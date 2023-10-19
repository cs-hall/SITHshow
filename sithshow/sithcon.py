from .cmecell import CMECell

from typing import Optional
from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

class SITHConCell(eqx.Module):
    """A single step of SITHCon with the CME backend. See `SITHCon_Classifier` for example usage.
    
    (`CMECell` -> `eqx.nn.Conv2d` -> maxpool over tau_stars -> `eqx.nn.Linear` -> activation)
    """
    
    in_size: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_width: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    
    sith_cell: CMECell
    conv: eqx.nn.Conv2d
    linear: eqx.nn.Linear
    activation: Optional[Callable]
    
    def __init__(self,
                 in_size,
                 out_channels=5,
                 kernel_width=5,
                 dilation=1,
                 activation=None,
                 *,
                 key: PRNGKeyArray,
                 **kwargs):
        """**Arguments:**

        - `in_size`: The input size. The input to the layer should be a vector of shape `(in_size,)`
        - `out_channels`: The output features produced by the convolution. Defaults to 5.
        - `kernel_width`: The width of the convolutional kernel. Defaults to 5.
        - `dilation`: The dilation of the convolution. Defaults to 1.
        - `activation`: The activation function after the convolution and linear layer. Defaults to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation. (Keyword only argument.)
        """
        
        key1, key2 = jrandom.split(key, 2)
        
        self.in_size = in_size
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.dilation = dilation
        
        sith_params = kwargs.get('sith_params')
        self.sith_cell = CMECell(in_size, **sith_params) if sith_params is not None else CMECell(in_size)
        
        assert(self.kernel_width <= self.sith_cell.n_taus)

        self.conv = eqx.nn.Conv2d(in_channels=1,
                                  out_channels=out_channels,
                                  kernel_size=(in_size, kernel_width),
                                  dilation=(1, dilation),
                                  use_bias=False,
                                  key=key1)
        
        self.linear = eqx.nn.Linear(out_channels, 
                                    out_channels,
                                    key=key2)     
        self.activation = activation

    def __call__(self, F: Array, f: Array, alpha: Array) -> Array:        
        """ Update the Laplace state `F` with input `f`, modulated by `alpha`. Then, perform the SITHCon.
        
        **Arguments:**

        - `F`: The current state of the Laplace transform, which should be a JAX array of shape `(self.CMECell.n_taus, self.CMECell.fn_evals)`
        - `f`: The current input state, which should be a JAX array of shape `(in_size,)`.
        - `alpha`: a JAX array of shape `(in_size,)`.

        **Returns:**

        A tuple of JAX arrays (F_prime, out)
        
        - `F_prime`: the updated state of the Laplace transform, a JAX array of shape `(self.CMECell.n_taus, self.CMECell.fn_evals)`
        - `out`: the transformed SITHCon input, a JAX array of shape `(out_channels,)`.
        """
        
        F, f = self.sith_cell(F, f, alpha)
        f = jnp.array(f, dtype=jnp.float32)  
        f = self.conv(f[None, ...]).squeeze(axis=1) # convolve over (1, in_size, n_taus) -> (out_channels, 1, n_taus) -> (out_channels, n_taus)
        f = jnp.max(f, axis=1) # select the max tau at each channel -> (out_channels)
        
        f = self.linear(f)
        
        if self.activation:
            f = self.activation(f)
        
        return F, jnp.array(f, dtype=jnp.float64)
    

class SITHCon_Classifier(eqx.Module):
    """A classifier built on three `SITHConCell`s. After observing a timeseries, returns the log probability for each class."""
    
    cell1: SITHConCell
    cell2: SITHConCell
    cell3: SITHConCell

    def __init__(self, in_size, out_size, conv_features=20, kernel_width=5, dilation=2, *, key, **kwargs):
        sith_params = kwargs.get('sith_params')
        
        key1, key2, key3 = jrandom.split(key, 3)
        self.cell1 = SITHConCell(in_size,
                                 out_channels=conv_features,
                                 kernel_width=kernel_width,
                                 dilation=dilation,
                                 activation=jax.nn.relu,
                                 sith_params=sith_params,
                                 key=key1)
        self.cell2 = SITHConCell(self.cell1.out_channels,
                                 out_channels=conv_features,
                                 kernel_width=kernel_width,
                                 dilation=dilation,
                                 activation=jax.nn.relu,
                                 sith_params=sith_params, 
                                 key=key2)
        self.cell3 = SITHConCell(self.cell2.out_channels,
                                 out_channels=out_size,
                                 kernel_width=kernel_width,
                                 dilation=dilation,
                                 activation=jax.nn.log_softmax,
                                 sith_params=sith_params, 
                                 key=key3)

    def __call__(self, input):
        """**Arguments:**

        - `input`: A JAX array of shape `(in_size, seq_len)`

        **Returns:**

        A JAX array of shape `(out_size,)`
        """
        # swap the time and feature axis (scan only operates on the leading axis)
        input = jnp.swapaxes(input, 0, 1)
        
        # initialize starting states
        init_F1, init_F2, init_F3 = (self.cell1.sith_cell.get_init_F(), 
                                     self.cell2.sith_cell.get_init_F(),
                                     self.cell3.sith_cell.get_init_F()) 
        # this is just a placeholder for the output
        init_out = jnp.zeros(self.cell3.out_channels)
        
        def step(out_Fs, f):
            _, F1, F2, F3 = out_Fs

            F1, f = self.cell1(F1, f, jnp.ones_like(f))
            F2, f = self.cell2(F2, f, jnp.ones_like(f))
            F3, out = self.cell3(F3, f, jnp.ones_like(f))
            
            return (out, F1, F2, F3), None  # ("carryover", "accumulated")
        
        out_Fs, _ = jax.lax.scan(step, 
                                 (init_out, init_F1, init_F2, init_F3),
                                 input)
        out = out_Fs[0]
        
        return out
