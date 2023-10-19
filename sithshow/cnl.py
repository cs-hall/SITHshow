from .cmecell import CMECell
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx


class CNL(eqx.Module):
    in_size: int = eqx.field(static=True)
    sith_cell: CMECell
    get_alpha: eqx.nn.Linear
    out_linear: eqx.nn.Linear

    def __init__(self, in_size, *, key, **kwargs):
        key1, key2 = jrandom.split(key, 2)
        sith_params = kwargs.get('sith_params')

        self.in_size = in_size

        self.get_alpha = eqx.nn.Linear(in_size, 1, key=key1)        
        self.sith_cell = (CMECell(1, **sith_params) if sith_params is not None else CMECell(in_size))
        self.out_linear = eqx.nn.Linear(self.sith_cell.n_taus, 1, key=key2)        
        
    def __call__(self, input):
        # swap the time and feature axis (scan only operates on the leading axis)
        input = jnp.swapaxes(input, 0, 1)
        init_F = self.sith_cell.get_init_F(delta=True) # initialize starting states
        
        def step(F, f):
            alpha = self.get_alpha(f)
            F, til_f = self.sith_cell(F, jnp.array([0.]), alpha)
            out = self.out_linear(til_f.ravel())
            
            return F, jax.nn.log_sigmoid(out)  # ("carryover", "accumulated")
        
        final_F, pred = jax.lax.scan(step, init_F, input)
        
        return pred.ravel()