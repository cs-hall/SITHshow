from .exprel import exprel

import json
import importlib.resources
from typing import Literal, Union

import jax
import jax.numpy as jnp
from jax.config import config
from jaxtyping import Array
import equinox as eqx


config.update("jax_enable_x64", True)

def load_cme_params() -> dict:
    with importlib.resources.open_text("sithshow", "iltcme.json") as file:
        cme_params = json.load(file)  

    return cme_params

CME_PARAMS = load_cme_params()


class CMECell(eqx.Module):
    """A single step of SITH with the CME backend.
    
        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        sith_cell = CMECell(in_size='scalar')

        def track_signal_over_time(signal):    
            def step(F, f):
                F, til_f = sith_cell(F, f, alpha=jnp.ones_like(f))

                return F, til_f

            final_F, til_fs = jax.lax.scan(step, sith_cell.get_init_F(delta=False), signal)

            return til_fs
        ```
    """
    
    in_size: Union[int, Literal['scalar']] = eqx.field(static=True)
    tau_min: float = eqx.field(static=True)
    tau_max: float = eqx.field(static=True)
    n_taus: int = eqx.field(static=True)
    max_fn_evals: int = eqx.field(static=True)
    fn_evals: int = eqx.field(static=True)
    g: int = eqx.field(static=True)

    c: Array = eqx.field(static=True)
    tau_stars: Array = eqx.field(static=True)
    beta: Array = eqx.field(static=True)
    eta: Array = eqx.field(static=True)
    s: Array = eqx.field(static=True)

    def __init__(self, 
                 in_size: Union[int, Literal['scalar']], 
                 tau_min: float = 0.1,
                 tau_max: float = 100.0,
                 n_taus: int = 50,
                 max_fn_evals: int = 10,
                 g: int = 0.0,
                 **kwargs):
        """**Arguments:**

        - `in_size`: The input size. The input to the layer should be a vector of
            shape `(in_size,)`
        - `tau_min`: The center of the temporal receptive field for the first taustar produced. 
        - `tau_max`: The center of the temporal receptive field for the last taustar produced. 
        - `n_taus`: Number of taustars produced, spread out logarithmically.
        - `max_fn_evals`: When this number is higher, the inverse Laplace approximation is more accurate 
            (temporal specificity of the taustars is higher and numerical imprecision is lower) at the cost of increased memory usage.
        - `g`: <<TODO>>
        
        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.
        """        
        # adapted from: https://github.com/ghorvath78/iltcme/blob/master/ipython_ilt.ipynb
        # find the most steep CME satisfying maxFnEvals
        params = CME_PARAMS[0]
        for p in CME_PARAMS:
            if p["cv2"] < params["cv2"] and p["n"] + 1 <= max_fn_evals:
                params = p
        
        a, b, c = params["a"], params["b"], params["c"]
        n, omega, mu1 = params["n"], params["omega"], params["mu1"]
        
        eta_real = jnp.array((c, *a), dtype=jnp.float64)
        eta_imag = jnp.array((0, *b), dtype=jnp.float64)
        
        beta_real = jnp.ones(n + 1, dtype=jnp.float64)
        beta_imag = jnp.arange(n + 1, dtype=jnp.float64) * omega
        
        self.in_size = in_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_taus = n_taus
        self.max_fn_evals = max_fn_evals
        self.fn_evals = n
        self.g = g

        self.c = (tau_max / tau_min)**(1.0 / (n_taus - 1)) - 1
        self.tau_stars = tau_min * (1 + self.c)**jnp.arange(n_taus, dtype=jnp.float64)
        #self.tau_stars = jnp.geomspace(tau_min, tau_max, n_taus, dtype=jnp.float64)
        self.beta = (beta_real + beta_imag * 1.0j) * mu1
        self.eta = (eta_real + eta_imag * 1.0j) * mu1
        self.s = jnp.outer(1 / self.tau_stars, self.beta)
    
        # HACK to add more n taus while keeping old tau_stars the same 
        if len(kwargs) != 0:
            self.n_taus = kwargs['_n_taus_']
            self.tau_stars = kwargs['_tau_stars_']
            self.tau_max = float(kwargs['_tau_max_'])
            self.s = kwargs['_s_']
    

    def get_init_F(self, delta=False) -> Array:
        """
        Get an initial Laplace state `F`.  
        
        **Arguments:**

        - `delta`: Whether or not to present a delta function before returning `F`. Defaults to False.
        
        **Returns:**
        
        - `F_prime`: A starting state of the Laplace transform, a JAX array of shape `(self.n_taus, self.fn_evals)`
        """                        
        _in_size = () if self.in_size == 'scalar' else (self.in_size,) 
        
        F = jnp.zeros((*_in_size, *self.s.shape), dtype=jnp.complex128)

        if delta:
            F, _ = self(F, f=jnp.ones(_in_size), alpha=jnp.ones(_in_size))
            
        return F


    def _call(self, F, f, alpha):
        # === Forward Laplace Transform ===
        s_mul_a = self.s * alpha
        hh = jnp.exp(-s_mul_a)  # hidden -> hidden
        ih = exprel(-s_mul_a)  # input -> hidden
        
        b = f * ih
        F = F * hh + b
        
        # === Inverse Laplace Transform ===
        til_f = jnp.inner(self.eta, F).real / self.tau_stars
        # if g=1, multiply by tau_stars and divide by number of s per til_f
        til_f = til_f * (self.tau_stars / self.fn_evals) ** self.g
        
        return F, til_f


    def __call__(self, 
                 F: Array,
                 f: Array,
                 alpha: Array) -> tuple[Array, Array]:
        """
        Update the Laplace state `F` with input `f`, modulated by `alpha`.
        
        **Arguments:**

        - `F`: The current state of the Laplace transform, which should be a JAX array of shape `(self.n_taus, self.fn_evals)`
        - `f`: The current input state, which should be a JAX array of shape `(in_size,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `alpha`: a JAX array of shape `(in_size,)`. (Or shape `()` if `in_features="scalar"`.)

        **Returns:**

        A tuple of JAX arrays `(F_prime, til_f)`
        
        - `F_prime`: the updated state of the Laplace transform, a JAX array of shape `(self.n_taus, self.fn_evals)`
        - `til_f`: the inverse Laplace transform at each tau, a JAX array of shape `(in_size, self.n_taus)`
        """
        
        if self.in_size == "scalar":
            return self._call(F, f, alpha)
        else:
            return jax.vmap(self._call)(F, f, alpha)