import jax
import jax.numpy as jnp
import equinox as eqx
from .cmecell import CMECell


def cast_params_to_float32(model):
    cast_to_float32 = lambda leaf: jnp.array(leaf, dtype=jnp.float32) if eqx.is_array(leaf) else leaf  
    
    return eqx.tree_at(where=jax.tree_util.tree_leaves, pytree=model, replace_fn=cast_to_float32)


def increase_n_taus(cme_cell, n_taus):
    # HACK: weird stuff happens when you run this more than once because of edge cases I didn't think about
    
    tau_stars = cme_cell.tau_min * (1 + cme_cell.c)**jnp.arange(n_taus, dtype=jnp.float64)
    tau_max = cme_cell.tau_stars[-1]
    s = jnp.outer(1 / tau_stars, cme_cell.beta)
    
    return CMECell(cme_cell.in_size,
                   cme_cell.tau_min,
                   cme_cell.tau_max,
                   cme_cell.n_taus,
                   cme_cell.max_fn_evals,
                   cme_cell.g,
                   _n_taus_ = n_taus,
                   _tau_stars_ = tau_stars,
                   _tau_max_ = tau_max,
                   _s_ = s)


def increase_n_taus_sithcon(sithcon, n_taus):
    # increase the number of taus in the sithcon after training
    # HACK: weird stuff happens when you run this more than once because of edge cases I didn't think about
        
    sithcon = eqx.tree_at(where=lambda sithcon: sithcon.cell1.sith_cell, 
                          pytree=sithcon,
                          replace=increase_n_taus(sithcon.cell1.sith_cell, n_taus))
    
    sithcon = eqx.tree_at(where=lambda sithcon: sithcon.cell2.sith_cell, 
                          pytree=sithcon,
                          replace=increase_n_taus(sithcon.cell2.sith_cell, n_taus))

    sithcon = eqx.tree_at(where=lambda sithcon: sithcon.cell3.sith_cell, 
                          pytree=sithcon,
                          replace=increase_n_taus(sithcon.cell3.sith_cell, n_taus))

    return sithcon

