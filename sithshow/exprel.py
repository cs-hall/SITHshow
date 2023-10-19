import jax.numpy as jnp
from jax import custom_jvp


@custom_jvp
def exprel(x: jnp.array) -> jnp.ndarray:
    # exprel(x) = (exp(x) - 1.0) / x
    # x = jnp.where(x == 0, jax.lax.stop_gradient(x), x)
    return jnp.where(x != 0, jnp.expm1(x) / x, jnp.ones_like(x))


exprel.defjvps(lambda t, ans, x: jnp.where(jnp.isclose(x, 0.0), 0.5, (ans * (x - 1) + 1) / x) * t)
