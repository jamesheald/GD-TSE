import jax
import jax.numpy as jnp

def sample_time_step(logits, horizon, max_horizon, key):

    mask = jnp.arange(max_horizon) >= horizon
    logits_trunc = jnp.where(mask, -1e30, logits)

    H_step = jax.random.categorical(key, logits_trunc)

    return H_step