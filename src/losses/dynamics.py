from typing import Any
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from src.utils.utils import Transition, get_mean_and_log_std

def dynamics_loss(dynamics_params: Any,
                  transitions: Transition,
                  key: jnp.ndarray,
                  dynamics_model: Any,
                  d_args: Any,
                  eps: float = 1e-3) -> jnp.ndarray:
    """
    Compute the negative log-likelihood loss of the dynamics model using a 
    bounded multivariate normal distribution for delta observations.

    Parameters
    ----------
    dynamics_params : Any
        Parameters of the dynamics model.
    transitions : Transition
        Named tuple containing state and action sequences:
            - s_t: current states, shape (B, T, state_dim)
            - s_tm1: previous states, shape (B, T, state_dim)
            - a_t: actions, shape (B, T, action_dim)
            - d_s: delta observations, shape (B, T, state_dim)
    key : jnp.ndarray
        JAX PRNG key for stochastic sampling.
    dynamics_model : Any
        Flax/JAX dynamics model with an `apply` method.
    d_args (Any):
        Environment-specific arguments (e.g., delta_obs_min, delta_obs_max).
    min_log_std : float, optional
        Minimum log standard deviation, by default -20.
    max_log_std : float, optional
        Maximum log standard deviation, by default 2.
    eps : float, optional
        Small epsilon for numerical stability in bounding, by default 1e-3.

    Returns
    -------
    loss : jnp.ndarray
        Scalar mean negative log-likelihood per state dimension.
    info : dict
        Dictionary containing:
            - 'dynamics_loss': same as `loss`.
    """
    
    s_t = transitions.s_t     # (batch_size_per_device, context_len, state_dim)
    a_t = transitions.a_t     # (batch_size_per_device, context_len, action_dim)
    s_tm1 = transitions.s_tm1 # (batch_size_per_device, context_len, state_dim)
    d_s = transitions.d_s     # (batch_size_per_device, context_len, state_dim)

    s_tm1_s_t = jnp.concatenate([s_tm1, s_t], axis=-1)
    
    s_p = dynamics_model.apply(dynamics_params,
                               s_tm1_s_t,
                               a_t,
                               key)

    s_mean, s_log_std = get_mean_and_log_std(s_p)

    base_dist = tfd.MultivariateNormalDiag(loc=s_mean, scale_diag=jnp.exp(s_log_std))

    bounded_bijector = tfb.Chain([
        tfb.Shift(shift=(d_args['delta_obs_min'] - eps/2)),
        tfb.Scale(scale=(d_args['delta_obs_max'] - d_args['delta_obs_min'] + eps)),
        tfb.Sigmoid(),
    ])

    dist = tfd.TransformedDistribution(distribution=base_dist, bijector=bounded_bijector)

    # negative log likelihood loss
    log_probs = dist.log_prob(d_s)
    loss = jnp.mean(-log_probs)

    # scale loss
    loss /= d_s.shape[-1] 

    return loss, {'dynamics_loss': loss}

dynamics_grad = jax.value_and_grad(dynamics_loss, has_aux=True)