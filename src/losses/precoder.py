from typing import Any
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from src.utils.utils import Transition

def precoder_loss(emp_params: Any,
                  transitions: Transition,
                  key: jnp.ndarray,
                  emp_model: Any,
                  dynamics_params: Any) -> jnp.ndarray:
    """
    Compute the precoder loss for the empowerment model, as the difference
    between info gain and mutual information.

    Parameters
    ----------
    emp_params : Any
        Parameters of the empowerment model.
    transitions : Transition
        Named tuple containing sequences:
            - s_t: current states, shape (B, T, state_dim)
            - s_tm1: previous states, shape (B, T, state_dim)
            - mask_t: mask tensor, shape (B, T, 1)
    key : jnp.ndarray
        JAX PRNG key for stochastic operations.
    emp_model : Any
        Empowerment model with an `apply` method.
    dynamics_params : Any
        Parameters of the dynamics model (used in rollout).

    Returns
    -------
    loss : jnp.ndarray
        Scalar precoder loss, computed as info_gain_loss - mutual_information.
    info : dict
        Dictionary containing:
            - 'precoder_loss': same as `loss`
            - 'mi': mutual information term
            - 'info_gain_loss': info gain term
    """

    s_t = transitions.s_t      # (batch_size_per_device, context_len, state_dim)
    s_tm1 = transitions.s_tm1  # (batch_size_per_device, context_len, state_dim)
    mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)

    emp_key, dropout_key = jax.random.split(key, 2)

    s_tm1_s_t = jnp.concatenate([s_tm1, s_t], axis=-1)

    mi, info_gain_loss = emp_model.apply(emp_params,
                                         s_tm1_s_t,
                                         mask,
                                         dynamics_params,
                                         emp_key,
                                         rngs={'dropout': dropout_key})

    loss = info_gain_loss - mi

    return loss, {'precoder_loss': loss,
                  'mi': mi,
                  'info_gain_loss': info_gain_loss}

precoder_grad = jax.value_and_grad(precoder_loss, has_aux=True)