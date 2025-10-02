from typing import Any, List
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from src.utils.utils import Transition

def CLVM_loss(vae_params: Any,
              transitions: Transition,
              key: jnp.ndarray,
              vae_model: Any,
              controlled_variables: List) -> jnp.ndarray:
    """
    Compute the loss for the Contextual Latent Variable Model (CLVM), combining
    KL divergence of latent variables and reconstruction loss of actions.

    Parameters
    ----------
    vae_params : Any
        Parameters of the CLVM (VAE) model.
    transitions : Transition
        Named tuple containing sequences:
            - s_t: current states, shape (B, T, state_dim)
            - s_tm1: previous states, shape (B, T, state_dim)
            - a_t: actions, shape (B, T, action_dim)
            - mask_t: mask tensor, shape (B, T, 1)
            - s_tp1: next states, used to extract controlled variables
    key : jnp.ndarray
        JAX PRNG key for stochastic operations.
    vae_model : Any
        CLVM model with an `apply` method.
    controlled_variables : List[int]
        Indices of controlled variables in state.

    Returns
    -------
    loss : jnp.ndarray
        Scalar total CLVM loss (KL + action reconstruction).
    info : dict
        Dictionary containing:
            - 'CLVM_loss': total loss
            - 'KL_loss': KL divergence term
            - 'action_decoder_loss': reconstruction term
    """
    
    s_t = transitions.s_t      # (batch_size_per_device, context_len, state_dim)
    a_t = transitions.a_t      # (batch_size_per_device, context_len, action_dim)
    s_tm1 = transitions.s_tm1  # (batch_size_per_device, context_len, state_dim)
    mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
    
    horizon = mask.sum(axis=1).astype(jnp.int32) # (B, 1)
    
    y_t = transitions.s_tp1[..., controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)

    vae_key, dropout_key = jax.random.split(key, 2)

    s_tm1_s_t = jnp.concatenate([s_tm1, s_t], axis=-1)

    kl_loss, action_decoder_loss = vae_model.apply(vae_params,
                                                   s_tm1_s_t,
                                                   a_t,
                                                   y_t,
                                                   horizon,
                                                   mask,
                                                   vae_key,
                                                   rngs={'dropout': dropout_key})
    
    loss = kl_loss + action_decoder_loss

    return loss, {'CLVM_loss': loss,
                  'KL_loss': kl_loss,
                  'action_decoder_loss': action_decoder_loss}

CLVM_grad = jax.value_and_grad(CLVM_loss, has_aux=True)