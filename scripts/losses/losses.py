from typing import Any, List
import jax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from decision_transformer.dt.utils import Transition

def dynamics_loss(dynamics_params: Any,
                  transitions: Transition,
                  key: jnp.ndarray,
                  dynamics_model: Any,
                  delta_obs_min: jnp.ndarray,
                  delta_obs_max: jnp.ndarray,
                  min_log_std: float = -20.,
                  max_log_std: float = 2. ,
                  eps: float = 1e-3) -> jnp.ndarray:
    
    s_t = transitions.s_t     # (batch_size_per_device, context_len, state_dim)
    a_t = transitions.a_t     # (batch_size_per_device, context_len, action_dim)
    s_tm1 = transitions.s_tm1 # (batch_size_per_device, context_len, state_dim)
    d_s = transitions.d_s     # (batch_size_per_device, context_len, state_dim)

    s_tm1_s_t = jnp.concatenate([s_tm1, s_t], axis=-1)
    
    s_p = dynamics_model.apply(dynamics_params,
                               s_tm1_s_t,
                               a_t,
                               key)
    
    s_mean, s_log_std = jnp.split(s_p, 2, axis=-1)
    s_log_std = jnp.clip(s_log_std, min_log_std, max_log_std)

    base_dist = tfd.MultivariateNormalDiag(loc=s_mean,
                                           scale_diag=jnp.exp(s_log_std))

    bounded_bijector = tfb.Chain([
        tfb.Shift(shift=(delta_obs_min - eps/2)),
        tfb.Scale(scale=(delta_obs_max - delta_obs_min + eps)),
        tfb.Sigmoid(),
    ])
    dist = tfd.TransformedDistribution(distribution=base_dist,
                                       bijector=bounded_bijector)

    log_probs = dist.log_prob(d_s)
    loss = jnp.mean(-log_probs)
    loss /= d_s.shape[-1] 

    return loss, {'dynamics_loss': loss}

def CLVM_loss(vae_params: Any,
              transitions: Transition,
              key: jnp.ndarray,
              vae_model: Any,
              controlled_variables: List) -> jnp.ndarray:
    
    s_t = transitions.s_t      # (batch_size_per_device, context_len, state_dim)
    a_t = transitions.a_t      # (batch_size_per_device, context_len, action_dim)
    s_tm1 = transitions.s_tm1  # (batch_size_per_device, context_len, state_dim)
    mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
    
    horizon = mask.sum(axis=1).astype(jnp.int32) # (B, 1)
    
    y_t = transitions.s_tp1[...,controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)

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
    
    CLVM_loss = kl_loss + action_decoder_loss

    return CLVM_loss, {'CLVM_loss': CLVM_loss,
                       'KL_loss': kl_loss,
                       'action_decoder_loss': action_decoder_loss}

def precoder_loss(emp_params: Any,
                  transitions: Transition,
                  key: jnp.ndarray,
                  emp_model: Any,
                  dynamics_apply: Any,
                  dynamics_params: Any) -> jnp.ndarray:
    
    s_t = transitions.s_t      # (batch_size_per_device, context_len, state_dim)
    s_tm1 = transitions.s_tm1  # (batch_size_per_device, context_len, state_dim)
    mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)

    emp_key, dropout_key = jax.random.split(key, 2)

    s_tm1_s_t = jnp.concatenate([s_tm1, s_t], axis=-1)

    posterior_likelihood, info_gain_loss = emp_model.apply(emp_params,
                                                           s_tm1_s_t,
                                                           mask,
                                                           dynamics_params,
                                                           emp_key,
                                                           rngs={'dropout': dropout_key})

    precoder_loss = info_gain_loss - posterior_likelihood

    return precoder_loss, {'precoder_loss': precoder_loss,
                           'posterior_likelihood': posterior_likelihood,
                           'info_gain_loss': info_gain_loss}