import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, List

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from src.networks.networks import GRU_Precoder, MLP, sinusoidal_pos_emb, mish
from src.utils.utils import get_mean_and_log_std

def get_rollout_function(dynamics_apply,
                         n_dynamics_ensembles,
                         obs_dim,
                         delta_obs_min,
                         delta_obs_max,
                         delta_obs_scale,
                         delta_obs_shift,
                         eps=1e-3):
    """
    Create a batched rollout function using an ensemble of dynamics models.

    Args:
        dynamics_apply (Callable): Function to compute dynamics predictions for a single ensemble member.
        n_dynamics_ensembles (int): Number of ensemble dynamics models.
        obs_dim (int): Dimensionality of the observation/state vector.
        delta_obs_min (float): Minimum bound for delta observation (used in bijector).
        delta_obs_max (float): Maximum bound for delta observation (used in bijector).
        delta_obs_scale (float): Scaling factor applied to delta observation.
        delta_obs_shift (float): Shift added to delta observation after scaling.
        eps (float, optional): Small epsilon to prevent numerical issues. Defaults to 1e-3.

    Returns:
        Callable: A vectorized rollout function `batch_peform_rollout(state, key, actions, dynamics_params)` that returns:
            next_state (array): Next states for each step in the rollout.
            info_gain (array): Information gain for each step in the rollout.
    """

    # sample a state sequence autoregressively from the learned markov dynamics model
    def peform_rollout(state, key, actions, dynamics_params):
        
        def step_fn(carry, action):
            state, key, dynamics_params = carry
            key, dropout_key, sample_i_key, sample_s_key = jax.random.split(key, 4)
            
            # predict the delta observation
            dropout_keys = jax.random.split(dropout_key, n_dynamics_ensembles)
            s_dist_params = jax.vmap(dynamics_apply, in_axes=(0,None,None,0))(dynamics_params, state, action, dropout_keys)
            s_mean, s_log_std = get_mean_and_log_std(s_dist_params)

            # calculate the info gain
            al_std = jnp.clip(jnp.sqrt(jnp.square(jnp.exp(s_log_std)).mean(0)), min=1e-3)
            ep_std = s_mean.std(axis=0)
            ratio = jnp.square(ep_std / al_std)
            info_gain = jnp.log(1 + ratio).mean(axis=-1)

            # sample a delta observation from the ensemble
            idx = jax.random.categorical(sample_i_key, jnp.ones(n_dynamics_ensembles), axis=-1)
            bounded_bijector = tfb.Chain([
                tfb.Shift(shift=(delta_obs_min - eps/2)),
                tfb.Scale(scale=(delta_obs_max - delta_obs_min + eps)),
                tfb.Sigmoid(),
            ])
            s_base_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))
            s_dist = tfd.TransformedDistribution(distribution=s_base_dist, bijector=bounded_bijector)
            delta_s = s_dist.sample(seed=sample_s_key)

            # calculate the next observation by adding the delta observation to the current observation
            delta_s = delta_s * delta_obs_scale + delta_obs_shift
            s_curr = state[..., obs_dim:]
            s_next = s_curr + delta_s

            # concatenate the current observation with the next observation to define the state
            next_state = jnp.concatenate([s_curr, s_next], axis=-1)

            carry = next_state, key, dynamics_params
            return carry, (s_next, info_gain)

        carry = state, key, dynamics_params
        _, (next_state, info_gain) = jax.lax.scan(step_fn, carry, actions)
        
        return next_state, info_gain

    batch_peform_rollout = jax.vmap(peform_rollout, in_axes=(0,0,0,None))

    return batch_peform_rollout

class CLVM(nn.Module):
    """
    Contextual/conditional/context-conditional latent variable model (CLVM) for inferring latent actions given 
    current state, future controlled variables, and horizon.

    Attributes:
        args (Any): Configuration object with hyperparameters.
        d_args (Any): Environment-specific arguments (e.g., action dimension).
        controlled_variables (List[int]): Indices of controlled variables.

    Methods:
        setup():
            Initializes encoder, precoder, optional state-dependent prior, and horizon embedding MLP.

        __call__(s_t, a_t, y_t, horizon, mask, key):
            Computes KL divergence loss and action reconstruction loss.

            Args:
                s_t (array): Current states, shape (batch, context_len, obs_dim*2)
                a_t (array): Observed actions, shape (batch, context_len, act_dim)
                y_t (array): Future controlled variables, shape (batch, context_len, controlled_variables_dim)
                horizon (array or int): Horizon step(s) for sampling future time step
                mask (array): Mask for valid timesteps, shape (batch, context_len, 1)
                key (PRNGKey): Random key for stochastic sampling

            Returns:
                kl_loss (float): KL divergence between posterior and prior over latent z
                a_decoder_loss (float): MSE between decoded actions and ground truth
    """
    args: Any
    d_args: Any
    controlled_variables: List

    def setup(self):

        self.horizon_mlp = nn.Sequential([sinusoidal_pos_emb(self.args.horizon_embed_dim),
                                          nn.Dense(self.args.horizon_embed_dim * 4),
                                          mish(),
                                          nn.Dense(self.args.horizon_embed_dim)])

        if self.args.state_dependent_prior:
            self.prior = MLP(out_dim=self.args.controlled_variables_dim*2,
                             h_dims=self.args.h_dims_prior,
                             drop_out_rates=[0.,0.])
            
        self.encoder = MLP(out_dim=self.args.controlled_variables_dim*2,
                           h_dims=self.args.h_dims_encoder,
                           drop_out_rates=self.args.encoder_dropout_rates)

        self.precoder = GRU_Precoder(act_dim=self.d_args['act_dim'],
                                     context_len=self.args.context_len,
                                     hidden_size=self.args.h_dims_GRU,
                                     autonomous=self.args.autonomous)
        
    def __call__(self, s_t, a_t, y_t, horizon, mask, key):

        key, subkey, sample_h_key, encoder_key = jax.random.split(key, 4)
        
        ###################### sample future time step from a geometric distribution ###################### 
        
        def sample_time_step(y_t, logits, horizon, key):

            mask = jnp.arange(self.args.context_len) >= horizon  # shape (N,)
            logits_trunc = jnp.where(mask, -1e30, logits)

            H_step = jax.random.categorical(key, logits_trunc)

            y_H_step = jnp.take_along_axis(y_t, H_step[None, None], axis=0)

            return H_step, y_H_step

        batch_get_H_step = jax.vmap(sample_time_step, in_axes=(0, None, 0, 0))

        logits = jnp.arange(self.args.context_len) * jnp.log(self.args.gamma)
        sample_h_keys = jax.random.split(sample_h_key, s_t.shape[0])
        H_step, y_H_step = batch_get_H_step(y_t, logits, horizon, sample_h_keys)

        ###################### infer latent action z given current state, future time step and the controlled variable at that future time step ###################### 

        horizon_embedding = self.horizon_mlp(H_step)

        z_dist_params = self.encoder(jnp.concatenate([s_t[:,0,:], y_H_step[:,0,:], horizon_embedding], axis=-1), encoder_key)

        z_mean, z_log_std = get_mean_and_log_std(z_dist_params)
        dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        z_samp = dist_z_post.sample(seed=subkey)

        ###################### precode latent action z ###################### 

        a_shape = a_t.shape

        actions = self.precoder(s_t, z_samp)

        mask = jnp.arange(self.args.context_len)[None] <= H_step[:,None] # shape (N,)
        a_mse = 0.5 * ((actions - a_t)**2 * mask[:,:,None]).sum(axis=-1).reshape(-1)

        ###################### loss ###################### 

        if self.args.state_dependent_prior:

            prior_params = self.prior(s_t[:,0,:])
            prior_mean, prior_log_std  = get_mean_and_log_std(prior_params)
            dist_z_prior = tfd.MultivariateNormalDiag(loc=prior_mean, scale_diag=jnp.exp(prior_log_std))

        else:
            
            # standard normal prior
            dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_mean.shape), scale_diag=jnp.ones(z_log_std.shape))
        
        kl_loss = tfd.kl_divergence(dist_z_post, dist_z_prior).mean()

        valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)

        # mean across batch, sum across time
        a_decoder_loss = jnp.sum(a_mse * valid_mask) / a_shape[0]

        # normalize
        kl_loss /= self.d_args['act_dim']
        a_decoder_loss /= self.d_args['act_dim']

        return kl_loss, a_decoder_loss