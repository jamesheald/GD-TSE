import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, List

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from jax.lax import stop_gradient

from src.networks import GRU_Precoder, MLP, sinusoidal_pos_emb, mish
from src.rollout import get_rollout_function

class empowerment(nn.Module):
    """
    Empowerment model that learns precoder to maximize mutual information
    and minimize information gain with a learned dynamics model.

    Attributes:
        args (Any): Configuration object with hyperparameters.
        d_args (Any): Environment-specific arguments (e.g., action dimension).
        controlled_variables (List[int]): Indices of controlled variables.
        dynamics_apply (Callable): Dynamics model apply function.

    Methods:
        setup():
            Initializes encoder, precoder, optional state-dependent prior, horizon MLP, 
            and batched rollout function.

        __call__(s_t, mask, dynamics_params, key):
            Computes mutual information and info gain loss.

            Args:
                s_t (array): Current states, shape (batch, context_len, obs_dim*2)
                mask (array): Mask for valid timesteps, shape (batch, context_len, 1)
                dynamics_params (dict): Parameters for the dynamics model ensemble
                key (PRNGKey): Random key for stochastic sampling

            Returns:
                mi (float): Mutual information between latent z and future controlled variables
                info_gain_loss (float): Information gain from dynamics predictions
    """
    args: Any
    d_args: Any
    controlled_variables: List
    dynamics_apply: Any

    def setup(self):

        self.horizon_mlp = nn.Sequential([sinusoidal_pos_emb(self.args.horizon_embed_dim),
                                          nn.Dense(self.args.horizon_embed_dim * 4),
                                          mish(),
                                          nn.Dense(self.args.horizon_embed_dim)])

        if self.args.state_dependent_prior:
            self.prior = MLP(out_dim=self.args.controlled_variables_dim*2,
                             h_dims=self.args.h_dims_prior,
                             drop_out_rates=[0.,0.])
        
        self.precoder = GRU_Precoder(act_dim=self.d_args['act_dim'],
                                     context_len=self.args.context_len,
                                     hidden_size=self.args.h_dims_GRU,
                                     autonomous=self.args.autonomous)

        self.encoder = MLP(out_dim=self.args.controlled_variables_dim*2,
                    h_dims=self.args.h_dims_encoder,
                    drop_out_rates=self.args.encoder_dropout_rates)
            
        self.batch_peform_rollout = get_rollout_function(self.dynamics_apply,
                                                        self.args.n_dynamics_ensembles,
                                                        self.d_args['obs_dim'],
                                                        self.d_args['delta_obs_min'],
                                                        self.d_args['delta_obs_max'],
                                                        self.d_args['delta_obs_scale'],
                                                        self.d_args['delta_obs_shift'])

    def __call__(self, s_t, mask, dynamics_params, key):

        sample_z_key, encoder_key = jax.random.split(key)

        ###################### state-dependent prior ###################### 

        if self.args.state_dependent_prior:

            source_params = self.prior(s_t[:,0,:])
            source_mean, source_log_std = jnp.split(source_params, 2, axis=-1)
            min_log_std = -20.
            max_log_std = 2.
            source_log_std = jnp.clip(source_log_std, min_log_std, max_log_std)
            source_dist = tfd.MultivariateNormalDiag(loc=source_mean, scale_diag=jnp.exp(stop_gradient(source_log_std)))
            z_samp = source_dist.sample(seed=sample_z_key)

        else:

            source_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros((s_t.shape[0], self.args.controlled_variables_dim)),
                                                     scale_diag=jnp.ones((s_t.shape[0], self.args.controlled_variables_dim)))

            z_samp = source_dist.sample(seed=sample_z_key)

        ###################### precode ###################### 

        actions = self.precoder(s_t, z_samp)

        ###################### Markov dynamics ###################### 

        def sample_one_traj(s_t, actions, key, dynamics_params):        

            dynamics_keys = jax.random.split(key, actions.shape[0])
            next_state, info_gain = self.batch_peform_rollout(s_t[:,0,:], dynamics_keys, actions, dynamics_params)

            y_samp = next_state[...,self.controlled_variables]

            return y_samp, info_gain

        y_samp, info_gain = sample_one_traj(s_t, actions, key, dynamics_params)

        horizon_embedding = self.horizon_mlp(jnp.arange(1,self.args.context_len+1))[None].repeat(s_t.shape[0],axis=0)
        s_t_expand = s_t[:,:1,:].repeat(self.args.context_len, axis=1)
        z_dist_params = self.encoder(jnp.concatenate([s_t_expand, y_samp, horizon_embedding], axis=-1), encoder_key)

        # posterior variance should be less than prior variance
        z_mean, z_log_std = jnp.split(z_dist_params, 2, axis=-1)
        min_log_std = -20.
        if self.args.state_dependent_prior:
            max_log_std = source_log_std
            z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std[:,None,:])
        else:
            max_log_std = 0.
            z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std)
        post_dist = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))

        ###################### loss ###################### 

        z_samp_expand = z_samp[:,None,:].repeat(self.args.context_len, axis=1)

        log_prob_z = post_dist.log_prob(z_samp_expand)

        per_horizon_mi = log_prob_z + source_dist.entropy()[:,None] # B X T

        # gamma discounting
        gamma_geom = self.args.gamma ** jnp.arange(self.args.context_len)

        # gamma discounted mutual information
        discounted_per_horizon_mi = gamma_geom[None] * per_horizon_mi * mask[:,:,0]

        # sum mi across time, mean across batch
        mi = discounted_per_horizon_mi.sum(axis=-1).mean()

        # cumulative info gain across time, mean across batch
        cum_info_gain = jnp.cumsum(info_gain * mask[:, :, 0], axis=-1)

        # gamma discounted cumulative info gain
        discounted_info_gain = gamma_geom[None] * cum_info_gain

        # mean across batch
        info_gain_loss = discounted_info_gain.mean()

        # scale loss terms
        mi /= (self.args.context_len * z_samp.shape[-1])
        info_gain_loss /= (self.args.context_len * z_samp.shape[-1])
        
        return mi, info_gain_loss