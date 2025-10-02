import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, List

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from src.networks.networks import GRU_Precoder, MLP, posterior
from src.utils.utils import get_mean_and_log_std
from src.models.horizon_sampler import sample_time_step

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
            Initializes posterior, precoder, optional state-dependent prior, and horizon embedding MLP.

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

        if self.args.state_dependent_prior:
            self.prior = MLP(out_dim=self.args.controlled_variables_dim*2,
                             h_dims=self.args.h_dims_prior,
                             drop_out_rates=[0.,0.])
        
        self.precoder = GRU_Precoder(self.args, self.d_args)

        self.q_posterior = posterior(self.args, self.d_args)
        
    def __call__(self, s_t, a_t, y_t, horizon, mask, key):

        key, subkey, sample_h_key, posterior_key = jax.random.split(key, 4)
        
        ###################### sample future time step from a geometric distribution ###################### 
        
        def sample_future(y_t, logits, horizon, key):

            H_step = sample_time_step(logits, horizon, self.args.context_len, key)
            y_H_step = jnp.take_along_axis(y_t, H_step[None, None], axis=0)

            return H_step, y_H_step 

        batch_sample_future = jax.vmap(sample_future, in_axes=(0,None,0,0))

        logits = jnp.arange(self.args.context_len) * jnp.log(self.args.gamma)
        sample_h_keys = jax.random.split(sample_h_key, s_t.shape[0])
        H_step, y_H_step = batch_sample_future(y_t, logits, horizon, sample_h_keys)

        ###################### infer latent action z given current state, future time step and the controlled variable at that future time step ###################### 

        z_dist_params = self.q_posterior(s_t[:,0,:], H_step, y_H_step[:,0,:], posterior_key)

        z_mean, z_log_std = get_mean_and_log_std(z_dist_params)
        dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        z_samp = dist_z_post.sample(seed=subkey)

        ###################### precode latent action z ###################### 

        a_shape = a_t.shape

        actions = self.precoder(s_t, z_samp)

        mask = jnp.arange(self.args.context_len)[None] <= H_step[:,None]
        a_sse = 0.5 * ((actions - a_t)**2 * mask[:,:,None]).sum(axis=-1).reshape(-1)

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
        a_decoder_loss = jnp.sum(a_sse * valid_mask) / a_shape[0]

        # scale loss terms
        kl_loss /= self.d_args['act_dim']
        a_decoder_loss /= self.d_args['act_dim']

        return kl_loss, a_decoder_loss