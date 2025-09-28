"""
From https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
Causal transformer (GPT) implementation
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, List

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import distrax
from flax.linen.initializers import zeros_init, constant

from jax.lax import stop_gradient

from decision_transformer.dt.networks.networks import GRU_Precoder, MLP, Transformer, FeedForwardModel, sinusoidal_pos_emb, mish

def make_transformer(state_dim: int,
                     act_dim: int,
                     controlled_variables_dim: int,
                     n_blocks: int,
                     h_dim: int,
                     context_len: int,
                     n_heads: int,
                     drop_p: float,
                     transformer_type: str) -> Transformer:
    """Creates a Transformer model.
    Args:
        state_dim: dimension of state
        act_dim: dimension of action
        n_blocks: number of attention blocks in transformer
        h_dim: size of hidden unit for liner layers
        context_len: length of context
        n_heads: number of attention heads in in transformer
        drop_p: dropout rate
    Returns:
        a model
    """
    module = Transformer(
        state_dim=state_dim,
        act_dim=act_dim,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=drop_p,
        transformer_type=transformer_type)

    return module


def make_transformer_networks(state_dim: int,
                              act_dim: int,
                              controlled_variables_dim: int,
                              n_blocks: int,
                              h_dim: int,
                              context_len: int,
                              n_heads: int,
                              drop_p: float,
                              trajectory_version: bool,
                              transformer_type: str) -> FeedForwardModel:
    batch_size = 1
    dummy_timesteps = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
    dummy_states = jnp.zeros((batch_size, context_len, state_dim*2))
    dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
    if trajectory_version:
        dummy_latent = jnp.zeros((batch_size, context_len * controlled_variables_dim))
    else:
        dummy_latent = jnp.zeros((batch_size, controlled_variables_dim))
    if trajectory_version or transformer_type=='dynamics':
        dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
    else:
        dummy_controlled_variables = jnp.zeros((batch_size, 1, controlled_variables_dim))
    dummy_rtg = jnp.zeros((batch_size, context_len, 1))
    dummy_horizon = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    deterministic = False

    def transformer_model_fn():
        class TransformerModule(nn.Module):
            @nn.compact
            def __call__(self,
                         timesteps: jnp.ndarray,
                         states: jnp.ndarray,
                         latent: jnp.ndarray,
                         actions: jnp.ndarray,
                         next_controlled_variables: jnp.ndarray,
                         returns_to_go: jnp.ndarray,
                         horizon: jnp.ndarray,
                         deterministic):
                outputs = make_transformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    controlled_variables_dim=controlled_variables_dim,
                    n_blocks=n_blocks,
                    h_dim=h_dim,
                    context_len=context_len,
                    n_heads=n_heads,
                    drop_p=drop_p,
                    transformer_type=transformer_type)(timesteps, states, latent, actions, next_controlled_variables, returns_to_go, horizon, deterministic)
                return outputs

        transformer_module = TransformerModule()
        transformer = FeedForwardModel(
            init=lambda key: transformer_module.init(
                key, dummy_timesteps, dummy_states, dummy_latent, dummy_actions, dummy_controlled_variables, dummy_rtg, dummy_horizon, deterministic),
            apply=transformer_module.apply)
        return transformer
    return transformer_model_fn()

class CLVM(nn.Module): # contextual/conditional/context-conditional latent variabl model
    state_dim: int
    act_dim: int
    controlled_variables_dim: int
    controlled_variables: List
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float
    n_dynamics_ensembles: int
    horizon_embed_dim: int
    gamma: float
    Markov_dynamics: True
    encoder_dropout_rates: List
    trajectory_version: bool = False
    state_dependent_prior: bool = True
    state_dependent_encoder: bool = True
    autonomous: bool = False

    def setup(self):

        self.horizon_mlp = nn.Sequential([sinusoidal_pos_emb(self.horizon_embed_dim),
                                          nn.Dense(self.horizon_embed_dim * 4),
                                          mish(),
                                          nn.Dense(self.horizon_embed_dim)])

        if self.state_dependent_prior:
            self.prior = MLP(out_dim=self.controlled_variables_dim*2,
                             h_dims=[256,256],
                             drop_out_rates=[0., 0.])
            
        self.encoder = MLP(out_dim=self.controlled_variables_dim*2,
                           h_dims=[256,256],
                           drop_out_rates=self.encoder_dropout_rates)

        self.precoder = GRU_Precoder(act_dim=self.act_dim,
                                     context_len=self.context_len,
                                     hidden_size=128,
                                     autonomous=self.autonomous)
        
    def __call__(self, s_t, a_t, y_t, horizon, mask, key):

        def get_mean_and_log_std(x, min_log_std = -20., max_log_std = 2.):
            x_mean, x_log_std = jnp.split(x, 2, axis=-1)
            x_log_std = jnp.clip(x_log_std, min_log_std, max_log_std)
            return x_mean, x_log_std

        key, subkey, dropout_key, sample_h_key, encoder_key = jax.random.split(key, 5)
        
        ###################### sample future time point from a geometric distribution ###################### 
        
        def sample_time_step(y_t, logits, horizon, key):

            mask = jnp.arange(self.context_len) >= horizon  # shape (N,)
            logits_trunc = jnp.where(mask, -1e30, logits)

            H_step = jax.random.categorical(key, logits_trunc)

            y_H_step = jnp.take_along_axis(y_t, H_step[None, None], axis=0)

            return H_step, y_H_step

        batch_get_H_step = jax.vmap(sample_time_step, in_axes=(0, None, 0, 0))

        logits = jnp.arange(self.context_len) * jnp.log(self.gamma)
        sample_h_keys = jax.random.split(sample_h_key, s_t.shape[0])
        H_step, y_H_step = batch_get_H_step(y_t, logits, horizon, sample_h_keys)

        ###################### infer latent action z ###################### 

        horizon_embedding = self.horizon_mlp(H_step)

        z_dist_params = self.encoder(jnp.concatenate([s_t[:,0,:], y_H_step[:,0,:], horizon_embedding], axis=-1), encoder_key)

        z_mean, z_log_std = get_mean_and_log_std(z_dist_params)
        dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        z_samp = dist_z_post.sample(seed=subkey)

        ###################### precode latent action z ###################### 

        a_shape = a_t.shape

        actions = self.precoder(s_t, z_samp)

        mask = jnp.arange(self.context_len)[None] <= H_step[:,None] # shape (N,)
        a_mse = 0.5 * ((actions - a_t)**2 * mask[:,:,None]).sum(axis=-1).reshape(-1)

        ###################### loss ###################### 

        if self.state_dependent_prior:

            prior_params = self.prior(s_t[:,0,:])
            prior_mean, prior_log_std = jnp.split(prior_params, 2, axis=-1)
            min_log_std = -20.
            max_log_std = 2.
            prior_log_std = jnp.clip(prior_log_std, min_log_std, max_log_std)
            dist_z_prior = tfd.MultivariateNormalDiag(loc=prior_mean, scale_diag=jnp.exp(prior_log_std))

        else:
            
            # standard normal prior
            dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_mean.shape), scale_diag=jnp.ones(z_log_std.shape))
        
        kl_loss = tfd.kl_divergence(dist_z_post, dist_z_prior).mean()

        valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)

        # mean across batch, sum across time
        a_decoder_loss = jnp.sum(a_mse * valid_mask) / a_shape[0]

        # normalize
        kl_loss /= self.act_dim
        a_decoder_loss /= self.act_dim

        return kl_loss, a_decoder_loss
    
class flow_model(nn.Module):
    h_dims_conditioner: int
    num_bijector_params: int
    num_coupling_layers: int
    z_dim: int

    def setup(self):

        # final linear layer of each conditioner initialised to zero so that the flow is initialised to the identity function
        self.conditioners = [nn.Sequential([nn.Dense(features=self.h_dims_conditioner), nn.relu,\
                                            nn.Dense(features=self.h_dims_conditioner), nn.relu,\
                                            nn.Dense(features=self.num_bijector_params*self.z_dim, bias_init=constant(jnp.log(jnp.exp(1.)-1.)), kernel_init=zeros_init())])
                             for layer_i in range(self.num_coupling_layers)]
        
    def __call__(self):

        def make_flow():
        
            mask = jnp.arange(self.z_dim) % 2 # every second element is masked
            mask = mask.astype(bool)

            def bijector_fn(params):
                shift, arg_soft_plus = jnp.split(params, 2, axis=-1)
                return distrax.ScalarAffine(shift=shift-jnp.log(jnp.exp(1.)-1.), scale=jax.nn.softplus(arg_soft_plus)+1e-3)
        
            layers = []
            for layer_i in range(self.num_coupling_layers):
                layer = distrax.MaskedCoupling(mask=mask, bijector=bijector_fn, conditioner=self.conditioners[layer_i])
                layers.append(layer)
                mask = jnp.logical_not(mask) # flip mask after each layer
            
            # return distrax.Inverse(distrax.Chain(layers)) # invert the flow so that the `forward` method is called with `log_prob`
            return distrax.Chain(layers)

        return make_flow()

class empowerment(nn.Module):
    state_dim: int
    act_dim: int
    controlled_variables_dim: int
    controlled_variables: List
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float
    gamma: float
    n_dynamics_ensembles: int
    delta_obs_scale: Any
    delta_obs_shift: Any
    delta_obs_min: Any
    delta_obs_max: Any
    horizon_embed_dim: int
    n_particles: int
    encoder_dropout_rates: List
    Markov_dynamics: bool = True
    alternate_training: bool = False
    sample_one_model: bool = True
    use_flow: bool = False
    state_dependent_source: bool = True
    learn_dynamics_std: bool = True
    autonomous: bool = False

    def setup(self):

        self.horizon_mlp = nn.Sequential([sinusoidal_pos_emb(self.horizon_embed_dim),
                                          nn.Dense(self.horizon_embed_dim * 4),
                                          mish(),
                                          nn.Dense(self.horizon_embed_dim)])

        if self.state_dependent_source:
            self.prior = MLP(out_dim=self.controlled_variables_dim*2,
                          h_dims=[256,256],
                          drop_out_rates=[0., 0.])
        
        self.precoder = GRU_Precoder(act_dim=self.act_dim,
                                     context_len=self.context_len,
                                     hidden_size=128,
                                     autonomous=self.autonomous)

        self.encoder = MLP(out_dim=self.controlled_variables_dim*2,
                    h_dims=[256,256],
                    drop_out_rates=self.encoder_dropout_rates)

        if self.use_flow:
            self.flow = flow_model(h_dims_conditioner=256,
                                num_bijector_params=2,
                                num_coupling_layers=2,
                                z_dim=self.controlled_variables_dim)

    def __call__(self, s_t, mask, dynamics_apply, dynamics_params, key):

        sample_z_key, encoder_key = jax.random.split(key)

        def get_mean_and_log_std(x, min_log_std = -20., max_log_std = 2.):
            x_mean, x_log_std = jnp.split(x, 2, axis=-1)
            x_log_std = jnp.clip(x_log_std, min_log_std, max_log_std)
            return x_mean, x_log_std

        ###################### state-dependent prior ###################### 

        if self.state_dependent_source:

            source_params = self.prior(s_t[:,0,:])
            source_mean, source_log_std = jnp.split(source_params, 2, axis=-1)
            min_log_std = -20.
            max_log_std = 2.
            source_log_std = jnp.clip(source_log_std, min_log_std, max_log_std)
            source_dist = tfd.MultivariateNormalDiag(loc=source_mean, scale_diag=jnp.exp(stop_gradient(source_log_std)))
            z_samp = source_dist.sample(seed=sample_z_key)

        else:

            # source_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_t.shape), scale_diag=jnp.ones(z_t.shape))
            source_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros((s_t.shape[0], self.controlled_variables_dim)),
                                                     scale_diag=jnp.ones((s_t.shape[0], self.controlled_variables_dim)))

            z_samp = source_dist.sample(seed=sample_z_key)

        ###################### precode ###################### 

        actions = self.precoder(s_t, z_samp)

        ###################### Markov dynamics ###################### 

        def sample_one_traj(s_t, actions, key, dynamics_params):

            # sample a state sequence autoregressively from the learned markov dynamics model
            def peform_rollout(state, key, actions, dynamics_params):
                
                def step_fn(carry, action):
                    state, key, dynamics_params = carry
                    key, dropout_key, sample_i_key, sample_s_key = jax.random.split(key, 4)
                    
                    # multiple samples
                    dropout_keys = jax.random.split(dropout_key, self.n_dynamics_ensembles)
                    s_dist_params = jax.vmap(dynamics_apply, in_axes=(0,None,None,0))(dynamics_params, state, action, dropout_keys)
                    # s_dist_params = dynamics_apply(dynamics_params, state, action, dropout_key)
                    s_mean, s_log_std = get_mean_and_log_std(s_dist_params)
                    
                    # disagreement = jnp.var(s_mean, axis=0).mean()

                    al_std = jnp.clip(jnp.sqrt(jnp.square(jnp.exp(s_log_std)).mean(0)), min=1e-3)
                    ep_std = s_mean.std(axis=0)
                    ratio = jnp.square(ep_std / al_std)
                    info_gain = jnp.log(1 + ratio).mean(axis=-1)#.reshape(-1, 1)

                    idx = jax.random.categorical(sample_i_key, jnp.ones(self.n_dynamics_ensembles), axis=-1)
                    eps = 1e-3
                    bounded_bijector = tfb.Chain([
                        tfb.Shift(shift=(self.delta_obs_min-eps/2)),
                        tfb.Scale(scale=(self.delta_obs_max - self.delta_obs_min + eps)),
                        tfb.Sigmoid(),
                    ])
                    # if self.learn_dynamics_std:
                    #     s_base_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))
                    #     s_dist = tfd.TransformedDistribution(distribution=s_base_dist, bijector=bounded_bijector)
                    #     delta_s = s_dist.sample(seed=sample_s_key)
                    # else:
                    #     s_base_dist = tfd.MultivariateNormalDiag(loc=bounded_bijector.forward(s_mean[idx]), scale_diag=jnp.exp(s_log_std[idx]))
                    #     delta_s = s_base_dist.sample(seed=sample_s_key)
                    s_base_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))
                    s_dist = tfd.TransformedDistribution(distribution=s_base_dist, bijector=bounded_bijector)
                    delta_s = s_dist.sample(seed=sample_s_key)

                    delta_s = delta_s * self.delta_obs_scale + self.delta_obs_shift

                    s_curr = state[...,self.state_dim:]
                    s_next = s_curr + delta_s

                    next_state = jnp.concatenate([s_curr, s_next], axis=-1)

                    carry = next_state, key, dynamics_params
                    return carry, (s_next, info_gain)

                carry = state, key, dynamics_params
                _, (next_state, info_gain) = jax.lax.scan(step_fn, carry, actions)
                
                return next_state, info_gain
            
            batch_peform_rollout = jax.vmap(peform_rollout, in_axes=(0,0,0,None))

            dynamics_keys = jax.random.split(key, actions.shape[0])
            next_state, info_gain = batch_peform_rollout(s_t[:,0,:], dynamics_keys, actions, dynamics_params)

            # y_samp = jnp.take_along_axis(next_state, horizon[..., None]-1, axis=1)[...,self.controlled_variables]
            y_samp = next_state[...,self.controlled_variables]

            return y_samp, info_gain

        batch_sample_one_traj = jax.vmap(sample_one_traj, in_axes=(None,None,0,None))

        particle_keys = jax.random.split(key, self.n_particles)
        y_samp, info_gain = batch_sample_one_traj(s_t, actions, particle_keys, dynamics_params)

        horizon_embedding = self.horizon_mlp(jnp.arange(1,self.context_len+1))[None].repeat(s_t.shape[0],axis=0)
        horizon_embedding = horizon_embedding[None].repeat(self.n_particles, axis=0)

        s_t_expand = s_t[:,:1,:].repeat(self.context_len, axis=1)
        s_t_expand = s_t_expand[None].repeat(self.n_particles, axis=0)

        z_dist_params = self.encoder(jnp.concatenate([s_t_expand, y_samp, horizon_embedding], axis=-1), encoder_key)

        ###################### encode NF ###################### 

        if self.use_flow:

            z_mean, z_log_std = jnp.split(z_dist_params, 2, axis=-1)
            min_log_std = -20.
            if self.state_dependent_source:
                max_log_std = source_log_std
            else:
                max_log_std = 0.
            z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std)
            base_post_dist = distrax.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
            bijector = self.flow()
            post_dist = distrax.Transformed(base_post_dist, bijector)
            
        else:

            # posterior variance should be less than prior variance
            z_mean, z_log_std = jnp.split(z_dist_params, 2, axis=-1)
            min_log_std = -20.
            if self.state_dependent_source:
                max_log_std = source_log_std
                z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std[None,:,None,:])
            else:
                max_log_std = 0.
                z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std)
            post_dist = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))

        ###################### loss ###################### 

        # log_prob_z = post_dist.log_prob(z_samp)
        # loss = -jnp.mean(log_prob_z + source_dist.entropy())
        # loss /= z_samp.shape[-1]

        z_samp_expand = z_samp[:,None,:].repeat(self.context_len, axis=1)
        z_samp_expand = z_samp_expand[None].repeat(self.n_particles, axis=0)

        log_prob_z = post_dist.log_prob(z_samp_expand)

        per_horizon_loss = -1 * (log_prob_z + source_dist.entropy()[None,:,None]) # P x B X T

        # discount and mask time/horizon
        gamma_geom = self.gamma ** jnp.arange(self.context_len)
        discounted_per_horizon_loss = gamma_geom[None,None] * per_horizon_loss * mask[None,:,:,0]

        info_gain_loss = (info_gain * mask[:,:,0]).sum(axis=-1).mean()
        info_gain_loss /= self.context_len
        info_gain_loss /= z_samp.shape[-1]
        
        # mean across batch, sum across time
        # loss = jnp.sum(-gamma_log_prob_z[:,:,None] * mask) / log_prob_z.shape[0] - jnp.mean(source_dist.entropy()[:,None] * horizon)

        loss = discounted_per_horizon_loss.sum(axis=-1).mean()
        loss /= self.context_len
        loss /= z_samp.shape[-1]
        
        return loss, info_gain_loss