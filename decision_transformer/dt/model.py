"""
From https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
Causal transformer (GPT) implementation
"""
import dataclasses
import jax
import jax.numpy as jnp

from flax import linen
from flax.linen.initializers import lecun_normal, zeros
from typing import Any, Callable, List, Optional

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from functools import partial

class dynamics(linen.Module):
    h_dims_dynamics: List
    state_dim: int
    drop_out_rates: List
    deterministic: bool = False
    
    def setup(self):

        self.dynamics = [linen.Sequential([linen.Dense(features=h_dim), linen.LayerNorm(), linen.relu]) for h_dim in self.h_dims_dynamics]
        self.dynamics_out = linen.Dense(features=self.state_dim*2)

        self.dropout = [linen.Dropout(rate=layer_i_rate) for layer_i_rate in self.drop_out_rates] 

    def __call__(self, obs, actions, key):

        x = jnp.concatenate((obs, actions), axis=-1)
        for i, fn in enumerate(self.dynamics):
            x = fn(x)
            if self.drop_out_rates[i] > 0.:
                key, subkey = jax.random.split(key)
                x = self.dropout[i](x, self.deterministic, subkey)
        x = self.dynamics_out(x)

        return x

@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any


class MaskedCausalAttention(linen.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    use_causal_mask: bool = True

    def setup(self):
        self.mask = jnp.tril(jnp.ones((self.max_T, self.max_T))).reshape(1, 1, self.max_T, self.max_T)

    @linen.compact
    def __call__(self, src: jnp.ndarray, mask: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        B, T, C = src.shape # batch size, seq length, h_dim * n_heads
        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim
        
        # rearrange q, k, v as (B, N, T, D)
        q = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        
        # weights (B, N, T, T)
        weights = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(D)

        if self.use_causal_mask:
            # causal mask applied to weights
            # mask == True --> weights, mask == False --> -jnp.inf
            weights = jnp.where(self.mask[..., :T, :T], weights, -jnp.inf)
        else:
            weights = jnp.where(jax.vmap(lambda x: jnp.outer(x, x).reshape(1, self.max_T, self.max_T))(mask)[..., :T, :T], weights, -1e30)
        # normalize weights, all -1e30 -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)

        attention = linen.Dropout(
            rate=self.drop_p,
            deterministic=deterministic)(normalized_weights @ v)
        
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N*D)

        projection = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(attention)

        out = linen.Dropout(
            rate=self.drop_p,
            deterministic=deterministic)(projection)

        return out


class Block(linen.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    use_causal_mask: bool = True

    @linen.compact
    def __call__(self, src: jnp.ndarray, mask: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        src = src + MaskedCausalAttention(
            h_dim=self.h_dim,
            max_T=self.max_T,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            use_causal_mask=self.use_causal_mask
        )(src, mask, deterministic) # residual
        src = linen.LayerNorm(dtype=self.dtype)(src)

        src2 = linen.Dense(
            self.h_dim*4,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src)
        src2 = jax.nn.gelu(src2)
        src2 = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src2)
        src2 = linen.Dropout(
            rate=self.drop_p,
            deterministic=deterministic)(src2)

        src = src + src2 # residual
        src = linen.LayerNorm(dtype=self.dtype)(src)
        return src
    
class Transformer(linen.Module):
    state_dim: int
    act_dim: int
    controlled_variables_dim: int
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float
    dtype: Any = jnp.float32
    max_timestep: int = 4096
    use_action_tanh: bool = True
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    transformer_type: str = 'dynamics' # 'encoder', 'precoder' or 'dynamics'
    trajectory_version: bool = False
    apply_conv: bool = False

    def setup(self):
        None
    
    @linen.compact
    def __call__(self,
                 timesteps: jnp.ndarray,
                 states: jnp.ndarray,
                 latent: jnp.ndarray,
                 actions: jnp.ndarray,
                 next_controlled_variables: jnp.ndarray,
                 returns_to_go: jnp.ndarray,
                 horizon: jnp.ndarray,
                 deterministic: bool) -> jnp.ndarray:
        B, T, _ = states.shape

        horizon_embeddings = linen.Embed(
            num_embeddings=self.context_len,
            features=self.h_dim)(horizon-1)
        
        if self.transformer_type == 'encoder':
            max_T = next_controlled_variables.shape[1] + 1 # + 1 for initial state
        elif self.transformer_type == 'vae_encoder':
            max_T = actions.shape[1] + next_controlled_variables.shape[1] + 1 # + 1 for initial state
        else:
            max_T = self.context_len + 1 # + 1 for initial state

        positions = jnp.arange(max_T)[None,:].repeat(states.shape[0], axis=0)
        positional_embeddings = linen.Embed(
            num_embeddings=max_T,
            features=self.h_dim)(positions)

        initial_state_embedding = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(states[:,0,:]) 
        initial_state_embedding += positional_embeddings[:,0,:]

        if self.transformer_type == 'encoder': # infer latent variable z_0 given s_0, y_1, ..., y_H

            # condiiton the encoder on the horizon length
            # currently conditioning by adding horizon embeddings
            # an alternative is to add a seperate horizon embedding token and allow other variables to attend to it
            initial_state_embedding += horizon_embeddings.squeeze()
            
            controlled_variables_embeddings = linen.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(next_controlled_variables) 
            controlled_variables_embeddings += positional_embeddings[:,1:,:] 
            controlled_variables_embeddings += horizon_embeddings
            
            # concatenate initial state and controlled variables
            # (s_0, y_1, ..., y_H)
            # (B x [T + 1] x h_dim)
            h = jnp.concatenate((initial_state_embedding[:,None,:], controlled_variables_embeddings), axis=1)

        elif self.transformer_type == 'vae_encoder': # infer latent variable z_0 given s_0, a_0, ..., a_H-1

            # condiiton the encoder on the horizon length
            # currently conditioning by adding horizon embeddings
            # an alternative is to add a seperate horizon embedding token and allow other variables to attend to it
            initial_state_embedding += horizon_embeddings.squeeze()

            controlled_variables_embeddings = linen.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(next_controlled_variables) 
            controlled_variables_embeddings += positional_embeddings[:,1:next_controlled_variables.shape[1]+1:,:] 
            controlled_variables_embeddings += horizon_embeddings
            
            action_embeddings = linen.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(actions) 
            action_embeddings += positional_embeddings[:,next_controlled_variables.shape[1]+1:,:] 
            action_embeddings += horizon_embeddings

            # stacked = jnp.stack([controlled_variables_embeddings, action_embeddings], axis=2)
            # # Step 2: Reshape to interleave along axis=1 → (B, 2T, D)
            # interleaved = stacked.reshape(stacked.shape[0], -1, stacked.shape[-1])
            # # Step 3: Expand initial_state_embedding to shape (B, 1, D)
            # initial_expanded = initial_state_embedding[:, None, :]
            # # Step 4: Concatenate along time axis
            # h = jnp.concatenate([initial_expanded, interleaved], axis=1)
                        
            # concatenate initial state and actions
            # (s_0, a_0, ..., a_H-1)
            # (B x [T + 1] x h_dim)
            h = jnp.concatenate((initial_state_embedding[:,None,:], controlled_variables_embeddings, action_embeddings), axis=1)

        elif self.transformer_type == 'precoder' or self.transformer_type == 'action_decoder': # generate action variable a_h given s_0, z_0, a_0, ..., a_h-1
            
            initial_state_embedding += horizon_embeddings.squeeze()
            
            latent_embedding = linen.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(latent) 
            latent_embedding += positional_embeddings[:,1,:]
            latent_embedding += horizon_embeddings.squeeze()
            
            action_embeddings = linen.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(actions[:,:-1,:])
            action_embeddings += positional_embeddings[:,2:,:] 
            action_embeddings += horizon_embeddings
            
            # concatenate initial state, latent and actions
            # (s_0, z_0, a_0, ..., a_H-1)
            # (B x [T + 1] x h_dim)
            h = jnp.concatenate((initial_state_embedding[:,None,:], latent_embedding[:,None,:], action_embeddings), axis=1)

        elif self.transformer_type == 'dynamics': # predict next controlled variables given s_0, a_0, ..., a_H-1

            # initial_state_embedding += horizon_embeddings # dynamics don't depend on horizon
           
            action_embeddings = linen.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(actions) + positional_embeddings[:,1:,:]
            
            # concatenate initial state and actions
            # (s_0, a_0, a_1 ..., a_H-1)
            # (B x [T + 1] x h_dim)
            h = jnp.concatenate((initial_state_embedding[:,None,:], action_embeddings), axis=1)

        h = linen.LayerNorm(dtype=self.dtype)(h)

        arange = jnp.arange(max_T)[None, :]
        if self.transformer_type == 'encoder':
            padded_mask = jnp.concatenate((jnp.ones((B,1)),
                                           arange[:,:next_controlled_variables.shape[1]] < horizon),
                                           axis=-1).astype(jnp.float32)[..., None]
        elif self.transformer_type == 'vae_encoder':
            padded_mask = jnp.concatenate((jnp.ones((B,1)),
                                           arange[:,:next_controlled_variables.shape[1]] < horizon,
                                           arange[:,:actions.shape[1]] < horizon),
                                           axis=-1).astype(jnp.float32)[..., None]
        else:
            padded_mask = jnp.zeros(1) # dummy
        # transformer and prediction
        for _ in range(self.n_blocks):
            h = Block(
                h_dim=self.h_dim,
                max_T=max_T,
                n_heads=self.n_heads,
                drop_p=self.drop_p,
                use_causal_mask=False if (self.transformer_type == 'encoder' or self.transformer_type == 'vae_encoder') else True)(h, padded_mask, deterministic)
            
        if self.transformer_type == 'encoder' or self.transformer_type == 'vae_encoder':
            # Multiply to zero-out unmasked values
            h_masked = h * padded_mask  # shape (B, T, D)

            # Sum only the masked positions
            sum_h = jnp.sum(h_masked, axis=1)  # shape (B, D)

            # Count of masked positions per batch
            count = jnp.sum(padded_mask, axis=1)  # shape (B, 1)

            # Pool (mean) token embeddings
            h = sum_h / count

            output_dim = next_controlled_variables.shape[1] * self.controlled_variables_dim * 2
        elif self.transformer_type == 'precoder':
            h = h[:, 1:]
            output_dim = self.act_dim
        elif self.transformer_type == 'action_decoder':
            h = h[:, 1:]
            # smooth hidden representations
            # if self.apply_conv:
            #     h = linen.Conv(features=self.h_dim, kernel_size=7, padding="SAME")(h) # kernel_size=3,5,7
            output_dim = self.act_dim * 2
        elif self.transformer_type == 'dynamics':
            h = h[:, -next_controlled_variables.shape[1]:]
            output_dim = self.controlled_variables_dim * 2
            
        # get outputs
        output = linen.Dense(
            output_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h)
        
        # if self.transformer_type == 'action_decoder' and self.apply_conv:
        #     x_mean, x_log_std = jnp.split(output, 2, axis=-1)
        #     # smooth mean actions
        #     x_mean = linen.Conv(features=self.act_dim, kernel_size=7, padding="SAME", feature_group_count=self.act_dim)(x_mean) # kernel_size=3,5,7
        #     output = jnp.concatenate([x_mean, x_log_std], axis=-1)

        return output

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
    dummy_states = jnp.zeros((batch_size, context_len, state_dim))
    dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
    if trajectory_version:
        dummy_latent = jnp.zeros((batch_size, context_len * controlled_variables_dim))
        dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
    else:
        dummy_latent = jnp.zeros((batch_size, controlled_variables_dim))
        dummy_controlled_variables = jnp.zeros((batch_size, 1, controlled_variables_dim))
    dummy_rtg = jnp.zeros((batch_size, context_len, 1))
    dummy_horizon = jnp.zeros((batch_size, 1), dtype=jnp.int32)

    def transformer_model_fn():
        class TransformerModule(linen.Module):
            @linen.compact
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
                key, dummy_timesteps, dummy_states, dummy_latent, dummy_actions, dummy_controlled_variables, dummy_rtg, dummy_horizon),
            apply=transformer_module.apply)
        return transformer
    return transformer_model_fn()

class VAE(linen.Module):
    state_dim: int
    act_dim: int
    controlled_variables_dim: int
    controlled_variables: List
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float

    def setup(self):

        self.encoder = Transformer(state_dim=self.state_dim,
                                    act_dim=self.act_dim,
                                    controlled_variables_dim=self.controlled_variables_dim,
                                    n_blocks=self.n_blocks,
                                    h_dim=self.h_dim,
                                    context_len=self.context_len,
                                    n_heads=self.n_heads,
                                    drop_p=self.drop_p,
                                    transformer_type='encoder')
                                    # transformer_type='vae_encoder')

        self.decoder = Transformer(state_dim=self.state_dim,
                                    act_dim=self.act_dim,
                                    controlled_variables_dim=self.controlled_variables_dim,
                                    n_blocks=self.n_blocks,
                                    h_dim=self.h_dim,
                                    context_len=self.context_len,
                                    n_heads=self.n_heads,
                                    drop_p=self.drop_p,
                                    transformer_type='action_decoder',
                                    apply_conv=False)

    def __call__(self, ts, s_t, z_t, a_t, y_t, rtg_t, horizon, mask, dynamics_apply, dynamics_params, key):

        def true_fn(x_mean, x_log_std, x_t, key=None):
            x_dist = tfd.MultivariateNormalDiag(loc=x_mean, scale_diag=jnp.exp(x_log_std))
            if key is not None:
                x_dist = tfd.TransformedDistribution(distribution=x_dist, bijector=tfb.Tanh())
                x_samp = x_dist.sample(seed=key)
            else:
                x_samp = jnp.zeros(x_t.shape)
            return x_dist.log_prob(x_t), x_samp

        def false_fn(x_mean, x_log_std, x_t, key=None):
            return 0., jnp.zeros(x_t.shape)
        
        def get_log_prob(mask, x_mean, x_log_std, x_t, key=None):
            return jax.lax.cond(mask, true_fn, false_fn, x_mean, x_log_std, x_t, key)
        batch_get_log_prob = jax.vmap(get_log_prob)

        def reshape_variables(x_mean, x_log_std, x_t, dim):
            x_mean = x_mean.reshape(-1, dim)
            x_log_std = x_log_std.reshape(-1, dim)
            x_t = x_t.reshape(-1, dim)
            return x_mean, x_log_std, x_t

        def get_mean_and_log_std(x, min_log_std = -20., max_log_std = 2.):
            x_mean, x_log_std = jnp.split(x, 2, axis=-1)
            x_log_std = jnp.clip(x_log_std, min_log_std, max_log_std)
            return x_mean, x_log_std
        
        def compute_smoothness(x: jnp.ndarray) -> jnp.ndarray:
            # x: (B, T, D)
            dx = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, D)
            smoothness = jnp.mean(jnp.square(dx), axis=1)  # (B, D)
            return smoothness

        ###################### keys ###################### 

        key, subkey = jax.random.split(key)
        
        ###################### encode ###################### 
        z_dist_params = self.encoder(ts, s_t, z_t, a_t, y_t, rtg_t, horizon, deterministic=False)
        z_mean, z_log_std = get_mean_and_log_std(z_dist_params)
        dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        z_t = dist_z_post.sample(seed=subkey)

        ###################### action decoder ###################### 

        a_shape = a_t.shape

        # use teacher forcing - fast(er) training
        a_dist_params = self.decoder(ts, s_t, z_t, a_t, y_t, rtg_t, horizon, deterministic=False)
        a_mean, a_log_std = get_mean_and_log_std(a_dist_params)

        # # no teacher forcing - slow training
        # actions = jnp.zeros(a_t.shape)
        # for t in range(self.context_len):
        #     a_dist_params = self.decoder(ts, s_t, z_t, actions, y_t, rtg_t, horizon)
        #     a_mean, _ = jnp.split(a_dist_params, 2, axis=-1)
        #     actions = actions.at[:,t,:].set(jnp.tanh(a_mean[:, t, :]))
        # a_mean, a_log_std = get_mean_and_log_std(a_dist_params)
        
        # reshape
        valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)
        a_mean, a_log_std, a_t = reshape_variables(a_mean,
                                                   a_log_std,
                                                   a_t,
                                                   self.act_dim)

        key, action_key, dynamics_key = jax.random.split(key, 3)
        action_keys = jax.random.split(action_key, valid_mask.shape[0])
        clipped_a_t = jnp.clip(a_t, -1+1e-6, 1-1e-6)
        # a_log_probs, a_samp = partial(batch_get_log_prob, tanh=True)(valid_mask, a_mean, a_log_std, clipped_a_t, action_keys)
        a_log_probs, a_samp = batch_get_log_prob(valid_mask, a_mean, a_log_std, clipped_a_t, action_keys)

        ###################### dynamics ###################### 

        y_shape = y_t.shape

        # predict future states
        # actions = a_samp.reshape(a_shape)
        actions = jnp.tanh(a_mean).reshape(a_shape)

        # sample a state sequence autoregressively from the learned markov dynamics model
        def peform_rollout(state, key, actions):
            
            def step_fn(carry, action):
                state, key = carry
                key, dropout_key, sample_key = jax.random.split(key, 3)
                s_dist_params = dynamics_apply(dynamics_params, state, action, dropout_key)
                s_mean, s_log_std = get_mean_and_log_std(s_dist_params)
                s_dist = tfd.MultivariateNormalDiag(loc=s_mean, scale_diag=jnp.exp(s_log_std))
                delta_s = s_dist.sample(seed=sample_key)
                next_state = state + delta_s
                carry = next_state, key
                return carry, s_dist_params

            carry = state, key
            _, s_dist_params = jax.lax.scan(step_fn, carry, actions)
            
            return s_dist_params
        
        batch_peform_rollout = jax.vmap(peform_rollout)

        dynamics_keys = jax.random.split(dynamics_key, actions.shape[0])
        s_dist_params = batch_peform_rollout(s_t[:,0,:], dynamics_keys, actions)

        # readout controlled variable(s)
        if y_shape[1] == 1:
            y_p = jnp.squeeze(jnp.take_along_axis(s_dist_params, horizon[..., None]-1, axis=1), axis=1)
        else:
            y_p = s_dist_params.copy()

        y_mean, y_log_std = get_mean_and_log_std(y_p)
        
        # reshape
        y_mean, y_log_std, y_t = reshape_variables(y_mean[...,self.controlled_variables], 
                                                   y_log_std[...,self.controlled_variables],
                                                   y_t,
                                                   self.controlled_variables_dim)

        if y_shape[1] == 1:
            y_log_probs, _ = true_fn(y_mean, y_log_std, y_t)
        else:
            y_log_probs, _ = batch_get_log_prob(valid_mask, y_mean, y_log_std, y_t)

        ###################### loss ###################### 

        # standard normal prior
        dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_mean.shape), scale_diag=jnp.ones(z_log_std.shape))
        kl_loss = tfd.kl_divergence(dist_z_post, dist_z_prior).mean()

        # a_decoder_loss = jnp.sum(-a_log_probs * valid_mask) / jnp.sum(valid_mask)
        a_decoder_loss = jnp.sum(-a_log_probs * valid_mask) / a_shape[0]

        if y_shape[1] == 1:
            y_decoder_loss = -y_log_probs.mean()
        else:
            # y_decoder_loss = jnp.sum(-y_log_probs * valid_mask) / jnp.sum(valid_mask)
            y_decoder_loss = jnp.sum(-y_log_probs * valid_mask) / a_shape[0]

        # smoothness = compute_smoothness(actions)

        # normalize
        kl_loss /= self.act_dim
        a_decoder_loss /= self.act_dim
        y_decoder_loss /= self.act_dim

        return kl_loss, a_decoder_loss, y_decoder_loss
    
class empowerment(linen.Module):
    state_dim: int
    act_dim: int
    controlled_variables_dim: int
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float

    def setup(self):

        self.encoder = Transformer(state_dim=self.state_dim,
                                    act_dim=self.act_dim,
                                    controlled_variables_dim=self.controlled_variables_dim,
                                    n_blocks=self.n_blocks,
                                    h_dim=self.h_dim,
                                    context_len=self.context_len,
                                    n_heads=self.n_heads,
                                    drop_p=self.drop_p,
                                    transformer_type='encoder')

        self.precoder = Transformer(state_dim=self.state_dim,
                                    act_dim=self.act_dim,
                                    controlled_variables_dim=self.controlled_variables_dim,
                                    n_blocks=self.n_blocks,
                                    h_dim=self.h_dim,
                                    context_len=self.context_len,
                                    n_heads=self.n_heads,
                                    drop_p=self.drop_p,
                                    transformer_type='action_decoder')

    def __call__(self, ts, s_t, z_t, a_t, y_t, rtg_t, horizon, mask, dynamics_apply, dynamics_params, key):

        dropout_key, sample_z_key, sample_y_key = jax.random.split(key, 3)

        ###################### sample from prior ###################### 
        
        dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_t.shape), scale_diag=jnp.ones(z_t.shape))
        z_samp = dist_z_prior.sample(seed=sample_z_key)

        ###################### precode ###################### 

        actions = jnp.zeros(a_t.shape)
        for t in range(self.context_len):
            a_dist_params = self.precoder(ts, s_t, z_samp, actions, y_t, rtg_t, horizon)
            a_mean, _ = jnp.split(a_dist_params, 2, axis=-1)
            actions = actions.at[:,t,:].set(jnp.tanh(a_mean[:, t, :]))

        ###################### dynamics ###################### 
        y_p = dynamics_apply(dynamics_params, ts, s_t, z_t, actions, y_t, rtg_t, horizon, rngs={'dropout': dropout_key})

        y_mean, y_log_std = jnp.split(y_p, 2, axis=-1)
        min_log_std = -20.
        max_log_std = 2.
        y_log_std = jnp.clip(y_log_std, min_log_std, max_log_std)
        dist_y = tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=jnp.exp(y_log_std))
        y_samp = dist_y.sample(seed=sample_y_key)

        ###################### sample controlled variable ###################### 
        z_dist_params = self.encoder(ts, s_t, z_t, a_t, y_samp, rtg_t, horizon)

        z_mean, z_log_std = jnp.split(z_dist_params, 2, axis=-1)
        min_log_std = -20.
        max_log_std = 2.
        z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std)
        post_dist = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))

        ###################### loss ###################### 

        log_prob_z = post_dist.log_prob(z_samp)

        loss = -jnp.mean(log_prob_z)

        loss /= z_samp.shape[-1]
        
        return loss