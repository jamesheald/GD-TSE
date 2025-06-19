"""
From https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
Causal transformer (GPT) implementation
"""
import dataclasses
import jax
import jax.numpy as jnp

from flax import linen
from flax.linen.initializers import lecun_normal, zeros
from typing import Any, Callable, Optional

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

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
    deterministic: bool = False if drop_p > 0.0 else True
    use_causal_mask: bool = True

    def setup(self):
        self.mask = jnp.tril(jnp.ones((self.max_T, self.max_T))).reshape(1, 1, self.max_T, self.max_T)

    @linen.compact
    def __call__(self, src: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
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
            weights = jnp.where(jax.vmap(lambda x: jnp.outer(x, x).reshape(1, self.max_T, self.max_T))(mask)[..., :T, :T], weights, -jnp.inf)
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)

        attention = linen.Dropout(
            rate=self.drop_p,
            deterministic=self.deterministic)(normalized_weights @ v)
        
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N*D)

        projection = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(attention)

        out = linen.Dropout(
            rate=self.drop_p,
            deterministic=self.deterministic)(projection)

        return out


class Block(linen.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    deterministic: bool = False if drop_p > 0.0 else True
    use_causal_mask: bool = True

    @linen.compact
    def __call__(self, src: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        src = src + MaskedCausalAttention(
            h_dim=self.h_dim,
            max_T=self.max_T,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            use_causal_mask=self.use_causal_mask,
        )(src, mask) # residual
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
            deterministic=self.deterministic)(src2)

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
                 horizon: jnp.ndarray) -> jnp.ndarray:
        B, T, _ = states.shape

        horizon_embeddings = linen.Embed(
            num_embeddings=self.context_len,
            features=self.h_dim)(horizon-1)
        
        if self.transformer_type == 'encoder':
            max_T = next_controlled_variables.shape[1] + 1 # + 1 for initial state
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

        elif self.transformer_type == 'action_encoder': # infer latent variable z_0 given s_0, a_0, ..., a_H-1

            # condiiton the encoder on the horizon length
            # currently conditioning by adding horizon embeddings
            # an alternative is to add a seperate horizon embedding token and allow other variables to attend to it
            initial_state_embedding += horizon_embeddings.squeeze()
            
            action_embeddings = linen.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(actions) 
            action_embeddings += positional_embeddings[:,1:,:] 
            action_embeddings += horizon_embeddings
            
            # concatenate initial state and actions
            # (s_0, a_0, ..., a_H-1)
            # (B x [T + 1] x h_dim)
            h = jnp.concatenate((initial_state_embedding[:,None,:], action_embeddings), axis=1)
        
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

        # first_elem = mask[:, :1, :]
        # padded_mask = jnp.concatenate([first_elem, mask], axis=1)
        if next_controlled_variables.shape[1] == 1 and self.transformer_type == 'encoder': # in this setting, padded_mask plays no role (multiply by 1)
            padded_mask = jnp.ones((B, max_T, 1), dtype=jnp.float32)
        else:
            arange = jnp.arange(max_T)[None, :]  # Shape: (1, T)
            padded_mask = (arange <= horizon).astype(jnp.float32)[..., None]  # Shape: (B, T, 1)
        
        # transformer and prediction
        for _ in range(self.n_blocks):
            h = Block(
                h_dim=self.h_dim,
                max_T=max_T,
                n_heads=self.n_heads,
                drop_p=self.drop_p,
                use_causal_mask=False if self.transformer_type == 'encoder' else True)(h, padded_mask)
            
        if self.transformer_type == 'encoder' or self.transformer_type == 'action_encoder':
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
                         horizon: jnp.ndarray):
                outputs = make_transformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    controlled_variables_dim=controlled_variables_dim,
                    n_blocks=n_blocks,
                    h_dim=h_dim,
                    context_len=context_len,
                    n_heads=n_heads,
                    drop_p=drop_p,
                    transformer_type=transformer_type)(timesteps, states, latent, actions, next_controlled_variables, returns_to_go, horizon)
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
                                    transformer_type='action_encoder')

        self.decoder = Transformer(state_dim=self.state_dim,
                                    act_dim=self.act_dim,
                                    controlled_variables_dim=self.controlled_variables_dim,
                                    n_blocks=self.n_blocks,
                                    h_dim=self.h_dim,
                                    context_len=self.context_len,
                                    n_heads=self.n_heads,
                                    drop_p=self.drop_p,
                                    transformer_type='action_decoder')

    def __call__(self, ts, s_t, z_t, a_t, y_t, rtg_t, horizon, mask, dynamics_apply, dynamics_params, key):
        
        ###################### encode ###################### 
        z_dist_params = self.encoder(ts, s_t, z_t, a_t, y_t, rtg_t, horizon)

        z_mean, z_log_std = jnp.split(z_dist_params, 2, axis=-1)
        min_log_std = -20.
        max_log_std = 2.
        z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std)
        dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        z_t = dist_z_post.sample(seed=key)

        ###################### decoder ###################### 

        # use teacher forcing
        a_dist_params = self.decoder(ts, s_t, z_t, a_t, y_t, rtg_t, horizon)

        a_mean, a_log_std = jnp.split(a_dist_params, 2, axis=-1)
        min_log_std = -20.
        max_log_std = 2.
        a_log_std = jnp.clip(a_log_std, min_log_std, max_log_std)
        a_mean = a_mean.reshape(-1, self.act_dim)
        a_log_std = a_log_std.reshape(-1, self.act_dim)
        
        ###################### loss ###################### 

        def true_fn(a_mean, a_log_std, a_t):
            base_dist = tfd.MultivariateNormalDiag(loc=a_mean, scale_diag=jnp.exp(a_log_std))
            a_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
            return a_dist.log_prob(a_t)

        def false_fn(a_mean, a_log_std, a_t):
            return 0.
        
        def get_log_prob(mask, a_mean, a_log_std, a_t):
            log_prob = jax.lax.cond(mask, true_fn, false_fn, a_mean, a_log_std, a_t)
            return log_prob
        batch_get_log_prob = jax.vmap(get_log_prob)
        
        a_t = a_t.reshape(-1, self.act_dim)
        a_t = jnp.clip(a_t, -1+1e-6, -1+1e-6)
        valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)
        log_probs = batch_get_log_prob(valid_mask, a_mean, a_log_std, a_t)
        decoder_loss = jnp.sum(-log_probs * valid_mask) / jnp.sum(valid_mask)
        
        # independent standard normal prior
        dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_mean.shape), scale_diag=jnp.ones(z_log_std.shape))
        kl_loss = tfd.kl_divergence(dist_z_post, dist_z_prior).mean()

        return decoder_loss, kl_loss
    
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
        
        return loss