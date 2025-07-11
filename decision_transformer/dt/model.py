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

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Tuple
import math

class mish(nn.Module):

    @nn.compact
    def __call__(self, x):

        return x * nn.tanh(nn.softplus(x))

class sinusoidal_pos_emb(nn.Module):
    dim: int

    def __call__(self, x):
        
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis = -1)

        return emb

class AutonomousGRU(nn.Module):
    hidden_size: int
    act_dim: int
    context_len: int

    @nn.compact
    def __call__(self, s_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        # s_t: (batch, 1, state_dim) assumed
        # z_t: (batch, z_dim)
    
        def scan_gru(s_t, z_t):
            
            # Initial input: (batch, state_dim + z_dim)
            initial_input = jnp.concatenate([s_t[0, :], z_t], axis=-1)

            # Map to initial hidden state
            initial_carry = nn.Dense(self.hidden_size)(initial_input)

            # initial_carry = MLP(out_dim=self.hidden_size,
            #                     h_dims=[256,256])(initial_input)

            gru_cell = nn.GRUCell(features=self.hidden_size)

            # Wrap the GRUCell with nn.RNN
            # cell_size is typically the hidden state size of the cell
            rnn_layer = nn.RNN(gru_cell)

            # Apply the RNN layer to the inputs
            inputs = jnp.zeros((self.context_len,1))
            _, outputs = rnn_layer(inputs, initial_carry=initial_carry, return_carry=True)

            ys = nn.Dense(self.act_dim)(outputs)
            # ys = nn.Dense(self.act_dim)(jnp.concatenate((initial_input[None,:].repeat(outputs.shape[0], axis=0), outputs), axis=-1))
            actions = jnp.tanh(ys)
                
            return actions  # (T, act_dim)

        # Vectorize across batch
        actions = jax.vmap(scan_gru)(s_t, z_t)  # (batch, T, act_dim)

        return actions

class MLP(linen.Module):
    out_dim: int
    h_dims: List
    
    def setup(self):

        self.mlp = [linen.Sequential([linen.Dense(features=h_dim), linen.LayerNorm(), linen.relu]) for h_dim in self.h_dims]
        self.mlp_out = linen.Dense(features=self.out_dim)

    def __call__(self, x):

        for fn in self.mlp:
            x = fn(x)
        x = self.mlp_out(x)

        return x

class MLP_precoder(linen.Module):
    act_dim: int
    context_len: int
    h_dims: List
    apply_conv: bool = False
    
    def setup(self):

        self.precoder = [linen.Sequential([linen.Dense(features=h_dim), linen.LayerNorm(), linen.relu]) for h_dim in self.h_dims]
        self.precoder_out = linen.Dense(features=self.context_len*self.act_dim)

    def __call__(self, s_t, z_t):

        x = jnp.concatenate((s_t[:,0,:], z_t), axis=-1)
        for fn in self.precoder:
            x = fn(x)
        # if self.apply_conv:
            # x = linen.Conv(features=self.h_dims[-1], kernel_size=7, padding="SAME")(h) # kernel_size=3,5,7
        x = self.precoder_out(x).reshape(-1,self.context_len,self.act_dim)
        if self.apply_conv:
            x = linen.Conv(features=self.act_dim, kernel_size=7, padding="SAME", feature_group_count=self.act_dim)(x) # kernel_size=3,5,7
        actions = jnp.tanh(x)

        return actions

class dynamics(linen.Module):
    h_dims_dynamics: List
    state_dim: int
    drop_out_rates: List
    deterministic: bool = False
    
    def setup(self):

        self.dynamics = [linen.Sequential([linen.Dense(features=h_dim), linen.LayerNorm(), linen.relu]) for h_dim in self.h_dims_dynamics]
        self.dynamics_out = linen.Dense(features=self.state_dim*2)
        
        # if len(jnp.array(self.drop_out_rates > 0.)) > 0:
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
                key, dummy_timesteps, dummy_states, dummy_latent, dummy_actions, dummy_controlled_variables, dummy_rtg, dummy_horizon, deterministic),
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
    n_dynamics_ensembles: int
    trajectory_version: bool = False
    state_dependent_prior: bool = False
    state_dependent_encoder: bool = False

    def setup(self):

        if self.state_dependent_prior:
            self.prior = MLP(out_dim=self.controlled_variables_dim*2,
                          h_dims=[256,256])

        # self.encoder = Transformer(state_dim=self.state_dim,
        #                             act_dim=self.act_dim,
        #                             controlled_variables_dim=self.controlled_variables_dim,
        #                             n_blocks=self.n_blocks,
        #                             h_dim=self.h_dim,
        #                             context_len=self.context_len,
        #                             n_heads=self.n_heads,
        #                             drop_p=self.drop_p,
        #                             transformer_type='encoder')
        #                             # transformer_type='vae_encoder')

        # if self.trajectory_version:
        #     self.encoder = MLP(out_dim=self._context_len*self.controlled_variables_dim*2,
        #                        h_dims=[256,256])
        # else:
        #     self.encoder = MLP(out_dim=self.controlled_variables_dim*2,
        #                        h_dims=[256,256])
            
        self.encoder = MLP(out_dim=self.controlled_variables_dim*2,
                           h_dims=[256,256])

    # out_dim: int
    # h_dims: List

        # self.precoder = Transformer(state_dim=self.state_dim,
        #                             act_dim=self.act_dim,
        #                             controlled_variables_dim=self.controlled_variables_dim,
        #                             n_blocks=self.n_blocks,
        #                             h_dim=self.h_dim,
        #                             context_len=self.context_len,
        #                             n_heads=self.n_heads,
        #                             drop_p=self.drop_p,
        #                             transformer_type='precoder',
        #                             apply_conv=False)
        
        # self.precoder = MLP_precoder(act_dim=self.act_dim,
        #                              context_len=self.context_len,
        #                              h_dims=[256,256])

        self.precoder = AutonomousGRU(act_dim=self.act_dim,
                                      context_len=self.context_len,
                                      hidden_size=128)
        
        # self.precoder = MLP(out_dim=self.act_dim,
                            # h_dims=[256,256])

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

        # y_t = jnp.squeeze(jnp.take_along_axis(s_tp1, horizon[..., None]-1, axis=1), axis=1)[...,self.controlled_variables]

        ###################### keys ###################### 

        key, subkey = jax.random.split(key)
        
        ###################### encode ###################### 

        y_h = jnp.take_along_axis(y_t, horizon[..., None]-1, axis=1)

        if self.state_dependent_encoder:
            z_dist_params = self.encoder(jnp.concatenate([s_t[:,0,:], y_h[:,0,:]], axis=-1))
        else:
            z_dist_params = self.encoder(y_h[:,0,:])
        z_mean, z_log_std = get_mean_and_log_std(z_dist_params)
        dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        z_t = dist_z_post.sample(seed=subkey)

        ###################### precode ###################### 

        a_shape = a_t.shape

        actions = self.precoder(s_t, z_t)
        # actions = self.precoder(s_t, y_h[:,0,:])

        a_mse = 0.5 * ((actions - a_t)**2).sum(axis=-1).reshape(-1)

        ###################### Markov dynamics ###################### 

        # # predict future states
        # # sample a state sequence autoregressively from the learned markov dynamics model
        # def peform_rollout(state, key, actions, dynamics_params):
            
        #     def step_fn(carry, action):
        #         state, key, dynamics_params = carry
        #         key, dropout_key, sample_i_key, sample_s_key = jax.random.split(key, 4)
                
        #         # multiple samples
        #         dropout_keys = jax.random.split(dropout_key, self.n_dynamics_ensembles)
        #         s_dist_params = jax.vmap(dynamics_apply, in_axes=(0,None,None,0))(dynamics_params, state, action, dropout_keys)
        #         # s_dist_params = dynamics_apply(dynamics_params, state, action, dropout_key)
        #         s_mean, s_log_std = get_mean_and_log_std(s_dist_params)
        #         disagreement = jnp.var(s_mean, axis=0).mean()

        #         idx = jax.random.categorical(sample_i_key, jnp.ones(self.n_dynamics_ensembles), axis=-1)
        #         s_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))

        #         delta_s = s_dist.sample(seed=sample_s_key)
        #         # next_state = state + delta_s
        #         s_curr = state[...,self.state_dim:]
        #         s_next = s_curr + delta_s

        #         next_state = jnp.concatenate([s_curr, s_next], axis=-1)

        #         carry = next_state, key, dynamics_params
        #         return carry, (s_curr, s_dist_params, disagreement)

        #     carry = state, key, dynamics_params
        #     _, (s_curr, s_dist_params, disagreement) = jax.lax.scan(step_fn, carry, actions)
            
        #     return s_curr, s_dist_params, disagreement
        
        # batch_peform_rollout = jax.vmap(peform_rollout, in_axes=(0,0,0,None))

        # dynamics_keys = jax.random.split(key, actions.shape[0])
        # s_samp, s_dist_params, disagreement = batch_peform_rollout(s_t[:,0,:], dynamics_keys, actions, dynamics_params)

        # s_mean, s_log_std = jnp.split(s_dist_params, 2, axis=-1)
        # min_log_std = -20.
        # max_log_std = 2.
        # s_log_std = jnp.clip(s_log_std, min_log_std, max_log_std)

        # s_samp = jnp.take_along_axis(s_samp, horizon[..., None]-1, axis=1)[:,:,None,self.controlled_variables]
        # s_mean = jnp.take_along_axis(s_mean, horizon[..., None, None]-1, axis=1)[...,self.controlled_variables]
        # s_log_std = jnp.take_along_axis(s_log_std, horizon[..., None, None]-1, axis=1)[...,self.controlled_variables]

        # y_dist = tfd.MultivariateNormalDiag(loc=s_samp + s_mean, scale_diag=jnp.exp(s_log_std))
        # y_log_probs = y_dist.log_prob(y_h[:,:,None,:].repeat(self.n_dynamics_ensembles,axis=2))

        # # y_mse = 0.5 * ((s_samp + s_mean - y_h[:,:,None,:].repeat(self.n_dynamics_ensembles,axis=2))**2).sum(axis=-1)

        ###################### dynamics ###################### 

        deterministic=False
        y_p = jax.vmap(dynamics_apply, in_axes=(0,None,None,None,None,None,None,None,None))(dynamics_params,
                                                                ts,
                                                                s_t,
                                                                z_t,
                                                                actions,
                                                                y_t,
                                                                rtg_t,
                                                                horizon,
                                                                deterministic)

        def true_fn(y_mean, y_log_std, y_t):
            dist = tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=jnp.exp(y_log_std))
            return dist.log_prob(y_t)

        y_mean, y_log_std = jnp.split(y_p, 2, axis=-1)
        min_log_std = -20.
        max_log_std = 2.
        y_log_std = jnp.clip(y_log_std, min_log_std, max_log_std)
        delta_y_t = y_h - s_t[:,:1,self.controlled_variables]
        delta_y_t = delta_y_t[None].repeat(self.n_dynamics_ensembles,axis=0)

        y_mean = jnp.take_along_axis(y_mean, horizon[None, ..., None]-1, axis=2)
        y_log_std = jnp.take_along_axis(y_log_std, horizon[None, ..., None]-1, axis=2)

        y_log_probs = true_fn(y_mean, y_log_std, delta_y_t)

        ###################### loss ###################### 

        # standard normal prior
        if self.state_dependent_prior:

            prior_params = self.prior(s_t[:,0,:])
            prior_mean, prior_log_std = jnp.split(prior_params, 2, axis=-1)
            min_log_std = -20.
            max_log_std = 2.
            prior_log_std = jnp.clip(prior_log_std, min_log_std, max_log_std)
            dist_z_prior = tfd.MultivariateNormalDiag(loc=prior_mean, scale_diag=jnp.exp(prior_log_std))

        else:
            
            dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_mean.shape), scale_diag=jnp.ones(z_log_std.shape))
        
        kl_loss = tfd.kl_divergence(dist_z_post, dist_z_prior).mean()

        y_log_probs_mixture = jax.nn.logsumexp(y_log_probs, b=1/self.n_dynamics_ensembles, axis=2)
        y_decoder_loss = -y_log_probs_mixture.mean()

        # y_mse_mixture = jax.nn.logsumexp(y_mse, b=1/self.n_dynamics_ensembles, axis=2)
        # y_decoder_loss = y_mse_mixture.mean()

        valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)

        # mean across batch and time
        # a_decoder_loss = jnp.sum(a_mse * valid_mask) / jnp.sum(valid_mask)

        # mean across batch, sum across time
        a_decoder_loss = jnp.sum(a_mse * valid_mask) / a_shape[0]

        # normalize
        kl_loss /= self.act_dim
        a_decoder_loss /= self.act_dim
        y_decoder_loss /= self.act_dim
        # disagreement_loss = disagreement.mean() / self.act_dim
        disagreement_loss = 0.

        return kl_loss, a_decoder_loss, y_decoder_loss, disagreement_loss
    
import distrax
from flax.linen.initializers import zeros_init, ones_init, normal, orthogonal, constant
class flow_model(linen.Module):
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

from jax.lax import stop_gradient
class empowerment(linen.Module):
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
    alternate_training: bool = False
    sample_one_model: bool = True
    use_flow: bool = False
    state_dependent_source: bool = False
    horizon_embed_dim: int = 128

    def setup(self):

        self.horizon_mlp = nn.Sequential([sinusoidal_pos_emb(self.horizon_embed_dim),
                                          nn.Dense(self.horizon_embed_dim * 4),
                                          mish(),
                                          nn.Dense(self.horizon_embed_dim)])

        if self.state_dependent_source:
            self.prior = MLP(out_dim=self.controlled_variables_dim*2,
                          h_dims=[256,256])
        
        self.precoder = AutonomousGRU(act_dim=self.act_dim,
                                      context_len=self.context_len,
                                      hidden_size=128)

        if self.alternate_training:
            dummy_s_t = jnp.zeros((1,self.context_len,self.state_dim))
            dummy_z_t = jnp.zeros((1,self.controlled_variables_dim))
            _ = self.precoder(dummy_s_t, dummy_z_t)

        self.encoder = MLP(out_dim=self.controlled_variables_dim*2,
                    h_dims=[256,256])
        
        if self.alternate_training:
            dummy_s_t = jnp.zeros((1,self.state_dim))
            dummy_y_t = jnp.zeros((1,self.controlled_variables_dim))
            _ = self.encoder(jnp.concatenate([dummy_s_t, dummy_y_t], axis=-1))

        if self.use_flow:
            self.flow = flow_model(h_dims_conditioner=256,
                                num_bijector_params=2,
                                num_coupling_layers=2,
                                z_dim=self.controlled_variables_dim)

    def __call__(self, ts, s_t, z_t, a_t, y_t, rtg_t, horizon, mask, train_precoder, dynamics_apply, dynamics_params, key):

        sample_z_key, sample_i_key, sample_y_key = jax.random.split(key, 3)

        def get_mean_and_log_std(x, min_log_std = -20., max_log_std = 2.):
            x_mean, x_log_std = jnp.split(x, 2, axis=-1)
            x_log_std = jnp.clip(x_log_std, min_log_std, max_log_std)
            return x_mean, x_log_std

        ###################### sample from aggregate posterior (state-dependent prior) ###################### 

        # z_dist_params = self.encoder(jnp.concatenate([s_t[:,0,:], y_t[:,0,:]], axis=-1))
        # z_mean, z_log_std = jnp.split(jax.lax.stop_gradient(z_dist_params), 2, axis=-1)
        # min_log_std = -20.
        # max_log_std = 2.
        # z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std)
        # post_dist = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        # z_samp = dist_z_prior.sample(seed=sample_z_key)

        ###################### state-dependent prior ###################### 

        if self.state_dependent_source:

            source_params = self.prior(s_t[:,0,:])
            source_mean, source_log_std = jnp.split(source_params, 2, axis=-1)
            min_log_std = -20.
            max_log_std = 2.
            source_log_std = jnp.clip(source_log_std, min_log_std, max_log_std)
            source_dist = tfd.MultivariateNormalDiag(loc=source_mean, scale_diag=jnp.exp(source_log_std))
            z_samp = source_dist.sample(seed=sample_z_key)

        else:

            source_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(z_t.shape), scale_diag=jnp.ones(z_t.shape))
            # sample_z_keys = jax.random.split(sample_z_key, s_t.shape[0])
            # z_samp = jax.vmap(dist_z_prior.sample)(seed=sample_z_keys)
            z_samp = source_dist.sample(seed=sample_z_key)

        ###################### precode ###################### 

        if self.alternate_training:

            precoder_apply = self.precoder.apply
            def apply_with_grad(precoder_params, s_t, z_t):
                return precoder_apply({'params': precoder_params}, s_t, z_t)

            def apply_without_grad(precoder_params, s_t, z_t):
                return precoder_apply({'params': jax.lax.stop_gradient(precoder_params)}, s_t, z_t)
            actions = jax.lax.cond(train_precoder,
                                apply_with_grad,
                                apply_without_grad,
                                self.variables['params']['precoder'], s_t, z_t)
            
        else:

            actions = self.precoder(s_t, z_t)

        ###################### Markov dynamics ###################### 

        # # predict future states
        # # sample a state sequence autoregressively from the learned markov dynamics model
        # def peform_rollout(state, key, actions, dynamics_params):
            
        #     def step_fn(carry, action):
        #         state, key, dynamics_params = carry
        #         key, dropout_key, sample_i_key, sample_s_key = jax.random.split(key, 4)
                
        #         # multiple samples
        #         dropout_keys = jax.random.split(dropout_key, self.n_dynamics_ensembles)
        #         s_dist_params = jax.vmap(dynamics_apply, in_axes=(0,None,None,0))(dynamics_params, state, action, dropout_keys)
        #         # s_dist_params = dynamics_apply(dynamics_params, state, action, dropout_key)
        #         s_mean, s_log_std = get_mean_and_log_std(s_dist_params)
        #         disagreement = jnp.var(s_mean, axis=0).mean()

        #         idx = jax.random.categorical(sample_i_key, jnp.ones(self.n_dynamics_ensembles), axis=-1)
        #         s_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))

        #         delta_s = s_dist.sample(seed=sample_s_key)
        #         next_state = state + delta_s
        #         carry = next_state, key, dynamics_params
        #         return carry, (next_state, disagreement)

        #     carry = state, key, dynamics_params
        #     _, (next_state, disagreement) = jax.lax.scan(step_fn, carry, actions)
            
        #     return next_state, disagreement
        
        # batch_peform_rollout = jax.vmap(peform_rollout, in_axes=(0,0,0,None))

        # dynamics_keys = jax.random.split(key, actions.shape[0])
        # next_state, _ = batch_peform_rollout(s_t[:,0,:], dynamics_keys, actions, dynamics_params)

        # y_samp = jnp.take_along_axis(next_state, horizon[..., None]-1, axis=1)[...,self.controlled_variables]

        ###################### sample controlled variable ###################### 

        deterministic=False
        y_p = jax.vmap(dynamics_apply, in_axes=(0,None,None,None,None,None,None,None,None))(dynamics_params,
                                                                ts,
                                                                s_t,
                                                                z_t,
                                                                actions,
                                                                y_t,
                                                                rtg_t,
                                                                horizon,
                                                                deterministic)

        y_mean, y_log_std = jnp.split(y_p, 2, axis=-1)
        idx = jax.random.categorical(sample_i_key, jnp.ones(self.n_dynamics_ensembles), axis=-1)
        min_log_std = -20.
        max_log_std = 2.
        y_log_std = jnp.clip(y_log_std, min_log_std, max_log_std)

        # y_mean = jnp.take_along_axis(y_mean, horizon[None, ..., None]-1, axis=2)
        # y_log_std = jnp.take_along_axis(y_log_std, horizon[None, ..., None]-1, axis=2)
        
        if self.sample_one_model:

            # sample a dynamics model from ensemble
            delta_y_dist = tfd.MultivariateNormalDiag(loc=y_mean[idx],
                                                    scale_diag=jnp.exp(y_log_std[idx]))
            delta_y = delta_y_dist.sample(seed=sample_y_key)
            y_samp = s_t[:,:1,self.controlled_variables] + delta_y
        
        else:

            # use all dynamics models in ensemble
            delta_y_dist = tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=jnp.exp(y_log_std))
            delta_y = delta_y_dist.sample(seed=sample_y_key)
            y_samp = s_t[:,:1,self.controlled_variables][None] + delta_y

        if self.alternate_training:

            encoder_apply = self.encoder.apply
            def apply_encoder_with_grad(encoder_params, s_t_y_samp):
                return encoder_apply({'params': encoder_params}, s_t_y_samp)

            def apply_encoder_without_grad(encoder_params, s_t_y_samp):
                return encoder_apply({'params': jax.lax.stop_gradient(encoder_params)}, s_t_y_samp)
            z_dist_params = jax.lax.cond(train_precoder,
                                apply_encoder_without_grad,
                                apply_encoder_with_grad,
                                self.variables['params']['encoder'], jnp.concatenate([s_t[:,0,:], y_samp[:,0,:]], axis=-1))

        else:

            horizon_embedding = self.horizon_mlp(jnp.arange(1,self.context_len+1))[None].repeat(s_t.shape[0],axis=0)

            s_t_expand = s_t[:,:1,:].repeat(self.context_len, axis=1)

            # z_dist_params = self.encoder(jnp.concatenate([s_t[:,0,:], y_samp[:,0,:]], axis=-1))

            z_dist_params = self.encoder(jnp.concatenate([s_t_expand, y_samp, horizon_embedding], axis=-1))

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

            # here i clip the variance of the posterior to 1, as it was 1 in the prior
            # and the posterior variance should be less than the prior
            z_mean, z_log_std = jnp.split(z_dist_params, 2, axis=-1)
            min_log_std = -20.
            if self.state_dependent_source:
                max_log_std = source_log_std
            else:
                max_log_std = 0.
            z_log_std = jnp.clip(z_log_std, min_log_std, max_log_std)
            post_dist = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))

        ###################### loss ###################### 

        # log_prob_z = post_dist.log_prob(z_samp)
        # loss = -jnp.mean(log_prob_z + source_dist.entropy())
        # loss /= z_samp.shape[-1]

        z_samp_expand = z_samp[:,None,:].repeat(self.context_len, axis=1)

        log_prob_z = post_dist.log_prob(z_samp_expand)

        gamma_geom = self.gamma ** jnp.arange(self.context_len)

        gamma_log_prob_z = gamma_geom[None] * log_prob_z
        
        # mean across batch, sum across time
        loss = jnp.sum(-gamma_log_prob_z[:,:,None] * mask) / log_prob_z.shape[0] - jnp.mean(source_dist.entropy()[:,None] * horizon)

        loss /= z_samp.shape[-1]
        
        return loss