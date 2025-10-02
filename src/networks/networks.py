"""
Causal transformer (GPT) implementation
"""

# transformers for MDPS
# https://openreview.net/pdf?id=NHMuM84tRT - LONG SHORT
# https://openreview.net/pdf?id=af2c8EaKl8 - CONV


import dataclasses
import jax
import jax.numpy as jnp
from flax import linen as nn
import math

from flax.linen.initializers import lecun_normal, zeros, zeros_init, constant
from typing import Any, Callable, List

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import distrax

@dataclasses.dataclass
class FeedForwardModel:
    init: Any
    apply: Any

class MaskedCausalAttention(nn.Module):
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

    @nn.compact
    def __call__(self, src: jnp.ndarray, mask: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        B, T, C = src.shape # batch size, seq length, h_dim * n_heads
        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim
        
        # rearrange q, k, v as (B, N, T, D)
        q = nn.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = nn.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = nn.Dense(
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

        attention = nn.Dropout(
            rate=self.drop_p,
            deterministic=deterministic)(normalized_weights @ v)
        
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N*D)

        projection = nn.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(attention)

        out = nn.Dropout(
            rate=self.drop_p,
            deterministic=deterministic)(projection)

        return out

class Block(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float = 0.1
    dtype: Any = jnp.float32
    kernel_init: Callable[..., Any] = lecun_normal()
    bias_init: Callable[..., Any] = zeros
    use_causal_mask: bool = True

    @nn.compact
    def __call__(self, src: jnp.ndarray, mask: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        src = src + MaskedCausalAttention(
            h_dim=self.h_dim,
            max_T=self.max_T,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            use_causal_mask=self.use_causal_mask
        )(src, mask, deterministic) # residual
        src = nn.LayerNorm(dtype=self.dtype)(src)

        src2 = nn.Dense(
            self.h_dim*4,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src)
        src2 = jax.nn.gelu(src2)
        src2 = nn.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(src2)
        src2 = nn.Dropout(
            rate=self.drop_p,
            deterministic=deterministic)(src2)

        src = src + src2 # residual
        src = nn.LayerNorm(dtype=self.dtype)(src)
        return src
    
class Transformer(nn.Module):
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
    
    @nn.compact
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

        horizon_embeddings = nn.Embed(
            num_embeddings=self.context_len,
            features=self.h_dim)(horizon-1)
        
        if self.transformer_type == 'encoder':
            max_T = next_controlled_variables.shape[1] + 1 # + 1 for initial state
        elif self.transformer_type == 'vae_encoder':
            max_T = actions.shape[1] + next_controlled_variables.shape[1] + 1 # + 1 for initial state
        else:
            max_T = self.context_len + 1 # + 1 for initial state

        positions = jnp.arange(max_T)[None,:].repeat(states.shape[0], axis=0)
        positional_embeddings = nn.Embed(
            num_embeddings=max_T,
            features=self.h_dim)(positions)

        initial_state_embedding = nn.Dense(
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
            
            controlled_variables_embeddings = nn.Dense(
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

            controlled_variables_embeddings = nn.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(next_controlled_variables) 
            controlled_variables_embeddings += positional_embeddings[:,1:next_controlled_variables.shape[1]+1:,:] 
            controlled_variables_embeddings += horizon_embeddings
            
            action_embeddings = nn.Dense(
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
            
            latent_embedding = nn.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(latent) 
            latent_embedding += positional_embeddings[:,1,:]
            latent_embedding += horizon_embeddings.squeeze()
            
            action_embeddings = nn.Dense(
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
           
            action_embeddings = nn.Dense(
                self.h_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(actions) + positional_embeddings[:,1:,:]
            
            # concatenate initial state and actions
            # (s_0, a_0, a_1 ..., a_H-1)
            # (B x [T + 1] x h_dim)
            h = jnp.concatenate((initial_state_embedding[:,None,:], action_embeddings), axis=1)

        h = nn.LayerNorm(dtype=self.dtype)(h)

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
            #     h = nn.Conv(features=self.h_dim, kernel_size=7, padding="SAME")(h) # kernel_size=3,5,7
            output_dim = self.act_dim * 2
        elif self.transformer_type == 'dynamics':
            h = h[:, -next_controlled_variables.shape[1]:]
            output_dim = self.controlled_variables_dim * 2
            
        # get outputs
        output = nn.Dense(
            output_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h)
        
        # if self.transformer_type == 'action_decoder' and self.apply_conv:
        #     x_mean, x_log_std = jnp.split(output, 2, axis=-1)
        #     # smooth mean actions
        #     x_mean = nn.Conv(features=self.act_dim, kernel_size=7, padding="SAME", feature_group_count=self.act_dim)(x_mean) # kernel_size=3,5,7
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

class GRU_Precoder(nn.Module):
    args: Any
    d_args: Any

    @nn.compact
    def __call__(self, s_t: jnp.ndarray, z_t: jnp.ndarray) -> jnp.ndarray:
        # s_t: (batch, 1, state_dim) assumed
        # z_t: (batch, z_dim)
    
        def scan_gru(s_t, z_t):
            
            # Initial input: (batch, state_dim + z_dim)
            initial_input = jnp.concatenate([s_t[0, :], z_t], axis=-1)

            initial_carry = MLP(out_dim=self.args.h_dims_GRU,
                                h_dims=[256,256],
                                drop_out_rates=[0., 0.])(initial_input)

            gru_cell = nn.GRUCell(features=self.args.h_dims_GRU)

            # Wrap the GRUCell with nn.RNN
            rnn_layer = nn.RNN(gru_cell)

            # Apply the RNN layer to the inputs
            if self.args.autonomous:
                inputs = jnp.zeros((self.args.context_len,1))
            else:
                inputs = initial_input[None].repeat(self.args.context_len, axis=0)
            _, outputs = rnn_layer(inputs, initial_carry=initial_carry, return_carry=True)

            ys = nn.Dense(self.d_args['act_dim'])(outputs)
            actions = jnp.tanh(ys)
                
            return actions  # (T, act_dim)

        # Vectorize across batch
        actions = jax.vmap(scan_gru)(s_t, z_t)  # (batch, T, act_dim)

        return actions

class MLP(nn.Module):
    out_dim: int
    h_dims: List
    drop_out_rates: List
    deterministic: bool = False
    
    def setup(self):

        self.mlp = [nn.Sequential([nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu]) for h_dim in self.h_dims]
        self.mlp_out = nn.Dense(features=self.out_dim)

        self.dropout = [nn.Dropout(rate=layer_i_rate) for layer_i_rate in self.drop_out_rates]

    def __call__(self, x, key=None):

        for i, fn in enumerate(self.mlp):
            x = fn(x)
            if self.drop_out_rates[i] > 0.:
                key, subkey = jax.random.split(key)
                x = self.dropout[i](x, self.deterministic, subkey)
        x = self.mlp_out(x)

        return x

class MLP_precoder(nn.Module):
    act_dim: int
    context_len: int
    h_dims: List
    apply_conv: bool = False
    
    def setup(self):

        self.precoder = [nn.Sequential([nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu]) for h_dim in self.h_dims]
        self.precoder_out = nn.Dense(features=self.context_len*self.act_dim)

    def __call__(self, s_t, z_t):

        x = jnp.concatenate((s_t[:,0,:], z_t), axis=-1)
        for fn in self.precoder:
            x = fn(x)
        # if self.apply_conv:
            # x = nn.Conv(features=self.h_dims[-1], kernel_size=7, padding="SAME")(h) # kernel_size=3,5,7
        x = self.precoder_out(x).reshape(-1,self.context_len,self.act_dim)
        if self.apply_conv:
            x = nn.Conv(features=self.act_dim, kernel_size=7, padding="SAME", feature_group_count=self.act_dim)(x) # kernel_size=3,5,7
        actions = jnp.tanh(x)

        return actions

class dynamics(nn.Module):
    args: Any
    d_args: Any
    
    def setup(self):

        self.dynamics = [nn.Sequential([nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu]) for h_dim in self.args.h_dims_dynamics]
        if self.args.learn_dynamics_std:
            self.dynamics_out = nn.Dense(features=self.d_args['obs_dim']*2)
        else:
            self.dynamics_out = nn.Dense(features=self.d_args['obs_dim'])

    def __call__(self, obs, actions, key=None):

        x = jnp.concatenate((obs, actions), axis=-1)
        for i, fn in enumerate(self.dynamics):
            x = fn(x)
        x = self.dynamics_out(x)

        if self.args.learn_dynamics_std:
            return x
        else:
            log_std = jnp.full_like(x, jnp.log(1e-3))
            return jnp.concatenate([x, log_std], axis=-1)

        return x

class posterior(nn.Module):
    args: Any
    d_args: Any

    def setup(self):


        self.horizon_mlp = nn.Sequential([sinusoidal_pos_emb(self.args.horizon_embed_dim),
                                          nn.Dense(self.args.horizon_embed_dim * 4),
                                          mish(),
                                          nn.Dense(self.args.horizon_embed_dim)])

        self.posterior_mlp = MLP(out_dim=self.args.controlled_variables_dim*2,
                           h_dims=self.args.h_dims_posterior,
                           drop_out_rates=self.args.posterior_dropout_rates)

    def __call__(self, state, horizon_steps, future_y, key):

        if horizon_steps.shape[0] != state.shape[0]:
            # mi training
            horizon_embedding = self.horizon_mlp(horizon_steps)[None].repeat(state.shape[0],axis=0)
        else:
            # CVLM training
            horizon_embedding = self.horizon_mlp(horizon_steps)

        z_dist_params = self.posterior_mlp(jnp.concatenate([state, future_y, horizon_embedding], axis=-1), key)

        return z_dist_params