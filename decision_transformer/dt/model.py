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
        self.mask = jnp.tril(
            jnp.ones((self.max_T, self.max_T))).reshape(1, 1, self.max_T, self.max_T)

    @linen.compact
    def __call__(self, src: jnp.ndarray) -> jnp.ndarray:
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
    def __call__(self, src: jnp.ndarray) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        src = src + MaskedCausalAttention(
            h_dim=self.h_dim,
            max_T=self.max_T,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            use_causal_mask=self.use_causal_mask,
        )(src) # residual
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


class DynamicsTransformer(linen.Module):
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

    def setup(self):
        self.input_seq_len = 3 * self.context_len
    
    @linen.compact
    def __call__(self,
                 timesteps: jnp.ndarray,
                 states: jnp.ndarray,
                 actions: jnp.ndarray,
                 returns_to_go: jnp.ndarray) -> jnp.ndarray:
        B, T, _ = states.shape

        # time_embeddings = linen.Embed(
        #     num_embeddings=self.max_timestep,
        #     features=self.h_dim)(timesteps)

        positions = jnp.arange(self.context_len+1)[None,:].repeat(states.shape[0], axis=0)
        time_embeddings = linen.Embed(
            num_embeddings=self.context_len+1,
            features=self.h_dim)(positions)

        # time embeddings are treated similar to positional embeddings
        initial_state_embedding = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(states[:,0,:]) + time_embeddings[:,0,:]
        action_embeddings = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(actions) + time_embeddings[:,1:,:]

        # concatenate initial state and actions
        # (s_0, a_0, a_2 ..., a_T)
        # (B x [T + 1] x h_dim)
        h = jnp.concatenate((initial_state_embedding[:,None,:], action_embeddings), axis=1)

        h = linen.LayerNorm(dtype=self.dtype)(h)

        # transformer and prediction
        for _ in range(self.n_blocks):
            h = Block(
                h_dim=self.h_dim,
                max_T=self.input_seq_len,
                n_heads=self.n_heads,
                drop_p=self.drop_p)(h)

        # get predictions
        next_controlled_variable_preds = linen.Dense(
            self.controlled_variables_dim*2,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h[:, 1:])     # predict next controlled variables given s_0, a_0, ..., a_t

        return next_controlled_variable_preds

class EncoderTransformer(linen.Module):
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
    use_causal_mask: bool = False

    def setup(self):
        self.input_seq_len = 3 * self.context_len
    
    @linen.compact
    def __call__(self,
                 timesteps: jnp.ndarray,
                 states: jnp.ndarray,
                 next_controlled_variables: jnp.ndarray,
                 returns_to_go: jnp.ndarray) -> jnp.ndarray:
        B, T, _ = states.shape

        positions = jnp.arange(self.context_len+1)[None,:].repeat(states.shape[0], axis=0)
        time_embeddings = linen.Embed(
            num_embeddings=self.context_len+1,
            features=self.h_dim)(positions)

        # time embeddings are treated similar to positional embeddings
        initial_state_embedding = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(states[:,0,:]) + time_embeddings[:,0,:]
        controlled_variables_embeddings = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(next_controlled_variables) + time_embeddings[:,1:,:]

        # concatenate initial state and controlled variables
        # (s_0, y_1, ..., y_H)
        # (B x [T + 1] x h_dim)
        h = jnp.concatenate((initial_state_embedding[:,None,:], controlled_variables_embeddings), axis=1)

        h = linen.LayerNorm(dtype=self.dtype)(h)

        # transformer and prediction
        for _ in range(self.n_blocks):
            h = Block(
                h_dim=self.h_dim,
                max_T=self.input_seq_len,
                n_heads=self.n_heads,
                drop_p=self.drop_p,
                use_causal_mask=self.use_causal_mask)(h)
            
        # pool token embeddings
        h_pooled = jnp.mean(h, axis=1)

        # get predictions
        latent_preds = linen.Dense(
            self.controlled_variables_dim*2,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h_pooled)     # predict latent variable given s_0, y_1, ..., y_H

        return latent_preds
    
class PrecoderTransformer(linen.Module):
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

    def setup(self):
        self.input_seq_len = 3 * self.context_len
    
    @linen.compact
    def __call__(self,
                 timesteps: jnp.ndarray,
                 states: jnp.ndarray,
                 latent: jnp.ndarray,
                 actions: jnp.ndarray,
                 next_controlled_variables: jnp.ndarray,
                 returns_to_go: jnp.ndarray) -> jnp.ndarray:
        B, T, _ = states.shape

        positions = jnp.arange(self.context_len+1)[None,:].repeat(states.shape[0], axis=0)
        time_embeddings = linen.Embed(
            num_embeddings=self.context_len+1,
            features=self.h_dim)(positions)

        # time embeddings are treated similar to positional embeddings
        initial_state_embedding = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(states[:,0,:]) + time_embeddings[:,0,:]
        latent_embedding = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(latent) + time_embeddings[:,1,:]
        action_embeddings = linen.Dense(
            self.h_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(actions) + time_embeddings[:,2:,:]

        # concatenate initial state and controlled variables
        # (s_0, z_0, a_1, ..., y_t)
        # (B x [T + 1] x h_dim)
        h = jnp.concatenate((initial_state_embedding[:,None,:], latent_embedding[:,None,:], action_embeddings), axis=1)

        h = linen.LayerNorm(dtype=self.dtype)(h)

        # transformer and prediction
        for _ in range(self.n_blocks):
            h = Block(
                h_dim=self.h_dim,
                max_T=self.input_seq_len,
                n_heads=self.n_heads,
                drop_p=self.drop_p)(h)
            
        # get predictions
        action_out = linen.Dense(
            self.act_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(h)     # predict action variable a_t given s_0, z_0, a_0, ..., a_t-1

        return action_out

def make_transformer(state_dim: int,
                     act_dim: int,
                     controlled_variables_dim: int,
                     n_blocks: int,
                     h_dim: int,
                     context_len: int,
                     n_heads: int,
                     drop_p: float) -> DynamicsTransformer:
    """Creates a DynamicsTransformer model.
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
    module = DynamicsTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=drop_p)

    return module


def make_transformer_networks(state_dim: int,
                              act_dim: int,
                              controlled_variables_dim: int,
                              n_blocks: int,
                              h_dim: int,
                              context_len: int,
                              n_heads: int,
                              drop_p: float) -> FeedForwardModel:
    batch_size = 1
    dummy_timesteps = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
    dummy_states = jnp.zeros((batch_size, context_len, state_dim))
    dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
    dummy_rtg = jnp.zeros((batch_size, context_len, 1))

    def policy_model_fn():
        class PolicyModule(linen.Module):
            @linen.compact
            def __call__(self,
                         timesteps: jnp.ndarray,
                         states: jnp.ndarray,
                         actions: jnp.ndarray,
                         returns_to_go: jnp.ndarray):
                y_ps = make_transformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    controlled_variables_dim=controlled_variables_dim,
                    n_blocks=n_blocks,
                    h_dim=h_dim,
                    context_len=context_len,
                    n_heads=n_heads,
                    drop_p=drop_p)(timesteps, states, actions, returns_to_go)
                return y_ps

        policy_module = PolicyModule()
        policy = FeedForwardModel(
            init=lambda key: policy_module.init(
                key, dummy_timesteps, dummy_states, dummy_actions, dummy_rtg),
            apply=policy_module.apply)
        return policy
    return policy_model_fn()
