import argparse
import csv
import flax
import functools
#import gym
import gymnasium as gym
import jax
import optax
import os
import pickle
import random
import sys

import jax.numpy as jnp
import numpy as np

from datetime import datetime
from typing import Any, Dict, Tuple

from functools import partial

import wandb
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import os
os.environ["MUJOCO_GL"] = "osmesa"

from decision_transformer.dt.model import make_transformer_networks, VAE, Transformer
from decision_transformer.dt.utils import ReplayBuffer, TrainingState, Transition
from decision_transformer.dt.utils import discount_cumsum, save_params, load_params
from decision_transformer.pmap import bcast_local_devices, synchronize_hosts, is_replicated

import jax
print(jax.devices())
from jax import config
config.update('jax_disable_jit', False)
config.update('jax_debug_nans', False)
config.update('jax_enable_x64', False)

def train(args):

    dataset = args.dataset          # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale      # normalize returns to go

    # use v3 env for evaluation because
    # Decision Transformer paper evaluates results on v3 envs

    if args.env == 'walker2d':
        env_name = 'Walker2d-v3'
        rtg_target = args.rtg_target if args.rtg_target is not None else 5000
        env_d4rl_name = f'walker2d-{dataset}-v2'

    elif args.env == 'halfcheetah':
        #env_name = 'HalfCheetah-v3'
        env_name = 'HalfCheetah-v4'
        rtg_target = args.rtg_target if args.rtg_target is not None else 6000
        env_d4rl_name = f'halfcheetah-{dataset}-v2'

    elif args.env == 'hopper':
        env_name = 'Hopper-v3'
        rtg_target = args.rtg_target if args.rtg_target is not None else 3600
        env_d4rl_name = f'hopper-{dataset}-v2'

    else:
        raise NotImplementedError

    env = gym.make(env_name)

    # device settings
    max_devices_per_host = args.max_devices_per_host
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    print(f'Device count: {jax.device_count()}, process count: {process_count} (id {process_id}), local device count: {local_device_count}, devices to be used count: {local_devices_to_use}')

    # seed for jax
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    global_key, global_key_vae, local_key, test_key = jax.random.split(key, 4)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    # seed for others
    random.seed(seed)
    np.random.seed(seed)
    #env.seed(seed)

    batch_size = args.batch_size            # training batch size
    batch_size_per_device = batch_size // local_devices_to_use
    grad_updates_per_step = args.grad_updates_per_step

    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    # load data from this file
    env_d4rl_name = 'relocate-expert-v1'
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}-qpos.pkl'

    # load dataset
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # to get status
    max_epi_len = -1
    min_epi_len = 10**6
    state_stats = []
    next_controlled_variables_stats = []
    for traj in trajectories:
        traj_len = traj['observations'].shape[0]
        min_epi_len = min(min_epi_len, traj_len)
        max_epi_len = max(max_epi_len, traj_len)
        state_stats.append(traj['observations'])
        next_controlled_variables_stats.append(traj['next_controlled_variables'])
        # convert
        traj['actions'] = jnp.array(traj['actions'])
        traj['observations'] = jnp.array(traj['observations'])
        traj['next_controlled_variables'] = jnp.array(traj['next_controlled_variables'])
        # calculate returns to go and rescale them
        traj['returns_to_go'] = jnp.array(discount_cumsum(traj['rewards'], 1.0) / rtg_scale).reshape(-1, 1)
        traj['timesteps'] = jnp.arange(start=0, stop=traj_len, step=1, dtype=jnp.int32).reshape(-1, 1)
        traj['traj_mask'] = jnp.ones(traj_len).reshape(-1, 1)

    # used for input normalization
    state_stats = jnp.concatenate(state_stats, axis=0)
    state_mean, state_std = jnp.mean(state_stats, axis=0), jnp.std(state_stats, axis=0) + 1e-8

    next_controlled_variables_stats = jnp.concatenate(next_controlled_variables_stats, axis=0)
    controlled_variables_mean, controlled_variables_std = jnp.mean(next_controlled_variables_stats, axis=0), jnp.std(next_controlled_variables_stats, axis=0) + 1e-6
    controlled_variables_dim = next_controlled_variables_stats.shape[-1]

    # state_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.shape[0]
    state_dim = trajectories[0]['observations'].shape[1]
    act_dim = trajectories[0]['actions'].shape[1]
    trans_dim = state_dim + act_dim + controlled_variables_dim + 1 + 1 + 1  # rtg, timesteps, mask

    # apply padding
    replay_buffer_data = []
    for traj in trajectories:
        traj_len = traj['observations'].shape[0]
        padding_len = (max_epi_len + context_len) - traj_len
        states = traj['observations']
        next_controlled_variables = traj['next_controlled_variables']

        # apply input normalization
        if not args.rm_normalization:
            states = (states - state_mean) / state_std
            next_controlled_variables = (next_controlled_variables - controlled_variables_mean) / controlled_variables_std

        states = jnp.concatenate([states, jnp.zeros((padding_len, state_dim))], axis=0)
        actions = jnp.concatenate([traj['actions'], jnp.zeros((padding_len, act_dim))], axis=0)
        next_controlled_variables = jnp.concatenate([next_controlled_variables, jnp.zeros((padding_len, controlled_variables_dim))], axis=0)
        returns_to_go = jnp.concatenate([traj['returns_to_go'], jnp.zeros((padding_len, 1))], axis=0)
        timesteps = jnp.concatenate([traj['timesteps'], jnp.zeros((padding_len, 1))], axis=0)
        traj_mask = jnp.concatenate([traj['traj_mask'], jnp.zeros((padding_len, 1))], axis=0)

        padding_data = jnp.concatenate([states, actions, next_controlled_variables, returns_to_go, timesteps, traj_mask], axis=-1)
        assert trans_dim == padding_data.shape[-1], padding_data.shape
        replay_buffer_data.append(padding_data)

    replay_buffer = ReplayBuffer(
        data=jnp.concatenate(replay_buffer_data, axis=0).reshape(local_devices_to_use, -1, max_epi_len + context_len, trans_dim)
    ) # (local_devices_to_use, num_epi, max_epi_len + context_len, trans_dim)

    import minari
    dataset = minari.load_dataset('D4RL/relocate/expert-v2')
    env = dataset.recover_environment(render_mode='rgb_array')

    prefix = "dt_" + env_d4rl_name
    start_time_str = '25-06-19-17-34-48'
    log_dir = os.path.join(args.log_dir, prefix, f'seed_{seed}', start_time_str)
    total_updates = 100000
    for model in ['vae', 'emp']:

        if model == 'vae':
            load_model_path = os.path.join(log_dir, model + '_model.pt')
            load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
            _vae_params = load_params(load_current_model_path)
            loaded_params = _vae_params['params']['decoder']
        elif model == 'emp':
            load_model_path = os.path.join(log_dir, model + '_model.pt')
            load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
            _vae_params = load_params(load_current_model_path)
            loaded_params = _vae_params['params']['precoder']

        batch_size = 1
        dummy_timesteps = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
        dummy_states = jnp.zeros((batch_size, context_len, state_dim))
        dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
        if args.trajectory_version:
            dummy_latent = jnp.zeros((batch_size, context_len * controlled_variables_dim))
            dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
        else:
            dummy_latent = jnp.zeros((batch_size, controlled_variables_dim))
            dummy_controlled_variables = jnp.zeros((batch_size, 1, controlled_variables_dim))
        dummy_rtg = jnp.zeros((batch_size, context_len, 1))
        dummy_horizon= jnp.ones((batch_size, 1), dtype=jnp.int32)
        dummy_mask = jnp.ones((batch_size, context_len, 1))

        precoder = Transformer(state_dim=state_dim,
                                act_dim=act_dim,
                                controlled_variables_dim=controlled_variables_dim,
                                n_blocks=n_blocks,
                                h_dim=embed_dim,
                                context_len=context_len,
                                n_heads=n_heads,
                                drop_p=dropout_p,
                                transformer_type='action_decoder')

        dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros((1,6)), scale_diag=jnp.ones((1,6)))

        def normalize_obs(obs):
            norm_obs = (obs - state_mean) / state_std
            return norm_obs

        actions = jnp.zeros((batch_size, context_len, act_dim))
        env.reset()
        # s_t = replay_buffer.data[0,:1,:1,:state_dim]
        s_t = jnp.concatenate((env.unwrapped.get_env_state()['qpos'], env.unwrapped.get_env_state()['qpos']))[None, None, :]
        key = jax.random.PRNGKey(seed)

        import imageio
        frames = []
        for t in range(200):
            frames.append(env.render())
            sample_key, dropout_key, key = jax.random.split(key, 3)
            z_t = dist_z_prior.sample(seed=sample_key)
            s_t = jnp.concatenate((env.unwrapped.get_env_state()['qpos'], env.unwrapped.get_env_state()['qpos']))[None, None, :]
            a_dist_params = precoder.apply({'params': loaded_params},
                                    dummy_timesteps,
                                    normalize_obs(s_t),
                                    z_t,
                                    actions,
                                    dummy_controlled_variables,
                                    dummy_rtg,
                                    (jnp.ones((1,1))*context_len).astype(jnp.int32),
                                    rngs={'dropout': dropout_key})
            a_mean, _ = jnp.split(a_dist_params, 2, axis=-1)
            obs, rew, terminated, truncated, info = env.step(jnp.tanh(a_mean[0, 0, :]))

        # for t in range(context_len):
        #     sample_key, dropout_key, key = jax.random.split(key, 3)
        #     z_t = dist_z_prior.sample(seed=sample_key)
        #     a_dist_params = precoder.apply({'params': loaded_params},
        #                             dummy_timesteps,
        #                             normalize_obs(s_t),
        #                             z_t,
        #                             actions,
        #                             dummy_controlled_variables,
        #                             dummy_rtg,
        #                             (jnp.ones((1,1))*context_len).astype(jnp.int32),
        #                             rngs={'dropout': dropout_key})
        #     a_mean, _ = jnp.split(a_dist_params, 2, axis=-1)
        #     actions = actions.at[:,t,:].set(jnp.tanh(a_mean[:, t, :]))
        
        # for t in range(context_len):
        #     sample_key, dropout_key, key = jax.random.split(key, 3)
        #     z_t = dist_z_prior.sample(seed=sample_key)
        #     a_dist_params = precoder.apply({'params': loaded_params},
        #                             dummy_timesteps,
        #                             normalize_obs(s_t),
        #                             z_t,
        #                             actions,
        #                             dummy_controlled_variables,
        #                             dummy_rtg,
        #                             (jnp.ones((1,1))*context_len).astype(jnp.int32),
        #                             rngs={'dropout': dropout_key})
        #     a_mean, _ = jnp.split(a_dist_params, 2, axis=-1)
        #     actions = actions.at[:,t,:].set(jnp.tanh(a_mean[:, t, :]))

        # import imageio
        # frames = []
        # for t in range(context_len):
        #     frames.append(env.render())
        #     obs, rew, terminated, truncated, info = env.step(actions[0,t,:])

        imageio.mimsave('output_video_' + model + '.mp4', frames, fps=30)
        
        from matplotlib import pyplot as plt
        for i in range(actions.shape[-1]):
            plt.plot(actions[0,:,i])
        plt.savefig(model+'action.png')
    
    breakpoint()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)
    parser.add_argument('--rtg_target', type=int, default=None)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.)
    parser.add_argument('--gradient_clipping', type=float, default=0.25)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_updates_per_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=1_000)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)
    parser.add_argument('--dynamics_save_iters', type=int, default=100)
    parser.add_argument('--vae_save_iters', type=int, default=100)
    parser.add_argument('--rm_normalization', action='store_true', help='Turn off input normalization')

    parser.add_argument('--max_devices_per_host', type=int, default=None)

    parser.add_argument('--trajectory_version', type=bool, default=False)

    args = parser.parse_args()

    train(args)
