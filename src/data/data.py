import jax.numpy as jnp
import numpy as np

import minari
from dataclasses import replace

from src.utils.utils import ReplayBuffer, discount_cumsum, get_local_devices_to_use, standardise_data

def get_dataset(args):

    minari_dataset = minari.load_dataset(args.env_d4rl_name)
    minari_env = minari_dataset.recover_environment(render_mode='rgb_array')
    learned_minari_env = minari_dataset.recover_environment(render_mode='rgb_array')

    # make hand and ball position relative to initial position of hand
    target_agnostic_minari_dataset = [replace(episode,
                                                observations=episode.observations - np.concatenate((np.zeros(33),
                                                                                                    episode.observations[0,33:36],
                                                                                                    episode.observations[0,33:36]))[None])
                                                for episode in minari_dataset]
    
    # extract obs and episode length range
    max_epi_len = -1
    min_epi_len = 10**6
    obs_stats = []
    for episode in target_agnostic_minari_dataset:
        traj_len = episode.observations.shape[0]-1
        min_epi_len = min(min_epi_len, traj_len)
        max_epi_len = max(max_epi_len, traj_len)
        obs_stats.append(episode.observations)

    obs_dim = episode.observations.shape[1]
    act_dim = episode.actions.shape[1]
    trans_dim = (obs_dim + act_dim + obs_dim
                 + 1 # rtg
                 + 1 # timesteps
                 + 1 # mask
                 + obs_dim + obs_dim)

    # obs mean and std
    obs_stats = jnp.concatenate(obs_stats, axis=0)
    obs_mean, obs_std = jnp.mean(obs_stats, axis=0), jnp.std(obs_stats, axis=0) + 1e-8

    # delta obs mean, std, max, and min
    delta_obs_stats = [ob[1:]-ob[:-1] for ob in obs_stats]
    delta_obs_stats = jnp.concatenate(delta_obs_stats, axis=0)
    delta_obs_mean, delta_obs_std = jnp.mean(delta_obs_stats, axis=0), jnp.std(delta_obs_stats, axis=0) + 1e-8
    delta_obs_min, delta_obs_max = jnp.min(delta_obs_stats, axis=0), jnp.max(delta_obs_stats, axis=0)
    delta_obs_min = standardise_data(delta_obs_min, delta_obs_mean, delta_obs_std)
    delta_obs_max = standardise_data(delta_obs_max, delta_obs_mean, delta_obs_std)

    delta_obs_scale = delta_obs_std / obs_std
    delta_obs_shift = delta_obs_mean / obs_std

    replay_buffer_data = []
    for episode in target_agnostic_minari_dataset:

        obs = jnp.array(episode.observations[:-1,:])
        next_obs = jnp.array(episode.observations[1:,:])
        delta_obs = next_obs-obs
        prev_obs = jnp.concatenate([jnp.array(episode.observations[:1,:]), # assume first previous step same as first step
                                    jnp.array(episode.observations[:-2,:])], axis=0)

        # standardise data
        obs = standardise_data(obs, obs_mean, obs_std)
        next_obs = standardise_data(next_obs, obs_mean, obs_std)
        delta_obs = standardise_data(delta_obs, delta_obs_mean, delta_obs_std)
        prev_obs = standardise_data(prev_obs, obs_mean, obs_std)

        # apply padding
        traj_len = episode.observations.shape[0]-1
        padding_len = (max_epi_len + args.context_len) - traj_len
        obs = jnp.concatenate([obs,
                               jnp.zeros((padding_len, obs_dim))], axis=0)
        
        actions = jnp.concatenate([jnp.array(episode.actions),
                                   jnp.zeros((padding_len, act_dim))], axis=0)
        
        next_obs = jnp.concatenate([next_obs,
                                    jnp.zeros((padding_len, obs_dim))], axis=0)
        
        delta_obs = jnp.concatenate([delta_obs,
                                    jnp.zeros((padding_len, obs_dim))], axis=0)
        
        prev_obs = jnp.concatenate([prev_obs,
                                    jnp.zeros((padding_len, obs_dim))], axis=0)
        
        returns_to_go = jnp.concatenate([jnp.array(discount_cumsum(episode.rewards, 1.0) / args.rtg_scale).reshape(-1, 1), 
                                         jnp.zeros((padding_len, 1))], axis=0)
        
        timesteps = jnp.concatenate([jnp.arange(start=0, stop=traj_len, step=1, dtype=jnp.int32).reshape(-1, 1),
                                     jnp.zeros((padding_len, 1))], axis=0)
        
        traj_mask = jnp.concatenate([jnp.ones(traj_len).reshape(-1, 1),
                                     jnp.zeros((padding_len, 1))], axis=0)

        padding_data = jnp.concatenate([obs,
                                        actions,
                                        next_obs,
                                        returns_to_go,
                                        timesteps,
                                        traj_mask,
                                        prev_obs,
                                        delta_obs], axis=-1)
        
        assert trans_dim == padding_data.shape[-1], padding_data.shape
        
        replay_buffer_data.append(padding_data)

    cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1, obs_dim, obs_dim])

    local_devices_to_use = get_local_devices_to_use(args)

    replay_buffer = ReplayBuffer(
        data=jnp.concatenate(replay_buffer_data, axis=0).reshape(local_devices_to_use, -1, max_epi_len + args.context_len, trans_dim)
    ) # (local_devices_to_use, num_epi, max_epi_len + context_len, trans_dim)

    d_args = {"obs_dim": obs_dim,
                "act_dim": act_dim,
                "trans_dim": trans_dim,
                "cumsum_dims": cumsum_dims,
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "delta_obs_min": delta_obs_min,
                "delta_obs_max": delta_obs_max,
                "delta_obs_scale": delta_obs_scale,
                "delta_obs_shift": delta_obs_shift,
                "max_epi_len": max_epi_len
                }

    return replay_buffer, minari_dataset, minari_env, learned_minari_env, d_args