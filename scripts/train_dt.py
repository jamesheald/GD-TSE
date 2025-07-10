# transformers for MDPS
# https://openreview.net/pdf?id=NHMuM84tRT - LONG SHORT
# https://openreview.net/pdf?id=af2c8EaKl8 - CONV


import os
os.environ["MUJOCO_GL"] = "egl"

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
from typing import Any, Dict, Tuple, List

from functools import partial

import wandb
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from decision_transformer.dt.model import make_transformer_networks, VAE, empowerment, Transformer, dynamics, MLP_precoder, AutonomousGRU, MLP
from decision_transformer.dt.utils import ReplayBuffer, TrainingState, Transition
from decision_transformer.dt.utils import discount_cumsum, save_params, load_params
from decision_transformer.pmap import bcast_local_devices, synchronize_hosts, is_replicated

import minari
import imageio
from matplotlib import pyplot as plt

from dataclasses import replace

import jax
print(jax.devices())
from jax import config
config.update('jax_disable_jit', False)
config.update('jax_debug_nans', True)
config.update('jax_enable_x64', False)

def train(args):

    # breakpoint()

    # import mujoco
    # model = minari_env.unwrapped.model
    # for j in range(model.njnt): print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))

    controlled_variables_dim = 3
    controlled_variables = [36 for i in range(controlled_variables_dim)] # hand pos and rot

    rtg_scale = args.rtg_scale      # normalize returns to go

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
    global_key, global_key_vae, global_key_emp, local_key, test_key = jax.random.split(key, 5)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    random.seed(seed)
    np.random.seed(seed)

    # batch_size = args.batch_size            # training batch size
    # batch_size_per_device = batch_size // local_devices_to_use
    dynamics_batch_size_per_device = args.dynamics_batch_size // local_devices_to_use
    vae_batch_size_per_device = args.vae_batch_size // local_devices_to_use
    emp_batch_size_per_device = args.emp_batch_size // local_devices_to_use
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
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}-fullnextstate.pkl'
    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.resume_start_time_str is None:
        prefix = "dt_" + env_d4rl_name
        start_time = datetime.now().replace(microsecond=0)
        start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
        log_dir = os.path.join(log_dir, prefix, f'seed_{seed}', start_time_str)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        prefix = "dt_" + env_d4rl_name
        log_dir = os.path.join(args.log_dir, prefix, f'seed_{seed}', args.resume_start_time_str)
        start_time = datetime.now().replace(microsecond=0)
        start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    log_csv_name = "log.csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = ([
        "duration",
        "num_updates",
        "action_loss"
    ])


    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("dataset path: " + dataset_path)
    print("log csv save path: " + log_csv_path)

    minari_dataset = minari.load_dataset('D4RL/relocate/expert-v2')
    minari_env = minari_dataset.recover_environment(render_mode='rgb_array')
    learned_minari_env = minari_dataset.recover_environment(render_mode='rgb_array')

    # make hand and ball position relative to initial position of hand
    target_agnostic_minari_dataset = [replace(episode,
                                              observations=episode.observations - np.concatenate((np.zeros(33),
                                                                                                  episode.observations[0,33:36],
                                                                                                  episode.observations[0,33:36]))[None])
                                              for episode in minari_dataset]

    # to get status
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
    trans_dim = obs_dim + act_dim + obs_dim + 1 + 1 + 1 + obs_dim # rtg, timesteps, mask

    # used for input normalization
    obs_stats = jnp.concatenate(obs_stats, axis=0)
    obs_mean, obs_std = jnp.mean(obs_stats, axis=0), jnp.std(obs_stats, axis=0) + 1e-8

    # apply padding
    replay_buffer_data = []
    for episode in target_agnostic_minari_dataset:
        traj_len = episode.observations.shape[0]-1
        padding_len = (max_epi_len + context_len) - traj_len
        obs = jnp.array(episode.observations[:-1,:])
        next_obs = jnp.array(episode.observations[1:,:])
        prev_obs = jnp.concatenate([#jnp.zeros((1, obs_dim)),
                                    jnp.array(episode.observations[:1,:]), # assume first previous step same as first step
                                    jnp.array(episode.observations[:-2,:])], axis=0)

        # apply input normalization
        if not args.rm_normalization:
            obs = (obs - obs_mean) / obs_std
            next_obs = (next_obs - obs_mean) / obs_std
            prev_obs = (prev_obs - obs_mean) / obs_std

        obs = jnp.concatenate([obs,
                               jnp.zeros((padding_len, obs_dim))], axis=0)
        actions = jnp.concatenate([jnp.array(episode.actions),
                                   jnp.zeros((padding_len, act_dim))], axis=0)
        next_obs = jnp.concatenate([next_obs,
                                    jnp.zeros((padding_len, obs_dim))], axis=0)
        prev_obs = jnp.concatenate([prev_obs,
                                    jnp.zeros((padding_len, obs_dim))], axis=0)
        returns_to_go = jnp.concatenate([jnp.array(discount_cumsum(episode.rewards, 1.0) / rtg_scale).reshape(-1, 1), 
                                         jnp.zeros((padding_len, 1))], axis=0)
        timesteps = jnp.concatenate([jnp.arange(start=0, stop=traj_len, step=1, dtype=jnp.int32).reshape(-1, 1),
                                     jnp.zeros((padding_len, 1))], axis=0)
        traj_mask = jnp.concatenate([jnp.ones(traj_len).reshape(-1, 1),
                                     jnp.zeros((padding_len, 1))], axis=0)

        padding_data = jnp.concatenate([obs, actions, next_obs, returns_to_go, timesteps, traj_mask, prev_obs], axis=-1)
        assert trans_dim == padding_data.shape[-1], padding_data.shape
        replay_buffer_data.append(padding_data)

    replay_buffer = ReplayBuffer(
        data=jnp.concatenate(replay_buffer_data, axis=0).reshape(local_devices_to_use, -1, max_epi_len + context_len, trans_dim)
    ) # (local_devices_to_use, num_epi, max_epi_len + context_len, trans_dim)

    ###################################### evaluation ###################################### 

    def get_mean_and_log_std(x, min_log_std = -20., max_log_std = 2.):
        x_mean, x_log_std = jnp.split(x, 2, axis=-1)
        x_log_std = jnp.clip(x_log_std, min_log_std, max_log_std)
        return x_mean, x_log_std

    def normalize_obs(obs):
        norm_obs = (obs - obs_mean) / obs_std
        return norm_obs
    
    def unnormalize_obs(norm_obs):
        obs = norm_obs * obs_std + obs_mean
        return obs

    # minari_dataset[0].observations[0,:30]-replay_buffer.data[0,0,0,:30]* state_std[:30] + state_mean[:30]
    # breakpoint()

    batch_size = 1
    eval_dummy_timesteps = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
    if args.trajectory_version:
        dummy_latent = jnp.zeros((batch_size, context_len * controlled_variables_dim))
        eval_dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
    else:
        dummy_latent = jnp.zeros((batch_size, controlled_variables_dim))
        eval_dummy_controlled_variables = jnp.zeros((batch_size, 1, controlled_variables_dim))
    eval_dummy_rtg = jnp.zeros((batch_size, context_len, 1))

    dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros((1,controlled_variables_dim)), scale_diag=jnp.ones((1,controlled_variables_dim)))

    eval_encoder = MLP(out_dim=controlled_variables_dim*2,
                       h_dims=[256,256])

    # eval_precoder = MLP(out_dim=act_dim,
    #                     h_dims=[256,256])

    eval_precoder = AutonomousGRU(act_dim=act_dim,
                                  context_len=context_len,
                                  hidden_size=128)

    prior = MLP(out_dim=controlled_variables_dim*2,
                          h_dims=[256,256])
    
    def get_actions(obs, key, precoder_params):

        # s_t = jnp.concatenate((minari_env.unwrapped.get_env_state()['qpos'], minari_env.unwrapped.get_env_state()['qpos']))[None, :]

        # # sample from posterior      
        # cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1])
        # y_t = replay_buffer.data[0,ep,t,cumsum_dims[1]:cumsum_dims[2]][jnp.array(controlled_variables)][None]
        # z_dist_params = jax.jit(eval_encoder.apply)(encoder_params, jnp.concatenate([normalize_obs(s_t), y_t], axis=-1))
        # z_mean, z_log_std = get_mean_and_log_std(z_dist_params)
        # dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_log_std))
        # z_t = dist_z_post.sample(seed=key)

        target_agnostic_obs = obs - np.concatenate((np.zeros(33),
                                                    obs[33:36],
                                                    obs[33:36]))

        target_agnostic_obs = normalize_obs(target_agnostic_obs)

        target_agnostic_obs = jnp.concatenate((target_agnostic_obs, target_agnostic_obs), axis=-1)
        
        # prior_z_params = jax.jit(prior.apply)(prior_params, target_agnostic_obs[None])
        # prior_mean, prior_log_std = jnp.split(prior_z_params, 2, axis=-1)
        # min_log_std = -20.
        # max_log_std = 2.
        # prior_log_std = jnp.clip(prior_log_std, min_log_std, max_log_std)
        # dist_z_prior = tfd.MultivariateNormalDiag(loc=prior_mean, scale_diag=jnp.exp(prior_log_std))
        # z_t = dist_z_prior.sample(seed=key)

        # sample from prior
        z_t = dist_z_prior.sample(seed=key)
        
        actions = jax.jit(eval_precoder.apply)(precoder_params, target_agnostic_obs[None,None, :], z_t)

        return actions[0,:,:]
    
    def eval_model(model, params, iter, loop='open'):

        # prior_params = {'params': params['params']['prior']}
        encoder_params = {'params': params['params']['encoder']}
        precoder_params = {'params': params['params']['precoder']}

        n_rollouts = 3
        for rollout in range(n_rollouts):
            obs, _ = minari_env.reset()
            # ep = 0
            # initial_state = unnormalize_obs(replay_buffer.data[0,ep,0,:72])
            # minari_env.unwrapped.set_state(qpos=initial_state[:36], qvel=initial_state[36:72])
            frames = []
            key = jax.random.PRNGKey(seed)
            actions = get_actions(obs, key, precoder_params)
            for t in range(50):
                # key, subkey = jax.random.split(key)
                # action = get_actions(obs, ep, t, subkey)[0,0,:]
                frames.append(minari_env.render())
                # cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1])
                # action = replay_buffer.data[0,ep,t,cumsum_dims[0]:cumsum_dims[1]]
                obs, rew, terminated, truncated, info = minari_env.step(actions[t,:])

            imageio.mimsave(os.path.join(log_dir, 'output_video_' + loop + '_' + model + '_' + str(iter) + '_' + str(rollout) + '.mp4'), frames, fps=30)

        plt.figure()
        for i in range(actions.shape[-1]):
            plt.plot(actions[:,i])
        plt.savefig(os.path.join(log_dir, 'actions_' + loop + '_' + model + '_' + str(iter) + '_' + str(rollout) + '.png'))
        plt.close()

        return None

    ###################################### evaluate dyanmics ###################################### 

    def eval_dynamics(key, precoder_params, dynamics_apply, dynamics_params, iter):

        # obs, _ = minari_env.reset()
        # x = minari_env.step(minari_env.action_space.sample())
        # minari_env.unwrapped.get_env_state()['qpos'][:30]-x[0][:30] = 0
        # (Pdb) np.isclose(obs[36:]-obj_pos_0, minari_env.unwrapped.get_env_state()['qpos'][30:33]-qpos_ob_0)
        # array([ True,  True,  True])
        # (Pdb) np.isclose(obs[36:]-obj_pos_0, minari_env.unwrapped.get_env_state()['qpos'][30:33])
        # array([ True,  True,  True])

        def get_predicted_obs(obs, key, actions, dynamics_params):

            def peform_rollout(state, key, actions, dynamics_params):
                
                def step_fn(carry, action):
                    state, key, dynamics_params = carry
                    key, dropout_key, sample_i_key, sample_s_key = jax.random.split(key, 4)
                    
                    # multiple samples
                    dropout_keys = jax.random.split(dropout_key, args.n_dynamics_ensembles)
                    s_dist_params = jax.vmap(dynamics_apply, in_axes=(0,None,None,0))(dynamics_params, state, action, dropout_keys)
                    # s_dist_params = dynamics_apply(dynamics_params, state, action, dropout_key)
                    s_mean, s_log_std = get_mean_and_log_std(s_dist_params)
                    disagreement = jnp.var(s_mean, axis=0).mean()

                    idx = jax.random.categorical(sample_i_key, jnp.ones(args.n_dynamics_ensembles), axis=-1)
                    s_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))

                    delta_s = s_dist.sample(seed=sample_s_key)
                    # next_state = state + delta_s
                    s_curr = state[...,obs_dim:]
                    s_next = s_curr + delta_s
                    s_next_mean = s_curr + s_mean.mean(axis=0)

                    next_state = jnp.concatenate([s_curr, s_next], axis=-1)

                    carry = next_state, key, dynamics_params
                    return carry, (s_curr, s_next_mean, s_dist_params, disagreement)

                carry = state, key, dynamics_params
                _, (s_curr, s_next_mean, s_dist_params, disagreement) = jax.lax.scan(step_fn, carry, actions)
                
                return s_curr, s_next_mean, s_dist_params, disagreement
            
            batch_peform_rollout = jax.vmap(peform_rollout, in_axes=(0,0,0,None))

            target_agnostic_obs = obs - np.concatenate((np.zeros(33),
                                            obs[33:36],
                                            obs[33:36]))

            target_agnostic_obs = normalize_obs(target_agnostic_obs)

            target_agnostic_obs = jnp.concatenate((target_agnostic_obs, target_agnostic_obs), axis=-1)

            dynamics_keys = jax.random.split(key, actions.shape[0])
            _, predicted_obs_traj, _, _ = batch_peform_rollout(target_agnostic_obs[None], dynamics_keys, actions, dynamics_params)

            return predicted_obs_traj

        n_rollouts = 3
        for rollout in range(n_rollouts):
            obs, _ = minari_env.reset()
            _ = learned_minari_env.reset()
            env_state = {'obj_pos': minari_env.unwrapped.get_env_state()['obj_pos'],
                         'qpos': minari_env.unwrapped.get_env_state()['qpos'],
                         'qvel': minari_env.unwrapped.get_env_state()['qvel'],
                         'target_pos': minari_env.unwrapped.get_env_state()['target_pos']}
            learned_minari_env.unwrapped.set_env_state(env_state)
            frames = []
            key = jax.random.PRNGKey(seed)
            actions = get_actions(obs, key, precoder_params)
            predicted_obs_traj = get_predicted_obs(obs, key, actions[None], dynamics_params)
            predicted_obs_traj = unnormalize_obs(predicted_obs_traj)
            predicted_obs_traj += np.concatenate((np.zeros(33),
                                                    obs[33:36],
                                                    obs[33:36]))[None,None]
            predicted_obs_traj = np.array(predicted_obs_traj)
            qpos_obj_pos = predicted_obs_traj[:, :, 36:] - obs[None,None,36:]
            qpos = np.zeros(36)
            frames.append(np.concatenate((minari_env.render(), learned_minari_env.render()), axis=1))
            for t in range(args.context_len):
                qpos[:30] = predicted_obs_traj[0, t, :30]
                qpos[30:33] = qpos_obj_pos[0, t, :]
                learned_minari_env.unwrapped.set_state(qpos=qpos, qvel=qpos*0.)
                _ = minari_env.step(actions[t,:])
                frames.append(np.concatenate((minari_env.render(), learned_minari_env.render()), axis=1))

            imageio.mimsave(os.path.join(log_dir, 'learned_dynamics_video_' + str(iter) + '_' + str(rollout) + '.mp4'), frames, fps=30)

    ###################################### dynamics training ###################################### 

    dynamics_model = dynamics(
        h_dims_dynamics=args.h_dims_dynamics,
        state_dim=obs_dim,
        drop_out_rates=args.dynamics_dropout_rates
    )

    if args.resume_start_time_str is not None and args.resume_dynamics is False:

        total_updates = 300000
        load_model_path = os.path.join(log_dir, "dynamics_model.pt")
        load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
        _dynamics_params = load_params(load_current_model_path)

        # dynamics_model = make_transformer_networks(
        #     state_dim=obs_dim,
        #     act_dim=act_dim,
        #     controlled_variables_dim=controlled_variables_dim,
        #     n_blocks=n_blocks,
        #     h_dim=embed_dim,
        #     context_len=context_len,
        #     n_heads=n_heads,
        #     drop_p=args.dynamics_dropout_p,
        #     trajectory_version=args.trajectory_version,
        #     transformer_type='dynamics'
        # )

        wandb.init(
                name=f'{env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
                group=env_d4rl_name,
                project='jax_dt',
                config=args
            )

    else:

        if args.resume_start_time_str is None or args.resume_dynamics is False:

            # dynamics_model = make_transformer_networks(
            #     state_dim=obs_dim,
            #     act_dim=act_dim,
            #     controlled_variables_dim=controlled_variables_dim,
            #     n_blocks=n_blocks,
            #     h_dim=embed_dim,
            #     context_len=context_len,
            #     n_heads=n_heads,
            #     drop_p=args.dynamics_dropout_p,
            #     trajectory_version=args.trajectory_version,
            #     transformer_type='dynamics'
            # )

            schedule_fn = optax.polynomial_schedule(
                init_value=lr * 1 / warmup_steps,
                end_value=lr,
                power=1,
                transition_steps=warmup_steps,
                transition_begin=0
            )
            dynamics_optimizer = optax.chain(
                optax.clip(args.gradient_clipping),
                optax.adamw(learning_rate=schedule_fn, weight_decay=wt_decay),
            )

            # batch_size = 1
            dummy_states = jnp.zeros((batch_size, obs_dim))
            dummy_actions = jnp.zeros((batch_size, act_dim))
            s_tm1_s_t = jnp.concatenate([dummy_states, dummy_states], axis=-1)
            key_params, key_dropout, key = jax.random.split(global_key, 3)
            dynamics_params_list = []
            # create ensemble of dynamics models
            for i in range(args.n_dynamics_ensembles):
                # dynamics_params_list.append(dynamics_model.init({'params': key_params, 'dropout': key_dropout}))
                dynamics_params_list.append(dynamics_model.init({'params': key_params}, s_tm1_s_t, dummy_actions, key_dropout))
                key_params, key_dropout, key = jax.random.split(key, 3)

            dynamics_params = jax.tree_util.tree_map(lambda *p: jnp.stack(p), *dynamics_params_list)

        elif args.resume_start_time_str is not None or args.resume_dynamics is True:

            dynamics_optimizer = optax.chain(
                optax.clip(args.gradient_clipping),
                optax.adamw(learning_rate=lr, weight_decay=wt_decay),
            )

            total_updates = 300000
            load_model_path = os.path.join(log_dir, "dynamics_model.pt")
            load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
            dynamics_params = load_params(load_current_model_path)

        dynamics_optimizer_state = jax.vmap(dynamics_optimizer.init)(dynamics_params)

        # count the number of parameters
        # param_count = sum(x.size for x in jax.tree_util.tree_leaves(dynamics_params_list[0]))
        # print(f'num_dynamics_param: {param_count}')

        dynamics_optimizer_state, dynamics_params = bcast_local_devices(
            (dynamics_optimizer_state, dynamics_params), local_devices_to_use)
        
        training_state = TrainingState(
            optimizer_state=dynamics_optimizer_state,
            params=dynamics_params,
            key=jax.random.split(local_key, args.n_dynamics_ensembles * local_devices_to_use).reshape(local_devices_to_use, args.n_dynamics_ensembles, -1),
            steps=jnp.zeros((local_devices_to_use, args.n_dynamics_ensembles)))

        def dynamics_loss(dynamics_params: Any,
                    transitions: Transition, key: jnp.ndarray) -> jnp.ndarray:
            s_t = transitions.s_t  # (batch_size_per_device, context_len, state_dim)
            a_t = transitions.a_t  # (batch_size_per_device, context_len, action_dim)
            s_tp1 = transitions.s_tp1  # (batch_size_per_device, context_len, state_dim)
            ts = transitions.ts.reshape(transitions.ts.shape[:2]).astype(jnp.int32)  # (batch_size_per_device, context_len)
            rtg_t = transitions.rtg_t  # (batch_size_per_device, context_len, 1)
            mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
            s_tm1 = transitions.s_tm1

            # horizon = mask.sum(axis=1).astype(jnp.int32) # (B, 1)
            # y_t = transitions.s_tp1[...,controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)
            # if args.trajectory_version:
            #     # y_t = transitions.s_tp1[...,controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)
            #     dummy_z_t = jnp.zeros((dynamics_batch_size_per_device, context_len * controlled_variables_dim))
            # else:
            #     # y_t = jnp.take_along_axis(s_tp1, horizon[..., None]-1, axis=1)[...,controlled_variables]
            #     dummy_z_t = jnp.zeros((dynamics_batch_size_per_device, controlled_variables_dim))

            # y_p = dynamics_model.apply(dynamics_params, ts, s_t, dummy_z_t, a_t, y_t, rtg_t, horizon, deterministic=True, rngs={'dropout': key})

            # def true_fn(y_mean, y_log_std, y_t):
            #     dist = tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=jnp.exp(y_log_std))
            #     return dist.log_prob(y_t)

            # def false_fn(y_mean, y_log_std, y_t):
            #     return 0.
            
            # def get_log_prob(mask, y_mean, y_log_std, y_t):
            #     log_prob = jax.lax.cond(mask, true_fn, false_fn, y_mean, y_log_std, y_t)
            #     return log_prob
            # batch_get_log_prob = jax.vmap(get_log_prob)

            # y_mean, y_log_std = jnp.split(y_p, 2, axis=-1)
            # min_log_std = -20.
            # max_log_std = 2.
            # y_log_std = jnp.clip(y_log_std, min_log_std, max_log_std)
            # y_mean = y_mean.reshape(-1, controlled_variables_dim)
            # y_log_std = y_log_std.reshape(-1, controlled_variables_dim)
            # delta_y_t = y_t - s_t[:,:1,controlled_variables]
            # delta_y_t = delta_y_t.reshape(-1, controlled_variables_dim)
            
            # # if args.trajectory_version:
            # #     valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)
            # #     log_probs = batch_get_log_prob(valid_mask, y_mean, y_log_std, delta_y_t)
            # #     loss = jnp.sum(-log_probs * valid_mask) / jnp.sum(valid_mask)
            # # else:
            # #     log_probs = jax.vmap(true_fn)(y_mean, y_log_std, delta_y_t)
            # #     loss = jnp.mean(-log_probs)

            # valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)
            # log_probs = batch_get_log_prob(valid_mask, y_mean, y_log_std, delta_y_t)
            # loss = jnp.sum(-log_probs * valid_mask) / jnp.sum(valid_mask)

            # loss /= controlled_variables_dim

            s_tm1_s_t = jnp.concatenate([s_tm1, s_t], axis=-1)
            s_p = dynamics_model.apply(dynamics_params, s_tm1_s_t, a_t, key)
            s_mean, s_log_std = jnp.split(s_p, 2, axis=-1)
            min_log_std = -20.
            max_log_std = 2. 
            s_log_std = jnp.clip(s_log_std, min_log_std, max_log_std)

            dist = tfd.MultivariateNormalDiag(loc=s_mean, scale_diag=jnp.exp(s_log_std))
            delta_s = s_tp1-s_t
            log_probs = dist.log_prob(delta_s)
            loss = jnp.mean(-log_probs)
            loss /= obs_dim 

            return loss 

        dynamics_grad = jax.jit(jax.value_and_grad(dynamics_loss))

        @jax.jit
        def update_step(
            state: TrainingState,
            transitions: jnp.ndarray,
        ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

            cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1, obs_dim])
            transitions = Transition(
                s_t=transitions[:, :, :cumsum_dims[0]],
                a_t=transitions[:, :, cumsum_dims[0]:cumsum_dims[1]],
                s_tp1=transitions[:, :, cumsum_dims[1]:cumsum_dims[2]],
                rtg_t=transitions[:, :, cumsum_dims[2]:cumsum_dims[3]],
                ts=transitions[:, :, cumsum_dims[3]:cumsum_dims[4]],
                mask_t=transitions[:, :, cumsum_dims[4]:cumsum_dims[5]],
                s_tm1=transitions[:, :, cumsum_dims[5]:cumsum_dims[6]]
            )

            key, key_dynamics = jax.random.split(state.key, 2)

            loss, dynamics_grads = dynamics_grad(state.params, transitions, key_dynamics)
            dynamics_grads = jax.lax.pmean(dynamics_grads, axis_name='i')
            dynamics_params_update, dynamics_optimizer_state = dynamics_optimizer.update(
                dynamics_grads, state.optimizer_state, state.params)
            dynamics_params = optax.apply_updates(state.params, dynamics_params_update)

            metrics = {'loss': loss}

            new_state = TrainingState(
                optimizer_state=dynamics_optimizer_state,
                params=dynamics_params,
                key=key,
                steps=state.steps + 1)
            return new_state, metrics

        def sample_data(training_state, replay_buffer, max_epi_len):
            # num_updates_per_iter
            key1, key2, key3 = jax.random.split(training_state.key, 3)
            epi_idx = jax.random.randint(
                key1, (int(dynamics_batch_size_per_device*grad_updates_per_step),),
                minval=0,
                maxval=replay_buffer.data.shape[0])  # from (0, num_epi)
            context_idx = jax.random.randint(
                key2, (int(dynamics_batch_size_per_device*grad_updates_per_step),),
                minval=0,
                maxval=max_epi_len)  # from (0, max_epi_len)

            def dynamic_slice_context(carry, x):
                traj, c_idx = x
                dynamics_context_len = 1
                return (), jax.lax.dynamic_slice(traj, (c_idx, 0), (dynamics_context_len, trans_dim))

            # (batch_size_per_device*num_updates_per_iter, max_epi_len + context_len, trans_dim)
            transitions = jnp.take(replay_buffer.data, epi_idx, axis=0, mode='clip')
            # (batch_size_per_device*num_updates_per_iter, context_len, trans_dim)
            _, transitions = jax.lax.scan(dynamic_slice_context, (), (transitions, context_idx))
            # (num_updates_per_iter, batch_size_per_device, context_len, trans_dim)
            transitions = jnp.reshape(transitions,
                                    [grad_updates_per_step, -1] + list(transitions.shape[1:]))

            training_state = training_state.replace(key=key3)
            return training_state, transitions

        def run_one_epoch(carry, unused_t, max_epi_len):
            training_state, replay_buffer = carry

            training_state, transitions = sample_data(training_state, replay_buffer, max_epi_len)
            training_state, metrics = jax.lax.scan(
                update_step, training_state, transitions, length=1)
            return (training_state, replay_buffer), metrics

        def run_training(training_state, replay_buffer, max_epi_len):
            synchro = is_replicated(
                training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')

            # vmap over ensemble
            vmapped_run_one_epoch = jax.vmap(
                lambda state, buffer: jax.lax.scan(
                    partial(run_one_epoch, max_epi_len=max_epi_len),
                    (state, buffer), None, length=num_updates_per_iter
                    ), in_axes=(0, None)
                )
            (training_state, _), metrics = vmapped_run_one_epoch(training_state, replay_buffer)

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return training_state, replay_buffer, metrics, synchro
        
        run_training = jax.pmap(partial(run_training, max_epi_len=max_epi_len), axis_name='i')

        total_updates = 0

        wandb.init(
                name=f'{env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
                group=env_d4rl_name,
                project='jax_dt',
                config=args
            )
    
        save_model_path = os.path.join(log_dir, "dynamics_model.pt")

        for i_train_iter in range(max_train_iters):
            log_dynamics_losses = []

            # optimization
            training_state, replay_buffer, training_metrics, synchro = run_training(
                training_state, replay_buffer)
            assert synchro[0], (current_step, training_state)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
            log_dynamics_losses.append(training_metrics['loss'])

            mean_dynamics_loss = np.mean(log_dynamics_losses)
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

            total_updates += num_updates_per_iter

            log_str = ("=" * 60 + '\n' +
                    "time elapsed: " + time_elapsed  + '\n' +
                    "train iter: " + str(i_train_iter)  + '\n' +
                    "num of updates: " + str(total_updates) + '\n' +
                    "dynamics loss: " +  format(mean_dynamics_loss, ".5f") + '\n'
                    )

            print(log_str)

            wandb.log({'mean_dynamics_loss': mean_dynamics_loss})

            log_data = [
                time_elapsed,
                total_updates,
                mean_dynamics_loss,
            ]

            csv_writer.writerow(log_data)

            # save model
            _dynamics_params = jax.tree_util.tree_map(lambda x: x[0], training_state.params)

            if i_train_iter % args.dynamics_save_iters == 0 or i_train_iter == max_train_iters - 1:
                save_current_model_path = save_model_path[:-3] + f"_{total_updates}.pt"
                print("saving current model at: " + save_current_model_path)
                save_params(save_current_model_path, _dynamics_params)

        synchronize_hosts()
        
        print("=" * 60)
        print("finished training dynamics!")
        print("=" * 60)
        end_time = datetime.now().replace(microsecond=0)
        time_elapsed = str(end_time - start_time)
        end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
        print("started training dynamics at: " + start_time_str)
        print("finished training dynamics at: " + end_time_str)
        print("total dynamics training time: " + time_elapsed)
        print("saved last updated model at: " + save_model_path)
        print("=" * 60)
        
        # cumsum_dims = np.cumsum([state_dim, act_dim, state_dim])
        # ep=0
        # s_t=replay_buffer.data[:, ep, :, :cumsum_dims[0]]
        # a_t=replay_buffer.data[:, ep, :, cumsum_dims[0]:cumsum_dims[1]]
        # s_tp1=replay_buffer.data[:, ep, :, cumsum_dims[1]:cumsum_dims[2]]
        # t=0
        # ensemble_keys = jax.random.split(jax.random.PRNGKey(0), 4)
        # pred = jax.vmap(dynamics_model.apply, in_axes=(0,None,None,0))(_dynamics_params, s_t[:,t,:], a_t[:,t,:], ensemble_keys)
        # s_mean, s_log_std = jnp.split(pred, 2, axis=-1)
        # min_log_std = -20.
        # max_log_std = 2. 
        # s_log_std = jnp.clip(s_log_std, min_log_std, max_log_std)
        # (s_tp1[:,t,controlled_variables]-s_t[:,t,controlled_variables])-s_mean[:,0,controlled_variables]
        # (abs((s_tp1[:,t,controlled_variables]-s_t[:,t,controlled_variables])-s_mean[:,0,controlled_variables])).mean()

        # minari_dataset = minari.load_dataset('D4RL/relocate/expert-v2')
        # minari_env = minari_dataset.recover_environment(render_mode='rgb_array')
        # minari_env.reset()
        # s_t = (jnp.concatenate((minari_env.unwrapped.get_env_state()['qpos'], minari_env.unwrapped.get_env_state()['qpos'])) - state_mean) / state_std
        # pred = jax.vmap(dynamics_model.apply, in_axes=(0,None,None,0))(_dynamics_params, s_t[None], a_t[:,10,:], ensemble_keys)
        # _ = minari_env.step(a_t[0,10,:])
        # s_tp1 = (jnp.concatenate((minari_env.unwrapped.get_env_state()['qpos'], minari_env.unwrapped.get_env_state()['qpos'])) - state_mean) / state_std
        # s_mean, s_log_std = jnp.split(pred, 2, axis=-1)
        # min_log_std = -20.
        # max_log_std = 2. 
        # s_log_std = jnp.clip(s_log_std, min_log_std, max_log_std)
        # idx = jnp.array(controlled_variables)
        # (s_tp1[idx]-s_t[idx])-s_mean[:,0,idx]
        # (abs((s_tp1[idx]-s_t[idx])-s_mean[:,0,idx])).mean()
        # breakpoint()

    ###################################### vae training ###################################### 

    vae_model = VAE(
        state_dim=obs_dim,
        act_dim=act_dim,
        controlled_variables=controlled_variables,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        n_dynamics_ensembles=args.n_dynamics_ensembles,
        trajectory_version=args.trajectory_version
    )
    
    if args.resume_start_time_str is None or args.resume_vae is False:

        schedule_fn = optax.polynomial_schedule(
            init_value=lr * 1 / warmup_steps,
            end_value=lr,
            power=1,
            transition_steps=warmup_steps,
            transition_begin=0
        )
        vae_optimizer = optax.chain(
            optax.clip(args.gradient_clipping),
            optax.adamw(learning_rate=schedule_fn, weight_decay=wt_decay),
        )

        batch_size = 1
        dummy_timesteps = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
        dummy_states = jnp.zeros((batch_size, context_len, obs_dim*2))
        dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
        if args.trajectory_version:
            dummy_latent = jnp.zeros((batch_size, context_len * controlled_variables_dim))
            # dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
        else:
            dummy_latent = jnp.zeros((batch_size, controlled_variables_dim))
            # dummy_controlled_variables = jnp.zeros((batch_size, 1, controlled_variables_dim))
        dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
        dummy_rtg = jnp.zeros((batch_size, context_len, 1))
        dummy_horizon= jnp.ones((batch_size, 1), dtype=jnp.int32)
        dummy_mask = jnp.ones((batch_size, context_len, 1))

        key_params, key_dropout = jax.random.split(global_key_vae)
        vae_params = vae_model.init({'params': key_params, 'dropout': key_dropout},
                                    ts=dummy_timesteps,
                                    s_t=dummy_states,
                                    z_t=dummy_latent,
                                    a_t=dummy_actions,
                                    y_t=dummy_controlled_variables,
                                    rtg_t=dummy_rtg,
                                    horizon=dummy_horizon,
                                    mask=dummy_mask,
                                    dynamics_apply=dynamics_model.apply,
                                    dynamics_params=_dynamics_params,
                                    key=key_params)
    
    else:

        vae_optimizer = optax.chain(
            optax.clip(args.gradient_clipping),
            optax.adamw(learning_rate=lr, weight_decay=wt_decay),
        )

        total_updates = 300000
        load_model_path = os.path.join(log_dir, "vae_model.pt")
        load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
        vae_params = load_params(load_current_model_path)
    
    vae_optimizer_state = vae_optimizer.init(vae_params)

    vae_optimizer_state, vae_params = bcast_local_devices(
        (vae_optimizer_state, vae_params), local_devices_to_use)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(vae_params))
    print(f'num_vae_param: {param_count}')

    def vae_loss(vae_params: Any,
                 transitions: Transition,
                 key: jnp.ndarray, #  ) -> jnp.ndarray:
                 w) -> jnp.ndarray:
        ts = transitions.ts.reshape(transitions.ts.shape[:2]).astype(jnp.int32)  # (batch_size_per_device, context_len)
        s_t = transitions.s_t  # (batch_size_per_device, context_len, state_dim)
        a_t = transitions.a_t  # (batch_size_per_device, context_len, action_dim)
        s_tp1 = transitions.s_tp1
        rtg_t = transitions.rtg_t  # (batch_size_per_device, context_len, 1)
        mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
        s_tm1 = transitions.s_tm1
        
        horizon = mask.sum(axis=1).astype(jnp.int32) # (B, 1)
        
        y_t = transitions.s_tp1[...,controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)
        if args.trajectory_version:
            # y_t = transitions.s_tp1[...,controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)
            dummy_z_t = jnp.zeros((vae_batch_size_per_device, context_len * controlled_variables_dim))
        else:
            # y_t = jnp.take_along_axis(s_tp1, horizon[..., None]-1, axis=1)[...,controlled_variables]
            dummy_z_t = jnp.zeros((vae_batch_size_per_device, controlled_variables_dim))

        vae_key, dropout_key = jax.random.split(key, 2)

        s_tm1_s_t = jnp.concatenate([s_tm1, s_t], axis=-1)

        kl_loss, action_decoder_loss, controlled_variable_decoder_loss, disagreement_loss = vae_model.apply(vae_params, ts, s_tm1_s_t, dummy_z_t, a_t, y_t, rtg_t, horizon, mask, dynamics_model.apply, _dynamics_params, vae_key, rngs={'dropout': dropout_key})

        # return kl_loss + action_decoder_loss + controlled_variable_decoder_loss + disagreement_loss * args.uncertainty_weight, (kl_loss, action_decoder_loss, controlled_variable_decoder_loss, disagreement_loss)
        return kl_loss + action_decoder_loss * (1-w) + controlled_variable_decoder_loss * w + disagreement_loss * args.uncertainty_weight, (kl_loss, action_decoder_loss, controlled_variable_decoder_loss, disagreement_loss)

    vae_grad = jax.jit(jax.value_and_grad(vae_loss, has_aux=True))

    @jax.jit
    def update_step_vae(
        state: TrainingState,
        transitions: jnp.ndarray,
    ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

        cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1, obs_dim])

        transitions = Transition(
            s_t=transitions[:, :, :cumsum_dims[0]],
            a_t=transitions[:, :, cumsum_dims[0]:cumsum_dims[1]],
            s_tp1=transitions[:, :, cumsum_dims[1]:cumsum_dims[2]],
            rtg_t=transitions[:, :, cumsum_dims[2]:cumsum_dims[3]],
            ts=transitions[:, :, cumsum_dims[3]:cumsum_dims[4]],
            mask_t=transitions[:, :, cumsum_dims[4]:cumsum_dims[5]],
            s_tm1=transitions[:, :, cumsum_dims[5]:cumsum_dims[6]]
        )

        key, key_vae = jax.random.split(state.key, 2)

        # w = jnp.where(state.steps/(num_updates_per_iter*1_000) < 0.5, 0., state.steps/(num_updates_per_iter*1_000) - 0.5)
        # w = jnp.clip(w, 0., 1.)

        w = state.steps/(num_updates_per_iter*1_000)
        if args.y_decoder_weight is None:
            w = jnp.clip(w, 0., 1.)
        else:
            w = jnp.clip(args.y_decoder_weight, 0., 1.)

        (loss, (kl_loss, a_decoder_loss, y_decoder_loss, disagreement_loss)), vae_grads = vae_grad(state.params, transitions, key_vae, w)
        vae_grads = jax.lax.pmean(vae_grads, axis_name='i')
        vae_params_update, vae_optimizer_state = vae_optimizer.update(
            vae_grads, state.optimizer_state, state.params)
        vae_params = optax.apply_updates(state.params, vae_params_update)

        metrics = {'loss': loss,
                   'kl_loss': kl_loss,
                   'a_decoder_loss': a_decoder_loss,
                   'y_decoder_loss': y_decoder_loss,
                   'weight': w,
                   'disagreement_loss': disagreement_loss}

        new_state = TrainingState(
            optimizer_state=vae_optimizer_state,
            params=vae_params,
            key=key,
            steps=state.steps + 1)
        return new_state, metrics

    def sample_data_vae(training_state, replay_buffer, max_epi_len):
            # num_updates_per_iter
            key1, key2, key3 = jax.random.split(training_state.key, 3)
            epi_idx = jax.random.randint(
                key1, (int(vae_batch_size_per_device*grad_updates_per_step),),
                minval=0,
                maxval=replay_buffer.data.shape[0])  # from (0, num_epi)
            context_idx = jax.random.randint(
                key2, (int(vae_batch_size_per_device*grad_updates_per_step),),
                minval=0,
                maxval=max_epi_len)  # from (0, max_epi_len)

            def dynamic_slice_context(carry, x):
                traj, c_idx = x
                return (), jax.lax.dynamic_slice(traj, (c_idx, 0), (context_len, trans_dim))

            # (batch_size_per_device*num_updates_per_iter, max_epi_len + context_len, trans_dim)
            transitions = jnp.take(replay_buffer.data, epi_idx, axis=0, mode='clip')
            # (batch_size_per_device*num_updates_per_iter, context_len, trans_dim)
            _, transitions = jax.lax.scan(dynamic_slice_context, (), (transitions, context_idx))
            # (num_updates_per_iter, batch_size_per_device, context_len, trans_dim)
            transitions = jnp.reshape(transitions,
                                    [grad_updates_per_step, -1] + list(transitions.shape[1:]))

            training_state = training_state.replace(key=key3)
            return training_state, transitions

    def run_one_epoch_vae(carry, unused_t, max_epi_len):
        training_state, replay_buffer = carry

        training_state, transitions = sample_data_vae(training_state, replay_buffer, max_epi_len)
        training_state, metrics = jax.lax.scan(
            update_step_vae, training_state, transitions, length=1)
        return (training_state, replay_buffer), metrics

    def run_training_vae(training_state, replay_buffer, max_epi_len):
        synchro = is_replicated(
            training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')
        (training_state, replay_buffer), metrics = jax.lax.scan(
            partial(run_one_epoch_vae, max_epi_len=max_epi_len), (training_state, replay_buffer), (),
            length=num_updates_per_iter)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, replay_buffer, metrics, synchro
    
    run_training_vae = jax.pmap(partial(run_training_vae, max_epi_len=max_epi_len), axis_name='i')

    vae_training_state = TrainingState(
        optimizer_state=vae_optimizer_state,
        params=vae_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        steps=jnp.zeros((local_devices_to_use,)))

    total_updates = 0

    save_model_path = os.path.join(log_dir, "vae_model.pt")

    for i_train_iter in range(max_train_iters):
        log_vae_losses = []
        log_kl_losses = []
        log_a_decoder_losses = []
        log_y_decoder_losses = []
        disagreement_losses = []
        weights = []

        # optimization
        vae_training_state, replay_buffer, training_metrics, synchro = run_training_vae(
            vae_training_state, replay_buffer)
        assert synchro[0], (current_step, vae_training_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        log_vae_losses.append(training_metrics['loss'])
        log_kl_losses.append(training_metrics['kl_loss'])
        log_a_decoder_losses.append(training_metrics['a_decoder_loss'])
        log_y_decoder_losses.append(training_metrics['y_decoder_loss'])
        weights.append(training_metrics['weight'])
        disagreement_losses.append(training_metrics['disagreement_loss'])

        mean_vae_loss = np.mean(log_vae_losses)
        mean_kl_loss = np.mean(log_kl_losses)
        mean_a_decoder_loss = np.mean(log_a_decoder_losses)
        mean_y_decoder_loss = np.mean(log_y_decoder_losses)
        mean_weights = np.mean(weights)
        mean_disagreement_losses = np.mean(disagreement_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed  + '\n' +
                   "train iter: " + str(i_train_iter)  + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "vae loss: " +  format(mean_vae_loss, ".5f") + '\n' +
                   "kl loss: " +  format(mean_kl_loss, ".5f") + '\n' +
                   "a decoder loss: " +  format(mean_a_decoder_loss, ".5f") + '\n' + 
                   "y decoder loss: " +  format(mean_y_decoder_loss, ".5f") + '\n' + 
                   "mean weights: " +  format(mean_weights, ".5f") + '\n' + 
                   "mean disagreement losses: " +  format(mean_disagreement_losses, ".5f") + '\n'
                )

        print(log_str)

        wandb.log({'mean_vae_loss': mean_vae_loss,
                   'mean_kl_loss': mean_kl_loss,
                   'mean_a_decoder_loss': mean_a_decoder_loss,
                   'mean_y_decoder_loss': mean_y_decoder_loss,
                   'mean_weight_(y_decoder)': mean_weights,
                   'mean disagreement losses': mean_disagreement_losses})

        log_data = [
            time_elapsed,
            total_updates,
            mean_vae_loss,
            mean_kl_loss,
            mean_a_decoder_loss,
            mean_y_decoder_loss
        ]

        csv_writer.writerow(log_data)

        # save model
        _vae_params = jax.tree_util.tree_map(lambda x: x[0], vae_training_state.params)

        if i_train_iter % args.vae_save_iters == 0 or i_train_iter == max_train_iters - 1:
            save_current_model_path = save_model_path[:-3] + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _vae_params)
            eval_model('vae', _vae_params, total_updates) # for model in ['vae', 'emp']:
            eval_dynamics(vae_training_state.key, {'params': _vae_params['params']['precoder']}, dynamics_model.apply, _dynamics_params, total_updates)

    synchronize_hosts()
    
    print("=" * 60)
    print("finished training vae!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training vae at: " + start_time_str)
    print("finished training vae at: " + end_time_str)
    print("total vae training time: " + time_elapsed)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)

    ###################################### empowerment training ###################################### 

    emp_model = empowerment(
        state_dim=obs_dim,
        act_dim=act_dim,
        controlled_variables_dim=controlled_variables_dim,
        controlled_variables=controlled_variables,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        n_dynamics_ensembles=args.n_dynamics_ensembles,
    )

    schedule_fn = optax.polynomial_schedule(
        init_value=lr * 1 / warmup_steps,
        end_value=lr,
        power=1,
        transition_steps=warmup_steps,
        transition_begin=0
    )
    emp_optimizer = optax.chain(
        optax.clip(args.gradient_clipping),
        optax.adamw(learning_rate=schedule_fn, weight_decay=wt_decay),
    )

    batch_size = 1
    dummy_timesteps = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
    dummy_states = jnp.zeros((batch_size, context_len, obs_dim))
    dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
    if args.trajectory_version:
        dummy_latent = jnp.zeros((batch_size, context_len * controlled_variables_dim))
        dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
    else:
        dummy_latent = jnp.zeros((batch_size, controlled_variables_dim))
        dummy_controlled_variables = jnp.zeros((batch_size, 1, controlled_variables_dim))
    dummy_rtg = jnp.zeros((batch_size, context_len, 1))
    dummy_horizon= jnp.zeros((batch_size, 1), dtype=jnp.int32)
    dummy_mask = jnp.zeros((batch_size, context_len, 1))

    key_params, key_dropout = jax.random.split(global_key_emp)
    emp_params = emp_model.init({'params': key_params, 'dropout': key_dropout},
                                ts=dummy_timesteps,
                                s_t=dummy_states,
                                z_t=dummy_latent,
                                a_t=dummy_actions,
                                y_t=dummy_controlled_variables,
                                rtg_t=dummy_rtg,
                                horizon=dummy_horizon,
                                mask=dummy_mask,
                                train_precoder=True,
                                dynamics_apply=dynamics_model.apply,
                                dynamics_params=_dynamics_params,
                                key=key_params)

    load_model_path = os.path.join(log_dir, "vae_model.pt")
    load_current_model_path = load_model_path[:-3] + f"_{max_train_iters*args.vae_save_iters}.pt"
    _vae_params = load_params(load_current_model_path)
    emp_params['params']['prior'] = _vae_params['params']['prior']
    emp_params['params']['precoder'] = _vae_params['params']['precoder']
    emp_params['params']['encoder'] = _vae_params['params']['encoder']
    del _vae_params
    
    emp_optimizer_state = emp_optimizer.init(emp_params)

    emp_optimizer_state, emp_params = bcast_local_devices(
        (emp_optimizer_state, emp_params), local_devices_to_use)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(emp_params))
    print(f'num_emp_param: {param_count}')

    def emp_loss(emp_params: Any,
                   transitions: Transition, key: jnp.ndarray, train_precoder: bool) -> jnp.ndarray:
        ts = transitions.ts.reshape(transitions.ts.shape[:2]).astype(jnp.int32)  # (batch_size_per_device, context_len)
        s_t = transitions.s_t  # (batch_size_per_device, context_len, state_dim)
        a_t = transitions.a_t  # (batch_size_per_device, context_len, action_dim)
        rtg_t = transitions.rtg_t  # (batch_size_per_device, context_len, 1)
        mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
        s_tp1 = transitions.s_tp1  # (batch_size_per_device, context_len, state_dim)

        horizon = mask.sum(axis=1).astype(jnp.int32) # (B, 1)
        y_t = transitions.s_tp1[...,controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)
        if args.trajectory_version:
            # y_t = transitions.s_tp1[...,controlled_variables]  # (batch_size_per_device, context_len, controlled_variables_dim)
            dummy_z_t = jnp.zeros((emp_batch_size_per_device, context_len * controlled_variables_dim))
        else:
            # y_t = jnp.take_along_axis(s_tp1, horizon[..., None]-1, axis=1)[...,controlled_variables]
            dummy_z_t = jnp.zeros((emp_batch_size_per_device, controlled_variables_dim))

        emp_key, dropout_key = jax.random.split(key, 2)

        loss = emp_model.apply(emp_params, ts, s_t, dummy_z_t, a_t, y_t, rtg_t, horizon, mask, train_precoder, dynamics_model.apply, _dynamics_params, emp_key, rngs={'dropout': dropout_key})

        return loss

    # emp_grad = jax.jit(jax.value_and_grad(emp_loss), static_argnames=('train_precoder'))
    emp_grad = jax.jit(jax.value_and_grad(emp_loss))

    @jax.jit
    # @partial(jax.jit, static_argnames=["train_precoder"])
    def update_step_emp(
        # state: TrainingState,
        carry,
        transitions: jnp.ndarray,
    ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

        state, train_precoder = carry

        cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1])

        transitions = Transition(
            s_t=transitions[:, :, :cumsum_dims[0]],
            a_t=transitions[:, :, cumsum_dims[0]:cumsum_dims[1]],
            s_tp1=transitions[:, :, cumsum_dims[1]:cumsum_dims[2]],
            rtg_t=transitions[:, :, cumsum_dims[2]:cumsum_dims[3]],
            ts=transitions[:, :, cumsum_dims[3]:cumsum_dims[4]],
            mask_t=transitions[:, :, cumsum_dims[4]:cumsum_dims[5]]
        )

        key, key_emp = jax.random.split(state.key, 2)

        loss, emp_grads = emp_grad(state.params, transitions, key_emp, train_precoder)
        emp_grads = jax.lax.pmean(emp_grads, axis_name='i')
        emp_params_update, emp_optimizer_state = emp_optimizer.update(
            emp_grads, state.optimizer_state, state.params)
        emp_params = optax.apply_updates(state.params, emp_params_update)

        metrics = {'loss': loss}

        new_state = TrainingState(
            optimizer_state=emp_optimizer_state,
            params=emp_params,
            key=key,
            steps=state.steps + 1)
        
        carry = new_state, train_precoder

        return carry, metrics
    
    def sample_data_emp(training_state, replay_buffer, max_epi_len):
        # num_updates_per_iter
        key1, key2, key3 = jax.random.split(training_state.key, 3)
        epi_idx = jax.random.randint(
            key1, (int(emp_batch_size_per_device*grad_updates_per_step),),
            minval=0,
            maxval=replay_buffer.data.shape[0])  # from (0, num_epi)
        context_idx = jax.random.randint(
            key2, (int(emp_batch_size_per_device*grad_updates_per_step),),
            minval=0,
            maxval=max_epi_len)  # from (0, max_epi_len)

        def dynamic_slice_context(carry, x):
            traj, c_idx = x
            return (), jax.lax.dynamic_slice(traj, (c_idx, 0), (context_len, trans_dim))

        # (batch_size_per_device*num_updates_per_iter, max_epi_len + context_len, trans_dim)
        transitions = jnp.take(replay_buffer.data, epi_idx, axis=0, mode='clip')
        # (batch_size_per_device*num_updates_per_iter, context_len, trans_dim)
        _, transitions = jax.lax.scan(dynamic_slice_context, (), (transitions, context_idx))
        # (num_updates_per_iter, batch_size_per_device, context_len, trans_dim)
        transitions = jnp.reshape(transitions,
                                [grad_updates_per_step, -1] + list(transitions.shape[1:]))

        training_state = training_state.replace(key=key3)
        return training_state, transitions

    def run_one_epoch_emp(carry, unused_t, max_epi_len):
        training_state, replay_buffer, train_precoder = carry

        training_state, transitions = sample_data_emp(training_state, replay_buffer, max_epi_len)
        (training_state, train_precoder), metrics = jax.lax.scan(
            update_step_emp, (training_state, train_precoder), transitions, length=1)
        return (training_state, replay_buffer, train_precoder), metrics

    def run_training_emp(training_state, replay_buffer, train_precoder, max_epi_len):
        synchro = is_replicated(
            training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')
        (training_state, replay_buffer, train_precoder), metrics = jax.lax.scan(
            partial(run_one_epoch_emp, max_epi_len=max_epi_len), (training_state, replay_buffer, train_precoder), (),
            length=num_updates_per_iter)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, replay_buffer, metrics, synchro
    
    train_precoder = True
    run_training_emp = jax.pmap(partial(run_training_emp, max_epi_len=max_epi_len, train_precoder=train_precoder), axis_name='i')
    # run_training_emp = partial(run_training_emp, max_epi_len=max_epi_len)

    emp_training_state = TrainingState(
        optimizer_state=emp_optimizer_state,
        params=emp_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        # key=local_key,
        steps=jnp.zeros((local_devices_to_use,)))
        # steps=jnp.zeros(1))

    total_updates = 0

    save_model_path = os.path.join(log_dir, "emp_model.pt")

    for i_train_iter in range(max_train_iters):
        log_emp_losses = []

        # optimization
        emp_training_state, replay_buffer, training_metrics, synchro = run_training_emp(
            emp_training_state, replay_buffer, train_precoder=jnp.array([True],))
        assert synchro[0], (current_step, emp_training_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        # precoder_loss = training_metrics['loss']

        # emp_training_state, replay_buffer, training_metrics, synchro = run_training_emp(
        #     emp_training_state, replay_buffer, train_precoder=jnp.array([False],))
        # assert synchro[0], (current_step, emp_training_state)
        # jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        # posterior_loss = training_metrics['loss']
        # log_emp_losses.append((precoder_loss + posterior_loss)/2)

        log_emp_losses.append(training_metrics['loss'])

        mean_emp_loss = np.mean(log_emp_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed  + '\n' +
                   "train iter: " + str(i_train_iter)  + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "emp loss: " +  format(mean_emp_loss, ".5f") + '\n'
                )

        print(log_str)

        wandb.log({'mean_emp_loss': mean_emp_loss})

        log_data = [
            time_elapsed,
            total_updates,
            mean_emp_loss,
        ]

        csv_writer.writerow(log_data)

        # save model
        _emp_params = jax.tree_util.tree_map(lambda x: x[0], emp_training_state.params)

        if i_train_iter % args.emp_save_iters == 0 or i_train_iter == max_train_iters - 1:
            save_current_model_path = save_model_path[:-3] + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _emp_params)
            eval_model('emp', _emp_params, total_updates)

    synchronize_hosts()
    
    print("=" * 60)
    print("finished training emp!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training emp at: " + start_time_str)
    print("finished training emp at: " + end_time_str)
    print("total emp training time: " + time_elapsed)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rtg_scale', type=int, default=1000)
    parser.add_argument('--rtg_target', type=int, default=None)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--context_len', type=int, default=50)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--gradient_clipping', type=float, default=0.25)
    
    parser.add_argument('--n_dynamics_ensembles', type=int, default=4)
    parser.add_argument('--h_dims_dynamics', type=List, default=[256,256])
    parser.add_argument('--dynamics_dropout_rates', type=List, default=[0.1, 0.1])
    parser.add_argument('--dynamics_dropout_p', type=float, default=0.)

    parser.add_argument('--dynamics_batch_size', type=int, default=2048)
    parser.add_argument('--vae_batch_size', type=int, default=256)
    parser.add_argument('--emp_batch_size', type=int, default=256)
    parser.add_argument('--grad_updates_per_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10_000)

    parser.add_argument('--max_train_iters', type=int, default=3000)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)
    parser.add_argument('--dynamics_save_iters', type=int, default=500)
    parser.add_argument('--vae_save_iters', type=int, default=100)
    parser.add_argument('--emp_save_iters', type=int, default=100)
    parser.add_argument('--rm_normalization', action='store_true', help='Turn off input normalization')

    parser.add_argument('--max_devices_per_host', type=int, default=None)

    parser.add_argument('--trajectory_version', action='store_true')
    parser.add_argument('--uncertainty_weight', type=float, default=0.)
    parser.add_argument('--y_decoder_weight', type=float, default=None)
    
    parser.add_argument('--resume_dynamics', action='store_true')
    parser.add_argument('--resume_vae', action='store_true')
    parser.add_argument('--resume_start_time_str', type=str, default='25-07-07-16-32-20') # None, '25-06-25-16-17-25'

    args = parser.parse_args()

    train(args)
