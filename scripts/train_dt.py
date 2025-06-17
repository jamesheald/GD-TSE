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

from decision_transformer.dt.model import make_transformer_networks, VAE
from decision_transformer.dt.utils import ReplayBuffer, TrainingState, Transition
from decision_transformer.dt.utils import discount_cumsum, save_params
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
    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = "dt_" + env_d4rl_name

    log_dir = os.path.join(log_dir, prefix, f'seed_{seed}', start_time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

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

    ###################################### dynamics training ###################################### 

    dynamics_model = make_transformer_networks(
        state_dim=state_dim,
        act_dim=act_dim,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        trajectory_version=args.trajectory_version,
        transformer_type='dynamics'
    )

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
    key_params, key_dropout = jax.random.split(global_key)
    dynamics_params = dynamics_model.init({'params': key_params, 'dropout': key_dropout})
    dynamics_optimizer_state = dynamics_optimizer.init(dynamics_params)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(dynamics_params))
    print(f'num_dynamics_param: {param_count}')

    dynamics_optimizer_state, dynamics_params = bcast_local_devices(
        (dynamics_optimizer_state, dynamics_params), local_devices_to_use)

    def dynamics_loss(dynamics_params: Any,
                   transitions: Transition, key: jnp.ndarray) -> jnp.ndarray:
        ts = transitions.ts.reshape(transitions.ts.shape[:2]).astype(jnp.int32)  # (batch_size_per_device, context_len)
        s_t = transitions.s_t  # (batch_size_per_device, context_len, state_dim)
        a_t = transitions.a_t  # (batch_size_per_device, context_len, action_dim)
        rtg_t = transitions.rtg_t  # (batch_size_per_device, context_len, 1)
        mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
        
        horizon = mask.sum(axis=1).astype(jnp.int32) # (B, 1)
        if args.trajectory_version:
            y_t = transitions.y_t  # (batch_size_per_device, context_len, controlled_variables_dim)
            dummy_z_t = jnp.zeros((batch_size, context_len * controlled_variables_dim))
        else:
            def slice_fn(y_t_b, start_t):
                return jax.lax.dynamic_slice(y_t_b, (start_t, 0), (1, controlled_variables_dim))  # (1, D)
            start_ts = (horizon - 1).squeeze(-1)  # shape: (B,)
            y_t = jax.vmap(slice_fn)(transitions.y_t, start_ts)  # (B, 1, D) y_t = transitions.y_t[:,horizon-1:horizon,:]
            dummy_z_t = jnp.zeros((batch_size, controlled_variables_dim))

        y_p = dynamics_model.apply(dynamics_params, ts, s_t, dummy_z_t, a_t, y_t, rtg_t, horizon, rngs={'dropout': key})

        def true_fn(y_mean, y_log_std, y_t):
            dist = tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=jnp.exp(y_log_std))
            return dist.log_prob(y_t)

        def false_fn(y_mean, y_log_std, y_t):
            return 0.
        
        def get_log_prob(mask, y_mean, y_log_std, y_t):
            log_prob = jax.lax.cond(mask, true_fn, false_fn, y_mean, y_log_std, y_t)
            return log_prob
        batch_get_log_prob = jax.vmap(get_log_prob)

        y_mean, y_log_std = jnp.split(y_p, 2, axis=-1)
        min_log_std = -20.
        max_log_std = 2.
        y_log_std = jnp.clip(y_log_std, min_log_std, max_log_std)
        y_mean = y_mean.reshape(-1, controlled_variables_dim)
        y_log_std = y_log_std.reshape(-1, controlled_variables_dim)
        y_t = y_t.reshape(-1, controlled_variables_dim)
        
        if args.trajectory_version:
            valid_mask = (mask.reshape(-1, 1) > 0).squeeze(-1)
            log_probs = batch_get_log_prob(valid_mask, y_mean, y_log_std, y_t)
            loss = jnp.sum(-log_probs * valid_mask) / jnp.sum(valid_mask)
        else:
            log_probs = jax.vmap(true_fn)(y_mean, y_log_std, y_t)
            loss = jnp.mean(-log_probs)

        return loss

    dynamics_grad = jax.jit(jax.value_and_grad(dynamics_loss))

    @jax.jit
    def update_step(
        state: TrainingState,
        transitions: jnp.ndarray,
    ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

        cumsum_dims = np.cumsum([state_dim, act_dim, controlled_variables_dim, 1, 1, 1])

        transitions = Transition(
            s_t=transitions[:, :, :cumsum_dims[0]],
            a_t=transitions[:, :, cumsum_dims[0]:cumsum_dims[1]],
            y_t=transitions[:, :, cumsum_dims[1]:cumsum_dims[2]],
            rtg_t=transitions[:, :, cumsum_dims[2]:cumsum_dims[3]],
            ts=transitions[:, :, cumsum_dims[3]:cumsum_dims[4]],
            mask_t=transitions[:, :, cumsum_dims[4]:cumsum_dims[5]]
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
            key1, (int(batch_size_per_device*grad_updates_per_step),),
            minval=0,
            maxval=replay_buffer.data.shape[0])  # from (0, num_epi)
        context_idx = jax.random.randint(
            key2, (int(batch_size_per_device*grad_updates_per_step),),
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

    def run_one_epoch(carry, unused_t, max_epi_len):
        training_state, replay_buffer = carry

        training_state, transitions = sample_data(training_state, replay_buffer, max_epi_len)
        training_state, metrics = jax.lax.scan(
            update_step, training_state, transitions, length=1)
        return (training_state, replay_buffer), metrics

    def run_training(training_state, replay_buffer, max_epi_len):
        synchro = is_replicated(
            training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')
        (training_state, replay_buffer), metrics = jax.lax.scan(
            partial(run_one_epoch, max_epi_len=max_epi_len), (training_state, replay_buffer), (),
            length=num_updates_per_iter)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, replay_buffer, metrics, synchro
    
    run_training = jax.pmap(partial(run_training, max_epi_len=max_epi_len), axis_name='i')

    training_state = TrainingState(
        optimizer_state=dynamics_optimizer_state,
        params=dynamics_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        steps=jnp.zeros((local_devices_to_use,)))

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

    ###################################### vae training ###################################### 

    vae_model = VAE(
        state_dim=state_dim,
        act_dim=act_dim,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p
    )

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
    dummy_states = jnp.zeros((batch_size, context_len, state_dim))
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
    
    vae_optimizer_state = vae_optimizer.init(vae_params)

    vae_optimizer_state, vae_params = bcast_local_devices(
        (vae_optimizer_state, vae_params), local_devices_to_use)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(vae_params))
    print(f'num_vae_param: {param_count}')

    def vae_loss(vae_params: Any,
                   transitions: Transition, key: jnp.ndarray) -> jnp.ndarray:
        ts = transitions.ts.reshape(transitions.ts.shape[:2]).astype(jnp.int32)  # (batch_size_per_device, context_len)
        s_t = transitions.s_t  # (batch_size_per_device, context_len, state_dim)
        a_t = transitions.a_t  # (batch_size_per_device, context_len, action_dim)
        rtg_t = transitions.rtg_t  # (batch_size_per_device, context_len, 1)
        mask = transitions.mask_t  # (batch_size_per_device, context_len, 1)
        
        horizon = mask.sum(axis=1).astype(jnp.int32) # (B, 1)
        if args.trajectory_version:
            y_t = transitions.y_t  # (batch_size_per_device, context_len, controlled_variables_dim)
            dummy_z_t = jnp.zeros((batch_size, context_len * controlled_variables_dim))
        else:
            def slice_fn(y_t_b, start_t):
                return jax.lax.dynamic_slice(y_t_b, (start_t, 0), (1, controlled_variables_dim))  # (1, D)
            start_ts = (horizon - 1).squeeze(-1)  # shape: (B,)
            y_t = jax.vmap(slice_fn)(transitions.y_t, start_ts)  # (B, 1, D) y_t = transitions.y_t[:,horizon-1:horizon,:]
            dummy_z_t = jnp.zeros((batch_size, controlled_variables_dim))

        vae_key, dropout_key = jax.random.split(key, 2)

        decoder_loss, kl_loss = vae_model.apply(vae_params, ts, s_t, dummy_z_t, a_t, y_t, rtg_t, horizon, mask, dynamics_model.apply, _dynamics_params, vae_key, rngs={'dropout': dropout_key})

        return decoder_loss + kl_loss, (decoder_loss, kl_loss)

    vae_grad = jax.jit(jax.value_and_grad(vae_loss, has_aux=True))

    @jax.jit
    def update_step_vae(
        state: TrainingState,
        transitions: jnp.ndarray,
    ) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:

        cumsum_dims = np.cumsum([state_dim, act_dim, controlled_variables_dim, 1, 1, 1])

        transitions = Transition(
            s_t=transitions[:, :, :cumsum_dims[0]],
            a_t=transitions[:, :, cumsum_dims[0]:cumsum_dims[1]],
            y_t=transitions[:, :, cumsum_dims[1]:cumsum_dims[2]],
            rtg_t=transitions[:, :, cumsum_dims[2]:cumsum_dims[3]],
            ts=transitions[:, :, cumsum_dims[3]:cumsum_dims[4]],
            mask_t=transitions[:, :, cumsum_dims[4]:cumsum_dims[5]]
        )

        key, key_vae = jax.random.split(state.key, 2)

        (loss, (decoder_loss, kl_loss)), vae_grads = vae_grad(state.params, transitions, key_vae)
        vae_grads = jax.lax.pmean(vae_grads, axis_name='i')
        vae_params_update, vae_optimizer_state = vae_optimizer.update(
            vae_grads, state.optimizer_state, state.params)
        vae_params = optax.apply_updates(state.params, vae_params_update)

        metrics = {'loss': loss,
                   'decoder_loss': decoder_loss,
                   'kl_loss': kl_loss}

        new_state = TrainingState(
            optimizer_state=vae_optimizer_state,
            params=vae_params,
            key=key,
            steps=state.steps + 1)
        return new_state, metrics

    def run_one_epoch_vae(carry, unused_t, max_epi_len):
        training_state, replay_buffer = carry

        training_state, transitions = sample_data(training_state, replay_buffer, max_epi_len)
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
    # run_training_vae = partial(run_training_vae, max_epi_len=max_epi_len)

    vae_training_state = TrainingState(
        optimizer_state=vae_optimizer_state,
        params=vae_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        # key=local_key,
        steps=jnp.zeros((local_devices_to_use,)))
        # steps=jnp.zeros(1))

    total_updates = 0

    # wandb.init(
    #         name=f'{env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
    #         group=env_d4rl_name,
    #         project='jax_dt',
    #         config=args
    #     )

    save_model_path = os.path.join(log_dir, "vae_model.pt")

    for i_train_iter in range(max_train_iters):
        log_vae_losses = []
        log_decoder_losses = []
        log_kl_losses = []

        # optimization
        vae_training_state, replay_buffer, training_metrics, synchro = run_training_vae(
            vae_training_state, replay_buffer)
        assert synchro[0], (current_step, vae_training_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        log_vae_losses.append(training_metrics['loss'])
        log_decoder_losses.append(training_metrics['decoder_loss'])
        log_kl_losses.append(training_metrics['kl_loss'])

        mean_vae_loss = np.mean(log_vae_losses)
        mean_decoder_loss = np.mean(log_decoder_losses)
        mean_kl_loss = np.mean(log_kl_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed  + '\n' +
                   "train iter: " + str(i_train_iter)  + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "vae loss: " +  format(mean_vae_loss, ".5f") + '\n' +
                   "decoder loss: " +  format(mean_decoder_loss, ".5f") + '\n' + 
                   "kl loss: " +  format(mean_kl_loss, ".5f") + '\n'
                )

        print(log_str)

        wandb.log({'mean_vae_loss': mean_vae_loss,
                   'mean_decoder_loss': mean_decoder_loss,
                   'mean_kl_loss': mean_kl_loss})

        log_data = [
            time_elapsed,
            total_updates,
            mean_vae_loss,
            mean_decoder_loss,
            mean_kl_loss
        ]

        csv_writer.writerow(log_data)

        # save model
        _vae_params = jax.tree_util.tree_map(lambda x: x[0], vae_training_state.params)

        if i_train_iter % args.vae_save_iters == 0 or i_train_iter == max_train_iters - 1:
            save_current_model_path = save_model_path[:-3] + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _vae_params)

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

    # xx=VAE_model.apply(VAE_params,
    #                 ts=dummy_timesteps,
    #                 s_t=dummy_states,
    #                 z_t=dummy_latent,
    #                 a_t=dummy_actions,
    #                 y_t=dummy_controlled_variables,
    #                 rtg_t=dummy_rtg,
    #                 horizon=dummy_horizon,
    #                 mask=dummy_mask,
    #                 dynamics_apply=dynamics_model.apply,
    #                 dynamics_params=dynamics_params_prebroadcast,
    #                 key=key_params,
    #                 rngs={'dropout': key_dropout})
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
    parser.add_argument('--dropout_p', type=float, default=0.1)
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
