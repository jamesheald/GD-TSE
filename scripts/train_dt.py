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

from decision_transformer.dt.model import make_transformer_networks, VAE, empowerment
from decision_transformer.dt.utils import ReplayBuffer, TrainingState, Transition
from decision_transformer.dt.utils import discount_cumsum, save_params, load_params
from decision_transformer.pmap import bcast_local_devices, synchronize_hosts, is_replicated

import jax
print(jax.devices())
from jax import config
config.update('jax_disable_jit', False)
config.update('jax_debug_nans', True)
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
    global_key, global_key_vae, global_key_emp, local_key, test_key = jax.random.split(key, 5)
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

    prefix = "dt_" + env_d4rl_name
    start_time_str = '25-06-19-17-34-48'
    log_dir = os.path.join(args.log_dir, prefix, f'seed_{seed}', start_time_str)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    # prefix = "dt_" + env_d4rl_name
    # log_dir = os.path.join(log_dir, prefix, f'seed_{seed}', start_time_str)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

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

    ###################################### dynamics ###################################### 

    import mujoco
    from mujoco import mjx

    import minari
    dataset = minari.load_dataset('D4RL/relocate/expert-v2')
    env = dataset.recover_environment(render_mode='rgb_array')
    mj_model = env.unwrapped.model
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # differentiation compatabile solver settings - https://github.com/google-deepmind/mujoco/issues/1182
    mjx_model = mjx_model.replace(opt=mjx_model.opt.replace(solver=mujoco.mjtSolver.mjSOL_NEWTON))
    mjx_model = mjx_model.replace(opt=mjx_model.opt.replace(iterations=1))
    mjx_model = mjx_model.replace(opt=mjx_model.opt.replace(ls_iterations=1))

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

    wandb.init(
            name=f'{env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
            group=env_d4rl_name,
            project='jax_dt',
            config=args
        )

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

        decoder_loss, kl_loss = vae_model.apply(vae_params, ts, s_t, dummy_z_t, a_t, y_t, rtg_t, horizon, mask, vae_key, rngs={'dropout': dropout_key})

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

    vae_training_state = TrainingState(
        optimizer_state=vae_optimizer_state,
        params=vae_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        # key=local_key,
        steps=jnp.zeros((local_devices_to_use,)))
        # steps=jnp.zeros(1))

    total_updates = 0

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

    ###################################### empowerment training ###################################### 

    emp_model = empowerment(
        state_dim=state_dim,
        act_dim=act_dim,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        model=mjx_model
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
    dummy_mjx_data = jax.vmap(lambda init_qpos: mjx_data.replace(qpos=init_qpos))(jnp.zeros((batch_size, mjx_data.qpos.shape[0])))

    # from decision_transformer.dt.model import MJXRollout
    # dynamics = MJXRollout(model=mjx_model)
    # y_samp = dynamics(dummy_actions, dummy_mjx_data)
    # get_jacobian = jax.jit(jax.jacrev(dynamics))
    # y_samp_grad = get_jacobian(dummy_actions, dummy_mjx_data)
    # breakpoint()

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
                                mjx_data=dummy_mjx_data,
                                key=key_params)

    total_updates = 1000
    load_model_path = os.path.join(log_dir, "vae_model.pt")
    load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
    _vae_params = load_params(load_current_model_path)
    # emp_params['params']['precoder'] = _vae_params['params']['precoder']
    # emp_params['params']['encoder'] = _vae_params['params']['encoder']
    emp_params['params']['precoder'] = _vae_params['params']['decoder']
    del _vae_params
    
    emp_optimizer_state = emp_optimizer.init(emp_params)

    emp_optimizer_state, emp_params = bcast_local_devices(
        (emp_optimizer_state, emp_params), local_devices_to_use)

    # count the number of parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(emp_params))
    print(f'num_emp_param: {param_count}')

    def emp_loss(emp_params: Any,
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

        emp_key, dropout_key = jax.random.split(key, 2)

        s_t_unnorm = s_t * state_std + state_mean
        qpos, qvel = jnp.split(s_t_unnorm[:,0,:], 2, axis=-1)

        batch_mjx_data = batch_mjx_data.replace(qpos=qpos)
        batch_mjx_data = batch_mjx_data.replace(qvel=qvel)

        loss = emp_model.apply(emp_params, ts, s_t, dummy_z_t, a_t, y_t, rtg_t, horizon, mask, batch_mjx_data, emp_key, rngs={'dropout': dropout_key})

        return loss

    emp_grad = jax.jit(jax.value_and_grad(emp_loss))

    @jax.jit
    def update_step_emp(
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

        key, key_emp = jax.random.split(state.key, 2)

        loss, emp_grads = emp_grad(state.params, transitions, key_emp)
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
        return new_state, metrics

    def run_one_epoch_emp(carry, unused_t, max_epi_len):
        training_state, replay_buffer = carry

        training_state, transitions = sample_data(training_state, replay_buffer, max_epi_len)
        training_state, metrics = jax.lax.scan(
            update_step_emp, training_state, transitions, length=1)
        return (training_state, replay_buffer), metrics

    def run_training_emp(training_state, replay_buffer, max_epi_len):
        synchro = is_replicated(
            training_state.replace(key=jax.random.PRNGKey(0)), axis_name='i')
        (training_state, replay_buffer), metrics = jax.lax.scan(
            partial(run_one_epoch_emp, max_epi_len=max_epi_len), (training_state, replay_buffer), (),
            length=num_updates_per_iter)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, replay_buffer, metrics, synchro
    
    run_training_emp = jax.pmap(partial(run_training_emp, max_epi_len=max_epi_len), axis_name='i')
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

    batch_mjx_data = jax.vmap(lambda init_qpos: mjx_data.replace(qpos=init_qpos))(jnp.zeros((batch_size_per_device, mjx_data.qpos.shape[0])))

    for i_train_iter in range(max_train_iters):
        log_emp_losses = []

        # optimization
        emp_training_state, replay_buffer, training_metrics, synchro = run_training_emp(
            emp_training_state, replay_buffer)
        assert synchro[0], (current_step, emp_training_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
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

    parser.add_argument('--max_train_iters', type=int, default=10)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)
    parser.add_argument('--dynamics_save_iters', type=int, default=100)
    parser.add_argument('--vae_save_iters', type=int, default=100)
    parser.add_argument('--emp_save_iters', type=int, default=100)
    parser.add_argument('--rm_normalization', action='store_true', help='Turn off input normalization')

    parser.add_argument('--max_devices_per_host', type=int, default=None)

    parser.add_argument('--trajectory_version', type=bool, default=False)

    args = parser.parse_args()

    train(args)
