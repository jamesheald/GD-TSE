import os
os.environ["MUJOCO_GL"] = "egl"

import jax
import optax
import random

import io

import jax.numpy as jnp
import numpy as np

from datetime import datetime
from typing import Any, Dict, Tuple

from functools import partial

import wandb
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import minari
import imageio
from matplotlib import pyplot as plt

from dataclasses import replace

print(jax.devices())
# from jax import config
# config.update('jax_disable_jit', False)
# config.update('jax_debug_nans', True)
# config.update('jax_enable_x64', False)

from decision_transformer.dt.networks.networks import dynamics, GRU_Precoder, MLP
from decision_transformer.dt.models.models import CLVM, empowerment
from decision_transformer.dt.utils import ReplayBuffer, TrainingState, Transition
from decision_transformer.dt.utils import discount_cumsum, save_params, load_params
from decision_transformer.pmap import bcast_local_devices, synchronize_hosts, is_replicated
from scripts.train_loop import create_one_train_iteration
from scripts.losses.losses import dynamics_loss, CLVM_loss, precoder_loss

import hydra
from omegaconf import DictConfig

cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'conf')
@hydra.main(config_path=cfg_path, config_name="config.yaml")
def train(args: DictConfig):
# def train(args):

    timestamp = datetime.now().strftime("%Y-%m-%d")
    BASE_SAVE_PATH = '/nfs/nhome/live/jheald/jax_dt/model_outputs'

    # import mujoco
    # model = minari_env.unwrapped.model
    # for j in range(model.njnt): print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))

    # controlled_variables_dim = 6
    # controlled_variables = [i for i in range(3)] # hand pos
    # controlled_variables += [36 + i for i in range(3)] # object pos
    controlled_variables_dim = 3
    controlled_variables = [36 + i for i in range(3)] # object pos

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
    # dataset_pathdataset_path = f'{args.dataset_dir}/{env_d4rl_name}-fullnextstate.pkl'

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

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
    trans_dim = obs_dim + act_dim + obs_dim + 1 + 1 + 1 + obs_dim + obs_dim # rtg, timesteps, mask

    # used for input normalization
    delta_obs_stats = [ob[1:]-ob[:-1]for ob in obs_stats]
    delta_obs_stats = jnp.concatenate(delta_obs_stats, axis=0)
    delta_obs_mean, delta_obs_std = jnp.mean(delta_obs_stats, axis=0), jnp.std(delta_obs_stats, axis=0) + 1e-8
    delta_obs_min, delta_obs_max = jnp.min(delta_obs_stats, axis=0), jnp.max(delta_obs_stats, axis=0)
    delta_obs_min = (delta_obs_min - delta_obs_mean) / delta_obs_std
    delta_obs_max = (delta_obs_max - delta_obs_mean) / delta_obs_std
    obs_stats = jnp.concatenate(obs_stats, axis=0)
    obs_mean, obs_std = jnp.mean(obs_stats, axis=0), jnp.std(obs_stats, axis=0) + 1e-8
    delta_obs_scale = delta_obs_std / obs_std
    # delta_obs_shift = (delta_obs_mean - obs_mean) / obs_std
    delta_obs_shift = delta_obs_mean / obs_std

    # apply padding
    replay_buffer_data = []
    for episode in target_agnostic_minari_dataset:
        traj_len = episode.observations.shape[0]-1
        padding_len = (max_epi_len + context_len) - traj_len
        obs = jnp.array(episode.observations[:-1,:])
        next_obs = jnp.array(episode.observations[1:,:])
        delta_obs = next_obs-obs
        prev_obs = jnp.concatenate([#jnp.zeros((1, obs_dim)),
                                    jnp.array(episode.observations[:1,:]), # assume first previous step same as first step
                                    jnp.array(episode.observations[:-2,:])], axis=0)

        # apply input normalization
        if not args.rm_normalization:
            obs = (obs - obs_mean) / obs_std
            next_obs = (next_obs - obs_mean) / obs_std
            delta_obs = (delta_obs - delta_obs_mean) / delta_obs_std
            prev_obs = (prev_obs - obs_mean) / obs_std

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
        returns_to_go = jnp.concatenate([jnp.array(discount_cumsum(episode.rewards, 1.0) / rtg_scale).reshape(-1, 1), 
                                         jnp.zeros((padding_len, 1))], axis=0)
        timesteps = jnp.concatenate([jnp.arange(start=0, stop=traj_len, step=1, dtype=jnp.int32).reshape(-1, 1),
                                     jnp.zeros((padding_len, 1))], axis=0)
        traj_mask = jnp.concatenate([jnp.ones(traj_len).reshape(-1, 1),
                                     jnp.zeros((padding_len, 1))], axis=0)

        padding_data = jnp.concatenate([obs, actions, next_obs, returns_to_go, timesteps, traj_mask, prev_obs, delta_obs], axis=-1)
        assert trans_dim == padding_data.shape[-1], padding_data.shape
        replay_buffer_data.append(padding_data)

    cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1, obs_dim, obs_dim])

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
                       h_dims=[256,256],
                       drop_out_rates=args.encoder_dropout_rates)

    # eval_precoder = MLP(out_dim=act_dim,
    #                     h_dims=[256,256])

    eval_precoder = GRU_Precoder(act_dim=act_dim,
                                 context_len=context_len,
                                 hidden_size=128,
                                 autonomous=args.autonomous)

    if args.state_dep_prior:

        prior = MLP(out_dim=controlled_variables_dim*2,
                    h_dims=[256,256],
                    drop_out_rates=[0., 0.])
    
    def get_actions(obs, key, precoder_params, prior_params=None):

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

        if args.state_dep_prior:
        
            prior_z_params = jax.jit(prior.apply)(prior_params, target_agnostic_obs[None])
            prior_mean, prior_log_std = jnp.split(prior_z_params, 2, axis=-1)
            min_log_std = -20.
            max_log_std = 2.
            prior_log_std = jnp.clip(prior_log_std, min_log_std, max_log_std)
            dist_z_prior = tfd.MultivariateNormalDiag(loc=prior_mean, scale_diag=jnp.exp(prior_log_std))
            z_t = dist_z_prior.sample(seed=key)

        else:

            # sample from prior
            dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros((1,controlled_variables_dim)), scale_diag=jnp.ones((1,controlled_variables_dim)))
            z_t = dist_z_prior.sample(seed=key)
        
        actions = jax.jit(eval_precoder.apply)(precoder_params, target_agnostic_obs[None,None, :], z_t)

        # target_obs = np.zeros(3)
        # y_h = target_obs - obs[33:36]
        # y_h = (y_h - obs_mean[36:]) / obs_std[36:]
        # actions = jax.jit(eval_precoder.apply)(precoder_params, target_agnostic_obs[None,None, :], y_h[None])

        return actions[0,:,:]
    
    def eval_model(model, params, iter, loop='open'):

        if args.state_dep_prior:
            prior_params = {'params': params['params']['prior']}
        else:
            prior_params = None
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
            actions = get_actions(obs, key, precoder_params, prior_params)
            for t in range(args.context_len):
                # key, subkey = jax.random.split(key)
                # action = get_actions(obs, ep, t, subkey)[0,0,:]
                frames.append(minari_env.render())
                # cumsum_dims = np.cumsum([obs_dim, act_dim, obs_dim, 1, 1, 1])
                # action = replay_buffer.data[0,ep,t,cumsum_dims[0]:cumsum_dims[1]]
                obs, rew, terminated, truncated, info = minari_env.step(actions[t,:])

            video_bytes = io.BytesIO()
            imageio.mimwrite(video_bytes, np.stack(frames, axis=0), format='mp4') #, codec='libx264')
            video_bytes.seek(0)

            wandb.log({f"video_{loop}_{model}_{rollout}/": wandb.Video(video_bytes, format="mp4")})

            fig = plt.figure()
            for i in range(actions.shape[-1]):
                plt.plot(actions[:,i])
            wandb.log({
                f"actions_{loop}_{model}_{rollout}/": wandb.Image(fig)
            })
            plt.close()

        return None

    ###################################### evaluate dyanmics ###################################### 

    def eval_dynamics(key, params, dynamics_apply, dynamics_params, iter):

        if params is None:
            prior_params = None
            encoder_params = None
            precoder_params = None
        else:
            if args.state_dep_prior:
                prior_params = {'params': params['params']['prior']}
            else:
                prior_params = None
            encoder_params = {'params': params['params']['encoder']}
            precoder_params = {'params': params['params']['precoder']}

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
                    s_base_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))
                    eps = 1e-3
                    bounded_bijector = tfb.Chain([
                        tfb.Shift(shift=(delta_obs_min-eps/2)),
                        tfb.Scale(scale=(delta_obs_max - delta_obs_min + eps)),
                        tfb.Sigmoid(),
                    ])
                    s_dist = tfd.TransformedDistribution(distribution=s_base_dist, bijector=bounded_bijector)

                    delta_s = s_dist.sample(seed=sample_s_key)
                    delta_s = delta_s * delta_obs_scale + delta_obs_shift
                    # next_state = state + delta_s
                    s_curr = state[...,obs_dim:]
                    s_next = s_curr + delta_s 

                    s_next_mean = s_curr + (bounded_bijector(s_mean).mean(axis=0) * delta_obs_scale + delta_obs_shift)

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
            if precoder_params is None:
                actions = minari_dataset[rollout].actions[:50,:]
            else:
                actions = get_actions(obs, key, precoder_params, prior_params)
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
            
            video_bytes = io.BytesIO()
            imageio.mimwrite(video_bytes, np.stack(frames, axis=0), format='mp4') #, codec='libx264')
            video_bytes.seek(0)

            wandb.log({f"learned_dynamics_video_{rollout}/": wandb.Video(video_bytes, format="mp4")})

    ###################################### dynamics training ###################################### 

    dynamics_model = dynamics(
        h_dims_dynamics=args.h_dims_dynamics,
        state_dim=obs_dim,
        drop_out_rates=args.dynamics_dropout_rates,
        learn_dynamics_std=args.learn_dynamics_std,
    )

    if args.resume_start_time_str is not None and args.resume_dynamics is False:
    # if True:

        # load_path = os.path.join(BASE_SAVE_PATH, '2025-09-28', '5gpmag7s') # learn_dynamics_std=False
        load_path = os.path.join(BASE_SAVE_PATH, '2025-09-28', '67j25zf6') # learn_dynamics_std=True
        total_updates = 100000
        load_model_path = os.path.join(load_path, "dynamics_model" + f"_{total_updates}.pt")
        _dynamics_params = load_params(load_model_path)

        wandb_run = wandb.init(
                name=f'{env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
                group=env_d4rl_name,
                project='jax_dt',
                config=dict(args)
            )
        
        save_path = os.path.join(BASE_SAVE_PATH, timestamp, wandb_run.id)
        os.makedirs(save_path, exist_ok=True)

    else:

        if args.resume_start_time_str is None or args.resume_dynamics is False:

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
                if args.Markov_dynamics:
                    dynamics_params_list.append(dynamics_model.init({'params': key_params}, s_tm1_s_t, dummy_actions, key_dropout))
                else:
                    dynamics_params_list.append(dynamics_model.init({'params': key_params, 'dropout': key_dropout}))
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

        dynamics_optimizer_state, dynamics_params = bcast_local_devices(
            (dynamics_optimizer_state, dynamics_params), local_devices_to_use)
        
        training_state = TrainingState(
            optimizer_state=dynamics_optimizer_state,
            params=dynamics_params,
            key=jax.random.split(local_key, args.n_dynamics_ensembles * local_devices_to_use).reshape(local_devices_to_use, args.n_dynamics_ensembles, -1),
            steps=jnp.zeros((local_devices_to_use, args.n_dynamics_ensembles)))

        dynamics_grad_fn = partial(dynamics_loss,
                                   dynamics_model=dynamics_model,
                                   delta_obs_min=delta_obs_min,
                                   delta_obs_max=delta_obs_max)
        
        dynamics_grad_fn = jax.jit(jax.value_and_grad(dynamics_grad_fn, has_aux=True))

        one_train_iteration = create_one_train_iteration(dynamics_optimizer,
                                           dynamics_grad_fn,
                                           dynamics_batch_size_per_device,
                                           args.grad_updates_per_step,
                                           args.num_updates_per_iter,
                                           max_epi_len,
                                           cumsum_dims,
                                           trans_dim,
                                           start_time,
                                           sample_horizon_len=1,
                                           ensemble=True)

        total_updates = 0

        wandb_run = wandb.init(
                name=f'{env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
                group=env_d4rl_name,
                project='jax_dt',
                config=dict(args)
            )

        save_path = os.path.join(BASE_SAVE_PATH, timestamp, wandb_run.id)
        save_model_path = os.path.join(save_path, "dynamics_model")
        os.makedirs(save_model_path, exist_ok=True)

        for i_train_iter in range(max_train_iters):
            
            total_updates, training_state, replay_buffer = one_train_iteration(training_state,
                                                                               replay_buffer,
                                                                               i_train_iter,
                                                                               total_updates)

            # save model
            _dynamics_params = jax.tree_util.tree_map(lambda x: x[0], training_state.params)

            if i_train_iter % args.dynamics_save_iters == 0 or i_train_iter == max_train_iters - 1:
                save_current_model_path = save_model_path + f"_{total_updates}.pt"
                print("saving current model at: " + save_current_model_path)
                save_params(save_current_model_path, _dynamics_params)
                if args.Markov_dynamics:
                    eval_dynamics(training_state.key, None, dynamics_model.apply, _dynamics_params, total_updates)

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

    ###################################### CLVM training ###################################### 

    vae_model = CLVM(
        state_dim=obs_dim,
        act_dim=act_dim,
        controlled_variables=controlled_variables,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
        gamma=args.gamma,
        state_dependent_prior=args.state_dep_prior,
        Markov_dynamics=args.Markov_dynamics,
        n_dynamics_ensembles=args.n_dynamics_ensembles,
        horizon_embed_dim=args.horizon_embed_dim,
        trajectory_version=args.trajectory_version,
        encoder_dropout_rates=args.encoder_dropout_rates,
        autonomous=args.autonomous
    )

    if args.resume_start_time_str is not None and args.resume_vae is False:

        total_updates = 25000
        load_model_path = os.path.join(log_dir, "vae_model.pt")
        load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
        vae_params = load_params(load_current_model_path)

        wandb.init(
                name=f'{env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
                group=env_d4rl_name,
                project='jax_dt',
                config=dict(args)
            )

    else:

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
            dummy_states = jnp.zeros((batch_size, context_len, obs_dim*2))
            dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
            dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
            dummy_horizon= jnp.ones((batch_size, 1), dtype=jnp.int32)
            dummy_mask = jnp.ones((batch_size, context_len, 1))

            key_params, key_dropout = jax.random.split(global_key_vae)
            vae_params = vae_model.init({'params': key_params, 'dropout': key_dropout},
                                        s_t=dummy_states,
                                        a_t=dummy_actions,
                                        y_t=dummy_controlled_variables,
                                        horizon=dummy_horizon,
                                        mask=dummy_mask,
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

        vae_training_state = TrainingState(
            optimizer_state=vae_optimizer_state,
            params=vae_params,
            key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
            steps=jnp.zeros((local_devices_to_use,)))

        save_model_path = os.path.join(save_path, "vae_model")
        os.makedirs(save_model_path, exist_ok=True)

        CLVM_grad_fn = partial(CLVM_loss,
                               vae_model=vae_model,
                               controlled_variables=controlled_variables)
        
        CLVM_grad_fn = jax.jit(jax.value_and_grad(CLVM_grad_fn, has_aux=True))

        one_train_iteration = create_one_train_iteration(vae_optimizer,
                                           CLVM_grad_fn,
                                           vae_batch_size_per_device,
                                           args.grad_updates_per_step,
                                           args.num_updates_per_iter,
                                           max_epi_len,
                                           cumsum_dims,
                                           trans_dim,
                                           start_time,
                                           sample_horizon_len=args.context_len,
                                           ensemble=False)

        total_updates = 0

        for i_train_iter in range(max_train_iters):
            
            total_updates, vae_training_state, replay_buffer = one_train_iteration(vae_training_state,
                                                                               replay_buffer,
                                                                               i_train_iter,
                                                                               total_updates)

            # save model
            _vae_params = jax.tree_util.tree_map(lambda x: x[0], vae_training_state.params)

            if i_train_iter % args.vae_save_iters == 0 or i_train_iter == max_train_iters - 1:
                save_current_model_path = save_model_path + f"_{total_updates}.pt"
                print("saving current model at: " + save_current_model_path)
                save_params(save_current_model_path, _vae_params)
                eval_model('vae', _vae_params, total_updates) # for model in ['vae', 'emp']:
                eval_dynamics(vae_training_state.key, _vae_params, dynamics_model.apply, _dynamics_params, total_updates)

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
        gamma=args.gamma,
        Markov_dynamics=args.Markov_dynamics,
        state_dependent_source=args.state_dep_prior,
        delta_obs_scale=delta_obs_scale,
        delta_obs_shift=delta_obs_shift,
        delta_obs_min=delta_obs_min,
        delta_obs_max=delta_obs_max,
        n_dynamics_ensembles=args.n_dynamics_ensembles,
        horizon_embed_dim=args.horizon_embed_dim,
        n_particles=args.n_particles,
        encoder_dropout_rates=args.encoder_dropout_rates,
        learn_dynamics_std=args.learn_dynamics_std,
        autonomous=args.autonomous,
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
    dummy_states = jnp.zeros((batch_size, context_len, obs_dim*2))
    dummy_actions = jnp.zeros((batch_size, context_len, act_dim))
    dummy_controlled_variables = jnp.zeros((batch_size, context_len, controlled_variables_dim))
    dummy_horizon= jnp.zeros((batch_size, 1), dtype=jnp.int32)
    dummy_mask = jnp.zeros((batch_size, context_len, 1))

    key_params, key_dropout = jax.random.split(global_key_emp)
    emp_params = emp_model.init({'params': key_params, 'dropout': key_dropout},
                                s_t=dummy_states,
                                mask=dummy_mask,
                                dynamics_apply=dynamics_model.apply,
                                dynamics_params=_dynamics_params,
                                key=key_params)

    # vae_save_iters = 50000
    # load_model_path = os.path.join(save_path, "vae_model")
    # load_current_model_path = load_model_path + f"_{max_train_iters*args.vae_save_iters}.pt"
    # _vae_params = load_params(load_current_model_path)
    if args.state_dep_prior:
        emp_params['params']['prior'] = _vae_params['params']['prior']
    emp_params['params']['precoder'] = _vae_params['params']['precoder']
    emp_params['params']['encoder'] = _vae_params['params']['encoder']
    del _vae_params
    
    emp_optimizer_state = emp_optimizer.init(emp_params)

    emp_optimizer_state, emp_params = bcast_local_devices(
        (emp_optimizer_state, emp_params), local_devices_to_use)

    precoder_grad_fn = partial(precoder_loss,
                               emp_model=emp_model,
                            dynamics_apply=dynamics_model.apply,
                            dynamics_params=_dynamics_params)
    
    precoder_grad_fn = jax.jit(jax.value_and_grad(precoder_grad_fn, has_aux=True))

    one_train_iteration = create_one_train_iteration(emp_optimizer,
                                        precoder_grad_fn,
                                        emp_batch_size_per_device,
                                        args.grad_updates_per_step,
                                        args.num_updates_per_iter,
                                        max_epi_len,
                                        cumsum_dims,
                                        trans_dim,
                                        start_time,
                                        sample_horizon_len=args.context_len,
                                        ensemble=False)

    emp_training_state = TrainingState(
        optimizer_state=emp_optimizer_state,
        params=emp_params,
        key=jnp.stack(jax.random.split(local_key, local_devices_to_use)),
        steps=jnp.zeros((local_devices_to_use,)))

    total_updates = 0

    save_model_path = os.path.join(save_path, "emp_model")
    os.makedirs(save_model_path, exist_ok=True)

    for i_train_iter in range(max_train_iters):
        
        total_updates, emp_training_state, replay_buffer = one_train_iteration(emp_training_state,
                                                                            replay_buffer,
                                                                            i_train_iter,
                                                                            total_updates)

        # save model
        _emp_params = jax.tree_util.tree_map(lambda x: x[0], emp_training_state.params)

        if i_train_iter % args.emp_save_iters == 0 or i_train_iter == max_train_iters - 1:
            save_current_model_path = save_model_path + f"_{total_updates}.pt"
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

if __name__ == '__main__':
    train()