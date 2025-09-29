import jax
import jax.numpy as jnp
import numpy as np
import io

import wandb
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import imageio
from matplotlib import pyplot as plt

from decision_transformer.dt.models.models import get_rollout_function
from decision_transformer.dt.utils import get_mean_and_log_std, standardise_data, unstandardise_data

###################################### evaluation ###################################### 

def get_actions(obs, key, precoder_params, args, d_args, prior_params, precoder_apply, prior_apply):

    # precoder_apply
    # prior_apply

    # make hand and ball position relative to initial position of hand
    target_agnostic_obs = obs - np.concatenate((np.zeros(33),
                                                obs[33:36],
                                                obs[33:36])) 
    target_agnostic_obs = standardise_data(target_agnostic_obs, d_args['obs_mean'], d_args['obs_std'])
    target_agnostic_obs = jnp.concatenate((target_agnostic_obs, target_agnostic_obs), axis=-1)

    if args.state_dep_prior:
    
        # state-dependent prior
        prior_z_params = jax.jit(prior_apply)(prior_params, target_agnostic_obs[None])
        prior_mean, prior_log_std = get_mean_and_log_std(prior_z_params)
        dist_z_prior = tfd.MultivariateNormalDiag(loc=prior_mean,
                                                  scale_diag=jnp.exp(prior_log_std))
    
    else:

        # state-independent prior (standard normal)
        dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros((1, args.controlled_variables_dim)),
                                                  scale_diag=jnp.ones((1, args.controlled_variables_dim)))
        
    # sample from prior
    z_samp = dist_z_prior.sample(seed=key)
    
    # precoder latent action via precoder
    actions = jax.jit(precoder_apply)(precoder_params, target_agnostic_obs[None,None, :], z_samp)

    return actions[0,:,:]

def eval_model(model, params, key, minari_env, args, d_args, precoder_apply, prior_apply, loop='open'):

    key, actions_key = jax.random.split(key)

    if args.state_dep_prior:
        prior_params = {'params': params['params']['prior']}
    else:
        prior_params = None
    encoder_params = {'params': params['params']['encoder']}
    precoder_params = {'params': params['params']['precoder']}

    n_rollouts = args.n_rollouts
    for rollout in range(n_rollouts):
        obs, _ = minari_env.reset()
        frames = []
        actions = get_actions(obs, actions_key, precoder_params, args, d_args, prior_params, precoder_apply, prior_apply)
        for t in range(args.context_len):
            frames.append(minari_env.render())
            obs, rew, terminated, truncated, info = minari_env.step(actions[t,:])

        video_bytes = io.BytesIO()
        imageio.mimwrite(video_bytes, np.stack(frames, axis=0), format='mp4')
        video_bytes.seek(0)
        wandb.log({f"video_{loop}_{model}_{rollout}/": wandb.Video(video_bytes, format="mp4")})

        fig = plt.figure()
        for i in range(actions.shape[-1]):
            plt.plot(actions[:,i])
        wandb.log({
            f"actions_{loop}_{model}_{rollout}/": wandb.Image(fig)
        })
        plt.close()

    return key

###################################### evaluate dyanmics ###################################### 

def eval_dynamics(args, d_args, key, dynamics_apply, dynamics_params, minari_dataset, minari_env, learned_minari_env, params=None, precoder_apply=None, prior_apply=None):

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

    def get_predicted_obs(obs, key, actions, dynamics_params):

        batch_peform_rollout = get_rollout_function(dynamics_apply,
                                                        args.n_dynamics_ensembles,
                                                        d_args['obs_dim'],
                                                        d_args['delta_obs_min'],
                                                        d_args['delta_obs_max'],
                                                        d_args['delta_obs_scale'],
                                                        d_args['delta_obs_shift'])

        target_agnostic_obs = obs - np.concatenate((np.zeros(33),
                                        obs[33:36],
                                        obs[33:36]))

        target_agnostic_obs = standardise_data(target_agnostic_obs, d_args['obs_mean'], d_args['obs_std'])

        target_agnostic_obs = jnp.concatenate((target_agnostic_obs, target_agnostic_obs), axis=-1)

        dynamics_keys = jax.random.split(key, actions.shape[0])
        predicted_obs_traj, _ = batch_peform_rollout(target_agnostic_obs[None], dynamics_keys, actions, dynamics_params)

        return predicted_obs_traj

    n_rollouts = args.n_rollouts
    for rollout in range(n_rollouts):
        
        # reset the environment and get the initial state
        obs, _ = minari_env.reset()
        env_state = {'obj_pos': minari_env.unwrapped.get_env_state()['obj_pos'],
                        'qpos': minari_env.unwrapped.get_env_state()['qpos'],
                        'qvel': minari_env.unwrapped.get_env_state()['qvel'],
                        'target_pos': minari_env.unwrapped.get_env_state()['target_pos']}
        
        # instantiate a second environment to use to render the state predicted under the learned dynamics model
        _ = learned_minari_env.reset()

        # match the initial state 
        learned_minari_env.unwrapped.set_env_state(env_state)

        key, precoder_key, dynamics_key = jax.random.split(key, 3)

        if precoder_params is not None:
            # generate actions via precoder
            actions = get_actions(obs, precoder_key, precoder_params, args, d_args, prior_params, precoder_apply, prior_apply)
        else:
            # use actions from the original dataset if no precoder available
            actions = minari_dataset[rollout].actions[:50, :]

        predicted_obs_traj = get_predicted_obs(obs, dynamics_key, actions[None], dynamics_params)
        predicted_obs_traj = unstandardise_data(predicted_obs_traj, d_args['obs_mean'], d_args['obs_std'])
        predicted_obs_traj += np.concatenate((np.zeros(33),
                                                obs[33:36],
                                                obs[33:36]))[None,None]
        predicted_obs_traj = np.array(predicted_obs_traj)

        qpos_obj_pos = predicted_obs_traj[:, :, 36:] - obs[None,None,36:]
        
        frames = []
        frames.append(np.concatenate((minari_env.render(), learned_minari_env.render()), axis=1))
        qpos = np.zeros(36)
        for t in range(args.context_len):

            # step with action and render state
            obs, rew, terminated, truncated, info = minari_env.step(actions[t,:])
            true_state_rendered = minari_env.render()

            # update environment state to that predicted by learned dynamics model and render the predicted state
            qpos[:30] = predicted_obs_traj[0, t, :30]
            qpos[30:33] = qpos_obj_pos[0, t, :]
            learned_minari_env.unwrapped.set_state(qpos=qpos, qvel=qpos*0.) # qvel unused for rendering
            predicted_state_rendered = learned_minari_env.render()

            # render true state and predicted state side-by-side
            frames.append(np.concatenate((true_state_rendered, predicted_state_rendered), axis=1))
        
        # log video to wandb
        video_bytes = io.BytesIO()
        imageio.mimwrite(video_bytes, np.stack(frames, axis=0), format='mp4')
        video_bytes.seek(0)
        wandb.log({f"learned_dynamics_video_{rollout}/": wandb.Video(video_bytes, format="mp4")})

    return key