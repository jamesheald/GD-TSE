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

from src.utils.utils import get_mean_and_log_std, standardise_data, unstandardise_data

def get_rollout_function(dynamics_apply, args, d_args, eps=1e-3):
    """
    Create a batched rollout function using an ensemble of dynamics models.

    Args:
        dynamics_apply (Callable): Function to compute dynamics predictions for a single ensemble member.
        args (Any): Configuration object with hyperparameters.
        d_args (Any): Environment-specific arguments (e.g., obs dimension).
        eps (float, optional): Small epsilon to prevent numerical issues. Defaults to 1e-3.

    Returns:
        Callable: A vectorized rollout function `batch_peform_rollout(state, key, actions, dynamics_params)` that returns:
            next_state (array): Next states for each step in the rollout.
            info_gain (array): Information gain for each step in the rollout.
    """

    # sample a state sequence autoregressively from the learned markov dynamics model
    def peform_rollout(state, key, actions, dynamics_params):
        
        def step_fn(carry, action):
            state, key, dynamics_params = carry
            key, dropout_key, sample_i_key, sample_s_key = jax.random.split(key, 4)
            
            # predict the delta observation
            dropout_keys = jax.random.split(dropout_key, args.n_dynamics_ensembles)
            s_dist_params = jax.vmap(dynamics_apply, in_axes=(0,None,None,0))(dynamics_params, state, action, dropout_keys)
            s_mean, s_log_std = get_mean_and_log_std(s_dist_params)

            # calculate the info gain
            al_std = jnp.clip(jnp.sqrt(jnp.square(jnp.exp(s_log_std)).mean(0)), min=1e-3)
            ep_std = s_mean.std(axis=0)
            ratio = jnp.square(ep_std / al_std)
            info_gain = jnp.log(1 + ratio).mean(axis=-1)

            # sample a delta observation from the ensemble
            idx = jax.random.categorical(sample_i_key, jnp.ones(args.n_dynamics_ensembles), axis=-1)
            bounded_bijector = tfb.Chain([
                tfb.Shift(shift=(d_args['delta_obs_min'] - eps/2)),
                tfb.Scale(scale=(d_args['delta_obs_max'] - d_args['delta_obs_min'] + eps)),
                tfb.Sigmoid(),
            ])
            s_base_dist = tfd.MultivariateNormalDiag(loc=s_mean[idx], scale_diag=jnp.exp(s_log_std[idx]))
            s_dist = tfd.TransformedDistribution(distribution=s_base_dist, bijector=bounded_bijector)
            delta_s = s_dist.sample(seed=sample_s_key)
            
            # calculate the next observation by adding the delta observation to the current observation
            delta_s = unstandardise_data(delta_s, d_args['delta_obs_mean'], d_args['delta_obs_std'])
            delta_s = standardise_data(delta_s, d_args['obs_mean'], d_args['obs_std'])
            s_curr = state[..., d_args['obs_dim']:]
            s_next = s_curr + delta_s

            # concatenate the current observation with the next observation to define the state
            next_state = jnp.concatenate([s_curr, s_next], axis=-1)

            carry = next_state, key, dynamics_params
            return carry, (s_next, info_gain)

        carry = state, key, dynamics_params
        _, (next_state, info_gain) = jax.lax.scan(step_fn, carry, actions)
        
        return next_state, info_gain

    batch_peform_rollout = jax.vmap(peform_rollout, in_axes=(0,0,0,None))

    return batch_peform_rollout

def sample_actions(obs, key, precoder_params, args, d_args, prior_params, precoder_apply, prior_apply):

    # make hand and ball position relative to initial position of hand
    target_agnostic_obs = obs - np.concatenate((np.zeros(33),
                                                obs[33:36],
                                                obs[33:36])) 
    target_agnostic_obs = standardise_data(target_agnostic_obs, d_args['obs_mean'], d_args['obs_std'])
    target_agnostic_obs = jnp.concatenate((target_agnostic_obs, target_agnostic_obs), axis=-1)

    if args.state_dependent_prior:
    
        # state-dependent prior
        prior_z_params = jax.jit(prior_apply)(prior_params, target_agnostic_obs[None])
        prior_mean, prior_log_std = get_mean_and_log_std(prior_z_params)
        dist_z_prior = tfd.MultivariateNormalDiag(loc=prior_mean,
                                                  scale_diag=jnp.exp(prior_log_std))
    
    else:

        # state-independent prior (standard normal)
        dist_z_prior = tfd.MultivariateNormalDiag(loc=jnp.zeros((1, args.controlled_variables_dim)),
                                                  scale_diag=jnp.ones((1, args.controlled_variables_dim)))
        
    # sample latent action from prior
    z_samp = dist_z_prior.sample(seed=key)
    
    # precoder latent action
    actions = jax.jit(precoder_apply)(precoder_params, target_agnostic_obs[None,None, :], z_samp)

    return actions[0,:,:]

###################################### evaluate action generator ###################################### 

def eval_action_generator(model, params, key, minari_env, args, d_args, precoder_apply, prior_apply, loop='open'):

    key, actions_key = jax.random.split(key)

    if args.state_dependent_prior:
        prior_params = {'params': params['params']['prior']}
    else:
        prior_params = None
    q_posterior_params = {'params': params['params']['q_posterior']}
    precoder_params = {'params': params['params']['precoder']}

    n_rollouts = args.n_rollouts
    for rollout in range(n_rollouts):
        obs, _ = minari_env.reset()
        frames = []
        actions = sample_actions(obs, actions_key, precoder_params, args, d_args, prior_params, precoder_apply, prior_apply)
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

###################################### evaluate learned dyanmics model ###################################### 

def eval_dynamics_model(args, d_args, key, dynamics_apply, dynamics_params, minari_dataset, minari_env, learned_minari_env, params=None, precoder_apply=None, prior_apply=None):

    if params is None:
        prior_params = None
        q_posterior_params = None
        precoder_params = None
    else:
        if args.state_dependent_prior:
            prior_params = {'params': params['params']['prior']}
        else:
            prior_params = None
        q_posterior_params = {'params': params['params']['q_posterior']}
        precoder_params = {'params': params['params']['precoder']}

    def get_predicted_obs(obs, key, actions, dynamics_params):

        batch_peform_rollout = get_rollout_function(dynamics_apply, args, d_args)

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
            actions = sample_actions(obs, precoder_key, precoder_params, args, d_args, prior_params, precoder_apply, prior_apply)
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