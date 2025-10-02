import os
os.environ["MUJOCO_GL"] = "egl"

import jax
print(jax.devices())
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import io
import imageio
import wandb
import hydra
from omegaconf import DictConfig

from src.networks.networks import posterior, GRU_Precoder
from src.utils.utils import get_mean_and_log_std, standardise_data
from src.models.horizon_sampler import sample_time_step
from src.data import get_dataset
from src.utils import load_params, get_controlled_variables

cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'conf')
@hydra.main(config_path=cfg_path, config_name="config.yaml")
def control(args: DictConfig):

    def sample_actions(obs, goal_y, key, precoder_params, args, d_args, q_posterior_params, precoder_apply, q_posterior_apply, H_step):

        sample_z_key, H_step_key, posterior_key = jax.random.split(key, 3)

        if H_step is None:
            logits = jnp.arange(args.context_len) * jnp.log(args.gamma)
            H_step = sample_time_step(logits, args.context_len, args.context_len, H_step_key)

        obs = standardise_data(obs, d_args['obs_mean'], d_args['obs_std'])
        obs = jnp.concatenate((obs, obs), axis=-1)

        goal_y = standardise_data(goal_y,
                                  d_args['obs_mean'][args.controlled_variables_idx],
                                  d_args['obs_std'][args.controlled_variables_idx])
        
        # sample latent action from posterior
        z_dist_params = jax.jit(q_posterior_apply)(q_posterior_params,
                                                   obs[None],
                                                   H_step[None],
                                                   goal_y[None],
                                                   posterior_key)
        
        z_mean, z_log_std = get_mean_and_log_std(z_dist_params)
        dist_z_post = tfd.MultivariateNormalDiag(loc=z_mean,
                                                scale_diag=jnp.exp(z_log_std))
        z_samp = dist_z_post.sample(seed=sample_z_key)
        
        # precoder latent action
        actions = jax.jit(precoder_apply)(precoder_params, obs[None,None, :], z_samp)

        return actions[0,:int(H_step)+1,:]

    ###################################### evaluate action generator ###################################### 

    def remove_task_info_from_goal(obs, initial_obs):

        # make hand and ball position relative to initial position of hand
        target_agnostic_obs = obs - np.concatenate((np.zeros(33),
                                                    initial_obs[33:36],
                                                    initial_obs[33:36])) 

        return target_agnostic_obs

    def eval_controller(params, key, minari_env, args, d_args, precoder_apply, q_posterior_apply, loop='closed'):

        key, actions_key = jax.random.split(key)

        q_posterior_params = {'params': params['params']['q_posterior']}
        precoder_params = {'params': params['params']['precoder']}

        # H_step = jnp.array(5, dtype=jnp.int32)
        H_step = None

        n_rollouts = args.n_rollouts
        for rollout in range(n_rollouts):
            frames = []
            done = False
            obs, _ = minari_env.reset()
            initial_obs = obs.copy()
            target_agnostic_obs = remove_task_info_from_goal(obs, initial_obs)
            goal_y = -initial_obs[33:36] # target_pos - palm_pos_0; target_agnostic_obs[36:39] - obs[36:39] # obj_pos - palm_pos_0 - (obj_pos - target_pos) = target_pos - palm_pos_0
            while not done:

                target_agnostic_obs = remove_task_info_from_goal(obs, initial_obs)
                actions = sample_actions(target_agnostic_obs, goal_y, actions_key, precoder_params, args, d_args, q_posterior_params, precoder_apply, q_posterior_apply, H_step)

                for action in actions:
                    frames.append(minari_env.render())
                    obs, rew, terminated, truncated, info = minari_env.step(action)
                    if terminated or truncated:
                        done = True
                        break
                if done:
                    break

            video_bytes = io.BytesIO()
            imageio.mimwrite(video_bytes, np.stack(frames, axis=0), format='mp4')
            video_bytes.seek(0)
            wandb.log({f"video_{loop}_{rollout}/": wandb.Video(video_bytes, format="mp4")})

    
    wandb.init(
        name=f'{args.env_d4rl_name}-control',
        group=args.env_d4rl_name,
        project='GD-TSE-control',
        config=dict(args)
    )

    args = get_controlled_variables(args)
    _, _, minari_env, _, d_args = get_dataset(args)
    
    precoder_apply = GRU_Precoder(args, d_args).apply
    q_posterior_apply = posterior(args, d_args).apply

    model_params = load_params(args.load_CVLM_path)

    args.base_save_path + '/2025-10-02/z50b3fy1/emp_model_1000000.pt'

    key = jax.random.PRNGKey(0)

    eval_controller(model_params, key, minari_env, args, d_args, precoder_apply, q_posterior_apply, loop='closed')

if __name__ == '__main__':
    control()