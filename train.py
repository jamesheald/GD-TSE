import os
os.environ["MUJOCO_GL"] = "egl"

import jax
print(jax.devices())
import jax.numpy as jnp
import numpy as np
import random
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from functools import partial
import wandb
from datetime import datetime
import hydra
from omegaconf import DictConfig

from src.networks import dynamics, MLP, GRU_Precoder
from src.models import CLVM, empowerment
from src.training import get_training_state, create_one_train_iteration
from src.losses import dynamics_loss, CLVM_loss, precoder_loss
from src.data import get_dataset
from src.rollout import eval_dynamics_model, eval_action_generator
from src.pmap import synchronize_hosts
from src.utils import get_local_devices_to_use, get_controlled_variables, save_params, load_params

cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'conf')
@hydra.main(config_path=cfg_path, config_name="config.yaml")
def train(args: DictConfig):

    start_time = datetime.now().replace(microsecond=0)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    BASE_SAVE_PATH = '/nfs/nhome/live/jheald/jax_dt/model_outputs'

    # set random seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    local_devices_to_use = get_local_devices_to_use(args)

    controlled_variables, args = get_controlled_variables(args)

    replay_buffer, minari_dataset, minari_env, learned_minari_env, d_args = get_dataset(args)

    ###################################### dynamics training ###################################### 

    dynamics_model = dynamics(args, d_args)

    key_dropout, subkey, key = jax.random.split(key, 3)
    dummy_states = jnp.zeros((1, d_args['obs_dim']))
    dummy_actions = jnp.zeros((1, d_args['act_dim']))
    model_kwargs = {'obs': jnp.concatenate([dummy_states, dummy_states], axis=-1),
                    'actions': dummy_actions,
                    'key': key_dropout}

    training_state, dynamics_optimizer = get_training_state(dynamics_model, model_kwargs, subkey, args, args.n_dynamics_ensembles)

    dynamics_grad_fn = partial(dynamics_loss,
                                dynamics_model=dynamics_model,
                                delta_obs_min=d_args['delta_obs_min'],
                                delta_obs_max=d_args['delta_obs_max'])
    
    dynamics_grad_fn = jax.jit(jax.value_and_grad(dynamics_grad_fn, has_aux=True))

    one_train_iteration = create_one_train_iteration(dynamics_optimizer,
                                        dynamics_grad_fn,
                                        args.dynamics_batch_size // local_devices_to_use,
                                        args.grad_updates_per_step,
                                        args.num_updates_per_iter,
                                        d_args['max_epi_len'],
                                        d_args['cumsum_dims'],
                                        d_args['trans_dim'],
                                        start_time,
                                        sample_horizon_len=1,
                                        ensemble=True)

    wandb_run = wandb.init(
            name=f'{args.env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
            group=args.env_d4rl_name,
            project='jax_dt',
            config=dict(args)
        )

    save_path = os.path.join(BASE_SAVE_PATH, timestamp, wandb_run.id)
    save_model_path = os.path.join(save_path, "dynamics_model")
    os.makedirs(save_model_path, exist_ok=True)

    total_updates = 0
    for i_train_iter in range(args.max_train_iters):
        
        total_updates, training_state, replay_buffer = one_train_iteration(training_state,
                                                                            replay_buffer,
                                                                            i_train_iter,
                                                                            total_updates)

        # save model
        _dynamics_params = jax.tree_util.tree_map(lambda x: x[0], training_state.params)

        if i_train_iter % args.dynamics_save_iters == 0 or i_train_iter == args.max_train_iters - 1:

            # render trajectories
            key = eval_dynamics_model(args, d_args, key, dynamics_model.apply, _dynamics_params, minari_dataset, minari_env, learned_minari_env)

            save_current_model_path = save_model_path + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _dynamics_params)

    synchronize_hosts()
    
    print("finished dynamics training!")

    ###################################### CLVM training ###################################### 

    vae_model = CLVM(args, d_args, controlled_variables)

    if args.state_dependent_prior:
        prior_apply = MLP(out_dim=args.controlled_variables_dim*2,
                          h_dims=args.h_dims_prior,
                          drop_out_rates=[0.,0.]).apply
    else:
        prior_apply = None
            
    precoder_apply = GRU_Precoder(act_dim=d_args['act_dim'],
                                  context_len=args.context_len,
                                  hidden_size=args.h_dims_GRU,
                                  autonomous=args.autonomous).apply

    subkey, key = jax.random.split(key)
    dummy_states = jnp.zeros((1, args.context_len, d_args['obs_dim']*2))
    dummy_actions = jnp.zeros((1, args.context_len, d_args['act_dim']))
    dummy_controlled_variables = jnp.zeros((1, args.context_len, args.controlled_variables_dim))
    dummy_horizon = jnp.zeros((1, 1), dtype=jnp.int32)
    dummy_mask = jnp.zeros((1, args.context_len, 1))
    model_kwargs = {'s_t': dummy_states,
                    'a_t': dummy_actions,
                    'y_t': dummy_controlled_variables,
                    'horizon': dummy_horizon,
                    'mask': dummy_mask,
                    'key': subkey}

    vae_training_state, vae_optimizer = get_training_state(vae_model, model_kwargs, subkey, args, ensemble_size=1)

    save_model_path = os.path.join(save_path, "vae_model")
    os.makedirs(save_model_path, exist_ok=True)

    CLVM_grad_fn = partial(CLVM_loss,
                           vae_model=vae_model,
                           controlled_variables=controlled_variables)
    
    CLVM_grad_fn = jax.jit(jax.value_and_grad(CLVM_grad_fn, has_aux=True))

    one_train_iteration = create_one_train_iteration(vae_optimizer,
                                        CLVM_grad_fn,
                                        args.vae_batch_size // local_devices_to_use,
                                        args.grad_updates_per_step,
                                        args.num_updates_per_iter,
                                        d_args['max_epi_len'],
                                        d_args['cumsum_dims'],
                                        d_args['trans_dim'],
                                        start_time,
                                        sample_horizon_len=args.context_len,
                                        ensemble=False)

    total_updates = 0
    for i_train_iter in range(args.max_train_iters):
        
        total_updates, vae_training_state, replay_buffer = one_train_iteration(vae_training_state,
                                                                            replay_buffer,
                                                                            i_train_iter,
                                                                            total_updates)

        # save model
        _vae_params = jax.tree_util.tree_map(lambda x: x[0], vae_training_state.params)

        if i_train_iter % args.vae_save_iters == 0 or i_train_iter == args.max_train_iters - 1:

            # render trajectories
            key = eval_action_generator('vae', _vae_params, key, minari_env, args, d_args, precoder_apply, prior_apply) # model in ['vae', 'emp']
            key = eval_dynamics_model(args, d_args, key, dynamics_model.apply, _dynamics_params, minari_dataset, minari_env, learned_minari_env, _vae_params, precoder_apply, prior_apply)

            save_current_model_path = save_model_path + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _vae_params)

    synchronize_hosts()
    
    print("finished CLVM training!")

    ###################################### mutual information training ###################################### 

    emp_model = empowerment(args, d_args, controlled_variables, dynamics_model.apply)

    subkey, key = jax.random.split(key)
    model_kwargs = {'s_t': dummy_states,
                    'mask': dummy_mask,
                    'dynamics_params': _dynamics_params,
                    'key': subkey}

    emp_training_state, emp_optimizer = get_training_state(emp_model, model_kwargs, subkey, args, ensemble_size=1)

    from flax.core import freeze, unfreeze
    def replace_params(training_state, target_params, key_to_replace):
        params = training_state.params
        params['params'][key_to_replace] = target_params['params'][key_to_replace]
        training_state = training_state.replace(params=params)
        return training_state

    keys_to_replace = ['encoder', 'precoder']
    if args.state_dependent_prior:
        keys_to_replace.append('prior')

    for key_to_replace in keys_to_replace:
        emp_training_state = replace_params(emp_training_state, vae_training_state.params, key_to_replace)

    precoder_grad_fn = partial(precoder_loss,
                               emp_model=emp_model,
                               dynamics_params=_dynamics_params)
    
    precoder_grad_fn = jax.jit(jax.value_and_grad(precoder_grad_fn, has_aux=True))

    one_train_iteration = create_one_train_iteration(emp_optimizer,
                                        precoder_grad_fn,
                                        args.emp_batch_size // local_devices_to_use,
                                        args.grad_updates_per_step,
                                        args.num_updates_per_iter,
                                        d_args['max_epi_len'],
                                        d_args['cumsum_dims'],
                                        d_args['trans_dim'],
                                        start_time,
                                        sample_horizon_len=args.context_len,
                                        ensemble=False)

    save_model_path = os.path.join(save_path, "emp_model")
    os.makedirs(save_model_path, exist_ok=True)

    total_updates = 0
    for i_train_iter in range(args.max_train_iters):
        
        total_updates, emp_training_state, replay_buffer = one_train_iteration(emp_training_state,
                                                                            replay_buffer,
                                                                            i_train_iter,
                                                                            total_updates)

        # save model
        _emp_params = jax.tree_util.tree_map(lambda x: x[0], emp_training_state.params)

        if i_train_iter % args.emp_save_iters == 0 or i_train_iter == args.max_train_iters - 1:

            # render trajectories
            key = eval_action_generator('emp', _emp_params, key, minari_env, args, d_args, precoder_apply, prior_apply) # model in ['vae', 'emp']

            save_current_model_path = save_model_path + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _emp_params)

    synchronize_hosts()
    
    print("finished mutual information training!")

if __name__ == '__main__':
    train()