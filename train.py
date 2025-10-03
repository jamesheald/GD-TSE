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
from src.losses import dynamics_grad, CLVM_grad, precoder_grad
from src.data import get_dataset
from src.rollout import eval_dynamics_model, eval_action_generator
from src.pmap import synchronize_hosts, bcast_local_devices
from src.utils import get_local_devices_to_use, get_controlled_variables, save_params, load_params, replace_params
from control import control

cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'conf')
@hydra.main(config_path=cfg_path, config_name="config.yaml")
def train(args: DictConfig):

    # set random seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    wandb_run = wandb.init(
        # name=f'{args.env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
        group=args.env_d4rl_name,
        project='GD-TSE-train',
        config=dict(args)
    )

    start_time = datetime.now().replace(microsecond=0)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(args.base_save_path, timestamp, wandb_run.id)
    save_model_path = os.path.join(save_path, "dynamics_model")
    os.makedirs(save_model_path, exist_ok=True)

    local_devices_to_use = get_local_devices_to_use(args)
    args = get_controlled_variables(args)
    replay_buffer, minari_dataset, minari_env, learned_minari_env, d_args = get_dataset(args)

    ###################################### dynamics training ###################################### 

    dynamics_model = dynamics(args, d_args)

    if args.load_dynamics_path is not None:

        # load dynamics model
        _dynamics_params = load_params(args.load_dynamics_path)

    else:

        # train dynamics model
        key_dropout, subkey, key = jax.random.split(key, 3)
        dummy_states = jnp.zeros((1, d_args['obs_dim']*2))
        dummy_actions = jnp.zeros((1, d_args['act_dim']))
        model_kwargs = {'obs': dummy_states,
                        'actions': dummy_actions,
                        'key': key_dropout}

        training_state, dynamics_optimizer = get_training_state(dynamics_model, model_kwargs, subkey, args, args.n_dynamics_ensembles)

        dynamics_grad_fn = partial(dynamics_grad, dynamics_model=dynamics_model, d_args=d_args)

        one_train_iteration = create_one_train_iteration(dynamics_optimizer,
                                                         dynamics_grad_fn,
                                                         args,
                                                         args.dynamics_batch_size // local_devices_to_use,
                                                         d_args,
                                                         start_time,
                                                         sample_horizon_len=1,
                                                         ensemble=True)

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
                save_params(save_current_model_path, _dynamics_params)

        synchronize_hosts()
        
        print("finished dynamics training!")

    ###################################### CLVM training ###################################### 

    vae_model = CLVM(args, d_args)

    if args.state_dependent_prior:
        prior_apply = MLP(out_dim=args.controlled_variables_dim*2,
                          h_dims=args.h_dims_prior,
                          drop_out_rates=[0.,0.]).apply
    else:
        prior_apply = None
            
    precoder_apply = GRU_Precoder(args, d_args).apply

    if args.load_CVLM_path is not None:

        # load CLVM model
        _vae_params = load_params(args.load_CVLM_path)

    else:

        # train CLVM model
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

        CLVM_grad_fn = partial(CLVM_grad, vae_model=vae_model, args=args)

        one_train_iteration = create_one_train_iteration(vae_optimizer,
                                                         CLVM_grad_fn,
                                                         args,
                                                         args.vae_batch_size // local_devices_to_use,
                                                         d_args,
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
                save_params(save_current_model_path, _vae_params)

        synchronize_hosts()
        
        print("finished CLVM training!")

    ###################################### mutual information training ###################################### 

    emp_model = empowerment(args, d_args, dynamics_model.apply)

    subkey, key = jax.random.split(key)
    dummy_states = jnp.zeros((1, args.context_len, d_args['obs_dim']*2))
    dummy_mask = jnp.zeros((1, args.context_len, 1))
    model_kwargs = {'s_t': dummy_states,
                    'mask': dummy_mask,
                    'dynamics_params': _dynamics_params,
                    'key': subkey}

    emp_training_state, emp_optimizer = get_training_state(emp_model, model_kwargs, subkey, args, ensemble_size=1)

    keys_to_replace = ['precoder', 'q_posterior']
    if args.state_dependent_prior:
        keys_to_replace.append('prior')

    vae_params = bcast_local_devices(_vae_params, local_devices_to_use)
    for key_to_replace in keys_to_replace:
        emp_training_state = replace_params(emp_training_state, vae_params, key_to_replace)

    precoder_grad_fn = partial(precoder_grad, emp_model=emp_model, dynamics_params=_dynamics_params)

    one_train_iteration = create_one_train_iteration(emp_optimizer,
                                                     precoder_grad_fn,
                                                     args,
                                                     args.emp_batch_size // local_devices_to_use,
                                                     d_args,
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
            save_params(save_current_model_path, _emp_params)

    synchronize_hosts()
    
    print("finished mutual information training!")

    ###################################### evaluate control ###################################### 

    # use the last saved (trained) model to evaluate control
    args.load_control_path = save_current_model_path
    control(args)

if __name__ == '__main__':
    train()