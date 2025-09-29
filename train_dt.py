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

from decision_transformer.dt.networks.networks import dynamics, MLP, GRU_Precoder
from decision_transformer.dt.models.models import CLVM, empowerment
from decision_transformer.dt.utils import get_local_devices_to_use, save_params, load_params
from decision_transformer.pmap import synchronize_hosts
from scripts.train_loop import create_one_train_iteration
from scripts.losses.losses import dynamics_loss, CLVM_loss, precoder_loss
from scripts.initialise_model import get_training_state
from scripts.get_data import get_dataset
from scripts.evaluate import eval_dynamics, eval_model

cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'conf')
@hydra.main(config_path=cfg_path, config_name="config.yaml")
def train(args: DictConfig):

    start_time = datetime.now().replace(microsecond=0)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    BASE_SAVE_PATH = '/nfs/nhome/live/jheald/jax_dt/model_outputs'

    # controlled_variables_dim = 6
    # controlled_variables = [i for i in range(3)] # hand pos
    # controlled_variables += [36 + i for i in range(3)] # object pos
    controlled_variables_dim = 3
    controlled_variables = [36 + i for i in range(3)] # object pos

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    local_devices_to_use = get_local_devices_to_use(args)

    replay_buffer, minari_dataset, minari_env, learned_minari_env, d_args = get_dataset(args)

    ###################################### dynamics training ###################################### 

    dynamics_model = dynamics(
        h_dims_dynamics=args.h_dims_dynamics,
        state_dim=d_args['obs_dim'],
        drop_out_rates=args.dynamics_dropout_rates,
        learn_dynamics_std=args.learn_dynamics_std,
    )

    # if args.resume_start_time_str is not None and args.resume_dynamics is False:
    # if True:

    # # load_path = os.path.join(BASE_SAVE_PATH, '2025-09-28', '5gpmag7s') # learn_dynamics_std=False
    # load_path = os.path.join(BASE_SAVE_PATH, '2025-09-28', '67j25zf6') # learn_dynamics_std=True
    # total_updates = 100000
    # load_model_path = os.path.join(load_path, "dynamics_model" + f"_{total_updates}.pt")
    # _dynamics_params = load_params(load_model_path)

    # wandb_run = wandb.init(
    #         name=f'{args.env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
    #         group=args.env_d4rl_name,
    #         project='jax_dt',
    #         config=dict(args)
    #     )
    
    # save_path = os.path.join(BASE_SAVE_PATH, timestamp, wandb_run.id)
    # os.makedirs(save_path, exist_ok=True)

    # else:

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
            key = eval_dynamics(args, d_args, key, dynamics_model.apply, _dynamics_params, minari_dataset, minari_env, learned_minari_env)

            save_current_model_path = save_model_path + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _dynamics_params)

    synchronize_hosts()
    
    print("finished dynamics training!")

    ###################################### CLVM training ###################################### 

    vae_model = CLVM(
        state_dim=d_args['obs_dim'],
        act_dim=d_args['act_dim'],
        controlled_variables=controlled_variables,
        controlled_variables_dim=controlled_variables_dim,
        n_blocks=args.n_blocks ,
        h_dim=args.embed_dim,
        context_len=args.context_len,
        n_heads=args.n_heads,
        drop_p=args.dropout_p,
        gamma=args.gamma,
        state_dependent_prior=args.state_dep_prior,
        Markov_dynamics=args.Markov_dynamics,
        n_dynamics_ensembles=args.n_dynamics_ensembles,
        horizon_embed_dim=args.horizon_embed_dim,
        trajectory_version=args.trajectory_version,
        encoder_dropout_rates=args.encoder_dropout_rates,
        autonomous=args.autonomous,
        h_dims_prior=args.h_dims_prior,
        h_dims_encoder=args.h_dims_encoder,
        h_dims_GRU=args.h_dims_GRU,
    )

    if args.state_dep_prior:
        prior_apply = MLP(out_dim=controlled_variables_dim*2,
                          h_dims=args.h_dims_prior,
                          drop_out_rates=[0.,0.]).apply
    else:
        prior_apply = None
            
    precoder_apply = GRU_Precoder(act_dim=d_args['act_dim'],
                                  context_len=args.context_len,
                                  hidden_size=args.h_dims_GRU,
                                  autonomous=args.autonomous).apply

    # if args.resume_start_time_str is not None and args.resume_vae is False:

    #     total_updates = 25000
    #     load_model_path = os.path.join(log_dir, "vae_model.pt")
    #     load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
    #     vae_params = load_params(load_current_model_path)

    #     wandb.init(
    #             name=f'{args.env_d4rl_name}-{random.randint(int(1e5), int(1e6) - 1)}',
    #             group=args.env_d4rl_name,
    #             project='jax_dt',
    #             config=dict(args)
    #         )

    # else:

    #     if args.resume_start_time_str is None or args.resume_vae is False:
        
        # else:

        #     vae_optimizer = optax.chain(
        #         optax.clip(args.gradient_clipping),
        #         optax.adamw(learning_rate=lr, weight_decay=wt_decay),
        #     )

        #     total_updates = 300000
        #     load_model_path = os.path.join(log_dir, "vae_model.pt")
        #     load_current_model_path = load_model_path[:-3] + f"_{total_updates}.pt"
        #     vae_params = load_params(load_current_model_path)

    subkey, key = jax.random.split(key)
    dummy_states = jnp.zeros((1, args.context_len, d_args['obs_dim']*2))
    dummy_actions = jnp.zeros((1, args.context_len, d_args['act_dim']))
    dummy_controlled_variables = jnp.zeros((1, args.context_len, controlled_variables_dim))
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
            key = eval_model('vae', _vae_params, key, minari_env, args, d_args, precoder_apply, prior_apply) # model in ['vae', 'emp']
            key = eval_dynamics(args, d_args, key, dynamics_model.apply, _dynamics_params, minari_dataset, minari_env, learned_minari_env, _vae_params, precoder_apply, prior_apply)

            save_current_model_path = save_model_path + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _vae_params)

    synchronize_hosts()
    
    print("finished CLVM training!")

    ###################################### mutual information training ###################################### 

    emp_model = empowerment(
        state_dim=d_args['obs_dim'],
        act_dim=d_args['act_dim'],
        controlled_variables_dim=controlled_variables_dim,
        controlled_variables=controlled_variables,
        n_blocks=args.n_blocks ,
        h_dim=args.embed_dim,
        context_len=args.context_len,
        n_heads=args.n_heads,
        drop_p=args.dropout_p,
        gamma=args.gamma,
        Markov_dynamics=args.Markov_dynamics,
        state_dependent_source=args.state_dep_prior,
        delta_obs_scale=d_args['delta_obs_scale'],
        delta_obs_shift=d_args['delta_obs_shift'],
        delta_obs_min=d_args['delta_obs_min'],
        delta_obs_max=d_args['delta_obs_max'],
        n_dynamics_ensembles=args.n_dynamics_ensembles,
        horizon_embed_dim=args.horizon_embed_dim,
        n_particles=args.n_particles,
        encoder_dropout_rates=args.encoder_dropout_rates,
        learn_dynamics_std=args.learn_dynamics_std,
        autonomous=args.autonomous,
        dynamics_apply=dynamics_model.apply,
        h_dims_prior=args.h_dims_prior,
        h_dims_encoder=args.h_dims_encoder,
        h_dims_GRU=args.h_dims_GRU,
    )

    subkey, key = jax.random.split(key)
    model_kwargs = {'s_t': dummy_states,
                    'mask': dummy_mask,
                    'dynamics_params': _dynamics_params,
                    'key': subkey}

    emp_training_state, emp_optimizer = get_training_state(emp_model, model_kwargs, subkey, args, ensemble_size=1)
    
    # vae_save_iters = 50000
    # load_model_path = os.path.join(save_path, "vae_model")
    # load_current_model_path = load_model_path + f"_{args.max_train_iters*args.vae_save_iters}.pt"
    # _vae_params = load_params(load_current_model_path)

    from flax.core import freeze, unfreeze
    def replace_params(training_state, target_params, key_to_replace):
        params = training_state.params
        params['params'][key_to_replace] = target_params['params'][key_to_replace]
        training_state = training_state.replace(params=params)
        return training_state

    keys_to_replace = ['encoder', 'precoder']
    if args.state_dep_prior:
        keys_to_replace.append('prior')

    for key_to_replace in keys_to_replace:
        emp_training_state = replace_params(emp_training_state, vae_training_state.params, key_to_replace)

    precoder_grad_fn = partial(precoder_loss,
                               emp_model=emp_model,
                               dynamics_apply=dynamics_model.apply,
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
            key = eval_model('emp', _emp_params, key, minari_env, args, d_args, precoder_apply, prior_apply) # model in ['vae', 'emp']

            save_current_model_path = save_model_path + f"_{total_updates}.pt"
            print("saving current model at: " + save_current_model_path)
            save_params(save_current_model_path, _emp_params)

    synchronize_hosts()
    
    print("finished mutual information training!")

if __name__ == '__main__':
    train()