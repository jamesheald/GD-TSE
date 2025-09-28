from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from datetime import datetime
import wandb

from decision_transformer.dt.utils import TrainingState, Transition, ReplayBuffer
from decision_transformer.pmap import is_replicated

# sample_horizon_len  # dynamics_context_len = 1, context_len

def create_one_train_iteration(optimizer,
                               grad_fn,
                               batch_size_per_device,
                               grad_updates_per_step,
                               num_updates_per_iter,
                               max_epi_len,
                               cumsum_dims,
                               trans_dim,
                               start_time,
                               sample_horizon_len,
                               ensemble):

    @jax.jit
    def update_step(state: TrainingState, transitions: jnp.ndarray) -> Tuple[TrainingState, bool, Dict[str, jnp.ndarray]]:
        
        transitions = Transition(
            s_t=transitions[:, :, :cumsum_dims[0]],
            a_t=transitions[:, :, cumsum_dims[0]:cumsum_dims[1]],
            s_tp1=transitions[:, :, cumsum_dims[1]:cumsum_dims[2]],
            rtg_t=transitions[:, :, cumsum_dims[2]:cumsum_dims[3]],
            ts=transitions[:, :, cumsum_dims[3]:cumsum_dims[4]],
            mask_t=transitions[:, :, cumsum_dims[4]:cumsum_dims[5]],
            s_tm1=transitions[:, :, cumsum_dims[5]:cumsum_dims[6]],
            d_s=transitions[:, :, cumsum_dims[6]:cumsum_dims[7]]
        )

        key, subkey = jax.random.split(state.key, 2)

        (loss, metrics), grads = grad_fn(state.params, transitions, subkey)

        grads = jax.lax.pmean(grads, axis_name='i')
        params_update, optimizer_state = optimizer.update(
            grads, state.optimizer_state, state.params)
        updated_params = optax.apply_updates(state.params, params_update)

        new_state = TrainingState(
            optimizer_state=optimizer_state,
            params=updated_params,
            key=key,
            steps=state.steps + 1)
        return new_state, metrics

    def sample_data(training_state, replay_buffer, max_epi_len):
        key1, key2, key3 = jax.random.split(training_state.key, 3)
        epi_idx = jax.random.randint(
            key1, (int(batch_size_per_device * grad_updates_per_step),),
            minval=0,
            maxval=replay_buffer.data.shape[0])  # from (0, num_epi)
        context_idx = jax.random.randint(
            key2, (int(batch_size_per_device * grad_updates_per_step),),
            minval=0,
            maxval=max_epi_len)

        def dynamic_slice_context(carry, x):
            traj, c_idx = x
            return (), jax.lax.dynamic_slice(traj, (c_idx, 0), (sample_horizon_len, trans_dim))

        transitions = jnp.take(replay_buffer.data, epi_idx, axis=0, mode='clip')
        _, transitions = jax.lax.scan(dynamic_slice_context, (), (transitions, context_idx))
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

        one_epoch = lambda state, buffer: jax.lax.scan(
            partial(run_one_epoch, max_epi_len=max_epi_len),
            (state, buffer), None, length=num_updates_per_iter)

        if ensemble:
            # vmap in case of ensemble of models
            one_epoch = jax.vmap(one_epoch, in_axes=(0, None))

        (training_state, _), metrics = one_epoch(training_state, replay_buffer)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, replay_buffer, metrics, synchro

    run_training = jax.pmap(partial(run_training, max_epi_len=max_epi_len), axis_name='i')

    def one_train_iteration(training_state: TrainingState,
                            replay_buffer: ReplayBuffer,
                            i_train_iter: int,
                            total_updates: int):

        # optimization
        training_state, replay_buffer, training_metrics, synchro = run_training(
            training_state, replay_buffer)
        assert synchro[0], (current_step, training_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)

        # compute means
        mean_metrics = {k: np.mean(v) for k, v in training_metrics.items()}

        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        # build log string automatically
        log_str = "=" * 60 + "\n"
        log_str += f"time elapsed: {time_elapsed}\n"
        log_str += f"train iter: {i_train_iter}\n"
        log_str += f"num of updates: {total_updates}\n"
        for k, v in mean_metrics.items():
            log_str += f"{k}: {format(v, '.5f')}\n"

        print(log_str)

        # log to wandb
        wandb.log(mean_metrics)

        return total_updates, training_state, replay_buffer

        # # save model
        # _dynamics_params = jax.tree_util.tree_map(lambda x: x[0], training_state.params)

        # if i_train_iter % args.dynamics_save_iters == 0 or i_train_iter == max_train_iters - 1:
        #     save_current_model_path = save_model_path + f"_{total_updates}.pt"
        #     print("saving current model at: " + save_current_model_path)
        #     save_params(save_current_model_path, _dynamics_params)
        #     if args.Markov_dynamics:
        #         eval_dynamics(training_state.key, None, dynamics_model.apply, _dynamics_params, total_updates)

    return one_train_iteration