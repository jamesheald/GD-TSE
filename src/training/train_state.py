import jax
from jax import numpy as jnp
import optax

from src.utils.utils import TrainingState, get_local_devices_to_use
from src.pmap.pmap import bcast_local_devices

def get_training_state(model, model_kwargs, key, args, ensemble_size):
    
    schedule_fn = optax.polynomial_schedule(
    init_value=args.lr * 1 / args.warmup_steps,
    end_value=args.lr,
    power=1,
    transition_steps=args.warmup_steps,
    transition_begin=0
    )

    optimizer = optax.chain(
        optax.clip(args.gradient_clipping),
        optax.adamw(learning_rate=schedule_fn, weight_decay=args.wt_decay),
    )

    # batch_size = 1
    key_params, key_dropout, subkey, key = jax.random.split(key, 4)
    key_dict = {'params': key_params,
                'dropout': key_dropout,
                'key': subkey}
    
    # initialise model(s)
    params_list = []
    for i in range(ensemble_size):
        params_list.append(model.init(key_dict, **model_kwargs))
        
        # generate new keys for the next member of the ensemble
        keys = list(key_dict.keys())
        new_keys = jax.random.split(key_dict['key'], len(keys))
        key_dict = {k: k_ for k, k_ in zip(keys, new_keys)}

    local_devices_to_use = get_local_devices_to_use(args)

    if ensemble_size > 1:
        params = jax.tree_util.tree_map(lambda *p: jnp.stack(p), *params_list)
        optimizer_state = jax.vmap(optimizer.init)(params)
        training_state_key = jax.random.split(key, args.n_dynamics_ensembles * local_devices_to_use).reshape(local_devices_to_use, args.n_dynamics_ensembles, -1)
        training_state_steps = jnp.zeros((local_devices_to_use, args.n_dynamics_ensembles))
    else:
        params = params_list[0]
        optimizer_state = optimizer.init(params)
        training_state_key = jnp.stack(jax.random.split(key, local_devices_to_use))
        training_state_steps = jnp.zeros((local_devices_to_use,))

    optimizer_state, params = bcast_local_devices((optimizer_state, params), local_devices_to_use)
    
    training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        key=training_state_key,
        steps=training_state_steps)

    return training_state, optimizer