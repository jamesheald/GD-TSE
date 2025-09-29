import flax
import jax
import optax
import pickle

import jax.numpy as jnp
import numpy as np

from typing import Any

def standardise_data(x, x_mean, x_std):
    return (x - x_mean) / x_std

def unstandardise_data(x, x_mean, x_std):
    return x * x_std + x_mean

def get_mean_and_log_std(x, min_log_std = -20., max_log_std = 2.):
    x_mean, x_log_std = jnp.split(x, 2, axis=-1)
    x_log_std = jnp.clip(x_log_std, min_log_std, max_log_std)
    return x_mean, x_log_std

def get_local_devices_to_use(args):

    max_devices_per_host = args.max_devices_per_host
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    return local_devices_to_use

def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

@flax.struct.dataclass
class ReplayBuffer:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray

@flax.struct.dataclass
class Transition:
    """Contains data for contextual-BC training step."""
    s_t: jnp.ndarray
    a_t: jnp.ndarray
    s_tp1: jnp.ndarray
    d_s: jnp.ndarray
    s_tm1: jnp.ndarray
    rtg_t: jnp.ndarray
    ts: jnp.ndarray
    mask_t: jnp.ndarray

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Any
    key: jnp.ndarray
    steps: jnp.ndarray

class File:
    """General purpose file resource."""
    def __init__(self, fileName: str, mode='r'):
        self.f = None
        if not self.f:
            self.f = open(fileName, mode)

    def __enter__(self):
        return self.f

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

def save_params(path: str, params: Any):
    """Saves parameters in Flax format."""
    with File(path, 'wb') as fout:
        fout.write(pickle.dumps(params))

def load_params(path: str) -> Any:
  with File(path, 'rb') as fin:
    buf = fin.read()
  return pickle.loads(buf)
