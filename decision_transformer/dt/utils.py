import flax
import jax
import optax
import pickle
import random
import time

import jax.numpy as jnp
import numpy as np

from typing import Any
from decision_transformer.d4rl_infos import D4RL_DATASET_STATS


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


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
