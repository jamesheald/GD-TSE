import minari
import numpy as np

# Load dataset
minari_dataset = minari.load_dataset('D4RL/relocate/expert-v2')

# Find the maximum episode length
T_max = max(episode.observations.shape[0] for episode in minari_dataset)

# Accumulators
sum_trajectory = np.zeros((T_max, 3))
count_trajectory = np.zeros((T_max, 1))  # to count how many episodes contribute at each time step

# Accumulate values across episodes
for episode in minari_dataset:
    # obs = episode.observations[:, 30:33]  # shape (T, 3)
    obs = episode.observations[:, -3:]  # shape (T, 3)
    T = obs.shape[0]
    sum_trajectory[:T] += obs
    count_trajectory[:T] += 1

# Compute average, avoiding divide-by-zero
avg_trajectory = sum_trajectory / count_trajectory

print("Average trajectory shape:", avg_trajectory.shape)

import pickle
with open("/nfs/nhome/live/jheald/jax_dt/data/relocate-expert-v1.pkl", "rb") as f:
    data = pickle.load(f)

sum_trajectory_pik = np.zeros((T_max, 3))
count_trajectory_pik  = np.zeros((T_max, 1))  # to count how many episodes contribute at each time step
for episode in range(len(data)):
    obs = data[episode]['observations'][:, -3:]
    T = obs.shape[0]
    sum_trajectory_pik[:T] += obs
    count_trajectory_pik[:T] += 1

avg_trajectory_pik = sum_trajectory_pik / count_trajectory_pik

from matplotlib import pyplot as plt
plt.figure()
for i in range(3):
    plt.plot(avg_trajectory[:,i])
plt.savefig('minari_obs.png')
plt.close()

from matplotlib import pyplot as plt
plt.figure()
for i in range(3):
    plt.plot(avg_trajectory_pik[:,i])
plt.savefig('d4rl_obs.png')
plt.close()
