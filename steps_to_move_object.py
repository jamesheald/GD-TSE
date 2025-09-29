import minari
import numpy as np

# Load the dataset
dataset = minari.load_dataset('D4RL/relocate/expert-v2')

# This will store the number of steps before the object moves in each episode
steps_until_object_moves = []

threshold = 5e-3

for episode in dataset.iterate_episodes():
    obs = episode.observations  # shape: (T, 39)
    initial_object_pos = obs[0, -3:]  # last 3 elements of first observation
    moved = False

    for t in range(1, len(obs)):
        current_object_pos = obs[t, -3:]
        if np.linalg.norm(current_object_pos - initial_object_pos) > threshold:
            steps_until_object_moves.append(t)
            moved = True
            break

    if not moved:
        steps_until_object_moves.append(len(obs))  # object never moved

# Print or use the result
print("Steps until object moved per episode:", steps_until_object_moves)

from matplotlib import pyplot as plt
plt.hist(steps_until_object_moves) # , bins=20, edgecolor='black'
filename = f"steps_until_object_moves_thresh_{threshold:.3f}.png"
plt.savefig(filename)

breakpoint()