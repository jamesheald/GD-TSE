import gym
import d4rl
import numpy as np

import collections
import pickle

datasets = []

for env_name in ['relocate']:
	for dataset_type in ['expert']:
		name = f'{env_name}-{dataset_type}-v1'
		env = gym.make(name)
		dataset = env.get_dataset()

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset:
			use_timeouts = True

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == 1000-1)
			for k in ['observations', 'actions', 'rewards', 'terminals', 'timeouts', 'infos/qvel']: # 'infos/hand_qpos', 'infos/obj_pos', 'infos/palm_pos', 'infos/qpos', 'infos/target_pos']:
				if k == 'infos/qvel':
					if i != N-1:
						if done_bool or final_timestep:
							# next controlled variables doesn't exist if last timestep (use nan)
							data_['next_controlled_variables'].append(np.full(dataset[k][i+1][-6:].shape, np.nan))
						else:
							data_['next_controlled_variables'].append(dataset[k][i+1][-6:])
					else:
						data_['next_controlled_variables'].append(np.full(dataset[k][i][-6:].shape, np.nan))
				else:
					data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(f'{name}.pkl', 'wb') as f:
			pickle.dump(paths, f)
