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
			for k in ['observations', 'actions', 'next_observations', 'rewards', 'terminals', 'timeouts', 'infos/target_pos']: # 'infos/hand_qpos', 'infos/obj_pos', 'infos/palm_pos', 'infos/qpos', 'infos/qvel', 'infos/target_pos']:
				if (i != N-1) and not (done_bool or final_timestep):
					if k == 'next_observations':
						# data_['next_controlled_variables'].append(dataset[k][i+1][-6:])
						data_[k].append(np.concatenate((dataset['infos/qpos'][i+1], dataset['infos/qvel'][i+1])))
					elif k == 'observations':
						data_[k].append(np.concatenate((dataset['infos/qpos'][i], dataset['infos/qvel'][i])))
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

		with open(f'{name}-fullnextstate.pkl', 'wb') as f:
			pickle.dump(paths, f)
