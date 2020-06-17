import os
import gym
import csv
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
import torch.distributions as dist

from model import A3CNet
from worker import Worker
from shared_adam import SharedAdam
from shared_rmsprop import SharedRMSprop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPS = 3000
#MAX_EPS = 10
NUM_LEARN_STEPS = 1
env_name = 'CartPole-v0'
#env_name = 'LunarLander-v2'

#results_folder = "{}_epsmax{}_epsmin{}_epsdec{}_disc{}".format(args.env_name, args.eps_max, args.eps_min, args.eps_decay, args.discount)
results_folder = 'CartPole-eps3000-learnsteps1-alosscoef-1.0-closs0.5'

if __name__ == '__main__':

	if not os.path.exists(results_folder):
		os.mkdir(results_folder)

	train_seed = 12321
	os.environ['PYTHONHASHSEED']=str(train_seed)
	np.random.seed(train_seed)
	torch.manual_seed(train_seed)

	#global_env = gym.make('LunarLander-v2')
	global_env = gym.make(env_name)
	global_env.seed(train_seed)

	global_network = A3CNet(global_env.observation_space.shape[0], global_env.action_space.n)
	global_network.share_memory()

	#global_optimizer = optim.RMSprop(global_network.parameters(), lr=1e-6)
	#global_optimizer = optim.Adam(global_network.parameters(), lr=1e-5)
	global_optimizer = None
	#global_optimizer = SharedAdam(global_network.parameters(), lr=1e-5) #, betas=(0.92, 0.999))
	#global_optimizer = SharedRMSprop(global_network.parameters(), lr=1e-5)


	global_ep_count = mp.Value('i', 0)
	res_queue = mp.Queue()

	num_workers = mp.cpu_count()
	#num_workers = 8

	workers = [Worker(global_network,
					global_optimizer,
					global_ep_count,
					res_queue,
					idx,
					device,
					env_name,
					train_seed=train_seed,
					max_eps=MAX_EPS,
					num_learn_steps=NUM_LEARN_STEPS,
					lr=1e-5,
					actor_loss_coeff=1.0,
					critic_loss_coeff=0.5,
					train_mode=True) for idx in range(num_workers)]
	[w.start() for w in workers]

	res = [] # record episode reward to plot
	while True:
		r = res_queue.get()
		if r is not None:
			res.append(r)
			#print('GN: ', global_network.state_dict()['value.weight'][0])
		else:
			#print('GN before break: ', global_network.state_dict()['value.weight'][0])
			break
	[w.join() for w in workers]
	global_network.save_model("{}/global_model".format(results_folder))

	print(type(res))
	with open('{}/train_results.csv'.format(results_folder), 'w') as train_results_file:
		results_writer = csv.writer(train_results_file, delimiter=',')
		results_writer.writerow(res)

	"""
	import matplotlib.pyplot as plt
	plt.plot(res)
	plt.ylabel('Moving average ep reward')
	plt.xlabel('Step')
	plt.show()
	"""
	test_seeds = [456, 12, 985234, 123, 3202]

	#print('GN Out: ', global_network.state_dict()['value.weight'][0])
	for seed_value in test_seeds:
		print('Testing with seed {}...'.format(seed_value))
		test_env = gym.make(env_name)
		test_env.seed(seed_value)

		scores = []

		for ep_cnt in range(100):
			state = test_env.reset()
			done = False
			total_reward = 0

			while not done:
				action = global_network.select_action(state, device)
				next_state, reward, done, _ = test_env.step(action)
				state = next_state
				total_reward += reward
			scores.append(total_reward)
			print('Ep: {}; Total reward: {}'.format(ep_cnt, total_reward))

		with open('{}/test_results_seed{}.csv'.format(results_folder, seed_value), 'w') as f:
			writer = csv.writer(f)
			writer.writerow(scores)
