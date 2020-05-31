import os
import gym
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np

from model import A3CNet
from worker import Worker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPS = 3000
#MAX_EPS = 100
NUM_LEARN_STEPS = 5
env_name = 'CartPole-v0'
#env_name = 'LunarLander-v2'

if __name__ == '__main__':

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

	print(res)

	import matplotlib.pyplot as plt
	plt.plot(res)
	plt.ylabel('Moving average ep reward')
	plt.xlabel('Step')
	plt.show()

	test_env = gym.make(env_name)
	test_env.seed(123)

	#print('GN Out: ', global_network.state_dict()['value.weight'][0])
	for ep_cnt in range(100):
		state = test_env.reset()
		done = False
		total_reward = 0

		while not done:
			state = torch.tensor([state], dtype=torch.float32).to(device)
			with torch.no_grad():
				action_prob, value = global_network.forward(state)
			action = torch.argmax(action_prob).item()
			next_state, reward, done, _ = test_env.step(action)
			state = next_state
			total_reward += reward
		print('Ep: {}; Total reward: {}'.format(ep_cnt, total_reward))
