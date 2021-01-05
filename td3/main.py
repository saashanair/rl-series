import argparse
import gym
import numpy as np
import os
import pickle
import torch

from td3 import TD3Agent


def fill_memory(env, td3_agent, epochs_fill_memory, exploration_noise):
	for _ in range(epochs_fill_memory):
		state = env.reset()
		done = False

		while not done:
			#action = td3_agent.select_action(state, exploration_noise=exploration_noise)
			action = env.action_space.sample() # do random action for warmup
			next_state, reward, done, _ = env.step(action)
			td3_agent.memory.store([state, action, next_state, reward, done])
			state = next_state


def train(env, td3_agent, epochs_train, epochs_fill_memory, batchsize, exploration_noise, results_folder):
	reward_history = []
	best_score = -np.inf 

	fill_memory(env, td3_agent, epochs_fill_memory, exploration_noise)
	print('Memory filled: ', len(td3_agent.memory))

	for iter_count in range(epochs_train):
		done = False
		state = env.reset()
		ep_reward = 0
		ep_steps = 0

		while not done:
			action = td3_agent.select_action(state, exploration_noise=exploration_noise)
			next_state, reward, done, _ = env.step(action)
			td3_agent.memory.store([state, action, next_state, reward, done])

			td3_agent.learn(current_iteration=iter_count, batchsize=batchsize)

			state = next_state
			ep_reward += reward
			ep_steps += 1

		reward_history.append(ep_reward)
		moving_avg_reward = np.mean(reward_history[-100:])

		print('Ep: {} | Ep reward: {} | Moving avg: {}'.format(iter_count, ep_reward, moving_avg_reward))

		if moving_avg_reward >= best_score:
			td3_agent.save(path='{}'.format(results_folder), model_name='best')
			best_score = moving_avg_reward

	with open('{}/reward_history.pkl'.format(results_folder), 'wb') as f:
		pickle.dump(reward_history, f)


def test():
	pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--num-train-episodes', type=int, default=3000)
	parser.add_argument('--num-memory-fill-episodes', type=int, default=10)
	parser.add_argument('--num_test_episodes', type=int, default=100)
	parser.add_argument('--memory-capacity', type=int, default=10000)
	parser.add_argument('--update-freq', type=int, default=2)
	parser.add_argument('--batchsize', type=int, default=64)
	parser.add_argument('--discount', type=float, default=0.99)
	parser.add_argument('--tau', type=float, default=0.005)
	parser.add_argument('--policy-noise-std', type=float, default=0.2)
	parser.add_argument('--policy-noise-clip', type=float, default=0.5)
	parser.add_argument('--exploration-noise', type=float, default=0.1)
	parser.add_argument('--actor-lr', type=float, default=1e-3)
	parser.add_argument('--critic-lr', type=float, default=1e-3)
	parser.add_argument('--results-folder', type=str)
	parser.add_argument('--env-name', type=str, default='MountainCarContinuous-v0')
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--cuda-device', type=str, default='cuda:0')
	parser.add_argument('--train-seed', type=int, default=12321, help='seed to use while training the model')
	parser.add_argument('--test-seed', type=int, nargs='+', default=[456, 12, 985234, 123, 3202], help='seeds to use while testing the model')
	args = parser.parse_args()

	device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")

	if args.results_folder is None:
		args.results_folder = 'results/{}_disc{}_actorlr{}_criticlr{}_tau{}_noisestd{}_noiseclip{}'.format(args.env_name, args.discount, args.actor_lr, args.critic_lr, args.tau, args.policy_noise_std, args.policy_noise_clip)

	if not os.path.exists('results'):
		os.mkdir('results')

	if not os.path.exists(args.results_folder):
		os.mkdir(args.results_folder)

	if args.train:
		os.environ['PYTHONHASHSEED']=str(args.train_seed)
		np.random.seed(args.train_seed)
		torch.manual_seed(args.train_seed)

		env = gym.make(args.env_name)
		env.seed(args.train_seed)

		td3_agent = TD3Agent(state_dim=env.observation_space.shape[0], 
								action_dim=env.action_space.shape[0], 
								min_action=env.action_space.low[0],  # clamp only works with numbers, not with arrays
								max_action=env.action_space.high[0], # clamp only works with numbers, not with arrays 
								device=device, 
								memory_capacity=args.memory_capacity, 
								discount=args.discount, 
								update_freq=args.update_freq, 
								tau=args.tau, 
								policy_noise_std=args.policy_noise_std, 
								policy_noise_clip=args.policy_noise_clip, 
								actor_lr=args.actor_lr, 
								critic_lr=args.critic_lr, 
								train_mode=args.train)

		train(env=env, 
				td3_agent=td3_agent, 
				epochs_train=args.num_train_episodes, 
				epochs_fill_memory=args.num_memory_fill_episodes, 
				batchsize=args.batchsize, 
				exploration_noise=args.exploration_noise,
				results_folder=args.results_folder)

		env.close()

	else:
		print('IN TEST!')
		test()










