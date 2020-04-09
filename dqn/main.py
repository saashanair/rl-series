"""
Main script where the DQN agent can be trained and tested. Allows to set various hyperparameter values to be used by the DQN agent.
"""

import os
import csv
import gym
import argparse
import numpy as np

from dqn_agent import DQNAgent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dqn_agent, max_train_episodes, learn_start, update_frequency, results_basepath):
	scores = []
	epsilon = []
	step_cnt = 0
	best_score = -np.inf

	print('Initializing Experience Replay Memory ...')
	for _ in range(max_train_episodes):
		done = False
		state = env.reset()

		ep_score = 0
		ep_steps = 0

		while not done:
			env.render()
			action = dqn_agent.select_action(state)
			next_state, reward, done, info = env.step(action)
			ep_score += reward
			dqn_agent.memory.store([state, action, next_state, reward, done])

			if step_cnt == learn_start:
				print('Init complete. Starting learning now ...')
			if step_cnt > learn_start:
				dqn_agent.learn(batch_size=64)
				if step_cnt % update_frequency == 0:
					dqn_agent.update_target_net()
			state = next_state
			step_cnt += 1
			ep_steps += 1

		if step_cnt > learn_start:
			epsilon.append(dqn_agent.epsilon)
			dqn_agent.update_epsilon()
			scores.append(ep_score)
			current_avg_score = np.mean(scores[-101:-1])
			ep = len(scores)

			print('Ep: {}, Steps: {}, Score: {}, Avg score: {}; {}'.format(ep, step_cnt, ep_score, current_avg_score, dqn_agent.epsilon))
		
			if current_avg_score >= best_score:
				dqn_agent.save_models('{}/policy_model_best'.format(results_basepath), '{}/target_model_best'.format(results_basepath))
				best_score = current_avg_score

	dqn_agent.save_models('{}/policy_model_final'.format(results_basepath), '{}/target_model_final'.format(results_basepath))

	with open('{}/train_results.csv'.format(results_basepath), 'w') as train_results_file:
		results_writer = csv.writer(train_results_file, delimiter=',')
		results_writer.writerows((zip(scores, epsilon)))
	pass


def test(dqn_agent, max_test_episodes, seed_value, results_basepath):
	step_cnt = 0
	scores = []

	for ep in range(max_test_episodes):
		score = 0
		done = False
		state = env.reset()
		while not done: # and step_cnt<1000:
			env.render()
			action = dqn_agent.select_action(state)
			next_state, reward, done, _ = env.step(action)
			score += reward
			state = next_state
			step_cnt += 1
		scores.append(score)
		print('Ep: {}, Score: {}'.format(ep, score))

	with open('{}/test_results_seed{}.csv'.format(results_basepath, seed_value), 'w') as f:
		writer = csv.writer(f)
		writer.writerow((scores))
		

if __name__ ==  '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--max-train-episodes', type=int, default=1050, help='specify the max episodes to train for (counts even the period of memory initialisation)')
	parser.add_argument('--max-test-episodes', type=int, default=100, help='specify the max episodes to test for')
	parser.add_argument('--learn-start', type=int, default=3000, help='number of timesteps after which learning should start (used to initialise the memory)')
	parser.add_argument('--update-frequency', type=int, default=1000, help='how frequently should the target network by updated')
	parser.add_argument('--train-seed', type=int, default=12321, help='seed to use while training the model')
	parser.add_argument('--test-seed', type=int, nargs='+', default=[456, 12, 985234, 123, 3202], help='seeds to use while testing the model')
	parser.add_argument('--discount', type=float, default=0.99, help='discounting value to determine how far-sighted the agent should be')
	parser.add_argument('--eps-max', type=float, default=1.0, help='max value for epsilon')
	parser.add_argument('--eps-min', type=float, default=0.01, help='min value for epsilon')
	parser.add_argument('--eps-decay', type=float, default=0.99, help='amount by which to decay the epsilon value for annealing strategy')
	parser.add_argument('--batchsize', type=int, default=64, help='number of samples to draw from memory for learning')
	parser.add_argument('--memory-capacity', type=int, default=5000, help='define the capacity of the replay memory')
	parser.add_argument('--results-folder', type=str, help='folder where the models and results of the current run must by stored')
	parser.add_argument('--env-name', type=str, default='MountainCar-v0', help='environment in which to train the agent')
	parser.add_argument('--train', action='store_true', help='train the agent')
	args = parser.parse_args()

	if args.results_folder is None:
		args.results_folder = "{}_epsmax{}_epsmin{}_epsdec{}_disc{}".format(args.env_name, args.eps_max, args.eps_min, args.eps_decay, args.discount)

	if not os.path.exists(args.results_folder):
		os.mkdir(args.results_folder)

	if args.train:
		os.environ['PYTHONHASHSEED']=str(args.train_seed)
		np.random.seed(args.train_seed)
		torch.manual_seed(args.train_seed)

		env = gym.make(args.env_name)
		env.seed(args.train_seed)

		dqn_agent = DQNAgent(device, 
								env.observation_space.shape[0], 
								env.action_space.n, 
								discount=args.discount, 
								eps_max=args.eps_max, 
								eps_min=args.eps_min, 
								eps_decay=args.eps_decay,
								memory_capacity=args.memory_capacity)

		train(dqn_agent, args.max_train_episodes, args.learn_start, args.update_frequency, args.results_folder)
		env.close()
	
	else:
		cntr = 1
		for seed_value in args.test_seed:
			print("Testing {}/{}, seed = {}".format(cntr, len(args.test_seed), seed_value))
			os.environ['PYTHONHASHSEED']=str(seed_value)
			np.random.seed(seed_value)
			torch.manual_seed(seed_value)

			env = gym.make(args.env_name)
			env.seed(seed_value)

			dqn_agent = DQNAgent(device, 
								env.observation_space.shape[0], 
								env.action_space.n, 
								discount=args.discount, 
								eps_max=0.0, # epsilon values should be zero to ensure no exploration in testing mode
								eps_min=0.0, 
								eps_decay=0.0)
			dqn_agent.load_models('{}/policy_model_best'.format(args.results_folder), '{}/target_model_best'.format(args.results_folder))

			test(dqn_agent, args.max_test_episodes, seed_value, args.results_folder)

			env.close()
			cntr += 1



