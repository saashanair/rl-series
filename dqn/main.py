import csv
import gym
import numpy as np
#from collections import deque

from dqn_agent import DQNAgent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MAX_TIMESTEPS = 10000
#MAX_EPISODES = 1040 ## values being used for MountainCar exploration test
MAX_EPISODES = 3000
TIMESTEPS_TO_START_LEARNING = 3000
TARGET_UPDATE = 1000
POLICY_NET_FILENAME='policy_net_model'
TARGET_NET_FILENAME='target_net_model'

train = True

#env = gym.make('CartPole-v0')
#lunar_env = gym.make('MountainCar-v0')
lunar_env = gym.make('LunarLander-v2')  # need to run 'pip install gym[box2d]' for this env
env = gym.wrappers.Monitor(lunar_env, './videos/train/', force=True)
dqn_agent = DQNAgent(device, env.observation_space.shape[0], env.action_space.n, discount=0.99, eps_max=0.1, eps_min=0.1)
print(dqn_agent.policy_net)

if train:
	scores = []
	epsilon = []
	#last_100_scores = deque([], maxlen=100)
	step_cnt = 0
	#ep = 0
	best_score = -np.inf


	print('Initializing Experience Replay Memory ...')
	for _ in range(MAX_EPISODES):
	#while step_cnt < MAX_TIMESTEPS:
		done = False
		state = env.reset()

		ep_score = 0
		ep_steps = 0

		while not done:
			if ep_steps >= 1000:
				print(ep_steps)
				break
			
			env.render()
			action = dqn_agent.select_action(state, explore=True)
			next_state, reward, done, info = env.step(action)
			ep_score += reward
			dqn_agent.memory.store([state, action, next_state, reward, done])
			if step_cnt == TIMESTEPS_TO_START_LEARNING:
				print('Init complete. Starting learning now ...')
			if step_cnt > TIMESTEPS_TO_START_LEARNING:
				dqn_agent.learn(batch_size=64)
				if step_cnt % TARGET_UPDATE == 0:
					dqn_agent.update_target_net()
			state = next_state
			step_cnt += 1
			ep_steps += 1
		#ep += 1

		if step_cnt > TIMESTEPS_TO_START_LEARNING:
			epsilon.append(dqn_agent.epsilon)
			dqn_agent.update_epsilon()
			#last_100_scores.append(ep_score)
			scores.append(ep_score)
			#current_avg_score = np.mean(last_100_scores)
			current_avg_score = np.mean(scores[-101:-1])
			ep = len(scores)

			print('Ep: {}, Steps: {}, Score: {}, Avg score: {}; {}'.format(ep, step_cnt, ep_score, current_avg_score, dqn_agent.epsilon))
		
			if current_avg_score >= best_score:
				dqn_agent.save_models(POLICY_NET_FILENAME, TARGET_NET_FILENAME)
				best_score = current_avg_score
		
	#print(dqn_agent.policy_net.state_dict().get('dense1.weight'))
	dqn_agent.save_models('policy_final', 'target_final')

	with open('results.csv', 'w') as results_file:
		results_writer = csv.writer(results_file, delimiter=',')
		results_writer.writerows((zip(scores, epsilon)))
else:
	#dqn_agent.load_models('policy_final', 'target_final')
	dqn_agent.load_models(POLICY_NET_FILENAME, TARGET_NET_FILENAME)
	dqn_agent.update_target_net()
	#print(dqn_agent.target_net.state_dict().get('dense1.weight'))
	step_cnt = 0
	scores = []
	for ep in range(100):
		score = 0
		done = False
		state = env.reset()
		while not done: # and step_cnt<1000:
			env.render()
			action = dqn_agent.select_action(state, explore=False)
			next_state, reward, done, _ = env.step(action)
			score += reward
			state = next_state
			step_cnt += 1
		scores.append(score)
		print('Ep: {}, Score: {}'.format(ep, score))

	with open('test_results_standard.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow((scores))
		#for s in score:
		#	writer.writerow(s)


env.close()
lunar_env.close()