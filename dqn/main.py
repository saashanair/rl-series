"""
Main script where the DQN agent can be trained and tested. Allows to set various hyperparameter values to be used by the DQN agent.
"""

import os
import csv
import gym
import argparse
import numpy as np
import pickle

from dqn_agent import DQNAgent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fill_memory(env, dqn_agent, num_memory_fill_eps):
    for _ in range(num_memory_fill_eps):
        done = False
        state = env.reset()

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)


def train(env, dqn_agent, results_basepath, num_train_eps=1000, num_memory_fill_eps=10, update_frequency=1000, batchsize=32):
    fill_memory(env, dqn_agent, num_memory_fill_eps)
    print('Memory filled. Current capacity: ', len(dqn_agent.memory))
    
    reward_history = []
    epsilon_history = []
    loss_history = []

    step_cnt = 0
    best_score = -np.inf

    for ep_cnt in range(num_train_eps):
        done = False
        state = env.reset()

        ep_score = 0

        while not done:
            #env.render()
            action = dqn_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            #print('r: ', reward)

            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)
            loss = dqn_agent.learn(batchsize)
            loss_history.append(loss)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        epsilon_history.append(dqn_agent.epsilon)
        dqn_agent.update_epsilon()

        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-100:]) # get average of last 100 scores

        print('Ep: {}, Total Steps: {}, Ep: Score: {}, Avg score: {}; Epsilon: {}'.format(ep_cnt, step_cnt, ep_score, current_avg_score, epsilon_history[-1]))
        
        if current_avg_score >= best_score:
            dqn_agent.save_models('{}/policy_model_best'.format(results_basepath))
            best_score = current_avg_score

    dqn_agent.save_models('{}/policy_model_final'.format(results_basepath))

    with open('{}/train_reward_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(reward_history, f)

    with open('{}/train_epsilon_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(epsilon_history, f)

    with open('{}/train_loss_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(loss_history, f)


    #with open('{}/train_results.csv'.format(results_basepath), 'w') as train_results_file:
    #   results_writer = csv.writer(train_results_file, delimiter=',')
    #   results_writer.writerows((zip(scores, epsilon)))


    


def test(env, dqn_agent, num_test_eps, seed_value, results_basepath):
    step_cnt = 0
    reward_history = []

    for ep in range(num_test_eps):
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
        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))

    #with open('{}/test_results_seed{}.csv'.format(results_basepath, seed_value), 'w') as f:
    #   writer = csv.writer(f)
    #   writer.writerow((scores))

    with open('{}/test_reward_history_{}.pkl'.format(results_basepath, seed_value), 'wb') as f:
        pickle.dump(reward_history, f)
        

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-train-eps', type=int, default=1000, help='specify the max episodes to train for (counts even the period of memory initialisation)')
    parser.add_argument('--num-test-eps', type=int, default=100, help='specify the max episodes to test for')
    parser.add_argument('--num-memory-fill-eps', type=int, default=10, help='number of timesteps after which learning should start (used to initialise the memory)')
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

    if args.train:

        os.environ['PYTHONHASHSEED']=str(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)

        env = gym.make(args.env_name)
        env.seed(args.train_seed)
        env.action_space.np_random.seed(args.train_seed)

        if args.results_folder is None:
            args.results_folder = "results/{}_epsmax{}_epsmin{}_epsdec{}_batchsize{}".format(args.env_name, args.eps_max, args.eps_min, args.eps_decay, args.batchsize)

        os.makedirs(args.results_folder, exist_ok=True)

        dqn_agent = DQNAgent(device, 
                                env.observation_space.shape[0], 
                                env.action_space.n, 
                                discount=args.discount, 
                                eps_max=args.eps_max, 
                                eps_min=args.eps_min, 
                                eps_decay=args.eps_decay,
                                memory_capacity=args.memory_capacity)

        train(env=env, 
                dqn_agent=dqn_agent, 
                results_basepath=args.results_folder, 
                num_train_eps=args.num_train_eps, 
                max_memory_fill_eps=args.num_memory_fill_eps, 
                update_frequency=args.update_frequency,
                batchsize=args.batchsize)

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
            env.action_space.np_random.seed(seed_value)

            dqn_agent = DQNAgent(device, 
                                env.observation_space.shape[0], 
                                env.action_space.n, 
                                discount=args.discount, 
                                eps_max=0.0, # epsilon values should be zero to ensure no exploration in testing mode
                                eps_min=0.0, 
                                eps_decay=0.0)
            dqn_agent.load_models('{}/policy_model_best'.format(args.results_folder))

            test(env, dqn_agent, args.num_test_eps, seed_value, args.results_folder)

            env.close()
            cntr += 1



