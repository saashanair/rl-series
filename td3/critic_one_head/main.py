"""
Script that contains the training and testing loops
"""

import argparse
import gym
import numpy as np
import os
import pickle
import torch

from td3 import TD3Agent


def fill_memory(env, td3_agent, epochs_fill_memory):
    """
    Function that performs a certain number of epochs of random interactions with the environment to populate the replay buffer

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    td3_agent: TD3Agent
        Agent to be trained
    epochs_fill_memory: int
        Number of epochs of interaction to be performed

    Returns
    ---
    none
    """

    for _ in range(epochs_fill_memory):
        state = env.reset()
        done = False

        while not done:
            action = env.action_space.sample() # do random action for warmup
            next_state, reward, done, _ = env.step(action)
            td3_agent.memory.store([state, action, next_state, reward, done]) # store the transition to memory
            state = next_state


def train(env, td3_agent, epochs_train, epochs_fill_memory, batchsize, exploration_noise, results_folder):
    """
    Function to train the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    td3_agent: TD3Agent
        Agent to be trained
    epochs_train: int
        Number of epochs/episodes of training to be performed
    epochs_fill_memory: int
        Number of epochs/episodes of interaction to be performed
    batchsize: int
        Number of transitions to be sampled from the replay buffer to perform an update
    exploration_noise: float
        Standard deviation, i.e. sigma, of the Gaussian noise applied to the agent to encourage exploration
    results_folder: str
        Location where models and other result files are saved

    Returns
    ---
    none
    """

    reward_history = [] # tracks the reward per episode
    best_score = -np.inf 

    fill_memory(env, td3_agent, epochs_fill_memory) # to populate the replay buffer before learning begins
    print('Memory filled: ', len(td3_agent.memory))

    total_steps = 0
    for ep_cnt in range(epochs_train):
        done = False
        state = env.reset()
        ep_reward = 0

        while not done:
            action = td3_agent.select_action(state, exploration_noise=exploration_noise) # generate noisy action
            next_state, reward, done, _ = env.step(action) # execute the action in the environment
            td3_agent.memory.store([state, action, next_state, reward, done]) # store the interaction in the replay buffer

            td3_agent.learn(current_iteration=total_steps, batchsize=batchsize) # update the networks

            state = next_state

            ep_reward += reward
            total_steps += 1

        reward_history.append(ep_reward)
        moving_avg_reward = np.mean(reward_history[-100:]) # tracks the mean of the last 100 scores

        print('Ep: {} | Ep reward: {} | Moving avg: {}'.format(ep_cnt, ep_reward, moving_avg_reward))

        if moving_avg_reward >= best_score:
            td3_agent.save(path='{}'.format(results_folder), model_name='best')
            best_score = moving_avg_reward

    # save the list of all episode rewards to a file
    with open('{}/reward_history.pkl'.format(results_folder), 'wb') as f:
        pickle.dump(reward_history, f)


def test(env, td3_agent, epochs_test, seed, results_folder):
    """
    Function to test the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    td3_agent: TD3Agent
        Agent to be trained
    epochs_test: int
        Number of epochs/episodes of testing to be performed
    seed: int
        Value of the seed used for testing
    results_folder: str
        Location where models and other result files are saved

    Returns
    ---
    none
    """

    reward_history = []
    for ep_cnt in range(epochs_test):
        state = env.reset()
        done = False

        ep_reward = 0

        while not done:
            action = td3_agent.select_action(state, exploration_noise=0.0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += 1
        print('Ep: {} | Ep reward: {}'.format(ep_cnt, ep_reward))
        reward_history.append(ep_reward)

    with open('{}/test_reward_history_{}.pkl'.format(results_folder, seed), 'wb') as f:
        pickle.dump(reward_history, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-train-episodes', type=int, default=3000)
    parser.add_argument('--num-memory-fill-episodes', type=int, default=10)
    parser.add_argument('--num-test-episodes', type=int, default=100)
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
    parser.add_argument('--env-name', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cuda-device', type=str, default='cuda:0')
    parser.add_argument('--train-seed', type=int, default=12321, help='seed to use while training the model')
    parser.add_argument('--test-seed', type=int, nargs='+', default=[456, 12, 985234, 123, 3202], help='seeds to use while testing the model')
    args = parser.parse_args()

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")

    ## training mode
    if args.train:

        if args.results_folder is None:
            args.results_folder = 'results/{}_disc{}_actorlr{}_criticlr{}_tau{}_noisestd{}_noiseclip{}'.format(args.env_name, args.discount, args.actor_lr, args.critic_lr, args.tau, args.policy_noise_std, args.policy_noise_clip)

        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(args.results_folder):
            os.mkdir(args.results_folder)

        os.environ['PYTHONHASHSEED']=str(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)

        env = gym.make(args.env_name)
        env.seed(args.train_seed)

        td3_agent = TD3Agent(state_dim=env.observation_space.shape[0], 
                                action_dim=env.action_space.shape[0],
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

    ## testing mode
    else:
        for seed in args.test_seed:
            print('=== TEST SEED: {} ==='.format(seed))
            os.environ['PYTHONHASHSEED']=str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            env = gym.make(args.env_name)
            env.seed(seed)

            td3_agent = TD3Agent(state_dim=env.observation_space.shape[0], 
                                    action_dim=env.action_space.shape[0],
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
            td3_agent.load(path=args.results_folder, model_name='best')

            test(env=env,
                td3_agent=td3_agent,
                epochs_test=args.num_test_episodes,
                seed=seed,
                results_folder=args.results_folder)

            env.close()


