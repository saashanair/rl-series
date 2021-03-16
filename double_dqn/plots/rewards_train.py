"""
Script to plot the reward history and moving average rewards of the agent during training
"""

import glob
import numpy as np 
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

env = 'LunarLander'
#path = '../pretrained/LunarLander-v2_epsmax1.0_epsmin0.01_epsdec0.995_batchsize64/'
path = '../results/LunarLander-v2-epsmax1.0_epsmin0.01_epsdec0.995_batchsize64_updfreq1000_memcap10000'
baseline_score = 200

#env = 'MountainCar'
#path = '../pretrained/MountainCar-v0_epsmax1.0_epsmin0.01_epsdec0.995_batchsize64'
#baseline_score = -110

#env = 'CartPole'
#path = '../pretrained/CartPole-v0_epsmax1.0_epsmin0.01_epsdec0.995_batchsize64'
#baseline_score = 195

def read_file(filename):
    """
    Function to read contents of the specifed pickle file

    Parameters
    ---
    filename: str
        Location of the file that needs to be read

    Returns
    ---
    scores: list of floats
        History of episodic scores received by the agent during training
    """
    with open(filename, 'rb') as f:
        scores = pickle.load(f)
        return scores

# read reward history from file
train_scores = read_file('{}/train_reward_history.pkl'.format(path))

# compute moving average of the 100 episodes
moving_avg = [np.mean(train_scores[i: i+100]) for i in range(len(train_scores)-100)]

# to mark baseline score for "completing" the environment
baseline_score = [baseline_score for _ in train_scores]

# since the x-axis for the baselines and raw scores differs from the one for the moving average score
x_axis = range(len(train_scores))
x_axis_moving_avg = [i+100 for i in range(len(moving_avg))] # starts plotting after 100th episode

# plots
plt.plot(x_axis, train_scores, color='#AED6F1', alpha=0.6)
plt.plot(x_axis, baseline_score, '--', color='r')
plt.plot(x_axis_moving_avg, moving_avg, color='#2874A6')

plt.xlim(left=0)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.tight_layout()

plt.savefig('{}_rewards_train_dqncodepost.png'.format(env))
#plt.show()

