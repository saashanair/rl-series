"""
Script to plot the reward history and moving average rewards of the agent during testing
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
        History of episodic scores received by the agent during testing
    """
    with open(filename, 'rb') as f:
        scores = pickle.load(f)
        return scores

# grab files for reward history across all seeds
test_files = glob.glob('{}/test_reward_history_*.pkl'.format(path))

# list of lists of reward histories
test_scores = []
for test_file in test_files:
    test_scores.append(read_file(test_file))

# compute mean and standard deviations per episode
test_scores_mean = np.mean(test_scores, axis=0)
test_scores_std = np.std(test_scores, axis=0)

# to mark baseline score for "completing" the environment
baseline_score = [baseline_score for _ in test_scores_mean]

x_axis = range(len(test_scores_mean))

plt.plot(x_axis, test_scores_mean, color='#2874A6')
plt.plot(x_axis, baseline_score, '--', color='r')

# to indicate deviation in rewards
plt.fill_between(x_axis, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='#AED6F1', alpha=0.5)

plt.xlim(left=0)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.tight_layout()

plt.savefig('{}_rewards_test_dqncodepost.png'.format(env))
#plt.show()