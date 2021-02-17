import glob
import matplotlib.pyplot as plt
import numpy as np 
import pickle

env = 'LunarLander'
#env = 'MountainCar' 

path = 'LunarLanderContinuous-v2_disc0.99_actorlr0.0001_criticlr0.001_tau0.005'
#path = 'LunarLanderContinuous-v2_disc0.99_actorlr0.0001_criticlr0.005_tau0.005'

def read_file(filename):
    with open(filename, 'rb') as f:
        scores = pickle.load(f)
        return scores

test_files = glob.glob('{}/test_reward_history_*.pkl'.format(path))

test_scores = []
for test_file in test_files:
	test_scores.append(read_file(test_file))

test_scores_mean = np.mean(test_scores, axis=0)
test_scores_std = np.std(test_scores, axis=0)

baseline_score = [200 for _ in test_scores_mean]

x_axis = range(len(test_scores_mean))

plt.plot(x_axis, test_scores_mean, color='#2874A6')
plt.plot(x_axis, baseline_score, '--', color='r')

plt.fill_between(x_axis, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='#AED6F1', alpha=0.5)

plt.xlim(left=0)
plt.ylim(bottom=-500, top=900)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.tight_layout()

plt.savefig('{}_test_clr1e3.png'.format(env))
#plt.savefig('{}_test_clr5e3.png'.format(env))
#plt.show()