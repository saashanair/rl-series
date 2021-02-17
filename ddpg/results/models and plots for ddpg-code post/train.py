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

train_scores = read_file('{}/reward_history.pkl'.format(path))
moving_avg = [np.mean(train_scores[i: i+100]) for i in range(len(train_scores)-100)]
baseline_score = [200 for _ in train_scores]


x_axis = range(len(train_scores))
x_axis_moving_avg = [i+100 for i in range(len(moving_avg))]

plt.plot(x_axis, train_scores, color='#AED6F1', alpha=0.6) ## F9E79F
plt.plot(x_axis, baseline_score, '--', color='r')
plt.plot(x_axis_moving_avg, moving_avg, color='#2874A6') ## B7950B 

plt.xlim(left=0)
plt.ylim(bottom=-500, top=900)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.tight_layout()

plt.savefig('{}_train_clr1e3.png'.format(env))
#plt.savefig('{}_train_clr5e3.png'.format(env))
#plt.show()


