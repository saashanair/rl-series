import csv
import numpy as np
import matplotlib.pyplot as plt

def read_dqn_scores(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		data = list(reader)
	score = [float(x[0]) for x in data]
	return score

def read_a3c_scores(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		data = list(reader)[0]
	return [float(x) for x in data]

def calculate_mean100score(scores):

	start_idx = 0
	end_idx = 100
	mean100scores = []
	while end_idx < len(scores):
		last_100_scores = scores[start_idx:end_idx]
		#print('YO: ', len(last_100_scores), start_idx, end_idx)
		mean100scores.append(np.mean(last_100_scores))
		start_idx += 1
		end_idx += 1
	return mean100scores

a3c_scores = read_a3c_scores('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5/train_results.csv')
dqn_scores = read_dqn_scores('CartPole-v0_epsmax1.0_epsmin0.01_epsdec0.99_disc0.99_dqn/train_results.csv')

mean100_dqn = calculate_mean100score(dqn_scores)
mean100_a3c = calculate_mean100score(a3c_scores)

plt.plot(mean100_dqn, color='#239B56', label='DQN')
plt.plot(mean100_a3c, color='#6C3483', label='A3C')
plt.ylim(top=240)
plt.legend(loc = 'upper left')
plt.xlabel('Episodes')
plt.ylabel('Average reward over the past 100 episodes')
plt.tight_layout()
#plt.show()
plt.savefig('train_a3c_vs_dqn.png')