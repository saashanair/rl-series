import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		data = list(reader)[0]

	return [float(x) for x in data]

def calculate_mean_100_episodes(scores):
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


aloss0_1_closs1_0 = read_file('CartPole-eps3000-learnsteps5-alosscoef-0.1-closs1.0/train_results.csv')
aloss0_5_closs1_0 = read_file('CartPole-eps3000-learnsteps5-alosscoef-0.5-closs1.0/train_results.csv')
aloss1_0_closs1_0 = read_file('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs1.0/train_results.csv')
aloss1_0_closs0_5 = read_file('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5/train_results.csv')
aloss1_0_closs0_1 = read_file('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.1/train_results.csv')

mean100_aloss0_1_closs1_0 = calculate_mean_100_episodes(aloss0_1_closs1_0)
mean100_aloss0_5_closs1_0 = calculate_mean_100_episodes(aloss0_5_closs1_0)
mean100_aloss1_0_closs1_0 = calculate_mean_100_episodes(aloss1_0_closs1_0)
mean100_aloss1_0_closs0_5 = calculate_mean_100_episodes(aloss1_0_closs0_5)
mean100_aloss1_0_closs0_1 = calculate_mean_100_episodes(aloss1_0_closs0_1)


plt.plot(mean100_aloss1_0_closs1_0, color='#CA6F1E', label='Actor coeff = 1.0')
plt.plot(mean100_aloss0_5_closs1_0, color='#239B56', label='Actor coeff = 0.5')
plt.plot(mean100_aloss0_1_closs1_0, color='#6C3483', label='Actor coeff = 0.1')

plt.ylim(top=290)
plt.legend(loc = 'upper left')
#plt.xlim(xmin=0)
#plt.ylim(ymin=-235, ymax=-65)
plt.xlabel('Episodes')
plt.ylabel('Average reward over the past 100 episodes')
plt.tight_layout()
#plt.show()
plt.savefig('train_actor_coeff.png')
#print(aloss0_1_closs1_0)
#print()

"""
plt.plot(mean100_aloss1_0_closs1_0, color='#CA6F1E', label='Critic coeff = 1.0')
plt.plot(mean100_aloss1_0_closs0_1, color='#6C3483', label='Critic coeff = 0.1')
plt.plot(mean100_aloss1_0_closs0_5, color='#239B56', label='Critic coeff = 0.5')
plt.legend(loc = 'upper left')

plt.ylim(top=290)
plt.xlabel('Episodes')
plt.ylabel('Average reward over the past 100 episodes')
plt.tight_layout()
#plt.show()
plt.savefig('train_critic_coeff.png')
"""
