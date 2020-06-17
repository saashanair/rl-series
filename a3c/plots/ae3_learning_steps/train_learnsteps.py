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

learnstep1 = read_file('CartPole-eps3000-learnsteps1-alosscoef-1.0-closs0.5/train_results.csv')
learnstep5 = read_file('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5/train_results.csv')
learnstep10 = read_file('CartPole-eps3000-learnsteps10-alosscoef-1.0-closs0.5/train_results.csv')
learnstep100 = read_file('CartPole-eps3000-learnsteps100-alosscoef-1.0-closs0.5/train_results.csv')
learnstep1000 = read_file('CartPole-eps3000-learnsteps1000-alosscoef-1.0-closs0.5/train_results.csv')

mean_learnstep1 = calculate_mean_100_episodes(learnstep1)
mean_learnstep5 = calculate_mean_100_episodes(learnstep5)
mean_learnstep10 = calculate_mean_100_episodes(learnstep10)
mean_learnstep100 = calculate_mean_100_episodes(learnstep100)
mean_learnstep1000 = calculate_mean_100_episodes(learnstep1000)

plt.plot(mean_learnstep1, color='#283747', label='Learn step = 1')
plt.plot(mean_learnstep5, color='#6C3483', label='Learn step = 5')
plt.plot(mean_learnstep10, color='#2874A6',  label='Learn step = 10')
plt.plot(mean_learnstep100, color='#239B56', label='Learn step = 100')
plt.plot(mean_learnstep1000, color='#CA6F1E', label='Learn step = 1000')
plt.ylim(top=290)
plt.legend(loc = 'upper left')
plt.xlabel('Episodes')
plt.ylabel('Average reward over the past 100 episodes')
plt.tight_layout()
#plt.show()
plt.savefig('train_learnsteps.png')
