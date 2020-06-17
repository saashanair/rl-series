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


aloss1_0_closs0_5 = read_file('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5/train_results.csv')
mean100_aloss1_0_closs0_5 = calculate_mean_100_episodes(aloss1_0_closs0_5)

meanscore_x = [x+100 for x in range(len(mean100_aloss1_0_closs0_5))]

plt.plot(aloss1_0_closs0_5, color='#2874A6', label='Reward at each episode')
plt.plot(meanscore_x , mean100_aloss1_0_closs0_5, color='#CA6F1E', label='Mean score of last 100 episodes')
plt.legend(loc='upper left')
plt.ylim(top=230)
plt.xlabel('Episodes')
plt.ylabel('Average reward over the past 100 episodes')
plt.tight_layout()
#plt.show()
plt.savefig('train_a3c.png')
