import csv
import numpy as np
import matplotlib.pyplot as plt

def read_scores(filename):
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

nonshared_adam_scores = read_scores('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-nonsharedadam/train_results.csv')
shared_adam_scores = read_scores('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-sharedadam/train_results.csv')
nonshared_rmsprop_scores = read_scores('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-nonsharedrmsprop/train_results.csv')
shared_rmsprop_scores = read_scores('CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-sharedrmsprop/train_results.csv')


mean100_nonshared_adam = calculate_mean100score(nonshared_adam_scores)
mean100_shared_adam = calculate_mean100score(shared_adam_scores)
mean100_nonshared_rmsprop = calculate_mean100score(nonshared_rmsprop_scores)
mean100_shared_rmsprop = calculate_mean100score(shared_rmsprop_scores)
#print(len(mean100_nonshared), len(mean100_shared))

plt.plot(mean100_nonshared_rmsprop, color='#2874A6', label='Non-shared RMSprop')
plt.plot(mean100_shared_rmsprop, color='#CA6F1E', label='Shared RMSprop')
plt.plot(mean100_nonshared_adam, color='#6C3483', label='Non-shared Adam')
plt.plot(mean100_shared_adam, color='#239B56', label='Shared Adam')
plt.ylim(top=250)
plt.legend(loc = 'upper left')
plt.xlabel('Episodes')
plt.ylabel('Average reward over the past 100 episodes')
plt.tight_layout()
#plt.show()
plt.savefig('train_shared_vs_nonshared.png')
