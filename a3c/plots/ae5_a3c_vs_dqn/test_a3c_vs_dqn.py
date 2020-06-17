import csv
import numpy as np
import matplotlib.pyplot as plt

def read_dqn_scores(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		data = list(reader)[0]
	score = [float(x) for x in data]
	return score

def read_a3c_scores(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		data = list(reader)[0]
	return [float(x) for x in data]

def prep_test_results(filenames, algo):
	data = []
	for filename in filenames:
		if algo == 'a3c':
			data.append(read_a3c_scores(filename))
		else:
			data.append(read_dqn_scores(filename))
	data = np.array(data)
	data_mean = np.mean(data, axis=0)
	data_std = np.std(data, axis=0)
	print(data.shape, data_mean.shape, data_std.shape)
	return data, data_mean, data_std

a3c_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5'
score_a3c, a3c_mean, a3c_std = prep_test_results(['{}/test_results_seed456.csv'.format(a3c_basepath), 
										'{}/test_results_seed12.csv'.format(a3c_basepath),
										'{}/test_results_seed985234.csv'.format(a3c_basepath),
										'{}/test_results_seed123.csv'.format(a3c_basepath),
										'{}/test_results_seed3202.csv'.format(a3c_basepath)], 'a3c')

dqn_basepath = 'CartPole-v0_epsmax1.0_epsmin0.01_epsdec0.99_disc0.99_dqn'
score_dqn, dqn_mean, dqn_std = prep_test_results(['{}/test_results_seed456.csv'.format(dqn_basepath), 
										'{}/test_results_seed12.csv'.format(dqn_basepath),
										'{}/test_results_seed985234.csv'.format(dqn_basepath),
										'{}/test_results_seed123.csv'.format(dqn_basepath),
										'{}/test_results_seed3202.csv'.format(dqn_basepath)], 'dqn')

print(dqn_mean)

x = list(range(len(dqn_mean)))
plt.plot(dqn_mean, color='#239B56', label='DQN')
plt.plot(a3c_mean, color='#6C3483', label='A3C')
plt.fill_between(x, dqn_mean - dqn_std, dqn_mean + dqn_std, color='#ABEBC6', alpha=0.5)
plt.fill_between(x, a3c_mean - a3c_std, a3c_mean + a3c_std, color='#D2B4DE', alpha=0.5) # fill_between requires an x set to work correctly
plt.legend(loc = 'upper left')
#plt.xlim(xmin=0)
plt.ylim(top=240)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
#plt.show()
plt.savefig('test_a3c_vs_dqn.png')

