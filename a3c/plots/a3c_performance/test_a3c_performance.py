import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		data = list(reader)[0]

	return [float(x) for x in data]

def prep_test_results(filenames):
	data = []
	for filename in filenames:
		data.append(read_file(filename))
	data = np.array(data)
	data_mean = np.mean(data, axis=0)
	data_std = np.std(data, axis=0)
	#print(data.shape, data_mean.shape, data_std.shape)
	return data, data_mean, data_std

aloss1_0_closs0_5_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5'
aloss1_0_closs0_5, aloss1_0_closs0_5_mean, aloss1_0_closs0_5_std = prep_test_results(['{}/test_results_seed456.csv'.format(aloss1_0_closs0_5_basepath), 
										'{}/test_results_seed12.csv'.format(aloss1_0_closs0_5_basepath),
										'{}/test_results_seed985234.csv'.format(aloss1_0_closs0_5_basepath),
										'{}/test_results_seed123.csv'.format(aloss1_0_closs0_5_basepath),
										'{}/test_results_seed3202.csv'.format(aloss1_0_closs0_5_basepath)])


x = list(range(len(aloss1_0_closs0_5_mean)))

plt.plot(aloss1_0_closs0_5_mean, color='#2874A6', label='Reward at each episode')
plt.fill_between(x, aloss1_0_closs0_5_mean - aloss1_0_closs0_5_std, aloss1_0_closs0_5_mean + aloss1_0_closs0_5_std, color='#ABEBC6', alpha=0.5)
plt.legend(loc='upper left')
plt.ylim(bottom=0, top=230)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
#plt.show()
plt.savefig('test_a3c.png')



