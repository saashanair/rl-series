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

learnsteps1_basepath = 'CartPole-eps3000-learnsteps1-alosscoef-1.0-closs0.5'
learnsteps1, learnsteps1_mean, learnsteps1_std = prep_test_results(['{}/test_results_seed456.csv'.format(learnsteps1_basepath), 
										'{}/test_results_seed12.csv'.format(learnsteps1_basepath),
										'{}/test_results_seed985234.csv'.format(learnsteps1_basepath),
										'{}/test_results_seed123.csv'.format(learnsteps1_basepath),
										'{}/test_results_seed3202.csv'.format(learnsteps1_basepath)])

learnsteps5_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5'
learnsteps5, learnsteps5_mean, learnsteps5_std = prep_test_results(['{}/test_results_seed456.csv'.format(learnsteps5_basepath), 
										'{}/test_results_seed12.csv'.format(learnsteps5_basepath),
										'{}/test_results_seed985234.csv'.format(learnsteps5_basepath),
										'{}/test_results_seed123.csv'.format(learnsteps5_basepath),
										'{}/test_results_seed3202.csv'.format(learnsteps5_basepath)])

learnsteps10_basepath = 'CartPole-eps3000-learnsteps10-alosscoef-1.0-closs0.5'
learnsteps10, learnsteps10_mean, learnsteps10_std = prep_test_results(['{}/test_results_seed456.csv'.format(learnsteps10_basepath), 
										'{}/test_results_seed12.csv'.format(learnsteps10_basepath),
										'{}/test_results_seed985234.csv'.format(learnsteps10_basepath),
										'{}/test_results_seed123.csv'.format(learnsteps10_basepath),
										'{}/test_results_seed3202.csv'.format(learnsteps10_basepath)])

learnsteps100_basepath = 'CartPole-eps3000-learnsteps100-alosscoef-1.0-closs0.5'
learnsteps100, learnsteps100_mean, learnsteps100_std = prep_test_results(['{}/test_results_seed456.csv'.format(learnsteps100_basepath), 
										'{}/test_results_seed12.csv'.format(learnsteps100_basepath),
										'{}/test_results_seed985234.csv'.format(learnsteps100_basepath),
										'{}/test_results_seed123.csv'.format(learnsteps100_basepath),
										'{}/test_results_seed3202.csv'.format(learnsteps100_basepath)])

learnsteps1000_basepath = 'CartPole-eps3000-learnsteps1000-alosscoef-1.0-closs0.5'
learnsteps1000, learnsteps1000_mean, learnsteps1000_std = prep_test_results(['{}/test_results_seed456.csv'.format(learnsteps1000_basepath), 
										'{}/test_results_seed12.csv'.format(learnsteps1000_basepath),
										'{}/test_results_seed985234.csv'.format(learnsteps1000_basepath),
										'{}/test_results_seed123.csv'.format(learnsteps1000_basepath),
										'{}/test_results_seed3202.csv'.format(learnsteps1000_basepath)])

x = list(range(len(learnsteps5_mean)))

plt.plot(learnsteps1_mean, color='#283747', label='Learn step = 1')
plt.plot(learnsteps5_mean, color='#6C3483', label='Learn step = 5')
plt.plot(learnsteps10_mean, color='#2874A6', label='Learn step = 10')
plt.plot(learnsteps100_mean, color='#239B56', label='Learn step = 100')
plt.plot(learnsteps1000_mean, color='#CA6F1E', label='Learn step = 1000')

plt.fill_between(x, learnsteps1_mean - learnsteps1_std, learnsteps1_mean + learnsteps1_std, color='#AEB6BF', alpha=0.5)
plt.fill_between(x, learnsteps10_mean - learnsteps10_std, learnsteps10_mean + learnsteps10_std, color='#AED6F1', alpha=0.5)
plt.fill_between(x, learnsteps100_mean - learnsteps100_std, learnsteps100_mean + learnsteps100_std, color='#ABEBC6', alpha=0.5)
plt.fill_between(x, learnsteps1000_mean - learnsteps1000_std, learnsteps1000_mean + learnsteps1000_std, color='#FAD7A0', alpha=0.5)
plt.fill_between(x, learnsteps5_mean - learnsteps5_std, learnsteps5_mean + learnsteps5_std, color='#D2B4DE', alpha=0.5) # fill_between requires an x set to work correctly
plt.legend(loc = 'upper left')
#plt.xlim(xmin=0)
plt.ylim(top=290)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
#plt.show()
plt.savefig('test_learnsteps.png')
