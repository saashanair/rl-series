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

aloss0_1_closs1_0_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-0.1-closs1.0'
aloss0_1_closs1_0, aloss0_1_closs1_0_mean, aloss0_1_closs1_0_std = prep_test_results(['{}/test_results_seed456.csv'.format(aloss0_1_closs1_0_basepath), 
										'{}/test_results_seed12.csv'.format(aloss0_1_closs1_0_basepath),
										'{}/test_results_seed985234.csv'.format(aloss0_1_closs1_0_basepath),
										'{}/test_results_seed123.csv'.format(aloss0_1_closs1_0_basepath),
										'{}/test_results_seed3202.csv'.format(aloss0_1_closs1_0_basepath)])

aloss0_5_closs1_0_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-0.5-closs1.0'
aloss0_5_closs1_0, aloss0_5_closs1_0_mean, aloss0_5_closs1_0_std = prep_test_results(['{}/test_results_seed456.csv'.format(aloss0_5_closs1_0_basepath), 
										'{}/test_results_seed12.csv'.format(aloss0_5_closs1_0_basepath),
										'{}/test_results_seed985234.csv'.format(aloss0_5_closs1_0_basepath),
										'{}/test_results_seed123.csv'.format(aloss0_5_closs1_0_basepath),
										'{}/test_results_seed3202.csv'.format(aloss0_5_closs1_0_basepath)])

aloss1_0_closs1_0_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs1.0'
aloss1_0_closs1_0, aloss1_0_closs1_0_mean, aloss1_0_closs1_0_std = prep_test_results(['{}/test_results_seed456.csv'.format(aloss1_0_closs1_0_basepath), 
										'{}/test_results_seed12.csv'.format(aloss1_0_closs1_0_basepath),
										'{}/test_results_seed985234.csv'.format(aloss1_0_closs1_0_basepath),
										'{}/test_results_seed123.csv'.format(aloss1_0_closs1_0_basepath),
										'{}/test_results_seed3202.csv'.format(aloss1_0_closs1_0_basepath)])

aloss1_0_closs0_5_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5'
aloss1_0_closs0_5, aloss1_0_closs0_5_mean, aloss1_0_closs0_5_std = prep_test_results(['{}/test_results_seed456.csv'.format(aloss1_0_closs0_5_basepath), 
										'{}/test_results_seed12.csv'.format(aloss1_0_closs0_5_basepath),
										'{}/test_results_seed985234.csv'.format(aloss1_0_closs0_5_basepath),
										'{}/test_results_seed123.csv'.format(aloss1_0_closs0_5_basepath),
										'{}/test_results_seed3202.csv'.format(aloss1_0_closs0_5_basepath)])

aloss1_0_closs0_1_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.1'
aloss1_0_closs0_1, aloss1_0_closs0_1_mean, aloss1_0_closs0_1_std = prep_test_results(['{}/test_results_seed456.csv'.format(aloss1_0_closs0_1_basepath), 
										'{}/test_results_seed12.csv'.format(aloss1_0_closs0_1_basepath),
										'{}/test_results_seed985234.csv'.format(aloss1_0_closs0_1_basepath),
										'{}/test_results_seed123.csv'.format(aloss1_0_closs0_1_basepath),
										'{}/test_results_seed3202.csv'.format(aloss1_0_closs0_1_basepath)])


x = list(range(len(aloss1_0_closs1_0_mean)))


plt.plot(aloss1_0_closs1_0_mean, color='#CA6F1E', label='Actor coeff = 1.0')
plt.plot(aloss0_5_closs1_0_mean, color='#239B56', label='Actor coeff = 0.5')
plt.plot(aloss0_1_closs1_0_mean, color='#6C3483', label='Actor coeff = 0.1')
plt.fill_between(x, aloss0_1_closs1_0_mean - aloss0_1_closs1_0_std, aloss0_1_closs1_0_mean + aloss0_1_closs1_0_std, color='#D2B4DE', alpha=0.5) # fill_between requires an x set to work correctly
plt.fill_between(x, aloss0_5_closs1_0_mean - aloss0_5_closs1_0_std, aloss0_5_closs1_0_mean + aloss0_5_closs1_0_std, color='#ABEBC6', alpha=0.5)
plt.fill_between(x, aloss1_0_closs1_0_mean - aloss1_0_closs1_0_std, aloss1_0_closs1_0_mean + aloss1_0_closs1_0_std, color='#FAD7A0', alpha=0.5)
plt.legend(loc = 'upper left')
#plt.xlim(xmin=0)
plt.ylim(top=290)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
#plt.show()
plt.savefig('test_actor_coeff.png')

"""
plt.plot(aloss1_0_closs1_0_mean, color='#CA6F1E', label='Critic coeff = 1.0')
plt.plot(aloss1_0_closs0_1_mean, color='#6C3483', label='Critic coeff = 0.1')
plt.plot(aloss1_0_closs0_5_mean, color='#239B56', label='Critic coeff = 0.5')
plt.fill_between(x, aloss1_0_closs0_1_mean - aloss1_0_closs0_1_std, aloss1_0_closs0_1_mean + aloss1_0_closs0_1_std, color='#D2B4DE', alpha=0.5) # fill_between requires an x set to work correctly
plt.fill_between(x, aloss1_0_closs0_5_mean - aloss1_0_closs0_5_std, aloss1_0_closs0_5_mean + aloss1_0_closs0_5_std, color='#ABEBC6', alpha=0.5)
plt.fill_between(x, aloss1_0_closs1_0_mean - aloss1_0_closs1_0_std, aloss1_0_closs1_0_mean + aloss1_0_closs1_0_std, color='#FAD7A0', alpha=0.5)
plt.legend(loc = 'upper left')
#plt.xlim(xmin=0)
plt.ylim(top=290)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
#plt.show()
plt.savefig('test_critic_coeff.png')
"""


