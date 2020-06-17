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

nonshared_adam_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-nonsharedadam'
nonshared_adam, nonshared_adam_mean, nonshared_adam_std = prep_test_results(['{}/test_results_seed456.csv'.format(nonshared_adam_basepath), 
										'{}/test_results_seed12.csv'.format(nonshared_adam_basepath),
										'{}/test_results_seed985234.csv'.format(nonshared_adam_basepath),
										'{}/test_results_seed123.csv'.format(nonshared_adam_basepath),
										'{}/test_results_seed3202.csv'.format(nonshared_adam_basepath)])

shared_adam_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-sharedadam'
shared_adam, shared_adam_mean, shared_adam_std = prep_test_results(['{}/test_results_seed456.csv'.format(shared_adam_basepath), 
										'{}/test_results_seed12.csv'.format(shared_adam_basepath),
										'{}/test_results_seed985234.csv'.format(shared_adam_basepath),
										'{}/test_results_seed123.csv'.format(shared_adam_basepath),
										'{}/test_results_seed3202.csv'.format(shared_adam_basepath)])

nonshared_rmsprop_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-nonsharedrmsprop'
nonshared_rmsprop, nonshared_rmsprop_mean, nonshared_rmsprop_std = prep_test_results(['{}/test_results_seed456.csv'.format(nonshared_rmsprop_basepath), 
										'{}/test_results_seed12.csv'.format(nonshared_rmsprop_basepath),
										'{}/test_results_seed985234.csv'.format(nonshared_rmsprop_basepath),
										'{}/test_results_seed123.csv'.format(nonshared_rmsprop_basepath),
										'{}/test_results_seed3202.csv'.format(nonshared_rmsprop_basepath)])

shared_rmsprop_basepath = 'CartPole-eps3000-learnsteps5-alosscoef-1.0-closs0.5-sharedrmsprop'
shared_rmsprop, shared_rmsprop_mean, shared_rmsprop_std = prep_test_results(['{}/test_results_seed456.csv'.format(shared_rmsprop_basepath), 
										'{}/test_results_seed12.csv'.format(shared_rmsprop_basepath),
										'{}/test_results_seed985234.csv'.format(shared_rmsprop_basepath),
										'{}/test_results_seed123.csv'.format(shared_rmsprop_basepath),
										'{}/test_results_seed3202.csv'.format(shared_rmsprop_basepath)])


x = list(range(len(nonshared_adam_mean)))

plt.plot(nonshared_rmsprop_mean, color='#2874A6', label='Non-shared RMSprop')
plt.plot(shared_rmsprop_mean, color='#CA6F1E', label='Shared RMSprop')
plt.plot(nonshared_adam_mean, color='#6C3483', label='Non-shared Adam')
plt.plot(shared_adam_mean, color='#239B56', label='Shared Adam')

plt.fill_between(x, nonshared_rmsprop_mean - nonshared_rmsprop_std, nonshared_rmsprop_mean + nonshared_rmsprop_std, color='#AED6F1', alpha=0.5)
plt.fill_between(x, shared_adam_mean - shared_adam_std, shared_adam_mean + shared_adam_std, color='#ABEBC6', alpha=0.5)
plt.fill_between(x, shared_rmsprop_mean - shared_rmsprop_std, shared_rmsprop_mean + shared_rmsprop_std, color='#FAD7A0', alpha=0.5)
plt.fill_between(x, nonshared_adam_mean - nonshared_adam_std, nonshared_adam_mean + nonshared_adam_std, color='#D2B4DE', alpha=0.5) # fill_between requires an x set to work correctly

plt.legend(loc = 'upper left')
#plt.xlim(xmin=0)
plt.ylim(top=250)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
#plt.show()
plt.savefig('test_shared_vs_nonshared.png')
