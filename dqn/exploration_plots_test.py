## color picker -- https://htmlcolorcodes.com

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
	print(data.shape, data_mean.shape, data_std.shape)
	return data, data_mean, data_std

score_noexplore_basepath = 'mountaincar_results/noexplore'
score_noexplore, score_noexplore_mean, score_noexplore_std = prep_test_results(['{}/test_results_seed456.csv'.format(score_noexplore_basepath), 
										'{}/test_results_seed12.csv'.format(score_noexplore_basepath),
										'{}/test_results_seed985234.csv'.format(score_noexplore_basepath),
										'{}/test_results_seed123.csv'.format(score_noexplore_basepath),
										'{}/test_results_seed3202.csv'.format(score_noexplore_basepath)])
#print(score_noexplore.shape, score_noexplore_mean.shape, score_noexplore_std.shape)

score_epsdecay99_basepath = 'mountaincar_results/epsannealing_decay0.99'
score_epsdecay99, score_epsdecay99_mean, score_epsdecay99_std = prep_test_results(['{}/test_results_seed456.csv'.format(score_epsdecay99_basepath), 
										'{}/test_results_seed12.csv'.format(score_epsdecay99_basepath),
										'{}/test_results_seed985234.csv'.format(score_epsdecay99_basepath),
										'{}/test_results_seed123.csv'.format(score_epsdecay99_basepath),
										'{}/test_results_seed3202.csv'.format(score_epsdecay99_basepath)])

#for m, s, l, h in zip( score_noexplore_mean, score_noexplore_std, score_noexplore_mean - score_noexplore_std, score_noexplore_mean + score_noexplore_std):
#	print(m, s, l, h)

score_eps0_01_nodecay_basepath = 'mountaincar_results/eps0.01_nodecay'
score_eps0_01_nodecay, score_eps0_01_nodecay_mean, score_eps0_01_nodecay_std = prep_test_results(['{}/test_results_seed456.csv'.format(score_eps0_01_nodecay_basepath), 
										'{}/test_results_seed12.csv'.format(score_eps0_01_nodecay_basepath),
										'{}/test_results_seed985234.csv'.format(score_eps0_01_nodecay_basepath),
										'{}/test_results_seed123.csv'.format(score_eps0_01_nodecay_basepath),
										'{}/test_results_seed3202.csv'.format(score_eps0_01_nodecay_basepath)])

score_pure_explore_basepath = 'mountaincar_results/pure_exploration'
score_pure_explore, score_pure_explore_mean, score_pure_explore_std = prep_test_results(['{}/test_results_seed456.csv'.format(score_pure_explore_basepath), 
										'{}/test_results_seed12.csv'.format(score_pure_explore_basepath),
										'{}/test_results_seed985234.csv'.format(score_pure_explore_basepath),
										'{}/test_results_seed123.csv'.format(score_pure_explore_basepath),
										'{}/test_results_seed3202.csv'.format(score_pure_explore_basepath)])

score_epsdecay9_basepath = 'mountaincar_results/epsannealing_decay0.9'
score_epsdecay9, score_epsdecay9_mean, score_epsdecay9_std = prep_test_results(['{}/test_results_seed456.csv'.format(score_epsdecay9_basepath), 
										'{}/test_results_seed12.csv'.format(score_epsdecay9_basepath),
										'{}/test_results_seed985234.csv'.format(score_epsdecay9_basepath),
										'{}/test_results_seed123.csv'.format(score_epsdecay9_basepath),
										'{}/test_results_seed3202.csv'.format(score_epsdecay9_basepath)])

score_epsdecay999_basepath = 'mountaincar_results/epsannealing_decay0.999'
score_epsdecay999, score_epsdecay999_mean, score_epsdecay999_std = prep_test_results(['{}/test_results_seed456.csv'.format(score_epsdecay999_basepath), 
										'{}/test_results_seed12.csv'.format(score_epsdecay999_basepath),
										'{}/test_results_seed985234.csv'.format(score_epsdecay999_basepath),
										'{}/test_results_seed123.csv'.format(score_epsdecay999_basepath),
										'{}/test_results_seed3202.csv'.format(score_epsdecay999_basepath)])


x = list(range(len(score_noexplore_mean)))

plt.axhline(y=-110, color='r', linestyle='--')

#plt.plot(x, score_noexplore_mean, color='#212F3D', label= 'No exploration')
plt.plot(x, score_epsdecay9_mean, color='#6C3483', label='Epsilon-greedy with annealing, decay = 0.9')
plt.plot(x, score_epsdecay99_mean, color='#239B56', label='Epsilon-greedy with annealing, decay = 0.99')
plt.plot(x, score_epsdecay999_mean, color='#CA6F1E', label='Epsilon-greedy with annealing, decay = 0.999')
#plt.plot(x, score_eps0_01_nodecay_mean, color='#2874A6', label='Epsilon-greedy (no annealing), eps = 0.01')
#plt.plot(x, score_pure_explore_mean, color='#B7950B', label='Pure exploration')

#plt.fill_between(x, score_noexplore_mean - score_noexplore_std, score_noexplore_mean + score_noexplore_std, color='#ABB2B9', alpha=0.5)
plt.fill_between(x, score_epsdecay99_mean - score_epsdecay99_std, score_epsdecay99_mean + score_epsdecay99_std, color='#ABEBC6', alpha=0.5)
plt.fill_between(x, score_epsdecay9_mean - score_epsdecay9_std, score_epsdecay9_mean + score_epsdecay9_std, color='#D2B4DE', alpha=0.5)
plt.fill_between(x, score_epsdecay999_mean - score_epsdecay999_std, score_epsdecay999_mean + score_epsdecay999_std, color='#FAD7A0', alpha=0.5)
#plt.fill_between(x, score_eps0_01_nodecay_mean - score_eps0_01_nodecay_std, score_eps0_01_nodecay_mean + score_eps0_01_nodecay_std, color='#AED6F1', alpha=0.5)
#plt.fill_between(x, score_pure_explore_mean - score_pure_explore_std, score_pure_explore_mean + score_pure_explore_std, color='#F9E79F', alpha=0.5)

plt.legend(loc = 'lower right')
plt.xlim(xmin=0)
plt.ylim(ymin=-235, ymax=-65)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
#plt.show()
plt.savefig('annealing_test.png')
