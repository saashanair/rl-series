import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
	with open(filename) as f:
		reader = csv.reader(f)
		data = list(reader)
	score = [float(x[0]) for x in data]
	eps = [float(x[1]) for x in data]
	return score, eps

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

score_noexplore, eps_noexplore = read_file('mountaincar_results/noexplore/train_results.csv')
score_epsdecay99, eps_epsdecay99 = read_file('mountaincar_results/epsannealing_decay0.99/train_results.csv')
score_epsdecay9, eps_epsdecay9 = read_file('mountaincar_results/epsannealing_decay0.9/train_results.csv')
score_epsdecay999, eps_epsdecay999 = read_file('mountaincar_results/epsannealing_decay0.9954/train_results.csv')
score_epsdecay5, eps_epsdecay5 = read_file('mountaincar_results/epsannealing_decay0.5/train_results.csv')
score_eps0_01_nodecay, eps_eps0_01_nodecay = read_file('mountaincar_results/eps0.01_nodecay/train_results.csv')
score_pure_explore, eps_pure_explore = read_file('mountaincar_results/pure_exploration/train_results.csv')


mean100_noexplore = calculate_mean100score(score_noexplore[:1000])
mean100_epsdecay99 = calculate_mean100score(score_epsdecay99[:1000])
mean100_epsdecay9 = calculate_mean100score(score_epsdecay9[:1000])
mean100_epsdecay9954 = calculate_mean100score(score_epsdecay999[:1000])
mean100_epsdecay5 = calculate_mean100score(score_epsdecay5[:1000])

mean100_eps0_01_nodecay = calculate_mean100score(score_eps0_01_nodecay[:1000])
mean100_pure_explore = calculate_mean100score(score_pure_explore[:1000])


meanscore_y = [x+100 for x in range(len(mean100_noexplore))]

plt.axhline(y=-110, color='r', linestyle='--')
#plt.plot(meanscore_y, mean100_noexplore, color='#212F3D', label='No exploration')
plt.plot(score_epsdecay9, color='#D2B4DE', alpha=0.6)
plt.plot(score_epsdecay99, color='#ABEBC6', alpha=0.6)
plt.plot(score_epsdecay999, color='#FAD7A0', alpha=0.6)
plt.plot(meanscore_y, mean100_epsdecay9, color='#6C3483', label='Epsilon-greedy (with annealing), decay = 0.9')
plt.plot(meanscore_y, mean100_epsdecay99, color='#239B56', label='Epsilon-greedy (with annealing), decay = 0.99')
plt.plot(meanscore_y, mean100_epsdecay9954, color='#CA6F1E', label='Epsilon-greedy (with annealing), decay = 0.999')
#plt.plot(meanscore_y, mean100_epsdecay5, color='#239B56', label='annealing')
#plt.plot(meanscore_y, mean100_eps0_01_nodecay, color='#2874A6', label='Epsilon-greedy (no annealing), eps = 0.01')
#plt.plot(meanscore_y, mean100_pure_explore, color='#B7950B', label='Pure exploration')

plt.legend(loc = 'lower right')
plt.xlim(xmin=0)
plt.ylim(ymin=-235, ymax=-65)
plt.xlabel('Episodes')
plt.ylabel('Average reward over the past 100 episodes')
plt.tight_layout()
#plt.show()
plt.savefig('annealing_train.png')


"""
#plt.plot(eps_noexplore, color='#212F3D', label='No exploration, eps = 0.00')
plt.plot(eps_epsdecay9, color='#6C3483', label='Epsilon-greedy (with annealing), decay = 0.9')
plt.plot(eps_epsdecay99, color='#239B56', label='Epsilon-greedy (with annealing), decay = 0.99')
plt.plot(eps_epsdecay999, color='#CA6F1E', label='Epsilon-greedy (with annealing), decay = 0.999')
#plt.plot(eps_eps0_01_nodecay, color='#2874A6', label='Epsilon-greedy (no annealing), eps = 0.01')

plt.legend(loc = 'upper right')
#plt.xlim(xmin=0)
#plt.ylim(ymin=-230, ymax=-65)
plt.xlabel('Episodes')
plt.ylabel('Epsilon value')
plt.tight_layout()
#plt.show()
plt.savefig('epsvalues_annealing_train.png')
"""
