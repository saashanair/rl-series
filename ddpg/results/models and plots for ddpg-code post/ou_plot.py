import glob
import matplotlib.pyplot as plt
import numpy as np 
import pickle

env = 'LunarLander'

path = 'ounoise'

def read_file(filename):
    with open(filename, 'rb') as f:
        n = pickle.load(f)
        return n

ounoise = read_file('{}/noise.pkl'.format(path))
print(len(ounoise))
#print(len(ounoise[:2]))
ounoise_main_engine = []
ounoise_left_engine = []
x_axis = []

for i, n in enumerate(ounoise[:1000]):
	#print(n)
	x_axis.append(i)
	ounoise_main_engine.append(n[0])
	ounoise_left_engine.append(n[1])

print(len(x_axis))


mean_value = [0.0] * len(x_axis)

plt.plot(ounoise_main_engine, color='#2874A6', label='OU Noise Main Engine')
plt.plot(x_axis, ounoise_left_engine, color='#B7950B', label='OU Noise Left Engine')
plt.plot(x_axis, mean_value, '--', color='black', label='Mean')
plt.xlabel('Time step')
plt.ylabel('Value from OU Process')
plt.tight_layout()

#plt.show()
plt.savefig('{}_ou.png'.format(env))
