"""
Script to plot annealing of the epsilon value during training
"""

import glob
import numpy as np 
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

path = '../pretrained/CartPole-v0_epsmax1.0_epsmin0.01_epsdec0.995_batchsize64'

def read_file(filename):
    """
    Function to read contents of the specifed pickle file

    Parameters
    ---
    filename: str
        Location of the file that needs to be read

    Returns
    ---
    scores: list of floats
        History of episodic scores received by the agent during testing
    """
    with open(filename, 'rb') as f:
        scores = pickle.load(f)
        return scores

epsilon_values = read_file('{}/train_epsilon_history.pkl'.format(path))

plt.plot(epsilon_values, color='#2874A6') 

plt.xlim(left=0)
plt.xlabel('Episodes')
plt.ylabel('Epsilon Values')
plt.tight_layout()

plt.savefig('epsilon_train.png'.format(env))
#plt.show()

