## initial thought was to use deque, but with a large replay memory it turns out to be not very efficient -- see https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
import random
import numpy as np

import torch

class ReplayMemory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.idx = 0

	def store(self, transition):
		if len(self.buffer) < self.capacity:
			self.buffer.append(transition)
		self.buffer[self.idx] = transition
		self.idx = (self.idx+1)%self.capacity

	def sample(self, batch_size, device):
		transitions = np.array(random.sample(self.buffer, batch_size))
		#print("T: ", transitions)
		states = torch.tensor(transitions[:, 0].tolist(), dtype=torch.float32).to(device)
		#print('States: ', states)
		actions = torch.tensor(transitions[:, 1].tolist(), dtype=torch.long).to(device)
		next_states = torch.tensor(transitions[:, 2].tolist(), dtype=torch.float32).to(device)
		rewards = torch.tensor(transitions[:, 3].tolist(), dtype=torch.float32).to(device)
		dones = torch.tensor(transitions[:, 4].tolist()).to(device)

		return states, actions, next_states, rewards, dones

	def __len__(self):
		return len(self.buffer)
