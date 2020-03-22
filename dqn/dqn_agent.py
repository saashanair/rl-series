import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ReplayMemory

class DQNNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(DQNNet, self).__init__()
		self.dense1 = nn.Linear(input_size, 256)
		self.dense2 = nn.Linear(256, 256)
		self.dense3 = nn.Linear(256, output_size)

		self.optimizer = optim.Adam(self.parameters())
		self.loss = nn.MSELoss()

	def forward(self, x):
		x = F.relu(self.dense1(x))
		x = F.relu(self.dense2(x))
		x = self.dense3(x)
		return x

	def save_model(self, filename):
		torch.save(self.state_dict(), filename)

	def load_model(self, filename):
		self.load_state_dict(torch.load(filename))


class DQNAgent:
	def __init__(self, device, state_size, action_size, discount=0.99, eps_max=1, eps_min=0.01, eps_decay=0.995):
		self.device = device

		self.epsilon = eps_max
		self.epsilon_min = eps_min
		self.epsilon_decay = eps_decay

		self.discount = discount
		self.state_size = state_size
		self.action_size = action_size

		self.policy_net = DQNNet(self.state_size, self.action_size).to(self.device)
		self.target_net = DQNNet(self.state_size, self.action_size).to(self.device)
		self.target_net.eval()

		self.memory = ReplayMemory(capacity=5000)

	def update_target_net(self):
		print('Updating target network...')
		self.target_net.load_state_dict(self.policy_net.state_dict())

	def update_epsilon(self):
		self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

	def random_exploration(self):
		return random.randrange(self.action_size)

	def select_action(self, state, explore=True):
		#self.eps = self.eps*self.eps_decay if self.eps > self.eps_min else self.eps_min
		#self.eps = max(self.eps_min, self.eps*self.eps_decay)

		if explore and random.random() <= self.epsilon:
			#print('In explore')
			return self.random_exploration()
			#return random.randrange(self.action_size)
		#print('Exploiting...')
		state = torch.tensor([state], dtype=torch.float32).to(self.device)
		#print('state: ', state)
		with torch.no_grad():
			action = self.policy_net.forward(state)
		#print('act: ', action)
		#print(action, torch.argmax(action), torch.argmax(action).item())
		return torch.argmax(action).item()

	def learn(self, batch_size):
		if len(self.memory) < batch_size:
			return

		states, actions, next_states, rewards, dones = self.memory.sample(batch_size, self.device)
		#print('Dones: ', dones)
		#print(states.shape, actions.shape)
		q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1)) # get q values of the actions that were taken; actions has to be explicitly reshaped to nx1 vector
		#print('Q pred: ', q_pred)

		q_target = self.target_net.forward(next_states).max(dim=1).values
		#print('Q target: ', q_target)
		#print('dones: ', dones)
		#print('qt of dones: ', q_target[dones])
		#print(type(q_target), type(dones))
		
		q_target[dones] = 0.0
		#print('Q target dones: ', q_target)
		y_j = rewards + (self.discount*q_target)
		y_j = y_j.view(-1, 1)
		#print('Yj: ', y_j, rewards)

		self.policy_net.optimizer.zero_grad() # manually clear out gradient coz backward() function accumulates gradients and you dont want them to mix up between minibatches
		#print('shape: ', y_j.shape, q_pred.shape)
		loss = self.policy_net.loss(y_j, q_pred)
		loss.backward() # calculates the gradients
		self.policy_net.optimizer.step() # updates the values of the weights
		

	def save_models(self, policy_net_filename, target_net_filename):
		print('Saving model...')
		self.policy_net.save_model(policy_net_filename)
		self.target_net.save_model(target_net_filename)

	def load_models(self, policy_net_filename, target_net_filename):
		print('Loading model...')
		self.policy_net.load_model(policy_net_filename)
		self.target_net.load_model(target_net_filename)












