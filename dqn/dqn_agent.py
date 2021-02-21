"""
Script that contains details about the architecture of the neural network and the learning methods used by the DQN agent
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ReplayMemory

class DQNNet(nn.Module):
    """
    Class that defines the architecture of the neural network of the DQN agent
    """
    def __init__(self, input_size, output_size):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, output_size)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def save_model(self, filename):
        """
        Function to save model parameters
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device):
        """
        Function to load model parameters
        """
        #if not torch.cuda.is_available():
        #self.load_state_dict(torch.load(filename, map_location='cpu'))
        self.load_state_dict(torch.load(filename, map_location=device)) # to ensure that a model that is trained on GPU can be run even on CPU



class DQNAgent:
    """
    Class that defines the functions required for training the DQN agent
    """
    def __init__(self, device, state_size, action_size, discount=0.99, eps_max=1.0, eps_min=0.01, eps_decay=0.995, memory_capacity=5000, train_mode=True):
        self.device = device

        # for epsilon-greedy exploration strategy
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        # for defining how far-sighted or myopic the agent should be
        self.discount = discount

        # details about the state vector recieved from the environment and the actions that the agent is allowed to perform
        self.state_size = state_size
        self.action_size = action_size

        # building the policy and target Q-networks for the agent, such that the target Q-network is kept frozen to avoid the training instability issues
        self.policy_net = DQNNet(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size).to(self.device)
        self.target_net.eval()

        # building the experience replay memory used to avoid training instability issues
        self.memory = ReplayMemory(capacity=memory_capacity)

        if not train_mode:
            self.policy_net.eval()


    def update_target_net(self):
        """
        Copies the current weights from the policy Q-network into the frozen target Q-network
        """
        #print('Updating target network...')
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def update_epsilon(self):
        """
        Used for reducing the epsilon value to allow for annealing with epsilon-greedy exploration strategy
        """
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


    def select_action(self, state):
        """
        Uses epsilon-greedy exploration such that, if the randomly generated number is less than epsilon then the agent performs random action, else the agent executes the action suggested by the policy Q-network
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.tensor([state], dtype=torch.float32).to(self.device)

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item()


    def learn(self, batchsize):
        """
        Function that performs the action learning in the agent by updating the weights in the required direction
        """

        # select n samples picked uniformly at random from the experience replay memory, such that n=batchsize
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)

        # get q values of the actions that were taken, i.e calculate qpred; actions has to be explicitly reshaped to nx1-vector
        q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1)) 
        
        #calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        q_target = self.target_net.forward(next_states).max(dim=1).values
        q_target[dones] = 0.0 # setting Q(s',a') to 0 when the current state is a terminal state
        y_j = rewards + (self.discount*q_target)
        y_j = y_j.view(-1, 1)
        
        # calculate the loss as the mean-squared error of yj and qpred and the corresponding gradients to update the weights of the network
        self.policy_net.optimizer.zero_grad() # manually clear out gradient coz backward() function accumulates gradients and you dont want them to mix up between minibatches
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward() # calculates the gradients
        self.policy_net.optimizer.step() # updates the values of the weights

        return loss
        

    def save_model(self, policy_net_filename):
        """
        Function to save the parameters of the policy and target models
        """
        self.policy_net.save_model(policy_net_filename)

    def load_model(self, policy_net_filename):
        """
        Function to load the parameters of the policy and target models
        """
        #print('Loading model...')
        self.policy_net.load_model(filename=policy_net_filename, device=self.device)












