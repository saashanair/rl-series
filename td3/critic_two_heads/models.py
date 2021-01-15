"""
Script that describes the details about the Actor and Critic architectures for the TD3 agent.
The architecture for the Critic is based on the code in the repo provided by the authors of the paper: https://github.com/sfujim/TD3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

class Actor(nn.Module):
    """
    Class that defines the neural network architecture for the Actor
    """
    def __init__(self, state_dim, action_dim, min_action, max_action, lr=1e-3):
        super(Actor, self).__init__()

        self.min_action = min_action
        self.max_action = max_action

        self.dense1 = nn.Linear(state_dim, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        x = torch.tanh(self.dense3(x))

        #scaling_factor = torch.tensor([self.max_action]).to(self.device)

        return  self.max_action * x ## assumes action range is symmetric


    def save_model(self, filename):
        """
        Function to save model parameters
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        """
        Function to load model parameters
        """
        self.load_state_dict(torch.load(filename))


class Critic(nn.Module):
    """
    Class that defines the neural network architecture for the Critic. 
    Encapsulates two copies of the same network, reperesentative of the two critic outputs Q1 and Q2 described in the paper
    """
    def __init__(self, state_dim, action_dim, lr=1e-3):
        super(Critic, self).__init__()

        # Architecture for Q1
        self.dense1 = nn.Linear(state_dim + action_dim, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, 1)

        # Architecture for Q2
        self.dense4 = nn.Linear(state_dim + action_dim, 400)
        self.dense5 = nn.Linear(400, 300)
        self.dense6 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action): 

        x = torch.cat([state, action], dim=1)

        # Forward pass for Q1
        q1 = F.relu(self.dense1(x))
        q1 = F.relu(self.dense2(q1))
        q1 = self.dense3(q1)

        # Forward pass for Q2
        q2 = F.relu(self.dense4(x))
        q2 = F.relu(self.dense5(q2))
        q2 = self.dense6(q2)

        return q1, q2

    def save_model(self, filename):
        """
        Function to save model parameters
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        """
        Function to load model parameters
        """
        self.load_state_dict(torch.load(filename))
