"""
Script that describes the details about the Actor and Critic architectures for the TD3 agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 


class Actor(nn.Module):
    """
    Class that defines the neural network architecture for the Actor
    """

    def __init__(self, state_dim, action_dim, max_action, lr=1e-3):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.dense1 = nn.Linear(state_dim, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        x = torch.tanh(self.dense3(x)) ## squashes the action output to a range of -1 to +1

        return  self.max_action * x ## assumes action range is symmetric


    def save_model(self, filename):
        """
        Function to save model parameters

        Parameters
        ---
        filename: str
            Name of the model to save (along with its location)

        Returns
        ---
        none
        """

        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        """
        Function to load model parameters

        Parameters
        ---
        filename: str
            Name of the model to load (along with its location)

        Returns
        ---
        none
        """

        self.load_state_dict(torch.load(filename))


class Critic(nn.Module):
    """
    Class that defines the neural network architecture for the Critic
    """

    def __init__(self, state_dim, action_dim, lr=1e-3):
        super(Critic, self).__init__()

        self.dense1 = nn.Linear(state_dim + action_dim, 400) ## the input to the network is a concatenation of the state and the action performed by the agent in that state
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action): 

        x = torch.cat([state, action], dim=1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x) ## the computed Q-value for the given state-action pair

        return x

    def save_model(self, filename):
        """
        Function to save model parameters

        Parameters
        ---
        filename: str
            Name of the model to save (along with its location)

        Returns
        ---
        none
        """

        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        """
        Function to load model parameters

        Parameters
        ---
        filename: str
            Name of the model to load (along with its location)

        Returns
        ---
        none
        """
        
        self.load_state_dict(torch.load(filename))
