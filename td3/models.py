import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

class Actor(nn.Module):
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
    def __init__(self, state_dim, action_dim, lr=1e-3):
        super(Critic, self).__init__()

        # Q1
        self.dense1 = nn.Linear(state_dim + action_dim, 400)
        self.dense2 = nn.Linear(400, 300)
        self.dense3 = nn.Linear(300, 1)

        # Q2
        self.dense4 = nn.Linear(state_dim + action_dim, 400)
        self.dense5 = nn.Linear(400, 300)
        self.dense6 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action): 

        x = torch.cat([state, action], dim=1)

        q1 = F.relu(self.dense1(x))
        q1 = F.relu(self.dense2(q1))
        q1 = self.dense3(q1)

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
