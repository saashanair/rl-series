"""
Script that describes the details about the Actor and Critic architectures for the DDPG agent
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

def get_fan_in_init_bound(layer):
    """
    Function to compute the initialisation bound at 1/sqrt(f), where f is the fan-in value (i.e., number of inputs) of the given layer

    Parameters
    ---
    layer: torch.nn.module
        The layer of the network to be initialised

    Returns
    ---
    the fan-in based upper bound to be used for initialisation, such that the lower bound is the negative of this value
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor.weight)
    #fan_in = layer.weight.size(1) ## a potential solution to computing fan-in when using linear layers as the shape of the weight of the linear layer is [fan_out, fan_in]
    return 1/math.sqrt(fan_in)


def apply_uniform_init(layer, bound=None):
    """
    Function to initialise the specified layer using either the provided bound value or the fan-in based bound (suggested in the DDPG paper for hidden layers)

    Parameters
    ---
    layer: torch.nn.module
        The layer of the network to be initialised

    bound: float or None
        Specifies the value for the upper bound of the initialisation, such that the lower bound is the negative of this value. If None, then use fan-in based initilisation

    Returns
    ---
    none
    """
    if bound is None:
        bound = get_fan_in_init_bound(layer)
    nn.init.uniform_(layer.weight, a=-bound, b=bound) # initalise the weights
    nn.init.uniform_(layer.bias, a=-bound, b=bound) # initialise the biases


class Actor(nn.Module):
    """
    Class that defines the neural network architecture for the Actor
    """

    def __init__(self, state_dim, action_dim, max_action, lr=1e-4):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.dense1 = nn.Linear(state_dim, 400)
        apply_uniform_init(self.dense1)

        #self.bn1 = nn.BatchNorm1d(400)

        self.dense2 = nn.Linear(400, 300)
        apply_uniform_init(self.dense2)

        #self.bn2 = nn.BatchNorm1d(300)

        self.dense3 = nn.Linear(300, action_dim)
        apply_uniform_init(self.dense3, bound=3*10e-3)

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

        self.dense1 = nn.Linear(state_dim, 400) ## the input to the network is a concatenation of the state and the action performed by the agent in that state
        apply_uniform_init(self.dense1)

        #self.bn1 = nn.BatchNorm1d(400)

        self.dense2 = nn.Linear(400 + action_dim, 300)
        apply_uniform_init(self.dense2)

        #self.bn2 = nn.BatchNorm1d(300)

        self.dense3 = nn.Linear(300, 1)
        apply_uniform_init(self.dense3, bound=3*10e-4)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-2)

    def forward(self, state, action): 

        #x = torch.cat([state, action], dim=1)

        x = F.relu(self.dense1(state))

        x = torch.cat([x, action], dim=1)
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
