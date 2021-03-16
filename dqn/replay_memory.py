"""
Script that contains the details about the experience replay buffer used in DQN to ensure training stability
"""

import random
import numpy as np

import torch

class ReplayMemory:
    """
    Class representing the replay buffer used for storing experiences for off-policy learning
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.buffer_done = []
        self.idx = 0

    def store(self, state, action, next_state, reward, done):
        """
        Function to add the provided experience to the memory, such that transition is a 5-tuple of the form (state, action, next_state, reward, done)

        Parameters
        ---
        state: numpy.ndarray
            Current state vector observed in the environment
        action: int
            Action performed by the agent in the current state
        next_state: numpy.ndarray
            State vector observed as a result of performing the action in the current state
        reward: float
            Reward obtained by the agent
        done: bool
            Indicates whether the agent has entered a terminal state or not

        Returns
        ---
        none
        """

        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state
            self.buffer_reward[self.idx] = reward
            self.buffer_done[self.idx] = done

        self.idx = (self.idx+1)%self.capacity # for circular memory

    def sample(self, batch_size, device):
        """
        Function to pick 'n' samples from the memory that are selected uniformly at random, such that n = batchsize

        Parameters
        ---
        batchsize: int
            Number of elements to randomly sample from the memory in each batch
        device: str
            Name of the device (cuda or cpu) on which the computations would be performed

        Returns
        ---
        Tensors representing a batch of transitions sampled from the memory
        """
        
        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)

        states = torch.from_numpy(np.array(self.buffer_state)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.buffer_action)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.buffer_next_state)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.buffer_reward)[indices_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.buffer_done)[indices_to_sample]).to(device)

        return states, actions, next_states, rewards, dones
        

    def __len__(self):
        """
        Function that specifies the number of elements persent in the replay memory


        Parameters
        ---
        none

        Returns
        ---
        int
            number of currently stored elements in the replay buffer
        """
        return len(self.buffer_state)
