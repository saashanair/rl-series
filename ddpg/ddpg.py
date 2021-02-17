"""
Script that contains the details about how the DDPG agent learns, generates actions and save/loads agents
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Actor, Critic
from replay_memory import ReplayMemory
from ou_noise import OrnsteinUhlenbeckNoise


class DDPGAgent:
    """
    Encapsulates the functioning of the DDPG agent
    """

    def __init__(self, state_dim, action_dim, max_action, device, memory_capacity=10000, discount=0.99, tau=0.005, sigma=0.2, theta=0.15, actor_lr=1e-4, critic_lr=1e-3, train_mode=True):
        self.train_mode = train_mode # whether the agent is in training or testing mode

        self.state_dim = state_dim # dimension of the state space
        self.action_dim = action_dim # dimension of the action space
        
        self.device = device # defines which cuda or cpu device is to be used to run the networks
        self.discount = discount # denoted a gamma in the equation for computation of the Q-value
        self.tau = tau # defines the factor used for Polyak averaging (i.e., soft updating of the target networks)
        self.max_action = max_action # the max value of the range in the action space (assumes a symmetric range in the action space)
        
        # create an instance of the replay buffer
        self.memory = ReplayMemory(memory_capacity)

        # create an instance of the noise generating process
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.action_dim), sigma=sigma, theta=theta)

        # instances of the networks for the actor and the critic
        self.actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.critic = Critic(state_dim, action_dim, critic_lr)

        # instance of the target networks for the actor and the critic
        self.target_actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.target_critic = Critic(state_dim, action_dim, critic_lr)

        # initialise the targets to the same weight as their corresponding current networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # since we do not learn/train on the target networks
        self.target_actor.eval()
        self.target_critic.eval()

        # for test mode
        if not self.train_mode:
            self.actor.eval()
            self.critic.eval()
            self.ounoise = None

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

    def select_action(self, state):
        """
        Function to return the appropriate action for the given state.
        During training, it adds a zero-mean OU noise to the action to encourage exploration.
        During testing, no noise is added to the action decision.

        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent
        
        Returns
        ---
        A numpy array representing the noisy action to be performed by the agent in the current state
        """

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
        
        self.actor.eval()
        act = self.actor(state).cpu().data.numpy().flatten() # performs inference using the actor based on the current state as the input and returns the corresponding np array
        self.actor.train()

        noise = 0.0

        ## for adding Gaussian noise (to use, update the code pass the exploration noise as input)
        #if self.train_mode:
        #	noise = np.random.normal(0.0, exploration_noise, size=act.shape) # generate the zero-mean gaussian noise with standard deviation determined by exploration_noise

        # for adding OU noise
        if self.train_mode:
            noise = self.ou_noise.generate_noise()

        noisy_action = act + noise
        noisy_action = noisy_action.clip(min=-self.max_action, max=self.max_action) # to ensure that the noisy action being returned is within the limit of "legal" actions afforded to the agent; assumes action range is symmetric

        return noisy_action

    def learn(self, batchsize):
        """
        Function to perform the updates on the 4 neural networks that run the DDPG algorithm.

        Parameters
        ---
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from

        Returns
        ---
        none
        """

        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device) # a batch of experiences randomly sampled form the memory

        # ensure that the actions and rewards tensors have the appropriate shapes
        actions = actions.view(-1, self.action_dim) 
        rewards = rewards.view(-1, 1)

        with torch.no_grad():
            # generate target actions
            target_action = self.target_actor(next_states)

            # calculate TD-Target
            target_q = self.target_critic(next_states, target_action)
            target_q[dones] = 0.0 # being in a terminal state implies there are no more future states that the agent would encounter in the given episode and so set the associated Q-value to 0
            y = rewards + self.discount * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y).mean()
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # actor loss is calculated by a gradient ascent along the crtic, thus need to apply the negative sign to convert to a gradient descent
        pred_current_actions = self.actor(states)
        pred_current_q = self.critic(states, pred_current_actions)
        actor_loss = - pred_current_q.mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # apply slow-update to the target networks
        self.soft_update_targets()


    def soft_update_net(self, source_net_params, target_net_params):
        """
        Function to perform Polyak averaging to update the parameters of the provided network

        Parameters
        ---
        source_net_params: list
            trainable parameters of the source, ie. current version of the network
        target_net_params: list
            trainable parameters of the corresponding target network

        Returns
        ---
        none
        """

        for source_param, target_param in zip(source_net_params, target_net_params):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def soft_update_targets(self):
        """
        Function that calls Polyak averaging on all three target networks

        Parameters
        ---
        none

        Returns
        ---
        none
        """

        self.soft_update_net(self.actor.parameters(), self.target_actor.parameters())
        self.soft_update_net(self.critic.parameters(), self.target_critic.parameters())

    def save(self, path, model_name):
        """
        Function to save the actor and critic networks

        Parameters
        ---
        path: str
            Location where the model is to be saved
        model_name: str
            Name of the model

        Returns
        ---
        none
        """

        self.actor.save_model('{}/{}_actor'.format(path, model_name))
        self.critic.save_model('{}/{}_critic'.format(path, model_name))

    def load(self, path, model_name):
        """
        Function to load the actor and critic networks

        Parameters
        ---
        path: str
            Location where the model is saved
        model_name: str
            Name of the model

        Returns
        ---
        none
        """

        self.actor.load_model('{}/{}_actor'.format(path, model_name))
        self.critic.load_model('{}/{}_critic'.format(path, model_name))


