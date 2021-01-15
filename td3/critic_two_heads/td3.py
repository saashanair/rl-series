"""
Script that contains the details about how the TD3 agent learns, generates actions and save/loads agents
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Actor, Critic
from replay_memory import ReplayMemory


class TD3Agent:
    """
    Encapsulates the functioning of the TD3 agent
    """

    def __init__(self, state_dim, action_dim, max_action, device, memory_capacity=10000, discount=0.99, update_freq=2, tau=0.005, policy_noise_std=0.2, policy_noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3, train_mode=True):
        self.train_mode = train_mode # whether the agent is in training or testing mode

        self.state_dim = state_dim # dimension of the state space
        self.action_dim = action_dim # dimension of the action space
        
        self.device = device # defines which cuda or cpu device is to be used to run the networks
        self.discount = discount # denoted a gamma in the equation for computation of the Q-value
        self.update_freq = update_freq # defines how frequently should the actor and target be updated
        self.tau = tau # defines the factor used for Polyak averaging (i.e., soft updating of the target networks)
        self.max_action = max_action # the max value of the range in the action space (assumes a symmetric range in the action space)
        self.policy_noise_clip = policy_noise_clip # max range within which the noise for the target policy smoothing must be contained
        self.policy_noise_std = policy_noise_std # standard deviation, i.e. sigma, of the Gaussian noise applied for target policy smoothing
        
        # create an instance of the replay buffer
        self.memory = ReplayMemory(memory_capacity)

        # instances of the networks for the actor and the two critics
        self.actor = Actor(state_dim, action_dim, max_action, actor_lr)
        self.critic = Critic(state_dim, action_dim, critic_lr) # the critic class encapsulates two copies of the neural network for the two critics used in TD3

        # instance of the target networks for the actor and the two critics
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

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)


    def select_action(self, state, exploration_noise=0.1):
        """
        Function to returns the appropriate action for the given state.
        During training, it returns adds a zero-mean gaussian noise with std=exploration_noise to the action to encourage exploration.
        No noise is added to the action decision during testing mode.

        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent
        exploration_noise: float, optional
            Standard deviation, i.e. sigma, of the Gaussian noise to be added to the agent's action to encourage exploration

        Returns
        ---
        A numpy array representing the noisy action to be performed by the agent in the current state
        """

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
            
        act = self.actor(state).cpu().data.numpy().flatten() # performs inference using the actor based on the current state as the input and returns the corresponding np array

        if not self.train_mode:
            exploration_noise = 0.0 # since we do not need noise to be added to the action during testing

        noise = np.random.normal(0.0, exploration_noise, size=act.shape) # generate the zero-mean gaussian noise with standard deviation determined by exploration_noise

        noisy_action = act + noise
        noisy_action = noisy_action.clip(min=-self.max_action, max=self.max_action) # to ensure that the noisy action being returned is within the limit of "legal" actions afforded to the agent; assumes action range is symmetric

        return noisy_action


    def learn(self, current_iteration, batchsize):
        """
        Function to perform the updates on the 6 neural networks that run the TD3 algorithm.

        Parameters
        ---
        current_iteration: int
            Total number of steps that have been performed by the agent
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

        # generate noisy target actions for target policy smoothing
        pred_action = self.target_actor(next_states)
        noise = torch.zeros_like(pred_action).normal_(0, self.policy_noise_std).to(self.device)
        noise = torch.clamp(noise, min=-self.policy_noise_clip, max=self.policy_noise_clip)
        noisy_pred_action = torch.clamp(pred_action + noise, min=-self.max_action, max=self.max_action)

        # calculate TD-Target using Clipped Double Q-learning
        target_q1, target_q2 = self.target_critic(next_states, noisy_pred_action)
        target_q = torch.min(target_q1, target_q2)
        target_q[dones] = 0.0 # being in a terminal state implies there are no more future states that the agent would encounter in the given episode and so set the associated Q-value to 0
        y = rewards + self.discount * target_q

        current_q1, current_q2 = self.critic(states, actions) # the critic class encapsulates two copies of the neural network thereby returning two Q values with each forward pass

        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y) # the losses of the two critics need to be added as there is only one optimiser shared between the two networks
        critic_loss = critic_loss.mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # delayed policy and target updates
        if current_iteration % self.update_freq == 0:

            # actor loss is calculated by a gradient ascent along crtic 1, thus need to apply the negative sign to convert to a gradient descent
            pred_current_actions = self.actor(states)
            pred_current_q1, _ = self.critic(states, pred_current_actions) # since we only need the Q-value from critic 1, we can ignore the second value obtained through the forward pass

            actor_loss = - pred_current_q1.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # apply slow-update to all three target networks
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


    def load(self, model_name):
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


