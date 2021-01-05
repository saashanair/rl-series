import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Actor, Critic
from replay_memory import ReplayMemory


class TD3Agent:
    def __init__(self, state_dim, action_dim, min_action, max_action, device, memory_capacity=10000, discount=0.99, update_freq=2, tau=0.005, policy_noise_std=0.2, policy_noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3, train_mode=True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.device = device
        self.discount = discount
        self.update_freq = update_freq
        self.tau = tau
        self.min_action = min_action
        self.max_action = max_action
        self.policy_noise_clip = policy_noise_clip
        self.policy_noise_std = policy_noise_std
        
        self.memory = ReplayMemory(memory_capacity)

        self.actor = Actor(state_dim, action_dim, min_action, max_action, actor_lr)
        self.critic = Critic(state_dim, action_dim, critic_lr)

        self.target_actor = Actor(state_dim, action_dim, min_action, max_action, actor_lr)
        self.target_critic = Critic(state_dim, action_dim, critic_lr)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.target_actor.eval()
        self.target_critic.eval()

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)

    def select_action(self, state, exploration_noise=0.1):
        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
            
        act = self.actor(state).cpu().data.numpy().flatten()

        noise = np.random.normal(0, exploration_noise, size=act.shape)

        noisy_action = act + noise
        noisy_action = noisy_action.clip(min=-self.max_action, max=self.max_action)

        return noisy_action

    def learn(self, current_iteration, batchsize):
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)

        actions = actions.view(-1, self.action_dim)
        rewards = rewards.view(-1, 1)

        pred_action = self.target_actor(next_states)
        noise = torch.zeros_like(pred_action).normal_(0, self.policy_noise_std).to(self.device)
        noise = torch.clamp(noise, min=-self.policy_noise_clip, max=self.policy_noise_clip)
        noisy_pred_action = torch.clamp(pred_action + noise, min=-self.max_action, max=self.max_action)

        target_q1, target_q2 = self.target_critic(next_states, noisy_pred_action)
        target_q = torch.min(target_q1, target_q2)
        target_q[dones] = 0.0
        y = rewards + self.discount * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        critic_loss = critic_loss.mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if current_iteration % self.update_freq == 0:
            pred_current_actions = self.actor(states)
            pred_current_q1, _ = self.critic(states, pred_current_actions)

            actor_loss = - pred_current_q1.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.soft_update_targets()


    def soft_update_net(self, source_net_params, target_net_params):
        for source_param, target_param in zip(source_net_params, target_net_params):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def soft_update_targets(self):
        self.soft_update_net(self.actor.parameters(), self.target_actor.parameters())
        self.soft_update_net(self.critic.parameters(), self.target_critic.parameters())

    def save(self, path, model_name):
        self.actor.save_model('{}/{}_actor'.format(path, model_name))
        self.critic.save_model('{}/{}_critic'.format(path, model_name))

    def load(self, model_name):
        self.actor.load_model('{}/{}_actor'.format(path, model_name))
        self.critic.load_model('{}/{}_critic'.format(path, model_name))


