import gym
import numpy as np 
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.distributions as dist

from model import A3CNet

class TransitionBuffer:
    def __init__(self, num_learn_steps):
        self.max_capacity = num_learn_steps
        self.__reset()

    def __reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.cntr = 0


    def store(self, state, action, reward, next_state):
        if self.cntr == self.max_capacity:
            print('Buffer is full')
            return
        #self.states[self.cntr] = state
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.cntr += 1

    def retrieve(self, device):
        states = torch.tensor(self.states, dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(self.next_states, dtype=torch.float32).to(device)
        self.__reset()
        return states, actions, rewards, next_states

class Worker(mp.Process):
    def __init__(self, 
        global_network, 
        global_optimizer, 
        global_ep_count, 
        res_queue, 
        name,
        device,
        env_name,
        train_seed=12321,
        max_eps=1000, 
        num_learn_steps=5,
        discount=0.9,
        actor_loss_coeff=1.0,
        critic_loss_coeff=0.5,
        entropy_regularisation_coeff=0.01,
        lr=1e-5,
        train_mode=True):
        super(Worker, self).__init__()
        
        self.name = 'w-{}'.format(name)
        self.device = device
        self.train_mode = train_mode

        self.global_network = global_network
        #self.global_optimizer = global_optimizer
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr)

        self.global_ep_count = global_ep_count
        self.res_queue = res_queue

        self.max_eps =  max_eps
        self.num_learn_steps = num_learn_steps

        self.env = gym.make(env_name)
        self.env.seed(train_seed+name)
        self.state_size = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.discount = discount
        self.actor_loss_coeff = actor_loss_coeff
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_regularisation_coeff = entropy_regularisation_coeff

        self.local_network = A3CNet(state_size=self.state_size, num_actions=self.num_actions)
        self.local_network.load_state_dict(self.global_network.state_dict())
        self.transition_buffer = TransitionBuffer(self.num_learn_steps)

    def select_action(self, state, mode='train'):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_prob, _ = self.local_network.forward(state)
        if mode=='train':
        #    action = np.random.choice(self.num_actions)
            #print('Action Prob: ', action_prob, action_prob.sum())
            action = dist.Categorical(action_prob).sample().item()
            #print('Action: {}, {}'.format(action_prob, action))
            return action
        action = torch.argmax(action_prob).item()
        #print('Action: {}, {}'.format(actions_prob, action))
        return action

    def learn(self, last_state, done):
        states, actions, rewards, next_states = self.transition_buffer.retrieve(self.device)
        
        R = 0.
        if not done:
            _, v_s_ = self.local_network.forward(torch.tensor([last_state], dtype=torch.float32).to(self.device))
            R = v_s_.data.numpy()[0, 0]
            #print('RRRRR: ', R)

        R_targets = []
        for reward in rewards.tolist()[::-1]:#reversed(rewards):
            R = reward + self.discount * R
            R_targets.append(R)

        R_targets.reverse()
        R_targets = torch.tensor(R_targets, dtype=torch.float32).to(self.device).view(-1, 1)
        #print('R TARGETS: ', done, R_targets, R_targets.shape)

        policy_preds, values_preds = self.local_network.forward(states)
        
        advantage = R_targets - values_preds
        critic_loss = advantage.pow(2).view(-1, 1)

        responsible_outputs = policy_preds.gather(1, actions.view(-1, 1).detach())
        log_policy_responsible_outputs = torch.log(responsible_outputs)

        #for pp, ro, lps, a in zip(policy_preds, responsible_outputs, log_policy_responsible_outputs, actions):
        #    print('YO: ', pp, a, ro, lps)

        #print('ADV SHAPES: ', R_targets.shape, values_preds.shape, advantage.shape)
        #for rt, v, a in zip(R_targets, values_preds, advantage):
        #    print('ADV: ', rt, v, a)

        entropy = -(torch.sum(policy_preds * torch.log(policy_preds), dim=1).view(-1, 1))
        
        actor_loss = -((log_policy_responsible_outputs * advantage.detach()).view(-1, 1))

        total_loss = critic_loss + actor_loss + 0.01 * entropy
        #print('WOOHOO: ', entropy.shape, actor_loss.shape, critic_loss.shape, total_loss.shape)
        total_loss = total_loss.mean()

        self.global_optimizer.zero_grad()
        total_loss.backward()
        for gp, lp in zip(self.global_network.parameters(), self.local_network.parameters()):
            gp._grad = lp.grad
        torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), 5.0)
        self.global_optimizer.step()
        
        self.local_network.load_state_dict(self.global_network.state_dict())


    def run(self):
        local_step_cntr = 0
        t_start = local_step_cntr
        while self.global_ep_count.value < self.max_eps:
            state = self.env.reset()
            #print('Called reset')
            ep_reward = 0
            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = np.clip(reward, a_min=-1, a_max=1)
                self.transition_buffer.store(state, action, reward, next_state)

                if done or local_step_cntr - t_start == self.num_learn_steps:
                    t_start = local_step_cntr
                    self.learn(next_state, done)
                    #print('Learning step done....')

                state = next_state
                ep_reward += reward
                local_step_cntr += 1
                if done: break

            with self.global_ep_count.get_lock():
                self.global_ep_count.value += 1
            self.res_queue.put(ep_reward)
            print('GLOBAL: {}, {}, {}'.format(self.name, self.global_ep_count.value, ep_reward))
        self.res_queue.put(None)



































