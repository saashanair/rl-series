import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class A3CNet(nn.Module):
    def __init__(self, state_size, num_actions):
        super(A3CNet, self).__init__()
        self.dense1 = nn.Linear(state_size, 512)
        self.dense2 = nn.Linear(512, 512)
        
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

        #self.optimizer = optim.RMSprop(self.parameters(), lr=1e-6)
        #self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        
        p = self.policy(x)
        actor_output = F.softmax(p, dim=1)
        
        critic_output = self.value(x)
        

        return actor_output, critic_output

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

    def select_action(self, state, device):
        state = torch.tensor([state], dtype=torch.float32).to(device)
        with torch.no_grad():
            action_prob, _ = self.forward(state)
        action = dist.Categorical(action_prob).sample().item()
        return action


"""
class A3CNet(nn.Module):
    def __init__(self, state_size, num_actions):
        super(A3CNet, self).__init__()
        self.dense1 = nn.Linear(state_size, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 256)

        self.dense6 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 128)

        self.dense7 = nn.Linear(256, 256)
        self.dense4 = nn.Linear(256, 128)

        self.policy = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

        #self.optimizer = optim.RMSprop(self.parameters(), lr=1e-6)
        #self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        #print('After dense1: ', x.size())
        x = F.relu(self.dense2(x))
        #print('After dense2: ', x.size())
        x = F.relu(self.dense5(x))

        xp = F.relu(self.dense6(x))
        xp = F.relu(self.dense3(xp))
        p = self.policy(xp)
        #print('After policy: ', p.size())
        actor_output = F.softmax(p, dim=1)
        #print('After softmax: ', actor_output.size())
        xv = F.relu(self.dense7(x))
        xv = F.relu(self.dense4(xv))
        critic_output = self.value(xv)
        #print('After value: ', critic_output.size())

        return actor_output, critic_output
"""

"""
class A3CNet(nn.Module):
    def __init__(self, state_size, num_actions):
        super(A3CNet, self).__init__()
        
        self.s_dim = state_size
        self.a_dim = num_actions

        self.dense1 = nn.Linear(self.s_dim, 128)
        self.policy = nn.Linear(128, self.a_dim)

        self.dense2 = nn.Linear(self.s_dim, 128)
        self.value = nn.Linear(128, 1)
        #set_init([self.pi1, self.pi2, self.v1, self.v2])
        #self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x1 = torch.tanh(self.dense1(x))
        logits = self.policy(x1)
        actor_output = F.softmax(logits, dim=1)

        x2 = torch.tanh(self.dense2(x))
        critic_output = self.value(x2)

        return actor_output, critic_output
"""
