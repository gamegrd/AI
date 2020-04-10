import os
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import Normal

from utils.tools import Pytorch, ImmutableDict

HYPER_PARAMS = ImmutableDict({
    'render': True,
    'gamma': 0.99,
    'tau': 0.005,
    'gradient_steps': 1,
    'batch_size': 128,
    'seed': False,
    'random_seed': 9527,

    'state_dim': 1,
    'action_dim': 1,
    'max_action': 1,
    'min_log_std': -20,
    'max_log_std': 2,

    # replay buffer size
    'capacity': 10000,
    'learning_rate': 3e-4,
    'transition': namedtuple('Transition', ['s', 'a', 'r', 's_', 'd']),
    'min_val': torch.tensor(1e-7).float(),

    'train_iteration': 100000,
    'train_log_interval': 2000,
})


class Actor(nn.Module):
    def __init__(self, hyper: dict):
        super(Actor, self).__init__()
        self.max_action = hyper['max_action']
        self.min_log_std = hyper['min_log_std']
        self.max_log_std = hyper['max_log_std']

        self.fc1 = nn.Linear(hyper['state_dim'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, 1)
        self.log_std_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, hyper: dict):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(hyper['state_dim'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, hyper: dict):
        super(Q, self).__init__()
        self.state_dim = hyper['state_dim']
        self.action_dim = hyper['action_dim']
        self.fc1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC():
    def __init__(self, hyper: dict):
        super(SAC, self).__init__()
        self.hyper = hyper
        self.device = Pytorch.device()
        self.policy_net = Actor(hyper).to(self.device)
        self.value_net = Critic(hyper).to(self.device)
        self.Q_net = Q(hyper).to(self.device)
        self.Target_value_net = Critic(hyper).to(self.device)

        self.replay_buffer = [hyper['transition']] * hyper['capacity']
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=hyper['learning_rate'])
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=hyper['learning_rate'])
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=hyper['learning_rate'])
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1
        self.writer = SummaryWriter('./exp-SAC')

        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action.item()  # return a scalar, float32

    def store(self, s, a, r, s_, d):
        index = self.num_transition % self.hyper['capacity']
        transition = self.hyper['transition'](s, a, r, s_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def get_action_log_prob(self, state):

        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + self.hyper['min_val'])
        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self):
        if self.num_training % 500 == 0:
            print("Training ... {} ".format(self.num_training))
        s = torch.tensor([t.s for t in self.replay_buffer]).float().to(self.device)
        a = torch.tensor([t.a for t in self.replay_buffer]).to(self.device)
        r = torch.tensor([t.r for t in self.replay_buffer]).to(self.device)
        s_ = torch.tensor([t.s_ for t in self.replay_buffer]).float().to(self.device)
        d = torch.tensor([t.d for t in self.replay_buffer]).float().to(self.device)

        for _ in range(self.hyper['gradient_steps']):
            # for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, False):
            index = np.random.choice(range(self.hyper['capacity']), self.hyper['batch_size'], replace=False)
            bn_s = s[index]
            bn_a = a[index].reshape(-1, 1)
            bn_r = r[index].reshape(-1, 1)
            bn_s_ = s_[index]
            bn_d = d[index].reshape(-1, 1)

            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + (1 - bn_d) * self.hyper['gamma'] * target_value

            excepted_value = self.value_net(bn_s)
            excepted_q = self.Q_net(bn_s, bn_a)

            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(bn_s)
            excepted_new_q = self.Q_net(bn_s, sample_action)
            next_value = excepted_new_q - log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)

            v_loss = self.value_criterion(excepted_value, next_value.detach())  # J_V
            v_loss = v_loss.mean()

            # Single Q_net this is different from original paper!!!
            q_loss = self.Q_criterion(excepted_q, next_q_value.detach())  # J_Q
            q_loss = q_loss.mean()

            log_policy_target = excepted_new_q - excepted_value

            pi_loss = log_prob * (log_prob - log_policy_target).detach()
            pi_loss = pi_loss.mean()

            self.writer.add_scalar('Loss/V_loss', v_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q_loss', q_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step=self.num_training)
            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            v_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q_optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
            self.Q_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # soft update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.hyper['tau']) + param * self.hyper['tau'])

            self.num_training += 1

    def save(self):
        torch.save(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.save(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.save(self.Q_net.state_dict(), './SAC_model/Q_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        torch.load(self.policy_net.state_dict(), './SAC_model/policy_net.pth')
        torch.load(self.value_net.state_dict(), './SAC_model/value_net.pth')
        torch.load(self.Q_net.state_dict(), './SAC_model/Q_net.pth')
        print()


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def train(hyper: dict):
    env = NormalizedActions(gym.make('Pendulum-v0'))

    if hyper['seed']:
        env.seed(hyper['random_seed'])
        torch.manual_seed(hyper['random_seed'])
        np.random.seed(hyper['random_seed'])

    hyper['state_dim'] = env.observation_space.shape[0]
    hyper['action_dim'] = env.action_space.shape[0]
    hyper['max_action'] = float(env.action_space.high[0])

    agent = SAC(hyper)

    ep_r = 0
    for i in range(hyper['train_iteration']):
        state = env.reset()
        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(np.float32(action))
            ep_r += reward
            if hyper['render']:
                env.render()
            agent.store(state, action, reward, next_state, done)

            if agent.num_transition >= hyper['capacity']:
                agent.update()

            state = next_state
            if done or t == 199:
                if i % 10 == 0:
                    print("Ep_i {}, the ep_r is {}, the t is {}".format(i, ep_r, t))
                break
        if i % hyper['train_log_interval'] == 0:
            agent.save()
        agent.writer.add_scalar('ep_r', ep_r, global_step=i)
        ep_r = 0


if __name__ == '__main__':
    hyper_ = HYPER_PARAMS.copy()
    train(hyper_)
