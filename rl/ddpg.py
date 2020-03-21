import os
from itertools import count

import gym
import numpy as np
import torch
import torch.nn.functional as func
from torch import nn, optim

from utils.log import logger
from utils.tools import Pytorch, TorchBoard
from utils.vars import MODULE_TRAIN

HYPER = {
    'tau': 0.005,
    'target_update_interval': 10,
    'test_iteration': 10,

    'learning_rate': 1e-3,
    'gamma': 0.99,
    # replay buffer size
    'capacity': 50000,
    'batch_size': 64,
    'seed': False,
    'random_seed': 9527,

    'sample_frequency': 256,
    'render': True,
    'log_interval': 50,
    'load': False,
    'render_interval': 100,
    'exploration_noise': 0.1,
    'max_episode': 100000,
    'max_length_of_trajectory': 2000,
    'print_log': 5,
    'update_iteration': 10,
}


class ReplayBuffer:
    """经验池"""

    def __init__(self, max_size=HYPER['capacity']):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            x_, y_, u_, r_, d_ = self.storage[i]
            x.append(np.array(x_, copy=False))
            y.append(np.array(y_, copy=False))
            u.append(np.array(u_, copy=False))
            r.append(np.array(r_, copy=False))
            d.append(np.array(d_, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, x, u):
        x = func.relu(self.l1(torch.cat([x, u], 1)))
        x = func.relu(self.l2(x))
        x = self.l3(x)
        return x


class Agent(object):
    def __init__(self, env, state_dim, action_dim, hidden_size=256):
        self.env = env
        self.actor = Actor(state_dim, action_dim, hidden_size).to(Pytorch.device())
        self.actor_target = Actor(state_dim, action_dim, hidden_size).to(Pytorch.device())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), HYPER['learning_rate'])

        self.critic = Critic(state_dim, action_dim).to(Pytorch.device())
        self.critic_target = Critic(state_dim, action_dim).to(Pytorch.device())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), HYPER['learning_rate'])

        self.replay_buffer = ReplayBuffer()
        self.writer = TorchBoard.writer('ddpg', MODULE_TRAIN)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(Pytorch.device())
        action = self.actor(state).cpu().data.numpy().flatten()
        env = self.env
        return (action + np.random.normal(0, HYPER['exploration_noise'], size=env.action_space.shape[0])).clip(
            env.action_space.low, env.action_space.high)

    def update(self):

        for it in range(HYPER['update_iteration']):
            x, y, u, r, d = self.replay_buffer.sample(HYPER['batch_size'])
            state = torch.FloatTensor(x).to(Pytorch.device())
            action = torch.FloatTensor(u).to(Pytorch.device())
            next_state = torch.FloatTensor(y).to(Pytorch.device())
            done = torch.FloatTensor(d).to(Pytorch.device())
            reward = torch.FloatTensor(r).to(Pytorch.device())

            # 计算target_Q的值
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = reward + ((1 - done) * HYPER['gamma'] * target_q).detach()

            # 获取当前的Q估计值
            current_q = self.critic(state, action)

            # Compute critic loss
            critic_loss = func.mse_loss(current_q, target_q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(HYPER['tau'] * param.data + (1 - HYPER['tau']) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(HYPER['tau'] * param.data + (1 - HYPER['tau']) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    @staticmethod
    def pkl_file(model='actor'):
        root_dirs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        pkl_dir = os.path.join(root_dirs, 'resources', 'pkl', 'ddpg')
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
        return os.path.join(pkl_dir, model + '.pkl')

    def save(self):
        torch.save(self.actor.state_dict(), Agent.pkl_file('actor'))
        torch.save(self.critic.state_dict(), Agent.pkl_file('critic'))

    def load(self):
        self.actor.load_state_dict(torch.load(Agent.pkl_file('actor')))
        self.critic.load_state_dict(torch.load(Agent.pkl_file('critic')))


def train():
    env = gym.make('Pendulum-v0').unwrapped

    if HYPER['seed']:
        env.seed(HYPER['random_seed'])
        torch.manual_seed(HYPER['random_seed'])
        np.random.seed(HYPER['random_seed'])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Agent(state_dim, action_dim, 256)
    ep_r = 0
    for i in range(HYPER['max_episode']):
        state = env.reset()
        for t in count():
            # 选取动作
            action = agent.select_action(state)
            # 获取奖励
            next_state, reward, done, info = env.step(action)
            ep_r += reward

            if HYPER['render'] and i >= HYPER['render_interval']:
                env.render()

            agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
            state = next_state
            if done or t >= HYPER['max_length_of_trajectory']:
                agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                if i % HYPER['print_log'] == 0:
                    logger.info("Step: %s... Reward: %s... Step: %s" % (t, ep_r, i))
                ep_r = 0
                break

        if i % HYPER['log_interval'] == 0:
            agent.save()
        if len(agent.replay_buffer.storage) >= HYPER['capacity'] - 1:
            agent.update()


if __name__ == '__main__':
    train()
