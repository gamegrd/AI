import pprint
from collections import namedtuple

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import VectorEnv, SubprocVectorEnv
from tianshou.policy import A2CPolicy
from tianshou.trainer import onpolicy_trainer

from rl.tianshou.discrete_net import Net, Actor, Critic
from utils.tools import ImmutableDict, Pytorch

HYPER_PARAMS = ImmutableDict({
    'render': 1,
    'gamma': 0.99,
    'tau': 0.005,
    'gradient_steps': 1,
    'batch_size': 128,
    'seed': False,
    'random_seed': 9527,

    'state_dim': 1,
    'action_dim': 1,
    'min_log_std': -20,
    'max_log_std': 2,
    'training_num': 8,
    'test_num': 8,
    'layer_num': 2,

    # replay buffer size
    'capacity': 10000,
    'learning_rate': 3e-4,
    'transition': namedtuple('Transition', ['s', 'a', 'r', 's_', 'd']),
    'min_val': torch.tensor(1e-7).float(),

    # A2C
    'vf_coef': 0.5,
    'ent_coef': 0.001,
    'max_grad_norm': None,

    'train_iteration': 100000,
    'train_log_interval': 2000,
    'step_per_epoch': 1000,
    'collect_per_step': 100,
    'repeat_per_collect': 1,
    'epoch': 100,
})


def train(hyper: dict):
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    hyper['state_dim'] = 4
    hyper['action_dim'] = 2

    train_envs = VectorEnv([lambda: gym.make(env_id) for _ in range(hyper['training_num'])])
    test_envs = SubprocVectorEnv([lambda: gym.make(env_id) for _ in range(hyper['test_num'])])

    if hyper['seed']:
        np.random.seed(hyper['random_seed'])
        torch.manual_seed(hyper['random_seed'])
        train_envs.seed(hyper['random_seed'])
        test_envs.seed(hyper['random_seed'])

    device = Pytorch.device()

    net = Net(hyper['layer_num'], hyper['state_dim'], device=device)
    actor = Actor(net, hyper['action_dim']).to(device)
    critic = Critic(net).to(device)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=hyper['learning_rate'])
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor, critic, optim, dist, hyper['gamma'], vf_coef=hyper['vf_coef'],
        ent_coef=hyper['ent_coef'], max_grad_norm=hyper['max_grad_norm'])
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(hyper['capacity']))
    test_collector = Collector(policy, test_envs)

    writer = SummaryWriter('./a2c')

    def stop_fn(x):
        if env.env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        else:
            return False

    result = onpolicy_trainer(
        policy, train_collector, test_collector, hyper['epoch'],
        hyper['step_per_epoch'], hyper['collect_per_step'], hyper['repeat_per_collect'],
        hyper['test_num'], hyper['batch_size'], stop_fn=stop_fn, writer=writer,
        task=env_id)
    train_collector.close()
    test_collector.close()
    pprint.pprint(result)
    # 测试
    env = gym.make(env_id)
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=hyper['render'])
    print(f'Final reward: {result["rew"]}, length: {result["len"]}')
    collector.close()


if __name__ == '__main__':
    hyper_ = HYPER_PARAMS.copy()
    train(hyper_)
