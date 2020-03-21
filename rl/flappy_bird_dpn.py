import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as func
from ple import PLE
from ple.games import FlappyBird
from ple.games.base import PyGameWrapper
from torch import nn, optim

from utils.log import logger
from utils.pytorch.mish import Mish
from utils.tools import Pytorch, ImmutableDict

HYPER_PARAMS = ImmutableDict({
    'gamma': 0.99,
    'max_epsilon': 1.0,
    'min_epsilon': 0.1,
    'epsilon_decay': 1 / 2000,
    # 状态、动作维度
    'obs_dim': 4,
    'action_dim': 2,
    # ReplayBuffer
    'capacity': 1000,
    'batch_size': 64,

    'epochs': 20000,
    'target_update': 100,
    'test': False,
    'target_update_interval': 10,
    'test_iteration': 10,
    'epoch_log': 100,
    'random_seed': 9527,

})


class Network(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super(Network, self).__init__()

        self.l1 = nn.Linear(obs_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)
        self.activate_func = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activate_func(self.l1(x))
        x = self.activate_func(self.l2(x))
        return self.l3(x)


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rew_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rews: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rews
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        num = self.batch_size if self.size > self.batch_size else self.size
        p = np.random.choice(self.size, size=num, replace=False)
        return dict(obs=self.obs_buf[p],
                    next_obs=self.next_obs_buf[p],
                    acts=self.acts_buf[p],
                    rews=self.rew_buf[p],
                    done=self.done_buf[p])

    def __len__(self) -> int:
        return self.size


class Agent:

    def __init__(self, hyper: dict, game: PyGameWrapper):
        self.hyper = hyper
        self.game = game
        self.p = PLE(game, fps=30, display_screen=True)
        self.p.init()

        self.memory = ReplayBuffer(hyper['obs_dim'], hyper['capacity'], hyper['batch_size'])
        self.epsilon_decay = hyper['epsilon_decay']
        self.epsilon = hyper['max_epsilon']
        self.max_epsilon = hyper['max_epsilon']
        self.min_epsilon = hyper['min_epsilon']
        self.gamma = torch.tensor(hyper['gamma']).to(Pytorch.device())
        self.target_update = hyper['target_update']

        self.dqn = Network(hyper['obs_dim'], hyper['action_dim']).to(Pytorch.device())
        self.dqn_target = Network(hyper['obs_dim'], hyper['action_dim']).to(Pytorch.device())
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.transition = list()
        self.is_test = hyper['test']
        self.epochs = hyper['epochs']
        self.batch_size = hyper['batch_size']
        self.epoch_log = hyper['epoch_log']

    def select_action(self, state: np.ndarray) -> int:
        def random_action(scale=1):
            action_max = int(scale * 100)
            r = random.randint(0, 100)
            if r <= action_max:
                return 1
            return 0
        """
        使用贪心（ ε—greedy ）搜索方法来对环境进行探索

        以 ε—greedy搜索以概率 ε 从所有可能的动作中随机选取一个动作
        以 1- ε 的概率选择已知的最好的动作（即当前状态下，Q值最大的那个动作）

        在初期， ε 的值应更大一些（即注重对环境的探索），随后逐渐减小 ε 的值（即注重对于Q值表的使用）
        self.epsilon会随着回合数减小，实现 ε 的值随着回合数的增加而递减。
        """
        if self.epsilon > np.random.random():
            selected_action = random_action()
        else:
            # 神经网络得到动作
            selected_action = self.dqn(torch.FloatTensor(state).to(Pytorch.device())).argmax()
            selected_action = selected_action.detach().cpu().item()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: int):
        reward = self.p.act(action)
        # 存储当前状态、行动、奖励、下一步状态、结束状态
        if not self.is_test:
            self.transition += [reward, self.state(), self.p.game_over()]
            self.memory.store(*self.transition)
        return reward

    def update_model(self):
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)
        loss = loss / len(samples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = Pytorch.device()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(device)

        loss = func.smooth_l1_loss(curr_q_value, target)
        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def state(self):
        obs = self.game.getGameState()
        return np.array([
            obs['player_y'], obs['player_vel'], obs['next_pipe_dist_to_player'],
            obs['next_pipe_top_y'], obs['next_pipe_bottom_y'],
            obs['next_next_pipe_dist_to_player'], obs['next_next_pipe_top_y'],
            obs['next_next_pipe_bottom_y']
        ])

    def train(self, ):
        self.is_test = False
        epsilons, losses, reward_records, update_cnt = [], [], [], 0,
        for frame_idx in range(1, self.epochs + 1):
            self.p.reset_game()
            reward = 0
            while not self.p.game_over():
                # 选取动作
                state = self.state()
                action = self.select_action(state)
                # 执行动作，获取更新的环境状态、奖励、是否完成等，并存储
                step_reward = self.step(action)
                reward = reward + step_reward

            # 计算损失函数，梯度下降
            loss = self.update_model()
            losses.append(loss)
            update_cnt += 1

            # 减少ε
            self.epsilon = max(self.min_epsilon,
                               self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
            epsilons.append(self.epsilon)

            # 更新target神经网络
            if update_cnt % self.target_update == 0:
                self._target_hard_update()

            reward_records.append(reward)
            if frame_idx % self.epoch_log == 0:
                avg_score = '%.2f' % np.mean(reward_records)
                logger.info("Epoch: %s, Score: %s, Avg-Score: %s, Loss: %s" % (frame_idx, reward, avg_score, loss))

    def test(self) -> None:
        self.is_test = True

        self.p.reset_game()
        total_reward = 0

        while not self.p.game_over():
            action = self.select_action(self.state())
            total_reward += self.step(action)
        logger.info("Total-Reward: %s" % total_reward)


def seed_torch(seed_):
    torch.manual_seed(seed_)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    hyper = HYPER_PARAMS.copy()
    hyper['obs_dim'] = 8
    hyper['action_dim'] = 2
    hyper['epoch_log'] = 1000
    hyper['epochs'] = 1000000

    game = FlappyBird()

    agent = Agent(hyper, game)
    agent.train()
    agent.test()
