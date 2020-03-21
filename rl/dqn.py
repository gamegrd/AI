from typing import Dict, Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as func
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
        p = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[p],
                    next_obs=self.next_obs_buf[p],
                    acts=self.acts_buf[p],
                    rews=self.rew_buf[p],
                    done=self.done_buf[p])

    def __len__(self) -> int:
        return self.size


class Agent:

    def __init__(self, hyper: dict, env: gym.Env):
        self.env = env
        self.hyper = hyper
        self.memory = ReplayBuffer(hyper['obs_dim'], hyper['capacity'], hyper['batch_size'])
        self.epsilon_decay = hyper['epsilon_decay']
        self.epsilon = hyper['max_epsilon']
        self.max_epsilon = hyper['max_epsilon']
        self.min_epsilon = hyper['min_epsilon']
        self.gamma = hyper['gamma']
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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        使用贪心（ ε—greedy ）搜索方法来对环境进行探索

        以 ε—greedy搜索以概率 ε 从所有可能的动作中随机选取一个动作
        以 1- ε 的概率选择已知的最好的动作（即当前状态下，Q值最大的那个动作）

        在初期， ε 的值应更大一些（即注重对环境的探索），随后逐渐减小 ε 的值（即注重对于Q值表的使用）
        self.epsilon会随着回合数减小，实现 ε 的值随着回合数的增加而递减。
        """
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            # 神经网络得到动作
            selected_action = self.dqn(torch.FloatTensor(state).to(Pytorch.device())).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        print(selected_action)

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)

        # 存储当前状态、行动、奖励、下一步状态、结束状态
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)
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

    def train(self, ):
        self.is_test = False
        state = self.env.reset()
        epsilons, losses, scores, score, update_cnt = [], [], [], 0, 0
        for frame_idx in range(1, self.epochs + 1):
            # 选取动作
            action = self.select_action(state)
            # 执行动作，获取更新的环境状态、奖励、是否完成等，并存储
            next_state, reward, done = self.step(action)
            state, score = next_state, score + reward

            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # 准备好训练神经网络
            if len(self.memory) >= self.batch_size:
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

                avg_score = '%.2f' % np.mean(scores)
                logger.info("Epoch: %s, Score: %s, Avg-Score: %s" % (frame_idx, score, avg_score))

        self.env.close()

    def test(self) -> None:
        self.is_test = True

        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            self.env.render()
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            total_reward += reward
        logger.info("Total-Reward: %s" % total_reward)
        self.env.close()


def seed_torch(seed_):
    torch.manual_seed(seed_)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    hyper = HYPER_PARAMS.copy()

    agent = Agent(hyper, env)
    agent.train()
    agent.test()
