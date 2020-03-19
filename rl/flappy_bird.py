import os

import torch
import torch.nn as nn
from ple import PLE
from ple.games import FlappyBird
from torch import optim

from utils.pytorch.mish import Mish


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 120)
        self.l1_2 = nn.Linear(120, 120)
        self.l2 = nn.Linear(120, 1)
        self.activate_func = Mish()

    def forward(self, x_):
        x_ = torch.div(x_, torch.tensor(500))
        x_ = self.activate_func(self.l1(x_))
        x_ = self.activate_func(self.l1_2(x_))
        x_ = self.l2(x_)
        x_ = torch.sigmoid(x_)
        return x_


class FlappyBirdGame:
    def __init__(self, reward_values=None, reward_discount=0.99, pip_gap=100,
                 display_screen=True, fps=30, force_fps=True):
        if reward_values is None:
            reward_values = {}
        self.game = PLE(FlappyBird(pipe_gap=pip_gap), reward_values=reward_values,
                        fps=fps, force_fps=force_fps, display_screen=display_screen)
        self.game.init()
        self.actions = self.game.getActionSet()
        self.reward_discount = reward_discount

    @staticmethod
    def random_agent(*args, **kwargs):
        return torch.rand(1)

    def calculate_trial_reward(self, rewards_tensor):
        rewards_output = torch.empty(rewards_tensor.shape[0])
        for i in range(rewards_tensor.shape[0]):
            discount_vector = torch.Tensor([self.reward_discount] * (rewards_tensor.shape[0] - i))
            pv_rewards = sum(
                rewards_tensor[i:] * discount_vector ** torch.FloatTensor(range(rewards_tensor.shape[0] - i)))
            rewards_output[i] = pv_rewards
        rewards_output = rewards_output.reshape((-1, 1))
        return rewards_output

    @staticmethod
    def observation_to_torch_tensor(observation):
        obs = [observation['player_y'], observation['player_vel'], observation['next_pipe_dist_to_player'],
               observation['next_pipe_top_y'], observation['next_pipe_bottom_y'],
               observation['next_next_pipe_dist_to_player'], observation['next_next_pipe_top_y'],
               observation['next_next_pipe_bottom_y']]
        obs_tensor = torch.FloatTensor(obs)
        obs_tensor = obs_tensor.reshape((1, 8))
        return obs_tensor

    def run_trial(self, agent=None, sample=True, verbose=False):
        if agent is None:
            agent = self.random_agent
        if self.game.game_over():
            self.game.reset_game()
        rewards = torch.empty(0)
        observations = torch.empty((0, 8))
        agent_decisions = torch.empty((0, 1))
        actual_decisions = torch.empty((0, 1))
        while not self.game.game_over():
            observation = self.observation_to_torch_tensor(self.game.getGameState())
            agent_decision = agent(observation)

            if sample:
                actual_decision = torch.bernoulli(agent_decision)
            else:
                actual_decision = torch.FloatTensor([1]) if agent_decision > 0.5 else torch.FloatTensor([0])

            actual_decision = actual_decision.reshape((1, 1))
            agent_decision = agent_decision.reshape((1, 1))
            if actual_decision == 1:
                action = self.actions[1]
            else:
                action = self.actions[0]

            reward = torch.FloatTensor([self.game.act(action)])

            # reward shaping
            # if (observation[0][0] < observation[0][4]) and (observation[0][0] > observation[0][3]):
            #     reward = torch.add(reward, torch.tensor(0.2))
            # else:
            #     reward = torch.add(reward, torch.tensor(-0.2))

            rewards = torch.cat((rewards, reward))
            observations = torch.cat((observations, observation))
            agent_decisions = torch.cat((agent_decisions, agent_decision))
            actual_decisions = torch.cat((actual_decisions, actual_decision))
            if verbose:
                print(f'action: {action}')
                print(f'observation: {observation}')
                print(f'reward: {reward}')

        return {'observations': observations,
                'rewards': self.calculate_trial_reward(rewards),
                'agent_decisions': agent_decisions,
                'actual_decisions': actual_decisions}

    def run_n_trials(self, n_trials, agent=None, sample=True):
        out_results = {'observations': torch.empty(0), 'rewards': torch.empty(0),
                       'agent_decisions': torch.empty(0), 'actual_decisions': torch.empty(0)}
        for i in range(n_trials):
            results = self.run_trial(agent, sample)
            out_results['observations'] = torch.cat((out_results['observations'], results['observations']))
            out_results['rewards'] = torch.cat((out_results['rewards'], results['rewards']))
            out_results['agent_decisions'] = torch.cat((out_results['agent_decisions'], results['agent_decisions']))
            out_results['actual_decisions'] = torch.cat((out_results['actual_decisions'], results['actual_decisions']))

        return out_results


class FlappyBot:
    def __init__(self, nn_agent=None, path=None, hyper=None):
        if hyper is None:
            hyper = {}
        self.nn_agent = nn_agent
        self.path = path
        self.game = FlappyBirdGame(**hyper)
        self.hyper = hyper
        if self.path is not None:
            self.nn_agent = torch.load(self.path)
        assert self.nn_agent is not None, 'flappy_bot needs an agent to be initialized'

    def init_game(self, flappy_bird_game_params):
        self.game = FlappyBirdGame(**flappy_bird_game_params)

    def train(self, epochs, trials_per_epoch=1, learning_rate=1e-3, verbose=False):
        optimizer = optim.Adam(self.nn_agent.parameters(), lr=learning_rate)

        for i in range(epochs):
            results = self.game.run_n_trials(trials_per_epoch, self.nn_agent.forward)
            if i % 50 == 0 and verbose:
                print(results['rewards'].max())
                print(results['actual_decisions'].std())

            rewards = results['rewards']
            rewards = (rewards - rewards.mean()) / rewards.std()
            agent_decisions = results['agent_decisions']
            actual_decisions = results['actual_decisions']

            # likelihood of path
            ll = actual_decisions * agent_decisions + (1 - actual_decisions) * (1 - agent_decisions)
            loss = torch.mul(torch.sum(torch.mul(torch.log(ll), rewards)), -1)

            optimizer.zero_grad()
            loss.backward()

            if i % 50 == 0 and verbose:
                print(loss.item())
            if self.path is not None and i % 50 == 0:
                self.save()

            optimizer.step()

        if self.path is not None:
            self.save()

    def run_trial(self, n_trials=1):
        self.game.run_n_trials(n_trials, self.nn_agent.forward, False)

    def save(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        file = os.path.join(root_dir, 'flappy.pkl')
        torch.save(self.nn_agent, file)


if __name__ == '__main__':
    flp_dict = {
        'force_fps': True,
        'display_screen': True,
        'reward_values': {
            "positive": 1.0,
            "tick": 0.1,
            "loss": -5.0
        },
        'reward_discount': 0.99
    }
    agent = Agent()

    train_bot = FlappyBot(hyper=flp_dict, nn_agent=agent)

    train_bot.train(epochs=1000000, verbose=True)
