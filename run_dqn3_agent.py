import random
import sys

import numpy as np
import torch
from easydict import EasyDict
from matplotlib import pyplot as plt

from agents.dqn3_agent import train_dqn3_agent
from environment import StocksEnv
from utils.experiment import ExperimentResult
from utils.plotting import plot_curves


class StocksEnvWithFeatureVectors(StocksEnv):

    def _avg_last(self, observation, index, n):
        return np.mean(observation[-n:, index])

    def _feature_vec(self, observation):
        return np.array([
            self._avg_last(observation, 0, 200),  # 200 day moving average
            self._avg_last(observation, 0, 50),  # 50 day moving average
            observation[-1, -1]  # most recent position
        ])

    def _get_observation(self) -> np.ndarray:
        observation = super()._get_observation()
        position_history = np.zeros(shape=(observation.history.shape[0], 1))
        position_history[-len(observation.position_history):, 0] = observation.position_history
        obs_with_history = np.hstack([observation.history, position_history])  # add position history
        return self._feature_vec(obs_with_history)


def main():
    name = ''
    if len(sys.argv) > 1:
        name = sys.argv[1]

    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    env = StocksEnvWithFeatureVectors(EasyDict({
        "env_id": 'stocks-dqn', "eps_length": 200,
        "window_size": 200, "train_range": None, "test_range": None,
        "stocks_data_filename": 'STOCKS_GOOGL'
    }))

    initial_obs = env.reset()

    # create training parameters
    train_parameters = {
        'observation_dim': len(initial_obs),
        'action_dim': 5,
        'action_space': env.action_space,
        'hidden_layer_num': 2,
        'hidden_layer_dim': 8,
        'gamma': 1,

        'max_time_step_per_episode': 200,

        'total_training_time_step': 500_000 * 2,

        'epsilon_start_value': 0.1,
        'epsilon_end_value': 0.1,
        'epsilon_duration': 400_000,

        'freq_update_target_policy': 20_000,

        'learning_rate': 1e-3,

        'final_policy_num_plots': 20,

        'model_name': "stocks_google.pt",
        'name': name
    }

    # create experiment
    train_returns, train_loss, train_profits = train_dqn3_agent(env, train_parameters)
    plot_curves([np.array([train_returns])], ['dqn'], ['r'], 'discounted return', 'DQN2')
    plt.savefig(f'dqn3_returns_{name}')
    plt.clf()
    plot_curves([np.array([train_loss])], ['dqn'], ['r'], 'training loss', 'DQN2')
    plt.savefig(f'dqn3_loss_{name}')
    plt.clf()
    plot_curves([np.array([train_profits]), np.array([(moving_average(train_profits, n=50))])],
                ['raw profits', '50-episode moving average'], ['r', 'g'], xlabel='Episode', ylabel='Profit ratio',
                title='DQN2')
    plt.grid()
    plt.savefig(f'dqn3_profits_avg_{name}')

    ExperimentResult(
        config=train_parameters,
        final_env=None,
        profits=train_profits,
        returns=train_returns,
        loss=train_loss,
        max_possible_profits=None,
        buy_and_hold_profits=None,
        algorithm=f'dqn3_{name}'
    ).to_file()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    main()
