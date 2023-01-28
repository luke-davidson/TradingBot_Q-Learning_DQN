import random
import sys

import numpy as np
import torch
from easydict import EasyDict
from matplotlib import pyplot as plt

from agents.dqn2_agent import train_dqn_agent
from environment import StocksEnv
from utils.experiment import ExperimentResult
from utils.plotting import plot_curves


class StocksEnvWithFeatureVectors(StocksEnv):

    def _get_observation(self) -> np.ndarray:
        observation = super()._get_observation()
        position_history = np.zeros(shape=(observation.history.shape[0], 1))
        position_history[-len(observation.position_history):, 0] = observation.position_history

        # gives a warning because it's not a state but shut the up
        # align the position history with where the stock was at that point
        return np.hstack([observation.history, position_history])


def main():
    name = sys.argv[1]

    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    env = StocksEnvWithFeatureVectors(EasyDict({
        "env_id": 'stocks-dqn', "eps_length": 200,
        "window_size": 200, "train_range": None, "test_range": None,
        "stocks_data_filename": 'STOCKS_GOOGL'
    }))

    # print('Length: ', len(env.df))
    # exit()

    initial_obs = env.reset()
    # np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)
    # print(repr(initial_obs[:50]))
    # exit()

    # create training parameters
    train_parameters = {
        'observation_dim': initial_obs.shape,
        'action_dim': 5,
        'action_space': env.action_space,
        'hidden_layer_num': 4,
        'hidden_layer_dim': 128,
        'gamma': 0.99,

        'max_time_step_per_episode': 200,

        'total_training_time_step': 500_000 // 10,

        'epsilon_start_value': 0.3,
        'epsilon_end_value': 0.00,
        'epsilon_duration': 250_000 // 10,

        'replay_buffer_size': 50000 // 10,
        'start_training_step': 2000 // 10,
        'freq_update_behavior_policy': 4,
        'freq_update_target_policy': 2000,

        'batch_size': 64,
        'learning_rate': 1e-3,

        'final_policy_num_plots': 20,

        'model_name': "stocks_google.pt",
        'name': name
    }

    # create experiment
    train_returns, train_loss, train_profits = train_dqn_agent(env, train_parameters)
    plot_curves([np.array([train_returns])], ['dqn'], ['r'], 'discounted return', 'DQN2')
    plt.savefig(f'dqn2_returns_{name}')
    plt.clf()
    plot_curves([np.array([train_loss])], ['dqn'], ['r'], 'training loss', 'DQN2')
    plt.savefig(f'dqn2_loss_{name}')
    plt.clf()
    # plot_curves([np.array([train_profits])], ['dqn'], ['r'], 'profit', 'DQN2')
    # plt.savefig(f'dqn2_profits_{name}')

    plot_curves([np.array([train_profits]), np.array([(moving_average(train_profits, n=50))])],
                ['raw profits', '50-episode moving average'], ['r', 'g'], 'profit', 'DQN2')
    plt.grid()
    plt.savefig(f'dqn2_profits_avg_{name}')

    ExperimentResult(
        config=train_parameters,
        final_env=None,
        profits=train_profits,
        returns=train_returns,
        loss=train_loss,
        max_possible_profits=None,
        buy_and_hold_profits=None,
        algorithm='dqn2_name'
    ).to_file()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    main()
