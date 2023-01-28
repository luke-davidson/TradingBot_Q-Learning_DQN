import random
import sys

import numpy as np
import torch
from easydict import EasyDict
from matplotlib import pyplot as plt

from agents.dqn6_agent import DQN6Agent, evaluate_dqn6_agent, train_dqn6_agent
from environment import Position, StocksEnv
from environment.trading_env import Mode
from utils.experiment import ExperimentResult
from utils.plotting import plot_curves

position_values = [p.value for p in list(Position)]


class StocksEnvWithFeatureVectors(StocksEnv):

    def _get_observation(self) -> np.ndarray:
        observation = super()._get_observation()
        position_history = np.zeros(shape=(observation.history.shape[0]))
        position_history[-len(observation.position_history):] = observation.position_history
        position_history_onehot = np.zeros(shape=(len(position_history), len(position_values)))
        position_history_onehot[range(len(position_history)),
                                (position_history - np.min(position_values)).astype(int)] = 1
        stacked = np.hstack([observation.history, position_history_onehot])
        # return stacked.reshape((1, stacked.shape[0], stacked.shape[1], 1))
        return stacked.reshape((-1, 1, stacked.shape[0], stacked.shape[1]))


def main():
    name = ''
    if len(sys.argv) > 1:
        name = sys.argv[1]

    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    env = StocksEnvWithFeatureVectors(EasyDict({
        "env_id": 'stocks-dqn', "eps_length": 200,
        "window_size": 200, "train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15,
        "stocks_data_filename": 'DIA', "mode": Mode.Train
    }))

    initial_obs = env.reset()

    # create training parameters
    train_parameters = {
        'observation_dim': len(initial_obs),
        'action_dim': env.action_space.n,
        'action_space': env.action_space,
        'gamma': 1,

        'max_time_step_per_episode': 200,

        'total_training_time_step': 100_000,

        'epsilon_start_value': 1.0,
        'epsilon_end_value': 0.001,
        'epsilon_duration': 80_000,

        'freq_update_target_policy': 2_000,

        'learning_rate': 5e-4,

        'model_name': "stocks_dow.pt",
        'name': name
    }

    # create experiment
    train_returns, train_loss, train_profits, validation_profits, agent = train_dqn6_agent(env, train_parameters)
    # model_file='dqn6_best.pt')

    plot_curves([np.array([train_returns])], ['dqn'], ['r'], xlabel='Episode', ylabel='Discounted return',
                title='DQN with Feedforward NN: Training set')
    plt.savefig(f'dqn6_returns_{name}')
    plt.clf()
    plot_curves([np.array([train_loss])], ['dqn'], ['r'], xlabel='Episode', ylabel='Loss',
                title='DQN with Feedforward NN: Training Set')
    plt.savefig(f'dqn6_loss_{name}')
    plt.clf()
    plot_curves([np.array([train_profits]), np.array([(moving_average(train_profits, n=20))])],
                ['Training Profits', '20-episode moving average'], ['r', 'g'], xlabel='Episode',
                ylabel='Profit ratio', title='DQN with Feedforward NN: Training Profits')
    plt.grid()
    plt.savefig(f'dqn6_profits_train_{name}')
    plt.clf()
    plot_curves([np.array([validation_profits]), np.array([(moving_average(validation_profits, n=20))])],
                ['Validation Profits', '20-episode moving average'], ['r', 'g'], xlabel='Episode',
                ylabel='Profit ratio', title='DQN with Feedforward NN: Validation Profits')
    plt.grid()
    plt.savefig(f'dqn6_profits_validation_{name}')

    ExperimentResult(
        config=train_parameters,
        final_env=None,
        profits=train_profits,
        returns=train_returns,
        loss=train_loss,
        max_possible_profits=None,
        buy_and_hold_profits=None,
        algorithm=f'dqn6_{name}'
    ).to_file()

    agent.to_file(f'{name}_model.pt')

    best_agent = DQN6Agent(train_parameters)
    best_agent.load_model('dqn6_best.pt')

    env.set_mode(Mode.Test)
    test_profits = evaluate_dqn6_agent(env=env, agent=best_agent, params={
        'episodes': 200,
        'episode_duration': 200,
    }, name=train_parameters['name'], save=True)
    plot_curves([np.array([test_profits])],
                ['Test time profits'], ['r'], xlabel='Episode', ylabel='Profit ratio',
                title='DQN with Feedforward NN: Test Set')
    plt.grid()
    plt.axhline(y=np.mean(test_profits), color='r', linestyle='--', label=f'Average Profit Ratio: '
                                                                          f'{np.mean(test_profits):.4f}')
    plt.legend()
    plt.savefig(f'dqn6_profits_test_{name}')


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    main()
