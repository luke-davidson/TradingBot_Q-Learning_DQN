import numpy as np
from easydict import EasyDict

from agents.agents import DQNConfig
from agents.dqn_agent import dqn_agent
from agents.features import history_and_action, seven_indicators
from environment import StocksEnv
from utils.experiment import ExperimentResult, visualize_experiment


def main():
    config = DQNConfig(
        num_episodes=10,
        max_timesteps=200,  # this must equal eps_length
        gamma=0.99,
        epsilon=0.01,
        env=StocksEnv(EasyDict({
            "env_id": 'stocks-dqn', "eps_length": 200,
            "window_size": 200, "train_range": None, "test_range": None,
            "stocks_data_filename": 'STOCKS_GOOGL'
        })),
        features=[history_and_action]
    )
    config.env.seed(0)
    final_env, profits, max_possible_profits, buy_and_hold_profits = dqn_agent(config)
    result = ExperimentResult(
        config=config,
        final_env=final_env,
        profits=profits,
        max_possible_profits=max_possible_profits,
        buy_and_hold_profits=buy_and_hold_profits,
        algorithm='dqn',
    )
    filename = result.to_file()
    visualize_experiment(filename)


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    main()
