import numpy as np
from easydict import EasyDict

from agents.agents import RLConfig
from agents.features import seven_indicators
from agents.semigradient_sarsa_agent import semigradient_sarsa
from environment import StocksEnv
from utils.experiment import ExperimentResult, visualize_experiment


def main():
    config = RLConfig(
        num_episodes=100,
        max_timesteps=500,  # this must equal eps_length
        alpha=0.001,
        gamma=0.99,
        epsilon=0.01,
        env=StocksEnv(EasyDict({
            "env_id": 'stocks-semigradient_sarsa', "eps_length": 500,
            "window_size": 50, "train_range": None, "test_range": None,
            "stocks_data_filename": 'STOCKS_GOOGL'
        })),
        features=[seven_indicators]
    )
    config.env.seed(0)
    final_env, profits, max_possible_profits, buy_and_hold_profits = semigradient_sarsa(config)
    result = ExperimentResult(
        config=config,
        final_env=final_env,
        profits=profits,
        max_possible_profits=max_possible_profits,
        buy_and_hold_profits=buy_and_hold_profits,
        algorithm='semigradient_sarsa',
    )
    filename = result.to_file()
    visualize_experiment(filename)


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    main()
