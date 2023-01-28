from easydict import EasyDict

from agents.agents import AgentConfig
from agents.random_agent import random_agent
from environment import StocksEnv
from utils.experiment import ExperimentResult, visualize_experiment


def main():
    config = AgentConfig(
        num_episodes=300,
        max_timesteps=1000,
        env=StocksEnv(EasyDict({
            "env_id": 'stocks-random_agent', "eps_length": 1000,
            "window_size": 300, "train_range": None, "test_range": None,
            "stocks_data_filename": 'STOCKS_GOOGL'
        })),
    )
    config.env.seed(0)
    final_env, profits, max_possible_profits, buy_and_hold_profits = random_agent(config)
    result = ExperimentResult(
        config=config,
        final_env=final_env,
        profits=profits,
        max_possible_profits=max_possible_profits,
        buy_and_hold_profits=buy_and_hold_profits,
        algorithm='random',
    )
    filename = result.to_file()
    visualize_experiment(filename)
    final_env.render()


if __name__ == '__main__':
    main()
