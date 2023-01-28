import numpy as np
import tqdm

from agents.agents import AgentConfig
from environment import Action


def random_agent(config: AgentConfig):
    allowed_actions = [Action.BUY, Action.HOLD, Action.SELL]

    profits, max_possible_profits, buy_and_hold_profits = [], [], []

    for _ in tqdm.trange(config.num_episodes):
        state = config.env.reset()

        state_history = [state.history]

        action = Action(np.random.choice(allowed_actions))

        for timestep in range(config.max_timesteps):
            next_state, reward, done, _ = config.env.step(action)
            state_history.append(next_state.history)

            next_action = Action(np.random.choice(allowed_actions))

            if done:
                break
            else:
                state = next_state
                action = next_action

        profits.append(config.env.final_profit())
        max_possible_profits.append(config.env.max_possible_profit())
        buy_and_hold_profits.append(state_history[-1][0][0] - state_history[0][-1][1])

    return config.env, profits, max_possible_profits, buy_and_hold_profits
