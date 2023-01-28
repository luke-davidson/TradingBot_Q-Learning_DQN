import random
from typing import List

import numpy as np
import tqdm

import environment
from agents.agents import RLConfig, FeatureExtractor
from environment import Action, State

Weights = np.ndarray


def semigradient_sarsa(config: RLConfig):
    s = config.env.reset()
    features_length = sum([len(feature(s, Action(config.env.random_action()))) for feature in config.features])
    w: Weights = np.zeros(features_length)

    allowed_actions = [Action.BUY, Action.HOLD, Action.SELL]

    profits = []
    max_possible_profits = []
    buy_and_hold_profits = []

    for _ in tqdm.trange(config.num_episodes):
        # reset the environment
        state = config.env.reset()

        state_history = [state.history]

        # find me an action
        action = epsilon_greedy_action_selection(state=state, action_space=allowed_actions,
                                                 features=config.features, w=w, epsilon=config.epsilon)

        for timestep in range(config.max_timesteps):
            # take the action and observe the next state and reward
            next_state, reward, done, _ = config.env.step(action)
            state_history.append(next_state.history)
            # print(f'{w} | {state.history[-1]} + {Action(action).name} => {reward}')

            # pick the next action epsilon greedily
            next_action = epsilon_greedy_action_selection(state=next_state, action_space=allowed_actions,
                                                          features=config.features, w=w, epsilon=config.epsilon)
            # update the weights
            w = update_weights(w, config.features, done, config.alpha, config.gamma, state, action, reward,
                               next_state, next_action)

            if done:
                break
            else:
                state = next_state
                action = next_action

        profits.append(config.env.final_profit())
        max_possible_profits.append(config.env.max_possible_profit())
        buy_and_hold_profits.append(state_history[-1][0][0] - state_history[0][-1][1])
        # buy at first open, sell at last close

    return config.env, profits, max_possible_profits, buy_and_hold_profits


def compute_g(rewards: List[float], gamma: float):
    """Computes discounted returns from a list of rewards, where the most recent reward is last"""
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
    return G


def update_weights(weights: Weights, features: List[FeatureExtractor], is_terminal: bool, alpha: float, gamma,
                   state: State, action: Action, reward: float, next_state: State, next_action: Action) -> Weights:
    """Computes the new weights using the semi-gradient SARSA approach"""
    gradient_q = compute_feature_vector(state, action, features).T
    q = compute_q_value(state, action, features, weights)

    if is_terminal:
        return weights + alpha * (reward - q) * gradient_q
    else:
        q_next = compute_q_value(next_state, next_action, features, weights)
        update = reward + (gamma * q_next) - q
        print(update)
        return weights + alpha * update * gradient_q


def compute_feature_vector(state: State, action: Action, features: List[FeatureExtractor]) -> np.ndarray:
    return np.hstack([feature(state, action) for feature in features])


def compute_q_value(state: State, action: Action, features: List[FeatureExtractor], w: np.ndarray) -> float:
    feature_vec = compute_feature_vector(state, action, features)
    return feature_vec.T @ w


def epsilon_greedy_action_selection(state: environment.State,
                                    action_space: List[environment.Action],
                                    features: List[FeatureExtractor],
                                    w: Weights,
                                    epsilon: float) -> environment.Action:
    if np.random.rand() < epsilon:  # random
        return Action(np.random.choice(action_space))
    else:  # greedy
        # shuffle list of actions randomly, so to break ties evenly
        actions = sorted(action_space, key=lambda k: random.random())
        q_vals = [compute_q_value(state, action, features, w) for action in actions]
        return actions[np.argmax(q_vals)]
