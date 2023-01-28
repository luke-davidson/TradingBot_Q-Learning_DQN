import random
from typing import List

import numpy as np
import tqdm
from keras.api._v2.keras import Model, Sequential
from keras.api._v2.keras.layers import Dense

import environment
from agents.agents import DQNConfig, FeatureExtractor
from environment import Action, Position, State


def create_model(n_features) -> Sequential:
    model = Sequential()
    model.add(Dense(500, input_shape=(1, n_features), activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(40, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    return model


def dqn_agent(config: DQNConfig):
    s = config.env.reset()
    features_length = sum([len(feature(s, Action(config.env.random_action()))) for feature in config.features])
    model = create_model(n_features=features_length)

    allowed_actions = [Action.BUY, Action.HOLD, Action.SELL]

    profits = []
    max_possible_profits = []
    buy_and_hold_profits = []

    for episode in tqdm.tqdm(range(config.num_episodes)):
        state = config.env.reset()

        state_history = [state.history]

        for _ in tqdm.tqdm(range(config.max_timesteps), position=0, leave=True):
            action = epsilon_greedy_action_selection(state=state, action_space=allowed_actions,
                                                     features=config.features, model=model, epsilon=config.epsilon)

            next_state, reward, done, _ = config.env.step(action)
            state_history.append(next_state.history)

            update_model(model=model, features=config.features, is_terminal=done, gamma=config.gamma, state=state,
                         action=action, reward=reward, next_state=next_state, action_space=allowed_actions)

            if done:
                break
            else:
                state = next_state

        profits.append(config.env.final_profit())
        max_possible_profits.append(config.env.max_possible_profit())
        buy_and_hold_profits.append(state_history[-1][0][0] - state_history[0][-1][1])
        # assuming buy at first open, sell at last close

        config.env.render_together(save=True, filename=f'data/plots/dqn/dqn_episode_{episode}_price_profit.png')

    return config.env, profits, max_possible_profits, buy_and_hold_profits


def update_model(model: Sequential, features: List[FeatureExtractor], is_terminal: bool, gamma,
                 state: State, action: Action, reward: float, next_state: State, action_space: List[Action]):
    """Train model using the TD target"""

    # compute state-action feature vector
    x = np.array([[compute_feature_vector(state, action, features)]])
    if is_terminal:
        model.fit(x=x, y=np.array([reward]), verbose=False)
    else:
        # randomize action ordering to break ties
        q_actions = [compute_q_value(next_state, next_action, features, model) for
                     next_action in sorted(action_space, key=lambda k: random.random())]
        max_q = np.mean(q_actions)
        td_target = np.array([reward + gamma * max_q])  # update using TD target
        # print(td_target)
        model.fit(x=x, y=td_target, verbose=False)


def compute_feature_vector(state: State, action: Action, features: List[FeatureExtractor]) -> np.ndarray:
    return np.hstack([feature(state, action) for feature in features])


def compute_q_value(state: State, action: Action, features: List[FeatureExtractor], model: Model) -> float:
    return model.predict(np.array([[compute_feature_vector(state, action, features)]]), verbose=False)[0, 0, 0]


# noinspection DuplicatedCode
def epsilon_greedy_action_selection(state: environment.State,
                                    action_space: List[environment.Action],
                                    features: List[FeatureExtractor],
                                    model: Model,
                                    epsilon: float) -> environment.Action:
    if np.random.rand() < epsilon:  # random
        return Action(np.random.choice(action_space))
    else:  # greedy
        # shuffle list of actions randomly, so to break ties evenly
        actions = sorted(action_space, key=lambda k: random.random())
        q_vals = [compute_q_value(state, action, features, model) for action in actions]
        return actions[np.argmax(q_vals)]
