import dataclasses
import os
from collections import namedtuple
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, List

import gym
import numpy as np
import pandas as pd
import torch
import tqdm
from gym import spaces
from matplotlib import pyplot as plt
from numba import jit


def load_dataset(name, index_name):
    # base_dir = os.path.dirname(os.path.abspath('STOCKS_GOOGL.csv'))
    base_dir = os.path.dirname(os.path.abspath('FOREX_EURUSD_1H_ASK.csv'))
    path = os.path.join(base_dir, name + '.csv')
    assert os.path.exists(
        path
    ), "You need to put the stock data under the \'environment/envs/data\' folder.\n \
        if using StocksEnv, you can download Google stocks data at \
        https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/datasets/data/STOCKS_GOOGL.csv"

    return pd.read_csv(path, parse_dates=True, index_col=index_name)


class Action(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


class Position(int, Enum):
    SHORT = -1.
    FLAT = 0.
    LONG = 1.


@dataclasses.dataclass
class State:
    avg_200: float
    avg_50: float
    current_position: int


def transform(position: Position, action: Action) -> Any:
    """
    Overview:
        used by environment.step().
        This func is used to transform the environment's position from
        the input (position, action) pair according to the status machine.
    Arguments:
        - position(Positions) : Long, Short or Flat
        - action(int) : Double_Sell, Sell, Hold, Buy, Double_Buy
    Returns:
        - next_position(Positions) : the position after transformation.
    """
    if action == Action.SELL:
        if position == Position.LONG:
            return Position.FLAT, False

        if position == Position.FLAT:
            return Position.SHORT, True

    if action == Action.BUY:
        if position == Position.SHORT:
            return Position.FLAT, False

        if position == Position.FLAT:
            return Position.LONG, True

    if action == Action.DOUBLE_SELL and (position == Position.LONG or position == Position.FLAT):
        return Position.SHORT, True

    if action == Action.DOUBLE_BUY and (position == Position.SHORT or position == Position.FLAT):
        return Position.LONG, True

    return position, False


class StocksEnv:
    Timestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])

    def __init__(self, cfg):
        self._cfg = cfg
        self._env_id = cfg.env_id
        # ======== param to plot =========
        self.cnt = 0

        if 'plot_freq' not in self._cfg:
            self.plot_freq = 10
        else:
            self.plot_freq = self._cfg.plot_freq
        if 'save_path' not in self._cfg:
            self.save_path = 'data/plots/'
        else:
            self.save_path = self._cfg.save_path
        # ================================

        self.train_range = cfg.train_range
        self.test_range = cfg.test_range
        self.window_size = cfg.window_size
        self.prices = None
        self.signal_features = None
        self.feature_dim_len = None
        self.shape = (cfg.window_size, 3)

        # ======== param about episode =========
        self._start_tick = 0
        self._end_tick = 0
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        # ======================================

        self._init_flag = True
        # init the following variables variable at first reset.
        self._action_space = None
        self._observation_space = None
        self._reward_space = None

        self._profit_history = [1.]
        # ====== load Google stocks data =======
        raw_data = load_dataset(self._cfg.stocks_data_filename, 'Date')
        self.raw_prices = raw_data.loc[:, 'Close'].to_numpy()
        EPS = 1e-10
        self.df = deepcopy(raw_data)
        if self.train_range is None or self.test_range is None:
            self.df = self.df.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        else:
            boundary = int(len(self.df) * self.train_range)
            train_data = raw_data[:boundary].copy()
            boundary = int(len(raw_data) * (1 + self.test_range))
            test_data = raw_data[boundary:].copy()

            train_data = train_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
            test_data = test_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
            self.df.loc[train_data.index, train_data.columns] = train_data
            self.df.loc[test_data.index, test_data.columns] = test_data
        # ======================================

        self.num_states = cfg.num_states

        # set costs
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    # override
    def _process_data(self, start_idx: int = None) -> Any:
        # ====== build feature map ========
        all_feature_name = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        all_feature = {k: self.df.loc[:, k].to_numpy() for k in all_feature_name}
        prices = self.df.loc[:, 'Close'].to_numpy()
        # =================================

        # select features you want
        selected_feature_name = ['Close']
        selected_feature = np.column_stack([all_feature[k] for k in selected_feature_name])
        feature_dim_len = len(selected_feature_name)

        # validate index
        if start_idx is None:
            if self.train_range is None or self.test_range is None:
                self.start_idx = np.random.randint(self.window_size, len(self.df) - self._cfg.eps_length)
            elif self._env_id[-1] == 'e':
                boundary = int(len(self.df) * (1 + self.test_range))
                assert len(
                    self.df) - self._cfg.eps_length > boundary + self.window_size, "parameter test_range is too large!"
                self.start_idx = np.random.randint(boundary + self.window_size, len(self.df) - self._cfg.eps_length)
            else:
                boundary = int(len(self.df) * self.train_range)
                assert boundary - self._cfg.eps_length > self.window_size, "parameter test_range is too small!"
                self.start_idx = np.random.randint(self.window_size, boundary - self._cfg.eps_length)
        else:
            self.start_idx = start_idx

        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self._cfg.eps_length - 1
        self.min_price = min(prices)
        self.max_price = max(prices)

        return prices, selected_feature, feature_dim_len

    # override
    def _calculate_reward(self, action: Action) -> float:
        step_reward = 0.
        current_price = (self.raw_prices[self._current_tick])
        last_trade_price = (self.raw_prices[self._last_trade_tick])
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent))

        if action == Action.BUY and self._position == Position.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Action.SELL and self._position == Position.LONG:
            step_reward = np.log(ratio) + cost

        if action == Action.DOUBLE_SELL and self._position == Position.LONG:
            step_reward = np.log(ratio) + cost

        if action == Action.DOUBLE_BUY and self._position == Position.SHORT:
            step_reward = np.log(2 - ratio) + cost

        step_reward = float(step_reward)

        return step_reward

    def reset(self, start_idx: int = None) -> State:
        self.cnt += 1
        self.prices, self.signal_features, self.feature_dim_len = self._process_data(start_idx)
        if self._init_flag:
            self.shape = (self.window_size, self.feature_dim_len)
            self._action_space = spaces.Discrete(len(Action))
            self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
            self._reward_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
            self._init_flag = False
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Position.FLAT
        self._position_history = [self._position]
        self._profit_history = [1.]
        self._total_reward = 0.

        return self._get_observation()

    def random_action(self) -> Any:
        return np.array([self.action_space.sample()])

    def position_history(self, n) -> List[float]:
        # Returns the n most recent positions
        if n < len(self._position) - 1:
            return self._position
        else:
            return self._position[-n:]

    def step(self, action: Action) -> Timestep:
        self._done = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._position, trade = transform(self._position, action)

        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        self._profit_history.append(float(np.exp(self._total_reward)))
        observation = self._get_observation()
        return self.Timestep(observation, step_reward, self._done, None)

    def _get_observation(self) -> State:
        obs = np.array(self.signal_features[(self._current_tick - self.window_size + 1):
                                            self._current_tick + 1]).astype(np.float32)

        return State(
            current_position=self._position,
            avg_200=np.mean(obs[-200:])[0],
            avg_50=np.mean(obs[-50:])[0]
        )

    def _discretize_state(self, State) -> State:
        ...

    def final_profit(self) -> float:
        return self._profit_history[-1]

    def render_together(self, save=True, filename: str = None) -> None:
        fig, axs = plt.subplots(2)

        axs[0].set_xlabel('trading days')
        axs[0].set_ylabel('close price')
        window_ticks = np.arange(len(self._position_history))
        eps_price = self.raw_prices[self._start_tick:self._end_tick + 1]
        axs[0].plot(eps_price)

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Position.SHORT:
                short_ticks.append(tick)
            elif self._position_history[i] == Position.LONG:
                long_ticks.append(tick)
            else:
                flat_ticks.append(tick)

        axs[0].plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        axs[0].plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        axs[0].plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        axs[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

        # profit history below
        axs[1].set_xlabel('trading days')
        axs[1].set_ylabel('profit')
        axs[1].plot(self._profit_history)

        if save:
            if not filename:
                filename = self.save_path + str(self._env_id) + '-price-profit.png'

            Path(filename[:filename.rindex('/')]).mkdir(parents=True, exist_ok=True)
            plt.savefig(filename)

    def close(self):
        plt.close()

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space


class LinearSchedule(object):
    """ This schedule returns the value linearly"""

    def __init__(self, start_value, end_value, duration):
        # start value
        self._start_value = start_value
        # end value
        self._end_value = end_value
        # time steps that value changes from the start value to the end value
        self._duration = duration
        # difference between the start value and the end value
        self._schedule_amount = end_value - start_value

    def get_value(self, time):
        # logic: if time > duration, use the end value, else use the scheduled value
        if time > self._duration:
            return self._end_value
        else:
            return self._start_value + (time / self._duration * (self._end_value - self._start_value))


class Policy:
    def __init__(self, num_states, num_actions):
        self.q_vals = {}

        for i in range(num_states):
            self.q_vals[i] = np.zeros(num_actions)

    def update_policy(self, state, action, reward, next_state):
        ...


class DQN3Agent(object):
    # initialize the agent
    def __init__(self,
                 params,
                 ):
        # save the parameters
        self.params = params

        # environment parameters
        self.action_dim = params['action_dim']
        self.obs_dim = params['observation_dim']

        # executable actions
        self.action_space = params['action_space']

        # create behavior policy network
        self.behavior_policy_net = Policy(num_states=params['num_states'], num_actions=params['num_actions'])
        # create target network
        self.target_policy_net = Policy(num_states=params['num_states'], num_actions=params['num_actions'])

    # get action
    def get_action(self, obs, eps):
        if np.random.random() < eps:  # with probability eps, the agent selects a random action
            action = self.action_space.sample()
            # print(f'Random action: {action}')
        else:  # with probability 1 - eps, the agent selects a greedy policy
            obs = self._arr_to_tensor(obs).view(1, -1)
            with torch.no_grad():
                q_values = self.behavior_policy_net(obs)
                action = q_values.max(dim=1)[1].item()  # index of the maximum value
                # print(f'NN predicted action: {action}')
        return action

    def update_behavior_policy(self, state, action, next_state, reward, done):
        state_tensor = self._arr_to_tensor(state)
        next_state_tensor = self._arr_to_tensor([next_state])

        q_estimate = self.behavior_policy_net(state_tensor)[action].reshape(1)
        with torch.no_grad():
            if done:
                td_target = torch.as_tensor(reward).reshape(1)
            else:
                q_max = torch.max(self.target_policy_net(next_state_tensor), dim=1).values
                td_target = reward + self.params['gamma'] * q_max

        td_loss = torch.nn.MSELoss()(q_estimate, td_target)

        self.behavior_policy_net.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

    # update behavior policy
    def update_behavior_policy_batch(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        """CODE HERE:
                Compute the predicted Q values using the behavior policy network
        """
        q_estimate = self.behavior_policy_net(obs_tensor).gather(dim=1, index=actions_tensor).flatten()
        q_max = torch.max(self.target_policy_net(next_obs_tensor), dim=1).values
        td_target = rewards_tensor.flatten() + self.params['gamma'] * q_max

        # done_idxs = (dones_tensor == 1).nonzero(as_tuple=True)[0]
        # td_target[done_idxs] = rewards_tensor[done_idxs, 0]

        # compute the loss
        td_loss = torch.nn.MSELoss()(q_estimate, td_target)

        # minimize the loss
        self.behavior_policy_net.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

    # update update target policy
    def update_target_policy(self):
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    # load trained model
    def load_model(self, model_file):
        # load the trained model
        self.behavior_policy_net.load_state_dict(torch.load(model_file, map_location=self.device))
        self.behavior_policy_net.eval()

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr = np.array(arr)
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        return batch_data_tensor


def train_dqn3_agent(env: StocksEnv, params):
    # create the DQN agent
    my_agent = DQN3Agent(params)

    # create the epsilon-greedy schedule
    my_schedule = LinearSchedule(start_value=params['epsilon_start_value'],
                                 end_value=params['epsilon_end_value'],
                                 duration=params['epsilon_duration'])

    num_episodes = params['total_training_time_step'] // params['max_time_step_per_episode']

    # training variables
    episode_t = 0
    rewards = []
    train_returns = []
    train_loss = []
    profits = []

    # reset the environment
    obs = env.reset(200)
    # print(obs)

    # start training
    pbar = tqdm.trange(params['total_training_time_step'])
    for t in pbar:
        # scheduled epsilon at time step t
        eps_t = my_schedule.get_value(t)
        # get one epsilon-greedy action
        action = my_agent.get_action(obs, eps_t)

        # step in the environment
        next_obs, reward, done, _ = env.step(action)
        train_loss.append(my_agent.update_behavior_policy(state=obs, action=action, reward=reward,
                                                          next_state=next_obs, done=done))

        rewards.append(reward)

        # update the target model
        if not np.mod(t, params['freq_update_target_policy']):
            # Update the target policy network
            my_agent.update_target_policy()

        if not done and episode_t != params['max_time_step_per_episode'] - 1:
            # increment
            obs = next_obs
            episode_t += 1
        else:  # we are done
            # compute the return
            G = compute_return(rewards, params['gamma'])
            # print(rewards)

            # store the return
            train_returns.append(G)
            profits.append(env.final_profit())
            episode_idx = len(train_returns)

            # print the information
            pbar.set_description(
                f"Ep={episode_idx} | "
                f"G={train_returns[-1] if train_returns else 0:.5f} | "
                f"Profit={env.final_profit():.3f} | "
                f"Eps={eps_t:.5f}"
            )

            # plot the last x episodes
            if episode_idx == 1 or episode_idx > num_episodes - params['final_policy_num_plots']:
                env.render_together(save=True, filename=f'data/plots/dqn3_{params["name"]}/episode_{episode_idx}')

            # reset the environment
            episode_t, rewards = 0, []
            obs = env.reset(200)

    # save the results
    return train_returns, train_loss, profits


@jit(nopython=True)
def compute_return(rewards, gamma):
    G = 0
    for reward in rewards:
        G = reward + gamma * G
    return G
