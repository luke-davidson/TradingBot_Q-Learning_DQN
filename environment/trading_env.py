import dataclasses
import os
from abc import abstractmethod
from cmath import inf
from enum import Enum
from pathlib import Path
from typing import Any, List

import gym
import numpy as np
import pandas as pd
from easydict import EasyDict
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt

from environment.BaseEnv import BaseEnv, BaseEnvTimestep


def load_dataset(name, index_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
    assert os.path.exists(
        path
    ), "You need to put the stock data under the \'environment/envs/data\' folder.\n \
        if using StocksEnv, you can download Google stocks data at \
        https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/datasets/data/STOCKS_GOOGL.csv"

    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    return df


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


class Mode(Enum):
    Train = 0
    Validation = 1
    Test = 2


@dataclasses.dataclass
class State:
    history: np.ndarray
    position_history: List[Position]
    tick: float


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


class TradingEnv(BaseEnv):

    def __init__(self, cfg: EasyDict) -> None:
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

        assert cfg.train_ratio + cfg.validation_ratio + cfg.test_ratio == 1.0
        self.train_ratio = cfg.train_ratio if hasattr(cfg, 'train_ratio') else 1
        self.validation_ratio = cfg.validation_ratio if hasattr(cfg, 'validation_ratio') else 0
        self.test_ratio = cfg.test_ratio if hasattr(cfg, 'test_ratio') else 0
        self.mode = cfg.mode if hasattr(cfg, 'mode') else Mode.Train

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

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        self.np_random, seed = seeding.np_random(seed)

    def set_mode(self, mode: Mode):
        self.mode = mode

    def reset(self, start_idx: int = None, mode: Mode = None) -> State:
        self.cnt += 1
        self.mode = mode if mode else self.mode  # keep same if no change
        self.prices, self.signal_features, self.feature_dim_len = self._process_data(start_idx)
        if self._init_flag:
            self.shape = (self.window_size, self.feature_dim_len)
            self._action_space = spaces.Discrete(len(Action))
            self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
            self._reward_space = gym.spaces.Box(-inf, inf, shape=(1,), dtype=np.float32)
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

    def step(self, action: Action) -> BaseEnvTimestep:
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
        info = dict(
            total_reward=self._total_reward,
            position=self._position.value,
        )

        if self._done:
            if self._env_id[-1] == 'e' and self.cnt % self.plot_freq == 0:
                self.render()
            info['max_possible_profit'] = np.log(self.max_possible_profit())
            info['final_eval_reward'] = self._total_reward

        return BaseEnvTimestep(observation, step_reward, self._done, info)

    def _get_observation(self) -> State:
        obs = np.array(self.signal_features[(self._current_tick - self.window_size + 1):
                                            self._current_tick + 1]).astype(np.float32)

        tick = (self._current_tick - self._last_trade_tick) / self._cfg.eps_length

        return State(history=obs,
                     position_history=self._position_history[-min(len(self._position_history), self.window_size):],
                     tick=tick)

    def render_profit(self, save=False):
        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('profit')
        plt.plot(self._profit_history)
        if save:
            plt.savefig(self.save_path + str(self._env_id) + "-profit.png")
        else:
            plt.show()

    def render_price(self, save=False):
        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('close price')
        window_ticks = np.arange(len(self._position_history))
        eps_price = self.raw_prices[self._start_tick:self._end_tick + 1]
        plt.plot(eps_price)

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

        plt.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        plt.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        plt.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        if save:
            plt.savefig(self.save_path + str(self._env_id) + '-price.png')
        else:
            plt.show()

    def final_profit(self) -> float:
        return self._profit_history[-1]

    def render(self, save=True) -> None:
        self.render_profit(save)
        self.render_price(save)

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

        if not filename:
            filename = self.save_path + str(self._env_id) + '-price-profit.png'

        Path(filename[:filename.rindex('/')]).mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()

    @abstractmethod
    def _process_data(self, start_idx: int = None):
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self, action):
        raise NotImplementedError

    @abstractmethod
    def max_possible_profit(self):
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Trading Env"
