from copy import deepcopy
from typing import Any

import numpy as np

from environment.trading_env import Action, Mode, Position, TradingEnv, load_dataset


class StocksEnv(TradingEnv):

    def __init__(self, cfg):
        super().__init__(cfg)

        # ====== load Google stocks data =======
        raw_data = load_dataset(self._cfg.stocks_data_filename, 'Date')
        self.raw_prices = raw_data.loc[:, 'Close'].to_numpy()
        EPS = 1e-10
        self.df = deepcopy(raw_data)

        # | <--- training ---> | <-- validation --> | <-- test --> |
        # [         ].....................[         ]....................[         ].........................
        #           ^                     ^         ^                    ^         ^                        ^
        #     train start             train end    valid start       valid end     test start           test end
        #
        # | <---------------------------> | <--------------------------> | <------------------------------> |
        # [.................] for any given interval that we want to sample points over:
        # [...]..........[..] we must exclude these two regions
        # window         episode length

        self.train_range = (self.window_size, int(np.floor(self.train_ratio * len(raw_data))))
        self.validation_range = (self.window_size + int(np.ceil(len(self.df) * self.test_ratio)),
                                 int(np.floor(len(self.df) * (self.validation_ratio + self.train_ratio))
                                 - self._cfg.eps_length))
        self.test_range = (self.window_size + int(np.ceil(len(self.df) * (self.validation_ratio + self.train_ratio))),
                           len(self.df) - self._cfg.eps_length)

        assert self.validation_range[1] > self.validation_range[0]
        assert self.test_range[1] > self.test_range[0]

        train_data = raw_data[self.train_range[0]:self.train_range[1]].copy()
        validation_data = raw_data[self.validation_range[0]:self.validation_range[1]].copy()
        test_data = raw_data[self.test_range[0]:self.test_range[1]].copy()

        # normalize datasets individually
        train_data = train_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        train_data -= train_data.min()
        train_data /= train_data.max()
        validation_data = validation_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        validation_data -= validation_data.min()
        validation_data /= validation_data.max()
        test_data = test_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        test_data -= test_data.min()
        test_data /= test_data.max()
        self.df.loc[train_data.index, train_data.columns] = train_data
        self.df.loc[validation_data.index, validation_data.columns] = validation_data
        self.df.loc[test_data.index, test_data.columns] = test_data

        # ======================================

        # set cost
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        # self.trade_fee_bid_percent = 0  # unit
        # self.trade_fee_ask_percent = 0  # unit

    # override
    def _process_data(self, start_idx: int = None) -> Any:
        """
        Overview:
            used by environment.reset(), process the raw data.
        Arguments:
            - start_idx (int): the start tick; if None, then randomly select.
        Returns:
            - prices: the close.
            - signal_features: feature map
            - feature_dim_len: the dimension length of selected feature
        """

        # ====== build feature map ========
        all_feature_name = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        all_feature = {k: self.df.loc[:, k].to_numpy() for k in all_feature_name}
        # add feature "Diff"
        prices = self.df.loc[:, 'Close'].to_numpy()
        diff = np.insert(np.diff(prices), 0, 0)
        all_feature_name.append('Diff')
        all_feature['Diff'] = diff
        # =================================

        # you can select features you want
        selected_feature_name = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        selected_feature = np.column_stack([all_feature[k] for k in selected_feature_name])
        feature_dim_len = len(selected_feature_name)

        if start_idx is None:
            if self.mode == Mode.Train:
                start, end = self.train_range
            elif self.mode == Mode.Validation:
                start, end = self.validation_range
            else:
                assert self.mode == Mode.Test
                start, end = self.test_range
            assert end > start
            self.start_idx = np.random.randint(start, end)
        else:
            self.start_idx = start_idx

        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self._cfg.eps_length - 1

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

    # override
    def max_possible_profit(self) -> float:
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            if self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (2 - (current_price / last_trade_price)) * (1 - self.trade_fee_ask_percent
                                                                                  ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] >= self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (current_price / last_trade_price) * (1 - self.trade_fee_ask_percent
                                                                            ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            last_trade_tick = current_tick - 1

        return profit

    def __repr__(self) -> str:
        return "DI-engine Stocks Trading Env"
