from typing import List

import numpy as np
from numba import njit
from numpy import ndarray

from environment import Action, State


def moving_average(window: int, state: State) -> float:
    # average the `window` most recent close prices
    return np.average(state.history[-window:, 0])


def on_balance_volume(state: State) -> float:
    # inefficient, ideally we'd like to use the previous value of obv
    # https://www.investopedia.com/terms/o/onbalancevolume.asp
    obv = 0
    for i in range(1, len(state.history) - 1):
        close = state.history[i][0]
        close_prev = state.history[i - 1][0]
        if close > close_prev:
            obv += state.history[i][5]
        elif close < close_prev:
            obv -= state.history[i][5]
    return obv


def accumulation_distribution(state: State) -> float:
    # https://www.investopedia.com/terms/a/accumulationdistribution.asp
    return accumulation_distribution_worker(state.history)


@njit
def accumulation_distribution_worker(history: np.ndarray) -> float:
    ad = 0
    for i in range(len(history)):
        close = history[i][0]
        low = history[i][3]
        high = history[i][2]
        volume = history[i][5]

        mfm = ((close - low) - (high - close)) / (high - low)
        mfv = mfm * volume
        ad += mfv
    return ad


def avg_directional_idx(state: State):
    ...


def aroon(state: State):
    # https://www.investopedia.com/terms/a/aroonoscillator.asp
    periods_since_high = 25 - np.argmax(state.history[-25:, 0])
    period_since_low = 25 - np.argmax(state.history[-25:, 0])
    up = 100 * (25 - periods_since_high) / 25
    down = 100 * (25 - period_since_low) / 25
    return up - down


def macd(state: State):
    return ema(state.history[-12:, 0], 12)[-1] - ema(state.history[-26:, 0], 26)[-1]


def ema(prices, days, smoothing=2):
    ema = [sum(prices[:days]) / days]
    for price in prices[days:]:
        ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
    return ema


# def ema(s, n):
#     """
#     returns an n period exponential moving average for
#     the time series s
#
#     s is a list ordered from oldest (index 0) to most
#     recent (index -1)
#     n is an integer
#
#     returns a numeric array of the exponential
#     moving average
#     """
#     ema = []
#     j = 1
#
#     # get n sma first and calculate the next n period ema
#     sma = sum(s[:n]) / n
#     multiplier = 2 / float(1 + n)
#     ema.append(sma)
#
#     # EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
#     ema.append(((s[n] - sma) * multiplier) + sma)
#
#     # now calculate the rest of the values
#     for i in s[n + 1:]:
#         tmp = ((i - ema[j]) * multiplier) + ema[j]
#         j = j + 1
#         ema.append(tmp)
#
#     return ema


def stochastic_oscillator(state: State) -> float:
    # https://www.investopedia.com/terms/s/stochasticoscillator.asp
    c = state.history[-1][0]
    low_14 = min(state.history[-14:, 3])
    high_14 = max(state.history[-14:, 2])
    return float((c - low_14) / (high_14 - low_14))


def moving_averages_with_state(windows: List[int], state: State, action: Action) -> ndarray:
    averages = [moving_average(w, state) for w in windows]
    action_onehot = np.zeros(shape=len(Action))
    action_onehot[action.value] = 1
    return np.hstack([averages, action_onehot])


def seven_indicators(state, action) -> np.ndarray:
    action_onehot = np.zeros(shape=len(Action))
    action_onehot[action.value] = 1

    vec = np.hstack([np.array([on_balance_volume(state),
                               accumulation_distribution(state),
                               aroon(state),
                               macd(state),
                               stochastic_oscillator(state)]),
                     action_onehot])
    # print(vec)
    return vec


def history_and_action(state: State, action: Action) -> np.ndarray:
    action_onehot = np.zeros(shape=len(Action))
    action_onehot[action.value] = 1

    position_history = np.zeros(shape=(state.history.shape[0]))
    position_history[0:len(state.position_history)] = state.position_history

    vec = np.hstack([state.history.flatten(), position_history, action_onehot])
    return vec
