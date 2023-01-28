from dataclasses import dataclass
from typing import Callable, List

import numpy as np

import environment
from environment import Action, State

FeatureExtractor = Callable[[State, Action], np.ndarray]


@dataclass
class AgentConfig:
    env: environment.TradingEnv
    num_episodes: int
    max_timesteps: int


@dataclass
class RLConfig(AgentConfig):
    alpha: float
    gamma: float
    epsilon: float
    features: List[FeatureExtractor]


@dataclass
class DQNConfig(AgentConfig):
    gamma: float
    epsilon: float
    features: List[FeatureExtractor]
