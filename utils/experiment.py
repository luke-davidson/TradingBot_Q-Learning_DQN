import dataclasses
import time
from datetime import datetime
from typing import List, Optional

import dill as pickle
import numpy as np
from dateutil import tz
from matplotlib import pyplot as plt

from agents.agents import AgentConfig
from environment import TradingEnv


@dataclasses.dataclass
class ExperimentResult:
    config: AgentConfig
    final_env: Optional[TradingEnv]
    profits: List[float]
    returns: Optional[List[float]]
    loss: Optional[List[float]]
    max_possible_profits: Optional[List[float]]
    buy_and_hold_profits: Optional[List[float]]
    algorithm: str
    timestamp: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.timestamp = datetime.utcfromtimestamp(time.time()).replace(tzinfo=tz.gettz('UTC')) \
            .astimezone(tz=tz.gettz('America/Boston')).strftime('%Y-%m-%d--%H-%M-%S')

    def to_file(self):
        with open(self.pickle_filename(), 'wb') as f:
            pickle.dump(self, f)
            return self.pickle_filename()

    def pickle_filename(self) -> str:
        return f'data/{self.algorithm}_{self.timestamp}.pickle'

    def name(self) -> str:
        return f'{self.algorithm}_{self.timestamp}'

    @staticmethod
    def from_file(path: str) -> 'ExperimentResult':
        with open(path, 'rb') as handle:
            return pickle.load(handle)


def visualize_experiment(filename: str):
    r = ExperimentResult.from_file(filename)
    plot_profits(r)
    r.final_env.render_together(save=True, filename='data/plots/' + str(r.name()) + "-price-profits.png")


def plot_profits(r: ExperimentResult):
    plt.plot(range(0, len(r.profits)), r.profits, 'blue', label='Agent profit')
    plt.plot(range(0, len(r.buy_and_hold_profits)), r.buy_and_hold_profits, 'green', label='Buy and hold profit')

    plt.title(f'Profits per training episode. Avg={np.average(r.profits)}')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.legend()
    plt.savefig('data/plots/' + str(r.name()) + "-profits.png")
