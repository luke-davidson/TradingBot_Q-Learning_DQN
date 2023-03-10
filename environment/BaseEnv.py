from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, List

import gym

BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])


# for solving multiple inheritance metaclass conflict between gym and ABC
class FinalMeta(type(ABC), type(gym.Env)):
    pass


class BaseEnv(gym.Env, ABC, metaclass=FinalMeta):
    """
    Overview:
        Basic environment class, extended from ``gym.Env``
    Interface:
        ``__init__``, ``reset``, ``close``, ``step``, ``random_action``, ``create_collector_env_cfg``, \
        ``create_evaluator_env_cfg``, ``enable_save_replay``
    """

    @abstractmethod
    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Lazy init, only related arguments will be initialized in ``__init__`` method, and the concrete \
            environment will be initialized the first time ``reset`` method is called.
        Arguments:
            - cfg (:obj:`dict`): Environment configuration in dict type.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        """
        Overview:
            Reset the environment to an initial state and returns an initial observation.
        Returns:
            - obs (:obj:`Any`): Initial observation after reset.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Overview:
            Close environment and all the related resources, it should be called after the usage of environment instance.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> 'BaseEnv.timestep':
        """
        Overview:
            Run one timestep of the environment's dynamics/simulation.
        Arguments:
            - action (:obj:`Any`): The ``action`` input to step with.
        Returns:
            - timestep (:obj:`BaseEnv.timestep`): The result timestep of environment executing one step.
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        """
        Overview:
            Set the seed for this environment's random number generator(s).
        Arguments:
            - seed (:obj:`Any`): Random seed.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Overview:
            Return the information string of this environment instance.
        Returns:
            - info (:obj:`str`): Information of this environment instance, like type and arguments.
        """
        raise NotImplementedError

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in environment manager \
            (a series of vectorized environment), and this method is mainly responsible for envs collecting data.
        Arguments:
            - cfg (:obj:`dict`): Original input environment config, which needs to be transformed into the type of creating \
                environment instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config collector envs.

        .. note::
            Elements(environment config) in collector_env_cfg/evaluator_env_cfg can be different, such as server ip and port.
        """
        collector_env_num = cfg.pop('collector_env_num')
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in environment manager \
            (a series of vectorized environment), and this method is mainly responsible for envs evaluating performance.
        Arguments:
            - cfg (:obj:`dict`): Original input environment config, which needs to be transformed into the type of creating \
                environment instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config evaluator envs.
        """
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [cfg for _ in range(evaluator_env_num)]

    # optional method
    def random_action(self) -> Any:
        """
        Overview:
            Return random action generated from the original action space, usually it is convenient for test.
        Returns:
            - random_action (:obj:`Any`): Action generated randomly.
        """
        pass
