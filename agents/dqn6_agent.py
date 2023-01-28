from typing import Dict

import numpy as np
import torch
import tqdm
from numba import jit
from torch import nn
from torch.nn import Conv2d, Flatten, LazyLinear, MaxPool2d, ReLU

from environment import TradingEnv
from environment.trading_env import Mode


# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)


class DeepQNet(nn.Module):
    def __init__(self, output_dim):
        super(DeepQNet, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 1)))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2d(kernel_size=(4, 1), stride=(2, 1)))

        self.layers.append(Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 1)))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2d(kernel_size=(4, 1), stride=(2, 1)))

        self.layers.append(Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 1)))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2d(kernel_size=(4, 1), stride=(2, 1)))

        self.layers.append(Flatten())
        self.layers.append(LazyLinear(out_features=64))
        self.layers.append(ReLU())

        self.layers.append(LazyLinear(out_features=output_dim))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            # print(f'Layer {layer} with input size {x.size()}')
            x = layer(x)
        return x


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


class DQN6Agent(object):
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
        self.behavior_policy_net = DeepQNet(output_dim=params['action_dim'])
        # create target network
        self.target_policy_net = DeepQNet(output_dim=params['action_dim'])

        # initialize target network with behavior network
        # self.behavior_policy_net.apply(customized_weights_init)
        # self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

        # send the agent to a specific device: cpu or gpu
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=params['learning_rate'])

    # get action
    def get_action(self, obs, eps):
        if np.random.random() < eps:  # with probability eps, the agent selects a random action
            action = self.action_space.sample()
        else:  # with probability 1 - eps, the agent selects a greedy policy
            obs = self._arr_to_tensor(obs)
            with torch.no_grad():
                q_values = self.behavior_policy_net(obs)
                action = q_values.max(dim=1)[1].item()  # index of the maximum value
        return action

    def update_behavior_policy(self, state, action, next_state, reward, done):
        state_tensor = self._arr_to_tensor(state)
        next_state_tensor = self._arr_to_tensor(next_state)

        q_estimate = self.behavior_policy_net(state_tensor)[0, action].reshape(1)
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

    # update update target policy
    def update_target_policy(self):
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    # load trained model
    def load_model(self, model_file):
        # load the trained model
        self.behavior_policy_net.load_state_dict(torch.load(model_file, map_location=self.device))
        self.behavior_policy_net.eval()

    def to_file(self, model_file):
        torch.save(self.behavior_policy_net.state_dict(), model_file)

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr = np.array(arr)
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor


def evaluate_dqn6_agent(env: TradingEnv, agent: DQN6Agent, params: Dict, verbose=True, name='', save=False):
    profits = []
    progress = tqdm.trange(params['episodes'], disable=not verbose)
    for episode in progress:
        state = env.reset()
        for timestep in range(params['episode_duration']):
            action = agent.get_action(state, eps=0)  # greedy action
            state, reward, done, _ = env.step(action)

            if done:
                break

        profits.append(env.final_profit())
        progress.set_description(
            f"TEST: Ep={episode} | "
            f"Profit={env.final_profit():.3f}"
        )

        if save:
            env.render_together(save=True, filename=f'data/plots/dqn6_{name}/evaluation/episode_{episode}')

    return profits


def train_dqn6_agent(env: TradingEnv, params, model_file: str = None):
    # create the DQN agent
    my_agent = DQN6Agent(params)
    if model_file:
        my_agent.load_model(model_file)

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
    validation_profits = []
    best_validation_profit = -1

    # reset the environment
    obs = env.reset(mode=Mode.Train)

    # start training
    pbar = tqdm.trange(params['total_training_time_step'])
    for t in pbar:
        # scheduled epsilon at time step t
        eps_t = my_schedule.get_value(t)
        # get one epsilon-greedy action
        action = my_agent.get_action(obs, eps_t)

        assert env.mode == Mode.Train
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
            G = compute_return(np.array(rewards), params['gamma'])
            train_returns.append(G)

            final_profit = env.final_profit()
            profits.append(final_profit)
            episode_idx = len(train_returns)

            # compute profits on validation dataset
            env.set_mode(Mode.Validation)
            validation = evaluate_dqn6_agent(env, agent=my_agent, params={
                'episodes': 20,
                'episode_duration': 200,
            }, verbose=False)
            avg_validation_profit = np.mean(validation)
            validation_profits.append(avg_validation_profit)
            if avg_validation_profit > best_validation_profit:
                best_validation_profit = avg_validation_profit
                my_agent.to_file('dqn6_best.pt')

            pbar.set_description(
                f"TRAIN: Ep={episode_idx} | "
                f"G={train_returns[-1] if train_returns else 0:.5f} | "
                f"Training Profit={final_profit:.3f} | "
                f"Validation Profit={avg_validation_profit:.3f} | "
                f"Eps={eps_t:.5f}"
            )

            # reset the environment
            episode_t, rewards = 0, []
            env.close()
            obs = env.reset(mode=Mode.Train)

    # save the results
    return train_returns, train_loss, profits, validation_profits, my_agent


@jit(nopython=True)
def compute_return(rewards, gamma):
    G = 0
    for reward in rewards:
        G = reward + gamma * G
    return G
