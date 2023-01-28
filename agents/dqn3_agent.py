import numpy as np
import torch
import tqdm
from numba import jit
from torch import nn

from environment import TradingEnv


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


# %%
class DeepQNet(nn.Module):
    def __init__(self, input_dim, num_hidden_layer, dim_hidden_layer, output_dim):
        super(DeepQNet, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_features=input_dim, out_features=dim_hidden_layer))
        self.layers.append(nn.ReLU())
        for i in range(num_hidden_layer):
            self.layers.append(nn.Linear(in_features=dim_hidden_layer, out_features=dim_hidden_layer))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=dim_hidden_layer, out_features=output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ReplayBuffer(object):
    """ Implement the Replay Buffer as a class, which contains:
            - self._data_buffer (list): a list variable to store all transition tuples.
            - add: a function to add new transition tuple into the buffer
            - sample_batch: a function to sample a batch training data from the Replay Buffer
    """

    def __init__(self, buffer_size):
        """Args:
               buffer_size (int): size of the replay buffer
        """
        # total size of the replay buffer
        self.total_size = buffer_size

        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, reward, next_obs, done):
        # create a tuple
        trans = (obs, act, reward, next_obs, done)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        """ Function to fetch the state, action, reward, next state, and done arrays.

            Args:
                indices (list): list contains the index of all sampled transition tuples.
        """
        # lists for transitions
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = [], [], [], [], []

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d = data
            # store to the list
            obs_list.append(np.array(obs, copy=False))
            actions_list.append(np.array(act, copy=False))
            rewards_list.append(np.array(reward, copy=False))
            next_obs_list.append(np.array(next_obs, copy=False))
            dones_list.append(np.array(d, copy=False))
        # return the sampled batch data as numpy arrays
        return np.array(obs_list), np.array(actions_list), np.array(rewards_list), np.array(next_obs_list), np.array(
            dones_list)

    def sample_batch(self, batch_size):
        """ Args:
                batch_size (int): size of the sampled batch data.
        """
        # sample indices with replaced
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)


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
        self.behavior_policy_net = DeepQNet(input_dim=params['observation_dim'],
                                            num_hidden_layer=params['hidden_layer_num'],
                                            dim_hidden_layer=params['hidden_layer_dim'],
                                            output_dim=params['action_dim'])
        # create target network
        self.target_policy_net = DeepQNet(input_dim=params['observation_dim'],
                                          num_hidden_layer=params['hidden_layer_num'],
                                          dim_hidden_layer=params['hidden_layer_dim'],
                                          output_dim=params['action_dim'])

        # initialize target network with behavior network
        self.behavior_policy_net.apply(customized_weights_init)
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

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


def train_dqn3_agent(env: TradingEnv, params):
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
