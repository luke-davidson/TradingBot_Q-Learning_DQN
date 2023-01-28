import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import Conv2d, Flatten, LazyLinear, Linear, MaxPool2d, ReLU


# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 1)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 1)


class DeepQNet(nn.Module):
    def __init__(self):
        super(DeepQNet, self).__init__()

        """CODE HERE: construct your Deep neural network
        """

        self.layers = nn.ModuleList()

        self.layers.append(Linear(in_features=1400, out_features=500))
        self.layers.append(ReLU())
        self.layers.append(LazyLinear(out_features=250))
        self.layers.append(ReLU())
        self.layers.append(LazyLinear(out_features=250))
        self.layers.append(ReLU())
        self.layers.append(LazyLinear(out_features=5))

        # self.layers.append(Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1)))
        # self.layers.append(ReLU())
        # self.layers.append(MaxPool2d(kernel_size=(4, 1), stride=(1, 1)))

        # self.layers.append(Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 1)))
        # self.layers.append(ReLU())
        # self.layers.append(MaxPool2d(kernel_size=(4, 1), stride=(2, 1)))

        # self.layers.append(Conv2d(in_channels=1, out_channels=1, kernel_size=(8, 1)))
        # self.layers.append(ReLU())
        # self.layers.append(MaxPool2d(kernel_size=(4, 1), stride=(2, 1)))

        # self.layers.append(Conv2d(in_channels=1, out_channels=1, kernel_size=(16, 1)))
        # self.layers.append(ReLU())
        # self.layers.append(MaxPool2d(kernel_size=(8, 1), stride=(2, 1)))

        # self.layers.append(Flatten())
        # self.layers.append(LazyLinear(out_features=500))
        # self.layers.append(ReLU())

        # self.layers.append(LazyLinear(out_features=5))

    def forward(self, x):
        """CODE HERE: implement your forward propagation
        """
        for idx, layer in enumerate(self.layers):
            # print(f'On layer {idx}({layer.__class__}) with input shape {x.size()}')
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
        """ CODE HERE: return the epsilon for each time step within the duration.
        """
        if time > self._duration:
            return self._end_value
        else:
            return self._start_value + (time / self._duration * (self._end_value - self._start_value))


class DQNAgent(object):
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

        # create behavior and target policy network
        self.behavior_policy_net = DeepQNet()
        self.target_policy_net = DeepQNet()

        # initialize target network with behavior network
        # self.behavior_policy_net.apply(customized_weights_init)
        # self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

        # send the agent to a specific device: cpu or gpu
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
                action = q_values.max(dim=1)[1].item()
        return action

    # update behavior policy
    def update_behavior_policy(self, batch_data):
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
        with torch.no_grad():
            q_max = torch.max(self.target_policy_net(next_obs_tensor), dim=1).values

            td_target = rewards_tensor.flatten() + self.params['gamma'] * q_max
            # done_idxs = (dones_tensor == 1).nonzero(as_tuple=True)[0]
            # td_target[done_idxs] = rewards_tensor[done_idxs, 0]

        # compute the loss
        td_loss = torch.nn.MSELoss()(q_estimate, td_target)
        td_loss.backward()

        # minimize the loss
        self.optimizer.step()
        self.behavior_policy_net.zero_grad()

        return td_loss.item()

    # update target policy
    def update_target_policy(self):
        """CODE HERE:
                Copy the behavior policy network to the target network
        """
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    # load trained model
    def load_model(self, model_file):
        # load the trained model
        self.behavior_policy_net.load_state_dict(torch.load(model_file, map_location=self.device))
        self.behavior_policy_net.eval()

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        # arr = np.array(arr).reshape((-1, 1, arr.shape[0], arr.shape[1]))
        arr_tensor = torch.from_numpy(np.array(arr).reshape(-1)).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        # batch_data_tensor['obs'] = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).reshape(
        #     (-1, 1, obs_arr.shape[1], obs_arr.shape[2]))  # reshape for CNN
        batch_data_tensor['obs'] = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        batch_data_tensor['action'] = torch.as_tensor(action_arr, device=self.device).long().view(-1, 1)
        batch_data_tensor['reward'] = torch.as_tensor(reward_arr, dtype=torch.float32, device=self.device).view(-1, 1)
        # batch_data_tensor['next_obs'] = torch.as_tensor(next_obs_arr, dtype=torch.float32, device=self.device).reshape(
        #     (-1, 1, obs_arr.shape[1], obs_arr.shape[2]))
        batch_data_tensor['next_obs'] = torch.as_tensor(next_obs_arr, dtype=torch.float32, device=self.device)
        batch_data_tensor['done'] = torch.as_tensor(done_arr, dtype=torch.float32, device=self.device).view(-1, 1)

        return batch_data_tensor


def train_dqn_agent(env, params):
    # create the DQN agent
    my_agent = DQNAgent(params)

    # create the epsilon-greedy schedule
    my_schedule = LinearSchedule(start_value=params['epsilon_start_value'],
                                 end_value=params['epsilon_end_value'],
                                 duration=params['epsilon_duration'])

    # create the replay buffer
    replay_buffer = ReplayBuffer(params['replay_buffer_size'])

    # training variables
    episode_t = 0
    rewards = []
    train_returns = []
    train_loss = []
    profits = []

    # reset the environment
    obs = env.reset()

    num_episodes = params['total_training_time_step'] // params['max_time_step_per_episode']

    # start training
    pbar = tqdm.trange(params['total_training_time_step'])
    for t in pbar:
        # scheduled epsilon at time step t
        eps_t = my_schedule.get_value(t)
        # get one epsilon-greedy action
        action = my_agent.get_action(obs, eps_t)

        # step in the environment
        next_obs, reward, done, _ = env.step(action)

        # add to the buffer
        replay_buffer.add(obs, action, reward, next_obs, done)
        rewards.append(reward)

        # check termination
        if done or episode_t == params['max_time_step_per_episode'] - 1:
            # compute the return
            G = 0
            for r in reversed(rewards):
                G = r + params['gamma'] * G

            # store the return
            train_returns.append(G)
            episode_idx = len(train_returns)

            # print the information
            pbar.set_description(
                f"Ep={episode_idx} | "
                f"G={np.mean(train_returns[-10:]) if train_returns else 0:.2f} | "
                f"Eps={eps_t:.5f}"
            )

            # plot the last x episodes
            if episode_idx == 1 or episode_idx > num_episodes - params['final_policy_num_plots']:
                env.render_together(save=True, filename=f'data/plots/dqn2_{params["name"]}/episode_{episode_idx}')

            # reset the environment
            episode_t, rewards = 0, []
            profits.append(env.final_profit())
            env.close()
            obs = env.reset()
        else:
            # increment
            obs = next_obs
            episode_t += 1
            # print(my_agent.behavior_policy_net.layers[0].weight)

        if t > params['start_training_step']:
            # update the behavior model
            if not np.mod(t, params['freq_update_behavior_policy']):
                """ CODE HERE:
                    Update the behavior policy network
                """
                train_loss.append(my_agent.update_behavior_policy(replay_buffer.sample_batch(params['batch_size'])))

            # update the target model
            if not np.mod(t, params['freq_update_target_policy']):
                """ CODE HERE:
                    Update the target policy network
                """
                my_agent.update_target_policy()

    # save the results
    return train_returns, train_loss, profits
