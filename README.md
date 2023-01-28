# Stock Trading with Reinforcement Learning

Marc Bacvanski and Luke Davidson

### Project Structure

| Folder        | Contents                                                                                                                                                                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agents`      | Code for each of the agents, specifically code to train and execute them                                                                                                                                                                                 |
| `data`        | Output data we collected from each of our experiments. These are mostly `.pickle` files, but there exist scripts to parse them.                                                                                                                          |
| `environment` | The trading environment shared by our agents. This is a heavily modified version of [DI-engine's version](https://github.com/opendilab/DI-engine/tree/main/dizoo/gym_anytrading/envs) of the [AnyTrading](https://github.com/AminHP/gym-anytrading) gym. |
| `examples`    | Some jupyter notebooks demonstrating the usage of the environments                                                                                                                                                                                       |
| `graphs`      | Collection of graphs and chart outputs from our experiments                                                                                                                                                                                              |
| `scratch`     | Scratchpaper? Jupyter notebooks? Nothing super interesting here                                                                                                                                                                                          |
| `utils`       | Utility methods to collect, save, and plot data                                                                                                                                                                                                          |

In the root folder exist scripts to run each of the agents, like `run_dqn5_agent.py` and `Q_learning_singleunits.py`.

### DQN Agents

* DQN1 is a foolish first attempt and doesn't seem to learn
* DQN2 uses a CNN to extract features from the entire observation space and approximate the Q values. It employs a target network and replay buffer. It sucks, most likely because it doesn't use one-hot vectors to encode past positions.
* DQN3 is like DQN2 but with a teeny tiny NN that uses precomputed features: a 200 day and 50 day moving average. Like DQN2 it also sucks because I hadn't figured out that you have to use one-hot vectors to encode agents' past positions.
* DQN4 finally figures out that you need to one-hot the position history. Duh. This uses a large feedforward neural network and finally produces some resemblance of profit.
* DQN5 is like DQN4 but uses a small neural network and precomputed features: 200-day moving average, 50-day moving average, and current position, one-hotted.
* DQN6 is like DQN4 but uses a CNN for function approximation over the entire observation space.

### Tabular Q-Learning

* Q_learning_stocks.py : Initial Q-Learning agent. Multiple updates made in Q_learning_singleunits.py
* Q_learning_singleunits.py : Single unit trader with tunable hyperparameters. Buys and Sells only 1 unit at a time. Made to see if a very general policy/profit can be made.
* Q_learning_multiunits.py : Multiple unit trader. Buy and Sell strategies can be changed and tuned in the __init__ method. Trader can buy as many units up to "max_buy_units", and sells a certain percentage of owned units. Buy points are stored and profits are calculated as they sell.