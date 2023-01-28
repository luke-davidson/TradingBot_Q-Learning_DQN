import numpy as np
import random
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Let state = the combination of an aggregated moving_avg + the current position of the trader

class Stocks_RL():
    def __init__(self, env, info):
        self.env = env
        self.prices = np.array(env.prices)
        self.long_term = info['long_term']
        self.short_term = info['short_term']
        self.epsilon = info['epsilon']
        self.alpha = info['alpha']
        self.gamma = info['gamma']
        self.num_trials = info['num_trials']
        self.inc = info['inc']
        self.holding = False
        self.actions = {True: ["sell", "hold"],
                        False: ["buy", "hold"]}
        self.moving_avg = 0
        self.profit_master = 0
        self.buy_price = 0
        self.sell_points = []
        self.buy_points = []
        self.profits = []
        self.amplifier = 10
        self.profit = 0
        self.trial_profit = 0
        self.trial_profits = []
        self.wait_days = info['wait_days']
        self.Waiting = False

        # Bins and Qs
        self.num_bins = info['num_bins']
        mini, maxi = self.find_minmax_movingavg()
        self.bin_edges = np.linspace(mini - 1e-3, maxi + 1e-3, self.num_bins+1)
        self.bin = {}
        self.Q = {}
        for b in range(self.num_bins):
            self.bin.update({b: [self.bin_edges[b], self.bin_edges[b+1]]})    # {bin_num: [min, max]}
            for holding in ["T", "F"]:
                name = str(b) + "_" + holding
                self.Q.update({name: [0, 0]})    # [sell, hold] for holding, [buy, hold] for NOT holding
        self.info = info

        # Rewards
        multiplier = info['multiplier']
        rewards = np.linspace(-1, 1, self.num_bins)*multiplier
        self.rewards = {}
        for string in ["holding_sell", "holding_hold", "NOTholding_buy", "NOTholding_hold"]:
            self.rewards.update({string: {}})
            if string == "holding_sell" or string == "NOTholding_hold":
                i = self.num_bins-1
                for b in range(self.num_bins):
                    self.rewards[string].update({b: rewards[i]})
                    i -= 1
            else:
                i = 0
                for b in range(self.num_bins):
                    self.rewards[string].update({b: rewards[i]})
                    i += 1

    def find_minmax_movingavg(self):
        avgs = []
        for t in range(self.long_term, self.prices.shape[0] - 1, self.inc):
            self.calc_moving_average(t)
            avgs.append(self.moving_avg)
        mini = min(avgs)
        maxi = max(avgs)
        return mini, maxi

    def determine_bin(self):
        """
        Determines the aggregated bin # based on the moving average, used for the identifying the state
        """
        for b in range(self.num_bins):
            if self.moving_avg >= self.bin[b][0] and self.moving_avg <= self.bin[b][1]:
                return b

    def state(self):
        """
        Determines the name of the state --> "{bin_num}_{T/F}"
        """
        if self.holding:
            bool = "T"
        else:
            bool = "F"
        return str(self.determine_bin()) + "_" + bool

    def epsilon_greedy(self, state):
        """
        Epsilon greedy breaking ties randomly
        """
        if self.Waiting:
            return 1, "hold"
        if random.random() < self.epsilon:
            # Random
            a = np.random.randint(0, 2)
            return a, self.actions[self.holding][a]
        else:
            # Greedy
            all_a, = np.where(self.Q[state] == np.amax(self.Q[state]))
            a = np.random.choice(all_a)
            action = self.actions[self.holding][a]
            return a, action

    def calc_moving_average(self, t):
        """
        Calculates moving average, "long_term" and "short_term" can be edited in the info dict input to the class
        """
        long_term_avg = np.sum(self.prices[t-self.long_term:t])/self.long_term
        short_term_avg = np.sum(self.prices[t-self.short_term:t])/self.short_term
        self.moving_avg = short_term_avg - long_term_avg
    
    def calc_reward(self, sit):
        """
        Calculates reward based on ???
        """
        b = self.determine_bin()
        return self.rewards[sit][b]

    def step(self, action):
        """
        Determines reward of taking the action given the current price and next price.
        """
        self.profit = self.prices[self.current_t + 1] - self.prices[self.current_t]
        if self.Waiting and self.wait_days != 0:
            self.wait_days -= 1
        elif self.wait_days == 0:
            self.Waiting = False
            self.wait_days = self.info['wait_days']
        
        if self.holding:
            if action == "sell":
                # Sold stock
                sell_price = self.prices[self.current_t]
                self.sell_points.append(self.current_t)
                if self.buy_price == 0:
                    pass
                else:
                    profit = sell_price - self.buy_price
                    self.profits.append(profit)
                    self.profit_master += profit
                    self.trial_profit += profit
                self.holding = False
                reward = self.calc_reward("holding_sell")
            elif action == "hold":
                # Held on to stock
                reward = self.calc_reward("holding_hold")
        else:
            if action == "buy":
                # Bought new stock
                self.buy_price = self.prices[self.current_t]
                self.buy_points.append(self.current_t)
                self.holding = True
                self.Waiting = True
                reward = self.calc_reward("NOTholding_buy")
            elif action == "hold":
                # Does not own stock and did not buy any new stock
                reward = self.calc_reward("NOTholding_hold")
        self.calc_moving_average(self.current_t+1)
        next_state = self.state()
        return reward, next_state
    
    def update_Q(self, s, a, r, next_s):
        """
        Updates Q --> Q(S, A) = Q(S, A) + alpha*(R + gamma*max_a{Q(S', a)} - Q(S, A))
        """
        self.Q[s][a] += self.alpha*(r + self.gamma*np.amax(self.Q[next_s]) - self.Q[s][a])

    def show_policy(self):
        holding = np.empty((self.num_bins, 2))
        for k, v in self.Q.items():
            if k.split('_')[1] == "T":
                holding[int(k.split('_')[0]), 0] = v[0]-v[1]
            else:
                holding[int(k.split('_')[0]), 1] = v[0]-v[1]
        plt.imshow(holding)
        plt.title("Learned Policy for Single-Unit Trading")
        plt.xlabel("Position of Agent (0: Holding, 1: Not Holding)")
        plt.ylabel("Aggregated Moving Average")
        plt.legend()
        plt.show()

    def run(self):
        """
        Run for a certain number of trials. Hope to see an increase in trial profit as trials go on
        """
        for _ in range(self.num_trials):
            for t in range(self.long_term, self.prices.shape[0] - 1, self.inc):
                self.current_t = t
                self.calc_moving_average(self.current_t)
                state = self.state()
                a, action = self.epsilon_greedy(state)
                reward, next_state = self.step(action)
                self.update_Q(state, a, reward, next_state)
            self.buy_price = 0
            if _ % 10 == 0:
                print(f"[INFO]: Completed {_+1} of {self.num_trials} trials: \t\t{round((_+1)/self.num_trials*100, 2)}%")
                print(f"[INFO]: Total profit for trial {_+1}: \t${round(self.trial_profit, 2)}\n")
            self.trial_profits.append(self.trial_profit)
            self.trial_profit = 0

    def render(self):
        """
        Plots
        """
            # True price, buy points and sell points
        # plt.plot(range(stock_env.prices.shape[0]), stock_env.prices, linewidth=1)
        # plt.scatter(self.buy_points, self.prices[self.buy_points], s=10, c="r")
        # plt.scatter(self.sell_points, self.prices[self.sell_points], s=10, c="g")
        # plt.title("Buy and Sell Points - Q-Learning Agent")
        # plt.xlabel("Price ($)")
        # plt.ylabel("Time")
        # plt.legend(["Price", "Buy Points", "Sell Points"])

            # Profits, trends
        plt.plot(range(len(self.trial_profits)), self.trial_profits)
        profit_filtered = savgol_filter(self.trial_profits, 151, 2)
        plt.plot(range(len(self.trial_profits)), profit_filtered)
        plt.title("Single-Unit Profit Trend - Q-Learning Agent")
        plt.xlabel("Trial #")
        plt.ylabel("Trial Profit ($)")
        plt.legend(["Profit", "Trend Line"])

        plt.grid()
        plt.show()

# Run
env = gym.make('forex-v0')
params = {"long_term": 60,              # 60
        "short_term": 20,              # 20
        "alpha": 0.05,                   # 0.1
        "epsilon": 0.01,                # 0.01
        "gamma": 0.99,                  # 0.99
        "num_bins": 20,                  # 8
        "num_trials": 1200, 
        "inc": 10,                      # 10
        "multiplier": 5,
        "wait_days": 0
    }
stock_env = Stocks_RL(env, params)
stock_env.run()
stock_env.show_policy()
stock_env.render()

# Print optimal policy
for key, value in stock_env.Q.items():
    if list(key)[-1] == "T":
        print(f"State: {key} \t Best Action: {stock_env.actions[True][np.argmax(value)]} \t Action Values: {np.around(np.array(value), 3)}")
    else:
        print(f"State: {key} \t Best Action: {stock_env.actions[False][np.argmax(value)]} \t Action Values: {np.around(np.array(value), 3)}")