import numpy as np
import random
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt

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

        ###########################
        ##### REWARD FUNCTION #####
        ###########################                                                                 # WANTS / THOUGHTS
        self.rewards = {"holding_sell": -10*self.moving_avg - 10*self.profit,                    # High: Decreasing moving_avg, decrease in price after sell
                                                                                # Low:  Increasing moving_avg, increase in price after sell
                        "holding_hold": -10*self.moving_avg - 10*self.profit,                  # High: Increasing moving_avg, increase in price after hold
                                                                                # Low:  Decreasing moving_avg, decrease in price after hold
                        "NOTholding_buy": 10*self.moving_avg + 10*self.profit,                   # High: ?
                                                                                # Low:  ?
                        "NOTholding_hold": 10*self.moving_avg + 10*self.profit}                # High: Decreasing moving_avg, decrease in price after staying
                                                                                # Low:  Increasing moving_avg, increase in price after staying
        ###########################
        ##### REWARD FUNCTION #####
        ###########################

        # Bins and Qs
        self.num_bins = info['num_bins']
        self.bin_edges = np.linspace(-0.015, 0.02, self.num_bins+1)
        self.bin = {}
        self.Q = {}
        for b in range(self.num_bins):
            self.bin.update({b: [self.bin_edges[b], self.bin_edges[b+1]]})    # {bin_num: [min, max]}
            for holding in ["T", "F"]:
                name = str(b) + "_" + holding
                self.Q.update({name: [0, 0]})    # [sell, hold] for holding, [buy, hold] for NOT holding

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
    
    def calc_reward(self):
        """
        Calculates reward based on ???
        """
        pass

    def step(self, action):
        """
        Determines reward of taking the action given the current price and next price.
        """
        self.profit = self.prices[self.current_t + 1] - self.prices[self.current_t]
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
                reward = self.rewards['holding_sell']
            elif action == "hold":
                # Held on to stock
                reward = self.rewards['holding_hold']
        else:
            if action == "buy":
                # Bought new stock
                self.buy_price = self.prices[self.current_t]
                self.buy_points.append(self.current_t)
                self.holding = True
                reward = self.rewards['NOTholding_buy']
            elif action == "hold":
                # Does not own stock and did not buy any new stock
                reward = self.rewards['NOTholding_hold']
        self.calc_moving_average(self.current_t+1)
        next_state = self.state()
        return reward, next_state
    
    def update_Q(self, s, a, r, next_s):
        """
        Updates Q --> Q(S, A) = Q(S, A) + alpha*(R + gamma*max_a{Q(S', a)} - Q(S, A))
        """
        self.Q[s][a] += self.alpha*(r + self.gamma*np.amax(self.Q[next_s]) - self.Q[s][a])

    def run(self):
        """
        Run for a certain number of trials. Hope to see an increase in trial profit as trials go on
        """
        for _ in range(self.num_trials):
            for t in range(200, self.prices.shape[0] - 1, 10):
                self.current_t = t
                self.calc_moving_average(self.current_t)
                state = self.state()
                a, action = self.epsilon_greedy(state)
                reward, next_state = self.step(action)
                self.update_Q(state, a, reward, next_state)
            self.buy_price = 0
            print(f"[INFO]: Completed {_+1} of {self.num_trials} trials: \t\t{round((_+1)/self.num_trials*100, 2)}%")
            print(f"[INFO]: Total profit for trial {_+1}: \t${round(self.trial_profit, 2)}\n")
            self.trial_profits.append(self.trial_profit)
            self.trial_profit = 0

    def render(self):
        """
        Plots
        """
            # True price, buy points and sell points
        plt.plot(range(stock_env.prices.shape[0]), stock_env.prices, linewidth=1)
        plt.scatter(self.buy_points, self.prices[self.buy_points], s=10, c="r")
        plt.scatter(self.sell_points, self.prices[self.sell_points], s=10, c="g")

            # Profits, trends
        # m, b = np.polyfit(range(len(self.profits)), self.profits, 1)
        # m, b = np.polyfit(range(len(self.trial_profits)), self.trial_profits, 1)
        # plt.plot(range(len(self.profits)), self.profits)
        # plt.plot(range(len(self.profits)), range(len(self.profits))*m + b)
        # plt.plot(range(len(self.trial_profits)), range(len(self.trial_profits))*m + b)
        # plt.plot(range(len(self.trial_profits)), self.trial_profits)

        plt.grid()
        plt.show()

# Run
env = gym.make('forex-v0')
params = {"long_term": 200,
        "short_term": 50,
        "alpha": 0.1,
        "epsilon": 0.01,
        "gamma": 0.9,
        "num_bins": 8,
        "num_trials": 500
    }

stock_env = Stocks_RL(env, params)
stock_env.run()
stock_env.render()
print(stock_env.profit_master)
for key, value in stock_env.Q.items():
    if list(key)[2] == "T":
        print(f"State: {key} \t Best Action: {stock_env.actions[True][np.argmax(value)]}")
    else:
        print(f"State: {key} \t Best Action: {stock_env.actions[False][np.argmax(value)]}")
