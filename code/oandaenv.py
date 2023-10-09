#
# Finance Environment
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#
import math
import tpqoa
import random
import numpy as np
import pandas as pd


class observation_space:
    def __init__(self, n):
        self.shape = (n,)


class action_space:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n - 1)


class OandaEnv:
    def __init__(self, symbol, start, end, granularity, price,
                 features, window, lags, leverage=1,
                 min_accuracy=0.5, min_performance=0.85,
                 mu=None, std=None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.granularity = granularity
        self.price = price
        self.api = tpqoa.tpqoa('../aiif.cfg')  # adjust path/filename
        self.features = features
        self.n_features = len(features)
        self.window = window
        self.lags = lags
        self.leverage = leverage
        self.min_accuracy = min_accuracy
        self.min_performance = min_performance
        self.mu = mu
        self.std = std
        self.observation_space = observation_space(self.lags)
        self.action_space = action_space(2)
        self._get_data()
        self._prepare_data()

    def _get_data(self):
        ''' Method to retrieve data from Oanda.
        '''
        self.fn = f'../data/'
        self.fn += f'oanda_{self.symbol}_{self.start}_{self.end}_'  
        self.fn += f'{self.granularity}_{self.price}.csv' 
        self.fn = self.fn.replace(' ', '_').replace('-', '_').replace(':', '_')
        print(self.fn)
        try:
            self.raw = pd.read_csv(self.fn, index_col=0, parse_dates=True)
        except:
            self.raw = self.api.get_history(self.symbol, self.start,
                                       self.end, self.granularity,
                                       self.price)
            self.raw.to_csv(self.fn)
        self.data = pd.DataFrame(self.raw['c'])
        self.data.columns = [self.symbol]

    def _prepare_data(self):
        ''' Method to prepare additional time series data
            (such as features data).
        '''
        self.data['r'] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        self.data['s'] = self.data[self.symbol].rolling(self.window).mean()
        self.data['m'] = self.data['r'].rolling(self.window).mean()
        self.data['v'] = self.data['r'].rolling(self.window).std()
        self.data.dropna(inplace=True)
        if self.mu is None:
            self.mu = self.data.mean()
            self.std = self.data.std()
        self.data_ = (self.data - self.mu) / self.std
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data['d'] = self.data['d'].astype(int)

    def _get_state(self):
        ''' Privat method that returns the state of the environment.
        '''
        return self.data_[self.features].iloc[self.bar -
                                    self.lags:self.bar].values

    def get_state(self, bar):
        ''' Method that returns the state of the environment.
        '''
        return self.data_[self.features].iloc[bar - self.lags:bar].values

    def reset(self):
        ''' Method to reset the environment.
        '''
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.bar = self.lags
        state = self._get_state()
        return state

    def step(self, action):
        ''' Method to step the environment forwards.
        '''
        correct = action == self.data['d'].iloc[self.bar]
        ret = self.data['r'].iloc[self.bar] * self.leverage
        reward_1 = 1 if correct else 0 
        reward_2 = abs(ret) if correct else -abs(ret)
        reward = reward_1 + reward_2 * self.leverage
        self.treward += reward_1
        self.bar += 1
        self.accuracy = self.treward / (self.bar - self.lags)
        self.performance *= math.exp(reward_2)
        if self.bar >= len(self.data):
            done = True
        elif reward_1 == 1:
            done = False
        elif (self.accuracy < self.min_accuracy and
              self.bar > self.lags + 15):
            done = True
        elif (self.performance < self.min_performance and
              self.bar > self.lags + 15):
            done = True
        else:
            done = False
        state = self._get_state()
        info = {}
        return state, reward, done, info
