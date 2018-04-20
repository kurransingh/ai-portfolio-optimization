import numpy as np
import math


class Portfolio:
    trading_cost = 10

    def __init__(self, rel_price, weights, prev_pf=None):
        self.rel_price = rel_price
        self.weights = weights
        self.cash = 1000

        if prev_pf is None:
            self.rel_price = None
            self.weights = [1, 0, 0]
            # self.u = None
            self.value = 1
            # self.rr = 0
            # self.log_rr = 0
            self.reward = 0
        else:
            # self.u = self.__calc_u(prev_pf)
            self.value = self.__calc_value(prev_pf)
            # self.rr = self.__calc_rr(prev_pf)
            # self.log_rr = self.__calc_log_rr(prev_pf)
            self.reward = self.__calc_reward(prev_pf)

    # def __calc_u(self, prev_pf):
    #     return Portfolio.trading_cost * \
    #            np.sum(np.linalg.norm(np.subtract(
    #                np.divide(np.multiply(self.rel_price, prev_pf.weights), np.dot(self.rel_price, prev_pf.weights)),
    #                self.weights)))

    # def __calc_value(self, prev_pf):
    #     return prev_pf.value * (1 - self.u) * np.dot(self.rel_price, prev_pf.weights)

    def __calc_value(self, prev_pf):
        return prev_pf.value * np.dot(self.rel_price, prev_pf.weights)

    # def __calc_rr(self, prev_pf):
    #     return (self.value / prev_pf.value) - 1
    #
    # def __calc_log_rr(self, prev_pf):
    #     return math.log(self.value / prev_pf.value)

    # def __calc_reward(self, prev_pf):
    #     return prev_pf.reward + math.log(self.u * np.dot(self.rel_price, prev_pf.weights))

    def __calc_reward(self, prev_pf):
        return prev_pf.reward + math.log(np.dot(self.rel_price, prev_pf.weights))

    def evaluate(self):
        return self.cash * self.value


# pf1 = Portfolio([1, 0.95, 0.86], [0.5, 0.1, 0.4])
# pf2 = Portfolio([1, 0.88, 0.97], [0.5, 0.20, 0.30], pf1)
# pf = Portfolio([1, 1.88, 5.97], [0.2, 0.2, 0.6], pf2)

# print(pf.rel_price)
# print(pf.weights)
# print(pf.u)
# print(pf.value)
# print(pf.rr)
# print(pf.log_rr)
# print(pf.reward)
# print(pf.evaluate())