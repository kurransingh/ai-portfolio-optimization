import statistics
import random
import uuid
import src.util as util
import src.constants as const


class Portfolio:
    trading_cost = const.COST

    def __init__(self, price, weights, shares):
        self.price = price
        self.weights = weights
        self.share_dist = shares
        self.value = self.__calc_value()
        self.id = self.__calc_id()

    # Returns portfolio value using share distribution and price vectors
    # Factors in trading cost
    def __calc_value(self):
        value = 0
        for idx in range(len(self.share_dist)):
            value += self.share_dist[idx] * self.price[idx] * (1 - Portfolio.trading_cost)
        return value

    # Returns a unique id for each portfolio based on the first security's weight
    # e.g. [0-5) : id1
    #      [5-10) : id2
    #      [10-15) : id3
    def __calc_id(self):
        random.seed(self.weights[0] * 100 // 5)
        a = "%32x" % random.getrandbits(128)
        rd = a[:12] + '4' + a[13:16] + 'a' + a[17:]
        return uuid.UUID(rd)

    # Returns share distribution vector for the next state using the previous state's weight vector and value
    def __next_share_dist(self):
        dist = []
        for idx in range(len(self.weights)):
            share_count = ((self.weights[idx]) * self.value) / self.price[idx]
            dist.append(share_count)
        return dist

    # Returns the next Portfolio given an action
    def next_state(self, action, next_price):
        stock1_prev_weight = self.weights[0]
        stock2_prev_weight = self.weights[1]
        if stock1_prev_weight > 0:
            diff = stock1_prev_weight * action / 100
            stock1_next_weight = round(stock1_prev_weight - diff, 4)
            stock2_next_weight = round(stock2_prev_weight + diff, 4)
        else:
            diff = stock2_prev_weight * action / 100
            stock1_next_weight = round(stock1_prev_weight + diff, 4)
            stock2_next_weight = round(stock2_prev_weight - diff, 4)

        next_weights = [stock1_next_weight, stock2_next_weight]
        next_share_dist = self.__next_share_dist()
        return Portfolio(next_price, next_weights, next_share_dist)

    def evaluate(self):
        return util.sigmoid(statistics.mean([self.__do_nothing(), self.__rebalance()]))

    def __do_nothing(self):
        weights = [0.5, 0.5]
        price = list(self.price)
        share_dist = []
        for idx in range(len(weights)):
            share_count = ((weights[idx]) * self.value) / price[idx]
            share_dist.append(share_count)
        return Portfolio(price, weights, share_dist).value

    def __rebalance(self):
        stock1_value = self.weights[0] * self.price[0]
        stock2_value = self.weights[1] * self.price[1]
        avg_value = statistics.mean([stock1_value, stock2_value])
        total_value = 2 * avg_value
        price = list(self.price)
        share_dist = [avg_value / self.price[0], avg_value / self.price[1]]
        weights = []
        for idx in range(len(share_dist)):
            sharePercent = (share_dist[idx] * self.price[idx] / total_value) * 100
            weights.append(sharePercent)
        return Portfolio(price, weights, share_dist).value

    def __str__(self) -> str:
        return 'id: ' + str(self.id) + '\n' + \
               'Price: ' + str(self.price) + '\n' + \
               'Weights: ' + str(self.weights) + '\n' + \
               'Shares: ' + str(self.share_dist) + '\n' + \
               'Value: ' + str(self.value)

