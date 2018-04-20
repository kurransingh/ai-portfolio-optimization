from source.csv_to_df import DataReader
from source.portfolio import Portfolio
import source.util as util
import random


class QLearningAgent:
    def __init__(self, alpha, discount, epsilon):
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        self.q_values = util.Counter()

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]

    @staticmethod
    def get_legal_actions():
        return [5*i for i in range(0, 21)]

    def value_from_q_values(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legal_actions = self.get_legal_actions()
        values = []

        for action in legal_actions:
            values.append(self.get_q_value(state, action))

        return max(values)

    def action_from_q_values(self, state):
        """
          Compute the best action to take in a state.
        """
        legal_actions = self.get_legal_actions()
        value_actions = []

        for action in legal_actions:
            value_action = (self.get_q_value(state, action), action)
            value_actions.append(value_action)

        best_actions = [x for x in value_actions if x == max(value_actions)]

        return random.choice(best_actions)[1]

    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
        """
        legal_actions = self.get_legal_actions()

        if util.flip_coin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.action_from_q_values(state)

        return action

    def update(self, state, action, next_state, reward):
        """
          Q-value update.
        """
        q_value = self.get_q_value(state, action)
        value = reward + self.discount * self.value_from_q_values(next_state)
        updated_value = (1 - self.alpha) * q_value + self.alpha * value
        self.q_values[(state, action)] = updated_value

    def get_policy(self, state):
        return self.action_from_q_values(state)

    def get_value(self, state):
        return self.value_from_q_values(state)


stock1 = DataReader('../data/AMZN.csv').get_df()
stock2 = DataReader('../data/GOOGL.csv').get_df()
iterations = 5
learning_agent = QLearningAgent(0.1, 0.9, 0.3)

prev_rel_price = [stock1.iloc[0]['Close'] / stock1.iloc[0]['Open'], stock2.iloc[0]['Close'] / stock2.iloc[0]['Open']]
prev_weights = [0.5, 0.25, 0.25]
state = Portfolio(prev_rel_price, prev_weights)

for i in range(0, iterations):
    for idx in range(1, stock1.shape[0]):
        stock1_open = stock1.iloc[idx]['Open']
        stock1_close = stock1.iloc[idx]['Close']
        stock2_open = stock2.iloc[idx]['Open']
        stock2_close = stock2.iloc[idx]['Close']
        stock1_rel = [stock1_close / stock1_open, stock2_close / stock2_open]

        action = learning_agent.get_action(state)

        stock1_prev_weight = prev_weights[1]
        stock2_prev_weight = prev_weights[2]
        if stock1_prev_weight > 0:
            diff = stock1_prev_weight * action / 100
            stock1_next_weight = stock1_prev_weight - diff
            stock2_next_weight = stock2_prev_weight + diff
        else:
            diff = stock2_prev_weight * action / 100
            stock1_next_weight = stock1_prev_weight + diff
            stock2_next_weight = stock2_prev_weight - diff

        rel_price = [1, stock1_close / stock1_open, stock2_close / stock2_open]
        weights = [prev_weights[0], stock1_next_weight, stock2_next_weight]
        next_state = Portfolio(rel_price, weights, state)

        learning_agent.update(state, action, next_state, next_state.reward)

        state = next_state
        prev_weights = weights