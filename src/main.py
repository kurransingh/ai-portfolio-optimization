from src.csv_to_df import DataReader
import src.constants as const
from src.portfolio import Portfolio
from src.learning_agent import QLearningAgent


def train(train_end):
    # Load data
    stock1_df = DataReader(const.STOCK1_FILE).get_df()
    stock2_df = DataReader(const.STOCK2_FILE).get_df()

    # Initialize parameters
    iterations = const.ITER
    alpha = const.ALPHA
    discount = const.DISCOUNT
    epsilon = const.EPSILON

    # Initialize learning agent
    agent = QLearningAgent(alpha, discount, epsilon)

    # Initialize starting portfolio state
    init_price = [stock1_df.iloc[0]['Open'], stock2_df.iloc[0]['Open']]
    init_weights = const.WEIGHTS
    init_share_dist = const.SHARE_DIST
    init_pf = Portfolio(init_price, init_weights, init_share_dist)

    print("------------------------TRAINING START------------------------")

    print("------------------------INITIAL VALUES------------------------")
    print(str(init_pf))
    prev_pf = init_pf
    print()

    # Update Q Values for a fixed number of iterations
    for i in range(iterations):
        print("------------------------ITERATION " + str(i) + "------------------------")

        # Update Q values for each day in the training period
        for idx in range(1, train_end):
            # Select best action based on Q values
            action = agent.get_action(prev_pf.id)

            # Update stock price
            next_price = [stock1_df.iloc[idx]['Open'], stock2_df.iloc[idx]['Open']]

            # Generate next portfolio state
            curr_pf = prev_pf.next_state(action, next_price)

            # Update Q value for previous state
            # Reward = curr_pf.value - prev_pf.value
            agent.update(prev_pf.id, action, curr_pf.id, curr_pf.value - prev_pf.value)

            print("Action : ", action)
            print(str(curr_pf))
            print()

            prev_pf = curr_pf

        print()

    print("------------------------TRAINING END------------------------")
    return agent


def test(test_start, trained_agent):
    # Initialize data
    stock1_df = DataReader(const.STOCK1_FILE).get_df()
    stock2_df = DataReader(const.STOCK2_FILE).get_df()

    # Initialize starting portfolio state
    init_price = [stock1_df.iloc[test_start]['Open'], stock2_df.iloc[test_start]['Open']]
    init_weights = const.WEIGHTS
    init_share_dist = const.SHARE_DIST
    init_pf = Portfolio(init_price, init_weights, init_share_dist)

    prev_pf = init_pf

    print("------------------------TESTING START------------------------")
    # Test model for duration of the test period
    for idx in range(test_start + 1, stock1_df.shape[0]):
        # Get best action based on Q values
        action = trained_agent.get_policy(prev_pf.id)

        # Update stock price
        next_price = [stock1_df.iloc[idx]['Open'], stock2_df.iloc[idx]['Open']]

        # Generate next portfolio state
        curr_pf = prev_pf.next_state(action, next_price)

        print("Action : ", action)
        print(str(curr_pf))
        print()

        prev_pf = curr_pf

    # Calculate profit at the end of testing
    print("PROFIT: " + str(prev_pf.value - init_pf.value))
    print("SCORE: " + str(prev_pf.evaluate()))


if __name__ == "__main__":
    agent = train(const.TRAIN_END)
    test(const.TRAIN_END + 1, agent)
