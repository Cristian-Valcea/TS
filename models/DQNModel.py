import math
import keras
import random
#import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque

@keras.saving.register_keras_serializable()
class DQN(keras.Model):
    def __init__(self, state_size, action_size):
        model = keras.models.Sequential()
        # Input Layer from course
#        model.add(keras.layers.Dense(units=32, input_dim=state_size, activation="relu"))
        # Input Layer from CHAT GPT
        model.add(keras.layers.Input(shape=(state_size,)))
        model.add(keras.layers.Dense(units=32, activation="relu"))
        # Hidden Layer
        model.add(keras.layers.Dense(units=8, activation="relu"))
        # Output Layer
        model.add(keras.layers.Dense(units=action_size, activation="linear"))
        # Compile Model
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))

        self.model = model


class Agent():
    def __init__(self, window_size, num_features, test_mode=False, model_name=''):
        self.window_size = window_size      # How many days of historical data do we want to include in our state representation
        self.num_features = num_features    # How many training features do we have
        self.state_size = window_size * num_features    # State size includes number of training features per day, and number of lookback days
        self.action_size = 3                # 0=hold, 1=buy, 2= sell
        self.memory = deque(maxlen=1000)    # Bound memory size: once the memory reaches 1000 units, the lefthand values are discarded as right 
        self.inventory = []                 # Inventory to hold trades
        self.model_name = model_name        # filename for saved model checkpoint loading
        self.test_mode = test_mode          # flag for testing (allows model load from checkpoint model_name)
        self.gamma = 0.95  
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.model = keras.models.load_model(model_name) if test_mode else self._model()

    # Deep Q-learning (DQL) model
    def _model(self):
        model = DQN(self.state_size,self.action_size).model
        return model

    # DQL Predict (with input reshaping)
    #  Input = State
    #  Output = Q-Table of actions Q-values
    def get_q_values_for_state(self, state):
        return self.model.predict(state.flatten().reshape(1, self.state_size))

    # DQL fit (with input reshaping)
    #  Input = State, Target Q-Table
    #  Output = MSE Loss between Target Q-Table and Actual Q-Table for state
    def fit_model(self, input_state, target_output):
        return self.model.fit(input_state.flatten().reshape(1, self.state_size), target_output, epochs=1, verbose=0)

    # Agent Action Selector
    #   Input = State
    #   Policy = epsilon-greedy (to minimize possibility of overfitting)
    #   Intitially high epsilon = more random, epsilon decay = less random later
    #   Output = Action (0, 1, or 2)
    def act(self, state):
        # Choose any action at random (Probablility = epsilon for training mode, 0% for testing mode)
        if (random.random() <= self.epsilon and not self.test_mode):
            return random.randrange(self.action_size)
        # Choose the action which has the highest Q-value (Probablitly = 1-epsilon for training mode, 100% for testing mode)
        options = self.get_q_values_for_state(state)
        return np.argmax(options[0])
       # return np.argmax(self.model.predict(state.flatten().reshape(1, self.state_size))[0])


    # Experience Replay (Learning Function)
    #   Input = Batch of (state, action, next_state) tuples
    #   Optimal Q Selection Policy = Bellman equation
    #   Important Notes = Model fitting step is in this function (fit_model)
    #                     Epsilon decay step is in this function
    #   Output = Model loss from fitting step
    def exp_replay(self, batch_size):
        losses = []
        mini_batch =[]
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

            for state, action, reward, next_state, done in mini_batch:
            # reminders: 
            #   - state is a vector containing close & MA values for the current time step
            #   - action is an integer representing the action taken by the act function at the current time step- buy, hold, or sell
            #   - reward represents the profit of a given action - it is either 0 (for buy, hold, and sells which loose money) or the profit in dollars (for a profitable sell)
            #   - next_state is a vector containing close & MA values for the next time step
            #   - done is a boolean flag representing whether or not we are in the last iteration of a training episode (i.e. True when next_state does not exist.)
                if done:
                    # special condition for last training epoch in batch (no next state)
                    optimal_q_for_action = reward
                else:
                    # target Q-values is updated using the Bellman equation: reward + gamma * max(predicted Q-values for next state))
                    optimal_q_for_action = reward + self.gamma * np.amax(self.get_q_values_for_state(next_state))
            # Get the predicted Q-values of the current state
            target_q_table = self.get_q_values_for_state(state)
            # Update the output Q table - replace the predicted Q value for action with the target Q value for action 
            target_q_table[0][action] = optimal_q_for_action
            # Fit the model where state is X and target_q_table is Y
            history = self.fit_model(state, target_q_table)
            losses += history.history['loss']
        # define epsilon decay (for the act function)     
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return losses



    def exp_replay_copilot(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state.flatten().reshape(1, self.state_size))
            if done:
                target[0][action] = reward
            else:
                t = self.model.predict(next_state.flatten().reshape(1, self.state_size))
                target[0][action] = reward + self.gamma * np.amax(t[0])
            self.fit_model(state, target)


keras.utils.disable_interactive_logging()

# Format price string
def format_price(n):
    return ('-$' if n < 0 else '$') + '{0:.2f}'.format(abs(n))

def sigmoid(x):
    if x < -709:  # math.exp(-709) is close to zero
        return 0.0
    return 1 / (1 + math.exp(-x))


# Plot behavior of trade output
def plot_behavior(data_input, bb_upper_data, bb_lower_data, states_buy, states_sell, profit, df, train=True):
    fig = plt.figure(figsize = (15,5))
    plt.plot(data_input, color='k', lw=2., label= 'Close Price')
    plt.plot(bb_upper_data, color='b', lw=2., label = 'Bollinger Bands')
    plt.plot(bb_lower_data, color='b', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='r', label = 'Buying signal', markevery = states_buy)
    plt.plot(data_input, 'v', markersize=10, color='g', label = 'Selling signal', markevery = states_sell)
    plt.title('Total gains: %f'%(profit))
    plt.legend()
    if train:
        plt.xticks(range(0, len(df.index.values), int(len(df.index.values)/15)), df.index.values[0:: int(len(df.index.values)/15)], rotation=45, fontsize='small')
    else:
        plt.xticks(range(0, len(df.index.values), int(len(df.index.values)/2)), df.index.values[0::int(len(df.index.values)/2)], rotation=45, fontsize='small')
    plt.show()

# Plot training loss
def plot_losses(losses, title):
    #fig = plt.figure(figsize = (15,5))
    plt.plot(losses)
    plt.title(title)
    plt.ylabel('MSE Loss Value')
    plt.xlabel('batch')
    plt.show()
    

# returns an an n-day state representation ending at time t
def get_state(data, t, n):    
    d = t - n
    if d >= 0:
        block = data[d:t] 
    else:
        block =  np.array([data[0]]*n) 
    res = []
    for i in range(n - 1):
        feature_res = []
        for feature in range(data.shape[1]):
            feature_res.append(sigmoid(block[i + 1, feature] - block[i, feature]))
        res.append(feature_res)
    # display(res)
    return np.array([res])


'''
# Train the model
keras.config.disable_traceback_filtering()  # disable built-in keras loading bars - they make the output difficult to read and monitor

# track number of examples in dataset (i.e. number of days to train on)
l = X_train[:,0].shape[0] - 1

print(l)

# batch size defines how often to run the exp_replay method
batch_size = 32

#An episode represents a complete pass over the data.
episode_count = 1

batch_losses = []
num_batches_trained = 0

for e in range(episode_count + 1):
    state = get_state(X_train, 0, window_size + 1)
    # initialize variables
    total_profit = 0
    total_winners = 0
    total_losers = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    for t in tqdm(range(l), desc=f'Running episode {e}/{episode_count}'):
        action = agent.act(state)   
        next_state = get_state(X_train, t + 1, window_size + 1)

        # initialize reward for the current time step
        reward = 0

        if action == 1: # buy
            # inverse transform to get true buy price in dollars
            buy_price = X_train[t, idx_close]
            agent.inventory.append(buy_price)
            states_buy.append(t)
            print(f'Buy: {format_price(buy_price)}')

        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)  
            # inverse transform to get true sell price in dollars
            sell_price = X_train[t, idx_close]

            # define reward as max of profit (close price at time of sell - close price at time of buy) and 0 
            trade_profit = sell_price - bought_price
            reward = max(trade_profit, 0)
            total_profit += trade_profit
            if trade_profit >=0:
                total_winners += trade_profit
            else:
                total_losers += trade_profit
            states_sell.append(t)
            print(f'Sell: {format_price(sell_price)} | Profit: {format_price(trade_profit)}')
        
        # flag for final training iteration
        done = True if t == l - 1 else False
        # append the details of the state action etc in the memory, to be used by the exp_replay function        
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        # print total profit and plot behaviour of the current episode when the episode is finished
        if done:
            print('--------------------------------')
            print(f'Episode {e}')
            print(f'Total Profit: {format_price(total_profit)}')
            print(f'Total Winners: {format_price(total_winners)}')
            print(f'Total Losers: {format_price(total_losers)}')
            print(f'Max Loss: {max(batch_losses[num_batches_trained:len(batch_losses)])}')
            print(f'Total Loss: {sum(batch_losses[num_batches_trained:len(batch_losses)])}')
            print('--------------------------------')
            if idx_bb_lower < X_train.shape[1] and idx_bb_upper < X_train.shape[1]:
                plot_behavior(X_train[:, idx_close].flatten(), X_train[:, idx_bb_upper].flatten(), X_train[:, idx_bb_lower].flatten(), states_buy, states_sell, total_profit)
            else:
                print("Index out of bounds for X_train")
            #plot_behavior(X_train[:, idx_close].flatten(), X_train[:, idx_bb_upper].flatten(), X_train[:, idx_bb_lower].flatten(), states_buy, states_sell, total_profit)
            plot_losses(batch_losses[num_batches_trained:len(batch_losses)], f'Episode {e} DQN model loss')
            num_batches_trained = len(batch_losses)

        if len(agent.memory) > batch_size:
            # when the size of the memory is greater than the batch size, run the exp_replay function on the batch to fit the model and get losses for the batch
            losses = agent.exp_replay(batch_size)    
            # then sum the losses for the batch and append them to the batch_losses list
            batch_losses.append(sum(losses))

    agent.model.save(f'model_ep{e}.keras')
'''



'''
# test the trained model

l_test = len(X_test) - 1
state = get_state(X_test, 0, window_size + 1)
total_profit = 0
done = False
states_sell_test = []
states_buy_test = []

#An episode represents a complete pass over the data.
episode_count = 1

#Get the trained model
agent = Agent(window_size, num_features=X_test.shape[1], test_mode=True, model_name=f'model_ep{episode_count}.keras')
agent.inventory = []

for t in range(l_test):
    action = agent.act(state)
    next_state = get_state(X_test, t + 1, window_size + 1)
    reward = 0

    if action == 1: # buy
        # inverse transform to get true buy price in dollars
        buy_price = X_test[t, idx_close]
        agent.inventory.append(buy_price)
        states_buy_test.append(t)
        print(f'Buy: {format_price(buy_price)}')

    elif action == 2 and len(agent.inventory) > 0: # sell
        bought_price = agent.inventory.pop(0)  
        # inverse transform to get true sell price in dollars
        sell_price = X_test[t, idx_close]

        # reward is max of profit (close price at time of sell - close price at time of buy)
        reward = max(sell_price - bought_price, 0)
        total_profit += sell_price - bought_price
        states_sell_test.append(t)
        print(f'Sell: {format_price(sell_price)} | Profit: {format_price(sell_price - bought_price)}')


    if t == l_test - 1:
        done = True
    
    # append to memory so we can re-train on 'live' (test) data later    
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print('------------------------------------------')
        print(f'Total Profit: {format_price(total_profit)}')
        print('------------------------------------------')
        
plot_behavior(X_test[:, idx_close].flatten(),X_test[:, idx_bb_upper].flatten(), X_test[:, idx_bb_lower].flatten(), states_buy_test, states_sell_test, total_profit, train=False)
'''


