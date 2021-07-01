# -*- coding: utf-8 -*-
"""
@author: Mikołaj Z., Szymon D.

based on: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import Dense
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os


action_resolution = 2
neurons = 20
rand_action=1

DISCOUNT = 0.99 #gamma

REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 100  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
MODEL_NAME = 'inv'
MIN_REWARD = -10  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 2000 #the longer, the better (usually)

epsilon = 1  
MIN_EPSILON = 0 

#  Stats settings
stats = 50  # episodes
SHOW_PREVIEW = False

show = 50

# Statistic
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': [], 'suc': []}
ep_times = []
aggr_ep_times = {'ep': [], 'avg': [], 'max': [], 'min': [], 'suc': []}
ep_times2 = []
aggr_ep_times2 = {'ep': [], 'avg': [], 'max': [], 'min': [], 'suc': []}



env = gym.make('CustomInvPendEnv4-v1', init_setup=(10, 'tustin', 'ql'))

if action_resolution % 2:
    action_step = (env.action_space.high - env.action_space.low)/(action_resolution-1) 
else:
    action_step = (env.action_space.high - env.action_space.low)/(action_resolution) 

# For stats
ep_rewards = [-500]


# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        
        model.add(Dense(neurons, activation='relu',input_shape=(4,))) 
        
        model.add(Dense(neurons, activation='relu')) 

        model.add(Dense(neurons, activation='relu')) 
        
        model.add(Dense(action_resolution, activation='linear')) 
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        
        current_qs_list = self.model.predict(current_states)
        
        

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        
        
        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action_n, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q
            # Update Q value for given state
            current_qs = current_qs_list[index]
            

            current_qs[action_n] = new_q
            
            #print(current_state)
            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

def get_action(action_n):
    action = action_n*action_step+env.action_space.low
    if not action_resolution % 2 and action>=0: #without 0
        action += action_step
    return action

agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    env.reset()
    current_state = np.array([0,0,0,np.random.uniform(low=-0.5, high=0.5)])
    Q_state = env.set_state(current_state)

    # Reset flag and start iterating until episode ends
    done = False
    timeup = 0
    timeup2 = 0
    while not done:

        if np.random.random() > epsilon or rand_action == 0:
            action_n = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action_n = np.random.randint(0, action_resolution)
        action = get_action(action_n)
        new_state, reward, done, _ = env.step(action)
        
        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        
        if abs(new_state[2]) < env.theta_up:
            timeup += 1
            timeup2 += 1
        else:

            timeup += 0
            timeup2 = 0
        
        if episode % show ==0:

            env.render()   

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action_n, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1



    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon -= 2/EPISODES
        epsilon = max(MIN_EPSILON, epsilon)
        
    ep_rewards.append(episode_reward)
    ep_times.append(timeup)
    ep_times2.append(timeup2)
    if not episode % stats:
        average_reward = sum(ep_rewards[-stats:])/stats
        average_time = sum(ep_times[-stats:])/stats
        average_time2 = sum(ep_times2[-stats:])/stats
        if episode == 0:
            average_reward = ep_rewards[0]
            average_time = ep_times[0]
            average_time2 = ep_times2[0]
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-stats:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-stats:]))  
        
        aggr_ep_times['ep'].append(episode)
        aggr_ep_times['avg'].append(average_time)
        aggr_ep_times['max'].append(max(ep_times[-stats:]))
        aggr_ep_times['min'].append(min(ep_times[-stats:])) 
        # Save model, but only when min reward is greater or equal a set value
        average_reward = sum(ep_rewards[-stats:])/len(ep_rewards[-stats:])
        min_reward = min(ep_rewards[-stats:])
        max_reward = max(ep_rewards[-stats:])
        min_time = min(ep_times[-stats:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{min_time:_>7.2f}time_up__{action_resolution:_>7.2f}a_res_{neurons:_>7.2f}neurons__{MINIBATCH_SIZE:_>7.2f}mini_size__{REPLAY_MEMORY_SIZE:_>7.2f}memory_size__{DISCOUNT:_>7.2f}discount__{rand_action:_>7.2f}rand_action__{int(time.time())}.model')

env.close()

plt.figure()
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

plt.figure()
plt.plot(aggr_ep_times['ep'], aggr_ep_times['avg'], label="average times up")
plt.plot(aggr_ep_times['ep'], aggr_ep_times['max'], label="max times up")
plt.plot(aggr_ep_times['ep'], aggr_ep_times['min'], label="min times up")
plt.legend(loc=4)
plt.xlabel('episode')
plt.ylabel('time')
plt.show()