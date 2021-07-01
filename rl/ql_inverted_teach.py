# -*- coding: utf-8 -*-
"""
@author: Mikołaj Z., Szymon D.

based on: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
"""




import math
import matplotlib.pyplot as plt
import numpy as np
import gym
from tqdm import tqdm
import os
    


env = gym.make('CustomInvPendEnv4-v1', init_setup=(10, 'heun', 'ql'))

# Create models folder
if not os.path.isdir('tables'):
    os.makedirs('tables')

episodes = 10000


Q_resolution = 10 
action_resolution = 10

learning_rate = .25 #alpha
discount = 0.99 #gamma
show = 500
stats = 100

env.observation_space.high[1] = 5
env.observation_space.high[3] = 5
env.observation_space.low[1] = -5
env.observation_space.low[3] = -5

min_rew = -500

action_step = (env.action_space.high - env.action_space.low)/action_resolution

Q_size = [Q_resolution+1]*len(env.observation_space.high) 
Q_table =  np.random.uniform(low=-2, high=0, size=(Q_size + [action_resolution+1]))

maxstep = (Q_resolution)/2


Q_step = (env.observation_space.high - env.observation_space.low)/([Q_resolution]*len(env.observation_space.high))

# Statistic
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': [], 'suc': []}
ep_times = []
aggr_ep_times = {'ep': [], 'avg': [], 'max': [], 'min': [], 'suc': []}
ep_times2 = []
aggr_ep_times2 = {'ep': [], 'avg': [], 'max': [], 'min': [], 'suc': []}

def get_Q_state(state): 
    Q_state = (state - env.observation_space.low)/Q_step
    for i in range (np.size(Q_state)):
        if Q_state[i] < 0:
            Q_state[i] = 0
        if Q_state[i] > Q_resolution-1 and i!=2:
            Q_state[i] = Q_resolution-1              
    return tuple(Q_state.astype(np.int))

def get_action(Q_state): 
    action_n = np.argmax(Q_table[Q_state])
    action = action_n*action_step+env.action_space.low
    return action, action_n


for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):
    env.reset()

    
    Q_state = get_Q_state(env.set_state([0,0,0,np.random.uniform(low=-0.5, high=0.5)])) 
    episode_reward = 0
    done = False
    timeup = 0
    timeup2 = 0
    new_state = []
    i=0
    while not done:

        action, action_n = get_action(Q_state)
        
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        

        if abs(new_state[2]) < env.theta_up:
            timeup += 1
            timeup2 += 1
        else:
            timeup += 0
            timeup2 = 0
            
        if episode % show ==0:
            env.render()       
            
        new_Q_state = get_Q_state(new_state)
        max_future_Q = np.max(Q_table[new_Q_state]) 
        current_Q = Q_table[Q_state + (action_n,)] 
        new_Q = (1 - learning_rate) * current_Q + learning_rate * (reward + discount * max_future_Q)
        Q_table[Q_state + (action_n,)] = new_Q 
        Q_state = new_Q_state
        
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
        
        aggr_ep_times2['ep'].append(episode)
        aggr_ep_times2['avg'].append(average_time2)
        aggr_ep_times2['max'].append(max(ep_times2[-stats:]))
        aggr_ep_times2['min'].append(min(ep_times2[-stats:]))  
        
        if average_reward > min_rew:
            min_rew = average_reward
            Q_table_save = Q_table
env.close()

np.save(f'tables/Q_table_average_reward_{min_rew:_>7.2f}', Q_table_save)


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