# -*- coding: utf-8 -*-
"""
@author: Mikołaj Z., Szymon D.

based on: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
"""



import math
import matplotlib.pyplot as plt
import numpy as np
import gym



def calculate_integral_indicator_entry(state_value, Ts, Tk, set_value = 0, variant = 'ise'):
    indicator_entry = 0
    if variant=='ise':
        indicator_entry = Ts * ((set_value - state_value)**2)
    elif variant=='itse':
        indicator_entry = Ts * Tk * ((set_value - state_value)**2)
    elif variant=='iae':
        indicator_entry = Ts * np.abs(set_value - state_value)
    elif variant=='itae':
        indicator_entry = Ts * Tk * np.abs(set_value - state_value)
    return indicator_entry

def calculate_integral_indicator(state_values, Ts, Tend, set_values = 0, variant = 'ise', plot_data_archived = False):
    time_vec = np.arange(Ts, Tend+Ts, Ts)
    if np.size(set_values) == 1:
        set_values = np.ones(state_values.shape)*set_values
    if plot_data_archived:
        indicator_storage = np.zeros(state_values.shape)
    indicator = 0
    for i in range(len(time_vec)):
        indicator += calculate_integral_indicator_entry(state_values[i], Ts, time_vec[i], set_value=set_values[i], variant=variant)
        if plot_data_archived:
            indicator_storage[i] = indicator
    if plot_data_archived:
        return indicator, indicator_storage
    else:
        return indicator

def calculate_integral_vec_indicator(state_values, Ts, Tend, set_values = np.array([0,0,0,0]), variant='ise', plot_data_archived = False):
    if set_values.size == 4:
        tmp = set_values
        set_values = np.ones(state_values.shape)
        for i in range(4):
            set_values[i, :] = set_values[i, :]*tmp[i]
    
    if plot_data_archived:
        indicator_storage = np.zeros(state_values.shape)
    
    indicator_vec = np.zeros((4,1))
    
    if plot_data_archived:
        for i in range(4):
            indicator_vec[i], indicator_storage[i, :] = calculate_integral_indicator(state_values[i, :], Ts, Tend, \
                                                                                      set_values=set_values[i, :], variant=variant, \
                                                                                      plot_data_archived=plot_data_archived)
        return indicator_vec, indicator_storage
    else:
        for i in range(4):
            indicator_vec[i] = calculate_integral_indicator(state_values[i, :], Ts, Tend, set_values=set_values[i, :], variant=variant,\
                                                            plot_data_archived=plot_data_archived)        
        return indicator_vec
    
    
    
    
    
np.random.seed(1)


env = gym.make('CustomInvPendEnv4-v1', init_setup=(10, 'heun', 'ql'))

#name of model
Q_table = np.load('tables/Q_table_average_reward__-2.85.npy') #sample table



Q_resolution = 10 
action_resolution = 10


env.observation_space.high[1] = 5
env.observation_space.high[3] = 5
env.observation_space.low[1] = -5
env.observation_space.low[3] = -5



action_step = (env.action_space.high - env.action_space.low)/action_resolution


maxstep = (Q_resolution)/2


Q_step = (env.observation_space.high - env.observation_space.low)/([Q_resolution]*len(env.observation_space.high))



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

sim_length = 500
obs_state_storage = np.zeros((4,sim_length))
ref_state_storage = np.zeros((4,sim_length))
control_values_storage = np.zeros((sim_length,))
ref_k = np.zeros((4,1))

States_loaded = env.reset()

States_loaded[0] = 0
States_loaded[1] = 0
States_loaded[2] = 0
States_loaded[3] = 0.5

Q_state = get_Q_state(env.set_state(States_loaded))
episode_reward = 0
done = False
timeup = 0
timeup2 = 0
s = 0
t = 0
while not done:

    action, action_n = get_action(Q_state)
    new_state, reward, done, _ = env.step(action)
    obs_state_storage[:, t] = new_state
    ref_state_storage[:, t] = ref_k.reshape((4,))
    
    env.render()        
    new_Q_state = get_Q_state(new_state)
    Q_state = new_Q_state
    t += 1
env.close()


Ts = 0.02
Tend = sim_length * Ts
t = np.arange(Ts, Tend+Ts, Ts)
indicator_vec = np.zeros((4,1))
indicator_storage = np.zeros((4, sim_length))

indicator_vec, indicator_storage = calculate_integral_vec_indicator(obs_state_storage, Ts, Tend, variant='ise', plot_data_archived=True)
print(indicator_vec)

plt.figure()
plt.plot(t, indicator_storage[0, :], label="x")
plt.plot(t, indicator_storage[1, :], label="x_dot")
plt.plot(t, indicator_storage[2, :], label="theta")
plt.plot(t, indicator_storage[3, :], label="theta_dot")
plt.legend()
plt.ylabel('ISE values')
plt.xlabel('Time [s]')
plt.show()
