# -*- coding: utf-8 -*-
"""
@author: Mikołaj Z., Szymon D.

based on: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
"""





import matplotlib.pyplot as plt
import numpy as np
import gym


import tensorflow as tf



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
    
    
    
 
    

env = gym.make('CustomInvPendEnv4-v1', init_setup=(10, 'tustin', 'ql'))

#name of model
test=tf.keras.models.load_model('models/inv____-0.05max___-0.35avg___-0.99min__500.00time_up_____2.00a_res___20.00neurons___100.00mini_size__25000.00memory_size_____0.99discount_____1.00rand_action__1616465364.model') #sample network



action_resolution = 2

if action_resolution % 2:
    action_step = (env.action_space.high - env.action_space.low)/(action_resolution-1) 
else:
    action_step = (env.action_space.high - env.action_space.low)/(action_resolution) 

def get_action(action_n):
    action = action_n*action_step+env.action_space.low
    if not action_resolution % 2 and action>=0: #without 0
        action += action_step
    return action

sim_length = 500
obs_state_storage = np.zeros((4,sim_length))

actions=np.zeros((sim_length,1))   


env.reset()

States_loaded = env.reset()


States_loaded[0] = 0
States_loaded[1] = 0
States_loaded[2] = 0
States_loaded[3] = 0.5

current_state = env.set_state(States_loaded) #starting state

episode_reward = 0
done = False
timeup = 0
timeup2 = 0
s = 0
t = 0
while not done:
   
    qs = test.predict(np.array(current_state).reshape(-1, *current_state.shape))[0]
    action = get_action(np.argmax(qs))
    
    new_state, reward, done, _ = env.step(action)
    obs_state_storage[:, t] = new_state
    actions[t,0]=action

    env.render()        
    current_state = new_state
    t += 1
        
env.close()


Ts = 0.02
Tend = sim_length * Ts
t = np.arange(Ts, Tend+Ts, Ts)
indicator_vec = np.zeros((4,1))
indicator_storage = np.zeros((4, sim_length))

indicator_vec, indicator_storage = calculate_integral_vec_indicator(obs_state_storage, Ts, Tend, variant='ise', plot_data_archived=True)

plt.figure()
plt.plot(t, indicator_storage[0, :], label="x")
plt.plot(t, indicator_storage[1, :], label="x_dot")
plt.plot(t, indicator_storage[2, :], label="theta")
plt.plot(t, indicator_storage[3, :], label="theta_dot")
plt.legend()
plt.ylabel('ISE values')
plt.xlabel('Time [s]')
plt.show()

