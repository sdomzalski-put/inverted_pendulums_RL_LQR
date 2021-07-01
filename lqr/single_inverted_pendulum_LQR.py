# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 14:48:06 2021

@author: Szymon
"""
import math
import gym
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

##############################################################################
#                   INTEGRAL CONTROL QUALITY INDICATORS                      #
##############################################################################
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

def calculate_integral_vec_indicator(state_values, Ts, Tend, set_values = np.array([0,0,0,0]), variant='ise', plot_data_archived = False, normalize = False):
    #time_vec = np.arange(Ts, Tend+Ts, Ts)
    if set_values.size == 4:
        tmp = set_values
        set_values = np.ones(state_values.shape)
        for i in range(4):
            set_values[i, :] = set_values[i, :]*tmp[i]
    
    if plot_data_archived:
        indicator_storage = np.zeros(state_values.shape)
    
    indicator_vec = np.zeros((4,1))
    
    if normalize:
        state_values[0, :] /= 2.4
        state_values[1, :] /= 2.4
        state_values[2, :] /= np.pi
        state_values[3, :] /= np.pi
        set_values[0, :] /= 2.4
        set_values[1, :] /= 2.4
        set_values[2, :] /= np.pi
        set_values[3, :] /= np.pi
        
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
    
##############################################################################
#                           LQR CONTROLLER CLASS                             #
##############################################################################
class LQR_controller():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.Q = np.zeros((n,n))
        np.fill_diagonal(self.Q, np.ones((n,1)))
        self.R = np.zeros((m,m))
        np.fill_diagonal(self.R, np.ones((m,1)))
        self.isReady = False
        
    def load_system(self, A, B, C = 1, D = 0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    
    def load_ctrl_params(self, Q, R):
        self.Q = Q
        self.R = R
         
    def check_system_controllability(self):
        self.S = np.zeros((self.n,self.n))
        for i in range(self.n):
            tmp = np.linalg.matrix_power(self.A, i) @ self.B
            self.S[:, i] = tmp.reshape((self.n,))
        self.S = self.S.reshape((self.n,self.n)).T
        self.rank = np.linalg.matrix_rank(self.S)
        if self.rank == self.n:
            return True
        else:
            return False
        
    def get_gain_matrix(self):
        controllable = self.check_system_controllability()
        if controllable:
            ricatti_solution = np.matrix(la.solve_continuous_are(self.A, self.B, self.Q, self.R))
            self.K = np.array(np.dot(la.inv(self.R), np.dot(self.B.T, ricatti_solution)))
            self.isReady = True
        else:
            print ('System is incontrallable! Inserting K zeros matrix!')
            self.K = np.zeros((self.m,self.n))
            self.isReady = False
        return self.K
    
    def get_lqr_eigenvalues(self):
        if self.isReady:
            self.eigVals, self.eigVecs = la.eig(self.A - (self.B@self.K))
            return self.eigVals
        else:
            print('Gain matrix is not ready! Please calculate K matrix before!')
            return None
    
    def get_lqr_eigenvectors(self):
        if self.isReady:
            self.eigVals, self.eigVecs = la.eig(self.A - (self.B@self.K))
            return self.eigVecs
        else:
            print('Gain matrix is not ready! Please calculate K matrix before!')
            return None
    
    def get_ctrl(self, estimated_state, reference_value):
        if self.isReady:
            return -(self.K @ estimated_state) + (self.K @ reference_value)
        else:
            print('Gain matrix is not ready! Please calculate K matrix before!')
            return None
        
    def setup_ctrl(self, Q, R, A, B, C=1, D=0):
        self.load_system(A, B, C, D)
        self.load_ctrl_params(Q, R)
        self.get_gain_matrix()
        

# #########################################
# MODEL PARAMETRIZATION 
# state = [x, x_dot, theta, theta_dot]
# #########################################

mi_c = 0.01                         # cart friction coefficient
mi_p = 0.01                         # pole friction coefficient
mass_cart = 1.0                     # cart mass
mass_pole = 0.1                     # pole mass
gravity = 9.81                      # gravitational acceleration
rod_length = 1.0                    # whole rod length
pole_length = rod_length/2.0        # pole length (between cart and mass center point)
max_force = 15.0                    # maximal applicable force to cart
Ts = 0.02                           # sampling time for simulation

param_setup={'gravity': gravity, 'mass_cart': mass_cart, 'mass_pole': mass_pole, 'length': pole_length, 'sampling': Ts,\
             'force_lim': max_force, 'rot_mi': mi_p, 'trans_mi': mi_c}

# SIMULATION 
Tend = 10
sim_length = int(math.ceil(Tend/Ts))

# CREATING LINEARIZED MODEL
# derived parameters
total_mass = mass_cart + mass_pole
L_i = pole_length*((4/3) - (mass_pole/total_mass))
# system dynamics definition for LQR
A_lqr = np.array([[0, 1, 0, 0], [0, 0, -(3*mass_pole*gravity)/(mass_pole+4*mass_cart), 0], 
                  [0, 0, 0, 1], [0, 0, 3*(mass_pole+mass_cart)*gravity/((mass_pole+4*mass_cart)*pole_length), 0]])
B_lqr = np.array([0, 4/(mass_pole+4*mass_cart), 0, -3/((mass_pole+4*mass_cart)*pole_length)], ndmin=2).T
# LQR parameterization
R = np.array([10], ndmin=2)
Q = np.zeros((4,4))
# np.fill_diagonal(Q, [1, 1, 1, 1])
# np.fill_diagonal(Q, [1, 0, 1, 0])
np.fill_diagonal(Q, [1000, 1, 1000, 1])
# np.fill_diagonal(Q, [1000, 0, 1000, 0])

# creating controller instance
lqr_ctrl = LQR_controller(4, 1)
lqr_ctrl.setup_ctrl(Q, R, A_lqr, B_lqr)

# creating environment instance
env = gym.make('CustomInvPend-v4', init_setup=(10, 'heun', 'lqr'), param_setup=param_setup)

# storages definition
obs_state_storage = np.zeros((4, sim_length))
ref_state_storage = np.zeros((4, sim_length))
control_values_storage = np.zeros((sim_length,))
indicator_vec = np.zeros((4,1))
indicator_storage = np.zeros((4, sim_length))

# starting state definition
init_state = np.array([0, 0, 0, 0.5])
obs_state = env.reset()
obs_state = env.set_state(init_state)

# reference starting vector
ref_k = np.zeros((4,1))

for t in range(sim_length):
    # REFERENCE VALUES CHANGES - UNCOMMENT IF YOU WANT TO CHECK POSITION CONTROL
    # if t == np.ceil(sim_length/3):
    #     ref_k = np.array([1, 0, 0, 0], ndmin=2).T
    # elif t == np.ceil(2*sim_length/3):
    #     ref_k = np.array([-0.5, 0, 0, 0], ndmin=2).T
    
    # rendering graphical environment
    env.render()
    # state need to be adjusted to acceptable format
    x_k = np.array(obs_state, ndmin=2).T
    # calculating control value
    ctrl_u = lqr_ctrl.get_ctrl(x_k, ref_k)[0]
    # getting state from observer, reward and done flag
    obs_state, reward, done, _ = env.step(ctrl_u)
    # archiving state values
    obs_state_storage[:, t] = obs_state
    ref_state_storage[:, t] = ref_k.reshape((4,))
    control_values_storage[t] = ctrl_u[0]
    # checking done flag to finish simulation
    if done:
        break
    
print('Episode finished after', (t+1)*Ts, ' seconds')

env.close()

# PLOTTING

t = np.arange(Ts, Tend+Ts, Ts)

plt.figure()
plt.plot(t, obs_state_storage[0, :], label="x")
plt.plot(t, ref_state_storage[0, :], label="$x_{ref}$")
plt.legend()
plt.ylabel('Location [m]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, obs_state_storage[1, :], label="$\dot{x}$")
plt.plot(t, ref_state_storage[1, :], label="$\dot{x}_{ref}$")
plt.legend()
plt.ylabel('Linear speed [m/s]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, obs_state_storage[2, :], label="$\Theta$")
plt.plot(t, ref_state_storage[2, :], label="$\Theta_{ref}$")
plt.legend()
plt.ylabel('Angle [rad]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, obs_state_storage[3, :], label="$\dot{\Theta}$")
plt.plot(t, ref_state_storage[3, :], label="$\dot{\Theta}_{ref}$")
plt.legend()
plt.ylabel('Angular speed [rad/s]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, control_values_storage, label="u")
plt.legend()
plt.ylabel('control values')
plt.xlabel('Time [s]')
plt.show()

indicator_vec, indicator_storage = calculate_integral_vec_indicator(obs_state_storage, Ts, Tend, set_values=ref_state_storage, variant='ise', plot_data_archived=True)

accumulated_ise = indicator_storage[0, :] + indicator_storage[1, :] + indicator_storage[2, :] + indicator_storage[3, :]
accumulated_position_ise = indicator_storage[0, :] + indicator_storage[2, :]

print('Single variable ISE:')
print(indicator_vec)
print('Accumulated ISE: ', accumulated_ise[-1])
print('Position accumulated ISE: ', accumulated_position_ise[-1])
print('')

plt.figure()
plt.plot(t, indicator_storage[0, :], label="x")
plt.legend()
plt.title('ISE')
plt.ylabel('ISE of x')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, indicator_storage[2, :], label="$\Theta$")
plt.legend()
plt.title('ISE')
plt.ylabel('ISE of $\Theta$')
plt.xlabel('Time [s]')
plt.show()
