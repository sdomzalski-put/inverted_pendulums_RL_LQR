# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:58:44 2021

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

def calculate_integral_vec_indicator(state_values, Ts, Tend, set_values = np.array([0,0,0,0,0,0]), variant='ise', plot_data_archived = False, normalize = False):
    #time_vec = np.arange(Ts, Tend+Ts, Ts)
    if set_values.size == 6:
        tmp = set_values
        set_values = np.ones(state_values.shape)
        for i in range(6):
            set_values[i, :] = set_values[i, :]*tmp[i]
    
    if plot_data_archived:
        indicator_storage = np.zeros(state_values.shape)
    
    indicator_vec = np.zeros((6,1))
    
    if normalize:
        state_values[0, :] /= 2.4
        state_values[3, :] /= 2.4
        state_values[1, :] /= np.pi
        state_values[2, :] /= np.pi
        state_values[4, :] /= np.pi
        state_values[5, :] /= np.pi
        set_values[0, :] /= 2.4
        set_values[3, :] /= 2.4
        set_values[1, :] /= np.pi
        set_values[2, :] /= np.pi
        set_values[4, :] /= np.pi
        set_values[5, :] /= np.pi
        
    if plot_data_archived:
        for i in range(6):
            indicator_vec[i], indicator_storage[i, :] = calculate_integral_indicator(state_values[i, :], Ts, Tend, \
                                                                                      set_values=set_values[i, :], variant=variant, \
                                                                                      plot_data_archived=plot_data_archived)
        return indicator_vec, indicator_storage
    else:
        for i in range(6):
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
# state = [x, x_dot, theta1, theta_dot1, theta2, theta_dot2]
# #########################################
m_c = 0.603                     # cart mass
m_p1 = 0.123                    # 1st pole mass
m_p2 = 0.092                    # 2nd pole mass
g = 9.81                        # gravitational acceleration
b_c = 2.21                      # viskotic friction cart coefficient
b_p1 = 0.00164                  # viskotic friction 1st pole coefficient
b_p2 = 0.0005                   # viskotic friction 2nd pole coefficient
l_p1 = 0.49                     # length of 1st pole
el_p1 = 0.245 #l_p1 / 2.0       # "effective" length of 1st pole (from cart to mass center point)
l_p2 = 0.49                     # length of 2nd pole
el_p2 = 0.245 #l_p2 / 2.0       # "effective" length of 2nd pole (from 1st pole to mass center point)
I_p1 = 0.005                    # intertia of 1st pole
I_p2 = 0.003                    # intertia of 2nd pole
max_force = 15.0                # maximal applicable force to cart
Ts = 0.02                       # sampling time

param_setup = {'gravity': g, 'mass_cart': m_c, 'mass_pole_1': m_p1, 'mass_pole_2': m_p2, \
                                                'b_cart': b_c, 'b_pole_1': b_p1, 'b_pole_2': b_p2, \
                                                'len_pole_1': l_p1, 'len_pole_2': l_p2, \
                                                'inertia_pole_1': I_p1, 'inertia_pole_2': I_p2, \
                                                'sampling': Ts, 'force_lim': max_force}

# SIMULATION
Tend = 10
sim_length = int(math.ceil(Tend/Ts))
friction_active = True

# derived parameters
m_total = m_c + m_p1 + m_p2

# linearized system dynamics definition for LQR
# mass matrix
m11 = m_total
m12 = m_p1*el_p1 + m_p2*l_p1
m13 = m_p2*el_p2
m21 = m12
m22 = I_p1 + m_p1*(el_p1**2) + m_p2*(l_p1**2)
m23 = m_p2*l_p1*el_p2
m31 = m13
m32 = m23
m33 = I_p2 + m_p2*(el_p2**2)

M_0 = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
M_0_inv = np.linalg.inv(M_0)

# coriolis forces matrix
if friction_active:
    f11 = b_c
    f12 = 0
    f13 = 0
    f21 = 0
    f22 = b_p1
    f23 = 0
    f31 = 0
    f32 = -b_p2
    f33 = b_p2
else:
    f11 = b_c                    # optional
    f12 = 0
    f13 = 0
    f21 = 0
    f22 = 0
    f23 = 0
    f31 = 0
    f32 = 0
    f33 = 0

F_0 = np.array([[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]])

# gravity matrix
N_1 = m_p1*el_p1 + m_p2*l_p1
N_2 = m_p2 * el_p2

G_0 = np.zeros((3,3))
np.fill_diagonal(G_0, [0, -N_1*g, -N_2*g])

# input matrix
H = np.array([[1], [0], [0]])

# concatenating submatrices
A_21 = np.concatenate((np.zeros((3,1)), M_0_inv @ np.array([[0,0], [N_1*g, 0], [0, N_2*g]])), axis=1)
A_22 = -M_0_inv @ F_0

# linearized state matrices
A_lqr = np.concatenate((np.concatenate((np.zeros((3,3)), np.eye(3)), axis=1), np.concatenate((A_21, A_22), axis=1)), axis=0)
B_lqr = np.concatenate((np.zeros((3,1)),M_0_inv @ H))

# LQR parameterization
if friction_active:
    # R = np.array([10], ndmin=2)
    R = np.array([20], ndmin=2)
    Q = np.eye(6)
    # np.fill_diagonal(Q, [1000, 100, 100, 1, 10, 10])
    np.fill_diagonal(Q, [1000, 1000, 0.1, 1, 1, 1])
else:
    R = np.array([10], ndmin=2)
    Q = np.eye(6)
    np.fill_diagonal(Q, [1000, 10000, 5000, 1, 1, 1])    

# creating controller instance
lqr_ctrl = LQR_controller(6, 1)
lqr_ctrl.setup_ctrl(Q, R, A_lqr, B_lqr)

# creating environment instance
env = gym.make('DoubleInvPend-v1', init_setup=(10, 'heun', 'lqr'), param_setup=param_setup)

# storages definition
obs_state_storage = np.zeros((6, sim_length))
ref_state_storage = np.zeros((6, sim_length))
control_values_storage = np.zeros((sim_length,))
indicator_vec = np.zeros((6,1))
indicator_storage = np.zeros((6, sim_length))

# starting state definition
init_state = np.array([0, 0, 0, 0.5, 0, 0]) 
obs_state = env.reset()
obs_state = env.set_state(init_state)

# starting reference vector definition
ref_k = np.zeros((6,1))

for t in range(sim_length):
    # REFERENCE VECTOR CHANGES - UNCOMMENT IF WANT CHECK CONTROLLING POSITION
    # if t == np.ceil(sim_length/3):
    #     ref_k = np.array([1, 0, 0, 0, 0, 0], ndmin=2).T
    # elif t == np.ceil(2*sim_length/3):
    #     ref_k = np.array([-1, 0, 0, 0, 0, 0], ndmin=2).T
    
    # graphical rendering
    env.render()
    # state need to be adjusted to acceptable format - difference between model vector state 
    x_k = np.array(obs_state, ndmin=2).T
    x_k = np.array([[x_k[0, 0], x_k[2, 0], x_k[4, 0], x_k[1, 0], x_k[3, 0], x_k[5, 0]]]).T
    # calculating control value
    ctrl_u = lqr_ctrl.get_ctrl(x_k, ref_k)[0]
    # getting state from observer, reward and done flag
    obs_state, reward, done, _ = env.step(ctrl_u)
    # archiving state values
    obs_state_storage[:, t] = np.array([obs_state[0], obs_state[2], obs_state[4], obs_state[1], obs_state[3], obs_state[5]])
    ref_state_storage[:, t] = ref_k.reshape((6,))
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
plt.plot(t, obs_state_storage[3, :], label="$\dot{x}$")
plt.plot(t, ref_state_storage[3, :], label="$\dot{x}_{ref}$")
plt.legend()
plt.ylabel('Linear speed [m/s]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, obs_state_storage[1, :], label="$\Theta_1$")
plt.plot(t, ref_state_storage[1, :], label="$\Theta_{1_{ref}}$")
plt.legend()
plt.ylabel('Angle [rad]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, obs_state_storage[4, :], label="$\dot{\Theta}_1$")
plt.plot(t, ref_state_storage[4, :], label="$\dot{\Theta}_{1_{ref}}$")
plt.legend()
plt.ylabel('Angular speed [rad/s]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, obs_state_storage[2, :], label="$\Theta_2$")
plt.plot(t, ref_state_storage[2, :], label="$\Theta_{2_{ref}}$")
plt.legend()
plt.ylabel('Angle [rad]')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, obs_state_storage[5, :], label="$\dot{\Theta}_2$")
plt.plot(t, ref_state_storage[5, :], label="$\dot{\Theta}_{2_{ref}}$")
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
+ indicator_storage[4, :] + indicator_storage[5, :]
accumulated_position_ise = indicator_storage[0, :] + indicator_storage[1, :] + indicator_storage[2, :]

print('Single variable ISE:')
print(indicator_vec)
print('Accumulated ISE: ', accumulated_ise[-1])
print('Position accumulated ISE: ', accumulated_position_ise[-1])
print('')

plt.figure()
plt.plot(t, indicator_storage[0, :], label="x")
plt.plot(t, indicator_storage[3, :], label="$\dot{x}$")
plt.plot(t, indicator_storage[1, :], label="$\Theta_1$")
plt.plot(t, indicator_storage[4, :], label="$\dot{\Theta}_1$")
plt.plot(t, indicator_storage[2, :], label="$\Theta_2$")
plt.plot(t, indicator_storage[5, :], label="$\dot{\Theta}_2$")
plt.legend()
plt.title('ISE')
plt.ylabel('ISE values')
plt.xlabel('Time [s]')
plt.show()

plt.figure()
plt.plot(t, accumulated_ise, label='All')
plt.plot(t, accumulated_position_ise, label='Positions')
plt.title('Accumulated ISE')
plt.ylabel('ISE values')
plt.xlabel('Time [s]')
plt.legend()
plt.show()

