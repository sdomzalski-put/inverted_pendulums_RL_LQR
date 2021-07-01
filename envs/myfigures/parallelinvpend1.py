# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:57:56 2021

@authors:   Szymon Domżalski
            Mikołaj Zwierzyński
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ParallelInvPendV1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    '''
    Init setup is a tuple which contains info about:
    1) seed (uint)
    2) integration method (str: euler, euler implicit, tustin)
    3) control algorithm (str: lqr, ql)
    
    Param setup is a tuple which contains info about:
    1) gravity acceleration scalar
    2) mass of a cart
    3) mass of a pole / rod
    4) half of a length of a pole/rod
    5) sampling time for model calculus
    6) max force applied to the cart
    7) rotational friction coefficient
    8) translational friction coefficient
    '''
    # def __init__(self, init_setup, param_setup=(9.81, 1.0, 0.1, 0.5, 0.02, 15.0, 0.01, 0.01)):
    def __init__(self, init_setup, param_setup={'gravity': 9.81, 'mass_cart': 2.0, 'mass_pole_1': 0.5, 'mass_pole_2': 0.25, \
                                                'b_cart': 1.0, 'b_pole_1': 0.15, 'b_pole_2': 0.1, \
                                                'len_pole_1': 0.6, 'len_pole_2': 0.4, 'sampling': 0.02, \
                                                'force_lim': 35.0}):
        # checking input parameters
        assert type(init_setup) == tuple
        assert np.size(init_setup) == 3
        assert type(init_setup[0]) == int
        assert type(init_setup[1]) == str
        assert type(init_setup[2]) == str
        
        assert type(param_setup) == dict
        assert len(param_setup) == 11

        # setting physics parameters
        self.gravity = param_setup['gravity']
        self.m_c = param_setup['mass_cart']
        self.m_p1 = param_setup['mass_pole_1']
        self.m_p2 = param_setup['mass_pole_2']
        self.m_total = self.m_c + self.m_p1 + self.m_p2
        self.length_p1 = param_setup['len_pole_1']
        self.length_p2 = param_setup['len_pole_2']
        self.l_1 = self.length_p1 / 2.0
        self.l_2 = self.length_p2 / 2.0
        self.b_cart = param_setup['b_cart']
        self.b_p1 = param_setup['b_pole_1']
        self.b_p2 = param_setup['b_pole_2']
        
        # max force applied to cart
        self.force_mag = param_setup['force_lim']
        
        # simulation timestep
        self.tau = param_setup['sampling']
        
        # numerical intergration method
        self.kinematics_integrator = init_setup[1]
        self.control_algorithm = init_setup[2]

        self.Nc = self.m_total * self.gravity
        
        # state variables limits
        self.theta_threshold_radians = math.pi # [rad]
        self.x_threshold = 2.4 # [m]
        
        self.theta_up = 20 * math.pi / 180.0 # 20 deg to [rad]

        # limit array - x, dx, theta, dtheta
        high = np.array([self.x_threshold, np.finfo(np.float32).max, self.theta_threshold_radians, np.finfo(np.float32).max, \
                         self.theta_threshold_radians, np.finfo(np.float32).max], dtype=np.float32)

        # observation space box
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # action space box (control only with cart force)
        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        
        # initialising
        self.seed(seed=init_setup[0])
        self.viewer = None
        self.state = None
        self.hist_acc = np.array([0,0,0])

    # method for setting same randomize seed every trial
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, f):

        # reseting state vector elements
        x, x_dot, theta1, theta_dot1, theta2, theta_dot2 = self.state
        # force in considered as action to do
        force = np.clip(f, -self.force_mag, self.force_mag)[0]
        
        # temporary variables with already calculated trigonometric functions values
        costheta1 = math.cos(theta1)
        sintheta1 = math.sin(theta1)
        costheta2 = math.cos(theta2)
        sintheta2 = math.sin(theta2)


        m11 = self.m_total
        m12 = self.m_p1*self.l_1*costheta1
        m13 = self.m_p2*self.l_2*costheta2
        m21 = m12
        m22 = 4*self.m_p1*(self.l_1**2)/3.0
        m23 = 0
        m31 = m13
        m32 = m23
        m33 = 4*self.m_p2*(self.l_2**2)/3.0
        
        M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
        M_inv = np.linalg.inv(M)
        
        f11 = self.b_cart
        f12 = -self.m_p1*self.l_1*sintheta1*theta_dot1
        f13 = -self.m_p2*self.l_2*sintheta2*theta_dot2
        f21 = 0
        f22 = self.b_p1
        f23 = 0
        f31 = 0
        f32 = 0
        f33 = self.b_p2
        
        F = np.array([[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]])
        
        
        g1 = force
        g2 = self.m_p1*self.l_1*self.gravity*sintheta1
        g3 = self.m_p2*self.l_2*self.gravity*sintheta2
        
        G = np.array([[g1], [g2], [g3]])
        
        x_dot_vec = np.array([[x_dot], [theta_dot1], [theta_dot2]])
        x_vec = np.array([[x], [theta1], [theta2]])
        
        # Model calculation        
        acc_vec = M_inv @ (G - (F @ x_dot_vec)) 
        
        if self.kinematics_integrator == 'euler':
            x_vec = x_vec + (self.tau * x_dot_vec)
            x_dot_vec = x_dot_vec + (self.tau * acc_vec)
            
        elif self.kinematics_integrator == 'heun':
            prev_x_dot_vec = x_dot_vec
            x_dot_vec_est = x_dot_vec + self.tau*acc_vec
            x_vec_est = x_vec + self.tau * x_dot_vec_est
            theta1 = x_vec_est[1,0]
            theta2 = x_vec_est[2,0]
            theta_dot1 = x_dot_vec_est[1,0]
            theta_dot2 = x_dot_vec_est[2,0]
            
            costheta1 = math.cos(theta1)
            sintheta1 = math.sin(theta1)
            costheta2 = math.cos(theta2)
            sintheta2 = math.sin(theta2)
    
    
            m11 = self.m_total
            m12 = self.m_p1*self.l_1*costheta1
            m13 = self.m_p2*self.l_2*costheta2
            m21 = m12
            m22 = 4*self.m_p1*(self.l_1**2)/3.0
            m23 = 0
            m31 = m13
            m32 = m23
            m33 = 4*self.m_p2*(self.l_2**2)/3.0
            
            M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
            M_inv = np.linalg.inv(M)
            
            f11 = self.b_cart
            f12 = -self.m_p1*self.l_1*sintheta1*theta_dot1
            f13 = -self.m_p2*self.l_2*sintheta2*theta_dot2
            f21 = 0
            f22 = self.b_p1
            f23 = 0
            f31 = 0
            f32 = 0
            f33 = self.b_p2
            
            F = np.array([[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]])
            
            
            g1 = force
            g2 = self.m_p1*self.l_1*self.gravity*sintheta1
            g3 = self.m_p2*self.l_2*self.gravity*sintheta2
            
            G = np.array([[g1], [g2], [g3]])
        
            acc_vec_est = M_inv @ (G - (F @ x_dot_vec_est)) 
            x_dot_vec = x_dot_vec + self.tau*0.5*(acc_vec + acc_vec_est)
            x_vec = x_vec + self.tau*0.5*(prev_x_dot_vec+x_dot_vec)
            
        else:  # semi-implicit euler
            x_dot_vec = x_dot_vec + self.tau * acc_vec
            x_vec = x_vec + self.tau * x_dot_vec

        self.hist_acc = acc_vec
        
        x = x_vec[0, 0]
        theta1 = x_vec[1, 0]
        theta2 = x_vec[2, 0]

        x_dot = x_dot_vec[0, 0]
        theta_dot1 = x_dot_vec[1, 0]
        theta_dot2 = x_dot_vec[2, 0]
        
        # amendment on angles greater than +/- 180 degrees
        if theta1 > math.pi:
            theta1 = theta1 - 2*math.pi
        if theta1 < -math.pi:
            theta1 = theta1 + 2*math.pi
            
        if theta2 > math.pi:
            theta2 = theta2 - 2*math.pi
        if theta2 < -math.pi:
            theta2 = theta2 + 2*math.pi
            
        # saving newly calculated state
        self.state = (x, x_dot, theta1, theta_dot1, theta2, theta_dot2)
        
        # calculating "done" flag - DEFINE STOP CONDITIONS FOR REINFORCEMENT LEARNING
        # done = bool(
        #     x < -self.x_threshold
        #     or x > self.x_threshold   
        #     #theta < -self.theta_up
        #     #or theta > self.theta_up
        # )
        done = False
        # assigning reward value
        if self.control_algorithm == 'ql':
            # DEFINE HERE YOUR REWARD FUNCTION
            reward = 2
        elif self.control_algorithm == 'lqr':
            reward = 1
        else:
            print('Unknown control algorithm! Setting reward to 0!')
            reward = 0
            
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        self.hist_acc = np.array([0, 0, 0])
        return np.array(self.state)

    def set_state(self, state_loaded, accs=np.array([0, 0, 0])):
        self.state = state_loaded
        self.hist_acc = np.array(accs)
        
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen1 = scale * self.length_p1
        polelen2 = scale * self.length_p2
        cartwidth = 50.0
        cartheight = 30.0
        point_step = 0.5
        point_offset = (self.x_threshold - self.x_threshold//1)
        
        self.points = list()

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen1 - polewidth / 2, -polewidth / 2
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(.8, .6, .4)
            self.poletrans1 = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.poletrans1)
            pole1.add_attr(self.carttrans)
            self.viewer.add_geom(pole1)
            
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen2 - polewidth / 2, -polewidth / 2
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.4, .6, .8)
            self.poletrans2 = rendering.Transform(translation=(0, axleoffset))
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)
            
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans1)
            # self.axle.add_attr(self.poletrans2)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            
            for i in range(int(world_width//point_step) + 1):
                point = rendering.Line(( scale*(i*point_step + point_offset), carty-10), (scale*(i*point_step + point_offset), carty+10))
                self.points.append(point)
                
                self.points[i].set_color(0, 0, 0)
                self.viewer.add_geom(self.points[i])

            self._pole_geom1 = pole1
            self._pole_geom2 = pole2
            
        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole1 = self._pole_geom1
        pole2 = self._pole_geom2
        
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen1 - polewidth / 2, -polewidth / 2
        pole1.v = [(l, b), (l, t), (r, t), (r, b)]
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen2 - polewidth / 2, -polewidth / 2
        pole2.v = [(l, b), (l, t), (r, t), (r, b)]
        
        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans1.set_rotation(-x[2])
        self.poletrans2.set_rotation(-x[4])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
