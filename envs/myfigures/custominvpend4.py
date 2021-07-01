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


class CustomInvPendEnv4(gym.Env):
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
    def __init__(self, init_setup, param_setup={'gravity': 9.81, 'mass_cart': 1.0, 'mass_pole': 0.2, 'length': 0.5, 'sampling': 0.02, \
                                                'force_lim': 15.0, 'rot_mi': 0.01, 'trans_mi': 0.01}):
        # checking input parameters
        assert type(init_setup) == tuple
        assert np.size(init_setup) == 3
        assert type(init_setup[0]) == int
        assert type(init_setup[1]) == str
        assert type(init_setup[2]) == str
        
        assert type(param_setup) == dict
        assert len(param_setup) == 8

        # setting physics parameters
        self.gravity = param_setup['gravity']
        self.masscart = param_setup['mass_cart']
        self.masspole = param_setup['mass_pole']
        self.length = param_setup['length']
        self.total_mass = (self.masspole + self.masscart)

        # max force applied to cart
        self.force_mag = param_setup['force_lim']
        
        # simulation timestep
        self.tau = param_setup['sampling']
        
        # numerical intergration method
        self.kinematics_integrator = init_setup[1]
        
        self.Nc = (self.total_mass * self.gravity)
        
        self.control_algorithm = init_setup[2]
        
        # friction coefficients - rotational and translational
        self.mip = param_setup['rot_mi']
        self.mic = param_setup['trans_mi']
        
        # state variables limits
        self.theta_threshold_radians = math.pi # [rad]
        self.x_threshold = 2.4 # [m]
        
        self.theta_up = 20 * math.pi / 180.0 # 20 deg to [rad]

        # limit array - x, dx, theta, dtheta
        high = np.array([self.x_threshold, np.finfo(np.float32).max, self.theta_threshold_radians, np.finfo(np.float32).max], \
                        dtype=np.float32)

        # observation space box
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # action space box (control only with cart force)
        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        
        # initialising
        self.seed(seed=init_setup[0])
        self.viewer = None
        self.state = None
        self.hist_acc = np.array([0,0])

    # method for setting same randomize seed every trial
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, f):

        # reseting state vector elements
        x, x_dot, theta, theta_dot = self.state
        # force in considered as action to do
        force = np.clip(f, -self.force_mag, self.force_mag)[0]
        
        # temporary variables with already calculated trigonometric functions values
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Model calculation
        # #############################################################

        sign_arg = self.Nc * x_dot
        
        thetaacc = (self.gravity*sintheta + costheta*(self.mic*self.gravity*np.sign(sign_arg) + \
            ((- force - ((sintheta + (self.mic*np.sign(sign_arg)*costheta))*self.masspole*self.length* \
            (theta_dot**2))) / self.total_mass)) - ((self.mip*theta_dot)/(self.masspole*self.length))) / \
            (self.length*((4/3)-(((self.masspole*costheta)/self.total_mass)*(costheta - (self.mic*np.sign(sign_arg))))))
            
        Nc_prev = self.Nc
        
        self.Nc = (self.total_mass*self.gravity) - (self.masspole*self.length*((thetaacc*sintheta)+(costheta*(theta_dot**2))))
        
        sign_arg = self.Nc * x_dot
        
        if np.sign(Nc_prev) != np.sign(self.Nc):
            
            thetaacc = (self.gravity*sintheta + costheta*(self.mic*self.gravity*np.sign(sign_arg) + \
                ((- force - ((sintheta + (self.mic*np.sign(sign_arg)*costheta))*self.masspole*self.length* \
                (theta_dot**2))) / self.total_mass)) - ((self.mip*theta_dot)/(self.masspole*self.length))) / \
                (self.length*((4/3)-(((self.masspole*costheta)/self.total_mass)*(costheta - (self.mic*np.sign(sign_arg))))))
            
        xacc = (force + self.masspole*self.length*((theta_dot**2)*sintheta - thetaacc*costheta) - \
            self.mic*self.Nc*np.sign(sign_arg)) / self.total_mass


        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            
        elif self.kinematics_integrator == 'heun':
            
            prev_theta_dot = theta_dot
            prev_x_dot = x_dot
            theta_dot_est = theta_dot + self.tau * thetaacc
            x_dot_est = x_dot + self.tau * xacc
            
            sign_arg = self.Nc * x_dot_est
        
            thetaacc_est = (self.gravity*sintheta + costheta*(self.mic*self.gravity*np.sign(sign_arg) + \
                ((- force - ((sintheta + (self.mic*np.sign(sign_arg)*costheta))*self.masspole*self.length* \
                (theta_dot_est**2))) / self.total_mass)) - ((self.mip*theta_dot_est)/(self.masspole*self.length))) / \
                (self.length*((4/3)-(((self.masspole*costheta)/self.total_mass)*(costheta - (self.mic*np.sign(sign_arg))))))
                
            Nc_prev = self.Nc
            
            self.Nc = (self.total_mass*self.gravity) - (self.masspole*self.length*((thetaacc_est*sintheta)+(costheta*(theta_dot_est**2))))
            
            sign_arg = self.Nc * x_dot_est
            
            if np.sign(Nc_prev) != np.sign(self.Nc):
                
                thetaacc_est = (self.gravity*sintheta + costheta*(self.mic*self.gravity*np.sign(sign_arg) + \
                    ((- force - ((sintheta + (self.mic*np.sign(sign_arg)*costheta))*self.masspole*self.length* \
                    (theta_dot_est**2))) / self.total_mass)) - ((self.mip*theta_dot_est)/(self.masspole*self.length))) / \
                    (self.length*((4/3)-(((self.masspole*costheta)/self.total_mass)*(costheta - (self.mic*np.sign(sign_arg))))))
                
            xacc_est = (force + self.masspole*self.length*((theta_dot_est**2)*sintheta - thetaacc_est*costheta) - \
                self.mic*self.Nc*np.sign(sign_arg)) / self.total_mass
            
            
            x_dot = x_dot + self.tau*((xacc_est+xacc)/2.0)
            theta_dot = theta_dot + self.tau*((thetaacc_est+thetaacc)/2.0)
            # positions
            x = x + self.tau*((prev_x_dot+x_dot)/2.0)
            theta = theta  + self.tau*((prev_theta_dot + theta_dot)/2.0)
            
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
            
        self.hist_acc[0] = xacc
        self.hist_acc[1] = thetaacc
        
        # amendment on angles greater than +/- 180 degrees
        if theta > math.pi:
            theta = theta - 2*math.pi
        if theta < -math.pi:
            theta = theta + 2*math.pi       
        
        # saving newly calculated state
        self.state = (x, x_dot, theta, theta_dot)
        
        # calculating "done" flag
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold   
        )

        # assigning reward value
        if self.control_algorithm == 'ql':
            reward = -((theta/self.theta_up) ** 2+ (x/self.x_threshold)** 2)
        elif self.control_algorithm == 'lqr':
            reward = 1
        else:
            print('Unknown control algorithm! Setting reward to 0!')
            reward = 0
            
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.hist_acc = np.array([0,0])
        self.Nc = (self.total_mass * self.gravity)  
        return np.array(self.state)

    def set_state(self, state_loaded, accs=np.array([0,0])):
        self.state = state_loaded
        self.hist_acc = np.array(accs)

        costheta = math.cos(self.state[2])
        sintheta = math.sin(self.state[2])
        self.Nc = (self.total_mass*self.gravity) - (self.masspole*self.length*((self.hist_acc[1]*sintheta)+(costheta*(self.state[3]**2))))

        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
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
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
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
                
            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
