#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from TurtleUtils import angle_diff, R
from robolab_turtlebot import Turtlebot, Rate

class TurtlebotOdometry:
    def __init__(self, turtle):
        self.turtle = turtle
        self.reset_odometry()
        self.yaw = self.turtle.get_odometry()[2]
        
    def get_odometry(self):
        odom = self.turtle.get_odometry()
        odom[2] = self._get_yaw(odom[2])
        return odom[:2], odom[2] # Position, yaw
    
    def reset_odometry(self):
        self.turtle.reset_odometry()
        self.yaw = 0.0
        time.sleep(0.5)
    
    def _get_yaw(self, odom_yaw):
        # Increment in yaw
        yaw_diff = self._normalize_angle(odom_yaw - self.yaw)
        self.yaw += yaw_diff
        return self.yaw
        
    def _normalize_angle(self, angle):
        angle = math.fmod(angle + np.pi, 2 * np.pi)
        angle = angle + np.pi if angle <= 0.0 else angle - np.pi
        return angle

class TurtlebotController:
    def __init__(self, turtle, rate):
        self.turtle = turtle
        self.rate = rate
        if isinstance(rate, (float, int)):
            self.rate = Rate(rate)
        self.odom = TurtlebotOdometry(turtle)
        
        ## Controller parameters
        self.min_yaw_rate = 0.48
        self.max_yaw_rate = 0.7
        
        self.max_linear_rate = 0.15
        self.min_linear_rate = 0.01
        
    def rotate(self, goal_yaw, relative = False, tol = 1e-3):
        _, current_yaw = self.get_odometry()
        if relative:
            goal_yaw += current_yaw
        yaw_error = angle_diff(current_yaw, goal_yaw)
        
        while abs(yaw_error) > tol:
            if yaw_error < 0.0:
                yaw_error = np.clip(yaw_error, -self.max_yaw_rate, -self.min_yaw_rate)
            else:
                yaw_error = np.clip(yaw_error, self.min_yaw_rate, self.max_yaw_rate)
                
            self.turtle.cmd_velocity(linear = 0.0, angular = yaw_error)
            self.rate.sleep()
            
            _, current_yaw = self.get_odometry()
            yaw_error = angle_diff(current_yaw, goal_yaw)
            
        self.turtle.cmd_velocity(linear = 0.0, angular = 0.0)
        time.sleep(0.2)
        
    def move_forward(self, distance, desired_yaw = None, tol = 1e-3):
        current_position, current_yaw = self.get_odometry()
        desired_yaw = current_yaw if desired_yaw is None else desired_yaw
        R_wb = R(desired_yaw)
        
        # Goal position expressed in the world frame
        goal = current_position + R_wb @ np.array([distance, 0.0])
        
        # Error vector expressed in the world frame
        error = goal - current_position
        
        # Error vector expressed in the body frame
        error_b = R_wb.T @ error
        
        while abs(error_b[0]) > tol:
            linear_velocity = np.clip(error_b[0], -self.max_linear_rate, self.max_linear_rate)
            angular_velocity = np.clip(desired_yaw - current_yaw, -self.max_yaw_rate, self.min_yaw_rate)
            
            # Current solution
            # self.turtle.cmd_velocity(linear = linear_velocity, angular = angular_velocity)
            
            # Upgraded solution
            linear_velocity = np.clip(linear_velocity, self.min_linear_rate, self.max_linear_rate) if linear_velocity > 0 else linear_velocity
            linear_velocity = np.clip(linear_velocity, -self.max_linear_rate, -self.min_linear_rate) if linear_velocity < 0 else  linear_velocity
            
            # Set velocity command
            self.turtle.cmd_velocity(linear = linear_velocity, angular = angular_velocity)
            
            self.rate.sleep()
            current_position, current_yaw = self.get_odometry()
            R_wb = R(current_yaw)
            error = goal - current_position
            error_b = R_wb.T @ error
        self.turtle.cmd_velocity(linear = 0, angular = 0)
        time.sleep(0.2)
        
    def move_to(self, position, yaw = None, relative = False):      
        if relative:
            self._move_to_rel(position, yaw)
        else:
            self._move_to_abs(position, yaw)
        
    def _move_to_abs(self, position, yaw):
        # Current position in world frame
        current_position, current_yaw = self.get_odometry()

        # Position error in the world frame
        error = position - current_position
        
        # Rotate so that turtlebot points at the desired position with its body frame x-axis
        phi = math.atan2(error[1], error[0])
        phi = current_yaw + angle_diff(current_yaw, phi) # TODO: Is this necessary?
        
        self.rotate(phi, relative = False)
        
        # Move towards the desired position
        self.move_forward(distance = np.linalg.norm(error), desired_yaw = phi)
        
        # Rotate to reach desired orientation
        if yaw is not None:
            self.rotate(yaw, relative = False)
    
    def _move_to_rel(self, position, yaw):
        # Position error in body frame
        _, current_yaw = self.get_odometry()
        error = position
        phi = math.atan2(error[1], error[0])
        self.rotate(current_yaw + phi, relative = False)
        self.move_forward(distance = np.linalg.norm(error), desired_yaw = current_yaw + phi)
        if yaw is not None:
            self.rotate(current_yaw + yaw, relative = False)
            
    def face_towards(self, position, relative = False):
        current_position, current_yaw = self.get_odometry()
        if relative:
            error = position
            phi = math.atan2(error[1], error[0])
            self.rotate(yaw, relative = False)
        else:
            error = position - current_position
            phi = math.atan2(error[1], error[0])
            phi = current_yaw + angle_diff(current_yaw, phi) 
            self.rotate(phi, relative = False)
            
    def cmd_velocity(self, linear = 0.0, angular = 0.0):
        self.turtle.cmd_velocity(linear = linear, angular = angular)
        
    def reset_odometry(self):
        self.odom.reset_odometry()
        
    def get_odometry(self):
        return self.odom.get_odometry()