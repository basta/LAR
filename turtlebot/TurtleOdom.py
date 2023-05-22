import math
import numpy as np
import time

from TurtleUtils import angle_diff, R
from robolab_turtlebot import Rate


class TurtlebotOdometry:
    def __init__(self, turtle):
        self.turtle = turtle
        self.reset_odometry()
        self.yaw = self.turtle.get_odometry()[2]

    def get_odometry(self):
        odom = self.turtle.get_odometry()
        odom[2] = self._get_yaw(odom[2])
        return odom[:2], odom[2]  # Position, yaw

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