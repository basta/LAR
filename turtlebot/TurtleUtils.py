#!/usr/bin/env python3

import math
import numpy as np

def angle_diff(from_angle, to_angle):
    diff = to_angle - from_angle
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return diff

def R_z(yaw):
    """
        Return a rotation matrix doing transformation from the body frame to the world frame.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s],
        [s,  c]
    ])
