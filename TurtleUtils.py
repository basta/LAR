#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt

def angle_diff(from_angle, to_angle):
    diff = to_angle - from_angle
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return diff

def polar2cartesian(rho, phi):
    return np.array([
         rho * math.cos(phi),
        -rho * math.sin(phi)
    ])

def mask_regions(mask, minimal_size = 10):
    starts = []
    ends = []
    state = "BLACK"
    for i, n in enumerate(mask):
        if n == 1 and state == "BLACK":
            state = "WHITE"
            starts.append(i)
        elif n == 0 and state == "WHITE":
            state = "BLACK"
            ends.append(i)
    if state == "WHITE":
        ends.append(i+1)
    
    regions = []
    for start,end in zip(starts, ends):
        size = end - start
        if size < minimal_size:
            continue
        regions.append((start, end, end-start))
    
    return sorted(regions, key = lambda reg: reg[2])

def R(yaw, degrees = False):
    """
        Return a rotation matrix doing transformation from the body frame to the world frame.
    """
    yaw = math.radians(yaw) if degrees else yaw
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s],
        [s,  c]
    ])

def robot2plt(points):
    return [np.array([-p[1], p[0]]) for p in points]

def plt2robot(points):
    return [np.array([p[1], -p[0]]) for p in points]

def robot2plt_numpy(points):
    """
    points.shape = (2,N)
    """
    for i in range(points.shape[1]):
        points[:,i] = np.array([-points[1,i], points[0,i]])
    return points

def plot_data(data, labels, walls = None ,xlim = (-2, 2), ylim = (-2,2), figsize = (10, 4)):
    assert len(data) == len(labels)
    
    # Create a new figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    
    # plot data
    for lbl, dt in zip(labels, data):
        ax.scatter(dt[0,:], dt[1,:], label = lbl)
        
    if walls is not None:
        plt.plot(walls[0], walls[1], color = "orange", linewidth = 4)
    
    # Add legend and grid
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set(xlim = xlim, ylim = ylim)
    plt.show()
    
    return ax
    
def plot_fitted_garage(garage, data,xlim = (-2.5, 2.5), ylim = (-2.5, 2.5), figsize = (10,10)):
    d = [data, garage.corners, garage.waypoints] # data
    l = ["Fitted data", "Garage corners", "Garage waypoints"] # labels
    walls = np.array([
        [garage.corners[0,2], garage.corners[0,0], garage.corners[0,1], garage.corners[0,3]],
        [garage.corners[1,2], garage.corners[1,0], garage.corners[1,1], garage.corners[1,3]],
    ])
    ax = plot_data(d,l, walls, xlim = xlim, ylim = ylim, figsize = figsize)
    
def show_img(img):
    plt.imshow(img)
    
    
def transform_points(points, position, yaw):
    rot = R(yaw)
    return [rot @ p + position for p in points]
    
def visualize_garage(ax, P1B = None, P2B = None, P_G = None, P_PG = None, P_GOAL = None):
    # TURTLEBOT
    circle1 = plt.Circle((0.0,0.0), 0.2, color='r')
#     fig, ax = # note we must use plt.subplots, not plt.subplot
    ax.cla()
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.add_patch(circle1)

    # P1
    if P1B is not None:
        circle2 = plt.Circle((-P1B[1], P1B[0]), 0.04, color='black')
        ax.add_patch(circle2)

    # P2
    if P2B is not None:
        circle3 = plt.Circle((-P2B[1], P2B[0]), 0.04, color='b')
        ax.add_patch(circle3)
        
    if P_G is not None:
        circle4 = plt.Circle((-P_G[1], P_G[0]), 0.04, color='red')
        ax.add_patch(circle4)
        
    if P_PG is not None:
        circle5 = plt.Circle((-P_PG[1], P_PG[0]), 0.04, color='green')
        ax.add_patch(circle5)
        
    if P_GOAL is not None:
        circle6 = plt.Circle((-P_GOAL[1], P_GOAL[0]), 0.04, color='yellow')
        ax.add_patch(circle6)