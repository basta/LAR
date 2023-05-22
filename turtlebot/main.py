#!/usr/bin/env python3
import concurrent.futures
from TurtleOdom import TurtlebotOdometry
from TurtleMPC import TurtlebotMPC, TurtlebotPolicy, TurtlebotDriver

import rospy
import numpy as np
import threading

from robolab_turtlebot import Turtlebot

turtle = Turtlebot(rgb = False, pc = False)
odom = TurtlebotOdometry(turtle)

policy = TurtlebotPolicy()
driver = TurtlebotDriver(policy, turtle, rate = 40)
mpc = TurtlebotMPC(policy, odom, rate = 5)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(mpc.start)
    executor.submit(driver.start)