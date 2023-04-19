#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

import abc
from typing import Dict

from robolab_turtlebot import Turtlebot
from TurtleControllers import TurtleDrive
from TurtleVision import TurtlebotVision
from TurtleMap import TurtlebotMap
from TurtleICP import TurtlebotICP
import TurtleUtils
import time
import loggingSetup
import logging

logger = logging.getLogger("robot")


class State(abc.ABC):
    name = "default"

    def __init__(self, automat: "Automat"):
        self.automat = automat

    def execute(self) -> "State":
        logger.info(f"Executing {self.name}")
        self.action()
        return self.get_next_state()

    @abc.abstractmethod
    def action(self):
        pass

    @abc.abstractmethod
    def get_next_state(self):
        pass


class Start(State):
    name = "Start"

    def get_next_state(self):
        return (
            self.automat.states["Task2"]
            if self.automat.memory.button_pressed
            else self.automat.states["Start"]
        )

    def action(self):
        time.sleep(0.5)


class End(State):
    name = "End"

    def get_next_state(self):
        return None

    def action(self):
        self.automat.turtle_controller.turtle.play_sound(5)
        logger.info("Problem solved!")


class Move_PG(State):
    name = "Move_PG"

    def action(self):
        self.automat.turtle_controller.move_to(self.automat.memory.P_PG, relative=False)

    def get_next_state(self):
        return self.automat.states["Move_G"]


class Move_G(State):
    name = "Move_G"

    def action(self):
        self.automat.turtle_controller.move_to(self.automat.memory.P_G, relative=False)

    def get_next_state(self):
        return self.automat.states["End"]


class Scan_T1(State):
    name = "Scan_T1"

    def action(self):
        """
        Find garage entrance
        """
        PURPLE_MIN_SIZE = 10
        YELLOW_MIN_SIZE = 30

        def reg_mean(regs):
            s, n = 0.0, len(regs) * 2
            for reg in regs:
                s += reg[0] + reg[1]
            return s / n

        turtle_controller = self.automat.turtle_controller
        turtle_controller.reset_odometry()
        time.sleep(0.4)
        turtle_vision = self.automat.turtle_vision

        # Find Garage center
        while True:

            # Find purple stripes
            regs_purple = turtle_vision.get_regions(
                color="purple", minimal_size=PURPLE_MIN_SIZE
            )

            # Find yellow stripe inbetween purple stripes
            regs_yellow = turtle_vision.get_regions(
                color="yellow", minimal_size=YELLOW_MIN_SIZE
            )

            # Want to see two purple pillars and a yellow region inbetween
            while len(regs_purple) < 2 or len(regs_yellow) == 0:
                turtle_controller.cmd_velocity(angular=0.5)
                regs_purple = turtle_vision.get_regions(
                    color="purple", minimal_size=PURPLE_MIN_SIZE
                )
                regs_yellow = turtle_vision.get_regions(
                    color="yellow", minimal_size=YELLOW_MIN_SIZE
                )
                turtle_controller.rate.sleep()

            # Want to look at the center of the garage entrance
            purple_mid = reg_mean(regs_purple)
            err = purple_mid - turtle_vision.img_width // 2

            if abs(err) < 10:
                # Garage entrance found, stop!
                break
            else:
                if err < 0.0:
                    err = np.clip(
                        err,
                        -turtle_controller.max_yaw_rate,
                        -turtle_controller.min_yaw_rate,
                    )
                else:
                    err = np.clip(
                        err,
                        turtle_controller.min_yaw_rate,
                        turtle_controller.max_yaw_rate,
                    )
                turtle_controller.cmd_velocity(linear=0, angular=-err)

            turtle_controller.rate.sleep()

        turtle_controller.cmd_velocity(linear=0, angular=0)

    def get_next_state(self):
        def generate_path(turtle_vision, visualize=False):
            self.automat.turtle_controller.reset_odometry()
            time.sleep(0.3)
            P1, P2, P_G, P_PG, P_GOAL = turtle_vision.garage_entry_waypoints()

            if visualize:
                fig, ax = plt.subplots()
                TurtleUtils.visualize_garage(ax, P1, P2, P_G, P_PG, P_GOAL)
            return P_PG, P_GOAL

        turtle_vision = self.automat.turtle_vision
        P_PG, P_G = generate_path(turtle_vision, visualize=True)

        P_PG_prev = self.automat.memory.P_PG
        self.automat.memory.P_PG = P_PG
        self.automat.memory.P_G = P_G

        if P_PG_prev is None:
            return self.automat.states["Move_PG"]
        return self.automat.states["Move_G"]


class Task1(State):
    name = "Task1"

    def action(self):
        self.automat.turtle_controller.reset_odometry()
        time.sleep(0.3)

    def get_next_state(self):
        return self.automat.states["Scan_T1"]


class Move_T2(State):
    name = "Move_T2"

    def action(self):
        garage = self.automat.memory.garage

        pt, idx, should_scan, go = None, None, None, True

        if self.automat.memory.t2_idx is None:
            odom = self.automat.turtle_controller.get_odometry()
            pos = odom[0]
            pt, idx, should_scan = garage.closest_waypoint(odom)
            should_scan = False

            logger.info("Distance to closest point is:", np.linalg.norm(pt - pos))
            if np.linalg.norm(pt - pos) < 0.2:
                go = False
                should_scan = False
                logger.info("No need to scan")

            logger.info(f"Moving to point {idx}")
        else:
            prev_idx = self.automat.memory.t2_idx
            pt, idx, should_scan = garage.closest_waypoint(self.automat.memory.t2_idx)
            logger.info(f"Old point index: {prev_idx}, New point index: {idx}")
        if np.linalg.norm(pt) > 1.5:  # more than 1.5 meters far
            self.automat.memory.reset = True

        # Move to point
        if go:
            self.automat.turtle_controller.move_to(pt, relative=False)
            if idx in [2, 3, 6, 7]:
                garage_mid = TurtleUtils.plt2robot(
                    [self.automat.memory.garage.waypoints[:, 0]]
                )[0].reshape(2)
                self.automat.turtle_controller.face_towards(garage_mid, relative=False)
                logger.info("Facing towards mid point")

            if idx in [1]:
                waypoint_2 = TurtleUtils.plt2robot(
                    [self.automat.memory.garage.waypoints[:, 2]]
                )[0].reshape(2)
                logger.info("Facing towards point 2")
                self.automat.turtle_controller.face_towards(waypoint_2, relative=False)

        # Book keeping
        self.automat.memory.t2_idx = idx
        self.automat.memory.should_scan = should_scan

    def get_next_state(self):

        if self.automat.can_reset and self.automat.memory.reset:
            self.automat.reset()
            logger.warning("Traveled to far, resetting...")

        self.automat.can_reset = False

        logger.info(
            "Scanning now:",
            self.automat.memory.should_scan or self.automat.memory.t2_idx == 1,
        )
        if self.automat.memory.should_scan or self.automat.memory.t2_idx == 1:
            self.automat.memory.t2_idx = None
            return self.automat.states["Task2"]

        logger.info("Moving successful, going to the next point")
        return self.automat.states["Move_T2"]


class Scan_T2(State):
    name = "Scan_T2"

    def see_yellow(self):
        YELLOW_MIN_SIZE = 100
        turtle_vision = self.automat.turtle_vision

        # Find purple stripes
        regs_yellow = turtle_vision.get_regions(
            color="yellow", minimal_size=YELLOW_MIN_SIZE
        )

        return len(regs_yellow) > 0

    def find_garage(self):
        turtle_controller = self.automat.turtle_controller
        while True:
            if self.see_yellow():
                break
            turtle_controller.cmd_velocity(linear=0, angular=0.60)
            turtle_controller.rate.sleep()
        turtle_controller.cmd_velocity(linear=0, angular=0)

    def lose_garage_from_sight(self):
        turtle_controller = self.automat.turtle_controller
        while True:
            if not self.see_yellow():
                break
            turtle_controller.cmd_velocity(linear=0, angular=-0.60)
            turtle_controller.rate.sleep()
        turtle_controller.cmd_velocity(linear=0, angular=0)

    def action(self):
        # Garage must initially be out of view!
        self.lose_garage_from_sight()

        # Find garage first
        self.find_garage()

        # Scan
        current_yaw = self.automat.turtle_controller.get_odometry()[1]
        angles = np.linspace(current_yaw, current_yaw + 2 * np.pi, 12)

        YELLOW_MIN_SIZE = 15
        PURPLE_MIN_SIZE = 4
        for angle in angles:

            if not self.see_yellow():
                break

            # Scan environment
            yellow_points = self.automat.turtle_vision.sample_garage(
                color="yellow", sampling_step=1, minimal_size=YELLOW_MIN_SIZE
            )
            purple_points = self.automat.turtle_vision.sample_garage(
                color="purple", sampling_step=1, minimal_size=PURPLE_MIN_SIZE
            )

            # Current odometry
            odom = self.automat.turtle_controller.get_odometry()

            # Add points to maps
            self.automat.turtle_map_yellow.add_points(yellow_points, odom)
            self.automat.turtle_map_purple.add_points(purple_points, odom)

            # Rotate to next position
            self.automat.turtle_controller.rotate(angle, relative=False, tol=0.1)
            time.sleep(0.1)

    def get_next_state(self):

        # More than two purple clusters => garage entrance found
        use_task1 = self.automat.turtle_map_purple.cluster_count >= 2
        if use_task1:
            clusters = self.automat.turtle_map_purple.clusters

            cluster_mids = list()

            position = self.automat.turtle_controller.get_odometry()[0]

            for i in range(len(clusters)):
                cl_mid = np.mean(clusters[i].squeeze(), axis=1).reshape(2)
                cluster_mids.append(cl_mid)

            cluster_mids = sorted(
                cluster_mids, key=lambda x: np.linalg.norm(position - x)
            )

            logger.info(f"Can see {len(clusters)} clusters")

            TurtleUtils.plot_data(
                clusters, [f"Purple cluster {i}" for i in range(len(clusters))]
            )

            # Some highschool math
            cluster1_mid = cluster_mids[0]
            cluster2_mid = cluster_mids[1]
            P1 = position - cluster1_mid
            t = cluster1_mid - cluster2_mid
            n = np.array([-t[1], t[0]])
            if np.dot(P1, n) < 0:
                n = -n
            p = cluster2_mid + t / 2 + n * 1.5
            logger.info(f"Moving to point: {p}")
            self.automat.turtle_controller.move_to(p, relative=False)
            return self.automat.states["Task1"]

        # Fit garage model using ICP
        yellow_downsampled = self.automat.turtle_map_yellow.points_downsampled
        yellow_downsampled = TurtleUtils.robot2plt_numpy(yellow_downsampled)

        TurtleUtils.plot_data([yellow_downsampled], ["filtered data"])

        purple_pts = self.automat.turtle_map_purple.points
        if purple_pts is not None:
            TurtleUtils.plot_data([purple_pts], ["purple points"])

        logger.info("Yellow points shape:", yellow_downsampled.shape)
        opt = self.automat.turtle_icp.optimize(yellow_downsampled, method="LS")
        self.automat.memory.garage = opt.garage

        # Visualization
        TurtleUtils.plot_fitted_garage(
            self.automat.memory.garage,
            TurtleUtils.robot2plt_numpy(
                np.copy(self.automat.turtle_map_yellow.points_filtered)
            ),
        )

        self.automat.turtle_map_purple.reset()
        return self.automat.states["Move_T2"]


class Task2(State):
    name = "Task2"

    def action(self):
        pass

    def get_next_state(self):
        return self.automat.states["Scan_T2"]


class Automat:
    def __init__(self, states: list = None):
        self.states: Dict[str, State] = {}
        for state_cls in states:
            state = state_cls(self)
            self.states[state.name] = state
            assert state.name != "default", f"Name for state {state_cls} is not set"

        self.state: State = list(self.states.values())[0]
        self.can_reset = True

    def execute(self):
        while self.state:
            self.state = self.state.execute()


class Pycomat(Automat):
    def __init__(self):
        super().__init__(
            [Start, Task2, Scan_T2, Move_T2, Task1, Scan_T1, Move_PG, Move_G, End]
        )
        self.turtle = Turtlebot(rgb=True, pc=True)
        self.turtle_controller = TurtleDrive(self.turtle, rate=40)
        self.reset()

        self.states["Start"].automat.turtle_controller.turtle.register_button_event_cb(
            self.button_callback
        )
        self.states["Start"].automat.turtle_controller.turtle.register_bumper_event_cb(
            self.bumper_callback
        )

        logger.info("Waiting for button press...")

    def reset(self):
        self.turtle_vision = TurtlebotVision(self.turtle)
        self.turtle_map_yellow = TurtlebotMap(
            filter_eps=0.5, filter_min_samples=12
        )  # Map for yellow points in images
        self.turtle_map_purple = TurtlebotMap(
            filter_eps=0.1, filter_min_samples=4
        )  # Map for purple points in images
        self.turtle_icp = TurtlebotICP()
        self.turtle_controller.reset_odometry()
        self.memory = FSMMemory()

    def button_callback(self, message):
        if not self.states["Start"].automat.memory.button_pressed:
            logger.info("Button pressed")
        self.states["Start"].automat.memory.button_pressed = True

    def bumper_callback(self, message):
        os._exit(1)


class FSMMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.garage = None
        self.should_scan = True
        self.t2_idx = None
        self.P_PG = None
        self.P_G = None
        self.reset = False
        self.button_pressed = False
