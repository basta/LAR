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
        Find garage entrance by aligning with purple and yellow markers.
        """
        CENTERING_ERROR_THRESHOLD = 10  # Pixels

        turtle_controller = self.automat.turtle_controller
        turtle_vision = self.automat.turtle_vision

        turtle_controller.reset_odometry()
        time.sleep(0.4) # Allow odometry to settle

        while True:
            # Attempt to find garage features and get alignment error.
            # Uses default min_sizes from TurtleVision.find_garage_alignment_error.
            alignment_error, status = turtle_vision.find_garage_alignment_error()

            # If features are not sufficient, rotate and try again.
            while status == "insufficient_regions":
                logger.info("Insufficient regions for garage alignment, rotating to find.")
                turtle_controller.cmd_velocity(angular=0.5)  # Rotate to find features.
                turtle_controller.rate.sleep()
                alignment_error, status = turtle_vision.find_garage_alignment_error() # Re-attempt.

            if status == "regions_found":
                if abs(alignment_error) < CENTERING_ERROR_THRESHOLD:
                    logger.info(f"Garage entrance centered. Error: {alignment_error:.2f} pixels.")
                    break  # Exit main loop, alignment is good
                else:
                    # Adjust robot orientation based on error
                    if alignment_error < 0.0: # Target is to the left of image center
                        # Error is negative. We want positive angular velocity to turn left.
                        angular_vel_command = np.clip(-alignment_error, # make it positive for clipping range
                                                      turtle_controller.min_yaw_rate,
                                                      turtle_controller.max_yaw_rate)
                    else: # Target is to the right of image center
                        # Error is positive. We want negative angular velocity to turn right.
                        angular_vel_command = np.clip(-alignment_error, # make it negative for clipping range
                                                      -turtle_controller.max_yaw_rate,
                                                      -turtle_controller.min_yaw_rate)
                    
                    logger.info(f"Aligning to garage. Error: {alignment_error:.2f} px. Proposed Angular vel: {angular_vel_command:.2f} rad/s.")
                    turtle_controller.cmd_velocity(linear=0.0, angular=angular_vel_command)
            elif status == "insufficient_regions":
                # This case should ideally be fully handled by the inner loop.
                # If we reach here, it means after rotating, regions are still not found.
                logger.warning("Still insufficient regions after rotation attempt. Retrying outer loop.")
            else:
                # Handle any other unexpected status from find_garage_alignment_error.
                logger.error(f"Unexpected status: {status}. Stopping alignment.")
                break # Safety break.

            turtle_controller.rate.sleep()

        turtle_controller.cmd_velocity(linear=0.0, angular=0.0) # Stop motion.
        logger.info("Scan_T1 action completed.")

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

            logger.info(f"Distance to closest point is: {np.linalg.norm(pt - pos)} m")
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
            f"Scanning now: {self.automat.memory.should_scan or self.automat.memory.t2_idx == 1}",
        )
        if self.automat.memory.should_scan or self.automat.memory.t2_idx == 1:
            self.automat.memory.t2_idx = None
            return self.automat.states["Task2"]

        logger.info("Moving successful, going to the next point")
        return self.automat.states["Move_T2"]


class Scan_T2(State):
    name = "Scan_T2"

    # Local helper methods see_yellow, find_garage, lose_garage_from_sight removed.

    def _perform_360_scan_and_map(self, 
                                  angles_to_scan, 
                                  yellow_visibility_min_size,
                                  yellow_sample_min_size, 
                                  purple_sample_min_size):
        logger.info("Starting 360-degree scan and mapping process.")
        turtle_vision = self.automat.turtle_vision
        turtle_controller = self.automat.turtle_controller
        turtle_map_yellow = self.automat.turtle_map_yellow
        turtle_map_purple = self.automat.turtle_map_purple

        for angle_rad in angles_to_scan:
            if not turtle_vision.is_color_visible(color="yellow", minimal_size=yellow_visibility_min_size):
                logger.warning("Lost sight of yellow marker during 360 scan, stopping scan early.")
                break

            logger.debug(f"Scanning at angle: {angle_rad:.2f} rad")
            yellow_points = turtle_vision.sample_garage(
                color="yellow", sampling_step=1, minimal_size=yellow_sample_min_size
            )
            purple_points = turtle_vision.sample_garage(
                color="purple", sampling_step=1, minimal_size=purple_sample_min_size
            )

            odom = turtle_controller.get_odometry()

            if yellow_points: 
                turtle_map_yellow.add_points(yellow_points, odom)
            if purple_points: 
                turtle_map_purple.add_points(purple_points, odom)

            logger.debug(f"Rotating to next scan angle: {angle_rad:.2f} rad")
            turtle_controller.rotate(angle_rad, relative=False, tol=0.1) 
            time.sleep(0.1) 
        
        logger.info("360-degree scan and mapping process completed.")

    def action(self):
        SCAN_T2_INIT_YELLOW_MIN_SIZE = 100 
        SCAN_LOOP_YELLOW_VISIBILITY_MIN_SIZE = 15 
        SCAN_YELLOW_SAMPLE_MIN_SIZE = 15
        SCAN_PURPLE_SAMPLE_MIN_SIZE = 4

        turtle_vision = self.automat.turtle_vision
        turtle_controller = self.automat.turtle_controller

        # Part 1: Initial positioning (lose and find yellow)
        turtle_vision.rotate_until_color_not_visible(
            turtle_controller,
            color="yellow",
            minimal_size=SCAN_T2_INIT_YELLOW_MIN_SIZE,
            angular_velocity=-0.60
        )
        turtle_vision.rotate_until_color_visible(
            turtle_controller,
            color="yellow",
            minimal_size=SCAN_T2_INIT_YELLOW_MIN_SIZE,
            angular_velocity=0.60
        )

        # Part 2: Setup for scanning loop
        current_yaw = turtle_controller.get_odometry()[1]
        angles = np.linspace(current_yaw, current_yaw + 2 * np.pi, 12)

        # Call the helper method to perform the scan.
        self._perform_360_scan_and_map(
            angles_to_scan=angles,
            yellow_visibility_min_size=SCAN_LOOP_YELLOW_VISIBILITY_MIN_SIZE,
            yellow_sample_min_size=SCAN_YELLOW_SAMPLE_MIN_SIZE,
            purple_sample_min_size=SCAN_PURPLE_SAMPLE_MIN_SIZE
        )

    def _calculate_task1_target_point(self, current_position, cluster1_mid, cluster2_mid):
        """
        Calculates a target navigation point for transitioning to Task1,
        based on the robot's position and two purple cluster midpoints.
        This encapsulates the "Some highschool math" logic.
        """
        NORMAL_DISTANCE_FACTOR = 1.5 # Factor to move along the normal vector
        
        vec_pos_from_c1 = current_position - cluster1_mid
        vec_c1_from_c2 = cluster1_mid - cluster2_mid
        normal_vec = np.array([-vec_c1_from_c2[1], vec_c1_from_c2[0]])

        if np.dot(vec_pos_from_c1, normal_vec) < 0:
            normal_vec = -normal_vec
        
        norm_of_normal = np.linalg.norm(normal_vec)
        if norm_of_normal < 1e-6: 
            logger.warning("Normal vector is near zero in _calculate_task1_target_point. Using zero vector.")
            unit_normal_vec = np.array([0.0, 0.0])
        else:
            unit_normal_vec = normal_vec / norm_of_normal

        target_point = cluster2_mid + (vec_c1_from_c2 / 2.0) + (unit_normal_vec * NORMAL_DISTANCE_FACTOR)
        
        return target_point

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

            # Geometric calculation replaced by helper method.
            # The check `use_task1` (i.e., purple_map.cluster_count >= 2)
            # and sorting of cluster_mids should generally ensure cluster_mids[0] and cluster_mids[1] are valid.
            if len(cluster_mids) >= 2: # Robustness check
                p = self._calculate_task1_target_point(position, cluster_mids[0], cluster_mids[1])
                logger.info(f"Calculated Task1 target point: {p}. Moving to point.")
                self.automat.turtle_controller.move_to(p, relative=False)
                return self.automat.states["Task1"]
            else:
                # This case should be rare if use_task1 is true and sorting is effective.
                logger.warning("Not enough cluster midpoints for geometric calculation. Proceeding to ICP.")


        # Fit garage model using ICP
        yellow_downsampled = self.automat.turtle_map_yellow.points_downsampled
        yellow_downsampled = TurtleUtils.robot2plt_numpy(yellow_downsampled)

        TurtleUtils.plot_data([yellow_downsampled], ["filtered data"])

        purple_pts = self.automat.turtle_map_purple.points
        if purple_pts is not None:
            TurtleUtils.plot_data([purple_pts], ["purple points"])

        logger.info(f"Yellow points shape: {yellow_downsampled.shape}")
        opt = self.automat.turtle_icp.optimize(yellow_downsampled)
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
        self.turtle_controller.cmd_velocity(linear = 0, angular = 0)
        time.sleep(0.005)
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
