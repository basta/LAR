#!/usr/bin/env python3

import numpy as np
import math, time, cv2
import logging

import TurtleUtils

logger = logging.getLogger(__name__)

YELLOW_HSV_MID = np.array([30, 170, 170])
YELLOW_HSV_RANGE = np.array([10, 100, 172])

PURPLE_HSV_MID = np.array([128, 170, 170])
PURPLE_HSV_RANGE = np.array([20, 66, 159])


class TurtlebotVision:
    UPPER_PIXEL_STRIP = 100
    LOWER_PIXEL_STRIP = 200
    DEFAULT_PURPLE_MIN_SIZE = 10
    DEFAULT_YELLOW_MIN_SIZE = 30

    def __init__(self, turtle):
        self.turtle = turtle

        # Strip parameters
        self.upper_pixel = self.UPPER_PIXEL_STRIP
        self.lower_pixel = self.LOWER_PIXEL_STRIP

        # Yellow HSV ranges
        self.set_yellow_range(YELLOW_HSV_MID, YELLOW_HSV_RANGE)

        # Purple HSV ranges
        self.set_purple_range(PURPLE_HSV_MID, PURPLE_HSV_RANGE)

        # Image data
        self.img_width = 640
        self.img_height = 480 # Corrected typo: heith -> height

        # Pixel to angle conversion constant (self.p2a).
        # This value typically represents the camera's focal length in pixels.
        # It is used in the `pixel2angle` method to convert a pixel coordinate
        # (distance from image center) into an angle (e.g., field of view).
        self.p2a = 620

    def set_yellow_range(self, yellow_mid, yellow_range):
        self.yellow_hsv_lower = yellow_mid - yellow_range
        self.yellow_hsv_upper = yellow_mid + yellow_range

    def set_purple_range(self, purple_mid, purple_range):
        self.purple_hsv_lower = purple_mid - purple_range
        self.purple_hsv_upper = purple_mid + purple_range

    # ===== Point Cloud =====

    def get_pc_image(self):
        return self.turtle.get_point_cloud()

    # ===== Depth Image =====

    def get_depth_image(self):
        pc_image = self.get_pc_image()
        return np.linalg.norm(pc_image, axis=2)

    def get_depth_image_strip(self):
        depth_image = self.get_depth_image()
        return depth_image[self.upper_pixel : self.lower_pixel]

    def get_depth_image_strip_filtered(self):
        depth_image_strip = self.get_depth_image_strip()
        column_mean = np.nanmean(depth_image_strip, axis=0)
        indices = np.where(np.isnan(depth_image_strip))
        depth_image_strip[indices] = np.take(column_mean, indices[1])
        return depth_image_strip

    def get_depth_strip(self):
        img = self.get_pc_image()
        return img[self.upper_pixel : self.lower_pixel]

    # ===== RGB Image =====

    def get_rgb_image(self):
        img = self.turtle.get_rgb_image()
        img[:, :, [0, 2]] = img[:, :, [2, 0]]  # BGR → RGB
        return img

    def get_rgb_strip(self):
        img = self.get_rgb_image()
        return img[self.upper_pixel : self.lower_pixel]

    def get_rgb_strip_mean(self):
        strip = self.get_rgb_strip()
        strip[:] = np.mean(strip, axis=0)
        return strip

    def get_hsv_strip(self):
        rgb_strip_mean = self.get_rgb_strip_mean()
        return cv2.cvtColor(rgb_strip_mean, cv2.COLOR_RGB2HSV)

    def get_rgb_mask(self, color="yellow"):
        # HSV strip
        hsv_strip = self.get_hsv_strip()

        # Get HSV color range
        if color.lower() == "yellow":
            hsv_lower = self.yellow_hsv_lower
            hsv_upper = self.yellow_hsv_upper
        elif color.lower() == "purple":
            hsv_lower = self.purple_hsv_lower
            hsv_upper = self.purple_hsv_upper
        else:
            raise Exception(f"Color '{color}' not supported.")

        # Detect color, create mask
        mask = cv2.inRange(hsv_strip, hsv_lower, hsv_upper)

        # Normalize mask
        if mask.max() != 0:
            mask = mask / mask.max()

        return mask

    def _reg_mean(self, regs):
        """
        Calculates the mean of the start and stop indices of regions.
        'regs' is expected to be a list of region tuples (start_idx, stop_idx, ...).
        """
        s, n_points = 0.0, 0
        if regs: # Ensure regs is not empty
            # Each region contributes its start and stop indices to the mean
            n_points = len(regs) * 2 
            if n_points == 0: # Should not happen if regs is not empty and regions are valid
                return 0.0 
            for reg_start, reg_stop, *_ in regs: # Unpack to use only start and stop
                s += reg_start + reg_stop
        return s / n_points if n_points > 0 else 0.0

    # ===== Garage detection =====
    def get_regions(self, color="yellow", minimal_size=10):
        mask = self.get_rgb_mask(color=color)
        return TurtleUtils.mask_regions(mask[0], minimal_size=minimal_size)

    def is_color_visible(self, color: str, minimal_size: int) -> bool:
        """Checks if any regions of the specified color and minimum size are visible."""
        regions = self.get_regions(color=color, minimal_size=minimal_size)
        return len(regions) > 0

    def rotate_until_color_visible(self, controller, color: str, minimal_size: int, angular_velocity: float = 0.6):
        """
        Rotates the robot until a region of the specified color and minimum size is visible.
        The controller object must provide cmd_velocity(linear, angular) and rate.sleep() methods.
        """
        logger.info(f"Rotating to find color '{color}' with min_size {minimal_size} using angular_velocity {angular_velocity:.2f} rad/s.")
        while not self.is_color_visible(color, minimal_size):
            controller.cmd_velocity(linear=0.0, angular=angular_velocity)
            controller.rate.sleep()
        controller.cmd_velocity(linear=0.0, angular=0.0) # Stop robot
        logger.info(f"Color '{color}' found.")

    def rotate_until_color_not_visible(self, controller, color: str, minimal_size: int, angular_velocity: float = -0.6):
        """
        Rotates the robot until a region of the specified color and minimum size is no longer visible.
        The controller object must provide cmd_velocity(linear, angular) and rate.sleep() methods.
        """
        logger.info(f"Rotating to lose sight of color '{color}' with min_size {minimal_size} using angular_velocity {angular_velocity:.2f} rad/s.")
        while self.is_color_visible(color, minimal_size):
            controller.cmd_velocity(linear=0.0, angular=angular_velocity)
            controller.rate.sleep()
        controller.cmd_velocity(linear=0.0, angular=0.0) # Stop robot
        logger.info(f"Color '{color}' no longer visible.")

    def pixel2angle(self, p):
        p_mid = p - self.img_width // 2
        return math.atan(p_mid / self.p2a)

    def sample_garage(
        self, color="yellow", sampling_step=20, minimal_size=10, r2p=False
    ):
        regions = self.get_regions(color=color, minimal_size=minimal_size)
        depth_img_strip = self.get_depth_image_strip_filtered()
        points = list()
        for start_idx, stop_idx, _ in regions:
            for i in range(start_idx + 2, stop_idx - 2, sampling_step):
                rho, phi = depth_img_strip[0][i], self.pixel2angle(i)
                if not np.isnan(rho):
                    points.append(TurtleUtils.polar2cartesian(rho, phi))

        # Convert points from robot's coordinate system to pyplot's coordinate system
        points = TurtleUtils.robot2plt(points) if r2p else points
        return points

    def find_garage_alignment_error(self, purple_min_size=None, yellow_min_size=None):
        # Use class defaults if specific sizes are not provided
        if purple_min_size is None:
            purple_min_size = self.DEFAULT_PURPLE_MIN_SIZE
        if yellow_min_size is None:
            yellow_min_size = self.DEFAULT_YELLOW_MIN_SIZE

        regs_purple = self.get_regions(color="purple", minimal_size=purple_min_size)
        regs_yellow = self.get_regions(color="yellow", minimal_size=yellow_min_size)

        if len(regs_purple) < 2 or len(regs_yellow) == 0:
            return None, "insufficient_regions"  # Status indicating not enough features

        # Calculate midpoint of purple regions
        # Assumes _reg_mean is available as a private method
        purple_mid = self._reg_mean(regs_purple)
        
        # Calculate error from image center
        # Assumes self.img_width is available
        alignment_error = purple_mid - self.img_width // 2
        
        return alignment_error, "regions_found"

    def garage_entry_waypoints(self):
        PRE_GARAGE_POINT_DISTANCE_FACTOR = 0.55
        GOAL_POINT_DISTANCE_FACTOR = 0.25
        regs = self.get_regions(color="purple")

        # Left pillar pixel
        p1 = int(0.5 * (regs[0][0] + regs[0][1]))

        # Right pillar pixel
        p2 = int(0.5 * (regs[1][0] + regs[1][1]))

        if p1 > p2:
            p1, p2 = p2, p1

        depth_img_strip = self.get_depth_image_strip_filtered()

        # Left pillar point
        rho1, phi1 = depth_img_strip[0][p1], self.pixel2angle(p1)
        P1 = TurtleUtils.polar2cartesian(rho1, phi1)

        # Right pillar point
        rho2, phi2, = depth_img_strip[0][
            p2
        ], self.pixel2angle(p2)
        P2 = TurtleUtils.polar2cartesian(rho2, phi2)

        # Garage entrance
        P_G = 0.5 * (P1 + P2)

        # Vector t
        t = P2 - P1

        R_90 = TurtleUtils.R(90, degrees=True)

        # Vector n1, points away from garage
        n1 = R_90.T @ t

        # Vector n2, points into garage
        n2 = -n1

        # Camera offset correction
        CAMERA_OFFSET = 0.12
        point_offset = t / np.linalg.norm(t) * CAMERA_OFFSET

        # Pre-garage point
        P_PG = P_G + n1 / np.linalg.norm(n1) * PRE_GARAGE_POINT_DISTANCE_FACTOR - point_offset

        # Goal point
        P_GOAL = P_G + n2 / np.linalg.norm(n2) * GOAL_POINT_DISTANCE_FACTOR - point_offset

        return [P1, P2, P_G, P_PG, P_GOAL]
