#!/usr/bin/env python3

import numpy as np
import math, time, cv2

import TurtleUtils

YELLOW_HSV_MID =  np.array([30, 170, 170])
YELLOW_HSV_RANGE = np.array([10,100,172])

PURPLE_HSV_MID = np.array([128, 170, 170])
PURPLE_HSV_RANGE = np.array([20,66,159])

class TurtlebotVision:
    def __init__(self, turtle):
        self.turtle = turtle
        
        # Strip parameters
        self.upper_pixel = 100
        self.lower_pixel = 200
        
        # Yellow HSV ranges
        self.set_yellow_range(YELLOW_HSV_MID, YELLOW_HSV_RANGE)
        
        # Purple HSV ranges
        self.set_purple_range(PURPLE_HSV_MID, PURPLE_HSV_RANGE)
        
        # Image data
        self.img_width = 640
        self.img_heith = 480
        
        # Pixel to angle conversion constant
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
        return np.linalg.norm(pc_image, axis = 2)
    
    def get_depth_image_strip(self):
        depth_image = self.get_depth_image()
        return depth_image[self.upper_pixel:self.lower_pixel]
    
    def get_depth_image_strip_filtered(self):
        depth_image_strip = self.get_depth_image_strip()
        column_mean = np.nanmean(depth_image_strip, axis=0)
        indices = np.where(np.isnan(depth_image_strip))
        depth_image_strip[indices] = np.take(column_mean, indices[1])
        return depth_image_strip
    
    def get_depth_strip(self):
        img = self.get_pc_image()
        return img[self.upper_pixel:self.lower_pixel]
    
    # ===== RGB Image =====
    
    def get_rgb_image(self):
        img = self.turtle.get_rgb_image()
        img[:,:,[0,2]] = img[:,:,[2,0]] # BGR → RGB
        return img
    
    def get_rgb_strip(self):
        img = self.get_rgb_image()
        return img[self.upper_pixel:self.lower_pixel]
    
    def get_rgb_strip_mean(self):
        strip = self.get_rgb_strip()
        strip[:] = np.mean(strip, axis = 0)
        return strip
    
    def get_hsv_strip(self):
        rgb_strip_mean = self.get_rgb_strip_mean()
        return cv2.cvtColor(rgb_strip_mean, cv2.COLOR_RGB2HSV)
    
    def get_rgb_mask(self, color = "yellow"):
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
    
    # ===== Garage detection =====
    def get_regions(self, color = "yellow", minimal_size = 10):
        mask = self.get_rgb_mask(color = color)
        return TurtleUtils.mask_regions(mask[0], minimal_size = minimal_size)
    
    def pixel2angle(self, p):
        p_mid = p - self.img_width // 2
        return math.atan(p_mid / self.p2a)
    
    def sample_garage(self, color = "yellow", sampling_step = 20, minimal_size = 10, 
                       r2p = False):
        regions = self.get_regions(color = color, minimal_size = minimal_size)
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
    
    def garage_entry_waypoints(self):
        regs = self.get_regions(color = "purple")
        
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
        rho2, phi2, = depth_img_strip[0][p2], self.pixel2angle(p2)
        P2 = TurtleUtils.polar2cartesian(rho2, phi2)
        
        # Garage entrance
        P_G = 0.5*(P1 + P2)
        
        # Vector t
        t = P2 - P1
        
        R_90 = TurtleUtils.R(90, degrees = True)
        
        # Vector n1, points away from garage
        n1 = R_90.T @ t
        
        # Vector n2, points into garage
        n2 = -n1
        
        # Camera offset correction
        CAMERA_OFFSET = 0.12
        point_offset = t / np.linalg.norm(t) * CAMERA_OFFSET
        
        # Pre-garage point
        P_PG = P_G + n1/np.linalg.norm(n1) * 0.55 - point_offset
        
        # Goal point
        P_GOAL = P_G + n2/np.linalg.norm(n2) * 0.25 - point_offset
        
        return [P1, P2, P_G, P_PG, P_GOAL]