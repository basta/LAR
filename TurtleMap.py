#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
import collections

import TurtleUtils

from sklearn.cluster import DBSCAN

class TurtlebotMap:
    def __init__(self, filter_eps = 0.85, filter_min_samples = 5):
        self.reset()

        # DBSCAN filter parameters
        self.filter_eps = filter_eps
        self.filter_min_samples = filter_min_samples

        # Downsampling parameters
        self.cell_size = 0.1

    def reset(self):
        self._points = None

    def add_points(self, points, odometry):
        # No points to add
        if len(points) == 0:
            return
       
        position, yaw = odometry
        points_transformed = np.array(TurtleUtils.transform_points(
                points, position, yaw)).T
        
        if self._points is None:
            self._points = points_transformed
        else:
            self._points = np.append(self._points , points_transformed, axis=1)

    def visualize(self):
        # Nothing to visualize
        if self._points is None:
            return
        
        points = self.points
        plt.scatter(points[0], points[1])
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        plt.show()

    def visualize_filtered(self):
        # Nothing to visualize
        if self._points is None:
            return
        
        points = self.points_filtered
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        plt.scatter(points[0], points[1])
        plt.show()

    @property
    def points(self):
        return self._points
    
    @property
    def cluster_count(self):
        if self._points is None:
            return 0
        # Find all possible clusters
        dbscan = DBSCAN(eps = self.filter_eps,
            min_samples = self.filter_min_samples).fit(self._points.T)
        # Remove outliers
        label_counter = collections.Counter(dbscan.labels_[dbscan.labels_ >= 0])
        return len(label_counter)

    @property
    def points_filtered(self):
        if self._points is None:
            return None

        dbscan = DBSCAN(eps = self.filter_eps,
            min_samples = self.filter_min_samples).fit(self._points.T)
        label_counter = collections.Counter(dbscan.labels_)
        most_common_label = sorted(list(label_counter.items()),
                        key = lambda x: x[1], reverse = True)[0][0]
        
        return self._points.T[np.where(dbscan.labels_ == most_common_label)].T
    
    @property
    def points_downsampled(self):
        if self._points is None:
            return None
        
        # Work with filtered points only
        points = self.points_filtered

        # Find the bounding box of the points
        x_min, x_max = np.min(points[0]), np.max(points[0])
        y_min, y_max = np.min(points[1]), np.max(points[1])

        # Grid size
        num_vox_x = int(math.ceil((x_max - x_min) / self.cell_size))
        num_vox_y = int(math.ceil((y_max - y_min) / self.cell_size))

        # Point grid
        point_grid = np.zeros((num_vox_x, num_vox_y, 2))
        count_grid = np.zeros((num_vox_x, num_vox_y))

        for i in points.shape[1]:
            pt = points[i]
            x_floored = math.floor((pt[0] - x_min) / self.cell_size)
            y_floored = math.floor((pt[1] - y_min) / self.cell_size)

            point_grid[x_floored, y_floored] += pt
            count_grid[x_floored, y_floored] += 1

        downsampled_points = []
        for i in range(num_vox_x):
            for j in range(num_vox_y):
                if count_grid[i, j] == 0:
                    continue
                downsampled_points.append(point_grid[i, j] / count_grid[i, j])

        return np.array(downsampled_points).T
