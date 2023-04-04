#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import collections

import TurtleUtils

from sklearn.cluster import DBSCAN

class TurtlebotMap:
    def __init__(self, filter_eps = 0.85, filter_min_samples = 5):
        self.reset()

        self.filter_eps = filter_eps
        self.filter_min_samples = filter_min_samples

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
    def points_filtered(self):
        if self._points is None:
            return None

        dbscan = DBSCAN(eps = self.filter_eps,
            min_samples = self.filter_min_samples).fit(self._points.T)
        label_counter = collections.Counter(dbscan.labels_)
        most_common_label = sorted(list(label_counter.items()),
                        key = lambda x: x[1], reverse = True)[0][0]
        
        return self._points.T[np.where(dbscan.labels_ == most_common_label)].T