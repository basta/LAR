import time,math,random
import itertools, collections, functools
import numpy as np
from sklearn.cluster import DBSCAN

# Visualization
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Utilities
from TurtleUtils import R, plot_fitted_garage, plt2robot
from TurtleControllers import TurtlebotController

class TurtlebotICP:
    def __init__(self):
        # ICP iterations
        self.max_iters = 10
        
        # Max distance, to filter out outliers
        self.threshold = 0.75
    
    def kernel(self, error):
        return 1.0 if np.linalg.norm(error) < self.threshold else 0.0
    
    def icp_svd(self, garage_model, Q):
        # ICP using SVD
        center_of_Q, Q_centered = self.center_data(Q)
        cost = None
        P_values = [garage_model.sampled.copy()]
        corresp_values = []
        exclude_indices = []
        
        def closest_point(data, point):
            delta = data - np.reshape(point, (-1, 1))
            norms = np.linalg.norm(delta, axis = 0)
            return np.min(norms)
    
        cost_angle = 0
        for i in range(self.max_iters):
            center_of_P, P_centered = self.center_data(garage_model.sampled, exclude_indices=exclude_indices)
            correspondences = self.get_correspondence_indices(P_centered, Q_centered)
            corresp_values.append(correspondences)
            cost = np.linalg.norm(P_centered - Q_centered)
            
            
            if i == self.max_iters - 1:
                g = (closest_point(P_centered, Q_centered[:,i])**2 for i in range(P_centered.shape[1]))
#                 g_minor = (closest_point(Q_centered, garage_model.corners[:,i]) * 0.0001 for i in range(4))
                cost = sum(g) #+ sum(g_minor)
            
            cov, exclude_indices = self.compute_cross_covariance(P_centered, Q_centered, correspondences)
            U, S, V_T = np.linalg.svd(cov)
            R = U.dot(V_T) 
            
            cost_angle += abs(math.degrees(math.atan2(R[0,0], R[1,0])) * 0.0001)
            
            t = center_of_Q - R.dot(center_of_P)  
            garage_model.apply_rotation(R)
            garage_model.apply_translation(t)
            P_values.append(garage_model.sampled.copy())
        corresp_values.append(corresp_values[-1])
        return P_values, cost, corresp_values

    def get_correspondence_indices(self,P, Q):
        # For each point in P, find the closest point in Q
        p_size = P.shape[1]
        q_size = Q.shape[1]
        correspondences = []
        for i in range(p_size):
            p_point = P[:, i]
            min_dist = float('inf')
            chosen_idx = -1
            for j in range(q_size):
                q_point = Q[:, j]
                dist = np.linalg.norm(q_point - p_point)
                if dist < min_dist:
                    min_dist = dist
                    chosen_idx = j
            correspondences.append((i, chosen_idx))
        return correspondences

    def error(self, x, p_point, q_point):
        rotation = R(x[2])
        translation = x[0:2]
        prediction = rotation.dot(p_point) + translation
        return prediction - q_point

    def center_data(self,data, exclude_indices=[]):
        reduced_data = np.delete(data, exclude_indices, axis=1)
        center = np.array([reduced_data.mean(axis=1)]).T
        return center, data - center

    def compute_cross_covariance(self,P, Q, correspondences):
        cov = np.zeros((2, 2))
        exclude_indices = []
        for i, j in correspondences:
            p_point = P[:, [i]]
            q_point = Q[:, [j]]
            weight = self.kernel(p_point - q_point)
            if weight < 0.01: exclude_indices.append(i)
            cov += weight * q_point.dot(p_point.T)
        return cov, exclude_indices
    
    def filter_data(self,data):
        dbscan = DBSCAN(eps = 0.85).fit(data.T)
        lbl = self.most_common_label(dbscan)
        
        print(data.T[np.where(dbscan.labels_ != lbl)].T)
        return data.T[np.where(dbscan.labels_ == lbl)].T
    
    def most_common_label(self,dbscan):
        cnt = collections.Counter(dbscan.labels_)
        lst = list(cnt.items())
        lst = sorted(lst, key = lambda x: x[1], reverse = True)
        
        print(f"Filtered {len(dbscan.labels_) - lst[0][1]} points")
        return lst[0][0]

    def optimize(self,data):
        MIN_ANGLE = 0
        MAX_ANGLE = 121
        ANGLE_STEP = 60
        optimum = None
        
        # Filter data
        data = self.filter_data(data)

        print("Optimizing ... ", end = "")
        t_start = time.perf_counter()
        
        # Optimize over initial garage orientations
        for abs_angle in range(MIN_ANGLE, MAX_ANGLE, ANGLE_STEP):
            
            for i in range(2):
                if abs_angle == 0 and i == 0:
                    continue
                    
                angle = abs_angle * (-1)**(i)
                
                print("Angle", angle)
                # Optimize over garage initial configurations
                for left, back, right in itertools.product([0, 1], repeat=3):
                    if not sum((left, back, right)):
                        continue
                    if sum((left , back , right)) == 1 and (left or right):
                        continue

                    parameters = {"rotation" : angle, "degrees" : True, "translation" : np.zeros(shape = (2,1)),
                                  "left" : left, "right" : right, "back" : back, "N" : data.shape[1],
                                  "height" : 0.49, "width" : 0.59}

                    garage = GarageModel(parameters)

                    P_values, cost, corresp_values = self.icp_svd(garage, data)

                    if optimum is None or cost < optimum.cost:
                        optimum = Optimum(P_values, cost, corresp_values, garage, data)

        t_finish = time.perf_counter()
        print("Done")
        print(f"Optimization time: {t_finish - t_start:.2f} s")
        return optimum
    
class GarageModel:
    def __init__(self, params):
        HEIGHT, WIDTH = params["height"], params["width"]
        
        # Garage:
        # LF        RF
        # |         |
        # |         |
        # LB ----- RB
        
        # LF, LB, RB, RF
        # | 
        # LB, RB, LF, RF
        self.corners = np.array([
            [0.0, WIDTH, 0.0,    WIDTH],
            [0.0, 0.0,   HEIGHT, HEIGHT]
        ])
        
        CLEARANCE = 0.7
        
        ## Waypoints
        # 0. Garage mid-points
        gmp = np.array([np.mean(self.corners[0,:]), np.mean(self.corners[1,:])]).reshape(2,1)
        
        # 1. Pre-Garage
        ent = np.array([np.mean(self.corners[0,:]), HEIGHT + 1.5 * CLEARANCE]).reshape(2,1)
        
        # 2.
        p2 = np.array([0.0, HEIGHT + CLEARANCE]).reshape(2,1)
        
        # 3.
        p3 = np.array([WIDTH, HEIGHT + CLEARANCE]).reshape(2,1)
        
        # 4.
        p4 = np.array([-CLEARANCE, HEIGHT]).reshape(2,1)
        
        # 5.
        p5 = np.array([WIDTH + CLEARANCE, HEIGHT]).reshape(2,1)
        
        # 6.
        p6 = np.array([-CLEARANCE, 0]).reshape(2,1)
        
        # 7.
        p7 = np.array([WIDTH + CLEARANCE, 0]).reshape(2,1)
        
        # 8.
        p8 = np.array([0, -CLEARANCE]).reshape(2,1)
        
        # 9.
        p9 = np.array([WIDTH, -CLEARANCE]).reshape(2,1)
        
        self.waypoints = np.hstack((gmp, ent, p2, p3, p4, p5, p6, p7, p8, p9))
        
        # Generate sampled garage model
        self.sampled = self._sample_garage(params)
        
        # Apply initial rotation and translation
        self.apply_rotation(R(params["rotation"], params["degrees"]))
        self.apply_translation(params["translation"])
        
        # Route to the garage
        self.route = {8:6, 6:4, 4:2, 2:1, 9:7, 7:5, 5:3, 3:1}
        
    def _sample_garage(self, params):
        HEIGHT, WIDTH = params["height"], params["width"]
        n_sides = sum((params["left"], params["right"], params["back"]))
        
        # Uniformly sample garage
        side_pts = [params["N"] // n_sides + (1 if x < params["N"] % n_sides else 0) for x in range(n_sides)]
        i = 0
        
        sides = list()
        
        # Generate left side
        if params["left"]:
            left, i = np.array([
                np.zeros(shape = (side_pts[i],)),
                np.linspace(start = 0.0, stop = HEIGHT, num = side_pts[i])
            ]), i + 1
            sides.append(left)
            
        # Generate back side
        if params["back"]:
            back, i = np.array([
                np.linspace(start = 0.0, stop = WIDTH, num = side_pts[i]),
                np.zeros(shape = (side_pts[i],))
            ]), i + 1
            sides.append(back)
            
        # Generate right side
        if params["right"]:
            right, i = np.array([
                np.ones(shape = (side_pts[i],)) * WIDTH,
                np.linspace(start = 0.0, stop = HEIGHT, num = side_pts[i])
            ]), i + 1
            sides.append(right)
        
        return np.hstack(sides)
        
    def apply_rotation(self, rot):
        self.corners = rot @ self.corners
        self.waypoints = rot @ self.waypoints
        self.sampled = rot @ self.sampled
        
    def apply_translation(self, t):
        self.corners += t
        self.waypoints += t
        self.sampled += t
        
    def closest_waypoint(self, position):
            
        # God odometry
        if isinstance(position, tuple):
            position = position[0]
        
        idx = None
        if isinstance(position, np.ndarray):
            position = position.reshape(2,1)
            waypoint_dist = np.linalg.norm(self.waypoints - position, axis = 0)
            idx = np.argmin(waypoint_dist)
        elif isinstance(position, (np.int64, np.int32, int)):
            idx = self.route[position]
            
        return plt2robot([self.waypoints[:,idx]])[0] , idx
    
class Optimum:
    def __init__(self, P_values, cost, corresp_values, garage, Q):
        self.P_values = P_values
        self.cost = cost
        self.corresp_values = corresp_values
        self.garage = garage
        self.Q = Q
        
    def animate_results(self, xlim=(-2, 2), ylim=(-2, 2), save_file = None):
        fig = plt.figure(figsize=(10, 6))
        anim_ax = fig.add_subplot(111)
        anim_ax.set(xlim=xlim, ylim=ylim)
        anim_ax.set_aspect('equal')
        plt.close()
        x_q, y_q = self.Q
        # draw initial correspondeces
        corresp_lines = []
        for i, j in self.corresp_values[0]:
            corresp_lines.append(anim_ax.plot([], [], 'grey')[0])
        # Prepare Q data.
        Q_line, = anim_ax.plot(x_q, y_q, 'o', color='orangered')
        # prepare empty line for moved data
        P_line, = anim_ax.plot([], [], 'o', color='#336699')

        def animate(i):
            P_inc = self.P_values[i]
            x_p, y_p = P_inc
            P_line.set_data(x_p, y_p)
            draw_inc_corresp(P_inc, self.Q, self.corresp_values[i])
            return (P_line,)

        def draw_inc_corresp(points_from, points_to, correspondences):
            for corr_idx, (i, j) in enumerate(correspondences):
                x = [points_from[0, i], points_to[0, j]]
                y = [points_from[1, i], points_to[1, j]]
                corresp_lines[corr_idx].set_data(x, y)

        anim = animation.FuncAnimation(fig, animate,
                                       frames=len(self.P_values), 
                                       interval=500, 
                                       blit=True)
    
        if save_file is not None:
            anim.save(save_file)
        return HTML(anim.to_jshtml())