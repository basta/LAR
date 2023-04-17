import time, math, itertools
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Utilities
from TurtleUtils import R, plt2robot, robot2plt, get_quadrant

class TurtlebotICP:
    def __init__(self):
        # ICP iterations
        self.max_iters = 21
        
    def get_correspondence_indices(self,P, Q):
        # For each point in P, find the closest point in Q
        p_size = P.shape[1]
        q_size = Q.shape[1]
        correspondences = []
        for i in range(p_size):
            p_point = P[:, i]
            j = np.argmin(np.linalg.norm(Q - p_point.reshape(2,1), axis=0))
            correspondences.append((i, j))
        return correspondences

    def error(self, x, p_point, q_point):
        rotation = R(x[2])
        translation = x[0:2]
        prediction = rotation.dot(p_point) + translation
        return prediction - q_point

    def optimize(self,data, method = "SVD", extra_points = 0):
        MIN_ANGLE = 0
        MAX_ANGLE = 121
        ANGLE_STEP = 30
        optimum = None
        
        print("Optimizing ... ")
        t_start = time.perf_counter()
        
        # Optimize over initial garage orientations
        for abs_angle in range(MIN_ANGLE, MAX_ANGLE, ANGLE_STEP):
            
            for i in range(2):
                if abs_angle == 0 and i == 0:
                    continue
                    
                angle = abs_angle * (-1)**(i)

                print(f"Optimizing with initial angle {angle} degrees ...")

                # Optimize over garage initial configurations
                for left, back, right in itertools.product([0, 1], repeat=3):
                    if not sum((left, back, right)):
                        continue
                    
                    if sum((left , back , right)) == 1 and (left or right):
                        continue
                        
                    parameters = {"rotation" : angle, "degrees" : True, "translation" : np.zeros(shape = (2,1)),
                                  "left" : left, "right" : right, "back" : back, "N" : data.shape[1] + extra_points,
                                  "height" : 0.49 - 0.07, "width" : 0.59 - 0.07}

                    # Move garage towards the center of the data
                    garage = GarageModel(parameters)
                    mean = np.mean(data, axis = 1).reshape(2,1)
                    garage_g = garage.waypoints[:,0].reshape(2,1)
                    garage.apply_translation(-garage_g + mean)
                    
                    # Optimize
                    P_values, cost, corresp_values = self.icp_least_squares(garage, data)

                    # Tom's trick
                    BM = garage.BM.reshape(2)
                    G = garage.waypoints[:,0]
                    q1 = get_quadrant(G)
                    q2 = get_quadrant(G - BM)

                    if q1 != q2 and sum((left, right, back)) == 1:
                        cost += float('inf')

                    if optimum is None or cost < optimum.cost:
                        optimum = Optimum(P_values, cost, corresp_values, garage, data)

        t_finish = time.perf_counter()
        print("Done")
        print(f"Optimization time: {t_finish - t_start:.2f} s")
        return optimum
    
    def prepare_system(self, x, garage, Q, correspondences):
        P = garage.sampled
        H = np.zeros((3, 3))
        g = np.zeros((3, 1))
        chi = 0

        def kernel(x):
            return 1 if np.linalg.norm(x) < 3.0 else 0

        for i, j in correspondences:
            p_point = P[:, [i]]
            q_point = Q[:, [j]]
            e = self.error(x, p_point, q_point)
            weight = kernel(e) 
            if weight != 0:
                J = self.jacobian(x, p_point)
                H += weight * J.T.dot(J)
                g += weight * J.T.dot(e)
                chi += e.T * e
        return H, g
    
    def icp_least_squares(self,garage_model, Q):
        x = np.zeros((3, 1))
        x_values = [x.copy()]  # Initial value for transformation.
        P_values = [garage_model.sampled.copy()]
        corresp_values = []
        for i in range(self.max_iters):
            rot = R(x[2])
            t = x[0:2]
            correspondences = self.get_correspondence_indices(garage_model.sampled, Q)
            corresp_values.append(correspondences)
            correspondences2 = self.get_correspondence_indices(Q, garage_model.sampled)
            correspondences2 = [(a,b) for b,a in correspondences2]
            correspondences.extend(correspondences2)
            H, g = self.prepare_system(x, garage_model, Q, correspondences)
            dx = np.linalg.lstsq(H, -g, rcond=None)[0]
            x += dx
            x[2] = math.atan2(math.sin(x[2]), math.cos(x[2])) # normalize angle
            x_values.append(x.copy())
            rot = R(x[2])
            t = x[0:2]
            garage_model.apply_rotation(rot)
            garage_model.apply_translation(t)
            P_values.append(garage_model.sampled.copy())
        corresp_values.append(corresp_values[-1])
    
        def closest_point(data, point):
                delta = data - point.reshape(2,1)
                norms = np.linalg.norm(delta, axis = 0)
                return np.min(norms)

        # Total cost
        cost_g = (closest_point(garage_model.sampled, Q[:,i])**2 for i in range(Q.shape[1]))
        cost = sum(cost_g)

        return P_values, cost, corresp_values

    def jacobian(self,x, p_point):
        theta = x[2]
        J = np.zeros((2, 3))
        J[0:2, 0:2] = np.identity(2)
        dR = np.array([[0, -1], [1, 0]])
        J[0:2, [2]] = (self.dR(theta)).dot(p_point)
        return J

    def dR(self,theta):
        s, c = math.sin(theta), math.cos(theta)
        return np.array([[-s, -c],
                         [c,  -s]])
    
class GarageModel:
    def __init__(self, params):
        HEIGHT, WIDTH = params["height"], params["width"]

        self.left_init = params["left"]
        self.right_init = params["right"]
        self.back_init = params["back"]
        
        # Garage:
        # LF        RF
        # |         |
        # |         |
        # LB --BM-- RB
        
        # LF, LB, RB, RF
        # | 
        # LB, RB, LF, RF
        self.corners = np.array([
            [0.0, WIDTH, 0.0,    WIDTH],
            [0.0, 0.0,   HEIGHT, HEIGHT]
        ])

        CLEARANCE = 0.4
        
        self.BM = np.array([WIDTH/2, 0]).reshape(2,1)
        
        ## Waypoints
        # 0. Garage mid-points
        self.gmp = np.array([np.mean(self.corners[0,:]), np.mean(self.corners[1,:])]).reshape(2,1)
        
        # 1. Pre-Garage
        ent = np.array([np.mean(self.corners[0,:]), HEIGHT + CLEARANCE]).reshape(2,1)
        
        # 2.
        p2 = np.array([-CLEARANCE, HEIGHT + CLEARANCE]).reshape(2,1)
        
        # 3.
        p3 = np.array([WIDTH + CLEARANCE, HEIGHT + CLEARANCE]).reshape(2,1)
        
        # 4.
        p4 = np.array([-CLEARANCE, HEIGHT/2]).reshape(2,1)
        
        # 5.
        p5 = np.array([WIDTH + CLEARANCE, HEIGHT/2]).reshape(2,1)

        # 6.
        p6 = np.array([-CLEARANCE, -CLEARANCE]).reshape(2,1)

        # 7.
        p7 = np.array([WIDTH + CLEARANCE, -CLEARANCE]).reshape(2,1)
        
        # 8.
        p8 = np.array([WIDTH / 2, -CLEARANCE]).reshape(2,1)
        
        self.waypoints = np.hstack((self.gmp.copy(), ent, p2, p3, p4, p5, p6, p7, p8))
        
        # Generate sampled garage model
        self.sampled = self._sample_garage(params)
        
        # Apply initial rotation and translation
        self.apply_rotation(R(params["rotation"], params["degrees"]))
        self.apply_translation(params["translation"])
        
        # Route to the garage
        self.route = {8:6, 6:4, 4:2, 2:1, 7:5, 5:3, 3:1, 1:0}
        
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
        self.gmp = rot @ self.gmp
        self.BM = rot @ self.BM
        
    def apply_translation(self, t):
        self.corners += t
        self.waypoints += t
        self.sampled += t
        self.gmp += t
        self.BM += t
        
    def closest_waypoint(self, position):
            
        # Got odometry
        if isinstance(position, tuple):
            position = position[0]
        
        idx = None
        if isinstance(position, np.ndarray):
            position_plt = robot2plt([position])[0]
            position_plt = position_plt.reshape(2,1)
            waypoint_dist = np.linalg.norm(self.waypoints[:,1:] - position_plt, axis = 0)
            idx = np.argmin(waypoint_dist) + 1
        elif isinstance(position, (np.int64, np.int32, int)):
            idx = self.route[position]

        should_scan = idx in [1,2,3,6,7]
            
        return plt2robot([self.waypoints[:,idx]])[0] , idx, should_scan
    
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