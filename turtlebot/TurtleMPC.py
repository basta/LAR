import casadi, rospy, threading, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class TurtlebotMPC:
    def __init__(self, policy, odom, rate):
        self.policy = policy
        self.odom = odom
        self.rate = rospy.Rate(rate)

        # Optimized trajectories and policies
        self.X_sol = None
        self.U_sol = None
        self.T_sol = None

        self.pi_x = None
        self.pi_y = None

    def setup(self):
        # Define differential drive kinematics
        x = casadi.SX.sym("x", 3, 1)
        u = casadi.SX.sym("u", 2, 1)

        n_states = x.numel()
        n_inputs = u.numel()

        x_dot = casadi.SX.sym("x_dot", 3, 1)
        x_dot[0] = casadi.cos(x[2]) * u[0]
        x_dot[1] = casadi.sin(x[2]) * u[0]
        x_dot[2] = u[1]

        f = casadi.Function(
            "f", [x, u], [x_dot], ["x", "u"], ["x_dot"]
        )

        # MPC parameters
        N = 50
        dt = 0.02
        T = N * dt
        print(f"MPC's prediction horizon is {T} seconds.")
        self.N = N
        self.dt = dt

        # State cost
        Qx, Qy, Qtheta = 100, 100, 10
        Q = casadi.diagcat(Qx, Qy, Qtheta)
        Qf = Q * 2

        # Input cost
        Rv, Romega = 3, 1
        R = casadi.diagcat(Rv, Romega) * 10

        # Input vel cost
        Vv, Vomega = 0.1, 0.1
        V = casadi.diagcat(Vv, Vomega) * 4000

        # Define optimization variables
        X = casadi.SX.sym("X", n_states, N+1)
        U = casadi.SX.sym("U", n_inputs, N)

        # Initial state parameter
        x_init = casadi.SX.sym("x_init", n_states, 1)

        # Reference state parameter
        x_ref = casadi.SX.sym("x_ref", n_states, 1)

        # Simulator
        def rk4(f, x, u, dt):
            k1 = f(x, u)
            k2 = f(x + dt/2 * k1, u)
            k3 = f(x + dt/2 * k2, u)
            k4 = f(x + dt * k3, u)
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Define kinematics constraints
        g = X[:, 0] - x_init
        for i in range(N):
            x_current, u_current, x_next = X[:, i], U[:, i], X[:, i+1]
            x_next_rk4 = rk4(f, x_current, u_current, dt)
            g = casadi.vertcat(g, x_next_rk4 - x_next)

        # Define cost
        cost = 0.0
        for i in range(N+1):
            x_current = X[:, i]
            x_error= x_current - x_ref
            
            if i < N:
                cost += x_error.T @ Q @ x_error 
            else:
                cost += x_error.T @ Qf @ x_error 
            
        for i in range(N):
            u_current = U[:, i]
            u_error = u_current
            cost += u_error.T @ R @ u_error
            
        for i in range(N-1):
            u_current, u_next = U[:, i], U[:, i+1]
            u_diff = u_next - u_current
            cost += u_diff.T @ V @ u_diff

        decision_variables = casadi.vertcat(
            X.reshape((-1, 1)),
            U.reshape((-1, 1))
        ) 

        # Setup bounds
        v_max = 1.0
        v_min = -v_max
        omega_max = 0.5
        omega_min = -omega_max


        lbx = casadi.DM.zeros((n_states*(N+1) + n_inputs*N, 1))
        ubx = casadi.DM.zeros((n_states*(N+1) + n_inputs*N, 1))

        lbg = casadi.DM.zeros((g.numel(), 1))
        ubg = casadi.DM.zeros((g.numel(), 1))

        lbx[0: n_states*(N+1): n_states] = -casadi.inf  # x lower bound
        lbx[1: n_states*(N+1): n_states] = -casadi.inf  # y lower bound
        lbx[2: n_states*(N+1): n_states] = -casadi.inf  # theta lower bound
        lbx[n_states*(N+1)::n_inputs] = v_min   # v lower bound
        lbx[n_states*(N+1)+1::n_inputs] = omega_min  # omega lower bound


        ubx[0: n_states*(N+1): n_states] = casadi.inf   # x upper bound
        ubx[1: n_states*(N+1): n_states] = casadi.inf   # y upper bound
        ubx[2: n_states*(N+1): n_states] = casadi.inf   # theta upper bound
        ubx[n_states*(N+1)::n_inputs] = v_max   # v lower bound
        ubx[n_states*(N+1)+1::n_inputs] = omega_max  # omega lower bound

        # lbx[n_states*(N+1)] = 0
        # lbx[n_states*(N+1) + 1] = 0
        # ubx[n_states*(N+1)] = 0
        # ubx[n_states*(N+1) + 1] = 0

        # Parameters
        P = casadi.vertcat(
            x_init.reshape((-1, 1)),
            x_ref.reshape((-1, 1))
        )

        nlp_prob = {
            'f': cost,
            'x': decision_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 5,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 1
        }

        self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)

        self.args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }

    def solve(self, initial_state, final_state):
        # Current time
        t0 = rospy.Time.now().to_sec()

        n_states = 3
        n_inputs = 2

        # Initial guess
        X0 = casadi.repmat(initial_state, 1, self.N+1)
        U0 = casadi.repmat(np.zeros(n_inputs), 1, self.N)

        # Initial guess for the decision variables
        dec0 = casadi.vertcat(
            X0.reshape((-1,1)),
            U0.reshape((-1,1))
        )

        s_param = casadi.vertcat(
            initial_state,
            final_state
        )

        solution = self.solver(
            x0 = dec0,
            lbx = self.args["lbx"],
            ubx = self.args["ubx"],
            lbg = self.args["lbg"],
            ubg = self.args["ubg"],
            p = s_param
        )

        self.X_sol = solution["x"][:n_states*(self.N+1)].reshape((n_states,-1))
        U_sol = solution["x"][n_states*(self.N+1):].reshape((n_inputs,-1))
        self.U_sol = casadi.horzcat(U_sol, U_sol[:,-1])
        self.T_sol = np.arange(0, self.N+1) * self.dt + t0

        self.pi_x = interp1d(self.T_sol, self.X_sol)
        self.pi_u = interp1d(self.T_sol, self.U_sol)

    def update_policy(self):
        lock = self.policy.policy_lock
        with lock:
            self.policy.pi_x = self.pi_x
            self.policy.pi_u = self.pi_u

        lock = self.policy.update_lock
        with lock:
            self.policy.policy_updated = True

    def start(self):
        self.setup()
        self.run()

    def run(self):
        while not rospy.is_shutdown():
            current_state = self.odom.get_odometry()
            FINAL_STATE = np.array([1.0, 0.0, 0.0])
            self.solve(current_state, FINAL_STATE)
            self.update_policy()
            self.rate.sleep()

class TurtlebotPolicy:
    def __init__(self):
        self.pi_x = None
        self.pi_y = None
        self.policy_lock = threading.Lock()

        self.policy_updated = False
        self.update_lock = threading.Lock()

class TurtlebotDriver:
    def __init__(self, policy, turtle, rate):
        self.policy = policy
        self.rate = rospy.Rate(rate)
        self.turtle = turtle

    def start(self):
        self.wait_for_policy()
        self.run()

    def wait_for_policy(self):
        while True:
            lock = self.policy.update_lock
            with lock:
                if self.policy.policy_updated:
                    break
            time.sleep(0.3)

    def check_new_policy(self):
        lock = self.policy.update_lock
        ret = False
        with lock:
            ret = self.policy.policy_updated
        return ret
    
    def update_policy(self):
        lock = self.policy.update_lock
        with lock:
            self.policy.policy_updated = False
        
        lock = self.policy.policy_lock
        with lock:
            self.pi_x = self.policy.pi_x
            self.pi_u = self.policy.pi_u

        print("New policy received!")
    
    def run(self):
        while not rospy.is_shutdown():
            if self.check_new_policy():
                self.update_policy()

            t =  rospy.Time.now().to_sec()
            x = self.pi_x(t)
            u = self.pi_u(t)

            linear_velocity = u[0]
            angular_velocity = u[1]

            self.turtle.cmd_velocity(
                linear = linear_velocity,
                angular = angular_velocity
            )

            self.rate.sleep()