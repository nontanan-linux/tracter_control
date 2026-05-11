#!/usr/bin/env python3
import os
# Limit OpenBLAS/Numpy threading to reduce CPU overhead for small matrix ops
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Point, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_auto_control_msgs.msg import AckermannControlCommand
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult
import numpy as np
import math
import sys
import os
import yaml
import csv
import csv

from ament_index_python.packages import get_package_share_directory

# MPCController class remains same...

# Load Config
config_path = os.path.join(get_package_share_directory('tracter_control'), 'config/mpc_config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Config file not found at {config_path}")
    sys.exit(1)

# --- 1. System Parameters ---
DT = config['system']['dt']
TRAJ_RESAMPLE_DIST = config['system']['traj_resample_dist']

# --- 2. Vehicle Parameters ---
L = config['vehicle']['wheelbase']
TRACK_WIDTH = config['vehicle']['track_width']
MAX_STEER = config['vehicle']['max_steer']
MAX_ACCEL = config['vehicle']['max_accel']
WHEEL_RADIUS = config['vehicle']['wheel_radius']
CHASSIS_LENGTH = config['vehicle']['chassis_length']
CHASSIS_WIDTH = config['vehicle']['chassis_width']
CHASSIS_HEIGHT = config['vehicle']['chassis_height']

# --- 3. Path Smoothing Parameters ---
REF_PATH_FILE = config['path_smoothing']['reference_path_file']
REF_TRAJ_TOPIC = config['path_smoothing']['ref_traj_topic']
CONTROL_CMD_TOPIC = config['system']['control_cmd_topic']

# --- 4. MPC Optimization Parameters ---
N = config['mpc']['prediction_horizon']
DT_PRED = config['mpc']['prediction_dt']
MIN_PRED_LEN = config['mpc']['min_prediction_length']

# Weights
Q_EY = config['mpc']['weights']['lat_error']
Q_ETHETA = config['mpc']['weights']['heading_error']
Q_EV = config['mpc']['weights']['velocity_error']

R_STEER = config['mpc']['weights']['steering_input']
R_ACCEL = config['mpc']['weights']['acceleration_input']

W_STEER_RATE = config['mpc']['weights']['steer_rate']
W_STEER_ACC = config['mpc']['weights']['steer_acc']
W_LAT_JERK = config['mpc']['weights']['lat_jerk']

# Hard Constraints
MAX_ACCEL = config['mpc']['constraints'].get('acceleration_limit', MAX_ACCEL)

# --- 5. Debug Parameters ---
DEBUG_LOG_INTERVAL = config['debug'].get('logging_interval', 10)



class MPCController:
    """
    Linearized Error-State Model Predictive Controller.
    State: x = [e_y, e_theta, e_v]
    Input: u = [delta_diff, accel] (Steering angle diff from ref, acceleration)
    """
    def __init__(self):
        # Weights matrices
        self.Q = np.diag([Q_EY, Q_ETHETA, Q_EV]) 
        self.R = np.diag([R_STEER, R_ACCEL])
        
        # Pre-calculate constant weight matrices to save CPU
        self.Q_bar = np.kron(np.eye(N), self.Q)
        self.R_bar = np.kron(np.eye(N), self.R)
        
        # Additional weights that need to be mutable
        self.w_steer_rate = W_STEER_RATE
        self.w_lat_jerk = W_LAT_JERK

    def update_weights(self, q_ey, q_etheta, q_ev, w_steer_rate, w_lat_jerk):
        """
        Update MPC weights dynamically from ROS parameters.
        """
        # Update Q matrix
        self.Q = np.diag([q_ey, q_etheta, q_ev])
        self.Q_bar = np.kron(np.eye(N), self.Q)
        
        # Update additional weights
        self.w_steer_rate = w_steer_rate
        self.w_lat_jerk = w_lat_jerk
        
    def get_linear_error_model(self, v_ref, delta_ref):
        """
        Returns A, B matrices for discrete time error dynamics.
        Uses prediction_dt (DT_PRED).
        """
        cos_sq_delta = math.cos(delta_ref)**2
        if cos_sq_delta < 0.01: cos_sq_delta = 0.01
        
        # A matrix
        A = np.array([
            [1.0, v_ref * DT_PRED, 0.0],
            [0.0, 1.0, (math.tan(delta_ref) / L) * DT_PRED],
            [0.0, 0.0, 1.0]
        ])
        
        # B matrix
        B = np.array([
            [0.0, 0.0],
            [(v_ref * DT_PRED) / (L * cos_sq_delta), 0.0],
            [0.0, DT_PRED]
        ])
        
        return A, B

    def solve(self, error_state, v_ref, kappa_ref):
        """
        Solves the MPC problem for lateral control.
        error_state: [e_y, e_theta]
        v_ref: current reference velocity
        kappa_ref: current reference curvature
        """
        # 1. Calculate Reference Steering (Ackermann)
        # delta_ref = atan(L * kappa)
        delta_ref = math.atan(L * kappa_ref)
        
        # 2. Get Linear Model matrices
        A, B = self.get_linear_error_model(v_ref, delta_ref)
        
        nx = 3
        nu = 2
        
        # 3. Formulate QP Matrices (Batch)
        # We want to minimize Sum( e^T Q e + (delta - delta_ref)^T R (delta - delta_ref) )
        # Let du = delta - delta_ref
        # e[k+1] = A e[k] + B du[k]
        
        S_x = np.zeros((N * nx, nx))
        S_u = np.zeros((N * nx, N * nu))
        
        S_x[0:nx, :] = A
        for i in range(1, N):
            S_x[i*nx:(i+1)*nx, :] = np.dot(A, S_x[(i-1)*nx:i*nx, :])
            
        S_u[0:nx, 0:nu] = B
        for i in range(1, N):
             S_u[i*nx:(i+1)*nx, 0:nu] = np.dot(A, S_u[(i-1)*nx:i*nx, 0:nu])
             
        for j in range(1, N):
            S_u[j*nx:, j*nu:(j+1)*nu] = S_u[0:(N-j)*nx, 0:nu]
        
        # Q_bar and R_bar are now pre-calculated in __init__
        
        # --- Advanced Weights (Smoothness) ---
        # Add Steering Rate and Acceleration Penalties
        # Matrix to penalize differences in inputs: D * U
        # D_block = [[1, 0], [-1, 1], [0, -1, 1]...]
        
        # Steering Rate Penalty (W_STEER_RATE)
        # We penalize (u_k - u_{k-1})
        H_rate = np.zeros((N * nu, N * nu))
        for i in range(1, N):
            # Penalize (delta_i - delta_{i-1})
            # index for delta_i is i*nu, delta_{i-1} is (i-1)*nu
            idx_curr = i * nu
            idx_prev = (i - 1) * nu
            
            # Rate weight for steering
            H_rate[idx_curr, idx_curr] += self.w_steer_rate
            H_rate[idx_prev, idx_prev] += self.w_steer_rate
            H_rate[idx_curr, idx_prev] -= self.w_steer_rate
            H_rate[idx_prev, idx_curr] -= self.w_steer_rate
            
            # Also apply some lateral jerk penalty proportional to steer rate * v
            # jerk ~ d(v^2 delta/L)/dt ~ (v^2/L) * (delta_rate)
            jerk_weight = self.w_lat_jerk * (v_ref**2 / L)**2
            H_rate[idx_curr, idx_curr] += jerk_weight
            H_rate[idx_prev, idx_prev] += jerk_weight
            H_rate[idx_curr, idx_prev] -= jerk_weight
            H_rate[idx_prev, idx_curr] -= jerk_weight

        # Steering Acceleration Penalty (W_STEER_ACC)
        # Penalize (delta_i - 2*delta_{i-1} + delta_{i-2})
        H_acc = np.zeros((N * nu, N * nu))
        for i in range(2, N):
            idx_0 = i * nu
            idx_1 = (i - 1) * nu
            idx_2 = (i - 2) * nu
            
            # Vector v = [1, -2, 1]
            # Contribution to H: w * v * v^T
            v_vec = np.array([1, -2, 1])
            comp = np.outer(v_vec, v_vec) * W_STEER_ACC
            
            for ii, ri in enumerate([idx_0, idx_1, idx_2]):
                for jj, rj in enumerate([idx_0, idx_1, idx_2]):
                    H_acc[ri, rj] += comp[ii, jj]

        # 4. Formulate QP Total
        x0 = error_state
        H = 2 * (np.dot(S_u.T, np.dot(self.Q_bar, S_u)) + self.R_bar + H_rate + H_acc)
        
        # f = 2 * (S_x * x0)^T * Q * S_u
        # Note: (Sx x0) is (N*nx, 1), Q is (N*nx, N*nx), Su is (N*nx, N*nu)
        # Result should be (1, N*nu) -> Transpose to (N*nu, 1) or vector
        
        term1 = np.dot(S_x, x0) # Predicted free response states
        f = 2 * np.dot(term1.T, np.dot(self.Q_bar, S_u)).flatten()
        
        # Solve Unconstrained: H U = -f
        try:
            # Add small regularization to H to ensure invertibility
            H += np.eye(H.shape[0]) * 1e-6
            du_opt_full = np.linalg.solve(H, -f) # Full horizon inputs
            du_opt = du_opt_full
        except np.linalg.LinAlgError:
            print("Singular matrix!")
            return delta_ref, 0.0, []

        # Optimal inputs
        # u = [delta_diff, accel]
        du_opt = du_opt[0:nu] # First step inputs
        delta_opt = delta_ref + du_opt[0]
        accel_opt = du_opt[1]
        
        # Predicted Error States for visualization
        X_pred = np.dot(S_x, x0) + np.dot(S_u, du_opt_full) # Need full U for prediction
        return delta_opt, accel_opt, X_pred

class MPCControllerNode(Node):
    def __init__(self):
        super().__init__('mpc_controller_node')
        
        self.controller = MPCController()
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/reference_path', 10)
        self.predicted_path_pub = self.create_publisher(Path, '/predicted_trajectory', 10)
        self.control_pub = self.create_publisher(AckermannControlCommand, CONTROL_CMD_TOPIC, 10)
        self.debug_pub = self.create_publisher(Marker, '/mpc/debug_target', 10)
        self.error_pub = self.create_publisher(Float64MultiArray, '/mpc/error_status', 10)
        
        # Subscribers
        self.traj_sub = self.create_subscription(Trajectory, REF_TRAJ_TOPIC + "_data", self.reference_traj_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        
        # State
        self.ref_traj = None 
        self.current_state = None # [x, y, theta, v]
        
        # Timer
        self.timer = self.create_timer(DT, self.timer_callback)

        # Dynamic Parameters (GWO Tuning)
        # Default to False if not in config
        gwo_enabled = config['mpc'].get('enable_gwo_tuning', False) 
        self.declare_parameter('enable_gwo_tuning', gwo_enabled)
        # Declare weights with initial config values
        self.declare_parameter('mpc.weights.lat_error', float(Q_EY))
        self.declare_parameter('mpc.weights.heading_error', float(Q_ETHETA))
        self.declare_parameter('mpc.weights.velocity_error', float(Q_EV))
        self.declare_parameter('mpc.weights.steer_rate', float(W_STEER_RATE))
        self.declare_parameter('mpc.weights.lat_jerk', float(W_LAT_JERK))

        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.get_logger().info("MPC Controller Node Started (Pure Controller)")

    def parameter_callback(self, params):
        # 1. Check flag status (current or new)
        enable_tuning = self.get_parameter('enable_gwo_tuning').value
        for p in params:
            if p.name == 'enable_gwo_tuning':
                enable_tuning = p.value

        # 2. Collect current weights
        current_weights = {
            'lat_error': self.get_parameter('mpc.weights.lat_error').value,
            'heading_error': self.get_parameter('mpc.weights.heading_error').value,
            'velocity_error': self.get_parameter('mpc.weights.velocity_error').value,
            'steer_rate': self.get_parameter('mpc.weights.steer_rate').value,
            'lat_jerk': self.get_parameter('mpc.weights.lat_jerk').value
        }

        # 3. Process updates
        weights_changed = False
        for p in params:
            if p.name == 'enable_gwo_tuning':
                if p.value:
                    self.get_logger().info("GWO Tuning Enabled: Accepting dynamic updates.")
                else:
                    self.get_logger().info("GWO Tuning Disabled: Parameters locked.")
                continue
            
            if p.name.startswith('mpc.weights.'):
                if not enable_tuning:
                    self.get_logger().warn(f"Rejecting update for {p.name}: Tuning disabled.")
                    return SetParametersResult(successful=False, reason="Tuning disabled (enable_gwo_tuning=False)")
                
                key = p.name.replace('mpc.weights.', '')
                if key in current_weights:
                    current_weights[key] = p.value
                    weights_changed = True

        # 4. Apply updates if any
        if weights_changed:
            self.controller.update_weights(
                current_weights['lat_error'],
                current_weights['heading_error'],
                current_weights['velocity_error'],
                current_weights['steer_rate'],
                current_weights['lat_jerk']
            )
            self.get_logger().info(f"Updated MPC Weights via Service: {current_weights}")

        return SetParametersResult(successful=True)

    def odometry_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Convert orientation to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        v = msg.twist.twist.linear.x
        self.current_state = np.array([x, y, theta, v])

    def reference_traj_callback(self, msg):
        # Convert Trajectory message to numpy array for internal calculations
        # [x, y, theta, v, kappa]
        traj_pts = []
        for p in msg.points:
            # Get yaw from quaternion
            q = p.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # kappa = tan(delta) / L
            kappa = math.tan(p.front_wheel_angle_rad) / L
            
            traj_pts.append([
                p.pose.position.x,
                p.pose.position.y,
                yaw,
                p.longitudinal_velocity_mps,
                kappa
            ])
        self.ref_traj = np.array(traj_pts)



    def get_closest_point(self, x, y):
        # Optimized: use squared distance for searching to avoid sqrt CPU cost
        dists_sq = (self.ref_traj[:,0] - x)**2 + (self.ref_traj[:,1] - y)**2
        min_idx = np.argmin(dists_sq)
        return min_idx, self.ref_traj[min_idx]

    def normalize_angle(self, angle):
        # Safer version: prevents infinite loops if angle is Inf/NaN
        if not np.isfinite(angle):
            return 0.0
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def calculate_errors(self, vehicle_state, ref_pt):
        x, y, theta, v = vehicle_state
        xr, yr, thetar, vr, kr = ref_pt
        
        # Lateral Error
        dx = x - xr
        dy = y - yr
        e_y = -dx * math.sin(thetar) + dy * math.cos(thetar)
        
        # Heading Error
        e_theta = self.normalize_angle(theta - thetar)
        
        # Velocity Error
        e_v = v - vr
        
        return np.array([e_y, e_theta, e_v])

    def timer_callback(self):
        # 0. Check if ready
        if self.ref_traj is None:
            if self.time_idx_debug % DEBUG_LOG_INTERVAL == 0:
                self.get_logger().info("Waiting for reference trajectory...")
            self.time_idx_debug += 1
            return
            
        if self.current_state is None:
            if self.time_idx_debug % DEBUG_LOG_INTERVAL == 0:
                self.get_logger().info("Waiting for odometry...")
            self.time_idx_debug += 1
            return
            
        # Robustness check: Ensure current_state is finite to prevent loops/crashes
        if not np.all(np.isfinite(self.current_state)):
            if self.time_idx_debug % DEBUG_LOG_INTERVAL == 0:
                self.get_logger().error("Non-finite values detected in current state!")
            return

        # 1. Get Current State
        x, y, theta, v = self.current_state
        
        # 2. Find Reference
        ref_idx, ref_pt = self.get_closest_point(x, y)
        xr, yr, thetar, vr, kr = ref_pt
        
        # 3. Calculate Errors
        error_state = self.calculate_errors(self.current_state, ref_pt)
        
        # 4. MPC Control (Coupled)
        delta_cmd, a_cmd, pred_errors = self.controller.solve(error_state, vr, kr)
        
        # Clip command to physical limits
        delta_cmd = np.clip(delta_cmd, -MAX_STEER, MAX_STEER)
        a_cmd = np.clip(a_cmd, -MAX_ACCEL, MAX_ACCEL)
        
        # 5. Publish Control command
        # Passing target speed (vr) along with acceleration
        self.publish_control(delta_cmd, a_cmd, vr)
        
        # 6. Visualization
        now = self.get_clock().now().to_msg()
        self.publish_path(now)
        self.publish_predicted_path(now, pred_errors, ref_pt)
        self.publish_debug(now, xr, yr)
        
        # 7. Publish Error Status for Tuning
        error_msg = Float64MultiArray()
        # [e_y, e_theta, e_v, steer_rate]
        # We need steer_rate... let's calculate it roughly or pass it?
        # Actually GWO just needs errors. Let's send e_y, e_theta, e_v.
        # Steer rate could be useful for smooth tuning.
        # But let's stick to state errors first as per CalculateErrors output.
        error_msg.data = [error_state[0], error_state[1], error_state[2]]
        self.error_pub.publish(error_msg)

        if self.time_idx_debug % (DEBUG_LOG_INTERVAL * 20) == 0:
            # Log active weights every 2 sec (at 10Hz)
            try:
                # Assuming controller instance has Q matrix to peek at
                lat_weight = self.controller.Q[0,0]
                heading_weight = self.controller.Q[1,1]
                vel_weight = self.controller.Q[2,2]
                
                steer_rate_weight = '?'
                if hasattr(self.controller, 'w_steer_rate'):
                    steer_rate_weight = f"{self.controller.w_steer_rate:.2f}"
                    
                self.get_logger().info(
                    f"[MPC Weights] Lat: {lat_weight:.2f}, Heading: {heading_weight:.2f}, "
                    f"Vel: {vel_weight:.2f}, Rate: {steer_rate_weight}"
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to read weights: {e}")

        # Debug Log
        if self.time_idx_debug % DEBUG_LOG_INTERVAL == 0:
            self.get_logger().info(f"e_y: {error_state[0]:.3f}, e_theta: {error_state[1]:.3f}, e_v: {error_state[2]:.3f}, steer: {delta_cmd:.3f}, accel: {a_cmd:.3f}")
        self.time_idx_debug += 1

    time_idx_debug = 0
    wheel_angle = 0.0
    # WHEEL_RADIUS loaded globally

    def publish_wheel_tfs(self, timestamp, steering_angle):
        tfs = self.vehicle.get_wheel_tfs(timestamp, steering_angle)
        for t in tfs:
            self.tf_broadcaster.sendTransform(t)

    def quaternion_from_euler(self, roll, pitch, yaw):
        # Standard ZYX convention
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]
        
    def quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return [x, y, z, w]


    def publish_control(self, steer, accel, speed):
        msg = AckermannControlCommand()
        msg.stamp = self.get_clock().now().to_msg()
        
        # Lateral
        msg.lateral.stamp = msg.stamp
        msg.lateral.steering_tire_angle = float(steer)
        
        # Longitudinal
        msg.longitudinal.stamp = msg.stamp
        msg.longitudinal.speed = float(speed)
        msg.longitudinal.acceleration = float(accel)
        
        self.control_pub.publish(msg)


    def publish_path(self, timestamp):
        # We don't really need to publish the reference path here anymore 
        # since path_publisher is doing it. 
        # But we can still do it if we want to show what the MPC is currently "seeing".
        if self.ref_traj is None: return
        
        path_msg = Path()
        path_msg.header.stamp = timestamp
        path_msg.header.frame_id = "map"
        
        # Downsample for vis
        for i in range(0, len(self.ref_traj), 10):
            pt = self.ref_traj[i]
            p = Odometry().pose.pose
            p.position.x = pt[0]
            p.position.y = pt[1]
            path_msg.poses.append(self.create_pose_stamped(p.position, "map", timestamp))
            
        self.path_pub.publish(path_msg)

    def publish_predicted_path(self, timestamp, pred_errors, ref_pt):
        # pred_errors: [e_y, e_theta, e_v] * (N+1) flattened
        # reshape pred_errors to (N+1, 3)
        nx = 3
        num_points = len(pred_errors) // nx
        errors = pred_errors.reshape((num_points, nx))
        
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = timestamp
        
        for i in range(num_points):
            e_y = errors[i, 0]
            
            xr, yr, thetar, vr, kr = ref_pt # kr is curvature
            s = i * vr * DT_PRED
            
            if abs(kr) > 1e-3:
                # Circular motion for reference
                R = 1.0 / kr
                d_theta = s / R
                cx = xr - R * math.sin(thetar)
                cy = yr + R * math.cos(thetar)
                xr_new = cx + R * math.sin(thetar + d_theta)
                yr_new = cy - R * math.cos(thetar + d_theta)
                thetar_new = thetar + d_theta
            else:
                # Straight line
                xr_new = xr + s * math.cos(thetar)
                yr_new = yr + s * math.sin(thetar)
                thetar_new = thetar
                
            gx = xr_new - e_y * math.sin(thetar_new)
            gy = yr_new + e_y * math.cos(thetar_new)
            
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = gx
            pose.pose.position.y = gy
            pose.pose.position.z = 0.0
            
            path_msg.poses.append(pose)
            
        self.predicted_path_pub.publish(path_msg)


    def publish_debug(self, timestamp, xr, yr):
        # Publish red sphere at reference point
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = timestamp
        marker.ns = "debug_target"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = xr
        marker.pose.position.y = yr
        self.debug_pub.publish(marker)

    def create_pose_stamped(self, position, frame_id, timestamp):
        from geometry_msgs.msg import PoseStamped
        p = PoseStamped()
        p.header.frame_id = frame_id
        p.header.stamp = timestamp
        p.pose.position = position
        p.pose.orientation.w = 1.0
        return p

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = MPCControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
