#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
import numpy as np
import yaml
import os
import sys
import copy
import time
from ament_index_python.packages import get_package_share_directory

# GWO Algorithm Implementation
class GWO:
    def __init__(self, dim, search_agents_no, max_iter, lb, ub):
        self.dim = dim
        self.search_agents_no = search_agents_no
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        # Initialize Alpha, Beta, Delta
        self.Alpha_pos = np.zeros(dim)
        self.Alpha_score = float("inf")
        
        self.Beta_pos = np.zeros(dim)
        self.Beta_score = float("inf")
        
        self.Delta_pos = np.zeros(dim)
        self.Delta_score = float("inf")
        
        # Initialize the positions of search agents
        self.Positions = np.random.uniform(0, 1, (search_agents_no, dim)) * (ub - lb) + lb
        
        # Convergence curve
        self.convergence_curve = np.zeros(max_iter)
        self.iter = 0

    def update(self, current_fitness_list):
        # Update Alpha, Beta, Delta
        for i in range(self.search_agents_no):
            fitness = current_fitness_list[i]
            
            # Update Alpha
            if fitness < self.Alpha_score:
                self.Alpha_score = fitness
                self.Alpha_pos = self.Positions[i, :].copy()
            
            # Update Beta
            if fitness > self.Alpha_score and fitness < self.Beta_score:
                self.Beta_score = fitness
                self.Beta_pos = self.Positions[i, :].copy()
                
            # Update Delta
            if fitness > self.Alpha_score and fitness > self.Beta_score and fitness < self.Delta_score:
                self.Delta_score = fitness
                self.Delta_pos = self.Positions[i, :].copy()
        
        # Main Loop logic for one iteration step
        a = 2 - self.iter * ((2) / self.max_iter) # a decreases linearly from 2 to 0
        
        for i in range(self.search_agents_no):
            for j in range(self.dim):
                
                # Alpha pack
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.Alpha_pos[j] - self.Positions[i, j])
                X1 = self.Alpha_pos[j] - A1 * D_alpha
                
                # Beta pack
                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.Beta_pos[j] - self.Positions[i, j])
                X2 = self.Beta_pos[j] - A2 * D_beta
                
                # Delta pack
                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.Delta_pos[j] - self.Positions[i, j])
                X3 = self.Delta_pos[j] - A3 * D_delta
                
                self.Positions[i, j] = (X1 + X2 + X3) / 3
                
        # Clip
        self.Positions = np.clip(self.Positions, self.lb, self.ub)
        
        self.iter += 1
        if self.iter >= self.max_iter:
            self.iter = 0 # Reset or loop? For continuous tuning, we might want to reset 'a' or oscillate it.
            # For this simple implementation, let's keep 'a' small or reset it.
            # Resetting 'a' might cause jump. Let's look at standard continuous GWO...
            # For now, let's just reset iter to keep exploring.
            pass

        return self.Alpha_pos, self.Alpha_score


class GWOTunerNode(Node):
    def __init__(self):
        super().__init__('gwo_tuner_node')
        
        # 1. Load Configurations
        self.load_configs()
        
        # 2. Parameters
        self.declare_parameter('calculation_rate_hz', 10.0)
        self.calc_rate = self.get_parameter('calculation_rate_hz').value
        
        self.declare_parameter('update_rate_hz', 0.1)
        self.update_rate = self.get_parameter('update_rate_hz').value
        
        self.declare_parameter('settling_time_s', 2.0)
        self.settling_time_s = self.get_parameter('settling_time_s').value
        
        self.declare_parameter('evaluation_duration_s', 5.0)
        self.evaluation_duration_s = self.get_parameter('evaluation_duration_s').value
        
        # 3. Setup GWO
        self.setup_gwo()
        
        # 4. ROS Interfaces
        self.error_sub = self.create_subscription(Float64MultiArray, '/mpc/error_status', self.error_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Service Client for MPC
        self.param_client = self.create_client(SetParameters, '/mpc_controller_node/set_parameters')
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('MPC parameter service not available, waiting...')

        # 5. State Variables
        self.error_buffer = []
        self.current_speed = 0.0
        self.is_tuning = False
        self.best_score_global = float("inf")
        self.best_pos_global = None
        self.last_update_time = self.get_clock().now()
        
        # Debug Counters
        self.debug_count_odom = 0
        self.debug_count_error = 0
        self.debug_count_calc = 0
        
        # Buffer for fitness evaluation
        self.buffer_size = int(self.calc_rate * self.evaluation_duration_s) # Assume error msg at 10Hz
        # Wait, buffer_size depends on how often error_callback is called. 
        # If MPC pubs at 20Hz, we get 20 pts/s.
        # Let's keep buffer logic based on length or time window in callback?
        # Ideally time window.
        
        # Timers
        # Calculation Timer (Fast): Monitor buffer, Calculate Fitness, Run GWO Logic
        self.calc_timer = self.create_timer(1.0/self.calc_rate, self.calculation_loop)
        
        # Update Timer (Slow): Handles strict update rate throttling if needed
        # Or we just use timestamp check in the main loop.
        # The user requested "update 5 hz". If we update params, we can do it at that rate.
        # But our logic is sequential (Wolf by Wolf).
        
        self.get_logger().info(f"GWO Tuner Started. Calc Rate: {self.calc_rate}Hz, Update Rate: {self.update_rate}Hz")
        
        # Flag to trigger optimization
        self.ready_to_optimize = False

    def load_configs(self):
        # Paths
        package_share_directory = get_package_share_directory('tracter_control')
        mpc_config_path = os.path.join(package_share_directory, 'config/mpc_config.yaml')
        gwo_config_path = os.path.join(package_share_directory, 'config/gwo_config.yaml')
        
        # Load MPC Config
        try:
            with open(mpc_config_path, 'r') as f:
                self.mpc_config = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load MPC config: {e}")
            sys.exit(1)
            
        # Load GWO Config
        try:
            with open(gwo_config_path, 'r') as f:
                self.gwo_config = yaml.safe_load(f)['gwo_node']['ros__parameters']
        except Exception as e:
            self.get_logger().error(f"Failed to load GWO config: {e}")
            sys.exit(1)

    # ... (setup_gwo, odom_callback, error_callback, calculate_fitness same)

    def calculation_loop(self):
        # FSM for Tuning
        if not hasattr(self, 'state'):
            self.state = 'SET_WEIGHTS'
            self.wolf_idx = 0
            self.gen = 0
            self.fitness_results = np.zeros(self.pop_size)

        if self.state == 'SET_WEIGHTS':
            # Throttle Update
            now = self.get_clock().now()
            dt = (now - self.last_update_time).nanoseconds / 1e9
            if dt < (1.0 / self.update_rate):
                return # Wait for next cycle (Throttled)

            # Set weights for current wolf
            wolf_pos = self.gwo.Positions[self.wolf_idx]
            self.get_logger().info(f"--- Gen {self.gen} Wolf {self.wolf_idx + 1}/{self.pop_size} ---")
            self.get_logger().info(f"Trying Weights: {wolf_pos}") # DEBUG
            
            self.update_mpc_weights(wolf_pos)
            self.last_update_time = self.get_clock().now()
            
            self.state = 'WAIT_UPDATE_CONFIRM'
            self.update_start_time = now

        elif self.state == 'WAIT_UPDATE_CONFIRM':
            # Check if update finished (we can use a flag set by callback)
            if self.update_finished:
                self.error_buffer = []  # clear before settling
                self.state = 'SETTLING'
                self.settling_start_time = self.get_clock().now()
                self.get_logger().info(f"Weights Applied. Settling for {self.settling_time_s}s...")
            else:
                # Timeout safety
                if (self.get_clock().now() - self.update_start_time).nanoseconds > 2e9: # 2s timeout
                     self.get_logger().warn("Parameter update timed out! Skipping wolf.")
                     self.wolf_idx += 1 
                     if self.wolf_idx >= self.pop_size:
                        self.state = 'UPDATE_GWO'
                     else:
                        self.state = 'SET_WEIGHTS'

        elif self.state == 'SETTLING':
            # Ignore errors here, just wait
            self.error_buffer = []
            now = self.get_clock().now()
            if (now - self.settling_start_time).nanoseconds / 1e9 >= self.settling_time_s:
                if self.current_speed > 0.1:
                    self.state = 'START_EVALUATION'
                    self.get_logger().info(f"Settled! Starting evaluation for {self.evaluation_duration_s}s...")
                else:
                    if self.debug_count_calc % 50 == 0:
                        self.get_logger().info("Settled, but waiting for vehicle to move...")
                    self.debug_count_calc += 1
            
        elif self.state == 'START_EVALUATION':
            self.error_buffer = [] # Clear buffer once more before start
            self.state = 'WAIT_BUFFER'

        elif self.state == 'WAIT_BUFFER':
            if len(self.error_buffer) >= self.buffer_size:
                self.state = 'EVALUATE'
                
        elif self.state == 'EVALUATE':
            fitness = self.calculate_fitness(self.error_buffer)
            self.fitness_results[self.wolf_idx] = fitness
            
            self.get_logger().info(f"Fitness Result: {fitness:.4f} (Best so far: {self.best_score_global:.4f})")
            
            self.wolf_idx += 1
            if self.wolf_idx >= self.pop_size:
                self.state = 'UPDATE_GWO'
            else:
                self.state = 'SET_WEIGHTS'
                
        elif self.state == 'UPDATE_GWO':
            alpha, score = self.gwo.update(self.fitness_results)
            self.get_logger().info(f"=== Generation {self.gen} Complete ===")
            self.get_logger().info(f"Best Fitness: {score:.4f}")
            self.get_logger().info(f"Best Weights: {alpha}")
            
            # Save if better
            if score < self.best_score_global:
                self.best_score_global = score
                self.best_pos_global = alpha
                self.save_best_params(alpha)
                self.get_logger().info("!!! NEW BEST FOUND AND SAVED !!!")
            
            # Ensure best weights are active on MPC
            self.update_mpc_weights(alpha)
            
            self.gen += 1
            self.wolf_idx = 0
            self.state = 'SET_WEIGHTS' # Start next gen

    def load_configs(self):
        # Paths
        package_share_directory = get_package_share_directory('tracter_control')
        mpc_config_path = os.path.join(package_share_directory, 'config/mpc_config.yaml')
        gwo_config_path = os.path.join(package_share_directory, 'config/gwo_config.yaml')
        
        # Load MPC Config
        try:
            with open(mpc_config_path, 'r') as f:
                self.mpc_config = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load MPC config: {e}")
            sys.exit(1)
            
        # Load GWO Config
        try:
            with open(gwo_config_path, 'r') as f:
                self.gwo_config = yaml.safe_load(f)['gwo_node']['ros__parameters']
        except Exception as e:
            self.get_logger().error(f"Failed to load GWO config: {e}")
            sys.exit(1)

    def setup_gwo(self):
        self.targets = self.gwo_config['targets'] # ['lat_error', 'heading_error', ...]
        self.pop_size = self.gwo_config['population_size']
        self.max_iter = self.gwo_config.get('max_iter', 100)
        range_pct = self.gwo_config['search_range_pct']
        
        # Initial values from MPC config
        initial_values = []
        for t in self.targets:
            # Map target name to config key structure
            # Assumes targets are keys in mpc.weights
            val = self.mpc_config['mpc']['weights'][t]
            initial_values.append(val)
        
        initial_values = np.array(initial_values)
        self.dim = len(initial_values)
        
        # Bounds
        lb = initial_values * (1.0 - range_pct)
        ub = initial_values * (1.0 + range_pct)
        
        self.gwo = GWO(self.dim, self.pop_size, self.max_iter, lb, ub)
        # Seed GWO with initial guess if supported, or rely on random around it
        # Custom init: First wolf is the initial guess
        self.gwo.Positions[0, :] = initial_values

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x
        if self.debug_count_odom % 50 == 0: # Log every 50 odom
            self.get_logger().info(f"[Odom] Speed: {self.current_speed:.2f}")
        self.debug_count_odom += 1

    def error_callback(self, msg):
        # msg.data = [e_y, e_theta, e_v]
        if self.current_speed < 0.1:
            # Don't collect errors if standing still (accumulates integral windup or invalid fitness)
            if self.debug_count_error % 50 == 0:
                self.get_logger().info(f"[Error] Skipping: Speed too low ({self.current_speed:.2f})")
            self.debug_count_error += 1
            return

        if getattr(self, 'state', None) == 'WAIT_BUFFER':
            self.error_buffer.append(msg.data)
            if self.debug_count_error % 50 == 0:
                self.get_logger().info(f"[Error] New Data. Buffer Size: {len(self.error_buffer)}/{self.buffer_size}")
        self.debug_count_error += 1

    def calculate_fitness(self, buffer):
        # Fitness = Weighted RMSE
        if not buffer: return float("inf")
        
        data = np.array(buffer)
        # data shape: (N, 3) -> lat, heading, vel
        
        rmse = np.sqrt(np.mean(data**2, axis=0))
        # Weights for fitness components (can be tuned or hardcoded)
        # Let's say we value lat_error and heading_error most
        w_fit = [1.0, 1.0, 0.5] 
        
        score = np.dot(rmse, w_fit)
        return score

    # control_loop replaced by calculation_loop defined above
    # Removing old control_loop to avoid duplication/confusion
    pass

    def update_mpc_weights(self, weights):
        self.update_finished = False
        req = SetParameters.Request()
        
        for i, name in enumerate(self.targets):
            param_name = f"mpc.weights.{name}"
            val = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(weights[i]))
            req.parameters.append(Parameter(name=param_name, value=val))
            
        self.get_logger().info(f"Sending Service Request for {len(weights)} params...") # DEBUG
        future = self.param_client.call_async(req)
        future.add_done_callback(self.update_done_callback)
        
    def update_done_callback(self, future):
        try:
            result = future.result()
            # result.results is list of SetParametersResult
            # check if all successful
            all_success = True
            for i, r in enumerate(result.results):
                if not r.successful:
                    all_success = False
                    self.get_logger().error(f"Param {i} Update Failed: {r.reason}")
            
            if all_success:
                self.update_finished = True
                self.get_logger().info("Service Call Successful: All params updated.")
            else:
                self.get_logger().warn("Some parameters failed to update.")
                self.update_finished = True # Proceed anyway to avoid stuck
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        
    def save_best_params(self, weights):
        package_share_directory = get_package_share_directory('tracter_control')
        save_path = os.path.join(package_share_directory, 'config/mpc_best_params.yaml')
        
        best_dict = {}
        for i, name in enumerate(self.targets):
            best_dict[name] = float(weights[i])
            
        try:
            with open(save_path, 'w') as f:
                yaml.dump({'best_mpc_weights': best_dict}, f)
            self.get_logger().info(f"Saved best params to {save_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save params: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GWOTunerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
