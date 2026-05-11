#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
import csv
import os
import yaml
import sys
import numpy as np

from ament_index_python.packages import get_package_share_directory

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        
        # Load Config
        package_share_directory = get_package_share_directory('tracter_control')
        config_path = os.path.join(package_share_directory, 'config/mpc_config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {e}")
            sys.exit(1)
            
        path_file = config['path_smoothing']['reference_path_file']
        self.topic = config['path_smoothing']['ref_traj_topic']
        
        # Resolve relative path
        if not os.path.isabs(path_file):
            path_file = os.path.join(package_share_directory, path_file)
            
        self.wheelbase = config['vehicle']['wheelbase']
        self.get_logger().info(f"Loading path from {path_file}")
        
        # Load data
        traj_data = []
        try:
            with open(path_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader) # Skip header
                for row in reader:
                    traj_data.append([float(x) for x in row])
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")
            sys.exit(1)
            
        self.traj_np = np.array(traj_data)
        
        # Create Messages
        self.path_msg = self.create_path_msg(self.traj_np)
        
        # Publishers
        self.viz_pub = self.create_publisher(Trajectory, self.topic, 10)
        self.data_pub = self.create_publisher(Trajectory, self.topic + "_data", 10)
        
        # Timer
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f"Publishing path to {self.topic} and data to {self.topic}_data")
        
    def create_path_msg(self, traj):
        msg = Trajectory()
        msg.header.frame_id = "map"
        for row in traj:
            p = TrajectoryPoint()
            p.pose.position.x = row[0]
            p.pose.position.y = row[1]
            yaw = row[2]
            p.pose.orientation.z = np.sin(yaw/2.0)
            p.pose.orientation.w = np.cos(yaw/2.0)
            p.longitudinal_velocity_mps = row[3]
            # Convert kappa to steering angle: delta = arctan(L * kappa)
            kappa = row[4]
            p.front_wheel_angle_rad = np.arctan(self.wheelbase * kappa)
            msg.points.append(p)
        return msg

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        self.viz_pub.publish(self.path_msg)
        self.data_pub.publish(self.path_msg) # Now both use the same Trajectory type

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
