from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('tracter_control')
    
    return LaunchDescription([
        # Path Publisher Node
        Node(
            package='tracter_control',
            executable='path_publisher_node',
            name='path_publisher',
            output='screen'
        ),
        
        # Vehicle Node (Simulation)
        Node(
            package='tracter_control',
            executable='vehicle_node',
            name='vehicle_plant',
            output='screen'
        ),
        
        # MPC Controller Node
        Node(
            package='tracter_control',
            executable='mpc_node',
            name='mpc_controller',
            output='screen',
            parameters=[os.path.join(pkg_share, 'config/mpc_config.yaml')]
        ),
        
        # GWO Tuner Node (Optional, kept commented out or enabled by arg)
        # Node(
        #     package='tracter_control',
        #     executable='gwo_tuner_node',
        #     name='gwo_tuner',
        #     output='screen'
        # ),
    ])
