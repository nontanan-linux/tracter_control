# Tracter Control Package

The `tracter_control` package is a ROS 2-based autonomous control system designed for tractor-trailer systems and standard vehicles. It features a Nonlinear Model Predictive Control (NMPC) module, a dynamic Grey Wolf Optimizer (GWO) for real-time parameter tuning, and necessary bridge nodes for interfacing with the Autoware autonomous driving stack.

## 🎯 Key Features

* **MPC Controller (`mpc_node`)**: A Model Predictive Control node that handles both lateral and longitudinal vehicle dynamics. It calculates the optimal steering and acceleration by minimizing state errors against a reference trajectory.
* **Tracter Control Node (`tracter_control.py`)**: The primary Autoware integration node. It listens to Autoware's scenario planner (`/planning/scenario_planning/trajectory`) and localization states, performs NMPC calculations, and acts as an external remote controller by publishing directly to Autoware's `external_cmd_selector`.
* **GWO Tuner (`gwo_tuner_node`)**: Uses the Grey Wolf Optimizer algorithm to dynamically adjust MPC penalty weights (Q and R matrices) in real-time based on tracking performance, ensuring a smooth and accurate ride.
* **Path Publisher (`path_publisher_node`)**: Loads reference path coordinates from `.csv` files and publishes them as a `nav_msgs/Path` along with detailed trajectory states (speed, curvature) for the basic MPC to follow.
* **Vehicle Simulation (`vehicle_node`)**: Provides a software-in-the-loop kinematics simulator for testing without real hardware. It calculates and publishes the vehicle's simulated odometry based on control commands.
* **Planning Simulator**: A comprehensive simulation setup that combines path generation, NMPC control, and vehicle dynamics to validate planning algorithms for tractor-trailer systems.
* **Control Selector (`control_selector_node`)**: An older bridging node previously used to intercept and replace the autonomous control stream directly before it reaches the vehicle interface.
* **Manual Control (`ackermann_teleop_node.py`)**: A keyboard-based teleoperation node designed to seamlessly interface with Autoware's External API to allow manual overriding and vehicle driving.

## 📂 Package Structure

```text
tracter_control/
├── config/              # Configuration files (e.g., mpc_config.yaml)
├── launch/              # ROS 2 Launch files
├── paths/               # Reference trajectories (.csv files)
├── urdf/                # Unified Robot Description Format files
├── simulation/          # Simulation & testing nodes (Standalone NMPC)
│   ├── __init__.py
│   ├── mpc_node.py
│   ├── gwo_tuner_node.py
│   ├── path_publisher.py
│   └── generate_path.py
└── tracter_control/     # Main control nodes
    ├── __init__.py
    ├── ackermann_teleop_node.py # Keyboard manual control
    ├── nmpc_control_node.py # NMPC Node (Tractor-Trailer)
    ├── tracter_control.py
    ├── vehicle_node.py
    ├── vehicle_chassis.py
    └── control_selector_node.py
```

## 🚀 Installation & Usage

1. **Build the Package**:
   Use the clean build script located in your workspace root.

   ```bash
   cd ~/AutoTract
   ./clean_build.sh
   source install/setup.bash
   ```

2. **Bringing up the System via Launch File** *(Example)*:

   ```bash
   ros2 launch tracter_control tracter_control.launch.py
   ```

3. **Running Standalone Nodes**:
   * **Start the Nonlinear MPC (Tractor-Trailer)**:
     `ros2 run tracter_control nmpc_control_node`
   * **Start the standalone MPC**:
     `ros2 run tracter_control mpc_node`
   * **Start the Autoware Integration Node (Tracter Control)**:
     `ros2 run tracter_control tracter_control_node`
   * **Start the GWO Tuner**:
     `ros2 run tracter_control gwo_tuner_node`

## ⚙️ MPC Parameters

All core parameters are located in `config/mpc_config.yaml`:
* `dt`: Control time step
* `prediction_horizon`: The number of steps the MPC looks ahead into the future
* `weights`: Default penalty weights (Lat/Heading Error, Steer Rate, etc.) used if GWO is disabled
* `vehicle`: Dimensions and physical limits constraints (wheelbase, max steer angle, etc.)

## 🎮 Planning Simulator
The planning simulator allows you to test the NMPC controller within the full Autoware environment.

1. **Launch Autoware Planning Simulator**:
   ```bash
   cd ~/autoware
   # Note: map_path must be an absolute path
   ros2 launch autoware_launch planning_simulator.launch.xml map_path:=/home/nontanan/autoware_data/CAV_GT
   ```
2. **Initialize and Set Goal**:
   In RViz, use the **2D Pose Estimate** tool to set the vehicle's starting position and the **2D Nav Goal** tool to set the destination.
3. **Run NMPC Control**:
   ```bash
   ros2 run tracter_control nmpc_control_node
   ```
4. **Switch Autoware to Remote Mode**:
   In a new terminal, call the service to activate external control:
   ```bash
   ros2 service call /control/external_cmd_selector/select_external_command tier4_control_msgs/srv/ExternalCommandSelect "{mode: {data: 1}}"
   ```

## 🕹️ Manual Control (Teleop)
You can manually control the vehicle using the keyboard. By default, this node integrates with Autoware's External API, publishing `ControlCommandStamped` (Throttle/Brake/Steer) and automatically handling Heartbeat and Gear signals.

**Run Teleop Node**:
```bash
ros2 run tracter_control ackermann_teleop_node
```

### Key Mapping:
| Key | Action |
| --- | --- |
| **W / X** | Increase / Decrease Throttle and Brake |
| **A / D** | Increase / Decrease Steering Angle |
| **S** | Stop (Apply Full Brake) |
| **Z** | Reset Steering to 0 |
| **Space** | **Toggle ENABLE / DISABLE** (Auto-engages and switches to Remote Mode) |
| **Ctrl+C** | Quit |

> [!TIP]
> **To start driving:** Run the node and press **Spacebar** once. This will enable the controller, switch Autoware to Remote Mode, and send the Engage command automatically. Then press **`w`** to accelerate. Press **Spacebar** again to disable and force a safe stop.

## 🔌 Autoware Integration

This package outputs commands using the standard `autoware_auto_control_msgs/msg/AckermannControlCommand`. You can inject these commands into the Autoware stack specifically using the **External/Remote Mode**:

Through `tracter_control_node`, the custom NMPC will publish its control commands directly to `~/input/remote/control_cmd` of Autoware's `external_cmd_selector`. To activate the custom MPC during runtime, simply use Autoware's API or RViz interface to change the active driving mode from **Auto** to **Remote**.
