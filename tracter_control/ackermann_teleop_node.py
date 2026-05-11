#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from autoware_auto_control_msgs.msg import AckermannControlCommand
import sys
import select
import termios
import tty

# Autoware Messages
try:
    from tier4_external_api_msgs.msg import Heartbeat, GearShiftStamped, GearShift, ControlCommandStamped
    from tier4_external_api_msgs.srv import Engage
    HAS_TIER4 = True
except ImportError:
    HAS_TIER4 = False

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

settings = termios.tcgetattr(sys.stdin)

class AckermannTeleop(Node):
    def __init__(self):
        super().__init__('ackermann_teleop')
        
        # Make space for the 16-line dashboard without clearing terminal history
        sys.stdout.write('\n' * 16)
        sys.stdout.flush()
        
        # Use Reliable + Transient Local QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Use ControlCommandStamped for Autoware Remote Mode
        if HAS_TIER4:
            self.control_pub = self.create_publisher(ControlCommandStamped, '/api/external/set/command/remote/control', qos_profile)
            self.heartbeat_pub = self.create_publisher(Heartbeat, '/api/external/set/command/remote/heartbeat', 10)
            self.gear_pub = self.create_publisher(GearShiftStamped, '/api/external/set/command/remote/shift', 10)
            self.engage_client = self.create_client(Engage, '/api/autoware/set/engage')
            try:
                from autoware_adapi_v1_msgs.srv import ChangeOperationMode
                self.remote_client = self.create_client(ChangeOperationMode, '/api/operation_mode/change_to_remote')
                self.has_adapi = True
            except ImportError:
                self.has_adapi = False
        else:
            self.control_pub = self.create_publisher(AckermannControlCommand, '/control/command/control_cmd', qos_profile)
        
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.step = 0.1
        self.steering_step = 0.05
        self.max_steering = 0.6
        self.enabled = False  # Start disabled for safety, require pressing Space to start
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.draw_dashboard()

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = None
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, settings)
        return key

    def call_engage(self):
        if not HAS_TIER4: return
        # Move below dashboard for logs so it doesn't break UI
        self.get_logger().info("Changing Operation Mode to REMOTE...")
        if hasattr(self, 'has_adapi') and self.has_adapi:
            from autoware_adapi_v1_msgs.srv import ChangeOperationMode
            req_remote = ChangeOperationMode.Request()
            self.remote_client.call_async(req_remote)
            
        self.get_logger().info("Calling Engage Service...")
        req = Engage.Request()
        req.engage = True
        self.engage_client.call_async(req)

    def timer_callback(self):
        key = self.get_key()
        if key == ' ':
            self.enabled = not self.enabled
            if self.enabled:
                self.call_engage()
        
        if key == 'w':
            self.throttle = min(1.0, self.throttle + self.step)
            self.brake = 0.0
        elif key == 'x':
            self.throttle = 0.0
            self.brake = min(1.0, self.brake + self.step)
        elif key == 'a':
            self.steering += self.steering_step
        elif key == 'd':
            self.steering -= self.steering_step
        elif key == 's':
            self.throttle = 0.0
            self.brake = 1.0
        elif key == 'z':
            self.steering = 0.0
        elif key == '\x03':  # Ctrl-C
            rclpy.shutdown()
            sys.exit()

        # Clamp steering
        self.steering = max(-self.max_steering, min(self.max_steering, self.steering))

        now = self.get_clock().now().to_msg()
        
        if not self.enabled:
            self.throttle = 0.0
            self.brake = 1.0

        if HAS_TIER4:
            # Send Heartbeat
            hb = Heartbeat()
            hb.stamp = now
            self.heartbeat_pub.publish(hb)
            
            # Send Gear DRIVE
            gs = GearShiftStamped()
            gs.stamp = now
            gs.gear_shift.data = GearShift.DRIVE
            self.gear_pub.publish(gs)

            # Send Control
            aw_cmd = ControlCommandStamped()
            aw_cmd.stamp = now
            aw_cmd.control.steering_angle = float(self.steering)
            aw_cmd.control.throttle = float(self.throttle)
            aw_cmd.control.brake = float(self.brake)
            self.control_pub.publish(aw_cmd)
        else:
            # Fallback
            cmd = AckermannControlCommand()
            cmd.stamp = now
            cmd.longitudinal.speed = float(self.throttle * 5.0)
            cmd.longitudinal.acceleration = float(self.throttle * 2.0)
            cmd.lateral.steering_tire_angle = float(self.steering)
            self.control_pub.publish(cmd)
            
        self.draw_dashboard()

    def draw_dashboard(self):
        # ANSI Colors
        GREEN = '\033[92m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        status_color = GREEN if self.enabled else RED
        status_text = " ENABLED " if self.enabled else "DISABLED!"
        
        # Progress bars (10 blocks)
        t_bars = int(self.throttle * 10)
        t_str = '█' * t_bars + '░' * (10 - t_bars)
        
        b_bars = int(self.brake * 10)
        b_str = '█' * b_bars + '░' * (10 - b_bars)
        
        # Steering bar (-0.6 to 0.6 mapped to 11 chars)
        s_norm = (self.steering + self.max_steering) / (2 * self.max_steering)
        s_idx = max(0, min(10, int(s_norm * 10)))
        s_arr = ['─'] * 11
        s_arr[5] = '┼'
        s_arr[s_idx] = '█'
        s_str = "".join(s_arr)

        dashboard = (
            f"\033[16A\033[J" # Move up 16 lines and clear downward
            f"{CYAN}{BOLD}================================================={RESET}\n"
            f"{BOLD}      🚗 Autoware Ackermann Teleop Control       {RESET}\n"
            f"{CYAN}{BOLD}================================================={RESET}\n"
            f"  STATUS: [{status_color}{BOLD}{status_text}{RESET}]   GEAR: [{GREEN} DRIVE {RESET}]\n"
            f"{CYAN}-------------------------------------------------{RESET}\n"
            f"  {YELLOW}THROTTLE{RESET} : [{GREEN}{t_str}{RESET}] {self.throttle*100:3.0f}%\n"
            f"  {YELLOW}BRAKE   {RESET} : [{RED}{b_str}{RESET}] {self.brake*100:3.0f}%\n"
            f"  {YELLOW}STEERING{RESET} : [{CYAN}{s_str}{RESET}] {self.steering:5.2f} rad\n"
            f"{CYAN}-------------------------------------------------{RESET}\n"
            f"  [W] Throttle Up     [X] Brake\n"
            f"  [A] Steer Left      [D] Steer Right\n"
            f"  [S] Emergency Stop  [Z] Center Steering\n"
            f"  [Space] Toggle Enable / Auto-Engage\n"
            f"{CYAN}-------------------------------------------------{RESET}\n"
            f"  Press [Ctrl+C] to quit\n"
            f"{CYAN}{BOLD}================================================={RESET}\n"
        )
        sys.stdout.write(dashboard)
        sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = AckermannTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
