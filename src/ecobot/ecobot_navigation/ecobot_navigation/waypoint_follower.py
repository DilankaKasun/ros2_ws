import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import math
import csv


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        self.declare_parameter('waypoint_file', '')
        self.declare_parameter('linear_gain', 0.5)
        self.declare_parameter('angular_gain', 1.5)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('max_linear_speed', 0.4)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('accel_limit', 0.4)
        self.declare_parameter('decel_limit', 0.5)
        self.declare_parameter('omega_accel', 1.5)
        self.declare_parameter('min_creep', 0.02)

        self.linear_gain = self.get_parameter('linear_gain').value
        self.angular_gain = self.get_parameter('angular_gain').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.max_linear = self.get_parameter('max_linear_speed').value
        self.max_angular = self.get_parameter('max_angular_speed').value
        self.accel_limit = self.get_parameter('accel_limit').value
        self.decel_limit = self.get_parameter('decel_limit').value
        self.omega_accel = self.get_parameter('omega_accel').value
        self.min_creep = self.get_parameter('min_creep').value

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.waypoints = []
        waypoint_file = self.get_parameter('waypoint_file').value
        if waypoint_file:
            self.load_waypoints(waypoint_file)

        self.waypoint_index = 0
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.last_ctrl_time = None

        self.cmd_pub = self.create_publisher(Twist, '/nav_cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.control_timer = self.create_timer(0.1, self.follow_loop)

    def load_waypoints(self, filename):
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                self.waypoints = [(float(r[0]), float(r[1])) for r in reader]
            self.get_logger().info(f'loaded {len(self.waypoints)} waypoints')
        except Exception as e:
            self.get_logger().error(f'failed to load waypoints: {e}')

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.robot_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))

    def _rate_limit(self, prev, target, dt, accel, decel=None):
        decel = accel if decel is None else decel
        limit = (accel if target >= prev else decel) * dt
        if abs(target - prev) <= limit:
            return target
        return prev + math.copysign(limit, target - prev)

    def _ctrl_dt(self):
        now = self.get_clock().now()
        if self.last_ctrl_time is None:
            self.last_ctrl_time = now
            return 0.1
        dt = (now - self.last_ctrl_time).nanoseconds * 1e-9
        self.last_ctrl_time = now
        return min(0.5, max(0.02, dt))

    def follow_loop(self):
        if not self.waypoints or self.waypoint_index >= len(self.waypoints):
            twist = Twist()
            self.cmd_pub.publish(twist)
            return

        gx, gy = self.waypoints[self.waypoint_index]
        dx = gx - self.robot_x
        dy = gy - self.robot_y
        dist = math.hypot(dx, dy)

        if dist < self.goal_tolerance:
            self.get_logger().info(f'reached waypoint {self.waypoint_index}')
            self.waypoint_index += 1
            self.prev_linear = 0.0
            self.prev_angular = 0.0
            return

        angle_to_goal = math.atan2(dy, dx)
        angle_error = angle_to_goal - self.robot_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        dt = self._ctrl_dt()

        lin_target = min(self.linear_gain * dist, self.max_linear)
        remaining = dist - self.goal_tolerance
        if remaining > 0.0:
            lin_target = min(
                lin_target,
                math.sqrt(2.0 * self.decel_limit * remaining))
        lin_target = max(self.min_creep, lin_target)
        linear = self._rate_limit(self.prev_linear, lin_target, dt,
                                  self.accel_limit, self.decel_limit)

        angular = self._rate_limit(self.prev_angular,
                                   self.angular_gain * angle_error,
                                   dt, self.omega_accel)

        twist = Twist()
        twist.linear.x = max(0.0, min(linear, self.max_linear))
        twist.angular.z = max(-self.max_angular,
                              min(self.max_angular, angular))
        self.prev_linear = twist.linear.x
        self.prev_angular = twist.angular.z
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
