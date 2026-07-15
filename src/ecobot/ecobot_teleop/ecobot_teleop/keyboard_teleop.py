import sys
import select
import tty
import termios
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')

        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('linear_step', 0.05)
        self.declare_parameter('angular_step', 0.1)

        self.max_linear = self.get_parameter('max_linear_speed').value
        self.max_angular = self.get_parameter('max_angular_speed').value
        self.linear_step = self.get_parameter('linear_step').value
        self.angular_step = self.get_parameter('angular_step').value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.settings = termios.tcgetattr(sys.stdin)

        self.linear_x = 0.0
        self.angular_z = 0.0

        self.get_logger().info(
            'ecobot keyboard teleop\n'
            '  w/s — forward/backward\n'
            '  a/d — turn left/right\n'
            '  space — stop\n'
            '  q — quit')

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        while rclpy.ok():
            key = self.get_key()
            if key == 'w':
                self.linear_x = min(self.linear_x + self.linear_step, self.max_linear)
            elif key == 's':
                self.linear_x = max(self.linear_x - self.linear_step, -self.max_linear)
            elif key == 'a':
                self.angular_z = min(self.angular_z + self.angular_step, self.max_angular)
            elif key == 'd':
                self.angular_z = max(self.angular_z - self.angular_step, -self.max_angular)
            elif key == ' ':
                self.linear_x = 0.0
                self.angular_z = 0.0
            elif key == 'q':
                break

            twist = Twist()
            twist.linear.x = self.linear_x
            twist.angular.z = self.angular_z
            self.cmd_pub.publish(twist)

        twist = Twist()
        self.cmd_pub.publish(twist)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        node.cmd_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
