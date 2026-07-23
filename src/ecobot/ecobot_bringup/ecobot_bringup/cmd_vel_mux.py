#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelMux(Node):
    """Relays the latest command from /nav_cmd_vel or /goto_cmd_vel to /cmd_vel."""

    def __init__(self):
        super().__init__('cmd_vel_mux')
        self._latest = None
        self._nav_sub = self.create_subscription(
            Twist, '/nav_cmd_vel', self._nav_cb, 10)
        self._goto_sub = self.create_subscription(
            Twist, '/goto_cmd_vel', self._goto_cb, 10)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('cmd_vel_mux: /nav_cmd_vel + /goto_cmd_vel -> /cmd_vel')

    def _nav_cb(self, msg):
        self._pub.publish(msg)

    def _goto_cb(self, msg):
        self._pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(CmdVelMux())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
