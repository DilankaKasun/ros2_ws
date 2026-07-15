#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')
        self.sub = self.create_subscription(
            Twist, '/nav_cmd_vel', self.cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('cmd_vel_mux: /nav_cmd_vel -> /cmd_vel')

    def cb(self, msg):
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(CmdVelMux())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
