#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from math import sin, cos


class SendGoal(Node):
    def __init__(self, x, y, yaw):
        super().__init__('send_goal')
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.goal_sent = False

        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 action server not available')
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = sin(yaw / 2.0)
        goal.pose.pose.orientation.w = cos(yaw / 2.0)

        self.get_logger().info(f'Sending goal: x={x}, y={y}, yaw={yaw}')
        send_future = self.client.send_goal_async(goal)
        send_future.add_done_callback(self.goal_response_cb)
        self.goal_sent = True

    def goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            rclpy.shutdown()
            return
        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_cb)

    def result_cb(self, future):
        result = future.result()
        if result.status == 4:
            self.get_logger().info('Goal SUCCEEDED')
        else:
            self.get_logger().error(f'Goal failed with status: {result.status}')
        rclpy.shutdown()


def main():
    if len(sys.argv) < 4:
        print('Usage: send_goal <x> <y> <yaw_radians>')
        return

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    yaw = float(sys.argv[3])

    rclpy.init()
    node = SendGoal(x, y, yaw)
    if node.goal_sent:
        rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
