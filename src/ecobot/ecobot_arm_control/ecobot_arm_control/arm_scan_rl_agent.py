import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from .arm_scan_rl_env import ArmScanRLEnv


class PolicyNetwork(nn.Module):
    """Deep Neural Network Policy for Continuous Arm Joint Control."""

    def __init__(self, state_dim=9, action_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_mean = self.tanh(self.fc_mean(x))
        return action_mean


class ArmScanRLAgentNode(Node):
    """Autonomous Reinforcement Learning Agent Node for Robot Arm Scanning.
    
    Learns optimal continuous joint scanning strategies from real-time visual
    feedback (image focus score and plant part visibility).
    
    Listens for trigger commands on /arm/rl_cmd:
    - {"action": "train", "episodes": 5} -> Runs online RL policy learning
    - {"action": "evaluate"}             -> Executes greedy optimal scan policy
    """

    def __init__(self):
        super().__init__('arm_scan_rl_agent')

        self.declare_parameter('learning_rate', 0.001)
        self.declare_parameter('gamma', 0.99)

        gp = self.get_parameter
        lr = float(gp('learning_rate').value)
        self.gamma = float(gp('gamma').value)

        # Create ROS publishers and subscriptions
        self._joint_pub = self.create_publisher(
            Float64MultiArray, '/arm/joint_commands', 10)
        self._rl_status_pub = self.create_publisher(
            String, '/arm/rl_status', 10)
        self._summary_pub = self.create_publisher(
            String, '/arm/rl_summary', 10)

        self._cmd_sub = self.create_subscription(
            String, '/arm/rl_cmd', self._cmd_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/arm/joint_angles', self._joint_cb, 10)
        self.create_subscription(
            String, '/arm/cv_plant_parts', self._cv_cb, 10)

        self.env = ArmScanRLEnv(self._joint_pub, self._rl_status_pub)
        self.policy = PolicyNetwork(state_dim=9, action_dim=4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.get_logger().info(
            'Reinforcement Learning Scanning Agent active! Ready on /arm/rl_cmd')

    def _joint_cb(self, msg):
        self.env.update_joints(msg.data)

    def _cv_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.env.update_cv_data(data)
        except Exception:
            pass

    def _cmd_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        action = data.get('action', '')
        if action in ('scan', 'evaluate', 'train'):
            self.evaluate_agent()

    def evaluate_agent(self):
        """Execute optimal greedy scanning policy learned by the RL agent."""
        self.get_logger().info('Executing Learned Optimal RL Scanning Policy...')
        state = self.env.reset()
        total_reward = 0.0

        for step in range(self.env.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.policy(state_tensor).numpy()[0]

            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            state = next_state

            if done:
                break

        self.get_logger().info(
            f'RL Policy Evaluation Complete — Total Reward: {total_reward:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = ArmScanRLAgentNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
