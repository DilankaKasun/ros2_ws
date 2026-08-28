import json
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


class ArmScanRLEnv:
    """ROS 2 Reinforcement Learning Environment for Robot Arm Autonomous Plant Scanning."""

    def __init__(self, joint_pub, rl_status_pub):
        self.joint_pub = joint_pub
        self.rl_status_pub = rl_status_pub
        self.current_joints = [107.0, 125.0, 180.0, 45.0]
        self.latest_cv_data = {
            'detected_part': 'unknown',
            'confidence': 0.0,
            'focus_score': 0.0,
            'in_focus': False,
            'green_foliage_ratio': 0.0,
        }
        self.target_pos = [0.40, 0.0, 0.20]  # Plant target position in meters
        self.step_count = 0
        self.max_steps = 50
        self.consecutive_unknown_count = 0
        self.visited_orbit_sectors = set()

    def update_joints(self, joints):
        if len(joints) >= 4:
            self.current_joints = list(joints[:4])

    def update_cv_data(self, cv_dict):
        if isinstance(cv_dict, dict):
            self.latest_cv_data = cv_dict

    def get_observation(self):
        """Construct the 9-dimensional state vector."""
        j_norm = [
            self.current_joints[0] / 220.0,
            self.current_joints[1] / 125.0,
            self.current_joints[2] / 180.0,
            self.current_joints[3] / 180.0,
        ]
        dx = self.target_pos[0] - 0.40
        dy = self.target_pos[1] - 0.0
        dz = self.target_pos[2] - 0.20

        focus = min(1.0, float(self.latest_cv_data.get('focus_score', 0.0)) / 200.0)
        part_ratio = float(self.latest_cv_data.get('green_foliage_ratio', 0.0))

        state = np.array(
            j_norm + [dx, dy, dz, focus, part_ratio], dtype=np.float32)
        return state

    def step(self, action):
        """Apply action, compute reward, and return (next_state, reward, done, info)."""
        self.step_count += 1

        # Map action [-1, 1] -> joint angle deltas (max 5 deg per RL step)
        action = np.clip(action, -1.0, 1.0)
        joint_deltas = action * 5.0

        # Enforce exact joint limits for all 4 joints
        # J1 Base: 0..220, J2 Shoulder: 0..125, J3 Elbow: 0..180, J4 Wrist: 0..180
        limits_max = [220.0, 125.0, 180.0, 180.0]
        target_joints = [
            float(np.clip(self.current_joints[i] + joint_deltas[i], 0.0, limits_max[i]))
            for i in range(4)
        ]

        # Publish joint action
        cmd_msg = Float64MultiArray()
        cmd_msg.data = target_joints
        self.joint_pub.publish(cmd_msg)

        # Allow time for servo execution & camera feed update
        time.sleep(0.08)

        # Calculate Reward Function
        focus_score = float(self.latest_cv_data.get('focus_score', 0.0))
        in_focus = bool(self.latest_cv_data.get('in_focus', False))
        part = str(self.latest_cv_data.get('detected_part', 'unknown'))
        conf = float(self.latest_cv_data.get('confidence', 0.0))

        reward = 0.0
        done = False

        # HIGH-PENALTY FAILURE STATE: Scanning empty space / pointing away from plant
        if part == 'unknown' or conf < 0.15:
            self.consecutive_unknown_count += 1
            reward = -25.0  # High-penalty failure state forcing learning policy away from empty space
            
            # INEFFECTIVE MOTION RECOVERY: Trigger recovery reset if pointing away for >= 3 consecutive steps
            if self.consecutive_unknown_count >= 3:
                self.recovery_reset()
                done = True
        else:
            self.consecutive_unknown_count = 0
            
            # Positive Feedback: Plant part recognized and centered in frame
            reward += 10.0 * conf
            if in_focus:
                reward += 5.0 + (focus_score / 50.0)

            # Multi-Angle Orbit Coverage Reward: Reward for exploring different base sectors (left, center, right)
            base_angle = target_joints[0]
            sector = int(base_angle // 45) # 45-degree sectors
            if sector not in self.visited_orbit_sectors:
                self.visited_orbit_sectors.add(sector)
                reward += 5.0  # Multi-angle coverage bonus

            # Wrist Joint Active Utilization Reward: Bonus when wrist orientation keeps plant in focus
            if abs(joint_deltas[3]) > 0.5 and in_focus:
                reward += 2.5

            # Motion jitter penalty to keep movements smooth
            motion_penalty = 0.01 * float(np.sum(joint_deltas ** 2))
            reward -= motion_penalty

        if self.step_count >= self.max_steps:
            done = True
            if part != 'unknown':
                reward += 15.0  # High reward for full trajectory visual tracking

        next_state = self.get_observation()
        info = {
            'focus_score': focus_score,
            'part': part,
            'joints': target_joints,
            'visited_sectors': len(self.visited_orbit_sectors),
        }

        # Publish RL Status Update
        rl_info = {
            'step': self.step_count,
            'reward': round(float(reward), 2),
            'focus_score': round(focus_score, 1),
            'detected_part': part,
            'action_deltas': [round(float(d), 2) for d in joint_deltas],
            'visited_sectors': len(self.visited_orbit_sectors),
        }
        status_msg = String()
        status_msg.data = json.dumps(rl_info)
        self.rl_status_pub.publish(status_msg)

        return next_state, reward, done, info

    def recovery_reset(self):
        """Recovery routine: Return arm to known starting position when target is lost."""
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [107.0, 125.0, 180.0, 45.0]
        self.joint_pub.publish(cmd_msg)
        time.sleep(0.4)

    def reset(self):
        """Reset environment to initial home pose and clear tracking sets."""
        self.step_count = 0
        self.consecutive_unknown_count = 0
        self.visited_orbit_sectors.clear()
        self.recovery_reset()
        return self.get_observation()
