#!/usr/bin/env python3
"""
EcoBot 4-DOF Robot Arm Interactive Keyboard Teleoperation.

Allows real-time manual jogging of all 4 arm joints:
  - Base (Joint 0 / Ch 1)
  - Shoulder (Joint 1 / Ch 4)
  - Elbow (Joint 2 / Ch 3)
  - Wrist (Joint 3 / Ch 2)

Operates in two modes:
  1. ROS 2 Mode: Publishes to `/arm/joint_commands` if `arm_control_node` is active.
  2. Direct I2C Mode: Directly commands PCA9685 if running standalone.
"""

import os
import sys
import time
import select
import tty
import termios
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from .servo_config import JOINTS, NUM_JOINTS, within_limits
from .pca9685_driver import PCA9685


class ArmKeyboardTeleop(Node):
    def __init__(self):
        super().__init__('arm_keyboard_teleop')

        self.declare_parameter('step_deg', 5.0)
        self.declare_parameter('direct_i2c', False)

        self._step_deg = float(self.get_parameter('step_deg').value)
        self._direct_i2c = bool(self.get_parameter('direct_i2c').value)

        # Current joint angles
        self._current_angles: List[float] = [float(j['home_angle']) for j in JOINTS]
        self._enabled = True

        # ROS 2 Publisher
        self._cmd_pub = self.create_publisher(Float64MultiArray, '/arm/joint_commands', 10)
        self._enable_pub = self.create_publisher(String, '/arm/enable', 10)

        # Standalone I2C Driver
        self._pca = None
        if self._direct_i2c:
            try:
                self._pca = PCA9685(bus=7, address=0x40, freq=50.0)
                self._write_pca_all()
            except Exception as e:
                self.get_logger().error(f"Failed to initialize direct PCA9685: {e}")

        self._settings = termios.tcgetattr(sys.stdin)

    def _write_pca_all(self):
        if self._pca and self._enabled:
            for i, j in enumerate(JOINTS):
                self._pca.set_angle(
                    j['channel'], self._current_angles[i],
                    j['pulse_min'], j['pulse_max'], j['servo_range']
                )

    def _publish_cmd(self):
        msg = Float64MultiArray()
        msg.data = list(self._current_angles)
        self._cmd_pub.publish(msg)
        if self._pca:
            self._write_pca_all()

    def _get_key(self) -> str:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)
        return key

    def _render_ui(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        status_str = "\033[1;32mACTIVE / ENABLED\033[0m" if self._enabled else "\033[1;31mDISABLED\033[0m"
        print("\033[1;36m=================================================================\033[0m")
        print("\033[1;37m              ECOBOT 4-DOF ARM MANUAL KEYBOARD TELEOP             \033[0m")
        print("\033[1;36m=================================================================\033[0m")
        print(f" Status: {status_str}  |  Step Size: \033[1;33m{self._step_deg:.1f}°\033[0m\n")

        print(" \033[1mJoint Name   Channel   Current Angle   Min..Max Range   Controls\033[0m")
        print(" -----------------------------------------------------------------")
        keys_dec = ['1', '2', '3', '4']
        keys_inc = ['q', 'w', 'e', 'r']

        for i, j in enumerate(JOINTS):
            ang = self._current_angles[i]
            bar_len = 20
            pct = (ang - j['min_angle']) / max(1.0, (j['max_angle'] - j['min_angle']))
            filled = int(pct * bar_len)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f" {j['label']:<10}  Ch {j['channel']}    \033[1;32m{ang:>5.1f}°\033[0m  [{bar}] {j['min_angle']:>3.0f}°..{j['max_angle']:>3.0f}°  [{keys_dec[i]}] - / [{keys_inc[i]}] +")

        print("\n \033[1;33mGlobal Controls:\033[0m")
        print("   [H] / [Space] : Go to HOME position ([107°, 125°, 180°, 45°])")
        print("   [+] / [-]     : Increase / Decrease step size (1°, 5°, 10°, 15°)")
        print("   [D]           : Disable / Relax all servos")
        print("   [E]           : Re-enable servos")
        print("   [X] / [Q] / ESC: Exit teleoperation")
        print("\033[1;36m=================================================================\033[0m")
        print(" Ready for keypress... ", end='', flush=True)

    def run(self):
        self._render_ui()
        self._publish_cmd()

        step_presets = [1.0, 2.5, 5.0, 10.0, 15.0]
        step_idx = step_presets.index(self._step_deg) if self._step_deg in step_presets else 2

        try:
            while rclpy.ok():
                key = self._get_key().lower()
                if not key:
                    rclpy.spin_once(self, timeout_sec=0.01)
                    continue

                updated = False

                # Joint 0: Base
                if key == '1':
                    self._current_angles[0] = max(JOINTS[0]['min_angle'], self._current_angles[0] - self._step_deg)
                    updated = True
                elif key == 'q':
                    self._current_angles[0] = min(JOINTS[0]['max_angle'], self._current_angles[0] + self._step_deg)
                    updated = True

                # Joint 1: Shoulder
                elif key == '2':
                    self._current_angles[1] = max(JOINTS[1]['min_angle'], self._current_angles[1] - self._step_deg)
                    updated = True
                elif key == 'w':
                    self._current_angles[1] = min(JOINTS[1]['max_angle'], self._current_angles[1] + self._step_deg)
                    updated = True

                # Joint 2: Elbow
                elif key == '3':
                    self._current_angles[2] = max(JOINTS[2]['min_angle'], self._current_angles[2] - self._step_deg)
                    updated = True
                elif key == 'e':
                    self._current_angles[2] = min(JOINTS[2]['max_angle'], self._current_angles[2] + self._step_deg)
                    updated = True

                # Joint 3: Wrist
                elif key == '4':
                    self._current_angles[3] = max(JOINTS[3]['min_angle'], self._current_angles[3] - self._step_deg)
                    updated = True
                elif key == 'r':
                    self._current_angles[3] = min(JOINTS[3]['max_angle'], self._current_angles[3] + self._step_deg)
                    updated = True

                # Home Pose
                elif key in ('h', ' '):
                    self._current_angles = [float(j['home_angle']) for j in JOINTS]
                    updated = True

                # Step size
                elif key in ('+', '='):
                    step_idx = min(len(step_presets) - 1, step_idx + 1)
                    self._step_deg = step_presets[step_idx]
                    self._render_ui()
                elif key in ('-', '_'):
                    step_idx = max(0, step_idx - 1)
                    self._step_deg = step_presets[step_idx]
                    self._render_ui()

                # Enable / Disable
                elif key == 'd':
                    self._enabled = False
                    en_msg = String()
                    en_msg.data = 'disable'
                    self._enable_pub.publish(en_msg)
                    if self._pca:
                        self._pca.disable()
                    self._render_ui()
                elif key == 'e':
                    self._enabled = True
                    en_msg = String()
                    en_msg.data = 'enable'
                    self._enable_pub.publish(en_msg)
                    if self._pca:
                        self._pca.enable()
                    updated = True

                # Exit
                elif key in ('x', '\x1b', '\x03'):  # 'x', ESC, Ctrl+C
                    break

                if updated:
                    self._publish_cmd()
                    self._render_ui()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)


def main(args=None):
    rclpy.init(args=args)
    teleop = ArmKeyboardTeleop()
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
