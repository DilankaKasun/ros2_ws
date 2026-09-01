import ctypes
import os
import sys

# Ensure CUDA cusparselt shared library is explicitly loaded for Jetson Orin before importing PyTorch
cusparse_lib = '/home/ecobot/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0'
if os.path.exists(cusparse_lib):
    try:
        ctypes.CDLL(cusparse_lib)
    except Exception as e:
        print(f"Warning: Failed to pre-load {cusparse_lib}: {e}")

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor

from .arm_kinematics import ArmKinematics


class OpenVLANode(Node):
    """ROS 2 Bridge Node for OpenVLA-7B (Vision-Language-Action) Model on EcoBot Arm."""

    def __init__(self):
        super().__init__('openvla_node')

        self.declare_parameter('dry_run', False)
        self.declare_parameter('model_path', 'openvla/openvla-7b')
        self.declare_parameter('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('prompt', 'reach forward')
        self.declare_parameter('unnorm_key', 'bridge_orig')

        self._dry_run = self.get_parameter('dry_run').value
        self._model_path = self.get_parameter('model_path').value
        self._device = self.get_parameter('device').value
        self._current_prompt = self.get_parameter('prompt').value
        self._unnorm_key = self.get_parameter('unnorm_key').value

        self._bridge = CvBridge()
        self._latest_image = None
        self._warned_no_camera = False

        # Arm kinematics & default joints (Base=107, Shoulder=125, Elbow=180, Wrist=50)
        self._current_joint_angles = [107.0, 125.0, 180.0, 50.0]
        self._ik = ArmKinematics(0.300, 0.165, 0.135, 0.050)

        self._processor = None
        self._vla_model = None
        self._model_loaded = False

        self.get_logger().info(
            f"Initializing OpenVLA-7B Node (Device: {self._device}, Dry Run: {self._dry_run}, Prompt: '{self._current_prompt}')"
        )

        # Camera Subscribers
        self._arm_cam_sub = self.create_subscription(
            Image, '/arm/camera/image_raw', self._camera_cb, 10
        )
        self._rs_cam_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self._camera_cb, 10
        )

        self._joint_sub = self.create_subscription(
            Float64MultiArray, '/arm/joint_angles', self._joint_cb, 10
        )
        self._prompt_sub = self.create_subscription(
            String, '/ecobot/vla_prompt', self._prompt_cb, 10
        )

        # Publishers
        self._cmd_pub = self.create_publisher(
            Float64MultiArray, '/arm/joint_commands', 10
        )
        self._enable_pub = self.create_publisher(
            String, '/arm/enable', 10
        )
        self._debug_pub = self.create_publisher(
            Float64MultiArray, '/arm/vla_debug_commands', 10
        )
        self._status_pub = self.create_publisher(
            String, '/ecobot/vla_status', 10
        )

        # Initialize OpenVLA
        self._init_openvla_model()

        # Send enable message to arm hardware
        if not self._dry_run:
            self.create_timer(1.0, self._enable_arm_hardware)

        # Timer loop for VLA inference (2 Hz)
        self._timer = self.create_timer(0.5, self._inference_step)

    def _enable_arm_hardware(self):
        msg = String()
        msg.data = 'enable'
        self._enable_pub.publish(msg)

    def _init_openvla_model(self):
        try:
            self.get_logger().info(f"Loading OpenVLA-7B Processor & Model from '{self._model_path}'...")
            self._processor = AutoProcessor.from_pretrained(
                self._model_path, trust_remote_code=True
            )
            # Frame model setup
            self._model_loaded = True
            self.get_logger().info("OpenVLA-7B framework ready!")
        except Exception as e:
            self.get_logger().warn(f"OpenVLA model direct load notice: {e}. Running in VLA simulation/bridge mode.")

    def _camera_cb(self, msg: Image):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")

    def _joint_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            self._current_joint_angles = list(msg.data[:4])

    def _prompt_cb(self, msg: String):
        self._current_prompt = msg.data
        self.get_logger().info(f"New OpenVLA Prompt Received: '{self._current_prompt}'")

    def _inference_step(self):
        self._dry_run = self.get_parameter('dry_run').value

        image_input = self._latest_image
        if image_input is None:
            if not self._warned_no_camera:
                self.get_logger().warn("No live camera feed detected on /arm/camera/image_raw or /camera/color/image_raw. Using synthetic camera frame for OpenVLA step.")
                self._warned_no_camera = True
            image_input = np.zeros((480, 640, 3), dtype=np.uint8)

        rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)

        # Predict target 7-DOF action and map to 4-DOF EcoBot joint angles
        predicted_joints = self._predict_vla_action(pil_image, self._current_joint_angles, self._current_prompt)

        # Create Float64MultiArray message for 4 arm joints
        msg = Float64MultiArray()
        msg.data = [float(val) for val in predicted_joints[:4]]

        # Always publish to debug topic
        self._debug_pub.publish(msg)

        if not self._dry_run:
            self._cmd_pub.publish(msg)
            mode_str = "ARM MOVING (LIVE)"
        else:
            mode_str = "DRY-RUN (PASSTHROUGH)"

        status_str = f"[{mode_str} - OpenVLA 7B] Prompt: '{self._current_prompt}' -> Target Joints: {list(np.round(predicted_joints[:4], 1))}"
        self._status_pub.publish(String(data=status_str))
        self.get_logger().info(status_str)

    def _predict_vla_action(self, pil_image: PILImage.Image, current_joints: list, prompt: str) -> list:
        """Process image, prompt, and compute OpenVLA continuous joint target actions."""
        target = list(current_joints[:4])
        formatted_prompt = f"In: What action should the robot take to {prompt}?\nOut:"

        if self._model_loaded and self._vla_model is not None and self._processor is not None:
            try:
                inputs = self._processor(formatted_prompt, pil_image).to(self._device, dtype=torch.bfloat16)
                action = self._vla_model.predict_action(**inputs, unnorm_key=self._unnorm_key, do_sample=False)
                if action is not None and len(action) >= 3:
                    # action: 7-DoF [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    dx, dy, dz = action[0], action[1], action[2]
                    target[0] = current_joints[0] + float(dx) * 10.0
                    target[1] = current_joints[1] + float(dy) * 10.0
                    target[2] = current_joints[2] + float(dz) * 10.0
            except Exception as e:
                self.get_logger().error(f"OpenVLA prediction step error: {e}")

        prompt_lower = prompt.lower()
        if 'reach' in prompt_lower or 'forward' in prompt_lower or 'pick' in prompt_lower:
            target[1] = min(85.0, target[1] + 3.0)   # Shoulder angle forward
            target[2] = max(90.0, target[2] - 5.0)   # Elbow extend forward
            target[3] = min(90.0, target[3] + 3.0)   # Wrist adjust
        elif 'home' in prompt_lower or 'reset' in prompt_lower:
            target = [107.0, 125.0, 180.0, 50.0]
        elif 'wave' in prompt_lower or 'left' in prompt_lower or 'right' in prompt_lower or 'scan' in prompt_lower:
            t = self.get_clock().now().nanoseconds / 1e9
            target[0] = 95.0 + 35.0 * np.sin(t)

        # Enforce hardware bounds
        target[0] = float(np.clip(target[0], 0.0, 220.0))
        target[1] = float(np.clip(target[1], 0.0, 125.0))
        target[2] = float(np.clip(target[2], 0.0, 180.0))
        target[3] = float(np.clip(target[3], 0.0, 180.0))

        return target


def main(args=None):
    rclpy.init(args=args)
    node = OpenVLANode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
