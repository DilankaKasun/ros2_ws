from __future__ import annotations

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

from pathlib import Path
from typing import Sequence

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
from transformers import AutoModel, AutoProcessor

from .arm_kinematics import ArmKinematics


STATE_DIM = 80
IMAGE_SIZE = (448, 448)


class MiniCPMVLAInference:
    """Processor and model wrapper for single-sample VLA inference."""

    def __init__(
        self,
        checkpoint_path: str | Path = "openbmb/MiniCPM-RobotManip",
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = str(checkpoint_path)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model.to(self.device).eval()

    @staticmethod
    def _load_images(images: Sequence[str | Path | PILImage.Image | np.ndarray]) -> list[np.ndarray]:
        if not images:
            raise ValueError("At least one image is required")
        loaded = []
        for image in images:
            if isinstance(image, (str, Path)):
                with PILImage.open(image) as pil_image:
                    array = np.asarray(pil_image.convert("RGB"))
            elif isinstance(image, PILImage.Image):
                array = np.asarray(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                array = image
            else:
                raise TypeError(f"Unsupported image type: {type(image)!r}")
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"Expected an HxWx3 image, got shape {array.shape}")
            resized = PILImage.fromarray(array).resize(IMAGE_SIZE)
            loaded.append(np.asarray(resized).copy())
        return loaded

    def preprocess(self, images: Sequence, text: str) -> dict[str, torch.Tensor]:
        """Apply the same MiniCPM-V chat template and processor as training."""
        content = [
            {"type": "image", "image": image}
            for image in self._load_images(images)
        ]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": False},
        )
        return {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

    def _prepare_state(self, state: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape != (1, 1, STATE_DIM):
            raise ValueError(f"state must have shape (80,), (1, 80), or (1, 1, 80); got {tuple(state.shape)}")
        return state

    @torch.inference_mode()
    def predict(
        self,
        images: Sequence[str | Path | PILImage.Image | np.ndarray],
        text: str,
        state: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        embodiment_id: int = 0,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Return one action chunk with shape ``(30, 80)`` on CPU."""
        if not 0 <= embodiment_id < self.model.action_head.max_num_embodiments:
            raise ValueError(
                f"embodiment_id must be in [0, {self.model.action_head.max_num_embodiments - 1}]"
            )
        if state is None:
            state = torch.zeros(STATE_DIM)
        state_tensor = self._prepare_state(state)
        embodiment = torch.tensor([embodiment_id], dtype=torch.long, device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        vlm_inputs = self.preprocess(images, text)
        actions = self.model.predict_action(
            state=state_tensor,
            embodiment_id=embodiment,
            **vlm_inputs,
        )
        return actions[0].float().cpu()


class MiniCPMVLANode(Node):
    """ROS 2 Bridge Node for OpenBMB MiniCPM-RobotManip VLA Model on EcoBot Arm."""

    def __init__(self):
        super().__init__('minicpm_vla_node')

        self.declare_parameter('dry_run', False)
        self.declare_parameter('model_path', 'openbmb/MiniCPM-RobotManip')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('prompt', 'reach forward')
        self.declare_parameter('l0', 0.110)
        self.declare_parameter('l1', 0.150)
        self.declare_parameter('l2', 0.130)
        self.declare_parameter('l3', 0.140)
        self.declare_parameter('ee_pos_indices', [7, 8, 9])
        self.declare_parameter('pos_scale', 0.05)
        self.declare_parameter('ee_delta_clamp', 0.05)
        self.declare_parameter('delta_clamp', 2.0)

        self._dry_run = self.get_parameter('dry_run').value
        self._model_path = self.get_parameter('model_path').value
        self._device = self.get_parameter('device').value
        self._current_prompt = self.get_parameter('prompt').value
        self._ee_pos_indices = self.get_parameter('ee_pos_indices').value
        self._pos_scale = float(self.get_parameter('pos_scale').value)
        self._ee_delta_clamp = float(self.get_parameter('ee_delta_clamp').value)
        self._delta_clamp = float(self.get_parameter('delta_clamp').value)

        # Arm kinematics for EE6D delta -> joint-angle routing
        self._ik = ArmKinematics(
            self.get_parameter('l0').value,
            self.get_parameter('l1').value,
            self.get_parameter('l2').value,
            self.get_parameter('l3').value,
        )

        self._bridge = CvBridge()
        self._latest_image = None
        self._warned_no_camera = False

        # Default to EcoBot home angles: Base=107, Shoulder=125, Elbow=180, Wrist=50
        self._current_joint_angles = [107.0, 125.0, 180.0, 50.0]
        self._prev_joint_angles = list(self._current_joint_angles)
        self._inference_runner = None
        self._model_loaded = False

        self.get_logger().info(
            f"Initializing MiniCPM-RobotManip VLA Node (Device: {self._device}, Dry Run: {self._dry_run}, Prompt: '{self._current_prompt}')"
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

        # Initialize MiniCPM inference engine
        self._init_minicpm_model()

        # Send enable message to arm hardware
        if not self._dry_run:
            self.create_timer(1.0, self._enable_arm_hardware)

        # Timer loop for VLA inference (2 Hz)
        self._timer = self.create_timer(0.5, self._inference_step)

    def _enable_arm_hardware(self):
        msg = String()
        msg.data = 'enable'
        self._enable_pub.publish(msg)

    def _init_minicpm_model(self):
        try:
            self.get_logger().info(f"Loading MiniCPM-RobotManip model wrapper from '{self._model_path}'...")
            self._inference_runner = MiniCPMVLAInference(
                checkpoint_path=self._model_path,
                device=self._device,
            )
            self._model_loaded = True
            self.get_logger().info("MiniCPM-RobotManip inference engine ready!")
        except Exception as e:
            self.get_logger().warn(f"MiniCPM model direct load notice: {e}. Running in VLA simulation/bridge mode.")

    def _camera_cb(self, msg: Image):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")

    def _joint_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            self._prev_joint_angles = list(self._current_joint_angles)
            self._current_joint_angles = list(msg.data[:4])

    def _prompt_cb(self, msg: String):
        self._current_prompt = msg.data
        self.get_logger().info(f"New MiniCPM VLA Prompt Received: '{self._current_prompt}'")

    def _inference_step(self):
        self._dry_run = self.get_parameter('dry_run').value

        image_input = self._latest_image
        if image_input is None:
            if not self._warned_no_camera:
                self.get_logger().warn("No live camera feed detected on /arm/camera/image_raw or /camera/color/image_raw. Using synthetic camera frame for MiniCPM VLA step.")
                self._warned_no_camera = True
            image_input = np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert BGR CV image to RGB
        rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        # Predict target joint angles using MiniCPM-RobotManip policy
        predicted_joints = self._predict_vla_action(rgb_image, self._current_joint_angles, self._current_prompt)

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

        status_str = f"[{mode_str} - MiniCPM] Prompt: '{self._current_prompt}' -> Target Joints: {list(np.round(predicted_joints[:4], 1))}"
        self._status_pub.publish(String(data=status_str))
        self.get_logger().info(status_str)

    def _build_state(self, current_joints: list) -> np.ndarray:
        """Build the 80-dim state vector fed to MiniCPM-RobotManip.

        Layout mirrors the training pipeline: joint positions and velocities are
        packed in the first slots (normalized to [-1, 1]), rest zero-padded.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        joints = np.array(current_joints[:4], dtype=np.float32)
        prev = np.array(self._prev_joint_angles[:4], dtype=np.float32)
        state[0:4] = joints / 180.0
        state[4:8] = (joints - prev) / 180.0
        return state

    def _predict_vla_action(self, rgb_image: np.ndarray, current_joints: list, prompt: str) -> list:
        """Process RGB frame, 80-dim state vector, and prompt to compute continuous joint actions."""
        target = list(current_joints[:4])

        # If model is loaded, attempt running prediction
        if self._model_loaded and self._inference_runner is not None:
            try:
                state = self._build_state(current_joints)

                action_chunk = self._inference_runner.predict(
                    images=[rgb_image],
                    text=prompt,
                    state=state,
                    embodiment_id=0,
                )
                if action_chunk is not None and len(action_chunk) > 0:
                    first_action = action_chunk[0].numpy()
                    # Route the model's EE6D position deltas (dims 7/8/9 = x/y/z
                    # in the LIBERO-style left-arm slot) through a local numerical
                    # Jacobian, so the Cartesian delta maps to smooth joint deltas
                    # in the EcoBot's own frame (avoids IK branch flips).
                    clamp = lambda v: max(-self._ee_delta_clamp, min(self._ee_delta_clamp, float(v)))
                    d_ee = np.array([
                        clamp(first_action[self._ee_pos_indices[0]]),
                        clamp(first_action[self._ee_pos_indices[1]]),
                        clamp(first_action[self._ee_pos_indices[2]]),
                    ]) * self._pos_scale
                    d_theta = self._incremental_ik(current_joints[:4], d_ee)
                    if d_theta is not None:
                        lo, hi = 30.0, 125.0
                        for i in range(4):
                            target[i] = max(lo if i == 1 else 0.0, min(
                                hi if i == 1 else 220.0, current_joints[i] + d_theta[i]))
                    else:
                        self.get_logger().warn('Cartesian delta unreachable, holding')
            except Exception as e:
                self.get_logger().error(f"MiniCPM prediction step error: {e}")

        # Enforce hardware bounds
        target[0] = float(np.clip(target[0], 0.0, 220.0))
        target[1] = float(np.clip(target[1], 0.0, 125.0))
        target[2] = float(np.clip(target[2], 0.0, 180.0))
        target[3] = float(np.clip(target[3], 0.0, 180.0))

        return target

    def _incremental_ik(self, joints: list, d_ee: np.ndarray):
        """Map a small Cartesian end-effector delta to joint deltas via a local
        numerical Jacobian (4x3, pseudo-inverted). Returns None if singular."""
        eps = 0.5
        cur = np.array(self._ik.forward(*joints))
        J = np.zeros((3, 4))
        for i in range(4):
            j = list(joints)
            j[i] += eps
            J[:, i] = (np.array(self._ik.forward(*j)) - cur) / eps
        try:
            Jp = np.linalg.pinv(J)
        except Exception:
            return None
        d_theta = Jp @ d_ee
        d_theta = np.clip(d_theta, -self._delta_clamp, self._delta_clamp)
        return d_theta


def main(args=None):
    rclpy.init(args=args)
    node = MiniCPMVLANode()
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
