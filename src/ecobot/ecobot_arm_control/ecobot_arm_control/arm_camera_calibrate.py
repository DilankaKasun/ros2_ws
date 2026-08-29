import json
import math
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, String

from .arm_kinematics import ArmKinematics
from .servo_config import JOINTS, NUM_JOINTS, to_ik


class ArmCameraCalibrate(Node):
    """Interactive calibration of the camera<->arm static transform.

    Drive the arm (via the dashboard sliders) so the wrist sits at a
    detected target's position — e.g. put the wrist next to a bottle the
    base camera can see. Then press Enter on each sample; the tool reads:

      * detection (xc, yc, zc) in camera optical frame
      * arm endpoint (xa, ya, za) in the arm IK frame, via FK on the
        live /arm/joint_angles

    and solves T_x = xa - zc, T_y = ya + xc, T_z = za + yc (yaw=0,
    camera z-forward / x-right / y-down maps to arm x-forward / y-left /
    z-up). Three or more spread-out samples are averaged, then saved to
    arm_tracking_params.yaml.

    Commands (stdin): <Enter> capture, 's'/'save' write file, 'q'/'quit' exit.
    """

    def __init__(self, out_yaml=None):
        super().__init__('arm_camera_calibrate')

        self.declare_parameter('l0', 0.320)
        self.declare_parameter('l1', 0.165)
        self.declare_parameter('l2', 0.140)
        self.declare_parameter('l3', 0.090)

        self._ik = ArmKinematics(
            self.get_parameter('l0').value,
            self.get_parameter('l1').value,
            self.get_parameter('l2').value,
            self.get_parameter('l3').value,
        )

        default_out = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'arm_tracking_params.yaml')
        self._out_yaml = out_yaml or default_out

        self._detections = []
        self._joint_angles = None
        self._samples = []

        self._det_sub = self.create_subscription(
            String, '/ecobot/detections', self._det_cb, 10)
        self._joint_sub = self.create_subscription(
            Float64MultiArray, '/arm/joint_angles', self._joint_cb, 10)

        self.get_logger().info(
            'Calibration ready. Move the wrist to a detected object, then:\n'
            "  <Enter>  capture this sample\n"
            "  s/save   average + save to %s\n"
            "  q/quit   exit without saving", self._out_yaml)

    def _det_cb(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self._detections = data
        except Exception:
            pass

    def _joint_cb(self, msg):
        if len(msg.data) >= NUM_JOINTS:
            self._joint_angles = list(msg.data[:NUM_JOINTS])

    def _nearest_detection(self):
        best = None
        best_z = math.inf
        for d in self._detections:
            z = float(d.get('z') or 0.0)
            if z <= 0.0 or z > 5.0:
                continue
            if z < best_z:
                best_z = z
                best = d
        return best

    def _capture(self):
        if self._joint_angles is None:
            self.get_logger().warn('No /arm/joint_angles yet — is the arm running?')
            return
        det = self._nearest_detection()
        if det is None:
            self.get_logger().warn(
                'No valid detection (z in 0..5m) on /ecobot/detections')
            return

        xc = float(det.get('x') or 0.0)
        yc = float(det.get('y') or 0.0)
        zc = float(det.get('z') or 0.0)
        name = str(det.get('class_name') or det.get('class') or '?')

        xa, ya, za = self._ik.forward(*to_ik(self._joint_angles))
        tx = xa - zc
        ty = ya + xc
        tz = za + yc

        self._samples.append((tx, ty, tz))
        self.get_logger().info(
            f'sample {len(self._samples)} [{name}]: cam='
            f'({xc:.3f},{yc:.3f},{zc:.3f}) arm=({xa:.3f},{ya:.3f},{za:.3f}) '
            f'-> T=({tx:.3f},{ty:.3f},{tz:.3f})')

    def _save(self):
        if not self._samples:
            self.get_logger().warn('No samples captured')
            return
        n = len(self._samples)
        avg = tuple(sum(s[i] for s in self._samples) / n for i in range(3))
        spread = max(
            math.hypot(s[0] - avg[0], s[1] - avg[1]) for s in self._samples)
        self.get_logger().info(
            f'averaged {n} samples -> T=({avg[0]:.3f},{avg[1]:.3f},{avg[2]:.3f}) '
            f'spread={spread:.3f}m')
        if spread > 0.03:
            self.get_logger().warn(
                'Samples are spread > 3cm apart — consider re-measuring')

        content = (
            'arm_target_tracker:\n'
            '  ros__parameters:\n'
            f"    transform_tx: {avg[0]:.4f}\n"
            f"    transform_ty: {avg[1]:.4f}\n"
            f"    transform_tz: {avg[2]:.4f}\n"
            '    transform_yaw: 0.0\n'
        )
        os.makedirs(os.path.dirname(self._out_yaml), exist_ok=True)
        with open(self._out_yaml, 'w') as f:
            f.write(content)
        self.get_logger().info(f'Saved -> {self._out_yaml}')

    def run_interactive(self):
        while rclpy.ok():
            try:
                line = input('>> ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if line in ('q', 'quit', 'exit'):
                break
            if line in ('s', 'save'):
                self._save()
            else:
                self._capture()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ArmCameraCalibrate()
    try:
        node.run_interactive()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
