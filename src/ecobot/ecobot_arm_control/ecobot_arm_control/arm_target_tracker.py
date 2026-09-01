import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, String

from .arm_kinematics import ArmKinematics
from .servo_config import (
    JOINTS, NUM_JOINTS, to_ik, ik_limits,
)


class ArmTargetTracker(Node):
    """Adaptive manipulation: keep the wrist camera gazing at a live-detected
    object after the base has parked near it.

    Detection points arrive in the camera optical frame (x=right, y=down,
    z=forward/depth). They are transformed into the arm IK frame
    (x=forward, y=left, z=up) using a static camera<->arm rigid transform
    calibrated with `arm_camera_calibrate`, then a standoff waypoint is
    computed on the base->object line and published as a pose goal.
    """

    def __init__(self):
        super().__init__('arm_target_tracker')

        self.declare_parameter('target_classes', ['bottle', 'cup'])
        self.declare_parameter('standoff', 0.10)
        self.declare_parameter('control_rate', 5.0)
        self.declare_parameter('lost_timeout', 2.0)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('transform_tx', 0.0)
        self.declare_parameter('transform_ty', 0.0)
        self.declare_parameter('transform_tz', 0.0)
        self.declare_parameter('transform_yaw', 0.0)
        self.declare_parameter('l0', 0.300)
        self.declare_parameter('l1', 0.165)
        self.declare_parameter('l2', 0.135)
        self.declare_parameter('l3', 0.050)
        self.declare_parameter('sweep_step', 0.02)
        self.declare_parameter('sweep_max', 0.15)

        self._classes = set(
            str(c).strip().lower()
            for c in self.get_parameter('target_classes').value)
        self._standoff = float(self.get_parameter('standoff').value)
        self._rate = float(self.get_parameter('control_rate').value)
        self._lost_timeout = float(self.get_parameter('lost_timeout').value)
        self._dry_run = bool(self.get_parameter('dry_run').value)
        self._tx = float(self.get_parameter('transform_tx').value)
        self._ty = float(self.get_parameter('transform_ty').value)
        self._tz = float(self.get_parameter('transform_tz').value)
        self._yaw = math.radians(float(self.get_parameter('transform_yaw').value))
        self._sweep_step = float(self.get_parameter('sweep_step').value)
        self._sweep_max = float(self.get_parameter('sweep_max').value)

        self._ik = ArmKinematics(
            self.get_parameter('l0').value,
            self.get_parameter('l1').value,
            self.get_parameter('l2').value,
            self.get_parameter('l3').value,
        )

        self._detections = []
        self._joint_angles = [float(j['home_angle']) for j in JOINTS]
        self._target_class = sorted(self._classes)[0] if self._classes else None
        self._tracking = False
        self._last_seen = None
        self._last_status = 'idle'
        self._last_target = None

        self._det_sub = self.create_subscription(
            String, '/ecobot/detections', self._det_cb, 10)
        self._joint_sub = self.create_subscription(
            Float64MultiArray, '/arm/joint_angles', self._joint_cb, 10)
        self._cmd_sub = self.create_subscription(
            String, '/arm/adaptive_target/cmd', self._cmd_cb, 10)

        self._pose_pub = self.create_publisher(
            Float64MultiArray, '/arm/pose_goal', 10)
        self._status_pub = self.create_publisher(
            String, '/arm/adaptive_target/status', 10)
        self._debug_pub = self.create_publisher(
            Float64MultiArray, '/arm/adaptive_target/debug', 10)

        self._timer = self.create_timer(1.0 / max(1.0, self._rate), self._tick)

        self.get_logger().info(
            f'Adaptive target tracker ready — classes={sorted(self._classes)} '
            f'standoff={self._standoff}m transform='
            f'({self._tx},{self._ty},{self._tz}) dry_run={self._dry_run}')

    # ── subscriptions ────────────────────────────────────────────

    def _det_cb(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self._detections = data
            elif isinstance(data, dict) and 'detections' in data:
                self._detections = data['detections']
        except Exception:
            pass

    def _joint_cb(self, msg):
        if len(msg.data) >= NUM_JOINTS:
            self._joint_angles = list(msg.data[:NUM_JOINTS])

    def _cmd_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        action = str(data.get('action', '')).strip().lower()
        if action == 'start':
            if data.get('class'):
                cls = str(data['class']).strip().lower()
                if self._classes and cls not in self._classes:
                    self.get_logger().warn(
                        f'Class "{cls}" not in target list '
                        f'{sorted(self._classes)}')
                self._target_class = cls
            self._tracking = True
            self._last_seen = None
            self._last_status = 'searching'
            self.get_logger().info(
                f'Tracking started for class "{self._target_class}"')
        elif action == 'stop':
            self._tracking = False
            self._last_status = 'idle'
            self.get_logger().info('Tracking stopped')
        elif action == 'set_class':
            cls = str(data.get('class', '')).strip().lower()
            if cls:
                self._target_class = cls
                self.get_logger().info(f'Target class set to "{cls}"')

    # ── transform helpers ────────────────────────────────────────

    def _cam_to_arm(self, xc, yc, zc):
        """Camera optical frame (x=right,y=down,z=forward) -> arm IK frame
        (x=forward,y=left,z=up), after yaw + translation."""
        # camera forward (+z) is arm forward (+x); camera right (+x) is arm
        # -y; camera up (-y) is arm +z.
        x = zc
        y = -xc
        z = -yc
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        rx = x * cos_y - y * sin_y
        ry = x * sin_y + y * cos_y
        return rx + self._tx, ry + self._ty, z + self._tz

    def _nearest_detection(self):
        best = None
        best_z = math.inf
        for d in self._detections:
            name = str(d.get('class_name') or d.get('class') or '').lower()
            if self._target_class and name != self._target_class:
                continue
            z = float(d.get('z') or 0.0)
            if z <= 0.0 or z > 5.0:
                continue
            if z < best_z:
                best_z = z
                best = d
        return best

    # ── main loop ────────────────────────────────────────────────

    def _tick(self):
        if not self._tracking:
            if self._last_status != 'idle':
                self._last_status = 'idle'
                self._publish_status('idle', None, 'Tracking off')
            return

        det = self._nearest_detection()
        now = time.monotonic()

        if det is None:
            if self._last_seen is not None and \
                    now - self._last_seen > self._lost_timeout:
                self._last_status = 'lost'
                self._last_target = None
                self._publish_status('lost', None, 'Target lost')
            elif self._last_seen is None:
                self._publish_status('searching', None, 'Waiting for target')
            return

        self._last_seen = now
        xc = float(det.get('x') or 0.0)
        yc = float(det.get('y') or 0.0)
        zc = float(det.get('z') or 0.0)

        ox, oy, oz = self._cam_to_arm(xc, yc, zc)

        # Waypoint on the base->object line, standoff meters in front of the
        # object, so the wrist (and its camera) points at it.
        r = math.hypot(ox, oy)
        if r < 1e-6:
            self._publish_status('error', (ox, oy, oz), 'Object at origin')
            return
        target = self._solve_target(ox, oy, oz)
        if target is None:
            self._last_status = 'unreachable'
            self._publish_status(
                'unreachable', (ox, oy, oz),
                f'No IK for object at ({ox:.2f},{oy:.2f},{oz:.2f})')
            return

        tx, ty, tz = target
        if self._dry_run:
            self._last_status = 'tracking'
            self._publish_status('tracking', (ox, oy, oz),
                                 f'DRY-RUN target ({tx:.2f},{ty:.2f},{tz:.2f})')
            self._publish_debug(ox, oy, oz, tx, ty, tz)
            return

        msg = Float64MultiArray()
        msg.data = [tx, ty, tz]
        self._pose_pub.publish(msg)
        self._last_status = 'tracking'
        self._last_target = (tx, ty, tz)
        self._publish_status('tracking', (ox, oy, oz),
                             f'Goal ({tx:.2f},{ty:.2f},{tz:.2f})')
        self._publish_debug(ox, oy, oz, tx, ty, tz)

    def _solve_target(self, ox, oy, oz):
        """Find a reachable waypoint standoff meters before the object,
        backing off along the base->object line if needed."""
        r = math.hypot(ox, oy)
        ux, uy = ox / r, oy / r

        # ik_limits keeps each pair ordered; a reversed joint maps its servo
        # minimum to the larger IK value, and an inverted range rejects
        # every pose.
        _lim = ik_limits()
        lo = [q[0] for q in _lim]
        hi = [q[1] for q in _lim]

        distance = max(0.0, r - self._standoff)
        max_back = self._standoff + self._sweep_max
        while distance >= 0.0 and (r - distance) <= max_back:
            tx, ty = ux * distance, uy * distance
            result = self._ik.inverse(
                tx, ty, oz,
                theta2_min=lo[1], theta2_max=hi[1],
                theta3_min=lo[2], theta3_max=hi[2],
                theta4_min=lo[3], theta4_max=hi[3],
                theta1_min=lo[0], theta1_max=hi[0],
            )
            if result is not None:
                return tx, ty, oz
            distance -= self._sweep_step
        return None

    def _publish_status(self, status, obj, detail):
        msg = String()
        msg.data = json.dumps({
            'status': status,
            'class': self._target_class,
            'tracking': self._tracking,
            'object': [round(float(v), 3) for v in obj] if obj else None,
            'detail': detail,
        })
        self._status_pub.publish(msg)

    def _publish_debug(self, ox, oy, oz, tx, ty, tz):
        msg = Float64MultiArray()
        msg.data = [
            ox, oy, oz,
            tx, ty, tz,
        ]
        self._debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmTargetTracker()
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
