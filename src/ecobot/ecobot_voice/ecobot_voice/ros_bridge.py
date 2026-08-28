"""ROS 2 <-> asyncio bridge for the LiveKit voice agent.

livekit-agents runs its own asyncio event loop; rclpy needs its executor
spinning continuously to service callbacks. Rather than interleave the two,
this runs a single rclpy node on a dedicated background thread and exposes
plain, thread-safe methods the agent's async tool functions can call.
Publishing from a non-executor thread is safe in rclpy (publish() does not
touch executor state); reads of subscription state go through a lock since
the executor thread writes them concurrently.
"""
import json
import math
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge
import cv2

from ecobot_arm_control.arm_kinematics import ArmKinematics
from ecobot_arm_control.servo_config import JOINTS, NUM_JOINTS, to_servo, to_ik, within_limits


def _twist(linear: float, angular: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(linear)
    msg.angular.z = float(angular)
    return msg


class _BridgeNode(Node):
    def __init__(self, wrist_camera_topic: str):
        super().__init__('ecobot_voice_bridge')
        self._lock = threading.Lock()

        self._arm_status = 'unknown'
        self._arm_pose = None
        self._joint_angles = None
        self._scanner_status = None
        self._detections = []
        self._tof_ranges = None
        self._odom = None
        self._wrist_frame = None
        self._wrist_frame_stamp = 0.0
        self._nav_status = {'state': 'idle'}

        self._bridge = CvBridge()

        self._pose_goal_pub = self.create_publisher(
            Float64MultiArray, '/arm/pose_goal', 10)
        self._joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/arm/joint_commands', 10)
        self._enable_pub = self.create_publisher(String, '/arm/enable', 10)
        self._scanner_cmd_pub = self.create_publisher(
            String, '/arm/scanner_cmd', 10)
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._nav_goal_handle = None

        self.create_subscription(
            String, '/arm/status', self._on_arm_status, 10)
        self.create_subscription(
            Float64MultiArray, '/arm/pose', self._on_arm_pose, 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)
        self.create_subscription(
            String, '/arm/scanner_status', self._on_scanner_status, 10)
        self.create_subscription(
            String, '/ecobot/detections', self._on_detections, 10)
        self.create_subscription(
            String, '/ecobot/tof_ranges', self._on_tof, 10)
        self.create_subscription(
            Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(
            Image, wrist_camera_topic, self._on_wrist_frame, 10)

    def _on_arm_status(self, msg):
        with self._lock:
            self._arm_status = msg.data

    def _on_arm_pose(self, msg):
        with self._lock:
            self._arm_pose = list(msg.data)

    def _on_joint_states(self, msg):
        with self._lock:
            self._joint_angles = list(msg.position)

    def _on_scanner_status(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._scanner_status = data

    def _on_detections(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        detections = data if isinstance(data, list) else data.get('detections', [])
        with self._lock:
            self._detections = detections

    def _on_tof(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._tof_ranges = data.get('ranges_m')

    def _on_odom(self, msg):
        with self._lock:
            self._odom = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'linear_x': msg.twist.twist.linear.x,
                'angular_z': msg.twist.twist.angular.z,
            }

    def _on_wrist_frame(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'wrist frame decode error: {e}')
            return
        with self._lock:
            self._wrist_frame = frame
            self._wrist_frame_stamp = time.time()

    def _on_nav_goal_response(self, future):
        goal_handle = future.result()
        with self._lock:
            self._nav_goal_handle = goal_handle
            if not goal_handle.accepted:
                self._nav_status = {'state': 'rejected'}
                return
            self._nav_status = {'state': 'navigating'}
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        result = future.result()
        # NavigateToPose's action_msgs/GoalStatus: 4=SUCCEEDED, 5=CANCELED,
        # 6=ABORTED (see action_msgs/msg/GoalStatus.msg) — mirrors
        # send_goal.py's status==4 check.
        state = 'succeeded' if result.status == 4 else f'failed(status={result.status})'
        with self._lock:
            self._nav_status = {'state': state}
            self._nav_goal_handle = None


class RosBridge:
    """Public, thread-safe surface used by the agent's function tools."""

    def __init__(self, wrist_camera_topic='/arm/camera/image_raw', ros_args=None):
        if not rclpy.ok():
            rclpy.init(args=ros_args)
        self._node = _BridgeNode(wrist_camera_topic)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

        l0 = self._node.declare_parameter('l0', 0.110).value
        l1 = self._node.declare_parameter('l1', 0.150).value
        l2 = self._node.declare_parameter('l2', 0.130).value
        l3 = self._node.declare_parameter('l3', 0.140).value
        self._ik = ArmKinematics(l0, l1, l2, l3)

    def shutdown(self):
        # Join before destroying the node/context — an unjoined spin thread
        # still touching DDS objects while the interpreter tears them down
        # aborts the process (observed: "terminate called without an active
        # exception" on interpreter exit).
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # ---- arm control -----------------------------------------------

    def move_arm_to(self, x: float, y: float, z: float):
        """Solve IK locally (mirrors arm_manual_node) so the agent gets an
        immediate, accurate reachable/unreachable answer instead of firing
        blind and polling for a status change that may never come."""
        lo = to_ik([JOINTS[i]['min_angle'] for i in range(NUM_JOINTS)])
        hi = to_ik([JOINTS[i]['max_angle'] for i in range(NUM_JOINTS)])
        result = self._ik.inverse(
            x, y, z,
            theta2_min=lo[1], theta2_max=hi[1],
            theta3_min=lo[2], theta3_max=hi[2],
            theta4_min=lo[3], theta4_max=hi[3],
        )
        if result is None:
            return False, f'({x:.2f}, {y:.2f}, {z:.2f}) is not reachable by the arm'

        angles = to_servo(result)
        if not within_limits(angles):
            return False, (
                f'({x:.2f}, {y:.2f}, {z:.2f}) needs a base angle outside the '
                'arm\'s allowed range')

        msg = Float64MultiArray()
        msg.data = [float(x), float(y), float(z)]
        self._node._pose_goal_pub.publish(msg)
        return True, f'moving to ({x:.2f}, {y:.2f}, {z:.2f})'

    def home_arm(self):
        msg = Float64MultiArray()
        msg.data = [float(j['home_angle']) for j in JOINTS]
        self._node._joint_cmd_pub.publish(msg)

    def set_arm_enabled(self, enabled: bool):
        msg = String()
        msg.data = 'enable' if enabled else 'disable'
        self._node._enable_pub.publish(msg)

    def start_scan(self, x: float, y: float, z: float):
        msg = String()
        msg.data = json.dumps({'action': 'scan', 'x': x, 'y': y, 'z': z})
        self._node._scanner_cmd_pub.publish(msg)

    def stop_scan(self):
        msg = String()
        msg.data = json.dumps({'action': 'stop'})
        self._node._scanner_cmd_pub.publish(msg)

    # ---- base movement / navigation -----------------------------------

    def set_cmd_vel(self, linear_x: float, angular_z: float):
        """Publish one /cmd_vel sample. Called at the sender's own rate (e.g.
        from a continuous teleop/action feed) — motor_control_node has its
        own cmd_vel_timeout (0.5s, see motor_control_node.py) that auto-stops
        the base if publishing stops, so no separate watchdog is needed here."""
        self._node._cmd_vel_pub.publish(_twist(float(linear_x), float(angular_z)))

    def navigate_to(self, x: float, y: float, yaw: float):
        """Send a Nav2 NavigateToPose goal (mirrors ecobot_bringup/send_goal.py).
        Fire-and-forget: returns once the goal is sent, not once it completes
        — poll get_nav_status() for progress."""
        if not self._node._nav_client.server_is_ready():
            return False, 'Nav2 action server is not available'

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        send_future = self._node._nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._node._on_nav_goal_response)
        return True, f'navigating to ({x:.2f}, {y:.2f}, yaw={yaw:.2f})'

    def get_nav_status(self):
        with self._node._lock:
            return dict(self._node._nav_status)

    def move(self, distance: float, linear: float = 0.3, angular: float = 0.0):
        """Drive the base a bounded distance, via a timed /cmd_vel burst on a
        background thread.

        Drive straight: `distance` metres forward (+) or backward (-) at
        `linear` m/s (clamped to 0.5). Rotate in place: pass `angular`
        (rad/s, clamped to 1.0) and `distance` is ignored (default 0).

        Works without a Nav2 goal. motor_control_node auto-stops 0.5s after
        the last sample; the thread also publishes an explicit zero at the
        end for a clean stop.
        """
        distance = float(distance)
        linear = max(0.0, min(abs(float(linear)), 0.5))
        angular = max(-1.0, min(float(angular), 1.0))

        if angular != 0.0:
            # Rotate in place for a fixed 0.4s at the requested rate.
            lin, ang, duration = 0.0, angular, 0.4
            ge = f'turning {angular:+.2f} rad/s'
        else:
            if distance == 0.0:
                return False, 'distance must be nonzero'
            lin = linear * (1.0 if distance > 0 else -1.0)
            ang = 0.0
            duration = abs(distance) / linear if linear > 0 else 0.0
            ge = f'moving {abs(distance):.2f}m'
        duration = min(max(duration, 0.2), 10.0)

        def _burst():
            step = 0.1
            ticks = max(1, int(duration / step))
            for _ in range(ticks):
                self._node._cmd_vel_pub.publish(_twist(lin, ang))
                time.sleep(step)
            self._node._cmd_vel_pub.publish(_twist(0.0, 0.0))

        threading.Thread(target=_burst, daemon=True).start()
        return True, f'{ge} for {duration:.2f}s (lin={lin:.2f}, ang={ang:.2f})'

    def emergency_stop(self):
        """Immediately zero the base velocity and cancel any active nav
        goal or scan. Published a few times, not once — a single Twist can
        be dropped or superseded by an in-flight continuous cmd_vel stream
        (e.g. the action-driven teleop path) racing on the same topic."""
        msg = Twist()
        for _ in range(5):
            self._node._cmd_vel_pub.publish(msg)
            time.sleep(0.05)
        if self._node._nav_goal_handle is not None:
            self._node._nav_goal_handle.cancel_goal_async()
        self.stop_scan()

    # ---- state / vision ----------------------------------------------

    def get_arm_status(self):
        with self._node._lock:
            return {
                'status': self._node._arm_status,
                'pose': self._node._arm_pose,
                'joint_angles': self._node._joint_angles,
            }

    def get_scanner_status(self):
        with self._node._lock:
            return self._node._scanner_status

    def get_latest_detections(self):
        with self._node._lock:
            return list(self._node._detections)

    def get_tof_ranges(self):
        with self._node._lock:
            return self._node._tof_ranges

    def get_odom(self):
        with self._node._lock:
            return self._node._odom

    def get_wrist_frame_jpeg(self, quality=80, max_age_s=2.0):
        """Latest wrist-camera frame as JPEG bytes, or None if stale/absent."""
        with self._node._lock:
            frame = self._node._wrist_frame
            stamp = self._node._wrist_frame_stamp
        if frame is None or (time.time() - stamp) > max_age_s:
            return None
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return buf.tobytes()

    def get_wrist_frame_rgb(self, max_age_s=2.0):
        """Latest wrist-camera frame as an RGB numpy array (for Portal's
        send_video_frame, which expects RGB, not cv2's native BGR), or None
        if stale/absent."""
        with self._node._lock:
            frame = self._node._wrist_frame
            stamp = self._node._wrist_frame_stamp
        if frame is None or (time.time() - stamp) > max_age_s:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
