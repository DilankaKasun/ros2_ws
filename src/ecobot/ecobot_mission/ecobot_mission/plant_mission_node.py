"""Autonomous per-plant mission orchestrator.

For each waypoint: navigate to a standoff pose facing the plant, settle,
trigger arm_scanner_node's existing multi-viewpoint scan, capture a
wrist-camera JPEG at each viewpoint, send them to Gemini for a health
assessment, then advance to the next waypoint (or wait for an operator
"next" command, depending on auto_advance).

Speaks the dashboard's /ecobot/plant_scan_cmd -> /ecobot/plant_scan_status /
/ecobot/scan_capture contract, consumed by the ecobot-ui dashboard over
rosbridge — no frontend changes needed. Talks to
arm_scanner_node purely over /arm/scanner_cmd + /arm/scanner_status; it is
never modified or imported.
"""
import functools
import json
import math
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from tf2_ros import (
    Buffer, ConnectivityException, ExtrapolationException, LookupException,
    TransformException, TransformListener,
)

from .gemini_client import GeminiClient

# Statuses a mission is actively running in — used to reject start/
# set_waypoints commands that would otherwise clobber an in-progress run.
_ACTIVE_STATUSES = {'NAVIGATING', 'SCANNING', 'ANALYZING', 'WAITING',
                    'SEARCHING'}


class PlantMissionNode(Node):
    def __init__(self):
        super().__init__('plant_mission_node')

        self.declare_parameter('plant_scan_cmd_topic', '/ecobot/plant_scan_cmd')
        self.declare_parameter('plant_scan_status_topic', '/ecobot/plant_scan_status')
        self.declare_parameter('scan_capture_topic', '/ecobot/scan_capture')
        self.declare_parameter('scanner_cmd_topic', '/arm/scanner_cmd')
        self.declare_parameter('scanner_status_topic', '/arm/scanner_status')
        self.declare_parameter('wrist_camera_topic', '/arm/camera/image_raw')
        self.declare_parameter('fallback_waypoints_topic', '/ecobot/waypoints')
        self.declare_parameter('nav_action_name', '/navigate_to_pose')

        self.declare_parameter('auto_advance', True)
        self.declare_parameter('default_frame', 'map')

        self.declare_parameter('approach_standoff_m', 0.4)
        # How near the plant the obstacle-avoidance layer is asked to stand
        # down. Deliberately small: the drive between plants keeps full
        # avoidance, only the last stretch up to the plant does not.
        self.declare_parameter('suppress_avoidance_radius_m', 1.2)
        self.declare_parameter('mission_suppress_topic',
                               '/ecobot/mission_suppress_avoidance')
        self.declare_parameter('goal_wait_timeout_s', 90.0)
        self.declare_parameter('nav_server_wait_s', 5.0)
        self.declare_parameter('nav_settle_s', 0.5)
        self.declare_parameter(
            'fallback_target_classes',
            ['potted plant', 'plant', 'pot', 'crop'])

        # Where to aim when scanning in place with no detected plant pose.
        # These sit inside the arm's workspace: the shoulder pivot is 38cm up
        # with a 35cm reach, so the old 0.18 target was unreachable and every
        # sampled viewpoint got filtered out, leaving a scan that never ran.
        self.declare_parameter('arm_scan_x', 0.30)
        self.declare_parameter('arm_scan_y', 0.0)
        self.declare_parameter('arm_scan_z', 0.50)
        # Photos per plant. The dashboard can change this between runs by
        # putting a 'samples' field on any /ecobot/plant_scan_cmd message.
        self.declare_parameter('scan_samples', 6)

        self.declare_parameter('capture_delay_s', 1.4)
        self.declare_parameter('frame_max_age_s', 2.0)
        self.declare_parameter('scan_settle_timeout_s', 20.0)

        self.declare_parameter('gemini_model', '')
        self.declare_parameter('gemini_timeout_s', 20.0)
        self.declare_parameter('gemini_max_retries', 1)
        self.declare_parameter('jpeg_quality', 85)

        gp = self.get_parameter
        self._auto_advance = bool(gp('auto_advance').value)
        self._default_frame = str(gp('default_frame').value)
        self._approach_standoff_m = float(gp('approach_standoff_m').value)
        self._suppress_radius_m = float(
            gp('suppress_avoidance_radius_m').value)
        self._nav_server_wait_s = float(gp('nav_server_wait_s').value)
        self._nav_settle_s = float(gp('nav_settle_s').value)
        self._fallback_target_classes = {
            str(c).strip().lower()
            for c in gp('fallback_target_classes').value}
        self._arm_scan_x = float(gp('arm_scan_x').value)
        self._arm_scan_y = float(gp('arm_scan_y').value)
        self._arm_scan_z = float(gp('arm_scan_z').value)
        self._scan_samples = int(gp('scan_samples').value)
        self._paused_from = None
        self._capture_delay_s = float(gp('capture_delay_s').value)
        self._frame_max_age_s = float(gp('frame_max_age_s').value)
        self._scan_settle_timeout_s = float(gp('scan_settle_timeout_s').value)
        self._jpeg_quality = int(gp('jpeg_quality').value)

        # -- mission state --
        self._waypoints = []
        self._idx = -1
        self._status = 'IDLE'
        self._error_msg = ''
        self._results = []
        self._current_result = None
        self._current_captures = []
        self._fallback_waypoints = []
        self._epoch = 0
        self._camera_path_paused = False

        self._nav_goal_handle = None
        self._nav_settle_due_time = None

        self._last_scanner_status = None
        self._scan_start_time = None
        self._capture_due_time = None
        self._pending_capture_label = None

        self._result_lock = threading.Lock()
        self._pending_gemini_result = None

        self._wrist_lock = threading.Lock()
        self._wrist_frame = None
        self._wrist_frame_stamp = 0.0
        self._bridge = CvBridge()

        try:
            self._gemini = GeminiClient(
                model=(gp('gemini_model').value or None),
                timeout_s=float(gp('gemini_timeout_s').value),
                max_retries=int(gp('gemini_max_retries').value))
        except RuntimeError as e:
            self.get_logger().warning(
                f'Gemini disabled: {e} — missions will still navigate/scan/'
                'capture, but plant-health results will be degraded')
            self._gemini = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._nav_client = ActionClient(
            self, NavigateToPose, str(gp('nav_action_name').value))

        self._scanner_cmd_pub = self.create_publisher(
            String, str(gp('scanner_cmd_topic').value), 10)
        self._status_pub = self.create_publisher(
            String, str(gp('plant_scan_status_topic').value), 10)
        # Hands the search-and-approach back to detection_goto, which owns
        # the drive toward whatever plant it finds.
        self._goto_cmd_pub = self.create_publisher(
            String, '/ecobot/goto_target', 10)
        # Asks obstacle_avoidance to stand down near a plant. Without it a
        # Nav2 approach never arrives: that layer sees the plant itself as
        # an obstacle at 0.9m and turns the robot away from the 0.4m
        # standoff it was sent to, and can reverse and spin on top of that.
        self._suppress_pub = self.create_publisher(
            Bool, str(gp('mission_suppress_topic').value), 10)

        self._scan_capture_pub = self.create_publisher(
            String, str(gp('scan_capture_topic').value), 10)

        self.create_subscription(
            String, str(gp('plant_scan_cmd_topic').value), self._on_cmd, 10)
        self.create_subscription(
            String, str(gp('scanner_status_topic').value),
            self._on_scanner_status, 10)
        self.create_subscription(
            Image, str(gp('wrist_camera_topic').value),
            self._on_wrist_frame, 10)

        # Must match detection_goto.py's publisher QoS (TRANSIENT_LOCAL) or
        # this subscription never connects / never receives the latched
        # last value.
        fallback_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(
            String, str(gp('fallback_waypoints_topic').value),
            self._on_fallback_waypoints, fallback_qos)

        self.create_timer(0.1, self._tick)

        self.get_logger().info('plant mission node ready')

    # ---- command dispatch --------------------------------------------

    def _on_cmd(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        action = data.get('action')

        # Any command may carry the shot count for the next scan.
        if data.get('samples') is not None:
            try:
                self._scan_samples = max(1, min(int(data['samples']), 40))
                self.get_logger().info(
                    f'scan sample count set to {self._scan_samples}')
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f'ignoring bad samples value: {data["samples"]!r}')

        if action == 'set_samples':
            self._publish_status()
            return
        if action == 'start':
            self._handle_start(data.get('waypoints', []))
        elif action == 'next':
            if self._status == 'WAITING':
                self._advance_and_navigate()
        elif action == 'stop':
            self._handle_stop()
        elif action == 'pause':
            self._handle_pause()
        elif action == 'resume':
            self._handle_resume()
        elif action == 'set_waypoints':
            self._handle_set_waypoints(data.get('waypoints', []))
        elif action == 'scan_here':
            self._handle_scan_here()

    def _handle_scan_here(self):
        """Scan in place — no navigation. For a caller (e.g.
        detection_goto.py's auto-tracked approach) that already
        positioned the robot itself and just wants this node's existing
        capture+Gemini+status pipeline, without a redundant Nav2 goal
        that could nudge the robot off the position it already found."""
        if self._status in _ACTIVE_STATUSES:
            self.get_logger().warning(
                f'ignoring scan_here command: mission already {self._status}')
            return
        self._waypoints = [{'x': 0.0, 'y': 0.0, 'frame': None}]
        self._results = []
        self._idx = 0
        self._error_msg = ''
        self._epoch += 1
        self._current_result = {
            'wp_idx': 0, 'captures': 0, 'nav_status': 'skipped(already in place)',
            'scan_status': None, 'health': None, 'confidence': None,
            'notes': None, 'timestamp': None,
        }
        self._start_arm_scan()

    def _handle_start(self, waypoints):
        if self._status in _ACTIVE_STATUSES:
            self.get_logger().warning(
                f'ignoring start command: mission already {self._status}')
            return
        wps = self._normalize_waypoints(waypoints)
        if not wps:
            wps = list(self._fallback_waypoints)
        if not wps:
            # No route given is the normal case for this robot: it is meant to
            # look around, drive to whatever plant it finds, and scan it.
            # Reporting ERROR here made the ordinary way of starting a mission
            # look broken. Hand off to detection_goto's search-and-approach
            # instead, which calls back with scan_here once it arrives.
            self._results = []
            self._idx = 0
            self._waypoints = [{'x': 0.0, 'y': 0.0, 'frame': None}]
            self._error_msg = ''
            self._epoch += 1
            self._goto_cmd_pub.publish(String(data=json.dumps(
                {'action': 'resume_auto_track'})))
            self._status = 'SEARCHING'
            self.get_logger().info(
                'no waypoints given — searching for a plant to scan')
            self._publish_status()
            return
        self._waypoints = wps
        self._results = []
        self._idx = -1
        self._error_msg = ''
        self._epoch += 1
        self._pause_camera_path()
        self._advance_and_navigate()

    def _handle_set_waypoints(self, waypoints):
        if self._status in _ACTIVE_STATUSES:
            self.get_logger().warning(
                f'ignoring set_waypoints command: mission already {self._status}')
            return
        self._waypoints = self._normalize_waypoints(waypoints)
        self._results = []
        self._idx = -1
        self._error_msg = ''
        self._status = 'WAYPOINTS_LOADED'
        self._publish_status()

    def _handle_stop(self):
        self._epoch += 1
        if self._nav_goal_handle is not None:
            self._nav_goal_handle.cancel_goal_async()
            self._nav_goal_handle = None
        self._scanner_cmd_pub.publish(String(data=json.dumps({'action': 'stop'})))
        self._nav_settle_due_time = None
        self._capture_due_time = None
        self._pending_capture_label = None
        self._scan_start_time = None
        self._current_captures = []
        self._current_result = None
        self._resume_camera_path()
        self._status = 'STOPPED'
        self._publish_status()

    def _handle_pause(self):
        """Halt the arm where it is and hold, keeping what has been captured.

        Unlike stop, the captures and the waypoint position survive, so
        resume picks the plant back up rather than starting the run over.
        """
        if self._status not in _ACTIVE_STATUSES:
            self.get_logger().warning(
                f'ignoring pause: mission is {self._status}')
            return
        self._paused_from = self._status
        # Bump the epoch so any scan callback still in flight is discarded.
        self._epoch += 1
        self._scanner_cmd_pub.publish(String(data=json.dumps({'action': 'stop'})))
        self._nav_settle_due_time = None
        self._capture_due_time = None
        self._pending_capture_label = None
        self._status = 'PAUSED'
        self.get_logger().info(
            f'paused ({len(self._current_captures)} captures held)')
        self._publish_status()

    def _handle_resume(self):
        """Carry on from a pause by rescanning the plant we were on."""
        if self._status != 'PAUSED':
            self.get_logger().warning(
                f'ignoring resume: mission is {self._status}')
            return
        self.get_logger().info('resuming — rescanning the current plant')
        self._start_arm_scan()

    def _pause_camera_path(self):
        """Stop detection_goto driving for the length of a waypoint run.

        Nav2 owns the wheels during one, but detection_goto is otherwise
        free to keep hunting plants, and it takes the wheels back one
        second after Nav2 falls silent — which is exactly when this node
        settles and starts the arm scan. The robot span while it was
        photographing. It keeps publishing a zero cmd_vel while paused, so
        the safety layer still sees a live command source.
        """
        self._goto_cmd_pub.publish(String(data=json.dumps(
            {'action': 'pause_auto_track'})))
        self._camera_path_paused = True

    def _resume_camera_path(self):
        """Hand the search-and-approach back once the waypoint run ends."""
        if not self._camera_path_paused:
            return
        self._camera_path_paused = False
        self._goto_cmd_pub.publish(String(data=json.dumps(
            {'action': 'resume_auto_track'})))

    def _distance_to_plant(self):
        """Metres from the robot to the plant it is currently working on,
        or None when the transform is not available."""
        if not 0 <= self._idx < len(self._waypoints):
            return None
        wp = self._waypoints[self._idx]
        frame = wp.get('frame') or self._default_frame
        try:
            # Non-blocking: this runs at 10Hz, so waiting on a missing
            # transform here would eat the whole tick period.
            tf = self._tf_buffer.lookup_transform(
                frame, 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.0))
        except TransformException:
            # Catch the base class, not the usual three: this runs inside
            # the tick, and a bad frame on a caller-supplied waypoint
            # raises InvalidArgumentException, which would escape and stop
            # the timer for good.
            return None
        return math.hypot(wp['x'] - tf.transform.translation.x,
                          wp['y'] - tf.transform.translation.y)

    def _suppress_avoidance_now(self):
        """Whether the robot is deliberately close to a plant right now.

        True for the last stretch of an approach and for the whole scan;
        false for the drive between plants, which keeps full avoidance.
        """
        if self._status in ('SCANNING', 'ANALYZING'):
            return True
        if self._status != 'NAVIGATING':
            return False
        dist = self._distance_to_plant()
        # No transform means no idea how close we are — leave avoidance on.
        return dist is not None and dist <= self._suppress_radius_m

    def _normalize_waypoints(self, waypoints):
        out = []
        for wp in waypoints or []:
            if not isinstance(wp, dict) or 'x' not in wp or 'y' not in wp:
                continue
            try:
                out.append({
                    'x': float(wp['x']), 'y': float(wp['y']),
                    'frame': wp.get('frame'),
                })
            except (TypeError, ValueError):
                continue
        return out

    def _on_fallback_waypoints(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        # /ecobot/waypoints (detection_goto.py) carries whatever object
        # class was tracked when the waypoint was saved — filter to
        # plant-like classes only, so a chair/table/etc. picked up by the
        # same detector is never mistaken for a crop to approach.
        if self._fallback_target_classes:
            data = [
                wp for wp in (data or [])
                if isinstance(wp, dict)
                and str(wp.get('class_name', '')).strip().lower()
                in self._fallback_target_classes
            ]
        self._fallback_waypoints = self._normalize_waypoints(data)

    # ---- mission progression ------------------------------------------

    def _advance_and_navigate(self):
        self._idx += 1
        self._current_result = {
            'wp_idx': self._idx, 'captures': 0, 'nav_status': None,
            'scan_status': None, 'health': None, 'confidence': None,
            'notes': None, 'timestamp': None,
        }
        self._status = 'NAVIGATING'
        self._publish_status()
        self._send_nav_goal(self._idx)

    def _finish_plant(self):
        if self._current_result is not None:
            self._current_result['timestamp'] = time.time()
            self._results.append(self._current_result)
            self._current_result = None
        self._current_captures = []

        if self._idx >= len(self._waypoints) - 1:
            self._resume_camera_path()
            self._status = 'COMPLETE'
            self._publish_status()
            return
        if self._auto_advance:
            self._advance_and_navigate()
        else:
            self._status = 'WAITING'
            self._publish_status()

    # ---- navigation -----------------------------------------------------

    def _send_nav_goal(self, idx):
        wp = self._waypoints[idx]
        frame = wp.get('frame') or self._default_frame

        pose = self._compute_standoff_pose(wp['x'], wp['y'], frame)
        if pose is None:
            if self._current_result is not None:
                self._current_result['nav_status'] = 'tf_lookup_failed'
            self._finish_plant()
            return
        gx, gy, gyaw = pose

        if not self._nav_client.wait_for_server(timeout_sec=self._nav_server_wait_s):
            self.get_logger().error('nav2 action server not available')
            self._resume_camera_path()
            self._status = 'ERROR'
            self._error_msg = 'nav2 action server not available'
            self._publish_status()
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.orientation.z = math.sin(gyaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(gyaw / 2.0)

        epoch = self._epoch
        self.get_logger().info(
            f'plant {idx}: navigating to standoff ({gx:.2f}, {gy:.2f}, '
            f'yaw={gyaw:.2f}) in frame "{frame}"')
        send_future = self._nav_client.send_goal_async(goal)
        send_future.add_done_callback(
            functools.partial(self._on_nav_goal_response, epoch=epoch, idx=idx))

    def _on_nav_goal_response(self, future, epoch, idx):
        if epoch != self._epoch:
            return
        goal_handle = future.result()
        if not goal_handle.accepted:
            if self._current_result is not None:
                self._current_result['nav_status'] = 'rejected'
            self._finish_plant()
            return
        self._nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            functools.partial(self._on_nav_result, epoch=epoch, idx=idx))

    def _on_nav_result(self, future, epoch, idx):
        if epoch != self._epoch:
            return
        self._nav_goal_handle = None
        result = future.result()
        # action_msgs/msg/GoalStatus SUCCEEDED == 4 — same convention as
        # ecobot_bringup/send_goal.py:46 and ros_bridge.py's nav handling.
        if result.status == 4:
            if self._current_result is not None:
                self._current_result['nav_status'] = 'ok'
            self._nav_settle_due_time = (
                self.get_clock().now() + Duration(seconds=self._nav_settle_s))
        else:
            if self._current_result is not None:
                self._current_result['nav_status'] = f'nav_failed(status={result.status})'
            self._finish_plant()

    def _compute_standoff_pose(self, px, py, frame):
        try:
            tf = self._tf_buffer.lookup_transform(
                frame, 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warning(f'tf lookup {frame}->base_footprint failed: {e}')
            return None

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        q = tf.transform.rotation
        ryaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))

        dx = px - rx
        dy = py - ry
        r = math.hypot(dx, dy)
        d = self._approach_standoff_m

        if r < 1e-3:
            # Robot is already essentially on top of the plant position —
            # no well-defined approach direction; keep current heading and
            # target the plant position itself.
            ux, uy = math.cos(ryaw), math.sin(ryaw)
        else:
            ux, uy = dx / r, dy / r

        gx = px - d * ux
        gy = py - d * uy
        gyaw = math.atan2(uy, ux)
        return gx, gy, gyaw

    # ---- arm scan ---------------------------------------------------------

    def _start_arm_scan(self):
        self._current_captures = []
        self._scan_start_time = self.get_clock().now()
        self._last_scanner_status = None
        # Defensive reset — arm_scanner_node can also auto-start a scan
        # from /ecobot/detections; this guards against one still running.
        self._scanner_cmd_pub.publish(String(data=json.dumps({'action': 'stop'})))
        self._scanner_cmd_pub.publish(String(data=json.dumps({
            'action': 'scan', 'x': self._arm_scan_x,
            'y': self._arm_scan_y, 'z': self._arm_scan_z,
            'samples': self._scan_samples})))
        self._status = 'SCANNING'
        self._publish_status()

    def _on_scanner_status(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        prev = self._last_scanner_status
        self._last_scanner_status = data
        if self._status != 'SCANNING':
            return

        if data.get('status') == 'scanning':
            # Only photograph the sampled viewpoints. Older scanners did not
            # send this flag, so treat its absence as "capture", keeping the
            # previous behaviour rather than silently taking no photos.
            wants_capture = data.get('capture', True)
            if wants_capture and (
                    prev is None or prev.get('status') != 'scanning'
                    or prev.get('viewpoint') != data.get('viewpoint')):
                self._pending_capture_label = data.get('current_label', '')
                self._capture_due_time = (
                    self.get_clock().now() + Duration(seconds=self._capture_delay_s))
        elif data.get('status') in ('idle', 'recovering') and prev is not None \
                and prev.get('status') in ('scanning', 'recovering'):
            self._on_scan_complete()

    def _on_scan_complete(self):
        self._scan_start_time = None
        self._capture_due_time = None
        if self._current_result is not None:
            self._current_result['captures'] = len(self._current_captures)
            self._current_result['scan_status'] = (
                'ok' if self._current_captures else 'no_captures')
            if self._last_scanner_status and 'parts_covered' in self._last_scanner_status:
                self._current_result['parts_covered'] = self._last_scanner_status['parts_covered']

        if self._gemini is None:
            if self._current_result is not None:
                self._current_result['health'] = 'unknown'
                self._current_result['notes'] = 'GOOGLE_API_KEY not set'
            self._finish_plant()
            return

        self._spawn_gemini_call()
        self._status = 'ANALYZING'
        self._publish_status()

    def _spawn_gemini_call(self):
        captures = list(self._current_captures)
        epoch = self._epoch
        idx = self._idx

        def _worker():
            labels = [label for label, _ in captures]
            images = [jpg for _, jpg in captures]
            # The Live streaming API only serves *-live-preview models; the
            # report runs on a pro model, so use the standard call.
            result = self._gemini.assess_plant(images, labels=labels)
            with self._result_lock:
                self._pending_gemini_result = (epoch, idx, result)

        threading.Thread(target=_worker, daemon=True).start()

    # ---- wrist camera capture -------------------------------------------

    def _on_wrist_frame(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f'wrist frame decode error: {e}')
            return
        with self._wrist_lock:
            self._wrist_frame = frame
            self._wrist_frame_stamp = time.time()

    def _get_wrist_jpeg(self):
        with self._wrist_lock:
            frame = self._wrist_frame
            stamp = self._wrist_frame_stamp
        if frame is None or (time.time() - stamp) > self._frame_max_age_s:
            return None
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return buf.tobytes() if ok else None

    def _do_capture(self, label):
        jpeg = self._get_wrist_jpeg()
        if jpeg is None:
            self.get_logger().warning(
                f'no fresh wrist frame for viewpoint "{label}", skipping capture')
            return
        self._current_captures.append((label, jpeg))
        self._publish_scan_capture(label, jpeg)

    # ---- publishing --------------------------------------------------------

    def _publish_status(self):
        payload = {
            'status': self._status,
            'idx': self._idx,
            'total': len(self._waypoints),
            'results': self._results,
            'waypoints': [{'x': wp['x'], 'y': wp['y']} for wp in self._waypoints],
            'samples': self._scan_samples,
            'captures': len(self._current_captures),
        }
        if self._error_msg:
            payload['error'] = self._error_msg
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _publish_scan_capture(self, label, jpeg_bytes):
        wp = self._waypoints[self._idx] if 0 <= self._idx < len(self._waypoints) else None
        payload = {
            # The dashboard decodes this with bytes.fromhex(...) —
            # this MUST be a hex string, not base64.
            'image_jpeg': jpeg_bytes.hex(),
            'capture_count': len(self._current_captures),
            # Repurposed: no detected-object class applies here, so this
            # carries the scan viewpoint label (front/right/left/top).
            'class': label,
        }
        if wp is not None:
            payload['pose'] = {'x': wp['x'], 'y': wp['y']}
        self._scan_capture_pub.publish(String(data=json.dumps(payload)))

    # ---- periodic tick ------------------------------------------------------

    def _tick(self):
        now = self.get_clock().now()
        self._publish_status()
        self._suppress_pub.publish(Bool(data=self._suppress_avoidance_now()))

        if self._nav_settle_due_time is not None and now >= self._nav_settle_due_time:
            self._nav_settle_due_time = None
            self._start_arm_scan()

        if self._capture_due_time is not None and now >= self._capture_due_time:
            label = self._pending_capture_label
            self._capture_due_time = None
            self._pending_capture_label = None
            self._do_capture(label)

        if self._status == 'SCANNING' and self._scan_start_time is not None:
            if (now - self._scan_start_time) > Duration(seconds=self._scan_settle_timeout_s):
                self.get_logger().warning('scan settle timeout, abandoning this plant')
                self._scan_start_time = None
                if self._current_result is not None:
                    self._current_result['scan_status'] = 'timeout'
                self._finish_plant()

        with self._result_lock:
            pending = self._pending_gemini_result
            self._pending_gemini_result = None
        if pending is not None:
            epoch, idx, result = pending
            if epoch == self._epoch and self._status == 'ANALYZING':
                if self._current_result is not None:
                    self._current_result['health'] = result['health']
                    self._current_result['confidence'] = result['confidence']
                    self._current_result['notes'] = result['notes']
                    if not self._current_result.get('scan_status'):
                        self._current_result['scan_status'] = (
                            'gemini_error' if result.get('error') else 'ok')
                self._finish_plant()


def main(args=None):
    rclpy.init(args=args)
    node = PlantMissionNode()
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
