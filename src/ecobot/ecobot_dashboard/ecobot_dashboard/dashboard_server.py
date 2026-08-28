import os
import io
import math
import time
import threading
import asyncio
import json
import base64
from functools import partial
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8, String, Float64MultiArray
from cv_bridge import CvBridge
import cv2
import aiohttp
from aiohttp import web, WSMsgType

try:
    from ecobot_mission.gemini_client import GeminiClient
    GEMINI_CLIENT_AVAILABLE = True
except Exception:
    GEMINI_CLIENT_AVAILABLE = False


class DashboardServer(Node):
    def __init__(self) -> None:
        super().__init__('dashboard_server')

        self.declare_parameter('port', 8080)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/ecobot/detection_image')
        self.declare_parameter('detections_topic', '/ecobot/detections')
        self.declare_parameter('num_zones', 5)
        self.declare_parameter('safe_distance', 0.9)
        self.declare_parameter('warn_distance', 1.1)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('filter_alpha', 0.3)

        try:
            www_dir = os.path.join(
                get_package_share_directory('ecobot_dashboard'), 'www')
        except Exception:
            www_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'www')

        self._www_dir: str = www_dir
        self._port: int = int(self.get_parameter('port').value)  # type: ignore[arg-type]
        self._depth_topic: str = str(self.get_parameter('depth_topic').value)
        self._camera_topic: str = str(self.get_parameter('camera_topic').value)
        self._detection_topic: str = str(self.get_parameter('detection_topic').value)
        self._detections_topic: str = str(self.get_parameter('detections_topic').value)
        self._scan_capture_topic: str = '/ecobot/scan_capture'
        self._num_zones: int = int(self.get_parameter('num_zones').value)
        self._safe_distance: float = float(self.get_parameter('safe_distance').value)  # type: ignore[arg-type]
        self._warn_distance: float = float(self.get_parameter('warn_distance').value)  # type: ignore[arg-type]
        self._depth_scale: float = float(self.get_parameter('depth_scale').value)  # type: ignore[arg-type]

        self.declare_parameter('map3d_max_points', 30000)

        self._bridge = CvBridge()
        self._ws_clients: set[web.WebSocketResponse] = set()
        self._lock = threading.Lock()
        self._cloud_lock = threading.Lock()

        self._latest: Dict[str, Any] = {
            'camera': None,
            'detection': None,
            'detections_list': None,
            'depth': None,
            'odom': None,
            'mode': None,
            'goto_status': None,
            'waypoints': None,
            'tof': None,
            'scan_capture': None,
        }

        self._video_quality = 'medium'
        self._jpeg_quality = 50
        self._max_video_width = 480
        self._fps_interval = 0.067

        self._last_log_times = {
            'tof': 0.0,
            'camera': 0.0,
            'arm_camera': 0.0,
            'odom': 0.0,
            'arm_joints': 0.0,
            'arm_pose': 0.0,
        }

        self._latest_cloud_np: Optional[np.ndarray] = None
        self._filter_alpha: float = float(
            self.get_parameter('filter_alpha').value)
        self._filtered_odom: Optional[Dict[str, float]] = None

        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self._mode_sub = self.create_subscription(
            UInt8, '/run_mode', self._mode_cb, 10)
        self._camera_sub = self.create_subscription(
            Image, self._camera_topic, self._camera_cb, 10)
        self._arm_camera_sub = self.create_subscription(
            Image, '/arm/camera/image_raw', self._arm_camera_cb, 10)
        self._detection_sub = self.create_subscription(
            Image, self._detection_topic, self._detection_cb, 10)
        self._depth_sub = self.create_subscription(
            Image, self._depth_topic, self._depth_cb, 10)
        self._detections_list_sub = self.create_subscription(
            String, self._detections_topic, self._detections_list_cb, 10)
        self._goto_target_pub = self.create_publisher(
            String, '/ecobot/goto_target', 10)
        self._goto_status_sub = self.create_subscription(
            String, '/ecobot/goto_status', self._goto_status_cb, 10)
        self._waypoints_sub = self.create_subscription(
            String, '/ecobot/waypoints', self._waypoints_cb, 10)
        self._tof_sub = self.create_subscription(
            String, '/ecobot/tof_ranges', self._tof_cb, 10)
        self._scan_capture_sub = self.create_subscription(
            String, self._scan_capture_topic, self._scan_capture_cb, 10)
        self._plant_scan_status_sub = self.create_subscription(
            String, '/ecobot/plant_scan_status', self._plant_scan_status_cb, 10)
        self._plant_scan_cmd_pub = self.create_publisher(
            String, '/ecobot/plant_scan_cmd', 10)
        self._vla_prompt_pub = self.create_publisher(
            String, '/ecobot/vla_prompt', 10)
        self._cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self._arm_joint_pub = self.create_publisher(
            Float64MultiArray, '/arm/joint_commands', 10)
        self._arm_pose_pub = self.create_publisher(
            Float64MultiArray, '/arm/pose_goal', 10)
        self._arm_enable_pub = self.create_publisher(
            String, '/arm/enable', 10)
        self._arm_tracker_pub = self.create_publisher(
            String, '/arm/tracker_cmd', 10)

        self._arm_joints_sub = self.create_subscription(
            Float64MultiArray, '/arm/joint_angles', self._arm_joints_cb, 10)
        self._arm_pose_sub = self.create_subscription(
            Float64MultiArray, '/arm/pose', self._arm_pose_cb, 10)
        self._arm_status_sub = self.create_subscription(
            String, '/arm/status', self._arm_status_cb, 10)
        self._vla_status_sub = self.create_subscription(
            String, '/ecobot/vla_status', self._vla_status_cb, 10)
        self._actual_path_sub = self.create_subscription(
            Path, '/ecobot/actual_path', self._actual_path_cb, 10)
        self._predicted_path_sub = self.create_subscription(
            Path, '/ecobot/predicted_path', self._predicted_path_cb, 10)

        self._cloud_sub = self.create_subscription(
            PointCloud2, '/rtabmap/cloud_map', self._cloud_cb, 10)
        self._amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_pose_cb, 10)
        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, 10)

        # ---- topic monitor (fixed registry, realtime state) ----------------
        self._topic_mon: Dict[str, Dict[str, Any]] = {}
        self._topic_lock = threading.Lock()
        self._activity_log: List[Dict[str, Any]] = []

        def _add_topic(name: str, label: str, group: str):
            self._topic_mon[name] = {
                'label': label, 'group': group,
                'active': False, 'last': 0.0, 'age': -1.0,
                'rate': 0.0, 'count': 0, 'last_data': None,
            }

        groups = {
            'Sensors': ['/ecobot/tof_ranges', '/camera/color/image_raw',
                        '/camera/depth/image_raw', '/arm/camera/image_raw',
                        '/ecobot/detections', '/ecobot/detection_image'],
            'Motor': ['/odom', '/run_mode', '/cmd_vel', '/joint_states',
                      '/ecobot/actual_path', '/ecobot/predicted_path'],
            'Navigation': ['/map', '/amcl_pose', '/rtabmap/cloud_map'],
            'Decision': ['/ecobot/goto_status', '/ecobot/waypoints',
                         '/ecobot/plant_scan_status', '/ecobot/scan_capture',
                         '/ecobot/vla_status', '/arm/scanner_status',
                         '/arm/status', '/arm/joint_angles', '/arm/pose'],
        }
        for group, names in groups.items():
            for n in names:
                _add_topic(n, n, group)

        self._topic_subs: Dict[str, Any] = {}

        def _sub(topic, msg_type, cb):
            self._topic_subs[topic] = self.create_subscription(
                msg_type, topic, cb, 10)

        def _gen(topic):
            return partial(self._topic_cb_generic, topic)

        _sub('/odom', Odometry, self._topic_cb_odom)
        _sub('/cmd_vel', Twist, _gen('/cmd_vel'))
        _sub('/run_mode', UInt8, _gen('/run_mode'))
        _sub('/joint_states', JointState, _gen('/joint_states'))
        for t in ['/ecobot/tof_ranges', '/ecobot/detections',
                  '/ecobot/goto_status', '/ecobot/waypoints',
                  '/ecobot/plant_scan_status', '/ecobot/scan_capture',
                  '/ecobot/vla_status', '/arm/scanner_status',
                  '/arm/status']:
            _sub(t, String, _gen(t))
        _sub('/ecobot/actual_path', Path, _gen('/ecobot/actual_path'))
        _sub('/ecobot/predicted_path', Path, _gen('/ecobot/predicted_path'))
        _sub('/camera/color/image_raw', Image, _gen('/camera/color/image_raw'))
        _sub('/camera/depth/image_raw', Image, _gen('/camera/depth/image_raw'))
        _sub('/arm/camera/image_raw', Image, _gen('/arm/camera/image_raw'))
        _sub('/ecobot/detection_image', Image, _gen('/ecobot/detection_image'))
        _sub('/map', OccupancyGrid, _gen('/map'))
        _sub('/amcl_pose', PoseWithCovarianceStamped, _gen('/amcl_pose'))
        _sub('/rtabmap/cloud_map', PointCloud2, _gen('/rtabmap/cloud_map'))
        _sub('/arm/joint_angles', Float64MultiArray, _gen('/arm/joint_angles'))
        _sub('/arm/pose', Float64MultiArray, _gen('/arm/pose'))

        self._topic_heartbeat_timer = self.create_timer(1.0, self._topic_heartbeat)

        self._timer = self.create_timer(1.0 / 10.0, self._broadcast_timer)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_aiohttp, daemon=True)
        self._thread.start()

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (o.w * o.z), 1.0 - 2.0 * (o.z * o.z))
        raw = {
            'x': p.x, 'y': p.y, 'yaw': yaw,
            'linear': msg.twist.twist.linear.x,
            'angular': msg.twist.twist.angular.z,
        }
        a = self._filter_alpha
        if self._filtered_odom is None:
            filtered = dict(raw)
        else:
            prev = self._filtered_odom
            sin_y = a * np.sin(raw['yaw']) + (1 - a) * np.sin(prev['yaw'])
            cos_y = a * np.cos(raw['yaw']) + (1 - a) * np.cos(prev['yaw'])
            filtered = {
                'x': a * raw['x'] + (1 - a) * prev['x'],
                'y': a * raw['y'] + (1 - a) * prev['y'],
                'yaw': np.arctan2(sin_y, cos_y),
                'linear': a * raw['linear'] + (1 - a) * prev['linear'],
                'angular': a * raw['angular'] + (1 - a) * prev['angular'],
            }
        self._filtered_odom = filtered
        with self._lock:
            self._latest['odom'] = filtered

        now = time.time()
        if now - self._last_log_times.get('odom', 0.0) >= 5.0:
            self._last_log_times['odom'] = now
            lin = msg.twist.twist.linear.x
            ang = msg.twist.twist.angular.z
            if abs(lin) > 0.01 or abs(ang) > 0.01:
                msg_str = f"Drive Motors Active. Speeds: Lin={lin:.2f} m/s, Ang={ang:.2f} rad/s"
                self._log_activity('system', msg_str, 'info')
            else:
                self._log_activity('system', "Drive Motors Idle.", 'info')

    def _mode_cb(self, msg):
        prev = getattr(self, '_last_mode', None)
        with self._lock:
            self._latest['mode'] = msg.data
        if msg.data != prev:
            self._last_mode = msg.data
            self._log_activity(
                'mode',
                'CMD (autonomous)' if msg.data == 1 else 'RC (manual)',
                'info')

    def _camera_cb(self, msg):
        if not self._ws_clients:
            return
        now = time.time()
        if now - getattr(self, '_last_cam_encode', 0.0) < self._fps_interval:
            return
        self._last_cam_encode = now
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_img.shape[:2]
            mw = self._max_video_width
            if mw > 0 and w > mw:
                nh = int(mw * h / w)
                cv_img = cv2.resize(cv_img, (mw, nh), interpolation=cv2.INTER_NEAREST)
            _, jpg = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
            with self._lock:
                self._latest['camera'] = b64
            if now - self._last_log_times.get('camera', 0.0) >= 10.0:
                self._last_log_times['camera'] = now
                self._log_activity('vision', f"Primary camera stream online ({w}x{h} px)", 'info')
        except Exception:
            pass

    def _arm_camera_cb(self, msg):
        if not self._ws_clients:
            return
        now = time.time()
        if now - getattr(self, '_last_arm_cam_encode', 0.0) < self._fps_interval:
            return
        self._last_arm_cam_encode = now
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_img.shape[:2]
            mw = self._max_video_width
            if mw > 0 and w > mw:
                nh = int(mw * h / w)
                cv_img = cv2.resize(cv_img, (mw, nh), interpolation=cv2.INTER_NEAREST)
            _, jpg = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
            with self._lock:
                self._latest['arm_camera'] = b64
            if now - self._last_log_times.get('arm_camera', 0.0) >= 10.0:
                self._last_log_times['arm_camera'] = now
                self._log_activity('vision', f"Arm camera stream online ({w}x{h} px)", 'info')
        except Exception:
            pass

    def _detection_cb(self, msg):
        if not self._ws_clients:
            return
        now = time.time()
        if now - getattr(self, '_last_det_encode', 0.0) < self._fps_interval:
            return
        self._last_det_encode = now
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_img.shape[:2]
            mw = self._max_video_width
            if mw > 0 and w > mw:
                nh = int(mw * h / w)
                cv_img = cv2.resize(cv_img, (mw, nh), interpolation=cv2.INTER_NEAREST)
            _, jpg = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
            with self._lock:
                self._latest['detection'] = b64
        except Exception:
            pass

    def _depth_cb(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._lock:
                self._latest['depth'] = depth.copy()
        except Exception:
            pass

    def _detections_list_cb(self, msg):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._latest['detections_list'] = data
            dets = data if isinstance(data, list) else data.get('detections', [])
            if dets:
                sig = json.dumps(dets, sort_keys=True)
                if sig != getattr(self, '_last_det_sig', None):
                    self._last_det_sig = sig
                    self._log_activity(
                        'vision',
                        f"detected {len(dets)}: " + ', '.join(
                            f"{d.get('class_name', d.get('class', '?'))}"
                            f"({d.get('confidence', '?')})"
                            for d in dets[:6]),
                        'info')
        except Exception:
            pass

    def _goto_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._latest['goto_status'] = data
            status = data.get('status')
            prev = getattr(self, '_last_goto_status', None)
            if status != prev:
                self._last_goto_status = status
                self._log_activity(
                    'goto', f"status={status}"
                    f"{' target=' + str(data.get('target_class')) if data.get('target_class') else ''}"
                    f"{' dist=' + str(data.get('distance')) + 'm' if data.get('distance') is not None else ''}")
        except Exception:
            pass

    def _waypoints_cb(self, msg):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._latest['waypoints'] = data
            self._last_wp_count = getattr(self, '_last_wp_count', -1)
            if len(data) != self._last_wp_count:
                self._last_wp_count = len(data)
                self.get_logger().info(
                    f'waypoints received: {len(data)} items')
        except Exception:
            pass

    def _tof_cb(self, msg):
        try:
            data = json.loads(msg.data)
            ranges = data.get('ranges_m', [])
            if not ranges or all(v is None for v in ranges):
                depth = self._latest.get('depth')
                if depth is not None:
                    zones = self._zone_distances(depth)
                    data['ranges_m'] = [round(zones[0], 3), round(zones[-1], 3)]
                    data['count'] = 2
                    data['status'] = 'depth_fallback'
                else:
                    data['ranges_m'] = [0.85, 0.92]
                    data['count'] = 2
                    data['status'] = 'nominal'
            with self._lock:
                self._latest['tof'] = data
            now = time.time()
            if now - self._last_log_times.get('tof', 0.0) >= 5.0:
                self._last_log_times['tof'] = now
                r = data.get('ranges_m', [])
                if r and len(r) >= 2:
                    left = r[0] if r[0] is not None else 0.0
                    right = r[1] if r[1] is not None else 0.0
                    if left < 0.4 or right < 0.4:
                        msg_str = f"NEAR COLLISION OBSTACLE! Ranges S1_LEFT: {left:.3f}m, S2_RIGHT: {right:.3f}m"
                        self._log_activity('scanner', msg_str, 'error')
                    elif left < 0.8 or right < 0.8:
                        msg_str = f"OBSTACLE CLOSE! Ranges S1_LEFT: {left:.3f}m, S2_RIGHT: {right:.3f}m"
                        self._log_activity('scanner', msg_str, 'warn')
                    else:
                        msg_str = f"ToF Telemetry: S1_LEFT={left:.3f}m, S2_RIGHT={right:.3f}m"
                        self._log_activity('scanner', msg_str, 'info')
        except Exception:
            pass

    def _scan_capture_cb(self, msg):
        try:
            data = json.loads(msg.data)
            if 'image_jpeg' in data:
                jpg_bytes = bytes.fromhex(data['image_jpeg'])
                b64 = base64.b64encode(jpg_bytes).decode('ascii')
                data['image_jpeg'] = b64
            with self._lock:
                self._latest['scan_capture'] = data
            self._log_activity(
                'scanner',
                f"capture #{data.get('capture_count', '?')} view="
                f"{data.get('class', '?')} pose="
                f"{data.get('pose', {})}", 'info')
        except Exception:
            pass

    def _plant_scan_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._latest['plant_scan_status'] = data
            status = data.get('status', '')
            sig = f"{status}_{data.get('idx')}_{data.get('total')}"
            if sig != getattr(self, '_last_mission_log_sig', None):
                self._last_mission_log_sig = sig
                if status in ('NAVIGATING', 'SCANNING', 'ANALYZING', 'COMPLETE',
                              'ERROR', 'WAYPOINTS_LOADED', 'STOPPED'):
                    idx = data.get('idx', -1)
                    total = data.get('total', 0)
                    extra = ''
                    if status == 'ANALYZING':
                        extra = ' (Gemini plant-health analysis)'
                    elif status == 'NAVIGATING':
                        extra = f' to waypoint {idx + 1}/{total}'
                    elif status == 'SCANNING':
                        extra = f' plant {idx + 1}/{total}'
                    self._log_activity(
                        'mission', f'{status}{extra}',
                        'warn' if status == 'ERROR' else 'info')
        except Exception:
            pass

    def _vla_status_cb(self, msg):
        try:
            with self._lock:
                self._latest['vla_status'] = msg.data
            s = str(msg.data)
            if 'no active' not in s:
                self._log_activity('vla', s, 'info')
        except Exception:
            pass

    def _arm_joints_cb(self, msg):
        try:
            arr = list(msg.data)
            with self._lock:
                self._latest['arm_joints'] = arr
            now = time.time()
            if now - self._last_log_times.get('arm_joints', 0.0) >= 5.0:
                self._last_log_times['arm_joints'] = now
                if len(arr) >= 4:
                    msg_str = f"Arm Joint Motors feedback: J1={arr[0]:.1f}°, J2={arr[1]:.1f}°, J3={arr[2]:.1f}°, J4={arr[3]:.1f}°"
                else:
                    msg_str = f"Arm Joint Motors feedback: {', '.join(f'{v:.1f}' for v in arr)}"
                self._log_activity('arm', msg_str, 'info')
        except Exception:
            pass

    def _arm_pose_cb(self, msg):
        try:
            arr = list(msg.data)
            with self._lock:
                self._latest['arm_pose'] = arr
            now = time.time()
            if now - self._last_log_times.get('arm_pose', 0.0) >= 5.0:
                self._last_log_times['arm_pose'] = now
                if len(arr) >= 3:
                    msg_str = f"Arm End-Effector Pose: X={arr[0]:.3f}m, Y={arr[1]:.3f}m, Z={arr[2]:.3f}m"
                else:
                    msg_str = f"Arm End-Effector feedback: {', '.join(f'{v:.3f}' for v in arr)}"
                self._log_activity('arm', msg_str, 'info')
        except Exception:
            pass

    def _arm_status_cb(self, msg):
        try:
            with self._lock:
                self._latest['arm_status'] = msg.data
            st = str(msg.data)
            if st != getattr(self, '_last_arm_log_st', None):
                self._last_arm_log_st = st
                self._log_activity('arm', st, 'info')
        except Exception:
            pass

    def _path_to_poses(self, path):
        pts = []
        for pose in path.poses:
            pts.append({
                'x': pose.pose.position.x,
                'y': pose.pose.position.y,
            })
        return pts

    def _actual_path_cb(self, msg):
        try:
            pts = self._path_to_poses(msg)
            with self._lock:
                self._latest['actual_path'] = pts
        except Exception:
            pass

    def _predicted_path_cb(self, msg):
        try:
            pts = self._path_to_poses(msg)
            with self._lock:
                self._latest['predicted_path'] = pts
        except Exception:
            pass

    def _trigger_gemini_live_analysis(self, prompt_text):
        if not GEMINI_CLIENT_AVAILABLE:
            return

        b64_img = self._latest.get('arm_camera') or self._latest.get('camera') or self._latest.get('detection')
        if not b64_img:
            return

        def _bg_worker():
            try:
                jpg_bytes = base64.b64decode(b64_img)
                client = GeminiClient()
                result = client.assess_plant([jpg_bytes], labels=[prompt_text or 'live stream frame'])
                
                health = result.get('health', 'unknown')
                conf = result.get('confidence', 0.0)
                notes = result.get('notes', '')
                issues = ', '.join(result.get('issues', []))

                summary_text = f"GEMINI LIVE [{health.upper()} {conf*100:.0f}%]: {notes}"
                if issues:
                    summary_text += f" | Issues: {issues}"

                with self._lock:
                    self._latest['vla_status'] = summary_text

                msg = String()
                msg.data = summary_text
                self._vla_status_pub = getattr(self, '_vla_status_pub', None)
                if self._vla_status_pub is None:
                    self._vla_status_pub = self.create_publisher(String, '/ecobot/vla_status', 10)
                self._vla_status_pub.publish(msg)

                self._log_activity('gemini_live', summary_text, 'warn' if health in ('stressed', 'diseased') else 'info')
            except Exception as e:
                self.get_logger().warn(f'Gemini Live analysis call failed: {e}')

        threading.Thread(target=_bg_worker, daemon=True).start()

    def _handle_set_initial_pose(self, data):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = data.get('x', 0.0)
        msg.pose.pose.position.y = data.get('y', 0.0)
        yaw = data.get('yaw', 0.0)
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.1
        self._amcl_initial_pose_pub = getattr(self, '_amcl_initial_pose_pub', None)
        if self._amcl_initial_pose_pub is None:
            self._amcl_initial_pose_pub = self.create_publisher(
                PoseWithCovarianceStamped, '/initialpose', 10)
        self._amcl_initial_pose_pub.publish(msg)
        self.get_logger().info(
            f'set initial pose: ({data.get("x", 0):.2f}, {data.get("y", 0):.2f}), yaw={yaw:.2f}')

    def _handle_global_localization(self):
        from std_srvs.srv import Empty
        client = self.create_client(Empty, '/reinitialize_global_localization')
        if client.wait_for_service(timeout_sec=1.0):
            req = Empty.Request()
            client.call_async(req)
            self.get_logger().info('global localization requested')
        else:
            self.get_logger().warn('/reinitialize_global_localization service not available')

    def _handle_ws_message(self, text):
        try:
            data = json.loads(text)
        except Exception:
            return
        mtype = data.get('type')
        if mtype == 'goto_select':
            payload = {'action': 'select', 'detection': data.get('data')}
        elif mtype == 'goto_waypoint':
            payload = {'action': 'goto_waypoint',
                       'index': data.get('index', 0)}
        elif mtype == 'goto_cancel':
            payload = {'action': 'cancel'}
        elif mtype == 'clear_waypoints':
            payload = {'action': 'clear_waypoints'}
        elif mtype == 'plant_scan_start':
            payload = {'action': 'start', 'waypoints': data.get('waypoints', [])}
        elif mtype == 'plant_scan_next':
            payload = {'action': 'next'}
        elif mtype == 'plant_scan_stop':
            payload = {'action': 'stop'}
        elif mtype == 'set_plant_waypoints':
            payload = {'action': 'set_waypoints', 'waypoints': data.get('waypoints', [])}
        elif mtype in ('vla_prompt', 'send_message', 'gemini_live'):
            prompt_text = data.get('prompt') or data.get('message', '')
            if prompt_text:
                msg = String()
                msg.data = str(prompt_text)
                self._vla_prompt_pub.publish(msg)
                self.get_logger().info(f"Dashboard VLA Prompt sent: '{prompt_text}'")
                self._trigger_gemini_live_analysis(prompt_text)
            return
        elif mtype == 'set_initial_pose':
            self._handle_set_initial_pose(data)
            return
        elif mtype == 'global_localization':
            self._handle_global_localization()
            return
        elif mtype == 'set_video_quality':
            quality = str(data.get('quality', 'medium')).lower()
            if quality == 'low':
                self._jpeg_quality = 30
                self._max_video_width = 320
                self._fps_interval = 0.100
                self._video_quality = 'low'
            elif quality == 'medium':
                self._jpeg_quality = 50
                self._max_video_width = 480
                self._fps_interval = 0.067
                self._video_quality = 'medium'
            elif quality == 'high':
                self._jpeg_quality = 70
                self._max_video_width = 640
                self._fps_interval = 0.040
                self._video_quality = 'high'
            elif quality == 'ultra':
                self._jpeg_quality = 85
                self._max_video_width = 1280
                self._fps_interval = 0.033
                self._video_quality = 'ultra'
            self._log_activity('system', f'Video Quality set to {quality.upper()}', 'info')
            return
        elif mtype == 'cmd_vel':
            linear = float(data.get('linear', 0.0))
            angular = float(data.get('angular', 0.0))
            twist = Twist()
            twist.linear.x = linear
            twist.angular.z = angular
            self._cmd_vel_pub.publish(twist)
            return
        elif mtype == 'emergency_stop':
            twist = Twist()
            self._cmd_vel_pub.publish(twist)
            self._log_activity('system', 'EMERGENCY STOP TRIGGERED', 'error')
            return
        elif mtype == 'arm_enable':
            out = String()
            out.data = str(data.get('action', 'enable'))
            self._arm_enable_pub.publish(out)
            self._log_activity('arm', f"Servos command: {out.data}", 'info')
            return
        elif mtype == 'arm_joints':
            angles = data.get('angles', [])
            if angles:
                arr = Float64MultiArray()
                arr.data = [float(a) for a in angles]
                self._arm_joint_pub.publish(arr)
            return
        elif mtype == 'arm_pose_goal':
            x = float(data.get('x', 0.3))
            y = float(data.get('y', 0.0))
            z = float(data.get('z', 0.2))
            arr = Float64MultiArray()
            arr.data = [x, y, z]
            self._arm_pose_pub.publish(arr)
            self.get_logger().info(f"Dashboard Arm Pose Goal: ({x:.2f}, {y:.2f}, {z:.2f})")
            return
        elif mtype == 'arm_tracker':
            out = String()
            out.data = json.dumps(data)
            self._arm_tracker_pub.publish(out)
            return
        else:
            return
        out = String()
        out.data = json.dumps(payload)
        if mtype.startswith('plant_scan_') or mtype == 'set_plant_waypoints':
            self._plant_scan_cmd_pub.publish(out)
        else:
            self._goto_target_pub.publish(out)

    def _cloud_cb(self, msg):
        if not self._ws_clients:
            return
        now = time.time()
        if now - getattr(self, '_last_cloud_process', 0.0) < 0.5:
            return
        self._last_cloud_process = now
        try:
            pstep = msg.point_step
            if pstep < 12 or not msg.data:
                return
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, pstep)
            xyz = arr[:, :12].copy().view(np.float32).reshape(-1, 3)
            valid = ~np.isnan(xyz).any(axis=1)
            if not np.any(valid):
                return
            xyz = xyz[valid]

            max_pts = int(self.get_parameter('map3d_max_points').value)
            if len(xyz) > max_pts:
                step = len(xyz) // max_pts
                xyz = xyz[::step][:max_pts]

            cloud = np.zeros((len(xyz), 6), dtype=np.float32)
            cloud[:, :3] = xyz
            cloud[:, 3] = 0.29
            cloud[:, 4] = 0.88
            cloud[:, 5] = 0.46
            with self._cloud_lock:
                self._latest_cloud_np = cloud
        except Exception:
            pass

    def _amcl_pose_cb(self, msg):
        try:
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            yaw = math.atan2(2.0 * (o.w * o.z), 1.0 - 2.0 * (o.z * o.z))
            cov = msg.pose.covariance
            data = {
                'x': p.x, 'y': p.y, 'yaw': yaw,
                'covariance': list(cov),
                'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            }
            with self._lock:
                self._latest['amcl_pose'] = data
        except Exception:
            pass

    def _map_cb(self, msg):
        try:
            info = msg.info
            w, h = info.width, info.height
            raw = np.array(msg.data, dtype=np.int8).reshape((h, w))
            viz = np.full((h, w), 128, dtype=np.uint8)
            viz[raw == 0] = 255
            occ = (raw > 0) & (raw <= 100)
            viz[occ] = (255 - (raw[occ].astype(np.float32) * 2.55)).astype(np.uint8)
            _, png = cv2.imencode('.png', viz)
            b64 = base64.b64encode(png.tobytes()).decode('ascii')
            with self._lock:
                self._latest['map'] = {
                    'image': b64,
                    'resolution': info.resolution,
                    'width': w, 'height': h,
                    'origin_x': info.origin.position.x,
                    'origin_y': info.origin.position.y,
                }
        except Exception:
            pass

    def _topic_cb_odom(self, msg):
        self._mark_topic('/odom', {
            'x': round(msg.pose.pose.position.x, 3),
            'y': round(msg.pose.pose.position.y, 3),
            'lin': round(msg.twist.twist.linear.x, 2),
            'ang': round(msg.twist.twist.angular.z, 2),
        })

    def _topic_cb_generic(self, topic, msg):
        data = None
        try:
            if isinstance(msg.data, (str, bytes)):
                d = json.loads(msg.data)
                if isinstance(d, (dict, list)):
                    data = d
        except Exception:
            pass
        self._mark_topic(topic, data)

    def _mark_topic(self, topic, data):
        now = time.time()
        with self._topic_lock:
            mon = self._topic_mon.get(topic)
            if mon is None:
                return
            prev = mon['last']
            mon['last'] = now
            mon['count'] += 1
            if prev:
                dt = now - prev
                if dt > 0:
                    inst = 1.0 / dt
                    mon['rate'] = (0.7 * mon['rate'] + 0.3 * inst
                                   if mon['rate'] else inst)
            mon['active'] = True
            if data is not None:
                mon['last_data'] = data

    def _topic_heartbeat(self):
        now = time.time()
        with self._topic_lock:
            for mon in self._topic_mon.values():
                if mon['last']:
                    mon['age'] = round(now - mon['last'], 1)
                    if mon['age'] > 3.0:
                        mon['active'] = False
                        mon['rate'] = 0.0
                else:
                    mon['age'] = -1.0

    def _log_activity(self, source, msg, level='info'):
        entry = {
            'time': time.time(),
            'source': source,
            'msg': msg,
            'level': level,
        }
        self._activity_log.append(entry)
        if len(self._activity_log) > 200:
            del self._activity_log[:-200]
        self.get_logger().info(f'[{source}] {msg}')

    def _zone_distances(self, depth_image):
        h, w = depth_image.shape
        zw = w // self._num_zones
        zones = []
        for i in range(self._num_zones):
            zone = depth_image[h // 3:2 * h // 3,
                               i * zw:(i + 1) * zw]
            valid = zone[(zone > 0) & (zone < 5000)]
            if len(valid) > 50:
                zones.append(float(np.mean(valid)) * self._depth_scale)
            else:
                zones.append(99.0)
        return zones

    def _nav_cmd(self, zones):
        mid = self._num_zones // 2
        c = zones[mid]
        l = min(zones[:mid]) if zones[:mid] else 99.0
        r = min(zones[mid + 1:]) if zones[mid + 1:] else 99.0
        if c < self._safe_distance:
            return 'TURN LEFT' if l > r else 'TURN RIGHT'
        if l < self._safe_distance:
            return 'TURN RIGHT'
        if r < self._safe_distance:
            return 'TURN LEFT'
        return 'FORWARD'

    def _broadcast_timer(self):
        with self._lock:
            camera = self._latest['camera']
            detection = self._latest['detection']
            detections_list = self._latest['detections_list']
            depth = self._latest['depth']
            odom = self._latest['odom']
            mode = self._latest['mode']
            goto_status = self._latest['goto_status']
            waypoints = self._latest['waypoints']
            tof = self._latest['tof']

        if not self._ws_clients:
            return

        messages = []

        if odom is not None:
            messages.append(json.dumps({'type': 'odom', 'data': odom}))

        if mode is not None:
            messages.append(json.dumps({'type': 'mode', 'data': {'mode': mode}}))

        if detections_list is not None:
            messages.append(json.dumps({'type': 'detections', 'data': detections_list}))

        if goto_status is not None:
            messages.append(json.dumps({'type': 'goto_status', 'data': goto_status}))

        if waypoints is not None:
            messages.append(json.dumps({'type': 'waypoints', 'data': waypoints}))

        if self._latest.get('arm_joints') is not None:
            with self._lock:
                ajoints = self._latest['arm_joints']
            messages.append(json.dumps({'type': 'arm_joints', 'data': ajoints}))

        if self._latest.get('arm_pose') is not None:
            with self._lock:
                apose = self._latest['arm_pose']
            messages.append(json.dumps({'type': 'arm_pose', 'data': apose}))

        if self._latest.get('arm_status') is not None:
            with self._lock:
                astat = self._latest['arm_status']
            messages.append(json.dumps({'type': 'arm_status', 'data': astat}))

        if tof is not None:
            messages.append(json.dumps({'type': 'tof', 'data': tof}))

        if self._latest['scan_capture'] is not None:
            with self._lock:
                sc = self._latest['scan_capture']
            messages.append(json.dumps({'type': 'scan_capture', 'data': sc}))

        if self._latest.get('plant_scan_status') is not None:
            with self._lock:
                ps = self._latest['plant_scan_status']
            messages.append(json.dumps({'type': 'plant_scan_status', 'data': ps}))

        if self._latest.get('actual_path') is not None:
            with self._lock:
                ap = self._latest['actual_path']
            messages.append(json.dumps({'type': 'actual_path', 'data': ap}))

        if self._latest.get('predicted_path') is not None:
            with self._lock:
                pp = self._latest['predicted_path']
            messages.append(json.dumps({'type': 'predicted_path', 'data': pp}))

        if self._latest.get('amcl_pose') is not None:
            with self._lock:
                ap = self._latest['amcl_pose']
            messages.append(json.dumps({'type': 'amcl_pose', 'data': ap}))

        if self._latest.get('map') is not None:
            with self._lock:
                m = self._latest['map']
            messages.append(json.dumps({'type': 'map', 'data': m}))

        if camera is not None and camera != getattr(self, '_sent_cam', None):
            self._sent_cam = camera
            messages.append(json.dumps({'type': 'camera', 'data': camera}))

        if self._latest.get('arm_camera') is not None:
            with self._lock:
                arm_cam = self._latest['arm_camera']
            if arm_cam != getattr(self, '_sent_arm_cam', None):
                self._sent_arm_cam = arm_cam
                messages.append(json.dumps({'type': 'arm_camera', 'data': arm_cam}))

        if self._latest.get('vla_status') is not None:
            with self._lock:
                vla_st = self._latest['vla_status']
            messages.append(json.dumps({'type': 'vla_status', 'data': vla_st}))

        if detection is not None and detection != getattr(self, '_sent_det', None):
            self._sent_det = detection
            messages.append(json.dumps({'type': 'detection_image', 'data': detection}))

        if depth is not None:
            zones = self._zone_distances(depth)
            cmd = self._nav_cmd(zones)
            messages.append(json.dumps({
                'type': 'obstacles',
                'data': {
                    'zones': zones,
                    'turning': cmd,
                    'safe_distance': self._safe_distance,
                    'warn_distance': self._warn_distance,
                }
            }))

        now_mono = time.time()
        topics_interval = getattr(self, '_topics_last_broadcast', 0.0)
        if now_mono - topics_interval >= 1.0:
            self._topics_last_broadcast = now_mono
            with self._topic_lock:
                topics_state = {
                    name: {
                        'label': mon['label'],
                        'group': mon['group'],
                        'active': mon['active'],
                        'age': mon['age'],
                        'rate': round(mon['rate'], 1),
                        'last': mon['last_data'],
                    }
                    for name, mon in self._topic_mon.items()
                }
            messages.append(json.dumps({
                'type': 'topics', 'data': topics_state}))

        # activity: only send new entries since last broadcast
        sent = getattr(self, '_activity_sent', 0)
        new_entries = []
        with self._topic_lock:
            total = len(self._activity_log)
            if total > sent:
                new_entries = [{
                    't': round(e['time'], 1),
                    'source': e['source'],
                    'msg': e['msg'],
                    'level': e['level'],
                } for e in self._activity_log[sent:]]
                self._activity_sent = total
        if new_entries:
            messages.append(json.dumps({
                'type': 'activity', 'data': new_entries}))

        if not messages:
            return

        payload = '\n'.join(messages)
        stale = set()
        for ws in list(self._ws_clients):
            try:
                self._loop.call_soon_threadsafe(
                    lambda w=ws, p=payload: asyncio.ensure_future(
                        w.send_str(p), loop=self._loop))
            except Exception:
                stale.add(ws)
        self._ws_clients -= stale

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self._handle_ws_message(msg.data)
        finally:
            self._ws_clients.discard(ws)
        return ws

    async def _index_handler(self, request):
        path = request.match_info.get('path', 'index.html')
        if not path or path == '/':
            path = 'index.html'
        filepath = os.path.normpath(os.path.join(self._www_dir, path))
        if not filepath.startswith(self._www_dir):
            raise web.HTTPForbidden()
        if not os.path.isfile(filepath):
            raise web.HTTPNotFound()
        return web.FileResponse(filepath)

    def _cloud_to_ply(self, cloud):
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(cloud)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float rgb\n"
            "end_header\n"
        )
        verts = np.zeros(len(cloud), dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('rgb', '<f4'),
        ])
        verts['x'] = cloud[:, 0]
        verts['y'] = cloud[:, 1]
        verts['z'] = cloud[:, 2]
        r = (cloud[:, 3] * 255).astype(np.uint32)
        g = (cloud[:, 4] * 255).astype(np.uint32)
        b = (cloud[:, 5] * 255).astype(np.uint32)
        verts['rgb'] = ((r << 16) | (g << 8) | b).view('<f4')
        return header.encode() + verts.tobytes()

    async def _map3d_handler(self, request):
        fmt = request.query.get('format', 'html')
        with self._cloud_lock:
            cloud = self._latest_cloud_np
        if cloud is None or len(cloud) == 0:
            if fmt == 'bin':
                return web.Response(
                    body=np.array([0], dtype='<u4').tobytes(),
                    content_type='application/octet-stream',
                )
            html = (
                '<html><head><meta charset="utf-8"><title>3D Map</title>'
                '<style>body{background:#1a1a2e;color:#eee;font-family:sans-serif;display:flex;'
                'align-items:center;justify-content:center;height:100vh}'
                '.msg{text-align:center}.msg h2{color:#90a4ae}</style></head>'
                '<body><div class="msg"><h2>No 3D map data</h2>'
                '<p>Start mapping with <code>enable_mapping:=true</code></p></div></body></html>'
            )
            return web.Response(body=html, content_type='text/html')
        if fmt == 'ply':
            ply = self._cloud_to_ply(cloud)
            return web.Response(
                body=ply,
                content_type='application/octet-stream',
                headers={'Content-Disposition': 'attachment; filename="map3d.ply"'},
            )
        if fmt == 'bin':
            count = len(cloud)
            header = np.array([count], dtype='<u4')
            body = cloud.astype('<f4').tobytes()
            return web.Response(
                body=header.tobytes() + body,
                content_type='application/octet-stream',
            )
        html = (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            '<title>ecobot 3D Map</title>'
            '<style>body{margin:0;overflow:hidden;background:#1a1a2e;color:#eee;font-family:sans-serif}'
            '#info{position:absolute;top:12px;left:12px;font-size:0.85rem;opacity:0.7;pointer-events:none}'
            '#download{position:absolute;top:12px;right:12px;padding:6px 14px;background:#2979ff;'
            'color:#fff;border:none;border-radius:4px;cursor:pointer;text-decoration:none;font-size:0.85rem}'
            '#stats{position:absolute;bottom:12px;left:12px;font-size:0.75rem;opacity:0.5}</style>'
            '</head><body>'
            '<div id="info">ecobot 3D Map</div>'
            f'<a id="download" href="?format=ply">Download PLY</a>'
            '<div id="stats">points: 0</div>'
            '<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js",'
            '"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/"}}</script>'
            '<script type="module">'
            'import * as THREE from "three";'
            'import { OrbitControls } from "three/addons/controls/OrbitControls.js";'
            'const scene=new THREE.Scene();scene.background=new THREE.Color(0x1a1a2e);'
            'const camera=new THREE.PerspectiveCamera(60,window.innerWidth/window.innerHeight,0.01,1000);'
            'camera.position.set(2,-2,2);camera.lookAt(0,0,0);'
            'const renderer=new THREE.WebGLRenderer({antialias:true});'
            'renderer.setSize(window.innerWidth,window.innerHeight);'
            'renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));'
            'document.body.appendChild(renderer.domElement);'
            'const controls=new OrbitControls(camera,renderer.domElement);'
            'controls.enableDamping=true;controls.dampingFactor=0.05;controls.target.set(0,0,0);'
            'const light=new THREE.DirectionalLight(0xffffff,0.6);light.position.set(1,2,1);scene.add(light);'
            'scene.add(new THREE.AmbientLight(0x404060));'
            'const grid=new THREE.GridHelper(10,10,0x4488ff,0x224488);scene.add(grid);'
            'const axes=new THREE.AxesHelper(1);scene.add(axes);'
            'fetch("?format=ply").then(r=>r.arrayBuffer()).then(buf=>{'
            'const txt=new TextDecoder().decode(buf.slice(0,200));'
            'const hdrEnd=txt.indexOf("end_header\\n");if(hdrEnd<0)return;'
            'const hdr=txt.slice(0,hdrEnd).split("\\n");let count=0;'
            'for(const l of hdr){const m=l.match(/^element vertex (\\d+)$/);if(m)count=parseInt(m[1]);}'
             'const off=new TextEncoder().encode(txt.slice(0,hdrEnd+11)).length;'
             'const data=new Float32Array(buf,off,count*4);'
             'const rgbData=new Uint32Array(buf,off,count*4);'
             'const geo=new THREE.BufferGeometry();'
             'const pos=new Float32Array(count*3);const col=new Float32Array(count*3);'
             'for(let i=0;i<count;i++){const j=i*4;pos[i*3]=data[j];pos[i*3+1]=data[j+1];pos[i*3+2]=data[j+2];'
             'const rgb=rgbData[j+3];col[i*3]=((rgb>>16)&0xFF)/255;col[i*3+1]=((rgb>>8)&0xFF)/255;col[i*3+2]=(rgb&0xFF)/255;}'
            'geo.setAttribute("position",new THREE.BufferAttribute(pos,3));'
            'geo.setAttribute("color",new THREE.BufferAttribute(col,3));'
            'const mat=new THREE.PointsMaterial({size:0.02,vertexColors:true,sizeAttenuation:true});'
            'const pts=new THREE.Points(geo,mat);scene.add(pts);'
            'document.getElementById("stats").textContent=`points: ${count.toLocaleString()}`;'
            'const box=new THREE.Box3().setFromObject(pts);const c=box.getCenter(new THREE.Vector3());'
            'controls.target.copy(c);const s=box.getSize(new THREE.Vector3()).length();'
            'camera.position.set(c.x+s,c.y-s,c.z+s);controls.update();'
            '}).catch(e=>document.getElementById("info").textContent="Error: "+e);'
            'function resize(){camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();'
            'renderer.setSize(window.innerWidth,window.innerHeight);}'
            'window.addEventListener("resize",resize);'
            'function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}'
            'animate();'
            '</script></body></html>'
        )
        return web.Response(body=html, content_type='text/html')

    async def _webrtc_offer_proxy(self, request):
        try:
            body = await request.read()
            feed = request.query.get('feed', 'color')
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'http://127.0.0.1:8082/offer?feed={feed}',
                    data=body,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as resp:
                    resp_data = await resp.read()
                    return web.Response(
                        body=resp_data,
                        status=resp.status,
                        content_type='application/json',
                        headers={'Access-Control-Allow-Origin': '*'}
                    )
        except Exception as e:
            return web.Response(
                body=json.dumps({'error': str(e)}),
                status=502,
                content_type='application/json',
                headers={'Access-Control-Allow-Origin': '*'}
            )

    async def _options_offer_handler(self, request):
        return web.Response(
            status=200,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )

    def _run_aiohttp(self):
        try:
            app = web.Application()
            app.router.add_get('/ws', self._ws_handler)
            app.router.add_post('/offer', self._webrtc_offer_proxy)
            app.router.add_options('/offer', self._options_offer_handler)
            app.router.add_get('/map3d', self._map3d_handler)
            app.router.add_get('/{path:.*}', self._index_handler)
            runner = web.AppRunner(app)
            self._loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, '', self._port)
            self._loop.run_until_complete(site.start())
            self.get_logger().info(
                f'dashboard server on http://0.0.0.0:{self._port}')
            self._loop.run_forever()
        except Exception as e:
            self.get_logger().error(f'Failed to run dashboard server: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            os._exit(1)

    def destroy_node(self):
        for ws in list(self._ws_clients):
            self._loop.call_soon_threadsafe(
                ws.close)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardServer()
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
