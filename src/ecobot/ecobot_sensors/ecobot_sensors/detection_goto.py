import json
import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tf2_ros
import numpy as np
from cv_bridge import CvBridge


class DetectionGoto(Node):
    def __init__(self):
        super().__init__('detection_goto')

        self.declare_parameter('cmd_vel_topic', '/goto_cmd_vel')
        self.declare_parameter('detections_topic', '/ecobot/detections')
        self.declare_parameter('target_topic', '/ecobot/goto_target')
        self.declare_parameter('status_topic', '/ecobot/goto_status')
        self.declare_parameter('waypoints_topic', '/ecobot/waypoints')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('stop_distance', 0.6)
        self.declare_parameter('wp_stop_distance', 0.15)
        self.declare_parameter('max_linear', 0.25)
        self.declare_parameter('max_angular', 0.8)
        self.declare_parameter('k_ang', 1.5)
        self.declare_parameter('k_lin', 0.6)
        self.declare_parameter('align_threshold', 0.35)
        self.declare_parameter('lost_timeout', 2.0)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('obstacle_stop_distance', 0.3)
        self.declare_parameter('search_timeout', 20.0)
        self.declare_parameter('search_speed', 0.5)
        self.declare_parameter('avoid_distance', 0.4)
        self.declare_parameter('avoid_angle', 1.05)
        self.declare_parameter('use_map_frame', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('blind_approach_limit', 1.0)

        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        target_topic = str(self.get_parameter('target_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)
        self.waypoints_topic = str(self.get_parameter('waypoints_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.wp_stop_distance = float(
            self.get_parameter('wp_stop_distance').value)
        self.max_linear = float(self.get_parameter('max_linear').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.k_ang = float(self.get_parameter('k_ang').value)
        self.k_lin = float(self.get_parameter('k_lin').value)
        self.align_threshold = float(
            self.get_parameter('align_threshold').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.obstacle_stop_dist = float(
            self.get_parameter('obstacle_stop_distance').value)
        self.search_timeout = float(self.get_parameter('search_timeout').value)
        self.search_speed = float(self.get_parameter('search_speed').value)
        self.avoid_distance = float(self.get_parameter('avoid_distance').value)
        self.avoid_angle = float(self.get_parameter('avoid_angle').value)
        self.use_map_frame = bool(self.get_parameter('use_map_frame').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self.blind_approach_limit = float(
            self.get_parameter('blind_approach_limit').value)

        self.detections = []
        self.target_class = None
        self.target_pos = (0.0, 0.0, 0.0)
        self.active = False
        self._mode = 'idle'
        self._wp_target = (0.0, 0.0)
        self._wp_frame = 'odom'
        self.last_seen = None
        self.status = 'IDLE'
        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_lock = threading.Lock()
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._odom_set = False
        self._map_available = False
        self._map_tx = 0.0
        self._map_ty = 0.0
        self._map_tyaw = 0.0

        self.waypoints = []
        self._saved_class = None
        self._last_tracked_det = None
        self._blind_start = None
        self._blind_last_z = 0.0

        self._avoid_state = 'none'
        self._avoid_dir = 1.0
        self._avoid_start = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        wp_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)
        self.wp_pub = self.create_publisher(
            String, self.waypoints_topic, wp_qos)
        self.create_subscription(String, detections_topic,
                                 self.detections_cb, 10)
        self.create_subscription(String, target_topic, self.target_cb, 10)
        self.create_subscription(Image, depth_topic, self.depth_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self._publish_waypoints()
        self.get_logger().info(
            f'detection_goto started — stop={self.stop_distance}m '
            f'wp_stop={self.wp_stop_distance}m '
            f'search={self.search_timeout}s@{self.search_speed}rad/s '
            f'avoid={self.avoid_distance}m@{math.degrees(self.avoid_angle):.0f}deg '
            f'map={self.use_map_frame}')

    def detections_cb(self, msg):
        try:
            dets = json.loads(msg.data)
            if isinstance(dets, list):
                self.detections = dets
        except Exception:
            pass

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._odom_x = p.x
        self._odom_y = p.y
        self._odom_yaw = math.atan2(
            2.0 * (o.w * o.z), 1.0 - 2.0 * (o.z * o.z))
        self._odom_set = True

    def _update_map_pose(self):
        if not self.use_map_frame or not self._odom_set:
            self._map_available = False
            return
        try:
            t = self.tf_buffer.lookup_transform(
                self._map_frame, 'odom',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
            self._map_tx = t.transform.translation.x
            self._map_ty = t.transform.translation.y
            self._map_tyaw = yaw
            self._map_available = True
        except Exception:
            self._map_available = False

    def _to_map(self, ox, oy):
        """Convert odom-frame (x, y) to map-frame (x, y)."""
        if not self._map_available:
            return ox, oy
        cos_m = math.cos(self._map_tyaw)
        sin_m = math.sin(self._map_tyaw)
        return (self._map_tx + cos_m * ox - sin_m * oy,
                self._map_ty + sin_m * ox + cos_m * oy)

    def _current_pose(self):
        """Return (x, y, yaw) in the navigation frame (map if available)."""
        if self._map_available:
            mx, my = self._to_map(self._odom_x, self._odom_y)
            myaw = self._odom_yaw + self._map_tyaw
            myaw = math.atan2(math.sin(myaw), math.cos(myaw))
            return mx, my, myaw
        return self._odom_x, self._odom_y, self._odom_yaw

    def target_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        action = data.get('action')
        if action == 'select':
            det = data.get('detection') or {}
            cls = det.get('class_name') or det.get('class')
            if cls:
                self._mode = 'tracking'
                self.target_class = cls
                self.target_pos = (float(det.get('x', 0.0)),
                                   float(det.get('y', 0.0)),
                                   float(det.get('z', 0.0)))
                self.active = True
                self.last_seen = self.get_clock().now()
                self._saved_class = None
                self._avoid_state = 'none'
                self._publish_status()
                self.get_logger().info(
                    f'goto target selected: {cls} '
                    f'pos=({self.target_pos[0]:.3f},{self.target_pos[2]:.3f})')
        elif action == 'goto_waypoint':
            idx = data.get('index', 0)
            if 0 <= idx < len(self.waypoints):
                wp = self.waypoints[idx]
                self._mode = 'waypoint'
                self._wp_target = (float(wp['x']), float(wp['y']))
                self._wp_frame = wp.get('frame', 'odom')
                self.target_class = wp.get('class_name', 'waypoint')
                self.active = True
                self._avoid_state = 'none'
                self._publish_status()
                self.get_logger().info(
                    f'goto waypoint {idx}: {self.target_class} '
                    f'({self._wp_target[0]:.3f},{self._wp_target[1]:.3f}) '
                    f'frame={self._wp_frame}')
        elif action == 'cancel':
            self.get_logger().info('goto cancelled')
            self._stop('IDLE')
        elif action == 'clear_waypoints':
            self.waypoints = []
            self._publish_waypoints()
            self.get_logger().info('waypoints cleared')

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = depth
        except Exception:
            pass

    def _obstacle_blocking(self):
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            return False
        h, w = depth.shape[:2]
        rows = slice(h // 2, 3 * h // 4)
        cols = slice(w // 3, 2 * w // 3)
        region = depth[rows, cols]
        valid = region[(region > 0) & (region < 5000)]
        if len(valid) < 50:
            return False
        return float(np.median(valid)) * self.depth_scale \
            < self.obstacle_stop_dist

    def _publish_status(self, distance=None):
        payload = {'status': self.status, 'target_class': self.target_class}
        if distance is not None:
            payload['distance'] = round(distance, 2)
        self.status_pub.publish(String(data=json.dumps(payload)))

    def _publish_waypoints(self):
        self.wp_pub.publish(String(data=json.dumps(self.waypoints)))

    def _set_status(self, status, distance=None):
        self.status = status
        self._publish_status(distance)

    def _stop(self, status):
        self.active = False
        self._mode = 'idle'
        self._avoid_state = 'none'
        self.cmd_pub.publish(Twist())
        self._set_status(status)

    def _save_waypoint(self, det):
        cls = self.target_class
        if not cls or cls == self._saved_class:
            return
        if not self._odom_set:
            self.get_logger().warn('cannot save waypoint: no odom')
            return
        self._saved_class = cls
        # estimate object position in odom frame from detection + robot pose
        dz = float(det.get('z', 0.0))
        dx = float(det.get('x', 0.0))
        obj_odom_x = self._odom_x + dz * math.cos(self._odom_yaw) \
                     - dx * math.sin(self._odom_yaw)
        obj_odom_y = self._odom_y + dz * math.sin(self._odom_yaw) \
                     + dx * math.cos(self._odom_yaw)
        self._update_map_pose()
        wx, wy = self._to_map(obj_odom_x, obj_odom_y)
        frame = self._map_frame if self._map_available else 'odom'
        nearby = []
        for d in self.detections:
            name = d.get('class_name') or d.get('class')
            if name and d.get('z') is not None:
                nearby.append({
                    'class_name': name,
                    'confidence': d.get('confidence', 0),
                    'distance': round(float(d['z']), 2),
                })
        wp = {
            'class_name': cls,
            'frame': frame,
            'x': round(wx, 3),
            'y': round(wy, 3),
            'z': round(dz, 3),
            'time': self.get_clock().now().nanoseconds * 1e-9,
            'nearby': nearby,
        }
        self.waypoints.append(wp)
        self._publish_waypoints()
        self.get_logger().info(
            f'waypoint saved: {cls} ({wp["x"]:.3f},{wp["y"]:.3f}) '
            f'frame={frame} obj_odom=({obj_odom_x:.3f},{obj_odom_y:.3f}) '
            f'dist={dz:.3f}m nearby={len(nearby)} total={len(self.waypoints)}')

    # ── main loop ────────────────────────────────────────────────

    def control_loop(self):
        self._publish_waypoints()
        self._update_map_pose()
        if not self.active:
            return
        now = self.get_clock().now()

        if self._avoid_state != 'none':
            self._control_avoid()
            return

        if self._mode == 'tracking':
            self._control_tracking(now)
        elif self._mode == 'blind_drive':
            self._control_blind_drive(now)
        elif self._mode == 'waypoint':
            self._control_waypoint()
        else:
            self._stop('IDLE')

    # ── AVOID state machine ──────────────────────────────────────

    def _control_avoid(self):
        px, py, _ = self._current_pose()
        if self._avoid_state == 'turn':
            elapsed = (self.get_clock().now() -
                       self._avoid_start).nanoseconds * 1e-9
            cmd = Twist()
            cmd.angular.z = self._avoid_dir * 0.35
            if elapsed >= 1.8:  # roughly 60° at 0.35 rad/s
                self._avoid_state = 'drive'
                self._avoid_start = self.get_clock().now()
            self.cmd_pub.publish(cmd)
            self._set_status('BLOCKED')
            return

        if self._avoid_state == 'drive':
            dist = math.hypot(px - self._avoid_base_x,
                              py - self._avoid_base_y)
            if dist >= self.avoid_distance or self._obstacle_blocking():
                self._avoid_state = 'none'
                self._avoid_dir *= -1.0
                self.get_logger().info('avoidance complete, resuming')
                return
            cmd = Twist()
            cmd.angular.z = 0.0
            cmd.linear.x = 0.2
            self.cmd_pub.publish(cmd)
            self._set_status('BLOCKED')
            return

    # ── blind drive (detection lost but close, just go straight) ──

    def _control_blind_drive(self, now):
        elapsed = (now - self._blind_start).nanoseconds * 1e-9
        est_traveled = elapsed * 0.15
        est_dist = self._blind_last_z - est_traveled

        if est_dist <= self.stop_distance:
            self.get_logger().info(
                f'blind reached {self.target_class} est={est_dist:.2f}m')
            self._mode = 'idle'
            self._stop('REACHED')
            return
        if elapsed > 8.0:
            self.get_logger().info(
                f'blind timeout {self.target_class}')
            self._mode = 'idle'
            self._stop('LOST')
            return

        cmd = Twist()
        # if obstacle right ahead, slow down but keep going
        if self._obstacle_blocking():
            cmd.linear.x = 0.05
        else:
            cmd.linear.x = 0.15
        self.cmd_pub.publish(cmd)
        self._set_status('TRACKING', distance=round(est_dist, 2))

    # ── tracking (camera-frame chase) ────────────────────────────

    def _control_tracking(self, now):
        best = None
        best_dist = float('inf')
        tx, tz = self.target_pos[0], self.target_pos[2]
        for d in self.detections:
            cls = d.get('class_name') or d.get('class')
            if cls != self.target_class:
                continue
            if d.get('z') is None:
                continue
            dx = float(d.get('x', 0.0)) - tx
            dz = float(d.get('z', 0.0)) - tz
            d2 = dx * dx + dz * dz
            if d2 < best_dist:
                best_dist = d2
                best = d

        if best is None:
            if self.last_seen is not None:
                elapsed = (now - self.last_seen).nanoseconds * 1e-9
                if elapsed > self.lost_timeout:
                    if self._last_tracked_det is not None:
                        last_z = float(
                            self._last_tracked_det.get('z', 99.0))
                        if last_z < self.blind_approach_limit:
                            self._mode = 'blind_drive'
                            self._blind_start = now
                            self._blind_last_z = last_z
                            self._save_waypoint(
                                self._last_tracked_det)
                            self.get_logger().info(
                                f'LOST close: {self.target_class} @'
                                f'{last_z:.2f}m — driving blind...')
                            return
                        # far: auto-waypoint navigation
                        self._save_waypoint(self._last_tracked_det)
                        if self.waypoints:
                            wp = self.waypoints[-1]
                            self._wp_target = (
                                float(wp['x']), float(wp['y']))
                            self._wp_frame = wp.get('frame',
                                                     'odom')
                            self._mode = 'waypoint'
                            self._avoid_state = 'none'
                            self._search_ticks = 0
                            self._saved_class = None
                            self.get_logger().info(
                                f'LOST far: {self.target_class}'
                                f' — waypoint navigation')
                            return
                    self._stop('LOST')
            return

        self.last_seen = now
        self._last_tracked_det = best
        x = float(best['x'])
        z = float(best['z'])
        angle = math.atan2(x, z)
        dist_err = z - self.stop_distance

        self._tick_log_count = getattr(self, '_tick_log_count', 0) + 1
        if self._tick_log_count % 30 == 0:
            self.get_logger().info(
                f'TRACKING {self.target_class}: dist={z:.2f}m '
                f'angle={math.degrees(angle):.0f}° '
                f'v_lin={min(self.max_linear, max(0.0, self.k_lin*dist_err)):.2f}')

        if dist_err <= 0.0:
            self.get_logger().info(
                f'reached {self.target_class} at {z:.2f}m')
            self._save_waypoint(best)
            self._stop('REACHED')
            return

        cmd = Twist()
        cmd.angular.z = max(-self.max_angular,
                            min(self.max_angular, -self.k_ang * angle))
        if abs(angle) < self.align_threshold:
            if self._obstacle_blocking():
                self._set_status('BLOCKED', distance=z)
                self._start_avoid()
                return
            cmd.linear.x = max(0.0,
                               min(self.max_linear, self.k_lin * dist_err))
        self.cmd_pub.publish(cmd)
        self._set_status('TRACKING', distance=z)

    # ── waypoint (odom/map frame) ────────────────────────────────

    def _control_waypoint(self):
        if not self._odom_set:
            return
        px, py, pyaw = self._current_pose()
        wx, wy = self._wp_target
        if self._wp_frame != 'odom' and not self._map_available:
            self.get_logger().warn(
                'waypoint frame is map but map not available — '
                'waiting for map frame')
            return
        dx = wx - px
        dy = wy - py
        dist_err = math.hypot(dx, dy) - self.wp_stop_distance

        _search = getattr(self, '_search_ticks', 0)
        if dist_err <= 0.0 or _search > 0:
            self._search_ticks = _search + 1
            if _search == 0:
                self.get_logger().info(
                    f'arrived near waypoint, searching for '
                    f'{self.target_class}...')
            cmd = Twist()
            cmd.angular.z = self.search_speed
            self.cmd_pub.publish(cmd)
            self._set_status('SEARCHING', distance=round(dist_err, 2))

            for d in self.detections:
                name = d.get('class_name') or d.get('class')
                if name == self.target_class and d.get('z') is not None:
                    self.get_logger().info(
                        f'found {self.target_class} during search, '
                        f'approaching...')
                    self._search_ticks = 0
                    self._mode = 'tracking'
                    self.target_pos = (float(d.get('x', 0.0)),
                                       float(d.get('y', 0.0)),
                                       float(d.get('z', 0.0)))
                    self.last_seen = self.get_clock().now()
                    self._saved_class = None
                    return

            max_ticks = int(self.search_timeout * 10)
            if self._search_ticks > max_ticks:
                self._search_ticks = 0
                self.get_logger().info(
                    f'search complete — stopped at waypoint '
                    f'{self.target_class}')
                self._stop('REACHED')
            return
        self._search_ticks = 0

        target_heading = math.atan2(dy, dx)
        angle_err = target_heading - pyaw
        angle_err = math.atan2(math.sin(angle_err), math.cos(angle_err))

        self._tick_log_count = getattr(self, '_tick_log_count', 0) + 1
        if self._tick_log_count % 30 == 0:
            self.get_logger().info(
                f'WAYPOINT {self.target_class}: dist={dist_err:.2f}m '
                f'angle_err={math.degrees(angle_err):.0f}°')

        cmd = Twist()
        cmd.angular.z = max(-self.max_angular,
                            min(self.max_angular, self.k_ang * angle_err))
        if abs(angle_err) < self.align_threshold:
            if self._obstacle_blocking():
                self._set_status('BLOCKED',
                                 distance=round(dist_err, 2))
                self._start_avoid()
                return
            cmd.linear.x = max(0.0,
                               min(self.max_linear, self.k_lin * dist_err))
        self.cmd_pub.publish(cmd)
        self._set_status('TRACKING', distance=round(dist_err, 2))

    def _start_avoid(self):
        self._avoid_state = 'turn'
        self._avoid_start = self.get_clock().now()
        self._avoid_base_x, self._avoid_base_y, _ = self._current_pose()
        self.get_logger().info(
            f'avoiding obstacle: {self._avoid_state}')


def main(args=None):
    rclpy.init(args=args)
    node = DetectionGoto()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
