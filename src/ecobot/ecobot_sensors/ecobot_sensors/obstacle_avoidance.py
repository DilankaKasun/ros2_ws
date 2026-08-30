import json
import socket
import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        if hasattr(socket, 'SO_REUSEPORT'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()


class ObstacleMJPEGHandler(BaseHTTPRequestHandler):
    frame = None
    frame_lock = threading.Lock()

    def do_GET(self):
        if self.path.startswith('/obstacle.mjpg'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            while rclpy.ok():
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.05)
                        continue
                    jpg = self.frame
                header = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' \
                       + str(len(jpg)).encode() + b'\r\n\r\n'
                try:
                    self.wfile.write(header + jpg + b'\r\n')
                except Exception:
                    break
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><img src="/obstacle.mjpg"/></body></html>')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.declare_parameter('safe_distance', 0.9)
        self.declare_parameter('warn_distance', 1.1)
        self.declare_parameter('zones', 5)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('reverse_speed', -0.2)
        self.declare_parameter('reverse_duration', 1.5)
        self.declare_parameter('turn_duration', 2.0)
        self.declare_parameter('show_viewer', False)
        self.declare_parameter('mjpeg_port', 8083)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('tof_threshold', 2.0)
        self.declare_parameter('avoid_hysteresis', 0.15)
        self.declare_parameter('mission_suppress_topic',
                               '/ecobot/mission_suppress_avoidance')

        self.safe_distance = self.get_parameter('safe_distance').value
        self.warn_distance = self.get_parameter('warn_distance').value
        self.avoid_hysteresis = self.get_parameter('avoid_hysteresis').value
        self.num_zones = self.get_parameter('zones').value
        self.max_linear = self.get_parameter('max_linear_speed').value
        self.max_angular = self.get_parameter('max_angular_speed').value
        self.reverse_speed = self.get_parameter('reverse_speed').value
        self.reverse_duration = self.get_parameter('reverse_duration').value
        self.turn_duration = self.get_parameter('turn_duration').value
        self.show_viewer = self.get_parameter('show_viewer').value
        mjpeg_port = self.get_parameter('mjpeg_port').value
        depth_topic = self.get_parameter('depth_topic').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.tof_threshold = self.get_parameter('tof_threshold').value
        mission_suppress_topic = str(
            self.get_parameter('mission_suppress_topic').value)

        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_lock = threading.Lock()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        ObstacleMJPEGHandler.frame = None
        self.mjpeg_server = ReusableHTTPServer(('', mjpeg_port), ObstacleMJPEGHandler)
        self.mjpeg_thread = threading.Thread(
            target=self.mjpeg_server.serve_forever, daemon=True)
        self.mjpeg_thread.start()
        self.get_logger().info(
            f'obstacle MJPEG stream on http://0.0.0.0:{mjpeg_port}/obstacle.mjpg')

        self.nav_vel_sub = self.create_subscription(
            Twist, '/nav_cmd_vel', self.nav_vel_cb, 10)
        self.latest_nav_vel = Twist()
        self.has_nav_vel = False
        self.nav_vel_timeout = 1.0
        self.last_nav_vel_time = self.get_clock().now()

        self.goto_vel_sub = self.create_subscription(
            Twist, '/goto_cmd_vel', self.goto_vel_cb, 10)
        self.latest_goto_vel = Twist()
        self.has_goto_vel = False
        self.goto_vel_timeout = 1.0
        self.last_goto_vel_time = self.get_clock().now()

        # detection_goto.py publishes True while it's driving toward an
        # auto-tracked target it deliberately wants to get close to (a
        # plant) — otherwise this node's own depth-based avoidance sees
        # the very thing being approached as an obstacle and swerves/
        # reverses away from it well before the target is even reached.
        # Same fail-safe timeout pattern as goto/nav_vel: an unrefreshed
        # signal (node died, was never launched) reverts to normal
        # avoidance rather than silently staying suppressed forever.
        self.suppress_avoidance_sub = self.create_subscription(
            Bool, '/ecobot/goto_suppress_avoidance',
            self.suppress_avoidance_cb, 10)
        self.suppress_avoidance = False
        self.suppress_avoidance_timeout = 1.0
        self.last_suppress_avoidance_time = self.get_clock().now()

        # plant_mission_node publishes here for the last stretch of a drive
        # up to a plant, and for the whole arm scan. Unlike the goto signal
        # above, this one is honoured while Nav2 is the command source: a
        # Nav2 approach to a plant is the one case where the thing filling
        # the depth image is the goal, not a hazard. Without it the layer
        # reads the plant as an obstacle at safe_distance (0.9m) and steers
        # the robot away from the standoff it was sent to (0.4m), so the
        # goal is never reached. Same 1s fail-safe as the others: if the
        # mission node dies, full avoidance comes back on its own.
        self.mission_suppress_sub = self.create_subscription(
            Bool, mission_suppress_topic, self.mission_suppress_cb, 10)
        self.mission_suppress = False
        self.mission_suppress_timeout = 1.0
        self.last_mission_suppress_time = self.get_clock().now()

        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_cb, 10)

        self.tof_sub = self.create_subscription(
            String, '/ecobot/tof_ranges', self.tof_cb, 10)
        self.latest_tof = None

        self.escape_state = 'NONE'
        self.escape_start_time = None
        self.escape_turn_dir = 'LEFT'
        # Latched with hysteresis so a min_dist hovering right at
        # safe_distance doesn't flip FORWARD/TURN every 100ms tick.
        self._danger_active = False

        self.get_logger().info(
            f'obstacle avoidance started — depth_topic={depth_topic} '
            f'safe<{self.safe_distance}m warn<{self.warn_distance}m '
            f'tof<{self.tof_threshold}m')

        self.timer = self.create_timer(1.0 / 10.0, self.avoidance_loop)

    def nav_vel_cb(self, msg):
        self.latest_nav_vel = msg
        self.has_nav_vel = True
        self.last_nav_vel_time = self.get_clock().now()

    def goto_vel_cb(self, msg):
        self.latest_goto_vel = msg
        self.has_goto_vel = True
        self.last_goto_vel_time = self.get_clock().now()

    def suppress_avoidance_cb(self, msg):
        self.suppress_avoidance = bool(msg.data)
        self.last_suppress_avoidance_time = self.get_clock().now()

    def mission_suppress_cb(self, msg):
        self.mission_suppress = bool(msg.data)
        self.last_mission_suppress_time = self.get_clock().now()

    def depth_cb(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = depth_image.copy()
        except Exception as e:
            self.get_logger().warn(f'depth callback error: {e}')

    def tof_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.latest_tof = data.get('ranges_m')
        except Exception:
            pass

    def zone_distances(self, depth_image):
        h, w = depth_image.shape
        zw = w // self.num_zones
        zones = []
        for i in range(self.num_zones):
            zone = depth_image[h // 3:2 * h // 3, i * zw:(i + 1) * zw]
            valid = zone[(zone > 0) & (zone < 5000)]
            zones.append(
                np.mean(valid) * self.depth_scale if len(valid) > 50 else 99.0)
        return zones

    def speed_ratio(self, min_dist):
        if min_dist >= self.warn_distance:
            return 1.0
        if min_dist <= 0.2:
            return 0.05
        return (min_dist - 0.2) / (self.warn_distance - 0.2)

    def avoidance_loop(self):
        nav2_expired = (
            self.get_clock().now() - self.last_nav_vel_time
        ).nanoseconds / 1e9 > self.nav_vel_timeout
        nav2_active = self.has_nav_vel and not nav2_expired

        goto_expired = (
            self.get_clock().now() - self.last_goto_vel_time
        ).nanoseconds / 1e9 > self.goto_vel_timeout
        goto_active = self.has_goto_vel and not goto_expired

        suppress_expired = (
            self.get_clock().now() - self.last_suppress_avoidance_time
        ).nanoseconds / 1e9 > self.suppress_avoidance_timeout
        mission_suppress_expired = (
            self.get_clock().now() - self.last_mission_suppress_time
        ).nanoseconds / 1e9 > self.mission_suppress_timeout
        # Two ways to stand down. The goto signal is for detection_goto's own
        # camera-driven approach, and stays barred while Nav2 is driving so a
        # stale flag can never disarm a Nav2 run. The mission signal is the
        # exception to that rule: plant_mission_node only raises it within
        # suppress_avoidance_radius_m of the plant it is approaching, and
        # while the arm is scanning one, so the long drive between plants
        # still keeps full avoidance.
        suppress_active = (
            (self.suppress_avoidance and not suppress_expired
             and goto_active and not nav2_active)
            or (self.mission_suppress and not mission_suppress_expired))

        # detection_goto is deliberately driving toward (or parked in front
        # of) the very thing depth sees as "blocked" — the plant it means to
        # reach. If suppression is active, any in-flight REVERSE/TURN escape
        # from a prior all_blocked snapshot must be aborted immediately so it
        # doesn't keep rotating the robot away from the approach target while
        # the approach is actively progressing.
        if suppress_active and self.escape_state != 'NONE':
            self.escape_state = 'NONE'
            self.escape_start_time = None
            self.get_logger().info(
                'escape aborted — auto-track approach in progress')

        with self.depth_lock:
            depth_available = self.latest_depth is not None
            if depth_available:
                depth_image = self.latest_depth.copy()

        cmd_source_active = nav2_active or goto_active
        if goto_active and not nav2_active:
            base_cmd = self.latest_goto_vel
        else:
            base_cmd = self.latest_nav_vel if nav2_active else Twist()

        zones = None
        h, w = 480, 640

        if not depth_available:
            tof = self.latest_tof
            if tof is not None and any(v is not None for v in tof):
                zones = [99.0] * self.num_zones
                if tof[0] is not None and tof[0] < self.tof_threshold:
                    for i in range(self.num_zones // 2):
                        zones[i] = tof[0]
                if tof[1] is not None and tof[1] < self.tof_threshold:
                    for i in range(self.num_zones // 2 + 1, self.num_zones):
                        zones[i] = tof[1]
                depth_image = np.zeros((h, w), dtype=np.float32)
            else:
                twist = Twist()
                if cmd_source_active:
                    twist = base_cmd
                else:
                    twist.linear.x = self.max_linear * 0.3
                self.cmd_pub.publish(twist)
                return
        else:
            zones = self.zone_distances(depth_image)
            h, w = depth_image.shape
            tof = self.latest_tof
            if tof is not None:
                if tof[0] is not None and tof[0] < self.tof_threshold:
                    for i in range(self.num_zones // 2):
                        zones[i] = min(zones[i], tof[0])
                if tof[1] is not None and tof[1] < self.tof_threshold:
                    for i in range(self.num_zones // 2 + 1, self.num_zones):
                        zones[i] = min(zones[i], tof[1])
        mid = self.num_zones // 2
        min_dist = min(zones)
        all_blocked = all(d < self.warn_distance for d in zones)
        left_avg = np.mean(zones[:mid]) if zones[:mid] else 99.0
        right_avg = np.mean(zones[mid + 1:]) if zones[mid + 1:] else 99.0

        cmd = 'FORWARD'
        sr = self.speed_ratio(min_dist)

        if not suppress_active and self.escape_state == 'NONE' and all_blocked:
            self.escape_state = 'REVERSE'
            self.escape_start_time = self.get_clock().now()
            self.escape_turn_dir = 'RIGHT' if left_avg > right_avg else 'LEFT'
            self.get_logger().info(f'all blocked — reversing, then turning {self.escape_turn_dir}')

        if self.escape_state == 'REVERSE':
            elapsed = (self.get_clock().now() - self.escape_start_time).nanoseconds / 1e9
            twist = Twist()
            twist.linear.x = self.reverse_speed
            twist.angular.z = self.max_angular * 0.6 * (
                1.0 if self.escape_turn_dir == 'LEFT' else -1.0)
            cmd = f'REVERSE {self.escape_turn_dir}'
            if elapsed >= self.reverse_duration:
                self.escape_state = 'TURN'
                self.escape_start_time = self.get_clock().now()
        elif self.escape_state == 'TURN':
            elapsed = (self.get_clock().now() - self.escape_start_time).nanoseconds / 1e9
            twist = Twist()
            twist.angular.z = self.max_angular * (
                1.0 if self.escape_turn_dir == 'LEFT' else -1.0)
            twist.linear.x = self.max_linear * 0.2
            cmd = f'TURN {self.escape_turn_dir}'
            if elapsed >= self.turn_duration:
                self.escape_state = 'NONE'
        else:
            if suppress_active:
                # An auto-tracked target (e.g. a plant) is being
                # deliberately approached — proximity to it isn't a
                # hazard, it's the goal. Don't second-guess detection_goto.
                self._danger_active = False
            elif self._danger_active:
                # Stay in avoidance until well clear, not just barely clear,
                # so the mode doesn't chatter on noisy readings near the edge.
                self._danger_active = min_dist < (
                    self.safe_distance + self.avoid_hysteresis)
            else:
                self._danger_active = min_dist < self.safe_distance

            if self._danger_active:
                if min(zones[:mid]) < min(zones[mid + 1:]):
                    cmd = 'TURN RIGHT'
                else:
                    cmd = 'TURN LEFT'

            if cmd == 'FORWARD':
                if cmd_source_active:
                    twist = base_cmd
                    twist.linear.x *= sr
                else:
                    twist = Twist()
                    twist.linear.x = self.max_linear * 0.6 * sr
            elif cmd == 'TURN LEFT':
                if cmd_source_active:
                    twist = base_cmd
                    twist.angular.z = self.max_angular
                    twist.linear.x = max(twist.linear.x * 0.3 * sr, 0.02)
                else:
                    twist = Twist()
                    twist.angular.z = self.max_angular
                    twist.linear.x = self.max_linear * 0.3 * sr
            else:
                if cmd_source_active:
                    twist = base_cmd
                    twist.angular.z = -self.max_angular
                    twist.linear.x = max(twist.linear.x * 0.3 * sr, 0.02)
                else:
                    twist = Twist()
                    twist.angular.z = -self.max_angular
                    twist.linear.x = self.max_linear * 0.3 * sr

        self.cmd_pub.publish(twist)

        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)
        zw = w // self.num_zones
        for i in range(self.num_zones):
            x1, x2 = i * zw, (i + 1) * zw
            d = zones[i]
            if d < self.safe_distance:
                col = (0, 0, 255)
            elif d < self.warn_distance:
                col = (0, 255, 255)
            else:
                col = (0, 255, 0)
            cv2.rectangle(colored, (x1, 0), (x2 - 1, h - 1), col, 2)
            label = f'{d:.1f}m' if d < 10 else '---'
            cv2.putText(colored, label, (x1 + 4, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        cx, cy = w // 2, h // 2 + 30
        if 'REVERSE' in cmd:
            arrow_col = (0, 0, 255)
            cv2.arrowedLine(colored, (cx, cy - 40), (cx, cy + 40), arrow_col, 4, tipLength=0.3)
            dir_col = (0, 0, 255)
            dir_sign = 1 if 'RIGHT' in cmd else -1
            for r in range(20, 60, 15):
                pts = np.array([
                    [cx + int(dir_sign * r * np.cos(t)), cy + 40 + int(r * np.sin(t))]
                    for t in np.linspace(0, np.pi / 3, 12)
                ], np.int32).reshape((-1, 1, 2))
                cv2.polylines(colored, [pts], False, (0, 0, 255), 2)
        elif 'TURN' in cmd and self.escape_state == 'TURN':
            arrow_col = (0, 200, 255)
            dir_sign = 1 if 'RIGHT' in cmd else -1
            for r in range(20, 60, 15):
                pts = np.array([
                    [cx + int(dir_sign * r * np.sin(t)), cy - int(r * np.cos(t))]
                    for t in np.linspace(0, np.pi / 2, 14)
                ], np.int32).reshape((-1, 1, 2))
                cv2.polylines(colored, [pts], False, (0, 200, 255), 2)
            cv2.arrowedLine(colored, (cx, cy), (cx + dir_sign * 50, cy - 30), (0, 200, 255), 4, tipLength=0.3)
        elif cmd == 'FORWARD':
            arrow_col = (0, int(255 * sr), 0)
            cv2.arrowedLine(colored, (cx, cy + 40), (cx, cy - 40), arrow_col, 4, tipLength=0.3)
        elif cmd == 'TURN LEFT':
            arrow_col = (0, 200, 255)
            cv2.ellipse(colored, (cx, cy), (50, 50), 0, 0, -90, arrow_col, 3)
            cv2.arrowedLine(colored, (cx, cy - 50), (cx - 30, cy - 50), arrow_col, 3, tipLength=0.3)
        elif cmd == 'TURN RIGHT':
            arrow_col = (0, 200, 255)
            cv2.ellipse(colored, (cx, cy), (50, 50), 0, 0, 90, arrow_col, 3)
            cv2.arrowedLine(colored, (cx, cy - 50), (cx + 30, cy - 50), arrow_col, 3, tipLength=0.3)

        cmd_label = cmd
        if self.escape_state == 'NONE' and cmd == 'FORWARD':
            cmd_label = f'FORWARD {min_dist:.1f}m'
        elif cmd.startswith('TURN') and self.escape_state == 'NONE':
            cmd_label = f'{cmd} {min_dist:.1f}m'

        cmd_colors = {
            'FORWARD': (0, 255, 0),
            'TURN LEFT': (0, 200, 255),
            'TURN RIGHT': (0, 200, 255),
        }
        for k in cmd_colors:
            if cmd_label.startswith(k):
                cc = cmd_colors[k]
                break
        else:
            cc = (255, 255, 255)
        cv2.putText(colored, cmd_label, (w // 2 - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, cc, 3)

        _, jpg = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with ObstacleMJPEGHandler.frame_lock:
            ObstacleMJPEGHandler.frame = jpg.tobytes()
        if self.show_viewer:
            cv2.imshow('ecobot — Obstacle Avoidance', colored)
            cv2.waitKey(1)

    def destroy_node(self):
        self.mjpeg_server.shutdown()
        self.mjpeg_server.server_close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
