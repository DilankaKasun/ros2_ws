import io
import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler


class ObstacleMJPEGHandler(BaseHTTPRequestHandler):
    server_node = None
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

        self.declare_parameter('safe_distance', 0.8)
        self.declare_parameter('warn_distance', 1.2)
        self.declare_parameter('zones', 5)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('show_viewer', False)
        self.declare_parameter('mjpeg_port', 8083)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_scale', 0.001)

        self.safe_distance = self.get_parameter('safe_distance').value
        self.warn_distance = self.get_parameter('warn_distance').value
        self.num_zones = self.get_parameter('zones').value
        self.max_linear = self.get_parameter('max_linear_speed').value
        self.max_angular = self.get_parameter('max_angular_speed').value
        self.show_viewer = self.get_parameter('show_viewer').value
        mjpeg_port = self.get_parameter('mjpeg_port').value
        depth_topic = self.get_parameter('depth_topic').value
        self.depth_scale = self.get_parameter('depth_scale').value

        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_lock = threading.Lock()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        ObstacleMJPEGHandler.frame = None
        self.mjpeg_server = HTTPServer(('', mjpeg_port), ObstacleMJPEGHandler)
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

        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_cb, 10)

        self.get_logger().info(
            f'obstacle avoidance started — depth_topic={depth_topic} '
            f'safe<{self.safe_distance}m warn<{self.warn_distance}m')

        self.timer = self.create_timer(1.0 / 10.0, self.avoidance_loop)

    def nav_vel_cb(self, msg):
        self.latest_nav_vel = msg
        self.has_nav_vel = True
        self.last_nav_vel_time = self.get_clock().now()

    def depth_cb(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = depth_image.copy()
        except Exception as e:
            self.get_logger().warn(f'depth callback error: {e}')

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

    def nav_cmd(self, zones):
        mid = self.num_zones // 2
        c = zones[mid]
        l = min(zones[:mid]) if zones[:mid] else 99.0
        r = min(zones[mid + 1:]) if zones[mid + 1:] else 99.0
        if c < self.safe_distance:
            return 'TURN LEFT' if l > r else 'TURN RIGHT'
        if l < self.safe_distance:
            return 'TURN RIGHT'
        if r < self.safe_distance:
            return 'TURN LEFT'
        return 'FORWARD'

    def avoidance_loop(self):
        nav2_expired = (
            self.get_clock().now() - self.last_nav_vel_time
        ).nanoseconds / 1e9 > self.nav_vel_timeout
        nav2_active = self.has_nav_vel and not nav2_expired

        with self.depth_lock:
            depth_available = self.latest_depth is not None
            if depth_available:
                depth_image = self.latest_depth.copy()

        cmd = 'FORWARD'
        zones = None

        if depth_available:
            zones = self.zone_distances(depth_image)
            cmd = self.nav_cmd(zones)

        twist = Twist()

        if cmd == 'FORWARD':
            if nav2_active:
                twist = self.latest_nav_vel
            elif depth_available:
                twist.linear.x = self.max_linear * 0.6
            else:
                twist.linear.x = self.max_linear * 0.3
        elif cmd == 'TURN LEFT':
            if nav2_active:
                twist = self.latest_nav_vel
                twist.angular.z = self.max_angular
                twist.linear.x = max(twist.linear.x * 0.3, 0.05)
            else:
                twist.angular.z = self.max_angular
                twist.linear.x = self.max_linear * 0.3
        elif cmd == 'TURN RIGHT':
            if nav2_active:
                twist = self.latest_nav_vel
                twist.angular.z = -self.max_angular
                twist.linear.x = max(twist.linear.x * 0.3, 0.05)
            else:
                twist.angular.z = -self.max_angular
                twist.linear.x = self.max_linear * 0.3

        self.cmd_pub.publish(twist)

        if not depth_available:
            return

        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)
        h, w = colored.shape[:2]
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
        cmd_colors = {
            'FORWARD': (0, 255, 0),
            'TURN LEFT': (0, 200, 255),
            'TURN RIGHT': (0, 200, 255),
        }
        cc = cmd_colors.get(cmd, (255, 255, 255))
        cv2.putText(colored, cmd, (w // 2 - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, cc, 3)
        _, jpg = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with ObstacleMJPEGHandler.frame_lock:
            ObstacleMJPEGHandler.frame = jpg.tobytes()
        if self.show_viewer:
            cv2.imshow('ecobot — Obstacle Avoidance', colored)
            cv2.waitKey(1)

    def destroy_node(self):
        self.mjpeg_server.shutdown()
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
