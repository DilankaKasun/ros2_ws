import io
import struct
import threading
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler


class MjpegStreamHandler(BaseHTTPRequestHandler):
    server_node = None
    frame = None
    frame_lock = threading.Lock()

    def do_GET(self):
        if self.path.startswith('/stream.mjpg'):
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
            self.wfile.write(b'<html><body><img src="/stream.mjpg"/></body></html>')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class CameraWebServer(Node):
    def __init__(self):
        super().__init__('camera_web_server')

        self.declare_parameter('port', 8081)
        self.declare_parameter('quality', 70)
        self.declare_parameter('topic', '/camera/color/image_raw')

        port = self.get_parameter('port').value
        self.quality = self.get_parameter('quality').value
        topic = self.get_parameter('topic').value

        self.bridge = CvBridge()
        MjpegStreamHandler.frame = None

        self.sub = self.create_subscription(
            Image, topic, self.image_callback, 10)

        self.server = HTTPServer(('', port), MjpegStreamHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.get_logger().info(f'camera web server on http://0.0.0.0:{port}/stream.mjpg')

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpg = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            with MjpegStreamHandler.frame_lock:
                MjpegStreamHandler.frame = jpg.tobytes()
        except Exception:
            pass

    def destroy_node(self):
        self.server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraWebServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
