import os
import io
import json
import socket
import threading
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        if hasattr(socket, 'SO_REUSEPORT'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()


class MjpegHandler(BaseHTTPRequestHandler):
    frame = None
    frame_lock = threading.Lock()

    def do_GET(self):
        if self.path.startswith('/stream.mjpg'):
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=--frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            last_send = 0.0
            interval = 1.0 / 15.0

            while rclpy.ok():
                now = time.time()
                if now - last_send < interval:
                    time.sleep(0.01)
                    continue

                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.01)
                        continue
                    jpg = self.frame

                header = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpg)).encode() +
                    b'\r\n\r\n'
                )
                try:
                    self.wfile.write(header + jpg + b'\r\n')
                    last_send = now
                except Exception:
                    break
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class ArmCameraServer(Node):
    """Arm camera streaming server and MP4/AVI scan video recorder."""

    def __init__(self):
        super().__init__('arm_camera_server')

        self.declare_parameter('port', 8084)
        self.declare_parameter('quality', 45)
        self.declare_parameter('topic', '/arm/camera/image_raw')
        self.declare_parameter('record_topic', '/arm/record_cmd')
        self.declare_parameter('output_dir', '/tmp/ecobot_recordings')

        port = int(self.get_parameter('port').value)
        self._quality = int(self.get_parameter('quality').value)
        topic = str(self.get_parameter('topic').value)
        record_topic = str(self.get_parameter('record_topic').value)
        self._output_dir = str(self.get_parameter('output_dir').value)

        os.makedirs(self._output_dir, exist_ok=True)

        self._bridge = CvBridge()
        MjpegHandler.frame = None

        self._video_writer = None
        self._recording = False
        self._record_lock = threading.Lock()
        self._recording_filename = ""

        self._sub = self.create_subscription(
            Image, topic, self._image_cb, 10)
        self._record_sub = self.create_subscription(
            String, record_topic, self._record_cmd_cb, 10)

        self._server = ReusableHTTPServer(('', port), MjpegHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f'Arm camera server active on http://0.0.0.0:{port}/stream.mjpg (record topic: {record_topic})')

    def _record_cmd_cb(self, msg):
        try:
            data = json.loads(msg.data)
            action = data.get('action', '').lower()
            if action == 'start':
                filename = data.get('filename') or f"arm_scan_{int(time.time())}.mp4"
                filepath = os.path.join(self._output_dir, filename)
                self.start_recording(filepath)
            elif action == 'stop':
                self.stop_recording()
        except Exception as e:
            self.get_logger().warn(f'record_cmd parse error: {e}')

    def start_recording(self, filepath):
        with self._record_lock:
            if self._recording:
                self.get_logger().info(f'Already recording to {self._recording_filename}')
                return
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._video_writer = cv2.VideoWriter(filepath, fourcc, 15.0, (640, 480))
                self._recording = True
                self._recording_filename = filepath
                self.get_logger().info(f'Started video recording to {filepath}')
            except Exception as e:
                self.get_logger().error(f'Failed to start video recording: {e}')
                self._recording = False

    def stop_recording(self):
        with self._record_lock:
            if not self._recording:
                return
            try:
                if self._video_writer:
                    self._video_writer.release()
                    self._video_writer = None
                self._recording = False
                self.get_logger().info(f'Stopped video recording. Video saved to {self._recording_filename}')
            except Exception as e:
                self.get_logger().error(f'Error stopping video recording: {e}')

    def _image_cb(self, msg):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # If video recording is active, write frame
            with self._record_lock:
                if self._recording and self._video_writer:
                    h, w = cv_img.shape[:2]
                    if (w, h) != (640, 480):
                        resized = cv2.resize(cv_img, (640, 480))
                        self._video_writer.write(resized)
                    else:
                        self._video_writer.write(cv_img)

            # MJPEG stream frame
            _, jpg = cv2.imencode(
                '.jpg', cv_img,
                [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            with MjpegHandler.frame_lock:
                MjpegHandler.frame = jpg.tobytes()
        except Exception:
            pass

    def destroy_node(self):
        self.stop_recording()
        self._server.shutdown()
        self._server.server_close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmCameraServer()
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
