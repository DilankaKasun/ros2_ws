import asyncio
import json
import math
import threading
import time
from urllib.parse import urlparse, parse_qs
from fractions import Fraction
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from std_msgs.msg import String, Float64MultiArray, UInt8
from cv_bridge import CvBridge
import cv2
import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import VideoFrame

STUN_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=[
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
        ])
    ]
)


def _create_placeholder(text: str):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (75, 226, 119), 2)
    return img


class ROSVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, name: str = "camera"):
        super().__init__()
        self.name = name
        self._frame = _create_placeholder(f"Initializing {name}...")
        self._lock = threading.Lock()
        self._last_pts = 0

    def set_frame(self, ndarray_rgb):
        with self._lock:
            self._frame = ndarray_rgb

    async def recv(self):
        while True:
            with self._lock:
                frame_data = self._frame
            if frame_data is not None:
                frame = VideoFrame.from_ndarray(frame_data, format="rgb24")
                now = time.time()
                frame.pts = int(now * 90000)
                frame.time_base = Fraction(1, 90000)
                return frame
            await asyncio.sleep(0.01)


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
        super().server_bind()


class WebRTCStreamer(Node):
    def __init__(self):
        super().__init__('webrtc_streamer')

        self.declare_parameter('signaling_port', 8082)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('detection_topic', '/ecobot/detection_image')
        self.declare_parameter('arm_camera_topic', '/arm/camera/image_raw')

        camera_topic = str(self.get_parameter('camera_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        arm_camera_topic = str(self.get_parameter('arm_camera_topic').value)

        self.bridge = CvBridge()
        self.color_track = ROSVideoTrack("Primary Camera")
        self.detection_track = ROSVideoTrack("Detection Camera")
        self.arm_track = ROSVideoTrack("Arm Camera")
        self._pcs = set()
        self._data_channels = set()
        self._dc_lock = threading.Lock()

        # Telemetry Cache
        self._latest_odom = None
        self._latest_tof = None
        self._latest_arm_pose = None
        self._latest_arm_status = None
        self._latest_mode = 1
        self._latest_detections = None
        self._latest_goto_status = None
        self._latest_waypoints = None
        self._latest_plant_scan = None
        self._latest_vla_status = None
        self._latest_actual_path = None
        self._latest_amcl_pose = None
        self._latest_map = None

        # Camera subscriptions
        self._color_sub = self.create_subscription(
            Image, camera_topic, self._color_cb, 10)
        self._det_sub = self.create_subscription(
            Image, detection_topic, self._det_cb, 10)
        self._arm_sub = self.create_subscription(
            Image, arm_camera_topic, self._arm_cb, 10)

        # Telemetry subscriptions
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self._mode_sub = self.create_subscription(
            UInt8, '/run_mode', self._mode_cb, 10)
        self._tof_sub = self.create_subscription(
            String, '/ecobot/tof_ranges', self._tof_cb, 10)
        self._arm_pose_sub = self.create_subscription(
            Float64MultiArray, '/arm/pose', self._arm_pose_cb, 10)
        self._arm_status_sub = self.create_subscription(
            String, '/arm/status', self._arm_status_cb, 10)
        self._detections_sub = self.create_subscription(
            String, '/ecobot/detections', self._detections_cb, 10)
        self._goto_status_sub = self.create_subscription(
            String, '/ecobot/goto_status', self._goto_status_cb, 10)
        self._waypoints_sub = self.create_subscription(
            String, '/ecobot/waypoints', self._waypoints_cb, 10)
        self._plant_scan_sub = self.create_subscription(
            String, '/ecobot/plant_scan_status', self._plant_scan_cb, 10)
        self._vla_status_sub = self.create_subscription(
            String, '/ecobot/vla_status', self._vla_status_cb, 10)
        self._actual_path_sub = self.create_subscription(
            Path, '/ecobot/actual_path', self._actual_path_cb, 10)
        self._amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_pose_cb, 10)
        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, 10)

        # Control publishers
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._arm_joint_pub = self.create_publisher(Float64MultiArray, '/arm/joint_commands', 10)
        self._arm_pose_pub = self.create_publisher(Float64MultiArray, '/arm/pose_goal', 10)
        self._vla_prompt_pub = self.create_publisher(String, '/ecobot/vla_prompt', 10)
        self._goto_target_pub = self.create_publisher(String, '/ecobot/goto_target', 10)
        self._plant_scan_pub = self.create_publisher(String, '/ecobot/plant_scan_cmd', 10)
        self._initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        # High-speed RTC Telemetry Broadcast at 25 Hz
        self._telemetry_timer = self.create_timer(1.0 / 25.0, self._broadcast_telemetry)

        port = int(self.get_parameter('signaling_port').value)
        self._server = ReusableHTTPServer(
            ('', port),
            lambda *a: _SignalingHandler(self, *a))
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

        self.get_logger().info(
            f'WebRTC full stack active on http://0.0.0.0:{port} (Video & Bidirectional DataChannels)')

    def _color_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.color_track.set_frame(rgb)
        except Exception:
            pass

    def _det_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.detection_track.set_frame(rgb)
        except Exception:
            pass

    def _arm_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.arm_track.set_frame(rgb)
        except Exception:
            pass

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = float(np.arctan2(2.0 * (o.w * o.z), 1.0 - 2.0 * (o.z * o.z)))
        self._latest_odom = {
            'x': round(float(p.x), 3), 'y': round(float(p.y), 3), 'yaw': yaw,
            'linear': round(float(msg.twist.twist.linear.x), 2),
            'angular': round(float(msg.twist.twist.angular.z), 2),
        }

    def _mode_cb(self, msg):
        self._latest_mode = int(msg.data)

    def _tof_cb(self, msg):
        try:
            self._latest_tof = json.loads(msg.data)
        except Exception:
            pass

    def _arm_pose_cb(self, msg):
        self._latest_arm_pose = [round(float(v), 2) for v in msg.data]

    def _arm_status_cb(self, msg):
        self._latest_arm_status = str(msg.data)

    def _detections_cb(self, msg):
        try:
            self._latest_detections = json.loads(msg.data)
        except Exception:
            pass

    def _goto_status_cb(self, msg):
        try:
            self._latest_goto_status = json.loads(msg.data)
        except Exception:
            pass

    def _waypoints_cb(self, msg):
        try:
            self._latest_waypoints = json.loads(msg.data)
        except Exception:
            pass

    def _plant_scan_cb(self, msg):
        try:
            self._latest_plant_scan = json.loads(msg.data)
        except Exception:
            pass

    def _vla_status_cb(self, msg):
        self._latest_vla_status = str(msg.data)

    def _actual_path_cb(self, msg):
        pts = []
        for pose in msg.poses:
            pts.append({'x': round(pose.pose.position.x, 2), 'y': round(pose.pose.position.y, 2)})
        self._latest_actual_path = pts

    def _amcl_pose_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = float(math.atan2(2.0 * (o.w * o.z), 1.0 - 2.0 * (o.z * o.z)))
        self._latest_amcl_pose = {
            'x': round(p.x, 2), 'y': round(p.y, 2), 'yaw': yaw,
            'covariance': list(msg.pose.covariance),
        }

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
            import base64
            b64 = base64.b64encode(png.tobytes()).decode('ascii')
            self._latest_map = {
                'image': b64,
                'resolution': info.resolution,
                'width': w, 'height': h,
                'origin_x': info.origin.position.x,
                'origin_y': info.origin.position.y,
            }
        except Exception:
            pass

    def _broadcast_telemetry(self):
        with self._dc_lock:
            if not self._data_channels:
                return
            channels = list(self._data_channels)

        payloads = []
        if self._latest_odom is not None:
            payloads.append(json.dumps({'type': 'odom', 'data': self._latest_odom}))
        payloads.append(json.dumps({'type': 'mode', 'data': {'mode': self._latest_mode}}))
        if self._latest_tof is not None:
            payloads.append(json.dumps({'type': 'tof', 'data': self._latest_tof}))
        if self._latest_arm_pose is not None:
            payloads.append(json.dumps({'type': 'arm_pose', 'data': self._latest_arm_pose}))
        if self._latest_arm_status is not None:
            payloads.append(json.dumps({'type': 'arm_status', 'data': self._latest_arm_status}))
        if self._latest_detections is not None:
            payloads.append(json.dumps({'type': 'detections', 'data': self._latest_detections}))
        if self._latest_goto_status is not None:
            payloads.append(json.dumps({'type': 'goto_status', 'data': self._latest_goto_status}))
        if self._latest_waypoints is not None:
            payloads.append(json.dumps({'type': 'waypoints', 'data': self._latest_waypoints}))
        if self._latest_plant_scan is not None:
            payloads.append(json.dumps({'type': 'plant_scan_status', 'data': self._latest_plant_scan}))
        if self._latest_vla_status is not None:
            payloads.append(json.dumps({'type': 'vla_status', 'data': self._latest_vla_status}))
        if self._latest_actual_path is not None:
            payloads.append(json.dumps({'type': 'actual_path', 'data': self._latest_actual_path}))
        if self._latest_amcl_pose is not None:
            payloads.append(json.dumps({'type': 'amcl_pose', 'data': self._latest_amcl_pose}))
        if self._latest_map is not None:
            payloads.append(json.dumps({'type': 'map', 'data': self._latest_map}))

        if not payloads:
            return

        msg_str = '\n'.join(payloads)
        for dc in channels:
            if dc.readyState == "open":
                try:
                    self._loop.call_soon_threadsafe(dc.send, msg_str)
                except Exception:
                    pass

    def _handle_dc_message(self, message_str: str):
        try:
            data = json.loads(message_str)
        except Exception:
            return
        mtype = data.get('type')
        if mtype == 'cmd_vel':
            twist = Twist()
            twist.linear.x = float(data.get('linear', 0.0))
            twist.angular.z = float(data.get('angular', 0.0))
            self._cmd_vel_pub.publish(twist)
        elif mtype == 'emergency_stop':
            twist = Twist()
            self._cmd_vel_pub.publish(twist)
            self.get_logger().warn("RTC Emergency Stop Executed!")
        elif mtype == 'arm_joints':
            angles = data.get('angles', [])
            if angles:
                arr = Float64MultiArray()
                arr.data = [float(a) for a in angles]
                self._arm_joint_pub.publish(arr)
        elif mtype == 'arm_pose_goal':
            arr = Float64MultiArray()
            arr.data = [float(data.get('x', 0.3)), float(data.get('y', 0.0)), float(data.get('z', 0.2))]
            self._arm_pose_pub.publish(arr)
        elif mtype == 'vla_prompt':
            prompt = data.get('prompt', '')
            if prompt:
                msg = String()
                msg.data = str(prompt)
                self._vla_prompt_pub.publish(msg)
        elif mtype in ('goto_select', 'goto_waypoint', 'goto_cancel', 'clear_waypoints'):
            out = String()
            out.data = json.dumps(data)
            self._goto_target_pub.publish(out)
        elif mtype and (mtype.startswith('plant_scan_') or mtype == 'set_plant_waypoints'):
            out = String()
            out.data = json.dumps(data)
            self._plant_scan_pub.publish(out)
        elif mtype == 'set_initial_pose':
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            msg.pose.pose.position.x = float(data.get('x', 0.0))
            msg.pose.pose.position.y = float(data.get('y', 0.0))
            yaw = float(data.get('yaw', 0.0))
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            self._initial_pose_pub.publish(msg)

    def _get_track(self, feed: str) -> VideoStreamTrack:
        feed = (feed or '').lower().strip()
        if 'arm' in feed:
            return self.arm_track
        elif 'det' in feed:
            return self.detection_track
        return self.color_track

    def _handle_offer_sync(self, sdp, sdp_type, feed="color"):
        future = asyncio.run_coroutine_threadsafe(
            self._handle_offer(sdp, sdp_type, feed), self._loop)
        return future.result(timeout=10)

    async def _handle_offer(self, sdp, sdp_type, feed="color"):
        pc = RTCPeerConnection(configuration=STUN_CONFIG)
        self._pcs.add(pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            with self._dc_lock:
                self._data_channels.add(channel)
            self.get_logger().info(f"WebRTC DataChannel connected: '{channel.label}'")

            @channel.on("message")
            def on_message(message):
                self._handle_dc_message(message)

            @channel.on("close")
            def on_close():
                with self._dc_lock:
                    self._data_channels.discard(channel)

        @pc.on("connectionstatechange")
        async def on_state():
            self.get_logger().info(
                f'WebRTC peer state changed: {pc.connectionState}')
            if pc.connectionState in ("failed", "closed", "disconnected"):
                self._pcs.discard(pc)

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp, sdp_type))

        selected_track = self._get_track(feed)
        pc.addTrack(selected_track)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await asyncio.sleep(0.2)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    def destroy_node(self):
        self._server.shutdown()
        self._loop.call_soon_threadsafe(self._loop.stop)
        super().destroy_node()


class _SignalingHandler(BaseHTTPRequestHandler):
    def __init__(self, streamer, *args):
        self._streamer = streamer
        super().__init__(*args)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _write(self, data):
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with self._streamer._dc_lock:
                dc_count = len(self._streamer._data_channels)
            status = {
                "status": "ok",
                "active_peers": len(self._streamer._pcs),
                "active_datachannels": dc_count,
                "feeds": ["color", "detection", "arm"],
            }
            self._write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/offer":
            qs = parse_qs(parsed.query)
            feed_param = qs.get("feed", ["color"])[0]

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                feed = data.get("feed", feed_param)
                result = self._streamer._handle_offer_sync(
                    data["sdp"], data["type"], feed=feed)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self._write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self._write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = WebRTCStreamer()
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
