import asyncio
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import VideoFrame


class ROSVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self._frame = None
        self._lock = threading.Lock()

    def set_frame(self, ndarray_rgb):
        with self._lock:
            self._frame = ndarray_rgb

    async def recv(self):
        while True:
            with self._lock:
                if self._frame is not None:
                    frame = VideoFrame.from_ndarray(self._frame, format="rgb24")
                    now = time.time()
                    frame.pts = int(now * 90000)
                    frame.time_base = 1 / 90000
                    return frame
            await asyncio.sleep(0.01)


class WebRTCStreamer(Node):
    def __init__(self):
        super().__init__('webrtc_streamer')

        self.declare_parameter('signaling_port', 8082)

        self.bridge = CvBridge()
        self.track = ROSVideoTrack()
        self._pcs = set()

        self._sub = self.create_subscription(
            Image, '/camera/color/image_raw', self._image_callback, 10)

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        port = self.get_parameter('signaling_port').value
        self._server = HTTPServer(
            ('', port),
            lambda *a: _SignalingHandler(self, *a))
        self._server.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

        self.get_logger().info(
            f'WebRTC signaling on http://0.0.0.0:{port}')

    def _image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.track.set_frame(rgb)

    def _handle_offer_sync(self, sdp, sdp_type):
        future = asyncio.run_coroutine_threadsafe(
            self._handle_offer(sdp, sdp_type), self._loop)
        return future.result(timeout=10)

    async def _handle_offer(self, sdp, sdp_type):
        pc = RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state():
            self.get_logger().info(
                f'WebRTC state: {pc.connectionState}')
            if pc.connectionState in (
                    "failed", "closed", "disconnected"):
                self._pcs.discard(pc)

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp, sdp_type))

        pc.addTrack(self.track)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await asyncio.sleep(0.5)

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
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _write(self, data):
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if self.path == "/offer":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)
            try:
                result = self._streamer._handle_offer_sync(
                    data["sdp"], data["type"])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self._write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self._write(str(e).encode())

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
