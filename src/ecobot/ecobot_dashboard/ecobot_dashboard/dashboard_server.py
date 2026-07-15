import os
import threading
import asyncio
import json
import base64
from typing import Any, Dict, Optional, List
import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8
from cv_bridge import CvBridge
import cv2
from aiohttp import web


class DashboardServer(Node):
    def __init__(self) -> None:
        super().__init__('dashboard_server')

        self.declare_parameter('port', 8080)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('num_zones', 5)
        self.declare_parameter('safe_distance', 0.8)
        self.declare_parameter('warn_distance', 1.2)
        self.declare_parameter('depth_scale', 0.001)

        try:
            www_dir = os.path.join(
                get_package_share_directory('ecobot_dashboard'), 'www')
        except Exception:
            www_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'www')

        self._www_dir: str = www_dir
        self._port: int = int(self.get_parameter('port').value)  # type: ignore[arg-type]
        self._depth_topic: str = str(self.get_parameter('depth_topic').value)  # type: ignore[arg-type]
        self._camera_topic: str = str(self.get_parameter('camera_topic').value)  # type: ignore[arg-type]
        self._num_zones: int = int(self.get_parameter('num_zones').value)  # type: ignore[arg-type]
        self._safe_distance: float = float(self.get_parameter('safe_distance').value)  # type: ignore[arg-type]
        self._warn_distance: float = float(self.get_parameter('warn_distance').value)  # type: ignore[arg-type]
        self._depth_scale: float = float(self.get_parameter('depth_scale').value)  # type: ignore[arg-type]

        self._bridge = CvBridge()
        self._ws_clients: set[web.WebSocketResponse] = set()
        self._lock = threading.Lock()

        self._latest: Dict[str, Any] = {
            'camera': None,
            'depth': None,
            'odom': None,
            'mode': None,
        }

        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self._mode_sub = self.create_subscription(
            UInt8, '/run_mode', self._mode_cb, 10)
        self._camera_sub = self.create_subscription(
            Image, self._camera_topic, self._camera_cb, 10)
        self._depth_sub = self.create_subscription(
            Image, self._depth_topic, self._depth_cb, 10)

        self._timer = self.create_timer(1.0 / 10.0, self._broadcast_timer)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_aiohttp, daemon=True)
        self._thread.start()

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (o.w * o.z), 1.0 - 2.0 * (o.z * o.z))
        with self._lock:
            self._latest['odom'] = {
                'x': p.x, 'y': p.y, 'yaw': yaw,
                'linear': msg.twist.twist.linear.x,
                'angular': msg.twist.twist.angular.z,
            }

    def _mode_cb(self, msg):
        with self._lock:
            self._latest['mode'] = msg.data

    def _camera_cb(self, msg):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpg = cv2.imencode('.jpg', cv_img,
                                  [cv2.IMWRITE_JPEG_QUALITY, 60])
            b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
            with self._lock:
                self._latest['camera'] = b64
        except Exception:
            pass

    def _depth_cb(self, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._lock:
                self._latest['depth'] = depth.copy()
        except Exception:
            pass

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
            depth = self._latest['depth']
            odom = self._latest['odom']
            mode = self._latest['mode']

        if not self._ws_clients:
            return

        messages = []

        if odom is not None:
            messages.append(json.dumps({'type': 'odom', 'data': odom}))

        if mode is not None:
            messages.append(json.dumps({'type': 'mode', 'data': {'mode': mode}}))

        if camera is not None:
            messages.append(json.dumps({'type': 'camera', 'data': camera}))

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

        if not messages:
            return

        payload = '\n'.join(messages)
        stale = set()
        for ws in self._ws_clients:
            try:
                ws.send_str(payload)
            except Exception:
                stale.add(ws)
        self._ws_clients -= stale

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.add(ws)
        async for _ in ws:
            pass
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

    def _run_aiohttp(self):
        app = web.Application()
        app.router.add_get('/ws', self._ws_handler)
        app.router.add_get('/{path:.*}', self._index_handler)
        runner = web.AppRunner(app)
        self._loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '', self._port)
        self._loop.run_until_complete(site.start())
        self.get_logger().info(
            f'dashboard server on http://0.0.0.0:{self._port}')
        self._loop.run_forever()

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
