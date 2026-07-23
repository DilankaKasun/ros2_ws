import os
import struct
import io
import threading
import asyncio
import json
import base64
from typing import Any, Dict, Optional, List
import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8, String
from cv_bridge import CvBridge
import cv2
from aiohttp import web, WSMsgType


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
        }

        self._latest_cloud_np: Optional[np.ndarray] = None

        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self._mode_sub = self.create_subscription(
            UInt8, '/run_mode', self._mode_cb, 10)
        self._camera_sub = self.create_subscription(
            Image, self._camera_topic, self._camera_cb, 10)
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

        self._cloud_sub = self.create_subscription(
            PointCloud2, '/rtabmap/cloud_map', self._cloud_cb, 10)

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

    def _detection_cb(self, msg):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpg = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
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
        except Exception:
            pass

    def _goto_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._latest['goto_status'] = data
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
            with self._lock:
                self._latest['tof'] = data
        except Exception:
            pass

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
        else:
            return
        out = String()
        out.data = json.dumps(payload)
        self._goto_target_pub.publish(out)

    def _cloud_cb(self, msg):
        try:
            points = list(point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True))
            if not points:
                return
            cloud = np.zeros((len(points), 6), dtype=np.float32)
            for i, p in enumerate(points):
                rgb_int = int(p[3])
                r = (rgb_int >> 16) & 0xFF
                g = (rgb_int >> 8) & 0xFF
                b = rgb_int & 0xFF
                cloud[i] = [p[0], p[1], p[2], r / 255.0, g / 255.0, b / 255.0]
            max_pts = int(self.get_parameter('map3d_max_points').value)
            if len(cloud) > max_pts:
                idx = np.random.choice(len(cloud), max_pts, replace=False)
                cloud = cloud[idx]
            with self._cloud_lock:
                self._latest_cloud_np = cloud
        except Exception as e:
            self.get_logger().warn(f'cloud callback: {e}')

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

        if tof is not None:
            messages.append(json.dumps({'type': 'tof', 'data': tof}))

        if camera is not None:
            messages.append(json.dumps({'type': 'camera', 'data': camera}))

        if detection is not None:
            messages.append(json.dumps(
                {'type': 'detection_image', 'data': detection}))

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

    async def _map3d_handler(self, request):
        with self._cloud_lock:
            cloud = self._latest_cloud_np
        if cloud is None or len(cloud) == 0:
            return web.Response(body=b'', content_type='application/octet-stream')
        buf = io.BytesIO()
        buf.write(struct.pack('<I', len(cloud)))
        buf.write(cloud.tobytes())
        return web.Response(body=buf.getvalue(), content_type='application/octet-stream')

    def _run_aiohttp(self):
        app = web.Application()
        app.router.add_get('/ws', self._ws_handler)
        app.router.add_get('/map3d', self._map3d_handler)
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
