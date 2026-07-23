import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class GroundMJPEGHandler(BaseHTTPRequestHandler):
    frame = None
    frame_lock = threading.Lock()

    def do_GET(self):
        if self.path.startswith('/ground.mjpg'):
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
            self.wfile.write(b'<html><body><img src="/ground.mjpg"/></body></html>')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class DepthGroundDetection(Node):
    def __init__(self):
        super().__init__('depth_ground_detection')

        self.declare_parameter('camera_height', 0.508)
        self.declare_parameter('ground_clearance', 0.02)
        self.declare_parameter('max_obstacle_height', 0.50)
        self.declare_parameter('min_obstacle_height', 0.005)
        self.declare_parameter('max_range', 3.0)
        self.declare_parameter('min_range', 0.2)
        self.declare_parameter('downsample', 2)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('mjpeg_port', 8084)
        self.declare_parameter('depth_scale', 0.001)

        self.camera_height = self.get_parameter('camera_height').value
        self.ground_clearance = self.get_parameter('ground_clearance').value
        self.max_obstacle_height = self.get_parameter('max_obstacle_height').value
        self.min_obstacle_height = self.get_parameter('min_obstacle_height').value
        self.max_range = self.get_parameter('max_range').value
        self.min_range = self.get_parameter('min_range').value
        self.downsample = self.get_parameter('downsample').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        depth_topic = self.get_parameter('depth_topic').value
        cam_info_topic = self.get_parameter('camera_info_topic').value
        mjpeg_port = self.get_parameter('mjpeg_port').value
        self.depth_scale = self.get_parameter('depth_scale').value

        self.bridge = CvBridge()
        self.fx = 430.0
        self.fy = 430.0
        self.cx = 320.0
        self.cy = 240.0
        self.intrinsics_received = False
        self.overlay_image = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        GroundMJPEGHandler.frame = None
        self.mjpeg_server = HTTPServer(('', mjpeg_port), GroundMJPEGHandler)
        self.mjpeg_thread = threading.Thread(
            target=self.mjpeg_server.serve_forever, daemon=True)
        self.mjpeg_thread.start()
        self.get_logger().info(
            f'ground detection MJPEG on http://0.0.0.0:{mjpeg_port}/ground.mjpg')

        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_cb, 10)

        self.cam_info_sub = self.create_subscription(
            CameraInfo, cam_info_topic, self.cam_info_cb, 10)

        self.pc_pub = self.create_publisher(
            PointCloud2, '/ground_obstacle_points', 10)

        self.debug_pub = self.create_publisher(
            Image, '/ground_debug_overlay', 10)

        self.get_logger().info(
            f'depth ground detection started — camera_height={self.camera_height}m '
            f'clearance={self.ground_clearance}m '
            f'range={self.min_range}-{self.max_range}m '
            f'downsample={self.downsample}')

    def cam_info_cb(self, msg: CameraInfo):
        if not self.intrinsics_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsics_received = True
            self.get_logger().info(
                f'camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}')

    def project_obstacle_points(self, depth_image):
        h, w = depth_image.shape
        ds = self.downsample
        rows = np.arange(0, h, ds)
        cols = np.arange(0, w, ds)
        vv, uu = np.meshgrid(rows, cols, indexing='ij')
        depths = depth_image[vv, uu].astype(np.float32) * self.depth_scale

        valid = (depths > self.min_range) & (depths < self.max_range) & (depths > 0)
        if not valid.any():
            return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        uu_v = uu[valid].astype(np.float32)
        vv_v = vv[valid].astype(np.float32)
        dd = depths[valid]

        x_cam = (uu_v - self.cx) * dd / self.fx
        y_cam = (vv_v - self.cy) * dd / self.fy
        z_cam = dd

        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, Time())
        except Exception:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, self.camera_frame, Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            except Exception as e:
                self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
                return [], np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        tz = t.transform.translation.z
        q = t.transform.rotation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z

        n = len(x_cam)
        pts_cam = np.column_stack([x_cam, y_cam, z_cam, np.ones(n)])

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        T = np.array([tx, ty, tz])

        pts_base = (R @ pts_cam[:, :3].T).T + T

        z_base = pts_base[:, 2]
        dist = np.linalg.norm(pts_base, axis=1)
        obstacle_mask = (
            (z_base > self.min_obstacle_height) &
            (z_base < self.max_obstacle_height) &
            (dist > self.min_range) &
            (dist < self.max_range)
        )

        if not obstacle_mask.any():
            return [], x_cam, y_cam, z_cam

        return pts_base[obstacle_mask], x_cam, y_cam, z_cam

    def depth_cb(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'depth conversion error: {e}')
            return

        obstacle_pts, x_cam, y_cam, z_cam = self.project_obstacle_points(depth_image)

        if len(obstacle_pts) > 0:
            self.publish_pointcloud(obstacle_pts)

        h, w = depth_image.shape
        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)

        if len(obstacle_pts) > 0:
            for pt in obstacle_pts[::5]:
                depth_val = pt[2]
                if depth_val <= 0:
                    continue
                u = int((pt[0] / depth_val) * self.fx + self.cx)
                v = int((pt[1] / depth_val) * self.fy + self.cy)
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(colored, (u, v), 2, (0, 255, 0), -1)

            if len(obstacle_pts) > 0:
                min_z = np.min(obstacle_pts[:, 2])
                max_z = np.max(obstacle_pts[:, 2])
                count = len(obstacle_pts)
                cv2.putText(colored, f'obstacles: {count}  z:{min_z:.2f}-{max_z:.2f}m',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            cv2.putText(colored, 'no obstacles',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        _, jpg = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with GroundMJPEGHandler.frame_lock:
            GroundMJPEGHandler.frame = jpg.tobytes()

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(colored, encoding='bgr8')
            overlay_msg.header = msg.header
            self.debug_pub.publish(overlay_msg)
        except Exception:
            pass

    def publish_pointcloud(self, points):
        if len(points) == 0:
            return
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        self.pc_pub.publish(msg)

    def destroy_node(self):
        self.mjpeg_server.shutdown()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthGroundDetection()
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
