import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseFeed(Node):
    def __init__(self):
        super().__init__('realsense_feed')

        self.declare_parameter('color_width', 640)
        self.declare_parameter('color_height', 480)
        self.declare_parameter('color_fps', 30)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        self.declare_parameter('depth_fps', 30)
        self.declare_parameter('show_viewer', True)

        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        self.color_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.aligned_depth_pub = self.create_publisher(
            Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.show_viewer = self.get_parameter('show_viewer').value
        self.pipeline = None
        self.running = False
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.align = None

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.color,
                self.get_parameter('color_width').value,
                self.get_parameter('color_height').value,
                rs.format.bgr8,
                self.get_parameter('color_fps').value,
            )
            config.enable_stream(
                rs.stream.depth,
                self.get_parameter('depth_width').value,
                self.get_parameter('depth_height').value,
                rs.format.z16,
                self.get_parameter('depth_fps').value,
            )
            self.pipeline.start(config)
            self.running = True
            self.colorizer = rs.colorizer()
            self.align = rs.align(rs.stream.color)
            profile = self.pipeline.get_active_profile()
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            self.get_logger().info(
                f'RealSense D415 feed started — depth intrinsics: '
                f'fx={self.depth_intrinsics.fx:.2f} fy={self.depth_intrinsics.fy:.2f} '
                f'cx={self.depth_intrinsics.ppx:.2f} cy={self.depth_intrinsics.ppy:.2f} | '
                f'color intrinsics: fx={self.color_intrinsics.fx:.2f} fy={self.color_intrinsics.fy:.2f} '
                f'cx={self.color_intrinsics.ppx:.2f} cy={self.color_intrinsics.ppy:.2f}')
            self.timer = self.create_timer(1.0 / 30.0, self.publish_frames)
        except Exception as e:
            self.pipeline = None
            self.get_logger().error(
                f'RealSense D415 not available ({e}) — camera will report offline on '
                f'/ecobot/hardware_status, no synthetic data will be published')

    def publish_frames(self):
        if not self.running:
            return

        if self.pipeline is None:
            return
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        # --- Raw color (frame: camera_color_optical_frame) ---
        color_image = np.asanyarray(color_frame.get_data())
        now = self.get_clock().now().to_msg()

        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        color_msg.header.stamp = now
        color_msg.header.frame_id = 'camera_color_optical_frame'
        self.color_pub.publish(color_msg)

        # --- Raw depth (frame: camera_depth_optical_frame) ---
        raw_depth_image = np.asanyarray(depth_frame.get_data())
        depth_msg = self.bridge.cv2_to_imgmsg(raw_depth_image, encoding='16UC1')
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = 'camera_depth_optical_frame'
        self.depth_pub.publish(depth_msg)

        # --- Depth camera_info ---
        if self.depth_intrinsics is not None:
            info = CameraInfo()
            info.header.stamp = now
            info.header.frame_id = 'camera_depth_optical_frame'
            info.width = self.depth_intrinsics.width
            info.height = self.depth_intrinsics.height
            info.distortion_model = 'plumb_bob'
            info.k = [
                self.depth_intrinsics.fx, 0.0, self.depth_intrinsics.ppx,
                0.0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy,
                0.0, 0.0, 1.0,
            ]
            info.d = list(self.depth_intrinsics.coeffs[:5]) if self.depth_intrinsics.model != rs.distortion.none else [0.0]*5
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [
                self.depth_intrinsics.fx, 0.0, self.depth_intrinsics.ppx, 0.0,
                0.0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
            self.depth_info_pub.publish(info)

        # --- Aligned depth (frame: camera_color_optical_frame) ---
        if self.align is not None:
            try:
                aligned_frames = self.align.process(frames)
                aligned_depth = aligned_frames.get_depth_frame()
                if aligned_depth:
                    aligned_depth_img = np.asanyarray(aligned_depth.get_data())
                    aligned_msg = self.bridge.cv2_to_imgmsg(
                        aligned_depth_img, encoding='16UC1')
                    aligned_msg.header.stamp = now
                    aligned_msg.header.frame_id = 'camera_color_optical_frame'
                    self.aligned_depth_pub.publish(aligned_msg)
            except RuntimeError as e:
                self.get_logger().warn(f'align failed: {e}')

        # --- Color camera_info ---
        if self.color_intrinsics is not None:
            cinfo = CameraInfo()
            cinfo.header.stamp = now
            cinfo.header.frame_id = 'camera_color_optical_frame'
            cinfo.width = self.color_intrinsics.width
            cinfo.height = self.color_intrinsics.height
            cinfo.distortion_model = 'plumb_bob'
            cinfo.k = [
                self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx,
                0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy,
                0.0, 0.0, 1.0,
            ]
            cinfo.d = list(self.color_intrinsics.coeffs[:5]) if self.color_intrinsics.model != rs.distortion.none else [0.0]*5
            cinfo.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            cinfo.p = [
                self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx, 0.0,
                0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
            self.color_info_pub.publish(cinfo)

        # --- Viewer ---
        if self.show_viewer:
            depth_colored = np.asanyarray(
                self.colorizer.colorize(depth_frame).get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_colored, alpha=0.03),
                cv2.COLORMAP_JET)
            display = np.hstack((color_image, depth_colormap))
            cv2.imshow('ecobot — Color | Depth', display)
            cv2.waitKey(1)

    def destroy_node(self):
        self.running = False
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseFeed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
