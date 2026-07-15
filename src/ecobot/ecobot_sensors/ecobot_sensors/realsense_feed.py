import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
        self.show_viewer = self.get_parameter('show_viewer').value
        self.pipeline = None
        self.running = False

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
            self.get_logger().info('RealSense D415 feed started')
            self.timer = self.create_timer(1.0 / 30.0, self.publish_frames)
        except RuntimeError as e:
            self.get_logger().error(f'failed to start RealSense: {e}')

    def publish_frames(self):
        if not self.running or self.pipeline is None:
            return
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())

        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        color_msg.header.stamp = self.get_clock().now().to_msg()
        color_msg.header.frame_id = 'camera_color_optical_frame'
        self.color_pub.publish(color_msg)

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.header.frame_id = 'camera_depth_optical_frame'
        self.depth_pub.publish(depth_msg)

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
