import glob
import os
import threading
import time
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


ARM_CAM_BY_ID = '/dev/v4l/by-id/usb-HRY_YDL_lens_USB_Camera_20210616_720-video-index0'


class USBCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')

        self.declare_parameter('device', ARM_CAM_BY_ID)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('topic', '/arm/camera/image_raw')

        device = self.get_parameter('device').value
        self._width = int(self.get_parameter('width').value)
        self._height = int(self.get_parameter('height').value)
        self._fps = int(self.get_parameter('fps').value)
        topic = str(self.get_parameter('topic').value)

        self._bridge = CvBridge()
        self._cap = self._open_camera(device)
        self._pub = self.create_publisher(Image, topic, 2)
        self._running = True

        if self._cap is not None:
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            self.get_logger().info(
                f'USB Arm camera capture thread started ({self._width}x{self._height}@{self._fps}fps)')
        else:
            self.get_logger().error('Could not initialize USB Arm camera.')

    def _configure_cap(self, cap):
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _open_camera(self, preferred):
        candidates = [preferred]
        if os.path.exists(ARM_CAM_BY_ID) and ARM_CAM_BY_ID not in candidates:
            candidates.insert(0, ARM_CAM_BY_ID)

        for dev in candidates:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap.isOpened():
                self._configure_cap(cap)
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.get_logger().info(f'Successfully opened ARM CAMERA device: {dev}')
                    return cap
                cap.release()

        self.get_logger().info('Scanning /dev/video* for USB Arm Camera...')
        scan_list = ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video4']
        all_devs = sorted(glob.glob('/dev/video*'))
        for dev in scan_list + [d for d in all_devs if d not in scan_list]:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap.isOpened():
                self._configure_cap(cap)
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.get_logger().info(f'Auto-selected camera device: {dev}')
                    return cap
                cap.release()
        return None

    def _capture_loop(self):
        target_dt = 1.0 / max(1, self._fps)
        while self._running and rclpy.ok():
            t0 = time.time()
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            try:
                msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'arm_camera_frame'
                self._pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Publish error: {e}')

            elapsed = time.time() - t0
            sleep_time = target_dt - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def destroy_node(self):
        self._running = False
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = USBCameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
