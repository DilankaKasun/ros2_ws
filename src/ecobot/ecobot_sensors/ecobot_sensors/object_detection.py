import os
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

COCO_NAMES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

COLORS = np.random.randint(0, 255, (len(COCO_NAMES), 3), dtype=np.uint8)


class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')

        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('inference_rate', 4)
        self.declare_parameter('model_path', '')

        conf = self.get_parameter('conf_threshold').value
        self.conf_threshold = float(conf)
        self.inference_rate = int(self.get_parameter('inference_rate').value)
        self.frame_count = 0
        self.latest_depth = None
        self.depth_h = 0
        self.depth_w = 0

        model_path = str(self.get_parameter('model_path').value)
        if not model_path:
            model_path = os.path.join(
                get_package_share_directory('ecobot_sensors'),
                'models', 'ssd_mobilenet.caffemodel')
        prototxt_path = os.path.join(
            get_package_share_directory('ecobot_sensors'),
            'models', 'ssd_mobilenet.prototxt')

        if not os.path.exists(model_path) or not os.path.exists(prototxt_path):
            self.get_logger().error(f'Model not found: {model_path}')
            self.net = None
        else:
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            size_mb = os.path.getsize(model_path) // 1024 // 1024
            self.get_logger().info(f'Detection model loaded ({size_mb}MB)')

        self.bridge = CvBridge()
        self.overlay_pub = self.create_publisher(Image, '/detection_overlay', 10)
        self.detections_pub = self.create_publisher(String, '/detections', 10)
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_cb, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_cb, 10)

    def depth_cb(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock() if hasattr(self, '_lock') else self._no_lock():
                self.latest_depth = depth.copy()
                self.depth_h, self.depth_w = depth.shape[:2]
        except Exception:
            pass

    def get_depth_at(self, cx, cy):
        if self.latest_depth is None:
            return None
        cy = int(round(cy))
        cx = int(round(cx))
        cy = max(0, min(cy, self.depth_h - 1))
        cx = max(0, min(cx, self.depth_w - 1))
        d = self.latest_depth[cy, cx]
        if d > 0 and d < 5000:
            return float(d) * 0.001
        roi = self.latest_depth[max(0, cy - 2):min(self.depth_h, cy + 3),
                                 max(0, cx - 2):min(self.depth_w, cx + 3)]
        valid = roi[(roi > 0) & (roi < 5000)]
        if len(valid) > 0:
            return float(np.median(valid)) * 0.001
        return None

    class _no_lock:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass

    def color_cb(self, msg):
        if self.net is None:
            return

        self.frame_count += 1
        if self.frame_count % self.inference_rate != 0:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        h, w = cv_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv_img, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        overlay = cv_img.copy()
        results = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.conf_threshold:
                continue

            class_id = int(detections[0, 0, i, 1])
            if class_id >= len(COCO_NAMES):
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = self.get_depth_at(cx, cy)

            label = f'{COCO_NAMES[class_id]} {confidence:.2f}'
            if dist is not None:
                label += f' {dist:.1f}m'

            color = tuple(int(c) for c in COLORS[class_id % len(COLORS)])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            results.append({
                'class': COCO_NAMES[class_id],
                'class_id': class_id,
                'confidence': round(confidence, 3),
                'distance': round(dist, 2) if dist is not None else None,
                'box': [int(x1), int(y1), int(x2), int(y2)],
            })

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)
        except Exception:
            pass

        if results:
            det_msg = String()
            det_msg.data = json.dumps(results)
            self.detections_pub.publish(det_msg)

    def _no_lock(self):
        return self.__class__._no_lock()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()
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
