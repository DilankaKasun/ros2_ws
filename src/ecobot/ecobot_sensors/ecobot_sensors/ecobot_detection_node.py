import os
import json
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False

COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
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

class TrtBackend:
    """Minimal YOLOv8 TensorRT runtime (fallback when ultralytics is missing)."""

    def __init__(self, engine_path, input_size):
        self.input_size = input_size
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('failed to deserialize engine')
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            host = cuda.pagelocked_empty(abs(trt.volume(shape)), trt.nptype(dtype))
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_host, self.input_device = host, device
            else:
                self.output_host, self.output_device = host, device

    def infer(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(w * scale), int(h * scale)
        dw, dh = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114,
                         dtype=np.float32)
        canvas[dh:dh + nh, dw:dw + nw] = \
            cv2.resize(img, (nw, nh)).astype(np.float32)
        blob = (canvas.transpose(2, 0, 1)[np.newaxis, ...] / 255.0
                ).astype(np.float32)
        np.copyto(self.input_host, blob.ravel())
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device,
                               self.stream)
        self.stream.synchronize()
        return self.output_host.copy(), scale, dw, dh


class EcobotDetectionNode(Node):
    def __init__(self):
        super().__init__('ecobot_detection_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('backend', 'auto')  # auto|ultralytics|tensorrt
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('inference_rate', 2)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('overlay_topic', '/ecobot/detection_image')
        self.declare_parameter('detections_topic', '/ecobot/detections')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        # How high the camera sits above the floor. Used to turn a
        # detection's vertical position into a height off the floor, which
        # is what the arm needs to aim at a plant. Must match the
        # base_footprint -> camera static transform in sensors.launch.py.
        self.declare_parameter('camera_height', 0.508)
        self.declare_parameter('detections_pointcloud_topic',
                               '/ecobot/detection_points')

        model_path = str(self.get_parameter('model_path').value)
        backend_want = str(self.get_parameter('backend').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.input_size = int(self.get_parameter('input_size').value)
        self.inference_rate = int(self.get_parameter('inference_rate').value)
        camera_topic = str(self.get_parameter('camera_topic').value)
        overlay_topic = str(self.get_parameter('overlay_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        pointcloud_topic = str(
            self.get_parameter('detections_pointcloud_topic').value)
        self.camera_height = float(self.get_parameter('camera_height').value)

        if not model_path:
            try:
                model_path = os.path.join(
                    get_package_share_directory('ecobot_sensors'),
                    'models', 'yolov8n.engine')
            except Exception:
                model_path = ''
        if not model_path or not os.path.exists(model_path):
            self.get_logger().warn(
                f'model not found: {model_path or "<unset>"} '
                f'— falling back to yolov8n.pt')
            model_path = 'yolov8n.pt'

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_header = None
        self.frame_lock = threading.Lock()
        self.tick_count = 0
        self.latest_depth = None
        self.depth_lock = threading.Lock()
        self.fx = 430.0
        self.fy = 430.0
        self.cx = 320.0
        self.cy = 240.0
        self.intrinsics_received = False

        self.backend = None
        self.model = None
        self.trt = None
        self._init_backend(backend_want, model_path)

        self.overlay_pub = self.create_publisher(Image, overlay_topic, 10)
        self.detections_pub = self.create_publisher(
            String, detections_topic, 10)
        self.color_sub = self.create_subscription(
            Image, camera_topic, self.color_cb, 10)
        # Depth and camera info are optional. Run against the wrist camera
        # there is neither, and the node is only asked WHETHER something is
        # in frame and WHERE in the frame — never how far away. An empty
        # topic name is not a legal subscription, so skip it rather than
        # crash on startup.
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_cb, 10) if depth_topic else None
        self.cam_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.cam_info_cb, 10) \
            if camera_info_topic else None
        if not depth_topic:
            self.get_logger().info(
                'no depth topic given — detections will carry no distance')
        self.pc_pub = self.create_publisher(
            PointCloud2, pointcloud_topic, 10)

        self.timer = self.create_timer(1.0 / 15.0, self.process_frame)
        self.get_logger().info(
            f'ecobot_detection_node started — backend={self.backend} '
            f'model={model_path} conf={self.conf_threshold} '
            f'every {self.inference_rate} ticks')

    def _init_backend(self, backend_want, model_path):
        if backend_want in ('auto', 'ultralytics'):
            if ULTRALYTICS_AVAILABLE:
                try:
                    self.model = YOLO(model_path, task='detect')
                    self.backend = 'ultralytics'
                    return
                except Exception as e:
                    self.get_logger().error(f'ultralytics load failed: {e}')
            elif backend_want == 'ultralytics':
                self.get_logger().error(
                    'ultralytics not installed. Install with: '
                    'pip install ultralytics (requires a working torch)')
        if backend_want in ('auto', 'tensorrt'):
            if not model_path.endswith('.engine'):
                if backend_want == 'tensorrt':
                    self.get_logger().error(
                        'tensorrt backend requires a .engine model_path')
            elif not (TRT_AVAILABLE and CUDA_AVAILABLE):
                self.get_logger().error('tensorrt/pycuda not available')
            else:
                try:
                    self.trt = TrtBackend(model_path, self.input_size)
                    self.backend = 'tensorrt'
                    return
                except Exception as e:
                    self.get_logger().error(f'TensorRT load failed: {e}')
        if self.backend is None:
            self.get_logger().error(
                'no inference backend available — node will idle. '
                'Install ultralytics or provide a valid .engine model.')

    def color_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_img
                self.latest_header = msg.header
        except Exception:
            pass

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = depth
        except Exception:
            pass

    def cam_info_cb(self, msg: CameraInfo):
        if not self.intrinsics_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsics_received = True

    def get_depth_at(self, u, v):
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            return None
        h, w = depth.shape[:2]
        if v < 0 or v >= h or u < 0 or u >= w:
            return None
        v0, v1 = max(0, v - 2), min(h, v + 3)
        u0, u1 = max(0, u - 2), min(w, u + 3)
        patch = depth[v0:v1, u0:u1]
        valid = patch[(patch > 0) & (patch < 5000)]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) * self.depth_scale

    def _infer_ultralytics(self, img):
        res = self.model.predict(
            img, conf=self.conf_threshold, iou=self.iou_threshold,
            imgsz=self.input_size, verbose=False)[0]
        detections = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        h, w = img.shape[:2]
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            detections.append({
                'class_id': int(cls),
                'class_name': res.names.get(cls, str(cls)),
                'confidence': round(float(conf), 3),
                'bbox': [int(max(0, x1)), int(max(0, y1)),
                         int(min(w - 1, x2)), int(min(h - 1, y2))],
            })
        return detections

    def _infer_tensorrt(self, img):
        output, scale, dw, dh = self.trt.infer(img)
        dets = output.reshape((1, 84, -1)).transpose(0, 2, 1)[0]
        h_img, w_img = img.shape[:2]
        boxes, scores, class_ids = [], [], []
        for det in dets:
            cls_id = int(np.argmax(det[4:]))
            score = float(det[4 + cls_id])
            if score < self.conf_threshold:
                continue
            cx, cy, bw, bh = det[:4]
            cx = (cx - dw) / scale
            cy = (cy - dh) / scale
            bw /= scale
            bh /= scale
            boxes.append([max(0.0, cx - bw / 2), max(0.0, cy - bh / 2),
                          min(float(w_img), cx + bw / 2),
                          min(float(h_img), cy + bh / 2)])
            scores.append(score)
            class_ids.append(cls_id)
        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold,
                                self.iou_threshold)
        detections = []
        for i in np.array(idxs).flatten():
            x1, y1, x2, y2 = boxes[i]
            cls = class_ids[i]
            detections.append({
                'class_id': int(cls),
                'class_name': COCO_NAMES[cls] if cls < len(COCO_NAMES)
                else str(cls),
                'confidence': round(float(scores[i]), 3),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
            })
        return detections

    def process_frame(self):
        self.tick_count += 1
        if self.tick_count % self.inference_rate != 0:
            return
        if self.backend is None:
            return
        with self.frame_lock:
            if self.latest_frame is None:
                return
            img = self.latest_frame
            header = self.latest_header
            self.latest_frame = None

        if self.backend == 'ultralytics':
            detections = self._infer_ultralytics(img)
        else:
            detections = self._infer_tensorrt(img)

        points_3d = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth_val = self.get_depth_at(cx, cy)
            det['distance'] = round(depth_val, 2) if depth_val is not None \
                else None
            if depth_val is not None and depth_val > 0:
                x_3d = (cx - self.cx) * depth_val / self.fx
                y_3d = (cy - self.cy) * depth_val / self.fy
                z_3d = depth_val
                det['x'] = round(x_3d, 3)
                det['y'] = round(y_3d, 3)
                det['z'] = round(z_3d, 3)
                # Physical size from the box and the depth at its centre.
                bbox_h_px = max(1, abs(y2 - y1))
                bbox_w_px = max(1, abs(x2 - x1))
                h_3d = (bbox_h_px * depth_val) / self.fy
                w_3d = (bbox_w_px * depth_val) / self.fx
                det['height'] = round(h_3d, 3)
                det['width'] = round(w_3d, 3)

                # How high off the floor the object is. In the optical
                # frame y points DOWN, so subtracting it from the camera's
                # mounting height gives height above the floor.
                #
                # The old z_top/z_bottom added half a HEIGHT to z, the
                # FORWARD distance — two different axes — so they described
                # nothing real. The arm aimed with them and swept past the
                # plant into the ceiling.
                center_h = self.camera_height - y_3d
                det['center_height'] = round(center_h, 3)
                det['top_height'] = round(center_h + h_3d / 2.0, 3)
                det['bottom_height'] = round(max(0.0, center_h - h_3d / 2.0), 3)
                det['plant_type'] = det.get('class_name', 'plant')
                points_3d.append([x_3d, y_3d, z_3d])

        det_msg = String()
        det_msg.data = json.dumps(detections)
        self.detections_pub.publish(det_msg)

        if points_3d:
            self._publish_pointcloud(np.array(points_3d, dtype=np.float32), header.stamp if header else None)

        if self.overlay_pub.get_subscription_count() > 0:
            try:
                overlay_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                overlay_msg.header = header
                self.overlay_pub.publish(overlay_msg)
            except Exception:
                pass

    def _publish_pointcloud(self, points, stamp=None):
        if len(points) == 0:
            return
        msg = PointCloud2()
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32,
                       count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        self.pc_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EcobotDetectionNode()
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
