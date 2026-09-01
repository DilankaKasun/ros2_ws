import socket
import threading
import time
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
from http.server import HTTPServer, BaseHTTPRequestHandler


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        if hasattr(socket, 'SO_REUSEPORT'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class YoloMJPEGHandler(BaseHTTPRequestHandler):
    frame = None
    frame_lock = threading.Lock()

    def do_GET(self):
        if self.path.startswith('/yolo.mjpg'):
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
            self.wfile.write(b'<html><body><img src="/yolo.mjpg"/></body></html>')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


COCO_SMALL_OBSTACLE_IDS = {
    39: 'bottle',
    44: 'cup',
    46: 'wine glass',
    64: 'potted plant',
    73: 'book',
    74: 'clock',
    76: 'scissors',
    84: 'book',
    47: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    65: 'bed',
    67: 'dining table',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush',
}

NUM_CLASSES = 80
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


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f'Host: {self.host.shape} Device: {self.device.shape}'


class YOLODetection(Node):
    def __init__(self):
        super().__init__('yolo_detection')

        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('inference_rate', 2)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('use_small_obstacle_filter', True)
        self.declare_parameter('mjpeg_port', 8087)
        self.declare_parameter('depth_scale', 0.001)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.input_size = self.get_parameter('input_size').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.use_small_filter = self.get_parameter('use_small_obstacle_filter').value
        mjpeg_port = self.get_parameter('mjpeg_port').value
        self.depth_scale = self.get_parameter('depth_scale').value

        self.bridge = CvBridge()
        self.fx = 430.0
        self.fy = 430.0
        self.cx = 320.0
        self.cy = 240.0
        self.intrinsics_received = False

        self.latest_color = None
        self.latest_depth = None
        self.frame_count = 0
        self.color_lock = threading.Lock()
        self.depth_lock = threading.Lock()

        self.trt_engine = None
        self.trt_context = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = None
        self.engine_loaded = False

        YoloMJPEGHandler.frame = None
        self.mjpeg_server = ReusableHTTPServer(('', mjpeg_port), YoloMJPEGHandler)
        self.mjpeg_thread = threading.Thread(
            target=self.mjpeg_server.serve_forever, daemon=True)
        self.mjpeg_thread.start()
        self.get_logger().info(
            f'yolo detection MJPEG on http://0.0.0.0:{mjpeg_port}/yolo.mjpg')

        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_cb, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_cb, 10)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.cam_info_cb, 10)

        self.overlay_pub = self.create_publisher(
            Image, '/yolo_detection_overlay', 10)
        self.detections_pub = self.create_publisher(
            String, '/yolo_detections_json', 10)
        self.pc_pub = self.create_publisher(
            PointCloud2, '/yolo_obstacle_points', 10)

        if not TRT_AVAILABLE:
            self.get_logger().error(
                'TensorRT not installed. Install with: sudo apt install nvidia-l4t-tensorrt')
            return
        if not CUDA_AVAILABLE:
            self.get_logger().error(
                'PyCUDA not installed. Install with: pip install pycuda')

        if model_path:
            self.load_engine(model_path)
        else:
            self.get_logger().warn(
                'No model path provided. Set model_path parameter to a .engine file.')

        self.get_logger().info(
            f'yolo detection started — model={model_path} '
            f'conf={self.conf_threshold} iou={self.iou_threshold} '
            f'rate=every {self.inference_rate} frame')

    def cam_info_cb(self, msg: CameraInfo):
        if not self.intrinsics_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.intrinsics_received = True

    def color_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.color_lock:
                self.latest_color = cv_img.copy()
                self.latest_header = msg.header
        except Exception:
            return

    def depth_cb(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = depth_img.copy()
        except Exception:
            return

    def load_engine(self, engine_path):
        if not TRT_AVAILABLE:
            return
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()
            self.allocate_buffers()
            self.engine_loaded = True
            self.get_logger().info(f'TensorRT engine loaded: {engine_path}')
        except Exception as e:
            self.get_logger().error(f'failed to load TensorRT engine: {e}')

    def allocate_buffers(self):
        if not TRT_AVAILABLE:
            return
        self.input_host = None
        self.output_host = None
        self.input_device = None
        self.output_device = None
        for i in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(i)
            mode = self.trt_engine.get_tensor_mode(name)
            shape = self.trt_engine.get_tensor_shape(name)
            dtype = self.trt_engine.get_tensor_dtype(name)
            size = abs(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.trt_context.set_tensor_address(name, int(device_mem))
            if mode == trt.TensorIOMode.INPUT:
                self.input_host = host_mem
                self.input_device = device_mem
                self.input_name = name
            else:
                self.output_host = host_mem
                self.output_device = device_mem
                self.output_name = name
        self.stream = cuda.Stream()

    def preprocess(self, img):
        h, w = img.shape[:2]
        input_w = self.input_size
        input_h = self.input_size
        scale = min(input_w / w, input_h / h)
        nw = int(w * scale)
        nh = int(h * scale)
        dw = (input_w - nw) // 2
        dh = (input_h - nh) // 2
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.float32)
        canvas[dh:dh+nh, dw:dw+nw] = resized.astype(np.float32)
        blob = canvas.transpose(2, 0, 1)[np.newaxis, ...] / 255.0
        return blob.astype(np.float32), scale, dw, dh

    def postprocess(self, output, scale, dw, dh, orig_w=640, orig_h=480):
        output = output.reshape((1, 84, -1)).transpose(0, 2, 1)
        dets = output[0]
        boxes, scores, class_ids = [], [], []
        for det in dets:
            bbox = det[:4]
            scores_list = det[4:]
            cls_id = int(np.argmax(scores_list))
            score = float(scores_list[cls_id])
            if score < self.conf_threshold:
                continue
            cx, cy, w, h = bbox
            cx = (cx - dw) / scale
            cy = (cy - dh) / scale
            w /= scale
            h /= scale
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = min(orig_w, cx + w / 2)
            y2 = min(orig_h, cy + h / 2)
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cls_id)
        if len(boxes) == 0:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        if len(indices) == 0:
            return []
        results = []
        for i in np.array(indices).flatten():
            x1, y1, x2, y2 = boxes[i]
            cls = class_ids[i]
            if self.use_small_filter and cls not in COCO_SMALL_OBSTACLE_IDS:
                continue
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(scores[i]),
                'class_id': cls,
                'class_name': COCO_NAMES[cls] if cls < len(COCO_NAMES) else str(cls),
            })
        return results

    def infer(self, img):
        if not self.engine_loaded:
            return []
        h, w = img.shape[:2]
        input_img, scale, dw, dh = self.preprocess(img)
        np.copyto(self.input_host, input_img.ravel())
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.trt_context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()
        output = self.output_host.copy()
        return self.postprocess(output, scale, dw, dh, orig_w=w, orig_h=h)

    def get_depth_at(self, u, v):
        with self.depth_lock:
            if self.latest_depth is None:
                return None
            depth = self.latest_depth
        if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
            return None
        v0 = max(0, v - 2)
        v1 = min(depth.shape[0], v + 3)
        u0 = max(0, u - 2)
        u1 = min(depth.shape[1], u + 3)
        patch = depth[v0:v1, u0:u1]
        valid = patch[(patch > 0) & (patch < 5000)]
        if len(valid) == 0:
            return None
        return float(np.median(valid)) * self.depth_scale

    def publish_loop(self):
        self.frame_count += 1
        if self.frame_count % self.inference_rate != 0:
            return
        with self.color_lock:
            if self.latest_color is None:
                return
            img = self.latest_color.copy()
            header = getattr(self, 'latest_header', None)
            self.latest_color = None

        if not self.engine_loaded:
            return

        results = self.infer(img)
        overlay = img.copy()
        points_3d = []

        for det in results:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth_val = self.get_depth_at(cx, cy)
            det['distance'] = depth_val
            if depth_val is not None and depth_val > 0:
                x_3d = (cx - self.cx) * depth_val / self.fx
                y_3d = (cy - self.cy) * depth_val / self.fy
                z_3d = depth_val
                points_3d.append([x_3d, y_3d, z_3d])

            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            if depth_val is not None:
                label += f' {depth_val:.2f}m'
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if points_3d:
            self.publish_pointcloud(np.array(points_3d, dtype=np.float32), header.stamp if header else None)

        _, jpg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with YoloMJPEGHandler.frame_lock:
            YoloMJPEGHandler.frame = jpg.tobytes()

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            self.overlay_pub.publish(overlay_msg)
        except Exception:
            pass

        self.detections_pub.publish(String(data=json.dumps(results)))

    def publish_pointcloud(self, points, stamp=None):
        if len(points) == 0:
            return
        msg = PointCloud2()
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
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
        msg.data = points.tobytes()
        self.pc_pub.publish(msg)

    def destroy_node(self):
        self.mjpeg_server.shutdown()
        self.mjpeg_server.server_close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetection()
    timer = node.create_timer(1.0 / 15.0, node.publish_loop)
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
