import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
import cv2
import traceback
from ultralytics import YOLO

class YoloTrackerNode(Node):
    def __init__(self):
        super().__init__('yolo_tracker')
        
        self.declare_parameter('topic', '/camera/color/image_raw')
        self.declare_parameter('publish_topic', '/ecobot/vision/plants')
        self.declare_parameter('model', 'yolov8n.pt') 
        
        topic = self.get_parameter('topic').value
        pub_topic = self.get_parameter('publish_topic').value
        model_path = self.get_parameter('model').value
        
        self.bridge = CvBridge()
        
        self.get_logger().info(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.get_logger().info("YOLO model loaded.")
        
        self.sub = self.create_subscription(Image, topic, self.image_callback, 1)
        self.pub = self.create_publisher(String, pub_topic, 10)
        
    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.track(
                cv_image, 
                persist=True, 
                classes=[58], 
                verbose=False,
                tracker="bytetrack.yaml"
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        if boxes.id is not None:
                            track_id = int(boxes.id[i].cpu().numpy())
                        else:
                            track_id = hash(tuple(xyxy)) % 10000
                            
                        detections.append({
                            "id": str(track_id),
                            "box_2d": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "label": "Plant",
                            "confidence": conf
                        })
            
            out_msg = String()
            out_msg.data = json.dumps(detections)
            self.pub.publish(out_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in YOLO tracking: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloTrackerNode()
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
