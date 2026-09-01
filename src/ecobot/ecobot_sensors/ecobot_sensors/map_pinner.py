import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import json
import math
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Point

class MapPinnerNode(Node):
    def __init__(self):
        super().__init__('map_pinner')
        self.sub = self.create_subscription(String, '/ecobot/map_pin', self.pin_callback, 10)
        self.pub = self.create_publisher(MarkerArray, '/ecobot/semantic_markers', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.markers = []
        self.marker_id_counter = 0
        self.get_logger().info("Semantic Map Pinner initialized.")

    def pin_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            plant_id = data.get('id', 'unknown')
            desc = data.get('desc', 'Plant')
            distance = float(data.get('distance', 1.0))
            
            # Get robot's current pose in odom frame
            try:
                trans = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
                
                # Simple projection: place marker distance meters in front of the robot
                # Extract yaw from quaternion
                q = trans.transform.rotation
                siny_cosp = 2 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                marker_x = trans.transform.translation.x + distance * math.cos(yaw)
                marker_y = trans.transform.translation.y + distance * math.sin(yaw)
                marker_z = trans.transform.translation.z
                
                # Create Marker
                marker = Marker()
                marker.header.frame_id = "odom"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "plants"
                marker.id = self.marker_id_counter
                self.marker_id_counter += 1
                
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                
                marker.pose.position.x = marker_x
                marker.pose.position.y = marker_y
                marker.pose.position.z = marker_z + 0.5 # Hover above ground
                
                marker.pose.orientation.w = 1.0
                
                marker.scale.z = 0.2 # Text height
                
                # Green text
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                
                marker.text = f"Plant {plant_id}\n{desc}"
                
                self.markers.append(marker)
                
                # Publish all markers
                array = MarkerArray()
                array.markers = self.markers
                self.pub.publish(array)
                
                self.get_logger().info(f"Pinned Plant {plant_id} at ({marker_x:.2f}, {marker_y:.2f})")
                
            except Exception as e:
                self.get_logger().warn(f"Could not get tf for pin: {e}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing map pin: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MapPinnerNode()
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
