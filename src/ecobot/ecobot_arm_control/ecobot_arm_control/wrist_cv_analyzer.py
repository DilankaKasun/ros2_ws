import json
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class WristCVAnalyzerNode(Node):
    """Real-time Wrist Camera Computer Vision Plant Part Analyzer.
    
    Analyses /arm/camera/image_raw in real time to distinguish:
    1. Leaves & Foliage (Green HSV color segmentation + contour analysis)
    2. Stem & Branches (Canny edge detection + vertical Hough lines + brown/woody HSV)
    3. Base, Roots & Pot Rim (Lower frame soil/pot contours & dark texture)
    
    Calculates image sharpness/focus using Laplacian variance:
        Focus Score = Var(Laplacian(Image))
    
    Publishes analysis JSON on /arm/cv_plant_parts and annotated debug image
    on /arm/camera/cv_overlay.
    """

    def __init__(self):
        super().__init__('wrist_cv_analyzer')

        self.declare_parameter('camera_topic', '/arm/camera/image_raw')
        self.declare_parameter('parts_topic', '/arm/cv_plant_parts')
        self.declare_parameter('overlay_topic', '/arm/camera/cv_overlay')
        self.declare_parameter('focus_threshold', 80.0)

        gp = self.get_parameter
        camera_topic = str(gp('camera_topic').value)
        parts_topic = str(gp('parts_topic').value)
        overlay_topic = str(gp('overlay_topic').value)
        self._focus_threshold = float(gp('focus_threshold').value)

        self._bridge = CvBridge()
        self._parts_pub = self.create_publisher(String, parts_topic, 10)
        self._overlay_pub = self.create_publisher(Image, overlay_topic, 10)

        self._sub = self.create_subscription(
            Image, camera_topic, self._on_frame, 10)

        self.get_logger().info(
            f'Wrist CV Plant Part Analyzer active on {camera_topic}')

    def _on_frame(self, msg):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        h, w, _ = cv_img.shape
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 1. Compute Focus Score (Sharpness) using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_score = float(np.var(laplacian))
        in_focus = focus_score >= self._focus_threshold

        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # 2. Leaves & Foliage Detection (Green HSV range)
        green_mask = cv2.inRange(
            hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_area = int(np.sum(green_mask > 0))

        # 3. Stem & Branches Detection (Canny edges + vertical Hough lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
        branch_count = 0
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if 45.0 <= angle <= 135.0:  # Vertical/angled branch lines
                    branch_count += 1

        # 4. Base & Roots / Pot Detection (Lower third frame texture & pot rim)
        lower_region = hsv[int(h * 0.65):, :]
        soil_pot_mask = cv2.inRange(
            lower_region, np.array([0, 20, 20]), np.array([30, 255, 120]))
        base_area = int(np.sum(soil_pot_mask > 0))

        # Determine dominant plant part in view
        detected_part = 'unknown'
        confidence = 0.0

        total_pixels = h * w
        green_ratio = green_area / float(total_pixels)
        base_ratio = base_area / float(w * (h * 0.35))

        if green_ratio >= 0.08:
            detected_part = 'leaves'
            confidence = min(0.98, round(0.5 + green_ratio * 1.5, 2))
        elif branch_count >= 3:
            detected_part = 'branches_stem'
            confidence = min(0.95, round(0.5 + branch_count * 0.05, 2))
        elif base_ratio >= 0.15:
            detected_part = 'base_roots'
            confidence = min(0.92, round(0.5 + base_ratio * 1.2, 2))
        elif green_ratio > 0.02:
            detected_part = 'leaves'
            confidence = 0.60

        # Calculate plant centroid & bounding box from combined mask (foliage + pot base)
        full_pot_mask = np.zeros_like(green_mask)
        full_pot_mask[int(h * 0.65):, :] = soil_pot_mask
        combined_mask = cv2.bitwise_or(green_mask, full_pot_mask)

        M = cv2.moments(combined_mask)
        cx_norm, cy_norm = 0.0, 0.0
        bbox = [0, 0, 0, 0]
        plant_present = False

        if M["m00"] > 100:
            cx_px = int(M["m10"] / M["m00"])
            cy_px = int(M["m01"] / M["m00"])
            cx_norm = float((cx_px - w / 2.0) / (w / 2.0))
            cy_norm = float((cy_px - h / 2.0) / (h / 2.0))
            bx, by, bw, bh = cv2.boundingRect(combined_mask)
            bbox = [int(bx), int(by), int(bx + bw), int(by + bh)]
            if detected_part in ('leaves', 'branches_stem', 'base_roots') and confidence >= 0.20:
                plant_present = True

        # Construct Analysis JSON
        analysis = {
            'detected_part': detected_part,
            'confidence': confidence,
            'focus_score': round(focus_score, 1),
            'in_focus': in_focus,
            'green_foliage_ratio': round(green_ratio, 3),
            'branch_line_count': branch_count,
            'base_pot_ratio': round(base_ratio, 3),
            'plant_present': plant_present,
            'centroid_x': round(cx_norm, 3),
            'centroid_y': round(cy_norm, 3),
            'bbox': bbox,
        }

        analysis_msg = String()
        analysis_msg.data = json.dumps(analysis)
        self._parts_pub.publish(analysis_msg)

        # Build Annotated Overlay Image
        overlay = cv_img.copy()
        if plant_present:
            cx_px = int((cx_norm + 1.0) * w / 2.0)
            cy_px = int((cy_norm + 1.0) * h / 2.0)
            cv2.circle(overlay, (cx_px, cy_px), 8, (255, 0, 255), -1)
            cv2.putText(overlay, 'PLANT CENTER', (cx_px + 10, cy_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        color = (0, 255, 0) if detected_part == 'leaves' else (
            (0, 165, 255) if detected_part == 'branches_stem' else (0, 0, 255))
        
        status_text = f"Part: {detected_part.upper()} ({confidence*100:.0f}%) | Focus: {focus_score:.1f} {'[OK]' if in_focus else '[BLUR]'}"
        cv2.putText(overlay, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw green leaf contours
        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 300:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(overlay, 'LEAF', (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw detected branch lines
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if 45.0 <= angle <= 135.0:
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)

        try:
            overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = msg.header
            self._overlay_pub.publish(overlay_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = WristCVAnalyzerNode()
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
