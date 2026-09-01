import os
import time
import json
import unittest
import urllib.request
import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge

from ecobot_arm_control.usb_camera_node import USBCameraNode
from ecobot_arm_control.arm_camera_server import ArmCameraServer, MjpegHandler
from ecobot_arm_control.wrist_cv_analyzer import WristCVAnalyzerNode
from ecobot_arm_control.arm_scanner_node import ArmScannerNode
from ecobot_sensors.detection_goto import DetectionGoto


class TestArmVideoStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def test_usb_camera_node_params_and_pub(self):
        """Step 2: Verify initialization of arm camera node."""
        node = USBCameraNode()
        self.assertEqual(node.get_parameter('width').value, 640)
        self.assertEqual(node.get_parameter('height').value, 480)
        self.assertEqual(node.get_parameter('topic').value, '/arm/camera/image_raw')
        node.destroy_node()

    def test_arm_camera_server_init(self):
        """Step 3: Verify video streaming service initialization on Jetson Nano."""
        port = 8094  # Use distinct test port
        node = ArmCameraServer()
        self.assertEqual(node.get_parameter('topic').value, '/arm/camera/image_raw')
        node.destroy_node()

    def test_mjpeg_stream_end_to_end(self):
        """Step 5: Verify live video stream accessibility via HTTP server endpoint."""
        port = 8095

        # Initialize streaming node
        server_node = ArmCameraServer()
        server_node._server.shutdown()
        server_node._server.server_close()

        # Re-bind server to test port
        from ecobot_arm_control.arm_camera_server import ReusableHTTPServer
        MjpegHandler.frame = None
        server_node._server = ReusableHTTPServer(('', port), MjpegHandler)
        server_node._thread = threading.Thread(
            target=server_node._server.serve_forever, daemon=True)
        server_node._thread.start()

        # Publish a synthetic frame (test plant image)
        bridge = CvBridge()
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw green plant foliage
        cv2.circle(test_img, (320, 240), 100, (0, 200, 0), -1)
        cv2.putText(test_img, 'TEST PLANT', (260, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        msg = bridge.cv2_to_imgmsg(test_img, encoding='bgr8')
        server_node._image_cb(msg)

        # Connect to stream endpoint http://127.0.0.1:<port>/stream.mjpg
        url = f'http://127.0.0.1:{port}/stream.mjpg'
        req = urllib.request.Request(url)
        
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get('Content-Type', '')
            self.assertIn('multipart/x-mixed-replace', content_type)
            
            # Read first chunk containing frame header and JPEG data
            chunk = resp.read(1024)
            self.assertTrue(b'--frame' in chunk or b'image/jpeg' in chunk or len(chunk) > 0)

        server_node.destroy_node()

    def test_wrist_cv_analyzer_plant_identification(self):
        """Step 4 & 6: Verify plant identification and visibility in video stream."""
        analyzer = WristCVAnalyzerNode()
        bridge = CvBridge()

        # Intercept publisher call to verify plant identification output directly
        published_msgs = []
        orig_pub = analyzer._parts_pub.publish
        analyzer._parts_pub.publish = lambda msg: published_msgs.append(json.loads(msg.data))

        # Create synthetic plant image with green leaves
        plant_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Green foliage area > 8%
        cv2.rectangle(plant_img, (100, 100), (540, 380), (35, 180, 35), -1)
        
        msg = bridge.cv2_to_imgmsg(plant_img, encoding='bgr8')
        analyzer._on_frame(msg)

        self.assertGreater(len(published_msgs), 0)
        latest = published_msgs[-1]
        self.assertEqual(latest['detected_part'], 'leaves')
        self.assertGreaterEqual(latest['confidence'], 0.5)
        self.assertTrue(latest['in_focus'])

        analyzer._parts_pub.publish = orig_pub
        analyzer.destroy_node()

    def test_stationary_arm_streaming_no_motion_commands(self):
        """Verify video streaming operates without publishing arm motion commands."""
        joint_cmd_received = []
        pose_goal_received = []

        sub_node = rclpy.create_node('test_motion_listener')
        sub_node.create_subscription(
            Float64MultiArray, '/arm/joint_commands',
            lambda m: joint_cmd_received.append(m), 10)
        sub_node.create_subscription(
            Float64MultiArray, '/arm/pose_goal',
            lambda m: pose_goal_received.append(m), 10)

        # Initialize streaming nodes (camera node + camera webserver + analyzer)
        cam_node = USBCameraNode()
        server_node = ArmCameraServer()
        analyzer_node = WristCVAnalyzerNode()

        # Send frame to trigger analyzer & server processing
        bridge = CvBridge()
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (540, 380), (35, 180, 35), -1)
        msg = bridge.cv2_to_imgmsg(test_img, encoding='bgr8')
        
        server_node._image_cb(msg)
        analyzer_node._on_frame(msg)

        # Allow time for any potential published messages
        rclpy.spin_once(sub_node, timeout_sec=0.1)

        # Verify zero joint commands or pose goals were published (arm stays completely stationary)
        self.assertEqual(len(joint_cmd_received), 0, "No joint movement commands should be sent during camera streaming")
        self.assertEqual(len(pose_goal_received), 0, "No pose goals should be sent during camera streaming")

        cam_node.destroy_node()
        server_node.destroy_node()
        analyzer_node.destroy_node()
        sub_node.destroy_node()

    def test_arm_scanner_motion_generation(self):
        """Verify arm_scanner_node generates joint commands when scan command received."""
        scanner = ArmScannerNode()
        published_joint_cmds = []

        orig_pub = scanner._joint_pub.publish
        scanner._joint_pub.publish = lambda msg: published_joint_cmds.append(list(msg.data))

        # Send scan command
        cmd_msg = String()
        cmd_msg.data = json.dumps({
            'action': 'scan',
            'x': 0.40,
            'y': 0.0,
            'z': 0.15,
            'plant_type': 'potted plant'
        })
        scanner._scanner_cmd_cb(cmd_msg)

        # Trigger timer callback to process scan path
        scanner._timer_cb()

        self.assertTrue(scanner._scanning)
        self.assertGreater(len(published_joint_cmds), 0)

        scanner._joint_pub.publish = orig_pub
        scanner.destroy_node()

    def test_revised_scanning_sequence_phases(self):
        """Verify the 4-phase revised scanning sequence is correctly generated and executed."""
        scanner = ArmScannerNode()
        
        # Send scan command
        cmd_msg = String()
        cmd_msg.data = json.dumps({
            'action': 'scan',
            'x': 0.40,
            'y': 0.0,
            'z': 0.15,
            'plant_type': 'test fern'
        })
        scanner._scanner_cmd_cb(cmd_msg)

        self.assertTrue(scanner._scanning)
        self.assertGreater(len(scanner._scan_queue), 0)

        # Retrieve waypoints in the generated queue
        waypoints = [wp[0] for wp in scanner._scan_queue]

        # 1. Phase 1: Move above plant (aimed at plant center, FK-verified)
        self.assertEqual(waypoints[0], "above_plant_start")

        # 2. Phase 2: Aimed transition (re-solved per step so the camera
        # stays pointed at the plant — replaces the old fixed-[b,s,e]
        # wrist/body sweeps whose aim error was 119-177°).
        transition_steps = [wp for wp in waypoints if wp.startswith("aimed_transition_step_")]
        self.assertGreaterEqual(len(transition_steps), 1)

        # 3. Phase 3: Detailed Component Scan (all waypoints must be aimed
        # at the plant — never the fake aim_err=0 body sweep).
        detailed_steps = [wp for wp in waypoints
                          if not (wp == "above_plant_start"
                                  or wp.startswith("aimed_transition_step_"))]
        self.assertGreater(len(detailed_steps), 0)
        for label, sx, sy, sz, ax, ay, az, angles, aim_err, dwell in scanner._scan_queue[1:]:
            self.assertLess(aim_err, 65.0,
                            f'waypoint {label} mis-aimed at {aim_err:.1f}°')

        # Verify that stop_scan correctly homed the arm and cleared the queue
        scanner._stop_scan()
        self.assertFalse(scanner._scanning)
        self.assertEqual(len(scanner._scan_queue), 0)
        
        scanner.destroy_node()

    def test_detection_goto_distance_and_stream_initiation(self):
        """Verify plant detection, stopping within 35cm-40cm, and successful initiation of video stream/recording."""
        # 1. Instantiate the nodes
        goto = DetectionGoto()
        scanner = ArmScannerNode()

        # Enforce target tracking class and auto-scanning mode
        goto.target_class = 'potted plant'
        goto._mode = 'tracking'
        goto.active = True
        goto.auto_scan_on_reach = True

        # Check and assert that stop_distance is strictly within the 35cm to 40cm range
        self.assertGreaterEqual(goto.stop_distance, 0.35)
        self.assertLessEqual(goto.stop_distance, 0.40)

        # Intercept scan command publish and route it to the scanner node
        published_scan_cmd = []
        goto.plant_scan_cmd_pub.publish = lambda msg: published_scan_cmd.append(msg.data)

        # Mock a plant detection that is too far (e.g., 0.60m)
        msg_far = String()
        msg_far.data = json.dumps([{
            'class_name': 'potted plant',
            'x': 0.0,
            'y': 0.0,
            'z': 0.60,
            'confidence': 0.85
        }])
        goto.detections_cb(msg_far)

        # Run control loop - robot should keep driving (not stop or reach yet)
        goto.control_loop()
        self.assertNotEqual(goto.status, 'REACHED')
        self.assertEqual(len(published_scan_cmd), 0)

        # Mock a plant detection that is within the target range (e.g., 0.35m)
        msg_close = String()
        msg_close.data = json.dumps([{
            'class_name': 'potted plant',
            'x': 0.0,
            'y': 0.0,
            'z': 0.35,
            'confidence': 0.90
        }])
        goto.detections_cb(msg_close)

        # Run control loop multiple times to allow the distance filter to converge and satisfy stop confirmation ticks
        for _ in range(15):
            goto.control_loop()

        # Robot must autonomously stop and transition to REACHED
        self.assertEqual(goto.status, 'REACHED')

        # Since auto_scan_on_reach is True, it must have triggered the scan
        self.assertEqual(len(published_scan_cmd), 1)

        # Verify the command received from detection_goto has 'scan_here' action
        scan_here_data = json.loads(published_scan_cmd[0])
        self.assertEqual(scan_here_data['action'], 'scan_here')

        # Pass a 'scan' command to the arm scanner node to initiate the actual scanning sequence and video recording stream
        scanner_msg = String()
        scanner_msg.data = json.dumps({
            'action': 'scan',
            'x': 0.40,
            'y': 0.0,
            'z': 0.15,
            'plant_type': 'potted plant'
        })
        # First we need to intercept the param client call so it doesn't fail on missing service
        scanner._param_client.service_is_ready = lambda: True
        scanner._param_client.call_async = lambda req: None

        scanner._scanner_cmd_cb(scanner_msg)

        # Verify successful initiation of scanning and video recording stream
        self.assertTrue(scanner._scanning)
        self.assertIsNotNone(scanner._video_writer)
        self.assertGreater(len(scanner._scan_queue), 0)

        # Verify first waypoint is "above_plant_start"
        self.assertEqual(scanner._scan_queue[0][0], "above_plant_start")

        # Cleanup
        scanner._stop_scan()
        goto.destroy_node()
        scanner.destroy_node()


if __name__ == '__main__':
    unittest.main()
