import unittest
import math
import json
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import rclpy
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState

from ecobot_arm_control.servo_config import (
    JOINTS, NUM_JOINTS, to_servo, to_ik, within_limits, apply_overrides,
    ik_limits,
)
from ecobot_arm_control.arm_kinematics import ArmKinematics
from ecobot_arm_control.arm_manual_node import ArmManualNode


class Test4DOFJointControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    # -------------------------------------------------------------------------
    # 1. Servo Configuration & 4-DOF Joint Specification Tests
    # -------------------------------------------------------------------------
    def test_4dof_joint_specifications(self):
        """Verify the 4-DOF arm joint definitions, channels, and physical bounds."""
        self.assertEqual(NUM_JOINTS, 4, "Robot arm must have exactly 4 degrees of freedom")
        self.assertEqual(len(JOINTS), 4)

        expected_joints = [
            ('arm_base_joint', 3, 0.0, 220.0, 270.0),
            ('arm_shoulder_joint', 0, 0.0, 125.0, 180.0),
            ('arm_elbow_joint', 1, 0.0, 180.0, 180.0),
            ('arm_wrist_joint', 2, 0.0, 180.0, 180.0),
        ]

        for i, (name, ch, min_a, max_a, servo_rng) in enumerate(expected_joints):
            j = JOINTS[i]
            self.assertEqual(j['name'], name)
            self.assertEqual(j['channel'], ch)
            self.assertGreaterEqual(j['home_angle'], min_a)
            self.assertLessEqual(j['home_angle'], max_a)
            self.assertGreaterEqual(j['min_angle'], min_a)
            self.assertLessEqual(j['max_angle'], max_a)
            self.assertEqual(j['servo_range'], servo_rng)
            self.assertGreaterEqual(j['pulse_min'], 0)
            self.assertLessEqual(j['pulse_max'], 4096)
            self.assertLess(j['pulse_min'], j['pulse_max'])

    def test_servo_ik_coordinate_transforms(self):
        """Verify round-trip mapping between IK coordinate frame and servo frame."""
        # Base has 95° offset, other joints 0° offset
        ik_sample = [0.0, 45.0, 90.0, 30.0]
        servo_sample = to_servo(ik_sample)
        
        # Base in servo space: ik 0 maps through offset -85
        self.assertAlmostEqual(servo_sample[0], -85.0, places=3)
        self.assertAlmostEqual(servo_sample[1], 45.0 - 143.0, places=3)
        self.assertAlmostEqual(servo_sample[2], 90.0 + 70.0, places=3)
        self.assertAlmostEqual(servo_sample[3], -30.0 + 92.0, places=3)

        # Round-trip transformation
        recovered_ik = to_ik(servo_sample)
        for original, recovered in zip(ik_sample, recovered_ik):
            self.assertAlmostEqual(original, recovered, places=3)

    def test_joint_limits_validation(self):
        """Verify joint angle limit checker accepts valid and rejects out-of-bounds angles."""
        # Valid home angles
        home_angles = [float(j['home_angle']) for j in JOINTS]
        self.assertTrue(within_limits(home_angles))

        # Base out of bounds (> 220)
        self.assertFalse(within_limits([225.0, 90.0, 90.0, 90.0]))
        # Shoulder out of bounds (> 125)
        self.assertFalse(within_limits([107.0, 130.0, 90.0, 90.0]))
        # Negative angle
        self.assertFalse(within_limits([107.0, -5.0, 90.0, 90.0]))

    def test_servo_config_overrides(self):
        """Verify runtime parameter override functionality."""
        orig_max = JOINTS[1]['max_angle']
        apply_overrides({'arm_shoulder_joint': {'max_angle': 120.0}})
        self.assertEqual(JOINTS[1]['max_angle'], 120.0)
        # Restore
        apply_overrides({'arm_shoulder_joint': {'max_angle': orig_max}})
        self.assertEqual(JOINTS[1]['max_angle'], orig_max)

    # -------------------------------------------------------------------------
    # 2. 4-DOF Kinematics (FK, IK, Orientation Aiming) Tests
    # -------------------------------------------------------------------------
    def test_4dof_forward_kinematics(self):
        """Verify forward kinematics computation for known 4-DOF arm geometry."""
        ik = ArmKinematics()

        # 1. Straight down pose: th1=0, th2=0, th3=0, th4=0
        # r = OFF_R, z = (L0+OFF_Z) - (L1+L2+L3) = 0.380 - 0.350 = 0.030
        x, y, z = ik.forward(0, 0, 0, 0)
        # x is the bracket offset, not zero: the shoulder sits off the axis.
        self.assertAlmostEqual(x, ik.pivot_r, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)
        self.assertAlmostEqual(z, ik.pivot_z - ik.span, places=3)

        # 2. Horizontal pose: th1=0, th2=90, th3=0, th4=0
        # r = OFF_R + L1+L2+L3 = 0.390, z = L0+OFF_Z = 0.380
        x, y, z = ik.forward(0, 90, 0, 0)
        self.assertAlmostEqual(x, ik.pivot_r + ik.span, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)
        self.assertAlmostEqual(z, ik.pivot_z, places=3)

        # 3. Base rotation 90 deg: x=0, y=0.390
        x, y, z = ik.forward(90, 90, 0, 0)
        self.assertAlmostEqual(x, 0.0, places=3)
        self.assertAlmostEqual(y, 0.390, places=3)
        self.assertAlmostEqual(z, ik.pivot_z, places=3)

    def test_4dof_inverse_kinematics_position(self):
        """Verify 4-DOF position-only inverse kinematics for reachable workspaces."""
        ik = ArmKinematics()

        # Test points within reachable envelope
        test_points = [
            (0.20, 0.0, 0.40),
            (0.0, 0.10, 0.60),
            (0.15, -0.15, 0.55),
            (0.25, 0.0, 0.45),
        ]

        for tx, ty, tz in test_points:
            self.assertTrue(ik.is_reachable(tx, ty, tz), f"Target ({tx}, {ty}, {tz}) should be reachable")
            sol = ik.inverse(tx, ty, tz, theta2_min=-90, theta2_max=180,
                             theta3_min=0, theta3_max=180, theta4_min=-90, theta4_max=180)
            self.assertIsNotNone(sol, f"IK failed for reachable point ({tx}, {ty}, {tz})")
            
            # Verify FK of IK solution reproduces target (x, y, z) within 2mm
            fx, fy, fz = ik.forward(*sol)
            self.assertAlmostEqual(fx, tx, delta=0.005)
            self.assertAlmostEqual(fy, ty, delta=0.005)
            self.assertAlmostEqual(fz, tz, delta=0.005)

    def test_4dof_unreachable_ik_handling(self):
        """Verify IK correctly identifies points outside reachable envelope."""
        ik = ArmKinematics()
        
        # Max reach is L1 + L2 + L3 = 0.390m
        self.assertFalse(ik.is_reachable(0.80, 0.0, 0.10))
        self.assertIsNone(ik.inverse(0.80, 0.0, 0.10))

    def test_4dof_orientation_aiming_ik(self):
        """Verify orientation-aware 4-DOF IK places wrist at standoff and aims camera at target."""
        # Use the arm's real geometry and limits rather than invented ones,
        # so this exercises poses the hardware can actually hold.
        ik = ArmKinematics()
        lim = ik_limits()

        # Standoff position and aim point (camera looking outward at a plant)
        sx, sy, sz = 0.20, 0.0, 0.40
        ax, ay, az = 0.32, 0.0, 0.40

        sol = ik.inverse_aim(sx, sy, sz, ax, ay, az,
                             theta2_min=lim[1][0], theta2_max=lim[1][1],
                             theta3_min=lim[2][0], theta3_max=lim[2][1],
                             theta4_min=lim[3][0], theta4_max=lim[3][1],
                             theta1_min=lim[0][0], theta1_max=lim[0][1])
        self.assertIsNotNone(sol, "inverse_aim should find solution for valid standoff/aim pair")
        
        # Check FK position at camera tip
        fx, fy, fz = ik.forward(*sol)
        self.assertAlmostEqual(fx, sx, delta=0.005)
        self.assertAlmostEqual(fy, sy, delta=0.005)
        self.assertAlmostEqual(fz, sz, delta=0.005)

        # Check aim error is minimal (< 5 degrees)
        err = ik.aim_error(sol[0], sol[1], sol[2], sol[3], ax, ay, az)
        self.assertLess(err, 5.0, f"Aim error {err:.2f}° exceeds allowable tolerance")

    # -------------------------------------------------------------------------
    # 3. 4-DOF ROS 2 Arm Controller Node Tests
    # -------------------------------------------------------------------------
    def test_arm_manual_node_init(self):
        """Verify ArmManualNode initializes with 4 joints, proper publishers, and subscribers."""
        node = ArmManualNode()
        self.assertEqual(len(node._current), 4)
        self.assertEqual(len(node._target), 4)
        self.assertIsNotNone(node._cmd_sub)
        self.assertIsNotNone(node._pose_sub)
        self.assertIsNotNone(node._enable_sub)
        self.assertIsNotNone(node._joint_pub)
        self.assertIsNotNone(node._angle_pub)
        self.assertIsNotNone(node._pose_pub)
        self.assertIsNotNone(node._status_pub)
        node.destroy_node()

    def test_arm_manual_node_joint_command_clamping(self):
        """Verify joint command callback clamps each of the 4 DOFs to its safe joint limits."""
        node = ArmManualNode()
        
        # Send angles exceeding limits: Base 250 (max 220), Shoulder 150 (max 125), Elbow 200 (max 180), Wrist 200 (max 180)
        msg = Float64MultiArray()
        msg.data = [250.0, 150.0, 200.0, 200.0]
        node._cmd_cb(msg)

        self.assertAlmostEqual(node._target[0], 220.0)
        self.assertAlmostEqual(node._target[1], 125.0)
        self.assertAlmostEqual(node._target[2], 180.0)
        self.assertAlmostEqual(node._target[3], 180.0)

        # Send negative / below-minimum angles: -10, -5, -2, -1
        # If all 4 are negative, it triggers servo disable
        msg_disable = Float64MultiArray()
        msg_disable.data = [-1.0, -1.0, -1.0, -1.0]
        node._cmd_cb(msg_disable)
        self.assertFalse(node._enabled)
        self.assertTrue(node._manually_disabled)

        node.destroy_node()

    def test_arm_manual_node_invalid_cmd_length(self):
        """Verify command callback ignores messages with fewer than 4 joint angles."""
        node = ArmManualNode()
        orig_target = list(node._target)
        
        # Send only 2 angles
        msg = Float64MultiArray()
        msg.data = [90.0, 45.0]
        node._cmd_cb(msg)

        # Targets should remain unchanged
        self.assertEqual(node._target, orig_target)
        node.destroy_node()

    def test_trapezoidal_trajectory_profiler(self):
        """Verify synchronized multi-joint smooth trapezoidal velocity profiling."""
        node = ArmManualNode()
        node._peak_speed = 90.0
        node._accel = 250.0
        node._current = [107.0, 125.0, 180.0, 45.0]
        node._target = [50.0, 60.0, 90.0, 90.0]

        node._plan_traj(start=node._current)

        self.assertGreater(node._traj_dur, 0.0)
        self.assertIsNotNone(node._traj_t0)

        # At t = 0, positions match start
        pos_t0 = node._traj_positions(0.0)
        for i in range(4):
            self.assertAlmostEqual(pos_t0[i], node._current[i], places=3)

        # At t = 0.5 * traj_dur, all joints are midway
        pos_mid = node._traj_positions(node._traj_dur * 0.5)
        for i in range(4):
            self.assertTrue(
                min(node._current[i], node._target[i]) <= pos_mid[i] <= max(node._current[i], node._target[i])
            )

        # At t = traj_dur, all joints reach target exactly
        pos_end = node._traj_positions(node._traj_dur)
        for i in range(4):
            self.assertAlmostEqual(pos_end[i], node._target[i], places=3)

        node.destroy_node()

    def test_arm_manual_node_pose_goal_execution(self):
        """Verify Cartesian [x, y, z] pose goal triggers 4-DOF IK and plans trajectory."""
        node = ArmManualNode()
        
        pose_msg = Float64MultiArray()
        pose_msg.data = [0.20, 0.0, 0.40]
        node._pose_cb(pose_msg)

        # Target should be updated with a valid IK solution within limits
        self.assertTrue(within_limits(node._target))
        
        # Verify FK of the target angles reaches approximately the requested pose
        ik_angles = to_ik(node._target)
        fx, fy, fz = node._ik.forward(*ik_angles)
        self.assertAlmostEqual(fx, 0.20, delta=0.01)
        self.assertAlmostEqual(fy, 0.0, delta=0.01)
        self.assertAlmostEqual(fz, 0.40, delta=0.01)

        node.destroy_node()

    def test_arm_manual_node_state_publishing(self):
        """Verify published ROS topics contain accurate 4-DOF joint states and FK pose."""
        node = ArmManualNode()
        
        published_joint_states = []
        published_angles = []
        published_poses = []
        published_statuses = []

        node._joint_pub.publish = lambda msg: published_joint_states.append(msg)
        node._angle_pub.publish = lambda msg: published_angles.append(msg)
        node._pose_pub.publish = lambda msg: published_poses.append(msg)
        node._status_pub.publish = lambda msg: published_statuses.append(msg)

        test_angles = [107.0, 90.0, 120.0, 45.0]
        node._current = list(test_angles)
        node._enabled = True

        node._publish_state()

        # 1. JointState message
        self.assertEqual(len(published_joint_states), 1)
        js = published_joint_states[0]
        self.assertEqual(len(js.name), 4)
        self.assertEqual(js.name, [j['name'] for j in JOINTS])
        self.assertEqual(list(js.position), test_angles)

        # 2. Angle Float64MultiArray
        self.assertEqual(len(published_angles), 1)
        self.assertEqual(list(published_angles[0].data), test_angles)

        # 3. Cartesian Pose Float64MultiArray
        self.assertEqual(len(published_poses), 1)
        self.assertEqual(len(published_poses[0].data), 3)
        expected_fk = node._ik.forward(*to_ik(test_angles))
        self.assertAlmostEqual(published_poses[0].data[0], expected_fk[0], places=4)
        self.assertAlmostEqual(published_poses[0].data[1], expected_fk[1], places=4)
        self.assertAlmostEqual(published_poses[0].data[2], expected_fk[2], places=4)

        # 4. Status String
        self.assertEqual(len(published_statuses), 1)
        self.assertEqual(published_statuses[0].data, 'enabled')

        node.destroy_node()

    # -------------------------------------------------------------------------
    # 4. PCA9685 PWM Driver Servo Angle Mapping Tests
    # -------------------------------------------------------------------------
    def test_pca9685_pwm_angle_mapping(self):
        """Verify angle-to-PWM pulse mapping calculation across all 4 servo channels."""
        from ecobot_arm_control.pca9685_driver import PCA9685
        
        with patch('smbus2.SMBus'):
            pca = PCA9685(bus=7, address=0x40, freq=50.0)
            pca.set_pwm = MagicMock()

            # Joint 0: DS 3218, range=270, pulse_min=150, pulse_max=600
            pca.set_angle(channel=0, angle_deg=0.0, pulse_min=150, pulse_max=600, servo_range=270)
            pca.set_pwm.assert_called_with(0, 0, 150)

            pca.set_angle(channel=0, angle_deg=135.0, pulse_min=150, pulse_max=600, servo_range=270)
            pca.set_pwm.assert_called_with(0, 0, 375)

            pca.set_angle(channel=0, angle_deg=270.0, pulse_min=150, pulse_max=600, servo_range=270)
            pca.set_pwm.assert_called_with(0, 0, 600)

            # Joint 1, 2, 3: TD 8130MG / MG 996R, range=180, pulse_min=150, pulse_max=600
            for ch in [1, 2, 3]:
                pca.set_angle(channel=ch, angle_deg=90.0, pulse_min=150, pulse_max=600, servo_range=180)
                pca.set_pwm.assert_called_with(ch, 0, 375)

    # -------------------------------------------------------------------------
    # 5. 4-DOF Arm Scanner & Target Tracker Integration Tests
    # -------------------------------------------------------------------------
    def test_arm_scanner_4dof_trajectory_generation(self):
        """Verify ArmScannerNode generates valid 4-DOF joint targets along scan path."""
        from ecobot_arm_control.arm_scanner_node import ArmScannerNode

        scanner = ArmScannerNode()
        cmd = String()
        cmd.data = json.dumps({'action': 'scan', 'x': 0.20, 'y': 0.0, 'z': 0.40, 'plant_type': 'test_crop'})
        scanner._scanner_cmd_cb(cmd)

        self.assertTrue(scanner._scanning)
        self.assertGreater(len(scanner._scan_queue), 0)

        # Verify all 4-DOF waypoints in queue have valid angles within physical limits
        for label, sx, sy, sz, ax, ay, az, angles, aim_err, dwell in scanner._scan_queue:
            self.assertEqual(len(angles), 4, f"Waypoint {label} must have 4 joint angles")
            self.assertTrue(within_limits(angles), f"Waypoint {label} angles {angles} outside joint limits")

        scanner._stop_scan()
        scanner.destroy_node()

    def test_arm_target_tracker_4dof_goal(self):
        """Verify ArmTargetTracker transforms detections and commands 4-DOF pose goals."""
        from ecobot_arm_control.arm_target_tracker import ArmTargetTracker

        tracker = ArmTargetTracker()
        tracker._tracking = True

        published_goals = []
        tracker._pose_pub.publish = lambda msg: published_goals.append(list(msg.data))

        # Detections arrive in the camera optical frame (x right, y down,
        # z forward), which _cam_to_arm remaps to the arm frame. These values
        # put the object at arm (0.30, 0, 0.45), inside the workspace.
        det_msg = String()
        det_msg.data = json.dumps([{
            'class': 'bottle',
            'x': 0.0,
            'y': -0.45,
            'z': 0.30,
            'confidence': 0.90
        }])
        tracker._det_cb(det_msg)
        tracker._tick()

        self.assertGreater(len(published_goals), 0, "Target tracker must publish pose goal for valid detection")
        goal = published_goals[0]
        self.assertEqual(len(goal), 3, "Pose goal must be 3D Cartesian coordinates [x, y, z]")
        self.assertAlmostEqual(goal[0], 0.20, delta=0.05)

        tracker.destroy_node()

    # -------------------------------------------------------------------------
    # 6. URDF 4-DOF Kinematic Chain Consistency Tests
    # -------------------------------------------------------------------------
    def test_urdf_consistency(self):
        """Verify the URDF file defines the 4 revolute joints matching the kinematic controller."""
        import os
        from ament_index_python.packages import get_package_share_directory

        urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'urdf', 'ecobot_arm.urdf'
        )
        self.assertTrue(os.path.exists(urdf_path), f"URDF file not found at {urdf_path}")

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Find all revolute joints
        revolute_joints = [
            j for j in root.findall('joint')
            if j.get('type') == 'revolute'
        ]
        self.assertEqual(len(revolute_joints), 4, "URDF must define exactly 4 revolute joints")

        joint_names = [j.get('name') for j in revolute_joints]
        self.assertIn('arm_base_joint', joint_names)
        self.assertIn('arm_shoulder_joint', joint_names)
        self.assertIn('arm_elbow_joint', joint_names)
        self.assertIn('arm_wrist_joint', joint_names)


if __name__ == '__main__':
    unittest.main()
