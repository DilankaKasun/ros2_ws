"""
ROS 2 node for ecobot motor control via serial interface.

Receives ``/cmd_vel`` velocity commands, converts them to per-motor RPM using
differential-drive kinematics, sends command packets over a serial port to the
motor driver, and publishes odometry, joint states, and run-mode information.
"""

import math
import time
import threading
import serial
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8
from tf2_ros import TransformBroadcaster

from ecobot_motor_control.serial_protocol import (
    build_send_packet, parse_receive_packet, cobs_encode, cobs_decode, PACKET_SIZE
)
from ecobot_motor_control.kinematics import (
    twist_to_rpm, encoder_delta_to_twist, CUGOV4_PARAMS
)


class MotorControlNode(Node):
    """
    ROS 2 node that drives a differential-drive ecobot platform.

    Subscribes to ``/cmd_vel`` (Twist), converts to left/right RPM, sends
    serial commands, and reads back encoder feedback to publish ``/odom``,
    ``/joint_states``, ``/run_mode``, and the odom→base_footprint transform.
    """
    def __init__(self):
        super().__init__('motor_control_node')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('max_rpm', 130.0)
        self.declare_parameter('product_id', 1)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        # The tracks scrub when the robot turns, so the turn the encoders
        # report is smaller than the turn the robot actually made — about
        # half of it on this machine. Everything that steers closes its
        # loop on this odometry, so the error has to be taken out here:
        # left in, the reported frame is not a rotated version of the real
        # world but a differently shaped one (drive at an angle and it is
        # recorded at half that angle), and no planner can work in it.
        #
        # MEASURE THIS PER ROBOT. Turn it a known amount by hand — a full
        # circle back to a floor mark is easiest — and divide the true
        # turn by what /odom reported. 1.0 means the encoders are right.
        self.declare_parameter('turn_calibration', 2.0)

        self.serial_port_name = self.get_parameter('serial_port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.control_freq = self.get_parameter('control_frequency').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.max_rpm = self.get_parameter('max_rpm').value
        self.product_id = self.get_parameter('product_id').value
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.base_frame = self.get_parameter('base_frame_id').value
        self.turn_calibration = float(
            self.get_parameter('turn_calibration').value)

        self.params = CUGOV4_PARAMS

        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.cmd_vel_stamp = 0.0
        self.cmd_lock = threading.Lock()

        self.encoder_l = 0
        self.encoder_r = 0
        self.prev_encoder_l = 0
        self.prev_encoder_r = 0
        self.first_encoder = True
        self.run_mode = 1

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0

        self.serial_conn = None
        self.serial_lock = threading.Lock()
        self.serial_reconnect_timer = None
        self.reconnect_count = 0

        self.last_loop_time = time.time()

        counts_per_wheel_rev = self.params.encoder_resolution * self.params.reduction_ratio
        max_counts_per_sec = self.max_rpm / 60.0 * counts_per_wheel_rev
        self.max_allowed_delta = int(max_counts_per_sec / self.control_freq * 1.5) + 1
        self.get_logger().info(
            f'encoder max_allowed_delta={self.max_allowed_delta} counts')

        self._actual_path = Path()
        self._actual_path.header.frame_id = self.odom_frame
        self._predicted_path = Path()
        self._predicted_path.header.frame_id = self.odom_frame
        self._max_path_poses = 500
        self._predicted_steps = 30
        self._predicted_dt = 0.1

        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.run_mode_pub = self.create_publisher(UInt8, '/run_mode', 10)
        self.actual_path_pub = self.create_publisher(Path, '/ecobot/actual_path', 10)
        self.predicted_path_pub = self.create_publisher(Path, '/ecobot/predicted_path', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.control_timer = self.create_timer(1.0 / self.control_freq, self.control_loop)

        self.get_logger().info(
            f'ecobot motor controller starting — port={self.serial_port_name}')
        self.open_serial()

    def open_serial(self):
        """
        Open (or reopen) the serial port connection.

        Closes any existing connection first, then attempts to open the port
        with the configured baudrate.  On failure, logs a warning (once) and
        schedules a background reconnect timer.
        """
        with self.serial_lock:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
            try:
                self.serial_conn = serial.Serial(
                    port=self.serial_port_name,
                    baudrate=self.baudrate,
                    timeout=0.05,
                )
                self.get_logger().info(f'serial port {self.serial_port_name} opened')
                self.reconnect_count = 0
            except serial.SerialException:
                self.reconnect_count += 1
                if self.reconnect_count == 1:
                    self.get_logger().warn(
                        f'serial port {self.serial_port_name} not available — '
                        f'will retry in background')
                self.serial_conn = None
                self.schedule_reconnect()

    def schedule_reconnect(self):
        """Schedule a periodic retry timer that calls :meth:`open_serial` every 3 seconds."""
        if self.serial_reconnect_timer is None:
            self.serial_reconnect_timer = self.create_timer(3.0, self.open_serial)

    def cmd_vel_callback(self, msg: Twist):
        """
        Store the latest commanded linear-x and angular-z speeds and dispatch immediately.
        """
        now = time.time()
        with self.cmd_lock:
            self.cmd_linear = msg.linear.x
            self.cmd_angular = msg.angular.z
            self.cmd_vel_stamp = now
            l_rpm, r_rpm = twist_to_rpm(self.cmd_linear, self.cmd_angular, self.params)
            l_rpm = max(-self.max_rpm, min(self.max_rpm, l_rpm))
            r_rpm = max(-self.max_rpm, min(self.max_rpm, r_rpm))

        # Instant sub-millisecond serial dispatch (<0.3ms to microcontroller)
        packet = build_send_packet(l_rpm, r_rpm, self.product_id)
        self.write_serial(packet)

    def read_serial(self):
        """
        Read and decode one packet from the serial port.

        Returns
        -------
        bytes or None
            COBS-decoded raw packet data, or ``None`` if no valid packet was
            available or the serial port is disconnected.
        """
        with self.serial_lock:
            if self.serial_conn is None or not self.serial_conn.is_open:
                return None
            try:
                raw = self.serial_conn.read_until(b'\x00')
                if not raw:
                    return None
                return cobs_decode(raw)
            except serial.SerialException:
                self.get_logger().warning('serial read error')
                self.serial_conn = None
                self.schedule_reconnect()
                return None

    def write_serial(self, data: bytes):
        """
        COBS-encode *data* and write it to the serial port.

        Parameters
        ----------
        data : bytes
            Raw (unencoded) packet payload to send.

        Returns
        -------
        bool
            ``True`` if the write succeeded, ``False`` otherwise.
        """
        with self.serial_lock:
            if self.serial_conn is None or not self.serial_conn.is_open:
                return False
            try:
                encoded = cobs_encode(data)
                self.serial_conn.write(encoded)
                return True
            except serial.SerialException:
                self.serial_conn = None
                self.schedule_reconnect()
                return False

    def control_loop(self):
        """
        Periodic control-loop callback (timer-driven).

        Converts the most recent ``/cmd_vel`` to per-motor RPM, sends a
        command packet, reads encoder feedback, and updates odometry.
        Commands are zeroed if the last velocity message is older than
        ``cmd_vel_timeout``.
        """
        now = time.time()
        dt = now - self.last_loop_time
        self.last_loop_time = now
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / self.control_freq

        with self.cmd_lock:
            if now - self.cmd_vel_stamp > self.cmd_vel_timeout:
                l_rpm = 0.0
                r_rpm = 0.0
            else:
                l_rpm, r_rpm = twist_to_rpm(
                    self.cmd_linear, self.cmd_angular, self.params)
                l_rpm = max(-self.max_rpm, min(self.max_rpm, l_rpm))
                r_rpm = max(-self.max_rpm, min(self.max_rpm, r_rpm))

        packet = build_send_packet(l_rpm, r_rpm, self.product_id)
        self.write_serial(packet)

        raw = self.read_serial()
        if raw is not None:
            try:
                fb = parse_receive_packet(raw)
            except ValueError:
                return
            self.run_mode = fb['run_mode']
            self.encoder_l = fb['encoder_l']
            self.encoder_r = fb['encoder_r']
            self.process_encoder_data(dt)

    def process_encoder_data(self, dt: float):
        """
        Publish joint states and update odometry from encoder counts.

        Parameters
        ----------
        dt : float
            Time delta (seconds) since the last encoder reading.  Used to
            compute velocities from encoder deltas.
        """
        mode_msg = UInt8()
        mode_msg.data = self.run_mode
        self.run_mode_pub.publish(mode_msg)

        dist_per_count = 2.0 * math.pi / (
            self.params.encoder_resolution * self.params.reduction_ratio)
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['left_crawler_joint', 'right_crawler_joint']
        joint_msg.position = [
            self.encoder_l * dist_per_count,
            self.encoder_r * dist_per_count,
        ]
        self.joint_state_pub.publish(joint_msg)

        if self.first_encoder:
            self.first_encoder = False
            self.prev_encoder_l = self.encoder_l
            self.prev_encoder_r = self.encoder_r
            return

        diff_l = self.encoder_l - self.prev_encoder_l
        diff_r = self.encoder_r - self.prev_encoder_r

        # Discard an implausible jump, but still take the new reading as the
        # baseline for the next one. Rolling the encoder back to prev instead
        # pinned prev in place, so every later delta was measured from that
        # stale value and grew until it tripped the guard too — one glitch
        # latched odometry off for the rest of the session, leaving the robot
        # reporting itself stationary while the wheels turned.
        if abs(diff_l) > self.max_allowed_delta:
            self.get_logger().warn(
                f'outlier left encoder delta={diff_l} > '
                f'{self.max_allowed_delta}; skipping this sample')
            diff_l = 0

        if abs(diff_r) > self.max_allowed_delta:
            self.get_logger().warn(
                f'outlier right encoder delta={diff_r} > '
                f'{self.max_allowed_delta}; skipping this sample')
            diff_r = 0

        linear_x, angular_z = encoder_delta_to_twist(
            diff_l, diff_r, dt, self.params)
        # Take out the scrubbing error before anything sees this reading.
        angular_z *= self.turn_calibration

        self.pose_yaw += angular_z * dt
        self.pose_x += linear_x * dt * math.cos(self.pose_yaw)
        self.pose_y += linear_x * dt * math.sin(self.pose_yaw)

        self.publish_odom(linear_x, angular_z)

        self.prev_encoder_l = self.encoder_l
        self.prev_encoder_r = self.encoder_r

    def publish_odom(self, linear_x: float, angular_z: float):
        """
        Publish an odometry message and the odom→base_footprint transform.

        Parameters
        ----------
        linear_x : float
            Current forward velocity (m/s).
        angular_z : float
            Current angular velocity (rad/s).
        """
        stamp = self.get_clock().now().to_msg()

        qz = math.sin(self.pose_yaw / 2.0)
        qw = math.cos(self.pose_yaw / 2.0)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.pose_x
        odom.pose.pose.position.y = self.pose_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = linear_x
        odom.twist.twist.angular.z = angular_z
        self.odom_pub.publish(odom)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = self.pose_x
        tf_msg.transform.translation.y = self.pose_y
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_msg)

        self._update_actual_path(stamp)
        self._update_predicted_path(stamp, linear_x, angular_z)

    def _update_actual_path(self, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.odom_frame
        pose.pose.position.x = self.pose_x
        pose.pose.position.y = self.pose_y
        pose.pose.orientation.z = math.sin(self.pose_yaw / 2.0)
        pose.pose.orientation.w = math.cos(self.pose_yaw / 2.0)
        self._actual_path.header.stamp = stamp
        self._actual_path.poses.append(pose)
        if len(self._actual_path.poses) > self._max_path_poses:
            self._actual_path.poses = self._actual_path.poses[-self._max_path_poses:]
        self.actual_path_pub.publish(self._actual_path)

    def _update_predicted_path(self, stamp, linear_x, angular_z):
        self._predicted_path.header.stamp = stamp
        self._predicted_path.poses = []
        px, py, pyaw = self.pose_x, self.pose_y, self.pose_yaw
        for i in range(1, self._predicted_steps + 1):
            t = i * self._predicted_dt
            if abs(angular_z) > 1e-6:
                r = linear_x / angular_z
                px = self.pose_x - r * math.sin(self.pose_yaw) + r * math.sin(self.pose_yaw + angular_z * t)
                py = self.pose_y + r * math.cos(self.pose_yaw) - r * math.cos(self.pose_yaw + angular_z * t)
                pyaw = self.pose_yaw + angular_z * t
            else:
                px = self.pose_x + linear_x * t * math.cos(pyaw)
                py = self.pose_y + linear_x * t * math.sin(pyaw)
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.odom_frame
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.orientation.z = math.sin(pyaw / 2.0)
            ps.pose.orientation.w = math.cos(pyaw / 2.0)
            self._predicted_path.poses.append(ps)
        self.predicted_path_pub.publish(self._predicted_path)

    def destroy_node(self):
        """Close the serial connection, then call the parent cleanup."""
        with self.serial_lock:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
        super().destroy_node()


def main(args=None):
    """
    Entry point: initialise rclpy, create the node, and spin.

    Shuts down cleanly on keyboard interrupt.
    """
    rclpy.init(args=args)
    node = MotorControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
