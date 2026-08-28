import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
import json
import math

from .pca9685_driver import PCA9685
from .servo_config import (
    JOINTS, NUM_JOINTS, to_servo, to_ik, within_limits, apply_overrides,
)
from .arm_kinematics import ArmKinematics


class ArmManualNode(Node):
    def __init__(self):
        super().__init__('arm_manual_node')

        self.declare_parameter('i2c_bus', 7)
        self.declare_parameter('pca9685_address', 0x40)
        self.declare_parameter('pwm_freq', 50.0)
        self.declare_parameter('cmd_timeout', 0.0)
        self.declare_parameter('move_interval_ms', 15)
        self.declare_parameter('home_ramp_steps', 50)
        self.declare_parameter('l0', 0.110)
        self.declare_parameter('l1', 0.150)
        self.declare_parameter('l2', 0.130)
        self.declare_parameter('l3', 0.140)
        # Smooth trapezoidal velocity profile. peak_speed deg/s, accel deg/s^2.
        self.declare_parameter('peak_speed', 90.0)
        self.declare_parameter('accel', 250.0)
        self.declare_parameter('overrides', '{}')

        self._i2c_bus = self.get_parameter('i2c_bus').value
        self._pca_addr = self.get_parameter('pca9685_address').value
        self._pwm_freq = self.get_parameter('pwm_freq').value
        self._cmd_timeout = self.get_parameter('cmd_timeout').value
        self._move_interval = self.get_parameter('move_interval_ms').value / 1000.0
        self._home_ramp_steps = self.get_parameter('home_ramp_steps').value
        self._peak_speed = self.get_parameter('peak_speed').value
        self._accel = self.get_parameter('accel').value
        try:
            apply_overrides(json.loads(
                str(self.get_parameter('overrides').value)))
        except Exception as e:
            self.get_logger().warn(f'ignoring servo overrides: {e}')

        self._current: list[float] = [float(j['home_angle']) for j in JOINTS]
        self._target: list[float] = list(self._current)

        self._pca = None
        self._enabled = False
        self._manually_disabled = False
        self._last_cmd_time = None
        self._ramping_home = False
        self._ramp_step = 0
        self._ramp_start = list(self._current)
        self._force_write = False

        # Smooth trapezoidal trajectory state.
        self._traj_dur = 0.0
        self._traj_t0 = None
        self._traj_T_acc = 0.0
        self._traj_d_acc = 0.0
        self._traj_D = 0.0
        self._traj_peak_n = 0.0
        self._traj_start = list(self._current)

        l0 = self.get_parameter('l0').value
        l1 = self.get_parameter('l1').value
        l2 = self.get_parameter('l2').value
        l3 = self.get_parameter('l3').value
        self._ik = ArmKinematics(l0, l1, l2, l3)

        try:
            self._pca = PCA9685(
                bus=self._i2c_bus,
                address=self._pca_addr,
                freq=self._pwm_freq,
            )
            self._enabled = True
            self.get_logger().info(
                f'PCA9685 initialized on I2C bus {self._i2c_bus}, '
                f'address 0x{self._pca_addr:02X}, freq {self._pwm_freq}Hz'
            )
            self._seed_current_from_hardware()
        except Exception as e:
            self.get_logger().error(f'Failed to init PCA9685: {e}')
            self.get_logger().warn(
                'Running in dry-run mode (no I2C). '
                'Check wiring and run: sudo i2cdetect -y -r 7'
            )

        self._cmd_sub = self.create_subscription(
            Float64MultiArray, '/arm/joint_commands',
            self._cmd_cb, 10)

        self._enable_sub = self.create_subscription(
            String, '/arm/enable',
            self._enable_cb, 10)

        self._pose_sub = self.create_subscription(
            Float64MultiArray, '/arm/pose_goal',
            self._pose_cb, 10)

        self._joint_pub = self.create_publisher(
            JointState, '/joint_states', 10)

        self._angle_pub = self.create_publisher(
            Float64MultiArray, '/arm/joint_angles', 10)

        self._pose_pub = self.create_publisher(
            Float64MultiArray, '/arm/pose', 10)

        self._status_pub = self.create_publisher(
            String, '/arm/status', 10)

        self._timer = self.create_timer(
            self._move_interval, self._timer_cb)

        self._start_home_ramp()

    def _seed_current_from_hardware(self):
        """Read back each channel's held pulse so the home ramp starts from
        where the arm actually is, not wherever JOINTS says home is.

        The PCA9685 keeps outputting its last commanded pulse across node
        restarts, so on a fresh start _current defaulting to the home angles
        made the ramp a no-op: diff was always zero and the arm snapped
        straight to home instead of ramping.
        """
        for i, j in enumerate(JOINTS):
            try:
                _, off = self._pca.get_pwm(j['channel'])
            except Exception as e:
                self.get_logger().warn(
                    f'Could not read back ch{j["channel"]} pulse, '
                    f'assuming home: {e}')
                continue
            if off <= 0:
                continue
            span = j['pulse_max'] - j['pulse_min']
            if span == 0:
                continue
            angle = (off - j['pulse_min']) / span * j['servo_range']
            self._current[i] = max(0.0, min(angle, float(j['servo_range'])))
        self._ramp_start = list(self._current)
        self.get_logger().info(f'Seeded current angles from hardware: {self._current}')

    def _start_home_ramp(self):
        self._ramping_home = True
        self._target = [float(j['home_angle']) for j in JOINTS]
        self._force_write = True
        self._plan_traj()
        self.get_logger().info(f'Ramping to home: {self._target}')

    def _cmd_cb(self, msg):
        self._ramping_home = False
        if len(msg.data) < NUM_JOINTS:
            self.get_logger().warn(
                f'Expected {NUM_JOINTS} angles, got {len(msg.data)}')
            return

        if all(v < 0 for v in msg.data[:NUM_JOINTS]):
            self._disable_servos()
            return

        self._manually_disabled = False

        if not self._enabled and self._pca is not None:
            try:
                self._pca.enable()
                self._enabled = True
                self._force_write = True
                self.get_logger().info('Servos re-enabled')
            except Exception as e:
                self.get_logger().error(f'Re-enable failed: {e}')
                self._pub_status('error')
                return

        for i in range(NUM_JOINTS):
            self._target[i] = float(max(
                JOINTS[i]['min_angle'],
                min(msg.data[i], JOINTS[i]['max_angle']),
            ))
        self._plan_traj()
        self._last_cmd_time = self.get_clock().now()
        self._pub_status('enabled')

    def _enable_cb(self, msg):
        if msg.data == 'enable':
            self._manually_disabled = False
            if not self._enabled and self._pca is not None:
                try:
                    self._pca.enable()
                    self._enabled = True
                    self._force_write = True
                    self._last_cmd_time = self.get_clock().now()
                    self.get_logger().info('Servos enabled via topic')
                    self._pub_status('enabled')
                except Exception as e:
                    self.get_logger().error(f'Enable failed: {e}')
        elif msg.data == 'disable':
            self._disable_servos()

    def _pose_cb(self, msg):
        self._ramping_home = False
        if len(msg.data) < 3:
            self.get_logger().warn(
                f'Expected [x, y, z], got {len(msg.data)} values')
            return

        x, y, z = msg.data[0], msg.data[1], msg.data[2]

        # Joint limits live in servo space; shift them into the IK frame so the
        # solver only returns poses the servos can actually hold.
        lo = to_ik([JOINTS[i]['min_angle'] for i in range(NUM_JOINTS)])
        hi = to_ik([JOINTS[i]['max_angle'] for i in range(NUM_JOINTS)])

        result = self._ik.inverse(
            x, y, z,
            theta2_min=lo[1], theta2_max=hi[1],
            theta3_min=lo[2], theta3_max=hi[2],
            theta4_min=lo[3], theta4_max=hi[3],
        )

        if result is None:
            self.get_logger().warn(
                f'Unreachable pose: ({x:.3f}, {y:.3f}, {z:.3f})')
            return

        angles = to_servo(result)
        if not within_limits(angles):
            self.get_logger().warn(
                f'Pose ({x:.3f}, {y:.3f}, {z:.3f}) needs base angle '
                f'{angles[0]:.0f}, outside joint limits')
            return

        th1, th2, th3, th4 = angles

        self.get_logger().info(
            f'Pose goal ({x:.3f},{y:.3f},{z:.3f}) '
            f'-> angles [{th1:.0f},{th2:.0f},{th3:.0f},{th4:.0f}]')

        self._manually_disabled = False
        if not self._enabled and self._pca is not None:
            try:
                self._pca.enable()
                self._enabled = True
                self._force_write = True
            except Exception as e:
                self.get_logger().error(f'Enable failed: {e}')
                return

        for i, ang in enumerate([th1, th2, th3, th4]):
            self._target[i] = float(ang)
        self._plan_traj()
        self._last_cmd_time = self.get_clock().now()
        self._pub_status('enabled')

    def _disable_servos(self):
        if self._pca is not None and self._enabled:
            try:
                self._pca.disable()
            except Exception as e:
                self.get_logger().warn(f'Disable warning: {e}')
        self._enabled = False
        self._manually_disabled = True
        self._last_cmd_time = None
        self.get_logger().info('Servos disabled')
        self._pub_status('disabled')

    def _pub_status(self, status):
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _plan_traj(self, start=None):
        """Synchronize a smooth trapezoidal velocity trajectory from the given
        (or current) joint state to _target, so all four joints depart and
        arrive together with an acceleration phase instead of jerking each
        joint independently at a fixed per-tick step."""
        if start is None:
            start = list(self._current)
        max_disp = 0.0
        for i in range(NUM_JOINTS):
            max_disp = max(max_disp, abs(self._target[i] - start[i]))

        if max_disp < 0.1:
            self._traj_dur = 0.0
            self._traj_t0 = None
            for i in range(NUM_JOINTS):
                self._current[i] = float(self._target[i])
            return

        self._peak_speed = self.get_parameter('peak_speed').value
        self._accel = self.get_parameter('accel').value
        v = max(1.0, float(self._peak_speed))
        a = max(1.0, float(self._accel))
        t_acc = v / a
        d_acc = 0.5 * a * t_acc * t_acc
        if max_disp >= 2.0 * d_acc:
            t_cruise = (max_disp - 2.0 * d_acc) / v
            self._traj_dur = 2.0 * t_acc + t_cruise
            self._traj_peak_n = v
            self._traj_T_acc = t_acc
            self._traj_d_acc = d_acc
            self._traj_D = max_disp
        else:
            # Triangular profile (never reach cruise). v_peak = sqrt(a*D).
            vp = math.sqrt(a * max_disp)
            t_acc = vp / a
            self._traj_dur = 2.0 * t_acc
            self._traj_peak_n = vp
            self._traj_T_acc = t_acc
            self._traj_d_acc = 0.5 * a * t_acc * t_acc
            self._traj_D = max_disp

        self._traj_start = list(start)
        self._traj_t0 = self.get_clock().now()
        self.get_logger().info(
            f'Traj to {[round(float(t),0) for t in self._target]} '
            f'dur={self._traj_dur:.2f}s')

    def _traj_positions(self, t_elapsed):
        """Joint positions along the live trapezoidal profile at elapsed sec."""
        D = self._traj_D
        T = self._traj_dur
        if D < 1e-6 or T <= 0:
            return list(self._traj_start)
        t = max(0.0, min(t_elapsed, T))
        T_acc = self._traj_T_acc
        v = self._traj_peak_n
        if t <= T_acc:
            s = 0.5 * (v / T_acc) * t * t
        elif t <= T - T_acc:
            s = self._traj_d_acc + v * (t - T_acc)
        else:
            tre = T - t
            s = D - 0.5 * (v / T_acc) * tre * tre
        frac = max(0.0, min(1.0, s / D))
        return [
            self._traj_start[i] + frac * (self._target[i] - self._traj_start[i])
            for i in range(NUM_JOINTS)
        ]

    def _timer_cb(self):
        now = self.get_clock().now()

        # Home ramp uses the same smooth trajectory machinery.
        if self._ramping_home:
            if self._traj_t0 is None or self._traj_dur <= 0.0:
                self._plan_traj()
            elapsed = (now - self._traj_t0).nanoseconds * 1e-9
            if elapsed >= self._traj_dur:
                self._current = [float(v) for v in self._target]
                for i in range(NUM_JOINTS):
                    self._write_servo(i, self._current[i])
                self._ramping_home = False
                self._traj_t0 = None
                self._last_cmd_time = now
                self.get_logger().info(
                    f'Home reached: {[round(v,0) for v in self._current]}')
                self._pub_status('enabled')
            else:
                self._current = self._traj_positions(elapsed)
                for i in range(NUM_JOINTS):
                    self._write_servo(i, self._current[i])
            self._publish_state()
            return

        if self._last_cmd_time is not None and self._cmd_timeout > 0:
            elapsed = (now - self._last_cmd_time).nanoseconds * 1e-9
            if elapsed > self._cmd_timeout:
                if self._enabled and self._pca is not None:
                    try:
                        self._pca.disable()
                    except Exception:
                        pass
                    self._enabled = False
                    self.get_logger().warn(
                        f'Command timeout ({elapsed:.1f}s),'
                        ' disabling servos')
                    self._pub_status('disabled')
                self._publish_state()
                return

        if self._force_write:
            self._force_write = False
            for i in range(NUM_JOINTS):
                self._write_servo(i, self._current[i])
            self._publish_state()

        # Follow the active trajectory toward the latest target.
        if self._traj_t0 is not None and self._traj_dur > 0.0:
            elapsed = (now - self._traj_t0).nanoseconds * 1e-9
            if elapsed >= self._traj_dur:
                self._current = [float(v) for v in self._target]
                self._traj_t0 = None
                for i in range(NUM_JOINTS):
                    self._write_servo(i, self._current[i])
            else:
                self._current = self._traj_positions(elapsed)
                if self._enabled:
                    for i in range(NUM_JOINTS):
                        self._write_servo(i, self._current[i])
            self._publish_state()
            return

        moved = False
        for i in range(NUM_JOINTS):
            if abs(self._current[i] - self._target[i]) < 0.1:
                continue
            speed = JOINTS[i]['speed']
            if not self._enabled:
                speed = max(speed, 3)
            if self._current[i] < self._target[i]:
                self._current[i] = min(
                    self._current[i] + speed, self._target[i])
            else:
                self._current[i] = max(
                    self._current[i] - speed, self._target[i])
            self._write_servo(i, self._current[i])
            moved = True

        self._publish_state()

    def _write_servo(self, index: int, angle: float):
        if self._pca is None or not self._enabled:
            return
        j = JOINTS[index]
        try:
            self._pca.set_angle(
                j['channel'], angle,
                j['pulse_min'], j['pulse_max'], j['servo_range'],
            )
        except Exception as e:
            self.get_logger().error(
                f'I2C write error on ch {j["channel"]}: {e}')

    def _publish_state(self):
        now = self.get_clock().now()
        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = [j['name'] for j in JOINTS]
        js.position = list(self._current)
        self._joint_pub.publish(js)

        fa = Float64MultiArray()
        fa.data = list(self._current)
        self._angle_pub.publish(fa)

        x, y, z = self._ik.forward(*to_ik(self._current))
        pe = Float64MultiArray()
        pe.data = [x, y, z]
        self._pose_pub.publish(pe)

        st_msg = String()
        st_msg.data = 'enabled' if self._enabled else 'disabled'
        self._status_pub.publish(st_msg)

    def destroy_node(self):
        if self._pca is not None:
            try:
                self._pca.disable()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmManualNode()
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
