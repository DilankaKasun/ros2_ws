#!/usr/bin/env python3
"""
EcoBot Hardware Diagnostic Node.

Periodically monitors and publishes hardware health status across the robot:
  - Publishes standard ROS 2 `diagnostic_msgs/msg/DiagnosticArray` to `/diagnostics`
  - Publishes clean JSON status to `/ecobot/hardware_status` (std_msgs/msg/String) for web UI
  - Provides `/ecobot/run_hardware_check` service (std_srvs/srv/Trigger) for on-demand self-tests
"""

import json
import os
import platform
import socket
import time
from typing import Dict, Any, List

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import String, UInt8
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Odometry

# Optional hardware libraries
try:
    import psutil
except ImportError:
    psutil = None

try:
    import smbus2
except ImportError:
    smbus2 = None

try:
    import sounddevice as sd
except ImportError:
    sd = None


class HardwareDiagnosticNode(Node):
    def __init__(self):
        super().__init__('hardware_diagnostic_node')

        # Parameters
        self.declare_parameter('check_interval', 5.0)
        self.declare_parameter('i2c_bus', 7)
        self.declare_parameter('pca9685_address', 0x40)
        self.declare_parameter('motor_serial_port', '/dev/ttyACM0')
        self.declare_parameter('tof_serial_port', '/dev/ttyUSB0')

        self.interval = float(self.get_parameter('check_interval').value)
        self.i2c_bus = int(self.get_parameter('i2c_bus').value)
        self.pca9685_addr = int(self.get_parameter('pca9685_address').value)
        self.motor_port = str(self.get_parameter('motor_serial_port').value)
        self.tof_port = str(self.get_parameter('tof_serial_port').value)

        # Topic monitoring trackers
        self._last_topics = {
            'realsense_color': {'stamp': 0.0, 'count': 0, 'fps': 0.0, 'width': 0, 'height': 0},
            'realsense_depth': {'stamp': 0.0, 'count': 0, 'fps': 0.0, 'width': 0, 'height': 0},
            'arm_camera': {'stamp': 0.0, 'count': 0, 'fps': 0.0, 'width': 0, 'height': 0},
            'odom': {'stamp': 0.0, 'count': 0, 'fps': 0.0, 'linear_vel': 0.0, 'angular_vel': 0.0},
            'joint_states': {'stamp': 0.0, 'count': 0, 'fps': 0.0},
            'run_mode': {'stamp': 0.0, 'val': 0},
            'tof_ranges': {'stamp': 0.0, 'data': None},
            'arm_status': {'stamp': 0.0, 'val': 'unknown'},
        }

        # Subscriptions for live topic monitoring
        self.create_subscription(Image, '/camera/color/image_raw', self._cb_realsense_color, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self._cb_realsense_depth, 10)
        self.create_subscription(Image, '/arm/camera/image_raw', self._cb_arm_camera, 10)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(JointState, '/joint_states', self._cb_joint_states, 10)
        self.create_subscription(UInt8, '/run_mode', self._cb_run_mode, 10)
        self.create_subscription(String, '/ecobot/tof_ranges', self._cb_tof_ranges, 10)
        self.create_subscription(String, '/arm/status', self._cb_arm_status, 10)

        # Publishers
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.status_pub = self.create_publisher(String, '/ecobot/hardware_status', 10)

        # Services for triggering on-demand self-tests
        self.srv_check = self.create_service(
            Trigger, '/ecobot/run_hardware_check', self._handle_trigger_check
        )
        self.srv_check_alias = self.create_service(
            Trigger, '/ecobot/trigger_hardware_check', self._handle_trigger_check
        )

        # Periodic timer
        self.timer = self.create_timer(self.interval, self._diagnostic_cycle)
        self._last_cycle_time = time.time()

        self.get_logger().info(
            f'Hardware Diagnostic Node started (Interval: {self.interval}s, I2C: {self.i2c_bus}@0x{self.pca9685_addr:02X})'
        )

    # --------------------------------------------------------------------------
    # Topic Callbacks for FPS & Activity Tracking
    # --------------------------------------------------------------------------
    def _cb_realsense_color(self, msg: Image):
        t = self._last_topics['realsense_color']
        t['stamp'] = time.time()
        t['count'] += 1
        t['width'] = msg.width
        t['height'] = msg.height

    def _cb_realsense_depth(self, msg: Image):
        t = self._last_topics['realsense_depth']
        t['stamp'] = time.time()
        t['count'] += 1
        t['width'] = msg.width
        t['height'] = msg.height

    def _cb_arm_camera(self, msg: Image):
        t = self._last_topics['arm_camera']
        t['stamp'] = time.time()
        t['count'] += 1
        t['width'] = msg.width
        t['height'] = msg.height

    def _cb_odom(self, msg: Odometry):
        t = self._last_topics['odom']
        t['stamp'] = time.time()
        t['count'] += 1
        t['linear_vel'] = round(msg.twist.twist.linear.x, 3)
        t['angular_vel'] = round(msg.twist.twist.angular.z, 3)

    def _cb_joint_states(self, msg: JointState):
        t = self._last_topics['joint_states']
        t['stamp'] = time.time()
        t['count'] += 1

    def _cb_run_mode(self, msg: UInt8):
        self._last_topics['run_mode']['stamp'] = time.time()
        self._last_topics['run_mode']['val'] = msg.data

    def _cb_tof_ranges(self, msg: String):
        t = self._last_topics['tof_ranges']
        t['stamp'] = time.time()
        try:
            t['data'] = json.loads(msg.data)
        except Exception:
            pass

    def _cb_arm_status(self, msg: String):
        self._last_topics['arm_status']['stamp'] = time.time()
        self._last_topics['arm_status']['val'] = msg.data

    # --------------------------------------------------------------------------
    # Diagnostic Processing Cycle
    # --------------------------------------------------------------------------
    def _diagnostic_cycle(self):
        now = time.time()
        dt = max(0.001, now - self._last_cycle_time)
        self._last_cycle_time = now

        # Compute topic frame rates
        for k in ['realsense_color', 'realsense_depth', 'arm_camera', 'odom', 'joint_states']:
            t = self._last_topics[k]
            t['fps'] = round(t['count'] / dt, 1)
            t['count'] = 0

        # Build Diagnostic Message
        diag_arr = DiagnosticArray()
        diag_arr.header.stamp = self.get_clock().now().to_msg()
        
        statuses: List[DiagnosticStatus] = []
        ui_summary: Dict[str, Any] = {
            'timestamp': now,
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall': 'HEALTHY',
            'failures_count': 0,
            'warnings_count': 0,
            'subsystems': {},
        }

        # 1. System & Compute Health
        sys_status, sys_dict = self._check_system()
        statuses.append(sys_status)
        ui_summary['subsystems']['system'] = sys_dict

        # 2. GPU & AI Acceleration
        gpu_status, gpu_dict = self._check_gpu()
        statuses.append(gpu_status)
        ui_summary['subsystems']['gpu'] = gpu_dict

        # 3. Base Motor Controller & Encoders
        motor_status, motor_dict = self._check_motors(now)
        statuses.append(motor_status)
        ui_summary['subsystems']['motors'] = motor_dict

        # 4. RealSense D415 Depth Camera
        rs_status, rs_dict = self._check_realsense(now)
        statuses.append(rs_status)
        ui_summary['subsystems']['realsense'] = rs_dict

        # 5. USB Arm Camera
        arm_cam_status, arm_cam_dict = self._check_arm_camera(now)
        statuses.append(arm_cam_status)
        ui_summary['subsystems']['arm_camera'] = arm_cam_dict

        # 6. PCA9685 Arm Servo Controller
        servo_status, servo_dict = self._check_pca9685()
        statuses.append(servo_status)
        ui_summary['subsystems']['arm_servos'] = servo_dict

        # 7. ToF Sensors
        tof_status, tof_dict = self._check_tof(now)
        statuses.append(tof_status)
        ui_summary['subsystems']['tof'] = tof_dict

        # 8. Audio Devices
        audio_status, audio_dict = self._check_audio()
        statuses.append(audio_status)
        ui_summary['subsystems']['audio'] = audio_dict

        # Determine overall system health
        err_count = sum(1 for s in statuses if s.level == DiagnosticStatus.ERROR)
        warn_count = sum(1 for s in statuses if s.level == DiagnosticStatus.WARN)
        ui_summary['failures_count'] = err_count
        ui_summary['warnings_count'] = warn_count

        if err_count > 0:
            ui_summary['overall'] = 'CRITICAL'
        elif warn_count > 0:
            ui_summary['overall'] = 'DEGRADED'
        else:
            ui_summary['overall'] = 'HEALTHY'

        # Publish /diagnostics
        diag_arr.status = statuses
        self.diag_pub.publish(diag_arr)

        # Publish /ecobot/hardware_status for web dashboard
        status_msg = String()
        status_msg.data = json.dumps(ui_summary)
        self.status_pub.publish(status_msg)

    # --------------------------------------------------------------------------
    # Subsystem Check Helpers
    # --------------------------------------------------------------------------
    def _check_system(self) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: System & Compute'
        stat.hardware_id = platform.node()

        data = {'status': 'PASS'}
        cpu_pct = psutil.cpu_percent() if psutil else 0.0
        mem_pct = psutil.virtual_memory().percent if psutil else 0.0
        disk_pct = psutil.disk_usage('/').percent if psutil else 0.0

        # Read Thermals
        thermals = {}
        max_t = 0.0
        thermal_dir = '/sys/devices/virtual/thermal'
        if os.path.exists(thermal_dir):
            for z in sorted(os.listdir(thermal_dir)):
                if z.startswith('thermal_zone'):
                    t_type_f = os.path.join(thermal_dir, z, 'type')
                    t_temp_f = os.path.join(thermal_dir, z, 'temp')
                    if os.path.exists(t_type_f) and os.path.exists(t_temp_f):
                        try:
                            with open(t_type_f, 'r') as f1, open(t_temp_f, 'r') as f2:
                                name = f1.read().strip()
                                val = float(f2.read().strip()) / 1000.0
                                thermals[name] = round(val, 1)
                                if val > max_t:
                                    max_t = val
                        except Exception:
                            pass

        cpu_t = thermals.get('cpu-thermal', thermals.get('CPU-therm', 0.0))
        gpu_t = thermals.get('gpu-thermal', thermals.get('GPU-therm', 0.0))

        stat.values = [
            KeyValue(key='CPU Usage %', value=f'{cpu_pct:.1f}'),
            KeyValue(key='RAM Usage %', value=f'{mem_pct:.1f}'),
            KeyValue(key='Disk Usage %', value=f'{disk_pct:.1f}'),
            KeyValue(key='CPU Temp °C', value=f'{cpu_t:.1f}'),
            KeyValue(key='GPU Temp °C', value=f'{gpu_t:.1f}'),
        ]

        if max_t > 85.0 or disk_pct > 95.0 or mem_pct > 95.0:
            stat.level = DiagnosticStatus.ERROR
            stat.message = f'System High Load/Temp (CPU {cpu_t}°C, RAM {mem_pct}%, Disk {disk_pct}%)'
            data['status'] = 'FAIL'
        elif max_t > 75.0 or disk_pct > 85.0 or mem_pct > 85.0:
            stat.level = DiagnosticStatus.WARN
            stat.message = f'System Resource Warning (CPU {cpu_t}°C, RAM {mem_pct}%)'
            data['status'] = 'WARN'
        else:
            stat.level = DiagnosticStatus.OK
            stat.message = f'Normal (CPU {cpu_pct:.0f}%, RAM {mem_pct:.0f}%, CPU {cpu_t}°C)'

        data.update({
            'cpu_pct': cpu_pct, 'mem_pct': mem_pct, 'disk_pct': disk_pct,
            'cpu_temp': cpu_t, 'gpu_temp': gpu_t, 'message': stat.message
        })
        return stat, data

    def _check_gpu(self) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: GPU & CUDA'
        stat.hardware_id = 'Jetson GPU'

        data = {'status': 'PASS', 'cuda': False}
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            data['cuda'] = has_cuda
            data['torch_version'] = torch.__version__
            if has_cuda:
                dev = torch.cuda.get_device_name(0)
                data['device'] = dev
                stat.level = DiagnosticStatus.OK
                stat.message = f'CUDA Available ({dev})'
                stat.values = [
                    KeyValue(key='CUDA Device', value=dev),
                    KeyValue(key='PyTorch Version', value=torch.__version__),
                ]
            else:
                stat.level = DiagnosticStatus.WARN
                stat.message = 'PyTorch installed without CUDA acceleration'
                data['status'] = 'WARN'
        except Exception as e:
            stat.level = DiagnosticStatus.WARN
            stat.message = f'PyTorch/CUDA check note: {e}'
            data['status'] = 'WARN'

        return stat, data

    def _check_motors(self, now: float) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: Base Motors (Pico)'
        stat.hardware_id = self.motor_port

        data = {'status': 'PASS', 'port': self.motor_port}
        odom_t = self._last_topics['odom']
        mode_t = self._last_topics['run_mode']
        age_sec = now - odom_t['stamp'] if odom_t['stamp'] > 0 else 999.0

        stat.values = [
            KeyValue(key='Serial Port', value=self.motor_port),
            KeyValue(key='Odom FPS', value=f'{odom_t["fps"]:.1f}'),
            KeyValue(key='Run Mode', value=str(mode_t['val'])),
            KeyValue(key='Last Message Age (s)', value=f'{age_sec:.1f}'),
        ]

        if odom_t['stamp'] > 0 and age_sec < 3.0:
            stat.level = DiagnosticStatus.OK
            stat.message = f'Pico OK ({odom_t["fps"]} Hz, Mode {mode_t["val"]})'
            data.update({'fps': odom_t['fps'], 'run_mode': mode_t['val'], 'active': True})
        elif os.path.exists(self.motor_port):
            stat.level = DiagnosticStatus.WARN
            stat.message = f'Pico port {self.motor_port} exists but no active /odom stream'
            data.update({'status': 'WARN', 'active': False})
        else:
            stat.level = DiagnosticStatus.ERROR
            stat.message = f'Motor Controller port {self.motor_port} not connected'
            data.update({'status': 'FAIL', 'active': False})

        return stat, data

    def _check_realsense(self, now: float) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: RealSense D415 Camera'
        stat.hardware_id = 'RealSense_D415'

        data = {'status': 'PASS'}
        c_t = self._last_topics['realsense_color']
        d_t = self._last_topics['realsense_depth']
        c_age = now - c_t['stamp'] if c_t['stamp'] > 0 else 999.0

        stat.values = [
            KeyValue(key='Color FPS', value=f'{c_t["fps"]:.1f}'),
            KeyValue(key='Depth FPS', value=f'{d_t["fps"]:.1f}'),
            KeyValue(key='Color Resolution', value=f'{c_t["width"]}x{c_t["height"]}'),
        ]

        if c_t['stamp'] > 0 and c_age < 3.0:
            stat.level = DiagnosticStatus.OK
            stat.message = f'Streaming ({c_t["fps"]} FPS RGB, {d_t["fps"]} FPS Depth)'
            data.update({'color_fps': c_t['fps'], 'depth_fps': d_t['fps'], 'streaming': True})
        else:
            stat.level = DiagnosticStatus.ERROR
            stat.message = 'No active RealSense image stream on /camera/*'
            data.update({'status': 'FAIL', 'streaming': False})

        return stat, data

    def _check_arm_camera(self, now: float) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: USB Arm Camera'
        stat.hardware_id = '/dev/video0'

        data = {'status': 'PASS'}
        a_t = self._last_topics['arm_camera']
        a_age = now - a_t['stamp'] if a_t['stamp'] > 0 else 999.0

        stat.values = [
            KeyValue(key='Arm Cam FPS', value=f'{a_t["fps"]:.1f}'),
            KeyValue(key='Resolution', value=f'{a_t["width"]}x{a_t["height"]}'),
        ]

        if a_t['stamp'] > 0 and a_age < 3.0:
            stat.level = DiagnosticStatus.OK
            stat.message = f'Streaming ({a_t["fps"]} FPS, {a_t["width"]}x{a_t["height"]})'
            data.update({'fps': a_t['fps'], 'streaming': True})
        elif os.path.exists('/dev/video0'):
            stat.level = DiagnosticStatus.WARN
            stat.message = 'Arm camera device /dev/video0 present but no stream'
            data.update({'status': 'WARN', 'streaming': False})
        else:
            stat.level = DiagnosticStatus.ERROR
            stat.message = 'Arm camera video device not found'
            data.update({'status': 'FAIL', 'streaming': False})

        return stat, data

    def _check_pca9685(self) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: Arm PCA9685 PWM Driver'
        stat.hardware_id = f'i2c-{self.i2c_bus}@0x{self.pca9685_addr:02X}'

        data = {'status': 'PASS', 'bus': self.i2c_bus, 'address': hex(self.pca9685_addr)}

        if smbus2 and os.path.exists(f'/dev/i2c-{self.i2c_bus}'):
            try:
                i2c = smbus2.SMBus(self.i2c_bus)
                mode1 = i2c.read_byte_data(self.pca9685_addr, 0x00)
                i2c.close()
                stat.level = DiagnosticStatus.OK
                stat.message = f'PCA9685 Responding on I2C-{self.i2c_bus} (MODE1={hex(mode1)})'
                stat.values = [
                    KeyValue(key='I2C Bus', value=str(self.i2c_bus)),
                    KeyValue(key='Address', value=hex(self.pca9685_addr)),
                    KeyValue(key='MODE1 Register', value=hex(mode1)),
                ]
                data['mode1'] = hex(mode1)
            except Exception as e:
                stat.level = DiagnosticStatus.ERROR
                stat.message = f'PCA9685 I2C Read Error: {e}'
                data['status'] = 'FAIL'
        else:
            stat.level = DiagnosticStatus.ERROR
            stat.message = f'I2C bus /dev/i2c-{self.i2c_bus} not available'
            data['status'] = 'FAIL'

        return stat, data

    def _check_tof(self, now: float) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: ESP32 ToF Sensors'
        stat.hardware_id = self.tof_port

        data = {'status': 'PASS'}
        tof_t = self._last_topics['tof_ranges']
        age = now - tof_t['stamp'] if tof_t['stamp'] > 0 else 999.0

        if tof_t['stamp'] > 0 and age < 4.0 and tof_t['data']:
            stat.level = DiagnosticStatus.OK
            stat.message = f'Active (Data: {tof_t["data"].get("ranges_mm", [])})'
            stat.values = [
                KeyValue(key='Left Range (mm)', value=str(tof_t['data'].get('left', 0))),
                KeyValue(key='Right Range (mm)', value=str(tof_t['data'].get('right', 0))),
            ]
            data.update({'active': True, 'data': tof_t['data']})
        elif os.path.exists(self.tof_port):
            stat.level = DiagnosticStatus.WARN
            stat.message = f'Port {self.tof_port} present but no JSON stream'
            data.update({'status': 'WARN', 'active': False})
        else:
            stat.level = DiagnosticStatus.WARN
            stat.message = f'ToF sensor port {self.tof_port} not connected (optional)'
            data.update({'status': 'WARN', 'active': False})

        return stat, data

    def _check_audio(self) -> (DiagnosticStatus, Dict[str, Any]):
        stat = DiagnosticStatus()
        stat.name = 'ecobot_hardware: Audio Subsystem'
        stat.hardware_id = 'Jetson Audio Sinks'

        data = {'status': 'PASS'}
        if sd:
            try:
                devs = sd.query_devices()
                ins = [d for d in devs if d['max_input_channels'] > 0]
                outs = [d for d in devs if d['max_output_channels'] > 0]
                stat.values = [
                    KeyValue(key='Playback Sinks', value=str(len(outs))),
                    KeyValue(key='Capture Sources', value=str(len(ins))),
                ]
                if len(ins) > 0 and len(outs) > 0:
                    stat.level = DiagnosticStatus.OK
                    stat.message = f'Audio OK ({len(outs)} Playback, {len(ins)} Mic inputs)'
                else:
                    stat.level = DiagnosticStatus.WARN
                    stat.message = f'Partial audio devices ({len(outs)} Out, {len(ins)} In)'
                    data['status'] = 'WARN'
                data.update({'inputs': len(ins), 'outputs': len(outs)})
            except Exception as e:
                stat.level = DiagnosticStatus.WARN
                stat.message = f'Audio query warning: {e}'
                data['status'] = 'WARN'
        else:
            stat.level = DiagnosticStatus.OK
            stat.message = 'Audio subsystem ALSA present'

        return stat, data

    # --------------------------------------------------------------------------
    # Service Callback: On-Demand Trigger Check
    # --------------------------------------------------------------------------
    def _handle_trigger_check(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().info('Received request to run full hardware check...')
        # Execute check cycle
        self._diagnostic_cycle()

        # Build quick summary response
        rs_fps = self._last_topics['realsense_color']['fps']
        arm_fps = self._last_topics['arm_camera']['fps']
        odom_fps = self._last_topics['odom']['fps']

        res_dict = {
            'timestamp': time.time(),
            'realsense_color_fps': rs_fps,
            'arm_camera_fps': arm_fps,
            'odom_fps': odom_fps,
            'status': 'HEALTHY' if (rs_fps > 0 and arm_fps > 0) else 'DEGRADED',
        }

        response.success = (res_dict['status'] == 'HEALTHY')
        response.message = json.dumps(res_dict)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = HardwareDiagnosticNode()
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
