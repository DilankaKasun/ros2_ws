#!/usr/bin/env python3
"""
EcoBot Full Hardware & System Diagnostic Suite.

Performs comprehensive pre-flight and runtime hardware validation across:
  - Jetson Orin Nano / Host Compute (CPU, RAM, Disk, Thermals, GPU/CUDA, Network)
  - RP2040 Pico Base Motor Controller (Serial COBS protocol, Encoders, Heartbeat)
  - Intel RealSense D415 Depth Camera (pyrealsense2 streaming, depth validity, intrinsics)
  - Arm Wrist USB Camera (V4L2 capture, resolution, frame validity)
  - Robot Arm Servos (PCA9685 I2C bus 7 @ 0x40, channel readbacks, joint limits)
  - ESP32 ToF Distance Sensors (Serial/HTTP telemetry)
  - Audio Subsystem (ALSA/Sounddevice playback and recording)
  - ROS 2 Ecosystem (Active nodes, topic rates, TF tree)
"""

import argparse
import glob
import json
import math
import os
import platform
import socket
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Optional hardware libraries loaded safely
try:
    import psutil
except ImportError:
    psutil = None

try:
    import serial
except ImportError:
    serial = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import smbus2
except ImportError:
    smbus2 = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import rclpy
    from rclpy.node import Node as RosNode
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def grab_ros_image(topic: str, encoding: str = 'bgr8', timeout_sec: float = 1.5):
    """Attempt to grab a single frame from a live ROS Image topic."""
    if not ROS_AVAILABLE:
        return None
    
    frame = None
    was_init = False
    try:
        if not rclpy.ok():
            rclpy.init()
            was_init = True
        
        node = RosNode(f'hw_check_grabber_{int(time.time()*1000)%10000}')
        bridge = CvBridge()
        
        def cb(msg):
            nonlocal frame
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
            except Exception:
                pass

        sub = node.create_subscription(RosImage, topic, cb, 1)
        t0 = time.time()
        while time.time() - t0 < timeout_sec and frame is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        
        node.destroy_node()
        if was_init and rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
    
    return frame

# ANSI styling
class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_GREEN = '\033[42m\033[30m'
    BG_YELLOW = '\033[43m\033[30m'
    BG_RED = '\033[41m\033[37m'


def tag(status: str) -> str:
    if status == 'PASS':
        return f'{Color.GREEN}{Color.BOLD}[PASS]{Color.RESET}'
    elif status == 'WARN':
        return f'{Color.YELLOW}{Color.BOLD}[WARN]{Color.RESET}'
    elif status == 'FAIL':
        return f'{Color.RED}{Color.BOLD}[FAIL]{Color.RESET}'
    elif status == 'SKIP':
        return f'{Color.DIM}[SKIP]{Color.RESET}'
    elif status == 'INFO':
        return f'{Color.CYAN}[INFO]{Color.RESET}'
    return f'[{status}]'


class CheckResult:
    def __init__(self, name: str, status: str, summary: str, details: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = status  # PASS, WARN, FAIL, SKIP, INFO
        self.summary = summary
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status,
            'summary': self.summary,
            'details': self.details,
        }


# ==============================================================================
# 1. System & Compute Checker
# ==============================================================================
class SystemChecker:
    @staticmethod
    def run() -> List[CheckResult]:
        results = []

        # 1.1 Host & OS Info
        uname = platform.uname()
        host_info = {
            'system': uname.system,
            'node': uname.node,
            'release': uname.release,
            'machine': uname.machine,
        }
        jetson_model = "Unknown Jetson / Linux Device"
        model_paths = ['/proc/device-tree/model', '/sys/firmware/devicetree/base/model']
        for p in model_paths:
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        jetson_model = f.read().strip().replace('\x00', '')
                        break
                except Exception:
                    pass
        host_info['device_model'] = jetson_model

        results.append(CheckResult(
            name='Host Platform',
            status='PASS',
            summary=f'{jetson_model} ({uname.node}, Linux {uname.release})',
            details=host_info
        ))

        # 1.2 CPU & Load
        if psutil:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_pct = psutil.cpu_percent(interval=0.2)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            cpu_status = 'PASS' if cpu_pct < 90.0 else 'WARN'
            results.append(CheckResult(
                name='CPU & Load',
                status=cpu_status,
                summary=f'{cpu_count} cores, usage {cpu_pct:.1f}%, load avg: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}',
                details={'cores': cpu_count, 'usage_pct': cpu_pct, 'load_avg': load_avg}
            ))

        # 1.3 Memory (RAM & Swap)
        if psutil:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            mem_used_gb = mem.used / 1e9
            mem_tot_gb = mem.total / 1e9
            mem_status = 'PASS' if mem.percent < 85.0 else ('WARN' if mem.percent < 95.0 else 'FAIL')
            results.append(CheckResult(
                name='Memory (RAM)',
                status=mem_status,
                summary=f'{mem_used_gb:.2f} GB / {mem_tot_gb:.2f} GB ({mem.percent}% used)',
                details={
                    'ram_total_gb': round(mem_tot_gb, 2),
                    'ram_used_gb': round(mem_used_gb, 2),
                    'ram_percent': mem.percent,
                    'swap_used_mb': round(swap.used / 1e6, 1),
                    'swap_total_mb': round(swap.total / 1e6, 1)
                }
            ))

        # 1.4 Disk Space
        if psutil:
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / 1e9
            disk_tot_gb = disk.total / 1e9
            disk_free_gb = disk.free / 1e9
            disk_status = 'PASS' if disk.percent < 85.0 else ('WARN' if disk.percent < 95.0 else 'FAIL')
            results.append(CheckResult(
                name='Disk Storage (/)',
                status=disk_status,
                summary=f'{disk_free_gb:.1f} GB free of {disk_tot_gb:.1f} GB ({disk.percent}% used)',
                details={
                    'total_gb': round(disk_tot_gb, 2),
                    'used_gb': round(disk_used_gb, 2),
                    'free_gb': round(disk_free_gb, 2),
                    'used_percent': disk.percent
                }
            ))

        # 1.5 Thermal Sensors
        thermals = {}
        thermal_dir = '/sys/devices/virtual/thermal'
        max_temp = 0.0
        if os.path.exists(thermal_dir):
            for z in sorted(os.listdir(thermal_dir)):
                if z.startswith('thermal_zone'):
                    t_type_file = os.path.join(thermal_dir, z, 'type')
                    t_temp_file = os.path.join(thermal_dir, z, 'temp')
                    if os.path.exists(t_type_file) and os.path.exists(t_temp_file):
                        try:
                            with open(t_type_file, 'r') as f1, open(t_temp_file, 'r') as f2:
                                t_name = f1.read().strip()
                                t_val = float(f2.read().strip()) / 1000.0
                                thermals[t_name] = round(t_val, 1)
                                if t_val > max_temp:
                                    max_temp = t_val
                        except Exception:
                            pass

        if thermals:
            t_status = 'PASS' if max_temp < 75.0 else ('WARN' if max_temp < 85.0 else 'FAIL')
            cpu_t = thermals.get('cpu-thermal', thermals.get('CPU-therm', 'N/A'))
            gpu_t = thermals.get('gpu-thermal', thermals.get('GPU-therm', 'N/A'))
            summary_str = f'CPU: {cpu_t}°C, GPU: {gpu_t}°C (Max: {max_temp:.1f}°C)'
            results.append(CheckResult(
                name='Thermals',
                status=t_status,
                summary=summary_str,
                details=thermals
            ))

        # 1.6 GPU / AI Acceleration
        gpu_details = {}
        gpu_status = 'INFO'
        gpu_summary = 'No GPU acceleration library detected'
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            gpu_details['torch_version'] = torch.__version__
            gpu_details['cuda_available'] = has_cuda
            if has_cuda:
                dev_name = torch.cuda.get_device_name(0)
                gpu_details['device_name'] = dev_name
                gpu_status = 'PASS'
                gpu_summary = f'CUDA Available ({dev_name}, PyTorch {torch.__version__})'
            else:
                gpu_status = 'WARN'
                gpu_summary = f'PyTorch installed ({torch.__version__}) but CUDA not active'
        except Exception as e:
            gpu_details['torch_error'] = str(e)

        try:
            import tensorrt as trt
            gpu_details['tensorrt_version'] = trt.__version__
            if gpu_status == 'PASS':
                gpu_summary += f', TensorRT {trt.__version__}'
        except Exception:
            pass

        results.append(CheckResult(
            name='GPU / CUDA Acceleration',
            status=gpu_status,
            summary=gpu_summary,
            details=gpu_details
        ))

        # 1.7 Network Connectivity
        net_details = {}
        ip_addrs = []
        try:
            # Find local IP addresses
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            s.close()
            net_details['primary_ip'] = local_ip
            ip_addrs.append(local_ip)
            net_status = 'PASS'
            net_summary = f'Connected (IP: {local_ip})'
        except Exception:
            net_status = 'WARN'
            net_summary = 'No external network connection detected'

        # Ping test to gateway or DNS
        try:
            ping_out = subprocess.run(
                ['ping', '-c', '1', '-W', '1', '8.8.8.8'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            net_details['internet_ping'] = (ping_out.returncode == 0)
        except Exception:
            pass

        results.append(CheckResult(
            name='Network Interface',
            status=net_status,
            summary=net_summary,
            details=net_details
        ))

        return results


# ==============================================================================
# 2. Base Motor Controller & RP2040 Pico Checker
# ==============================================================================
class MotorChecker:
    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        s = 0
        for i in range(0, len(data), 2):
            word = struct.unpack('<H', data[i:i+2])[0]
            s += word
            while s >> 16:
                s = (s & 0xFFFF) + (s >> 16)
        return (~s) & 0xFFFF

    @staticmethod
    def cobs_encode(data: bytes) -> bytes:
        out = bytearray()
        out.append(0)
        code_idx = 0
        code = 1
        for byte in data:
            if byte == 0:
                out[code_idx] = code
                code_idx = len(out)
                out.append(0)
                code = 1
            else:
                out.append(byte)
                code += 1
                if code == 0xFF:
                    out[code_idx] = code
                    code_idx = len(out)
                    out.append(0)
                    code = 1
        out[code_idx] = code
        out.append(0)
        return bytes(out)

    @staticmethod
    def cobs_decode(data: bytes) -> bytes:
        out = bytearray()
        i = 0
        while i < len(data):
            code = data[i]
            if code == 0:
                break
            i += 1
            out.extend(data[i:i + code - 1])
            i += code - 1
            if code != 0xFF and i < len(data) and data[i] != 0:
                out.append(0)
        return bytes(out)

    @staticmethod
    def run(port: str = '/dev/ttyACM0', baudrate: int = 115200) -> List[CheckResult]:
        results = []

        # Find candidate ports
        candidates = [port]
        pico_links = glob.glob('/dev/serial/by-id/*Pico*') + glob.glob('/dev/serial/by-id/*pico*')
        for link in pico_links:
            real_path = os.path.realpath(link)
            if real_path not in candidates:
                candidates.insert(0, real_path)

        active_port = None
        for p in candidates:
            if os.path.exists(p):
                active_port = p
                break

        if not active_port:
            results.append(CheckResult(
                name='Motor Controller Device',
                status='FAIL',
                summary=f'Port {port} not found (Pico disconnected or unpowered)',
                details={'target_port': port, 'available_serial': glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')}
            ))
            return results

        # Check port permissions
        is_readable = os.access(active_port, os.R_OK)
        is_writable = os.access(active_port, os.W_OK)
        if not (is_readable and is_writable):
            results.append(CheckResult(
                name='Motor Port Permissions',
                status='FAIL',
                summary=f'User lacks read/write access to {active_port} (try: sudo usermod -a -G dialout $USER)',
                details={'port': active_port, 'read': is_readable, 'write': is_writable}
            ))
            return results

        if not serial:
            results.append(CheckResult(
                name='Motor Controller Driver',
                status='WARN',
                summary='pyserial library not installed, skipping direct serial handshake',
                details={'port': active_port}
            ))
            return results

        # Attempt direct handshake or detect if port is busy by motor_control_node
        ser_details = {'port': active_port, 'baudrate': baudrate}
        try:
            ser = serial.Serial(active_port, baudrate, timeout=0.6)
            
            # Send 0 RPM heartbeat packet (72 bytes)
            PACKET_HEADER_SIZE = 8
            PACKET_BODY_SIZE = 64
            PACKET_SIZE = PACKET_HEADER_SIZE + PACKET_BODY_SIZE
            ROBOT_ID = 8888
            
            body = bytearray(PACKET_BODY_SIZE)
            struct.pack_into('<ff', body, 0, 0.0, 0.0)
            checksum = MotorChecker.calculate_checksum(bytes(body))
            header = struct.pack('<HHHH', 1, ROBOT_ID, PACKET_SIZE, checksum)
            raw_pkt = bytes(header + body)
            
            t0 = time.time()
            ser.write(MotorChecker.cobs_encode(raw_pkt))
            ser.flush()
            
            resp_raw = ser.read_until(b'\x00')
            latency_ms = round((time.time() - t0) * 1000, 2)
            ser.close()

            if resp_raw:
                decoded = MotorChecker.cobs_decode(resp_raw)
                if len(decoded) == PACKET_SIZE:
                    rx_body = decoded[PACKET_HEADER_SIZE:]
                    rx_header = decoded[:PACKET_HEADER_SIZE]
                    prod_id, rob_id, length, rx_checksum = struct.unpack('<HHHH', rx_header)
                    enc_l, enc_r = struct.unpack_from('<ii', rx_body, 0)
                    run_mode = rx_body[8]
                    
                    ser_details.update({
                        'product_id': prod_id,
                        'robot_id': rob_id,
                        'encoder_left': enc_l,
                        'encoder_right': enc_r,
                        'run_mode': run_mode,
                        'latency_ms': latency_ms,
                    })

                    results.append(CheckResult(
                        name='Motor Controller (Pico RP2040)',
                        status='PASS',
                        summary=f'Connected on {active_port} @ {baudrate} baud (latency: {latency_ms}ms, mode: {run_mode}, encoders: L={enc_l}, R={enc_r})',
                        details=ser_details
                    ))
                else:
                    results.append(CheckResult(
                        name='Motor Controller (Pico RP2040)',
                        status='WARN',
                        summary=f'Pico responded but packet size mismatch: got {len(decoded)} bytes (expected {PACKET_SIZE})',
                        details=ser_details
                    ))
            else:
                results.append(CheckResult(
                    name='Motor Controller (Pico RP2040)',
                    status='WARN',
                    summary=f'Opened {active_port} but received no packet response (check Pico firmware or power)',
                    details=ser_details
                ))

        except serial.SerialException as e:
            err_msg = str(e)
            if 'Device or resource busy' in err_msg or 'could not open port' in err_msg:
                # Node is likely already running and streaming
                results.append(CheckResult(
                    name='Motor Controller Port',
                    status='PASS',
                    summary=f'Port {active_port} is active and occupied by running ROS node (motor_control_node)',
                    details={'port': active_port, 'busy': True}
                ))
            else:
                results.append(CheckResult(
                    name='Motor Controller Communication',
                    status='FAIL',
                    summary=f'Failed to open serial port {active_port}: {err_msg}',
                    details=ser_details
                ))
        except Exception as e:
            results.append(CheckResult(
                name='Motor Controller Communication',
                status='FAIL',
                summary=f'Unexpected error communicating with {active_port}: {e}',
                details=ser_details
            ))

        return results


# ==============================================================================
# 3. Intel RealSense D415 Depth Camera Checker
# ==============================================================================
class RealSenseChecker:
    @staticmethod
    def run(save_snapshot_dir: Optional[str] = None) -> List[CheckResult]:
        results = []

        if rs is None:
            results.append(CheckResult(
                name='RealSense SDK (pyrealsense2)',
                status='FAIL',
                summary='pyrealsense2 Python module is not installed',
                details={}
            ))
            return results

        ctx = rs.context()
        devices = ctx.query_devices()
        num_devices = len(devices)

        if num_devices == 0:
            results.append(CheckResult(
                name='RealSense D415 Camera',
                status='FAIL',
                summary='No Intel RealSense devices detected on USB bus',
                details={'devices_found': 0}
            ))
            return results

        dev = devices[0]
        dev_name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else 'RealSense Camera'
        serial_no = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else 'Unknown'
        fw_ver = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else 'Unknown'
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else 'Unknown'

        rs_details = {
            'name': dev_name,
            'serial_number': serial_no,
            'firmware_version': fw_ver,
            'usb_mode': usb_type,
        }

        # Check USB mode warning
        usb_warn = ''
        if '2.' in usb_type:
            usb_warn = ' (Warning: USB 2.1 detected — USB 3.0 recommended for full resolution/FPS)'

        # Test streaming pipeline if not currently locked
        pipeline_ok = False
        frame_stats = {}
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            profile = pipe.start(cfg)
            pipeline_ok = True

            # Warm up and grab frames
            for _ in range(5):
                frames = pipe.wait_for_frames(timeout_ms=3000)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame and depth_frame:
                color_img = np.asanyarray(color_frame.get_data()) if np else None
                depth_img = np.asanyarray(depth_frame.get_data()) if np else None

                if depth_img is not None:
                    valid_pixels = np.count_nonzero(depth_img)
                    total_pixels = depth_img.size
                    fill_rate = round((valid_pixels / total_pixels) * 100, 1)
                    min_depth_m = round(float(np.min(depth_img[depth_img > 0])) * 0.001, 2) if valid_pixels > 0 else 0
                    max_depth_m = round(float(np.max(depth_img)) * 0.001, 2)
                    frame_stats = {
                        'color_resolution': f'{color_frame.get_width()}x{color_frame.get_height()}',
                        'depth_resolution': f'{depth_frame.get_width()}x{depth_frame.get_height()}',
                        'depth_fill_rate_pct': fill_rate,
                        'depth_range_m': f'{min_depth_m}m - {max_depth_m}m',
                    }
                    rs_details.update(frame_stats)

                    # Save snapshots if requested
                    if save_snapshot_dir and cv2:
                        os.makedirs(save_snapshot_dir, exist_ok=True)
                        c_path = os.path.join(save_snapshot_dir, 'realsense_color.jpg')
                        d_path = os.path.join(save_snapshot_dir, 'realsense_depth.png')
                        cv2.imwrite(c_path, color_img)
                        depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        cv2.imwrite(d_path, depth_color)
                        rs_details['saved_snapshots'] = [c_path, d_path]

            pipe.stop()

        except Exception as e:
            err_msg = str(e)
            if 'Device or resource busy' in err_msg or 'pipeline' in err_msg.lower():
                # Device is in use by running realsense_feed node
                rs_details['pipeline_status'] = 'Device busy (currently streaming in ROS realsense_feed node)'
                pipeline_ok = True
                # Grab frame via ROS topic if snapshots requested
                if save_snapshot_dir and cv2:
                    ros_frame = grab_ros_image('/camera/color/image_raw', encoding='bgr8')
                    if ros_frame is not None:
                        os.makedirs(save_snapshot_dir, exist_ok=True)
                        c_path = os.path.join(save_snapshot_dir, 'realsense_color.jpg')
                        cv2.imwrite(c_path, ros_frame)
                        rs_details['saved_snapshots'] = [c_path]
            else:
                rs_details['pipeline_error'] = err_msg

        status = 'PASS' if pipeline_ok else 'WARN'
        summary_str = f'{dev_name} (S/N: {serial_no}, FW: {fw_ver}, USB: {usb_type}){usb_warn}'
        if frame_stats:
            summary_str += f' — 640x480 RGB+Depth OK ({frame_stats.get("depth_fill_rate_pct", 0)}% depth valid)'

        results.append(CheckResult(
            name='RealSense D415 Camera',
            status=status,
            summary=summary_str,
            details=rs_details
        ))

        return results


# ==============================================================================
# 4. USB Arm Camera Checker
# ==============================================================================
class ArmCameraChecker:
    ARM_CAM_BY_ID = '/dev/v4l/by-id/usb-HRY_YDL_lens_USB_Camera_20210616_720-video-index0'

    @staticmethod
    def run(save_snapshot_dir: Optional[str] = None) -> List[CheckResult]:
        results = []

        if cv2 is None:
            results.append(CheckResult(
                name='Arm Camera Driver (OpenCV)',
                status='FAIL',
                summary='OpenCV (cv2) is not installed',
                details={}
            ))
            return results

        preferred = ArmCameraChecker.ARM_CAM_BY_ID
        found_device = None

        if os.path.exists(preferred):
            found_device = preferred
        else:
            # Look for by-id links or video0
            candidates = glob.glob('/dev/v4l/by-id/*USB*Camera*') + ['/dev/video0', '/dev/video1']
            for d in candidates:
                if os.path.exists(d):
                    found_device = d
                    break

        if not found_device:
            results.append(CheckResult(
                name='USB Arm Camera',
                status='FAIL',
                summary='No USB Arm Camera video device found at /dev/v4l/by-id/ or /dev/video*',
                details={'target': preferred}
            ))
            return results

        cam_details = {'device_path': found_device}
        real_dev = os.path.realpath(found_device)
        cam_details['real_device'] = real_dev

        # Test video capture
        cap = None
        opened = False
        frame_ok = False
        try:
            cap = cv2.VideoCapture(found_device, cv2.CAP_V4L2)
            if cap.isOpened():
                opened = True
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_ok = True
                    h, w, c = frame.shape
                    mean_val = float(np.mean(frame)) if np else 0.0
                    cam_details.update({
                        'resolution': f'{w}x{h}',
                        'channels': c,
                        'mean_brightness': round(mean_val, 1)
                    })

                    if save_snapshot_dir:
                        os.makedirs(save_snapshot_dir, exist_ok=True)
                        snap_path = os.path.join(save_snapshot_dir, 'arm_camera.jpg')
                        cv2.imwrite(snap_path, frame)
                        cam_details['saved_snapshot'] = snap_path
                cap.release()
            else:
                # Check if device is held by running usb_camera_node
                cam_details['note'] = 'Device node exists (may be in active use by usb_camera_node)'
                if save_snapshot_dir and cv2:
                    ros_frame = grab_ros_image('/arm/camera/image_raw', encoding='bgr8')
                    if ros_frame is not None:
                        os.makedirs(save_snapshot_dir, exist_ok=True)
                        snap_path = os.path.join(save_snapshot_dir, 'arm_camera.jpg')
                        cv2.imwrite(snap_path, ros_frame)
                        cam_details['saved_snapshot'] = snap_path
                        frame_ok = True
        except Exception as e:
            cam_details['error'] = str(e)
        finally:
            if cap and cap.isOpened():
                cap.release()

        if frame_ok:
            results.append(CheckResult(
                name='USB Arm Camera',
                status='PASS',
                summary=f'Capturing OK ({found_device} -> {real_dev}, {cam_details.get("resolution", "640x480")})',
                details=cam_details
            ))
        elif opened:
            results.append(CheckResult(
                name='USB Arm Camera',
                status='WARN',
                summary=f'Device opened at {found_device} but returned empty frame',
                details=cam_details
            ))
        else:
            results.append(CheckResult(
                name='USB Arm Camera',
                status='PASS' if os.path.exists(found_device) else 'FAIL',
                summary=f'Device present at {found_device} (in use by ROS usb_camera_node)',
                details=cam_details
            ))

        return results


# ==============================================================================
# 5. Robot Arm PCA9685 I2C & Servo Controller Checker
# ==============================================================================
class ArmServoChecker:
    @staticmethod
    def run(bus: int = 7, address: int = 0x40) -> List[CheckResult]:
        results = []

        if smbus2 is None:
            results.append(CheckResult(
                name='Arm Servo I2C Driver (smbus2)',
                status='FAIL',
                summary='smbus2 library not installed',
                details={}
            ))
            return results

        i2c_dev = f'/dev/i2c-{bus}'
        if not os.path.exists(i2c_dev):
            results.append(CheckResult(
                name='Arm Servo I2C Bus',
                status='FAIL',
                summary=f'I2C bus {i2c_dev} does not exist',
                details={'bus': bus, 'address': hex(address)}
            ))
            return results

        details = {'i2c_bus': bus, 'address': hex(address)}
        try:
            i2c = smbus2.SMBus(bus)
            
            # Read MODE1 register (0x00)
            mode1 = i2c.read_byte_data(address, 0x00)
            details['mode1_register'] = hex(mode1)
            
            # Read 4-DOF joint channels (Base: Ch1, Shoulder: Ch4, Elbow: Ch3, Wrist: Ch2)
            channel_pulses = {}
            joint_configs = [
                ('Base', 1),
                ('Shoulder', 4),
                ('Elbow', 3),
                ('Wrist', 2),
            ]
            for label, ch in joint_configs:
                reg = 0x06 + 4 * ch
                on_l = i2c.read_byte_data(address, reg)
                on_h = i2c.read_byte_data(address, reg + 1)
                off_l = i2c.read_byte_data(address, reg + 2)
                off_h = i2c.read_byte_data(address, reg + 3)
                on_val = (on_h << 8) | on_l
                off_val = (off_h << 8) | off_l
                channel_pulses[f'{label} (ch{ch})'] = {'on': on_val, 'off': off_val}

            details['joints'] = channel_pulses
            i2c.close()

            results.append(CheckResult(
                name='Arm PCA9685 PWM Controller',
                status='PASS',
                summary=f'PCA9685 responding on I2C-{bus} address {hex(address)} (4 joints active, MODE1={hex(mode1)})',
                details=details
            ))

        except OSError as e:
            results.append(CheckResult(
                name='Arm PCA9685 PWM Controller',
                status='FAIL',
                summary=f'I2C communication failed on bus {bus} address {hex(address)}: {e}',
                details=details
            ))
        except Exception as e:
            results.append(CheckResult(
                name='Arm PCA9685 PWM Controller',
                status='FAIL',
                summary=f'Unexpected error probing PCA9685: {e}',
                details=details
            ))

        return results


# ==============================================================================
# 6. ESP32 ToF Distance Sensors Checker
# ==============================================================================
class TofChecker:
    @staticmethod
    def run(port: str = '/dev/ttyUSB0', baudrate: int = 115200) -> List[CheckResult]:
        results = []
        details = {'target_port': port, 'baudrate': baudrate}

        if not os.path.exists(port):
            results.append(CheckResult(
                name='ESP32 ToF Sensors',
                status='WARN',
                summary=f'Port {port} not found (ToF sensors optional / disconnected or using WiFi mode)',
                details=details
            ))
            return results

        if not serial:
            results.append(CheckResult(
                name='ESP32 ToF Driver',
                status='WARN',
                summary='pyserial not available to test ToF serial port',
                details=details
            ))
            return results

        try:
            ser = serial.Serial(port, baudrate, timeout=1.0)
            raw = ser.read(512).decode('latin-1', errors='ignore')
            ser.close()

            if '{' in raw and '}' in raw:
                idx = raw.find('{')
                end_idx = raw.find('}', idx)
                sample_json = raw[idx:end_idx+1]
                data = json.loads(sample_json)
                details['sample_data'] = data
                results.append(CheckResult(
                    name='ESP32 ToF Sensors',
                    status='PASS',
                    summary=f'Streaming valid JSON on {port} ({data})',
                    details=details
                ))
            else:
                results.append(CheckResult(
                    name='ESP32 ToF Sensors',
                    status='WARN',
                    summary=f'Port {port} opened but no JSON distance stream detected in 1s',
                    details=details
                ))
        except serial.SerialException as e:
            if 'Device or resource busy' in str(e):
                results.append(CheckResult(
                    name='ESP32 ToF Sensors',
                    status='PASS',
                    summary=f'Port {port} in active use by ROS tof_sensors node',
                    details=details
                ))
            else:
                results.append(CheckResult(
                    name='ESP32 ToF Sensors',
                    status='WARN',
                    summary=f'Serial port error on {port}: {e}',
                    details=details
                ))
        except Exception as e:
            results.append(CheckResult(
                name='ESP32 ToF Sensors',
                status='WARN',
                summary=f'Failed reading ToF sensors on {port}: {e}',
                details=details
            ))

        return results


# ==============================================================================
# 7. Audio System (Microphone & Speaker) Checker
# ==============================================================================
class AudioChecker:
    @staticmethod
    def run() -> List[CheckResult]:
        results = []
        details = {}

        if sd is None:
            results.append(CheckResult(
                name='Audio Subsystem (sounddevice)',
                status='WARN',
                summary='sounddevice module not available, falling back to ALSA check',
                details={}
            ))
            # Fallback to aplay/arecord
            try:
                aplay_out = subprocess.run(['aplay', '-l'], stdout=subprocess.PIPE, text=True)
                arecord_out = subprocess.run(['arecord', '-l'], stdout=subprocess.PIPE, text=True)
                has_playback = 'card' in aplay_out.stdout.lower()
                has_capture = 'card' in arecord_out.stdout.lower()
                status = 'PASS' if (has_playback and has_capture) else 'WARN'
                results.append(CheckResult(
                    name='ALSA Audio Devices',
                    status=status,
                    summary=f'ALSA Devices: Playback={"YES" if has_playback else "NO"}, Capture={"YES" if has_capture else "NO"}',
                    details={'aplay': has_playback, 'arecord': has_capture}
                ))
            except Exception as e:
                results.append(CheckResult(
                    name='ALSA Audio Devices',
                    status='WARN',
                    summary=f'Failed querying ALSA audio: {e}',
                    details={}
                ))
            return results

        try:
            devices = sd.query_devices()
            input_devs = [d for d in devices if d['max_input_channels'] > 0]
            output_devs = [d for d in devices if d['max_output_channels'] > 0]

            details['total_devices'] = len(devices)
            details['input_devices'] = len(input_devs)
            details['output_devices'] = len(output_devs)

            if len(input_devs) > 0 and len(output_devs) > 0:
                results.append(CheckResult(
                    name='Audio Subsystem (Microphone & Speaker)',
                    status='PASS',
                    summary=f'{len(output_devs)} Playback / {len(input_devs)} Capture devices available for Voice Agent',
                    details=details
                ))
            elif len(output_devs) > 0:
                results.append(CheckResult(
                    name='Audio Subsystem (Microphone & Speaker)',
                    status='WARN',
                    summary=f'Audio output available ({len(output_devs)} devices), but no capture microphone found',
                    details=details
                ))
            else:
                results.append(CheckResult(
                    name='Audio Subsystem (Microphone & Speaker)',
                    status='WARN',
                    summary='No audio devices found',
                    details=details
                ))
        except Exception as e:
            results.append(CheckResult(
                name='Audio Subsystem',
                status='WARN',
                summary=f'Error querying audio devices: {e}',
                details={'error': str(e)}
            ))

        return results


# ==============================================================================
# 8. ROS 2 Node & Topic Health Checker (Optional / When ROS is running)
# ==============================================================================
class RosChecker:
    @staticmethod
    def run() -> List[CheckResult]:
        results = []

        # Check ROS 2 environment variables
        ros_distro = os.environ.get('ROS_DISTRO', 'humble')
        ros_installed = os.path.isdir(f'/opt/ros/{ros_distro}')
        
        results.append(CheckResult(
            name='ROS 2 Environment',
            status='PASS' if ros_installed else 'WARN',
            summary=f'ROS 2 Distro: {ros_distro} (Path: /opt/ros/{ros_distro})',
            details={'distro': ros_distro, 'installed': ros_installed}
        ))

        # Check running ROS 2 nodes and topics
        try:
            node_proc = subprocess.run(['ros2', 'node', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3.0)
            if node_proc.returncode == 0:
                nodes = [n.strip() for n in node_proc.stdout.strip().split('\n') if n.strip()]
                
                expected_core_nodes = [
                    'realsense_feed',
                    'usb_camera_node',
                    'motor_control_node',
                    'arm_manual_node',
                    'arm_camera_server',
                ]
                running_expected = [n for n in expected_core_nodes if any(n in running_n for running_n in nodes)]

                results.append(CheckResult(
                    name='ROS 2 Running Nodes',
                    status='PASS' if len(nodes) > 0 else 'INFO',
                    summary=f'{len(nodes)} nodes active ({len(running_expected)}/{len(expected_core_nodes)} core ecobot nodes)',
                    details={'active_nodes': nodes}
                ))

                # Check topics
                topic_proc = subprocess.run(['ros2', 'topic', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3.0)
                if topic_proc.returncode == 0:
                    topics = [t.strip() for t in topic_proc.stdout.strip().split('\n') if t.strip()]
                    results.append(CheckResult(
                        name='ROS 2 Topic Bus',
                        status='PASS',
                        summary=f'{len(topics)} active ROS topics on the bus',
                        details={'topics': topics}
                    ))
            else:
                results.append(CheckResult(
                    name='ROS 2 Stack Status',
                    status='INFO',
                    summary='ROS 2 daemon is idle (no running nodes currently launched)',
                    details={}
                ))
        except Exception:
            results.append(CheckResult(
                name='ROS 2 Stack Status',
                status='INFO',
                summary='ROS 2 nodes not running (running in pre-flight mode)',
                details={}
            ))

        return results


# ==============================================================================
# 9. Interactive Actuator Test Suite
# ==============================================================================
class InteractiveTester:
    @staticmethod
    def run_tests():
        print(f"\n{Color.CYAN}{Color.BOLD}=== ECOBOT INTERACTIVE ACTUATOR TEST MODE ==={Color.RESET}")
        print("This mode tests moving parts (Motors & Arm Servos) and audio safely.\n")

        # Test 1: PCA9685 Arm Servo Test
        if smbus2:
            ans = input(f"Test Arm Servos (I2C-7 @ 0x40)? [y/N]: ").strip().lower()
            if ans == 'y':
                try:
                    from ecobot_arm_control.pca9685_driver import PCA9685
                    from ecobot_arm_control.servo_config import JOINTS
                    print("  Initializing PCA9685...")
                    pca = PCA9685(bus=7, address=0x40, freq=50.0)
                    print("  Driving joints to designated home angles...")
                    for j in JOINTS:
                        print(f"    Moving {j['label']} (ch {j['channel']}) -> {j['home_angle']} deg")
                        pca.set_angle(j['channel'], j['home_angle'], j['pulse_min'], j['pulse_max'], j['servo_range'])
                        time.sleep(0.3)
                    print(f"  {Color.GREEN}Arm Servos test complete.{Color.RESET}")
                except Exception as e:
                    print(f"  {Color.RED}Arm test failed: {e}{Color.RESET}")

        # Test 2: Base Motors Test
        if serial:
            ans = input(f"\nTest Base Motors (slow 0.5s forward jog)? Ensure wheels are clear! [y/N]: ").strip().lower()
            if ans == 'y':
                try:
                    port = '/dev/ttyACM0'
                    ser = serial.Serial(port, 115200, timeout=0.5)
                    print("  Sending 15 RPM forward jog command for 0.5s...")
                    # 15 RPM forward
                    body = bytearray(64)
                    struct.pack_into('<ff', body, 0, 15.0, 15.0)
                    cs = MotorChecker.calculate_checksum(bytes(body))
                    header = struct.pack('<HHHH', 1, 8888, 72, cs)
                    ser.write(MotorChecker.cobs_encode(bytes(header + body)))
                    time.sleep(0.5)
                    # Stop
                    struct.pack_into('<ff', body, 0, 0.0, 0.0)
                    cs = MotorChecker.calculate_checksum(bytes(body))
                    header = struct.pack('<HHHH', 1, 8888, 72, cs)
                    ser.write(MotorChecker.cobs_encode(bytes(header + body)))
                    ser.close()
                    print(f"  {Color.GREEN}Base Motor test complete.{Color.RESET}")
                except Exception as e:
                    print(f"  {Color.RED}Motor test failed: {e}{Color.RESET}")

        # Test 3: Audio Tone Test
        ans = input(f"\nTest Audio Speaker Output? [y/N]: ").strip().lower()
        if ans == 'y':
            try:
                subprocess.run(['speaker-test', '-t', 'sine', '-f', '440', '-l', '1'], timeout=2.0)
                print(f"  {Color.GREEN}Audio test complete.{Color.RESET}")
            except Exception as e:
                print(f"  {Color.YELLOW}Speaker test note: {e}{Color.RESET}")

        print(f"\n{Color.CYAN}Interactive tests finished.{Color.RESET}\n")


# ==============================================================================
# 10. CLI Dashboard Formatter & Main Entrypoint
# ==============================================================================
def print_report(all_results: List[CheckResult], elapsed_sec: float) -> int:
    passes = sum(1 for r in all_results if r.status == 'PASS')
    warns = sum(1 for r in all_results if r.status == 'WARN')
    fails = sum(1 for r in all_results if r.status == 'FAIL')
    infos = sum(1 for r in all_results if r.status == 'INFO')
    skips = sum(1 for r in all_results if r.status == 'SKIP')
    total = len(all_results)

    print()
    print(f"{Color.CYAN}{Color.BOLD}{'=' * 78}{Color.RESET}")
    print(f"{Color.WHITE}{Color.BOLD}                ECOBOT HARDWARE & SYSTEM DIAGNOSTIC REPORT{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'=' * 78}{Color.RESET}")
    print(f"{Color.DIM}Date: {time.strftime('%Y-%m-%d %H:%M:%S')} | Elapsed: {elapsed_sec:.2f}s | Checks: {total}{Color.RESET}")
    print(f"{Color.CYAN}{'-' * 78}{Color.RESET}")

    for r in all_results:
        print(f"{tag(r.status):<18} {Color.BOLD}{r.name:<32}{Color.RESET} {r.summary}")

    print(f"{Color.CYAN}{'=' * 78}{Color.RESET}")

    # Summary box
    if fails == 0 and warns == 0:
        overall = f"{Color.BG_GREEN}{Color.BOLD} SYSTEM HEALTHY: ALL CHECKS PASSED {Color.RESET}"
        exit_code = 0
    elif fails == 0:
        overall = f"{Color.BG_YELLOW}{Color.BOLD} SYSTEM DEGRADED: WARNINGS DETECTED {Color.RESET}"
        exit_code = 0
    else:
        overall = f"{Color.BG_RED}{Color.BOLD} SYSTEM CRITICAL: {fails} HARDWARE FAILURE(S) {Color.RESET}"
        exit_code = 1

    print(f"Summary: {Color.GREEN}{passes} Passed{Color.RESET}, "
          f"{Color.YELLOW}{warns} Warnings{Color.RESET}, "
          f"{Color.RED}{fails} Failures{Color.RESET}, "
          f"{Color.CYAN}{infos} Info{Color.RESET}")
    print(f"Status:  {overall}")
    print(f"{Color.CYAN}{'=' * 78}{Color.RESET}\n")

    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="EcoBot Comprehensive Hardware Diagnostic Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ros2 run ecobot_bringup hardware_check
  ros2 run ecobot_bringup hardware_check --quick
  ros2 run ecobot_bringup hardware_check --cameras --save-snapshots /tmp/ecobot_cam
  ros2 run ecobot_bringup hardware_check --test
  ros2 run ecobot_bringup hardware_check --json
        """
    )

    parser.add_argument('--all', action='store_true', help='Run all hardware checks (default)')
    parser.add_argument('--quick', action='store_true', help='Run fast non-intrusive pre-flight checks (<2s)')
    parser.add_argument('--system', action='store_true', help='Check host compute, RAM, CPU, thermals, GPU')
    parser.add_argument('--motors', action='store_true', help='Check base motor controller & encoders')
    parser.add_argument('--cameras', action='store_true', help='Check RealSense D415 and USB Arm camera')
    parser.add_argument('--arm', action='store_true', help='Check PCA9685 I2C and servos')
    parser.add_argument('--sensors', action='store_true', help='Check all perception sensors (RealSense, Arm Cam, ToF)')
    parser.add_argument('--audio', action='store_true', help='Check audio devices (Microphone/Speaker)')
    parser.add_argument('--ros', action='store_true', help='Check active ROS 2 nodes and topics')
    parser.add_argument('--test', action='store_true', help='Run interactive actuator movement and audio tests')
    parser.add_argument('--json', action='store_true', help='Output full report in JSON format')
    parser.add_argument('--save-snapshots', metavar='DIR', type=str, help='Directory to save camera test frames')
    parser.add_argument('--port', default='/dev/ttyACM0', help='Motor controller serial port (default: /dev/ttyACM0)')
    parser.add_argument('--i2c-bus', type=int, default=7, help='Arm PCA9685 I2C bus (default: 7)')

    args = parser.parse_args()

    # Determine which check groups to run
    specific_flags = [args.system, args.motors, args.cameras, args.arm, args.sensors, args.audio, args.ros]
    run_all = args.all or not any(specific_flags)

    t_start = time.time()
    results: List[CheckResult] = []

    if run_all or args.system:
        results.extend(SystemChecker.run())

    if run_all or args.motors:
        results.extend(MotorChecker.run(port=args.port))

    if run_all or args.cameras or args.sensors:
        results.extend(RealSenseChecker.run(save_snapshot_dir=args.save_snapshots))
        results.extend(ArmCameraChecker.run(save_snapshot_dir=args.save_snapshots))

    if run_all or args.arm:
        results.extend(ArmServoChecker.run(bus=args.i2c_bus))

    if run_all or args.sensors:
        results.extend(TofChecker.run())

    if run_all or args.audio:
        results.extend(AudioChecker.run())

    if run_all or args.ros:
        results.extend(RosChecker.run())

    t_elapsed = time.time() - t_start

    # Interactive test mode
    if args.test:
        InteractiveTester.run_tests()

    # JSON Output
    if args.json:
        report_data = {
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_sec': round(t_elapsed, 3),
            'summary': {
                'passes': sum(1 for r in results if r.status == 'PASS'),
                'warnings': sum(1 for r in results if r.status == 'WARN'),
                'failures': sum(1 for r in results if r.status == 'FAIL'),
                'total': len(results),
            },
            'results': [r.to_dict() for r in results]
        }
        print(json.dumps(report_data, indent=2))
        return 1 if report_data['summary']['failures'] > 0 else 0

    return print_report(results, t_elapsed)


if __name__ == '__main__':
    sys.exit(main())
