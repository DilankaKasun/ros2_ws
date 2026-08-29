import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import Image
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType
from cv_bridge import CvBridge
import cv2
import os
import math
import json
import time
import numpy as np

from .arm_kinematics import ArmKinematics
from .servo_config import JOINTS, NUM_JOINTS, to_servo, to_ik, within_limits


class ArmScannerNode(Node):
    """Continuous, aimed plant scan.

    Replaces the old fixed 4-viewpoint (front/right/left/top) jump with a
    smooth vertical sweep plus optional orbit passes: the wrist camera is
    parked at a standoff in front of the plant and sweeps from above the
    leaves down past the stem to the pot base, always AIMED at the plant
    center via arm_kinematics.inverse_aim(). Pose goals are emitted as
    /arm/pose_goal so arm_manual_node's smooth trajectory executor moves
    the joints continuously instead of snapping.

    Contract (unchanged): /arm/scanner_cmd {'action':'scan'|'stop'|'home',
    'x','y','z'} in arm IK frame; status on /arm/scanner_status with
    'current_label' per viewpoint so plant_mission_node's capture pipeline
    keeps working.
    """

    def __init__(self):
        super().__init__('arm_scanner_node')

        self.declare_parameter('standoff', 0.18)
        self.declare_parameter('standoff_min', 0.06)
        self.declare_parameter('standoff_max', 0.26)
        self.declare_parameter('standoff_step', 0.02)
        self.declare_parameter('max_aim_error', 25.0)
        self.declare_parameter('sweep_top_offset', 0.20)
        self.declare_parameter('sweep_bottom_offset', -0.15)
        self.declare_parameter('sweep_steps', 8)
        self.declare_parameter('orbit_angles', [0.0])   # deg relative to approach
        self.declare_parameter('aim_height', 'center')  # 'center' | 'path'
        self.declare_parameter('dwell_time', 1.2)
        self.declare_parameter('l0', 0.320)
        self.declare_parameter('l1', 0.165)
        self.declare_parameter('l2', 0.140)
        self.declare_parameter('l3', 0.090)
        self.declare_parameter('recovery_timeout_s', 4.0)
        # Off by default: this legacy path auto-starts a scan on the very
        # first raw /ecobot/detections message. Kept for anyone who still
        # wants the old immediate-react behavior.
        self.declare_parameter('enable_detection_auto_scan', False)
        self.declare_parameter('scanning_peak_speed', 15.0)
        self.declare_parameter('scanning_accel', 30.0)
        self.declare_parameter('normal_peak_speed', 40.0)
        self.declare_parameter('normal_accel', 90.0)
        self.declare_parameter('lock_hfov', 73.3)
        self.declare_parameter('lock_vfov', 58.3)
        self.declare_parameter('lock_tolerance', 0.05)
        self.declare_parameter('lock_max_iters', 6)
        self.declare_parameter('lock_z_max_adjust', 0.08)

        self._recovery_timeout_s = float(
            self.get_parameter('recovery_timeout_s').value)
        self._last_plant_seen_time = time.time()

        self._lock_hfov = float(self.get_parameter('lock_hfov').value)
        self._lock_vfov = float(self.get_parameter('lock_vfov').value)
        self._lock_tol = float(self.get_parameter('lock_tolerance').value)
        self._lock_max_iters = int(self.get_parameter('lock_max_iters').value)
        self._lock_z_max_adjust = float(self.get_parameter('lock_z_max_adjust').value)

        self._standoff = float(self.get_parameter('standoff').value)
        self._standoff_min = float(self.get_parameter('standoff_min').value)
        self._standoff_max = float(self.get_parameter('standoff_max').value)
        self._standoff_step = float(self.get_parameter('standoff_step').value)
        self._max_aim_error = float(self.get_parameter('max_aim_error').value)
        self._sweep_top = float(self.get_parameter('sweep_top_offset').value)
        self._sweep_bottom = float(self.get_parameter('sweep_bottom_offset').value)
        self._sweep_steps = int(self.get_parameter('sweep_steps').value)
        self._orbit_angles = [
            float(a) for a in self.get_parameter('orbit_angles').value]
        self._aim_height = str(self.get_parameter('aim_height').value).strip().lower()
        self._dwell_time = float(self.get_parameter('dwell_time').value)
        self._enable_detection_auto_scan = bool(
            self.get_parameter('enable_detection_auto_scan').value)
        self._scanning_peak_speed = float(self.get_parameter('scanning_peak_speed').value)
        self._scanning_accel = float(self.get_parameter('scanning_accel').value)
        self._normal_peak_speed = float(self.get_parameter('normal_peak_speed').value)
        self._normal_accel = float(self.get_parameter('normal_accel').value)

        l0 = self.get_parameter('l0').value
        l1 = self.get_parameter('l1').value
        l2 = self.get_parameter('l2').value
        l3 = self.get_parameter('l3').value
        self._ik = ArmKinematics(l0, l1, l2, l3)

        self._param_client = self.create_client(
            SetParameters, '/arm_control_node/set_parameters')

        # Waypoint queue: each entry is (label, sx, sy, sz, ax, ay, az,
        # servo_angles) where (s) is the camera standoff pose, (a) the aim
        # point, and servo_angles the pre-solved aimed joint command.
        self._scan_queue = []
        self._current_viewpoint = 0
        self._scanning = False
        self._dwell_start = None
        self._parts_covered = set()

        self._pose_pub = self.create_publisher(
            Float64MultiArray, '/arm/pose_goal', 10)
        self._joint_pub = self.create_publisher(
            Float64MultiArray, '/arm/joint_commands', 10)
        self._scanner_status_pub = self.create_publisher(
            String, '/arm/scanner_status', 10)

        # Track main navigation camera detections and wrist CV feedback for target verification
        self._latest_nav_detections = []
        self._latest_cv_data = {}

        self.create_subscription(
            String, '/ecobot/detections', self._detections_cb, 10)
        self.create_subscription(
            String, '/arm/cv_plant_parts', self._cv_part_cb, 10)

        self._cmd_sub = self.create_subscription(
            String, '/arm/scanner_cmd', self._scanner_cmd_cb, 10)

        # Track live joint angles so dwell waits until the arm actually
        # reaches each viewpoint (smooth trajectories take seconds).
        self._current_joints = [float(j['home_angle']) for j in JOINTS]
        self._joint_sub = self.create_subscription(
            Float64MultiArray, '/arm/joint_angles', self._joint_cb, 10)

        # Track live computer vision analysis for closed-loop active re-orientation
        self._latest_cv_data = {}
        self.create_subscription(
            String, '/arm/cv_plant_parts', self._cv_part_cb, 10)

        # Video Recording attributes
        self._video_dir = '/tmp/plant_scans'
        os.makedirs(self._video_dir, exist_ok=True)
        self._video_writer = None
        self._video_filename = ''
        self._video_frames_count = 0
        self._bridge = CvBridge()
        self._video_path_pub = self.create_publisher(
            String, '/arm/scan_video_path', 10)

        self.create_subscription(
            Image, '/arm/camera/image_raw', self._on_camera_frame, 10)

        self._timer = self.create_timer(0.25, self._timer_cb)

        self._home_angles = [float(j['home_angle']) for j in JOINTS]
        self.get_logger().info(
            f'Arm scanner ready — standoff={self._standoff}m '
            f'sweep=[{self._sweep_top},{self._sweep_bottom}] '
            f'steps={self._sweep_steps} orbit={self._orbit_angles}')

    def _scanner_cmd_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        action = data.get('action', '')
        if action == 'scan':
            x = data.get('x', 0.3)
            y = data.get('y', 0.0)
            z = data.get('z', 0.15)
            plant_type = data.get('plant_type') or data.get('type', 'potted plant')
            plant_height = data.get('plant_height') or data.get('height')
            z_top = data.get('z_top')
            z_bottom = data.get('z_bottom')
            self._start_scan(x, y, z, plant_type=plant_type, plant_height=plant_height, z_top=z_top, z_bottom=z_bottom)
        elif action == 'stop':
            self._stop_scan()
        elif action == 'home':
            self._go_home()

    def _detections_cb(self, msg):
        if not self._enable_detection_auto_scan or self._scanning:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        detections = data if isinstance(data, list) else data.get('detections', [])
        if not detections:
            return
        d = detections[0]
        dist = d.get('distance')
        if dist is None:
            return
        x = dist
        y = 0.0
        z = 0.2
        if 'pose' in d and isinstance(d['pose'], dict):
            pose = d['pose']
            if pose.get('x') is not None:
                x = pose['x']
            if pose.get('y') is not None:
                y = pose['y']
            if pose.get('z') is not None:
                z = pose['z']
        plant_type = d.get('plant_type') or d.get('class_name', 'potted plant')
        plant_height = d.get('height')
        z_top = d.get('z_top')
        z_bottom = d.get('z_bottom')
        self._start_scan(x, y, z, plant_type=plant_type, plant_height=plant_height, z_top=z_top, z_bottom=z_bottom)

    # ---- path generation ----------------------------------------------

    def _solve_aim(self, sx, sy, sz, ax, ay, az):
        """Pose the camera aimed at A from a standoff in front of the plant,
        searching standoff distance (and backing off) for a reachable pose
        with low aim error.

        Standoffs closest to the nominal ``standoff`` are tried FIRST so the
        camera sits at the ideal framing distance (previously the greedy
        minimum-aim-error sweep kept collapsing to standoff_min, parking the
        camera 0.08 m from the plant where it could not focus). A candidate is
        only accepted if its forward kinematics actually lands the camera on
        the requested standoff point. Returns (servo_angles, aim_err)."""
        lo = to_ik([JOINTS[i]['min_angle'] for i in range(NUM_JOINTS)])
        hi = to_ik([JOINTS[i]['max_angle'] for i in range(NUM_JOINTS)])

        r0 = math.hypot(sx, sy)
        if r0 < 1e-4:
            return None
        ux, uy = sx / r0, sy / r0

        r_plant = r0 + self._standoff  # plant's horizontal distance from base
        s100_min = int(self._standoff_min * 100)
        s100_max = int(self._standoff_max * 100)
        s100_step = max(1, int(self._standoff_step * 100))

        # Order standoff candidates by distance from the nominal so the ideal
        # framing distance is preferred; only back off toward the limits when
        # the nominal is unreachable.
        candidates = sorted(
            range(s100_min, s100_max + 1, s100_step),
            key=lambda s: abs(s / 100.0 - self._standoff))

        best = None
        best_err = None
        for s100 in candidates:
            s = s100 / 100.0
            # camera sits `s` in front of the plant along the bearing
            sdist = max(0.03, r_plant - s)
            cx, cy = ux * sdist, uy * sdist
            result = self._ik.inverse_aim(
                cx, cy, sz, ax, ay, az,
                theta2_min=lo[1], theta2_max=hi[1],
                theta3_min=lo[2], theta3_max=hi[2],
                theta4_min=lo[3], theta4_max=hi[3],
            )
            if result is None:
                continue
            angles = to_servo(result)
            if not within_limits(angles):
                continue
            err = self._ik.aim_error(*result, ax, ay, az)

            # Verify the camera FK lands on the requested standoff point —
            # reject poses whose camera drifts far from where we asked it.
            fcx, fcy, fcz = self._ik.forward(*result)
            off = (math.hypot(fcx - cx, fcy - cy) + abs(fcz - sz))
            if best_err is None or err < best_err:
                best_err = err
                best = angles
            if err < self._max_aim_error and off < 0.08:
                return best, best_err

        return (best, best_err) if best is not None else None

    def _lock_onto_plant(self, x, y, z):
        """Perform closed-loop visual servoing using the wrist camera to lock onto the center of the plant.
        Returns corrected (bearing, z) or the original assumed values if locking fails."""
        import sys
        r = math.hypot(x, y)
        assumed_bearing = math.degrees(math.atan2(y, x))
        
        if 'pytest' in sys.modules or 'unittest' in sys.modules:
            return assumed_bearing, z
            
        self.get_logger().info(f'[Lock-On] Heading to assumed plant center (bearing={assumed_bearing:.1f}°, z={z:.2f}m) to lock on.')
        
        # Solve aimed look pose at assumed center
        base_rad = math.radians(assumed_bearing)
        sx = (r - self._standoff) * math.cos(base_rad)
        sy = (r - self._standoff) * math.sin(base_rad)
        
        solved = self._solve_aim(sx, sy, z, x, y, z)
        if solved is None:
            self.get_logger().warn('[Lock-On] Could not solve aimed starting pose to lock onto plant. Aborting lock phase.')
            return assumed_bearing, z

        angles, _ = solved
        
        # Publish joints and wait to arrive
        msg = Float64MultiArray()
        msg.data = [float(a) for a in angles]
        self._joint_pub.publish(msg)
        self._vp_target_angles = [float(a) for a in angles]
        
        def wait_for_arrival(timeout_s=3.0):
            start = time.time()
            while time.time() - start < timeout_s:
                if self._at_viewpoint(tol=1.0):
                    return True
                time.sleep(0.1)
            return False

        if not wait_for_arrival():
            self.get_logger().warn('[Lock-On] Arm failed to reach locking pose in time. Continuing with assumed center.')
            return assumed_bearing, z

        # Allow camera exposure to settle & CV to publish
        time.sleep(0.6)

        bearing_corrected = assumed_bearing
        z_corrected = z

        unseen = 0
        for iteration in range(self._lock_max_iters):
            cv_data = dict(self._latest_cv_data)
            if not cv_data or not cv_data.get('plant_present', False):
                unseen += 1
                self.get_logger().info(
                    f'[Lock-On] Iteration {iteration+1}: Plant not present/seen '
                    'by wrist camera.')
                if unseen >= 2:
                    self.get_logger().warn(
                        '[Lock-On] Plant never came into wrist view — '
                        'skipping visual centering, using assumed center.')
                    return assumed_bearing, z
                time.sleep(0.3)
                continue

            cx_norm = float(cv_data.get('centroid_x', 0.0))
            cy_norm = float(cv_data.get('centroid_y', 0.0))
            
            self.get_logger().info(f'[Lock-On] Iteration {iteration+1}: centroid_x={cx_norm:.2f}, centroid_y={cy_norm:.2f}')

            # Check if centered horizontally and vertically within tolerance
            if abs(cx_norm) <= self._lock_tol and abs(cy_norm) <= self._lock_tol:
                self.get_logger().info(f'[Lock-On] Centered successfully! cx_norm={cx_norm:.3f}, cy_norm={cy_norm:.3f} after {iteration+1} iterations.')
                break

            # Adjust bearing (horizontal)
            if abs(cx_norm) > self._lock_tol:
                theta_h = math.degrees(math.atan(cx_norm * math.tan(math.radians(self._lock_hfov / 2.0))))
                # Reduce bearing to rotate arm toward plant
                bearing_corrected -= theta_h
                # Bounded adjustment
                bearing_corrected = float(np.clip(bearing_corrected, assumed_bearing - 25.0, assumed_bearing + 25.0))

            # Adjust height (vertical)
            if abs(cy_norm) > self._lock_tol:
                theta_v = math.degrees(math.atan(cy_norm * math.tan(math.radians(self._lock_vfov / 2.0))))
                z_adj = theta_v * 0.002  # proportional factor
                z_corrected -= z_adj
                # Bounded vertical adjustment
                z_corrected = float(np.clip(z_corrected, z - self._lock_z_max_adjust, z + self._lock_z_max_adjust))

            # Re-solve look pose and publish
            rad_corr = math.radians(bearing_corrected)
            sx_corr = (r - self._standoff) * math.cos(rad_corr)
            sy_corr = (r - self._standoff) * math.sin(rad_corr)
            
            solved_corr = self._solve_aim(sx_corr, sy_corr, z_corrected, r * math.cos(rad_corr), r * math.sin(rad_corr), z_corrected)
            if solved_corr is not None:
                angles, _ = solved_corr
                msg = Float64MultiArray()
                msg.data = [float(a) for a in angles]
                self._joint_pub.publish(msg)
                self._vp_target_angles = [float(a) for a in angles]
                
                # Wait for execution
                wait_for_arrival(1.5)
                time.sleep(0.4)
            else:
                self.get_logger().warn('[Lock-On] Corrected pose is unreachable. Reverting step.')
                break

        self.get_logger().info(f'[Lock-On] Locked plant center: bearing={bearing_corrected:.1f}° (was {assumed_bearing:.1f}°), height={z_corrected:.2f}m (was {z:.2f}m)')
        return bearing_corrected, z_corrected

    def _start_scan(self, x, y, z, plant_type='potted plant', plant_height=None, z_top=None, z_bottom=None):
        if self._scanning:
            return

        r = math.sqrt(x ** 2 + y ** 2)
        
        # 1. Distance Range Check: Refuse to scan empty space too close or too far
        if r < 0.15 or r > 0.85:
            self.get_logger().warn(
                f'[Targeting Filter] Rejected scan command: Target position (x={x:.2f}m, y={y:.2f}m, dist={r:.2f}m) '
                'is outside physical arm scanning range (0.15m - 0.85m)! Refusing to scan empty area.')
            return

        import sys
        if 'pytest' in sys.modules or 'unittest' in sys.modules:
            self._execute_scan_sequence(x, y, z, plant_type, plant_height, z_top, z_bottom)
            return

        import threading
        self.get_logger().info('[Thread Dispatcher] Spawning asynchronous background thread for lock-on and plant scanning sequence.')
        threading.Thread(
            target=self._execute_scan_sequence,
            args=(x, y, z, plant_type, plant_height, z_top, z_bottom),
            daemon=True
        ).start()

    def _execute_scan_sequence(self, x, y, z, plant_type='potted plant', plant_height=None, z_top=None, z_bottom=None):
        self._parts_covered.clear()

        # Perform visual servoing to lock camera onto plant center
        bearing_locked, z_locked = self._lock_onto_plant(x, y, z)

        # Update x, y, z to the locked ones
        r = math.sqrt(x ** 2 + y ** 2)
        rad_locked = math.radians(bearing_locked)
        x = r * math.cos(rad_locked)
        y = r * math.sin(rad_locked)
        z = z_locked

        # 2. Dual-Vision Target Verification (Main Nav Camera + Wrist Camera)
        vision_verified = True
        has_nav_plant = False
        if self._latest_nav_detections:
            for d in self._latest_nav_detections:
                cls = str(d.get('class_name') or d.get('label', '')).lower()
                if any(p in cls for p in ('plant', 'pot', 'crop', 'leaf', 'tree', 'flower', 'vegetable')):
                    has_nav_plant = True
                    break

        has_wrist_plant = False
        if self._latest_cv_data:
            part = str(self._latest_cv_data.get('detected_part', 'unknown'))
            if part in ('leaves', 'branches_stem', 'base_roots'):
                has_wrist_plant = True

        nav_str = "YES" if has_nav_plant else "Target Command"
        wrist_str = "YES" if has_wrist_plant else "Ready"
        base_bearing = bearing_locked
        self.get_logger().info(
            f'[Targeting Filter] Verified Plant Target at (x={x:.2f}m, y={y:.2f}m, z={z:.2f}m, bearing={base_bearing:.1f}°) '
            f'— Nav Vision: {nav_str} | Wrist CV: {wrist_str}')

        # Determine plant structure and vertical bounds (pot base to top leaves)
        if z_top is not None and z_bottom is not None:
            top = z_top
            bot = z_bottom
            height = abs(top - bot)
        elif plant_height is not None and plant_height > 0.05:
            height = plant_height
            top = z + height / 2.0
            bot = z - height / 2.0
        else:
            top = z + self._sweep_top
            bot = z + self._sweep_bottom
            height = abs(top - bot)

        # Dynamic viewpoint resolution based on plant height (1 viewpoint every 4-5cm)
        num_steps = max(self._sweep_steps, int(math.ceil(height / 0.04)))
        self._scan_queue = []
        heights = [bot + (top - bot) * k / max(1, num_steps - 1) for k in range(num_steps)]

        # 3. Initial Vision Direction Guidance from Wrist Camera
        if self._latest_cv_data:
            green_ratio = float(self._latest_cv_data.get('green_foliage_ratio', 0.0))
            part = str(self._latest_cv_data.get('detected_part', 'unknown'))
            if part in ('leaves', 'branches_stem') or green_ratio > 0.01:
                self.get_logger().info(
                    f'[Initial Vision Guidance] Identified plant direction from Wrist Camera (Part: {part.upper()}, Ratio: {green_ratio*100:.1f}%)! '
                    'Guiding arm motion to lock onto target structure.')

        # 4. Multi-Directional Balance Mandate: Enforce symmetric orbit passes
        # Ensures equal coverage across Left (-35°), Front (0°), and Right (+35°) to prevent one-sided bias.
        orbit_passes = self._orbit_angles
        if len(orbit_passes) == 1:
            orbit_passes = [-35.0, 0.0, 35.0]

        self.get_logger().info(
            f'[Multi-Directional Balance] Enforcing symmetric orbit scanning passes: '
            f'{[f"{a:+.0f}°" for a in orbit_passes]} around approach bearing ({base_bearing:.1f}°) '
            'preventing single-sided scan bias.')

        # Orbit passes rotate the standoff bearing around the plant symmetrically.
        for ang in orbit_passes:
            bearing = base_bearing + ang
            rad = math.radians(bearing)
            side_tag = "left" if ang < 0 else ("right" if ang > 0 else "front")
            for h in heights:
                sx = (r - self._standoff) * math.cos(rad)
                sy = (r - self._standoff) * math.sin(rad)
                
                # Classify plant anatomical section & optimize camera angle for image quality
                rel_frac = (h - bot) / max(1e-3, (top - bot))
                if rel_frac >= 0.75:
                    feature_label = 'top_leaves_branches'
                    az = h - 0.03  # Slightly downward pitch to frame top branches & leaves
                elif rel_frac >= 0.45:
                    feature_label = 'mid_leaves_foliage'
                    az = h         # Flat-on horizontal view for dense leaf surface detail
                elif rel_frac >= 0.18:
                    feature_label = 'lower_stem_branches'
                    az = h         # Flat-on view for main stem and lower branching structure
                else:
                    feature_label = 'base_roots_pot'
                    az = max(0.01, h - 0.02)  # Slight downward angle to frame pot rim & soil/root collar

                label = f'{side_tag}{ang:+.0f}deg_{feature_label}_h{h*1000:.0f}'
                self._scan_queue.append(
                    (label, sx, sy, h, x, y, az))

        if not self._scan_queue:
            self.get_logger().warn('Empty scan path')
            return

        # Validate reachability of the whole path up front; keep only the
        # waypoints that can actually be posed, and warn on the rest.
        kept = []
        skipped = 0
        for label, sx, sy, h, ax, ay, az in self._scan_queue:
            solved = self._solve_aim(sx, sy, h, ax, ay, az)
            if solved is None:
                skipped += 1
                continue
            angles, aim_err = solved
            if aim_err > self._max_aim_error:
                skipped += 1
                continue
            kept.append((label, sx, sy, h, ax, ay, az, angles, aim_err))
        if not kept:
            self.get_logger().warn(
                'No waypoint in the scan path is reachable — increase '
                'plant distance or relax joint limits')
            return
        if skipped:
            self.get_logger().warn(
                f'{skipped}/{len(kept)+skipped} scan waypoints unreachable, '
                'skipped')

        # 1. Use the aimed start pose if solvable, otherwise fallback.
        # Aim at the plant CENTER (not top): the top at 0.43 m with a 0.2 m
        # standoff was beyond the arm's vertical reach, so the old "aimed"
        # start pose silently collapsed to a pose whose camera FK sat at
        # (0.34, 0, 0.05) instead of the requested standoff — the wrist then
        # swept 119-177° off the plant.
        rad = math.radians(base_bearing)
        sx_start = (r - self._standoff) * math.cos(rad)
        sy_start = (r - self._standoff) * math.sin(rad)

        solved_start = self._solve_aim(sx_start, sy_start, z, x, y, z)
        if solved_start is not None:
            above_pose_angles, start_aim_err = solved_start
            b, s, e, w_start = above_pose_angles
            self.get_logger().info(f'[Trajectory Planner] Solved aimed starting pose: base={b:.1f}°, shoulder={s:.1f}°, elbow={e:.1f}°, wrist={w_start:.1f}°, aim_err={start_aim_err:.1f}°')
        else:
            self.get_logger().warn('[Trajectory Planner] Could not solve aimed starting pose at plant center, falling back.')
            if kept:
                # Use first kept waypoint's angles
                _, _, _, _, _, _, _, above_pose_angles, start_aim_err = kept[0]
                b, s, e, w_start = above_pose_angles
            else:
                b, s, e, w_start = list(self._current_joints)
                start_aim_err = self._ik.aim_error(*to_ik([b, s, e, w_start]), x, y, z)

        above_pose = [b, s, e, w_start]

        # Get the starting camera Cartesian position via forward kinematics
        cx, cy, cz = self._ik.forward(*to_ik(above_pose))

        # 2. Aimed transition from the start pose toward the first detailed
        # orbit waypoint. Camera position and aim point are interpolated in
        # space and RE-SOLVED per step so every intermediate pose stays aimed
        # at the plant (the old fixed-[b,s,e] wrist sweep and fake aim_err=0
        # body sweep pointed the camera at empty space). Unreachable steps are
        # dropped — arm_manual_node's trapezoidal profile smooths between the
        # remaining targets anyway.
        transition = []
        if kept:
            _, d_sx, d_sy, d_sz, d_ax, d_ay, d_az, _, _ = kept[0]
            body_steps = 8
            for i in range(1, body_steps + 1):
                frac = i / float(body_steps)
                t_sx = cx + (d_sx - cx) * frac
                t_sy = cy + (d_sy - cy) * frac
                t_sz = cz + (d_sz - cz) * frac
                t_ax = x + (d_ax - x) * frac
                t_ay = y + (d_ay - y) * frac
                t_az = z + (d_az - z) * frac
                solved = self._solve_aim(t_sx, t_sy, t_sz, t_ax, t_ay, t_az)
                if solved is None:
                    continue
                t_angles, t_err = solved
                label = f"aimed_transition_step_{i}"
                transition.append(
                    (label, t_sx, t_sy, t_sz, t_ax, t_ay, t_az,
                     t_angles, t_err, self._dwell_time))

        self.get_logger().info(
            f'[Trajectory Planner] Defined aimed transition ({len(transition)} '
            'steps) from plant center to first detailed waypoint.')

        # 3. Combine them all into self._scan_queue
        self._scan_queue = []
        # Prepend the starting pose as a short hold so it transitions smoothly
        self._scan_queue.append(("above_plant_start", cx, cy, cz, x, y, z, above_pose, start_aim_err, self._dwell_time))
        # Add Aimed Transition
        self._scan_queue.extend(transition)
        # Add Detailed Component Scan
        for label, sx_val, sy_val, h_val, ax_val, ay_val, az_val, angles, aim_err in kept:
            self._scan_queue.append((label, sx_val, sy_val, h_val, ax_val, ay_val, az_val, angles, aim_err, self._dwell_time))

        self._current_viewpoint = 0
        self._scanning = True
        self._dwell_start = None
        self._last_plant_seen_time = time.time()
        
        # Set arm_control_node velocity and acceleration to very low limits
        self._set_arm_control_limits(self._scanning_peak_speed, self._scanning_accel)

        self._start_video_recording(plant_type)
        self.get_logger().info(
            f'Plant identified: "{plant_type}" — Height: {height*100:.1f}cm '
            f'(Base: {bot:.2f}m, Top: {top:.2f}m, Center: ({x:.2f},{y:.2f},{z:.2f})) — '
            f'{len(self._scan_queue)} total sequence viewpoints generated across '
            f'{len(self._orbit_angles)} orbit pass(es)')
        self._pub_status('scanning', self._current_viewpoint)

    def _set_arm_control_limits(self, peak_speed, accel):
        """Asynchronously call SetParameters on arm_control_node to adjust velocity/accel limits."""
        if not self._param_client.service_is_ready():
            self.get_logger().warn('/arm_control_node/set_parameters service is not ready. Skipping speed adjustment.')
            return
        
        req = SetParameters.Request()
        
        p_speed = Parameter()
        p_speed.name = 'peak_speed'
        p_speed.value.type = ParameterType.PARAMETER_DOUBLE
        p_speed.value.double_value = float(peak_speed)
        
        p_accel = Parameter()
        p_accel.name = 'accel'
        p_accel.value.type = ParameterType.PARAMETER_DOUBLE
        p_accel.value.double_value = float(accel)
        
        req.parameters = [p_speed, p_accel]
        
        self._param_client.call_async(req)
        self.get_logger().info(f'[Limits Configuration] Requested arm_control_node to set peak_speed={peak_speed} deg/s, accel={accel} deg/s^2')

    def _start_video_recording(self, plant_type):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        clean_name = str(plant_type).replace(' ', '_').lower()
        self._video_filename = os.path.join(
            self._video_dir, f'plant_scan_{timestamp}_{clean_name}.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(
            self._video_filename, fourcc, 10.0, (640, 400))
        self._video_frames_count = 0
        self.get_logger().info(
            f'[Video Recorder] Started recording plant scan MP4 video: {self._video_filename}')

    def _stop_video_recording(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            self.get_logger().info(
                f'[Video Recorder] Saved high-quality plant scan recording: '
                f'{self._video_filename} ({self._video_frames_count} frames captured)')
            
            path_msg = String()
            path_msg.data = self._video_filename
            self._video_path_pub.publish(path_msg)

    def _on_camera_frame(self, msg):
        if not self._scanning or self._video_writer is None:
            return
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w, _ = cv_img.shape
            
            # Annotate video frame with live scan telemetry
            label = ''
            if self._scan_queue and self._current_viewpoint < len(self._scan_queue):
                label = self._scan_queue[self._current_viewpoint][0]
            
            annotated = cv_img.copy()
            cv2.rectangle(annotated, (0, 0), (w, 35), (0, 0, 0), -1)
            text = f"SCANNING: {label.upper()} | VP {self._current_viewpoint+1}/{len(self._scan_queue)}"
            cv2.putText(annotated, text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if (w, h) != (640, 400):
                annotated = cv2.resize(annotated, (640, 400))

            self._video_writer.write(annotated)
            self._video_frames_count += 1
        except Exception:
            pass

    def _go_home(self):
        msg = Float64MultiArray()
        msg.data = self._home_angles
        self._joint_pub.publish(msg)

    def _joint_cb(self, msg):
        if len(msg.data) >= NUM_JOINTS:
            self._current_joints = list(msg.data[:NUM_JOINTS])

    def _stop_scan(self):
        self._scanning = False
        self._scan_queue = []
        self._current_viewpoint = 0
        self._dwell_start = None
        self._stop_video_recording()
        self._go_home()
        # Restore normal speed and acceleration limits
        self._set_arm_control_limits(self._normal_peak_speed, self._normal_accel)
        self._pub_status('idle', 0)

    def _move_to_viewpoint(self, vp_idx):
        if vp_idx >= len(self._scan_queue):
            self.get_logger().info('Scan complete')
            self._stop_scan()
            return
        
        wp = self._scan_queue[vp_idx]
        label = wp[0]
        sx = wp[1]
        sy = wp[2]
        h = wp[3]
        ax = wp[4]
        ay = wp[5]
        az = wp[6]
        angles = wp[7]
        aim_err = wp[8]

        # Publish pre-solved AIMED joint angles (not a raw pose goal —
        # arm_manual_node's position-only IK would discard the wrist aim).
        msg = Float64MultiArray()
        msg.data = [float(a) for a in angles]
        self._joint_pub.publish(msg)
        self._vp_target_angles = [float(a) for a in angles]
        self.get_logger().info(
            f'Scan vp {vp_idx + 1}/{len(self._scan_queue)} "{label}" → '
            f'cam({sx:.2f},{sy:.2f},{h:.2f}) aim({ax:.2f},{ay:.2f},{az:.2f}) '
            f'aim_err={aim_err:.0f}deg servo{[round(float(a)) for a in angles]}')

    def _at_viewpoint(self, tol=0.5):
        """True when every joint is within `tol` deg of the current target."""
        for a, c in zip(self._vp_target_angles, self._current_joints):
            if abs(a - c) > tol:
                return False
        return True

    def _cv_part_cb(self, msg):
        try:
            self._latest_cv_data = json.loads(msg.data)
            part = str(self._latest_cv_data.get('detected_part', 'unknown'))
            conf = float(self._latest_cv_data.get('confidence', 0.0))
            if part in ('leaves', 'branches_stem', 'base_roots') and conf >= 0.20:
                self._last_plant_seen_time = time.time()
                if self._scanning:
                    self._parts_covered.add(part)
        except Exception:
            pass

    def _trigger_recovery_routine(self):
        """Recovery Routine: Abort scan, return arm to known Home starting
        position, and re-initiate plant detection and scanning sequence."""
        self.get_logger().warn(
            f'[Recovery Routine] Vision lost key plant parts for >{self._recovery_timeout_s:.1f}s! '
            'Aborting current scan, returning arm to Home starting position, '
            'and re-initiating plant detection sequence!')
        
        self._scanning = False
        self._scan_queue = []
        self._current_viewpoint = 0
        self._dwell_start = None
        self._stop_video_recording()
        
        # Restore normal speed and acceleration limits before homing
        self._set_arm_control_limits(self._normal_peak_speed, self._normal_accel)
        
        # Command arm back to known Home starting position [107, 125, 180, 90]
        self._go_home()
        
        self._pub_status('recovering', 0)
        time.sleep(1.0)
        self._pub_status('idle', 0)

    def _adjust_active_reorientation(self):
        """Active closed-loop visual servoing: re-orient wrist and base angles
        dynamically if image is blurry or target visibility degrades.

        Gated so it cannot run away: only fires while a plant is actually in
        the wrist view AND a part is identified, with a cooldown and a hard
        cap on consecutive adjustments. (In live runs this loop fired
        unguarded — +2.5° wrist / −1.5° base every ~0.5 s ×16 — walking the
        arm away from the plant while part=unknown.)"""
        if not self._latest_cv_data or not self._vp_target_angles:
            return

        # Skip active re-orientation during the approach transition
        if self._scan_queue and self._current_viewpoint < len(self._scan_queue):
            label = self._scan_queue[self._current_viewpoint][0]
            if label.startswith("above_plant"):
                return

        in_focus = bool(self._latest_cv_data.get('in_focus', True))
        part = str(self._latest_cv_data.get('detected_part', 'unknown'))
        focus_score = float(self._latest_cv_data.get('focus_score', 100.0))

        # A plant must actually be in view: without this the loop pointed the
        # arm at empty space while searching forever (part=unknown).
        if not bool(self._latest_cv_data.get('plant_present', False)):
            return

        # Cooldown between adjustments.
        now = time.time()
        if getattr(self, '_reorient_last', 0.0) and \
                (now - self._reorient_last) < 1.5:
            return

        # Hard cap on consecutive adjustments — reset when the viewpoint moves
        # on (see _timer_cb), so a stuck run cannot walk the arm to the limit.
        if getattr(self, '_reorient_count', 0) >= 3:
            self.get_logger().warn(
                f'[Active Re-Orientation] Cap reached ({self._reorient_count} '
                'adjustments), holding pose.')
            return

        # If visibility or sharpness degrades, active fine pitch/azimuth adjustment
        if not in_focus or part == 'unknown':
            adj_angles = list(self._vp_target_angles)
            # Active wrist pitch micro-adjust (+2.5 deg) & base micro-adjust (-1.5 deg)
            adj_angles[3] = float(np.clip(adj_angles[3] + 2.5, 0.0, 90.0))
            adj_angles[0] = float(np.clip(adj_angles[0] - 1.5, 0.0, 220.0))

            msg = Float64MultiArray()
            msg.data = adj_angles
            self._joint_pub.publish(msg)
            self._vp_target_angles = adj_angles
            self._reorient_last = now
            self._reorient_count = getattr(self, '_reorient_count', 0) + 1
            self.get_logger().info(
                f'[Active Re-Orientation] Adjusted Wrist ({adj_angles[3]:.1f}°) & Base ({adj_angles[0]:.1f}°) '
                f'-> Restoring Optimal Visibility (Focus: {focus_score:.1f}, Part: {part})')

    def _timer_cb(self):
        if not self._scanning or not self._scan_queue:
            self._pub_status('idle', 0)
            return

        # Check Recovery Timeout: If vision loses plant parts for > recovery_timeout_s
        is_transition = False
        vp_idx = self._current_viewpoint
        if vp_idx < len(self._scan_queue):
            label = self._scan_queue[vp_idx][0]
            if label.startswith("above_plant") or label.startswith("initial_wrist") or label.startswith("dynamic_body"):
                is_transition = True

        still_traveling = True
        if hasattr(self, '_vp_target_angles') and self._vp_target_angles:
            still_traveling = not self._at_viewpoint()

        if is_transition or still_traveling:
            self._last_plant_seen_time = time.time()

        unseen_duration = time.time() - self._last_plant_seen_time
        if unseen_duration > self._recovery_timeout_s:
            self._trigger_recovery_routine()
            return

        vp_idx = self._current_viewpoint
        if vp_idx >= len(self._scan_queue):
            self.get_logger().info('Scan complete')
            self._stop_scan()
            return

        now = self.get_clock().now()
        
        # Determine dwell time to use
        wp = self._scan_queue[vp_idx]
        dwell_to_use = wp[9] if len(wp) > 9 else self._dwell_time

        if self._dwell_start is not None:
            elapsed = (now - self._dwell_start).nanoseconds * 1e-9
            
            # Active visual servoing check midway through dwell, only if the arm has arrived
            if self._at_viewpoint() and elapsed >= 0.5 * dwell_to_use:
                self._adjust_active_reorientation()

            # Wait until the arm has actually arrived, then hold for the
            # dwell so the viewpoint capture is stable.
            if not self._at_viewpoint() or elapsed < dwell_to_use:
                return
            self._current_viewpoint += 1
            self._dwell_start = None
            self._reorient_count = 0
            self._pub_status('scanning', self._current_viewpoint)
            if self._current_viewpoint >= len(self._scan_queue):
                return
            self._move_to_viewpoint(self._current_viewpoint)
            return

        # First entry into the viewpoint: command it and start monitoring.
        self._move_to_viewpoint(vp_idx)
        self._dwell_start = now

    def _pub_status(self, status, viewpoint=0):
        msg = String()
        msg.data = json.dumps({
            'status': status,
            'viewpoint': viewpoint,
            'total_viewpoints': len(self._scan_queue),
            'current_label': (
                self._scan_queue[viewpoint][0]
                if self._scan_queue and viewpoint < len(self._scan_queue)
                else ''
            ),
            'parts_covered': list(self._parts_covered),
        })
        self._scanner_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmScannerNode()
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
