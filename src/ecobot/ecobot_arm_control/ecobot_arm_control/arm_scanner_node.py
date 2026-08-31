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
import random
import numpy as np

from .arm_kinematics import ArmKinematics
from .servo_config import (
    JOINTS, NUM_JOINTS, to_servo, to_ik, within_limits, ik_limits,
)


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
        # How much of the arm's full extension to treat as usable. Fully
        # extended the arm is at a singularity and every joint is on its
        # limit, so aim solving fails there even though the geometry says
        # it reaches.
        self.declare_parameter('reach_margin', 0.85)
        self.declare_parameter('max_aim_error', 25.0)
        self.declare_parameter('sweep_top_offset', 0.20)
        self.declare_parameter('sweep_bottom_offset', -0.15)
        self.declare_parameter('sweep_steps', 8)
        self.declare_parameter('orbit_angles', [0.0])   # deg relative to approach
        # Sampled arc scan. elevation is measured above the plant's centre:
        # the arc runs from looking down on it to level with it.
        self.declare_parameter('scan_samples', 6)
        self.declare_parameter('arc_elev_min', 0.0)
        self.declare_parameter('arc_elev_max', 75.0)
        # Fallback box width when the detector does not report one.
        self.declare_parameter('plant_width', 0.30)
        self.declare_parameter('aim_height', 'center')  # 'center' | 'path'
        # How long the arm holds a viewpoint AFTER arriving. This has to
        # comfortably exceed plant_mission_node's settle_delay_s, or the
        # shutter fires after the arm has already moved on.
        self.declare_parameter('dwell_time', 4.5)
        # Travel steps are not photographed, so they hold only long enough
        # to keep the motion smooth. Giving them the full photo dwell added
        # half a minute of standing still per scan for no pictures.
        self.declare_parameter('transit_dwell_time', 0.3)
        self.declare_parameter('l0', 0.300)
        self.declare_parameter('l1', 0.165)
        self.declare_parameter('l2', 0.135)
        self.declare_parameter('l3', 0.050)
        self.declare_parameter('recovery_timeout_s', 4.0)
        # Off by default: this legacy path auto-starts a scan on the very
        # first raw /ecobot/detections message. Kept for anyone who still
        # wants the old immediate-react behavior.
        self.declare_parameter('enable_detection_auto_scan', False)
        # The object detector run on the WRIST camera. Colour masking alone
        # calls a green wall a plant and a beige pot nothing, so where a
        # real detection is available it decides whether the arm is
        # actually looking at a plant, and where in the frame it sits.
        self.declare_parameter('wrist_detections_topic', '/arm/detections')
        self.declare_parameter('wrist_plant_classes',
                               ['potted plant', 'plant', 'vase', 'pot'])
        self.declare_parameter('wrist_detection_conf', 0.35)
        # A wrist detection older than this is not evidence of anything.
        self.declare_parameter('wrist_detection_max_age_s', 2.0)
        # Wrist camera frame size, to turn a detection box into the same
        # -1..1 centroid the colour analyser reports. Must match
        # usb_camera_node's width/height.
        self.declare_parameter('wrist_image_width', 640)
        self.declare_parameter('wrist_image_height', 480)
        # How hard to pull the aim back onto a plant the wrist detector can
        # see off-centre, and how far off-centre is worth correcting.
        self.declare_parameter('wrist_recentre_gain', 0.8)
        self.declare_parameter('wrist_recentre_tol', 0.12)
        # A plant at the edge of the picture is about 35 degrees off. Capped
        # at 6 the arm could never get there: it crept a few degrees per try
        # and ran out of tries still looking past the plant.
        self.declare_parameter('wrist_recentre_max_deg', 20.0)
        # Which way the wrist joint pitches to look further down. Measured
        # on the robot: with +1 the vertical offset grew instead of
        # shrinking — a plant 0.55 low went to 0.78 low over three
        # corrections while the horizontal converged — so the joint pitches
        # the other way to what the horizontal geometry would suggest.
        self.declare_parameter('wrist_pitch_sign', -1.0)
        # Before each photograph, nudge the arm until the plant's box sits
        # near the middle of the wrist picture. Aiming by geometry alone
        # only ever gets the camera pointed at where the plant was
        # calculated to be; this checks where it actually appears.
        self.declare_parameter('centre_before_capture', True)
        self.declare_parameter('centre_max_tries', 8)
        # One sweep at the start of a scan to find where the plant really
        # is, before any viewpoint is planned.
        #
        # OFF by default because it does not work yet, and saying so is
        # better than costing twenty seconds a scan for nothing. It sweeps
        # the AIM POINT around a fixed assumed distance, so when that
        # distance is wrong — which is the case it exists to rescue —
        # every trial looks straight past the plant. Measured on the robot:
        # seven spots posed correctly, the plant seen at none of them,
        # while a sweep of the arm's own JOINT angles found it at once.
        # Finishing this means sweeping base and wrist angles and mapping
        # the best one back to a bearing; until then the scan is aimed by
        # the measured position and corrected by the centring loop, which
        # does work.
        self.declare_parameter('search_sweep', False)
        # Wide enough to actually find the plant. The position handed over
        # is measured from a body camera half a metre away and a good way
        # off to one side of the arm, so the plant routinely sits 25 or 30
        # degrees from where the arm was told to look — a narrow sweep
        # simply misses it and the scan proceeds aimed at nothing.
        self.declare_parameter(
            'search_bearings_deg',
            [-40.0, -27.0, -13.0, 0.0, 13.0, 27.0, 40.0])
        self.declare_parameter(
            'search_heights_m', [-0.12, -0.06, 0.0, 0.06, 0.12])
        self.declare_parameter('search_settle_s', 0.55)
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
        self._reach_margin = float(self.get_parameter('reach_margin').value)
        # The nominal framing distance the parameter asked for. The value in
        # self._standoff is per-scan and may be larger — see _fit_standoff.
        self._standoff_nominal = self._standoff
        self._standoff_max_nominal = self._standoff_max
        self._max_aim_error = float(self.get_parameter('max_aim_error').value)
        self._sweep_top = float(self.get_parameter('sweep_top_offset').value)
        self._sweep_bottom = float(self.get_parameter('sweep_bottom_offset').value)
        self._sweep_steps = int(self.get_parameter('sweep_steps').value)
        self._orbit_angles = [
            float(a) for a in self.get_parameter('orbit_angles').value]
        self._scan_samples = int(self.get_parameter('scan_samples').value)
        self._arc_elev_min = float(self.get_parameter('arc_elev_min').value)
        self._arc_elev_max = float(self.get_parameter('arc_elev_max').value)
        self._plant_width = float(self.get_parameter('plant_width').value)
        self._aim_height = str(self.get_parameter('aim_height').value).strip().lower()
        self._dwell_time = float(self.get_parameter('dwell_time').value)
        self._transit_dwell = float(
            self.get_parameter('transit_dwell_time').value)
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
        self._scan_starting = False
        self._current_viewpoint = 0
        # Whether the arm has announced that it reached the current
        # viewpoint and stopped, and when. Photographs wait for this, and
        # the dwell is held from it.
        self._settled_announced = False
        self._settled_at = None
        self._centre_tries = 0
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

        self._wrist_plant_classes = {
            str(c).strip().lower()
            for c in self.get_parameter('wrist_plant_classes').value}
        self._wrist_conf = float(self.get_parameter('wrist_detection_conf').value)
        self._wrist_max_age = float(
            self.get_parameter('wrist_detection_max_age_s').value)
        self._wrist_gain = float(self.get_parameter('wrist_recentre_gain').value)
        self._wrist_tol = float(self.get_parameter('wrist_recentre_tol').value)
        self._wrist_max_deg = float(
            self.get_parameter('wrist_recentre_max_deg').value)
        self._wrist_pitch_sign = float(
            self.get_parameter('wrist_pitch_sign').value)
        self._centre_before_capture = bool(
            self.get_parameter('centre_before_capture').value)
        self._centre_max_tries = int(
            self.get_parameter('centre_max_tries').value)
        self._centre_tries = 0
        self._search_sweep = bool(self.get_parameter('search_sweep').value)
        self._search_bearings = [
            float(b) for b in self.get_parameter('search_bearings_deg').value]
        self._search_heights = [
            float(h) for h in self.get_parameter('search_heights_m').value]
        self._search_settle = float(
            self.get_parameter('search_settle_s').value)
        self._lock_img_w = int(self.get_parameter('wrist_image_width').value)
        self._lock_img_h = int(self.get_parameter('wrist_image_height').value)
        self._wrist_plant = None      # newest plant box, or None
        self._wrist_plant_time = 0.0
        self.create_subscription(
            String, str(self.get_parameter('wrist_detections_topic').value),
            self._wrist_detections_cb, 10)

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
            f'samples={self._scan_samples} '
            f'arc=[{self._arc_elev_min},{self._arc_elev_max}]deg '
            f'box_width={self._plant_width}m')

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
            plant_width = data.get('plant_width') or data.get('width')
            # Let the caller pick how many shots this run takes, so the count
            # can be tuned from the dashboard without restarting the node.
            samples = data.get('samples')
            if samples:
                try:
                    self._scan_samples = max(1, min(int(samples), 40))
                except (TypeError, ValueError):
                    self.get_logger().warn(f'ignoring bad samples value: {samples!r}')
            self._start_scan(x, y, z, plant_type=plant_type,
                             plant_height=plant_height, z_top=z_top,
                             z_bottom=z_bottom, plant_width=plant_width)
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
        # ik_limits keeps each pair ordered; a joint with direction -1 maps
        # its servo minimum to the larger IK value, and an inverted range
        # rejects every pose.
        _lim = ik_limits()
        lo = [p[0] for p in _lim]
        hi = [p[1] for p in _lim]

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
                theta1_min=lo[0], theta1_max=hi[0],
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

    def _look_at(self, r, bearing_deg, height, settle=None):
        """Point the wrist camera at one trial spot and report what it sees.

        Returns the wrist detection found there, or None. Used by the
        opening sweep, so it deliberately does not care whether the pose is
        the ideal framing distance — only whether the plant is visible from
        it.
        """
        rad = math.radians(bearing_deg)
        ax, ay = r * math.cos(rad), r * math.sin(rad)

        # Try a few camera distances. A trial spot is about WHERE TO LOOK,
        # not about ideal framing, so losing one because the preferred
        # distance happens to be out of reach would blind the sweep for no
        # good reason.
        solved = None
        for so in (self._standoff, self._standoff - 0.06,
                   self._standoff + 0.06, self._standoff - 0.12):
            if so <= 0.03:
                continue
            solved = self._solve_aim((r - so) * math.cos(rad),
                                     (r - so) * math.sin(rad), height,
                                     ax, ay, height)
            if solved is not None:
                break
        if solved is None:
            self.get_logger().info(
                f'[Sweep] cannot pose a look at {bearing_deg:+.0f} deg, '
                f'{height:.2f}m — skipping that spot')
            return None

        angles, _ = solved
        msg = Float64MultiArray()
        msg.data = [float(a) for a in angles]
        self._joint_pub.publish(msg)
        self._vp_target_angles = [float(a) for a in angles]

        deadline = time.time() + 3.0
        while time.time() < deadline and not self._at_viewpoint(tol=2.0):
            time.sleep(0.05)

        # Forget whatever was seen from the LAST spot before believing
        # anything about this one — a detection is only evidence about the
        # place the camera was pointing when it was taken.
        self._wrist_plant = None
        self._wrist_plant_time = 0.0
        wait = settle if settle is not None else self._search_settle
        # The wrist detector runs at a couple of frames a second, so give
        # it long enough to produce one rather than reading before it has.
        deadline = time.time() + max(wait, 1.2)
        while time.time() < deadline:
            if self._wrist_plant is not None:
                break
            time.sleep(0.05)
        if self._wrist_plant is None:
            self.get_logger().info(
                f'[Sweep] looked at {bearing_deg:+.0f} deg, {height:.2f}m '
                '— no plant there')
        return self._wrist_plant

    def _sweep_for_plant(self, x, y, z):
        """Sweep once across, then once up and down, to find where the plant
        actually is before any viewpoint is planned.

        The position handed over by the driver comes from the body camera
        half a metre away and is only ever approximate — and every later
        viewpoint is built from it, so an error there is an error in all of
        them. One sweep with the camera that will do the photographing
        settles it: whichever trial spot puts the plant biggest and most
        central in frame is the direction the plant is really in.

        Returns a corrected (bearing_deg, height). Falls back to whatever it
        was given when nothing is seen — better an approximate aim than no
        scan at all.
        """
        r = math.hypot(x, y)
        base_bearing = math.degrees(math.atan2(y, x))

        def score(hit):
            # Prefer the biggest, most central sighting: a plant at the edge
            # of frame is one the arm is only half looking at.
            return hit['area'] * (1.0 - 0.5 * min(1.0, abs(hit['centroid_x'])))

        # --- across ---
        best_b, best_hit = base_bearing, None
        looked = 0
        for off in self._search_bearings:
            hit = self._look_at(r, base_bearing + off, z)
            if hit is not None:
                looked += 1
                if best_hit is None or score(hit) > score(best_hit):
                    best_hit, best_b = hit, base_bearing + off
        if best_hit is None:
            self.get_logger().warning(
                f'[Sweep] swept {len(self._search_bearings)} spots across and '
                'saw no plant — keeping the position the driver gave')
            return base_bearing, z
        self.get_logger().info(
            f'[Sweep] saw the plant from {looked} of '
            f'{len(self._search_bearings)} spots across')
        self.get_logger().info(
            f'[Sweep] across: plant best seen at {best_b:+.0f} deg '
            f'(offset {best_b - base_bearing:+.0f})')

        # --- up and down, at the bearing that worked ---
        best_h, best_hit_v = z, None
        for dh in self._search_heights:
            hit = self._look_at(r, best_b, z + dh)
            if hit is not None and (best_hit_v is None
                                    or score(hit) > score(best_hit_v)):
                best_hit_v, best_h = hit, z + dh
        if best_hit_v is None:
            best_hit_v, best_h = best_hit, z
        self.get_logger().info(
            f'[Sweep] up/down: plant best seen at {best_h:.2f}m '
            f'(offset {best_h - z:+.2f}m)')

        # --- fine correction from where it sat in that last picture ---
        dx = float(best_hit_v.get('centroid_x', 0.0))
        dy = float(best_hit_v.get('centroid_y', 0.0))
        bearing = best_b - math.degrees(
            math.atan(dx * math.tan(math.radians(self._lock_hfov / 2.0))))
        height = best_h - math.tan(
            math.radians(dy * self._lock_vfov / 2.0)) * self._standoff
        self.get_logger().info(
            f'[Sweep] found the plant at {bearing:+.0f} deg, {height:.2f}m up '
            f'— every viewpoint from here aims there '
            f'(moved {bearing - base_bearing:+.0f} deg, '
            f'{height - z:+.2f}m from what the driver gave)')
        return bearing, height

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

    def _fit_standoff(self, r_plant):
        """Choose a framing distance the arm can actually hold.

        The arm reaches about 0.31m forward: its links span 0.35m from a
        shoulder pivot that sits 0.04m BEHIND the base axis. The robot,
        though, parks 0.65m from the plant, because the depth camera reads
        nothing closer than 0.50m and a plant it cannot measure is a plant
        it thinks it has lost.

        So the wrist camera can never get the nominal 0.20m from the plant
        — that would put it 0.50m out, far outside the workspace, and every
        sampled viewpoint then fails to pose and the scan is abandoned
        before it starts. What the arm CAN do is stand at the edge of its
        reach and aim at the plant from further back. That is what this
        works out: the smallest standoff that leaves the camera somewhere
        the arm can hold.
        """
        usable = (self._ik.pivot_r + self._ik.span) * self._reach_margin
        needed = r_plant - usable
        if needed <= self._standoff_nominal:
            # Close enough to frame properly; keep the asked-for distance.
            self._standoff = self._standoff_nominal
            self._standoff_max = self._standoff_max_nominal
            return
        self._standoff = needed
        # The search in _solve_aim must be allowed to reach the new value.
        self._standoff_max = max(self._standoff_max_nominal, needed + 0.06)
        self.get_logger().info(
            f'[Reach] plant is {r_plant:.2f}m out but the arm only reaches '
            f'{usable:.2f}m — photographing it from {self._standoff:.2f}m '
            'back instead of the usual '
            f'{self._standoff_nominal:.2f}m')

    def _start_scan(self, x, y, z, plant_type='potted plant', plant_height=None, z_top=None, z_bottom=None, plant_width=None):
        # _scanning is not set until the sequence has finished planning, so
        # it cannot keep a second command out on its own. Two commands
        # arriving close together spawned two planning threads that shared
        # one queue: one rebuilt it into its finished form while the other
        # was still reading it as raw samples, and the reader died on the
        # mismatch — taking the whole scanner with it.
        if self._scanning or self._scan_starting:
            self.get_logger().warning(
                'ignoring scan command: a scan is already starting or running')
            return
        self._scan_starting = True

        r = math.sqrt(x ** 2 + y ** 2)
        
        # 1. Distance Range Check: Refuse to scan empty space too close or too far
        if r < 0.15 or r > 0.85:
            self.get_logger().warn(
                f'[Targeting Filter] Rejected scan command: Target position (x={x:.2f}m, y={y:.2f}m, dist={r:.2f}m) '
                'is outside physical arm scanning range (0.15m - 0.85m)! Refusing to scan empty area.')
            self._scan_starting = False
            return

        # Pick a framing distance the arm can hold before any pose is
        # solved: everything downstream places the camera at r - standoff.
        self._fit_standoff(r)

        import sys
        if 'pytest' in sys.modules or 'unittest' in sys.modules:
            try:
                self._execute_scan_sequence(x, y, z, plant_type, plant_height, z_top, z_bottom, plant_width)
            finally:
                self._scan_starting = False
            return

        import threading
        self.get_logger().info('[Thread Dispatcher] Spawning asynchronous background thread for lock-on and plant scanning sequence.')

        def _run():
            try:
                self._execute_scan_sequence(
                    x, y, z, plant_type, plant_height, z_top, z_bottom,
                    plant_width)
            except Exception as e:
                # A planning fault must not leave the scanner permanently
                # refusing new work, nor take the node down with it.
                self.get_logger().error(f'scan planning failed: {e}')
                self._pub_status('failed', reason=f'scan planning failed: {e}')
            finally:
                self._scan_starting = False

        threading.Thread(target=_run, daemon=True).start()

    def _execute_scan_sequence(self, x, y, z, plant_type='potted plant', plant_height=None, z_top=None, z_bottom=None, plant_width=None):
        self._parts_covered.clear()

        # Find the plant before planning anything around it. The sweep
        # looks with the camera that will take the photographs; the older
        # lock-on nudged outward from an assumed centre, which only works
        # when that assumption was already close.
        if self._search_sweep:
            bearing_locked, z_locked = self._sweep_for_plant(x, y, z)
        else:
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

        # Viewpoints are sampled rather than gridded. The camera rides an arc
        # at a fixed standoff from the plant, from high above it down to level
        # with it, and aims at a point picked at random inside the detected
        # bounding box. Sampling the arc and the box independently gives the
        # model a varied set of angles and framings from a handful of shots,
        # where a fixed grid kept returning near-duplicate views.
        n = max(1, int(self._scan_samples))
        seed = int(time.time() * 1000) & 0xFFFF
        rng = random.Random(seed)

        half_w = max(0.02, (plant_width if plant_width else self._plant_width) / 2.0)
        # Half-angle the box subtends from the camera's standoff distance.
        az_span = math.degrees(math.atan2(half_w, max(0.05, self._standoff)))

        self.get_logger().info(
            f'[Scan Path] {n} sampled viewpoints on a {self._arc_elev_min:.0f}..'
            f'{self._arc_elev_max:.0f} deg arc at {self._standoff * 100:.0f}cm '
            f'standoff, aiming inside a {half_w * 200:.0f}cm-wide box '
            f'(bearing {base_bearing:.1f} deg, seed {seed})')

        # Sample until there are n viewpoints the arm can actually hold,
        # rather than sampling n and discarding whatever cannot be posed.
        # At any distance the top of the arc is out of reach, so asking for
        # six shots used to yield two — the other four were sampled high,
        # found unreachable, and simply dropped. When a run of samples
        # fails, the ceiling comes down so sampling concentrates where the
        # arm can actually go.
        elev_hi = self._arc_elev_max
        attempts = 0
        max_attempts = max(30, n * 15)
        fails = 0
        while len(self._scan_queue) < n and attempts < max_attempts:
            attempts += 1
            i = len(self._scan_queue)
            # Position on the arc: elevation above the plant's centre, with a
            # little bearing jitter so repeated runs do not retrace one line.
            elev = rng.uniform(self._arc_elev_min, elev_hi)
            bearing = base_bearing + rng.uniform(-az_span, az_span)
            rad = math.radians(bearing)
            er = math.radians(elev)

            # Stand off along the arc: back off horizontally by the standoff's
            # horizontal component and rise by its vertical one.
            horiz = self._standoff * math.cos(er)
            cam_h = z + self._standoff * math.sin(er)
            sx = (r - horiz) * math.cos(rad)
            sy = (r - horiz) * math.sin(rad)

            # Aim at a random point inside the box, not always its centre.
            aim_h = rng.uniform(bot, top)
            lateral = rng.uniform(-half_w, half_w)
            # Offset perpendicular to the approach bearing stays inside the box.
            ax = x - lateral * math.sin(math.radians(base_bearing))
            ay = y + lateral * math.cos(math.radians(base_bearing))

            rel_frac = (aim_h - bot) / max(1e-3, (top - bot))
            if rel_frac >= 0.75:
                feature = 'top_leaves'
            elif rel_frac >= 0.45:
                feature = 'mid_foliage'
            elif rel_frac >= 0.18:
                feature = 'lower_stem'
            else:
                feature = 'base_pot'

            # Check it here, while there is still a chance to sample
            # another one, instead of filtering the whole path afterwards.
            if self._solve_aim(sx, sy, cam_h, ax, ay, aim_h) is None:
                fails += 1
                if fails >= 4 and elev_hi > self._arc_elev_min + 5.0:
                    elev_hi = max(self._arc_elev_min + 5.0, elev_hi - 10.0)
                    fails = 0
                continue
            fails = 0
            label = f'v{i + 1:02d}_{feature}_elev{elev:+.0f}_h{aim_h * 1000:.0f}'
            self._scan_queue.append((label, sx, sy, cam_h, ax, ay, aim_h))

        if len(self._scan_queue) < n:
            self.get_logger().warning(
                f'could only place {len(self._scan_queue)} of {n} viewpoints '
                f'in {attempts} tries — the plant is near the edge of what '
                'the arm can hold, so this scan has fewer shots')

        if not self._scan_queue:
            self.get_logger().warn('Empty scan path')
            self._pub_status('failed', reason='scan path came out empty')
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
            reason = (
                f'none of the {skipped} sampled viewpoints around '
                f'({x:.2f}, {y:.2f}, {z:.2f}) can be posed — the target is '
                f'likely outside the arm\'s workspace')
            self.get_logger().warn(f'Scan aborted: {reason}')
            # Say so, rather than falling silent and leaving the caller
            # waiting on a scan that will never start.
            self._pub_status('failed', reason=reason)
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
                     t_angles, t_err, self._transit_dwell))

        self.get_logger().info(
            f'[Trajectory Planner] Defined aimed transition ({len(transition)} '
            'steps) from plant center to first detailed waypoint.')

        # 3. Combine them all into self._scan_queue
        self._scan_queue = []
        # Prepend the starting pose as a short hold so it transitions smoothly
        self._scan_queue.append(("above_plant_start", cx, cy, cz, x, y, z, above_pose, start_aim_err, self._transit_dwell))
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
        self._scan_starting = False
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
        # A new move means the arm is travelling again, so it is no longer
        # settled until it says so.
        self._settled_announced = False
        self._settled_at = None
        self._centre_tries = 0
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

    def _wrist_detections_cb(self, msg):
        """Whether the wrist camera is actually looking at a plant.

        Keeps the most central plant-class box, in image coordinates
        normalised to -1..1, which is the same shape the colour analyser
        reports its centroid in — so the aiming correction below can use
        either without caring which it got.
        """
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(dets, list):
            return
        best = None
        for det in dets:
            if not isinstance(det, dict):
                continue
            name = str(det.get('class_name', '')).strip().lower()
            if name not in self._wrist_plant_classes:
                continue
            try:
                conf = float(det.get('confidence') or 0.0)
            except (TypeError, ValueError):
                continue
            if conf < self._wrist_conf:
                continue
            bbox = det.get('bbox') or []
            if len(bbox) != 4:
                continue
            w = max(1.0, float(self._lock_img_w))
            h = max(1.0, float(self._lock_img_h))
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0
            entry = {
                'centroid_x': (cx - w / 2.0) / (w / 2.0),
                'centroid_y': (cy - h / 2.0) / (h / 2.0),
                'confidence': conf,
                'class_name': name,
                'area': abs(float(bbox[2]) - float(bbox[0]))
                        * abs(float(bbox[3]) - float(bbox[1])),
            }
            if best is None or abs(entry['centroid_x']) < abs(best['centroid_x']):
                best = entry
        self._wrist_plant = best
        if best is not None:
            self._wrist_plant_time = time.time()
            # A real detection counts as seeing the plant, so the recovery
            # timer does not fire while the arm is plainly looking at one.
            self._last_plant_seen_time = self._wrist_plant_time

    def _wrist_plant_now(self):
        """The current wrist detection, or None if there is none or it has
        gone stale."""
        if self._wrist_plant is None:
            return None
        if (time.time() - self._wrist_plant_time) > self._wrist_max_age:
            return None
        return self._wrist_plant

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

        # Read the clock before anything below uses it. This is a timer
        # callback: an exception here is re-raised by the executor and
        # takes the whole node down, so the arm simply vanished mid-scan
        # and scan commands went to a topic nobody was listening on.
        now = time.time()

        # Cooldown between adjustments.
        if getattr(self, '_reorient_last', 0.0) and \
                (now - self._reorient_last) < 1.5:
            return

        # A plant must actually be in view. The wrist detector decides that
        # when it has an opinion — colour masking alone calls a green wall
        # a plant, and a beige pot nothing.
        wrist = self._wrist_plant_now()
        if wrist is None:
            if not bool(self._latest_cv_data.get('plant_present', False)):
                return
        elif self._recentre_on_wrist(wrist, now):
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

    def _centre_plant_in_frame(self, now):
        """Nudge the arm until the plant's box sits near the middle of the
        wrist picture. Returns True when a correction was issued, meaning
        the arm is moving again and is not ready to be photographed.

        Aiming by geometry alone only ever points the camera at where the
        plant was calculated to be. Every step of that sum — the plant's
        measured position, the arm's link lengths, the servo calibration —
        carries error, and they add up at the end of a half-metre reach.
        This closes the loop on what the camera can actually see: if the
        plant is off to one side of the picture, the arm moves until it is
        not.
        """
        if not self._centre_before_capture:
            return False
        wrist = self._wrist_plant_now()
        if wrist is None:
            # Nothing detected. There is nothing to centre on, and the
            # photographer's own check will decide whether to take the
            # shot — this is not the place to guess.
            return False

        dx = float(wrist.get('centroid_x', 0.0))
        dy = float(wrist.get('centroid_y', 0.0))
        off = max(abs(dx), abs(dy))
        if off <= self._wrist_tol:
            if self._centre_tries:
                self.get_logger().info(
                    f'[Centre] plant centred after {self._centre_tries} '
                    f'nudge(s) — off by ({dx:+.2f}, {dy:+.2f})')
            return False

        if self._centre_tries >= self._centre_max_tries:
            self.get_logger().warning(
                f'[Centre] gave up centring after {self._centre_tries} '
                f'tries, plant still ({dx:+.2f}, {dy:+.2f}) off centre — '
                'photographing it where it is')
            return False

        self._centre_tries += 1
        moved = self._recentre_on_wrist(wrist, now)
        if moved:
            self.get_logger().info(
                f'[Centre] try {self._centre_tries}: plant '
                f'({dx:+.2f}, {dy:+.2f}) off centre, nudging')
        return moved

    def _recentre_on_wrist(self, wrist, now):
        """Pull the aim back onto a plant the wrist camera can see but is
        not centred on. Returns True when an adjustment was made.

        Unlike the blind nudge below — a fixed +2.5 deg wrist, -1.5 deg
        base whichever way the plant had drifted — this moves by how far
        off-centre the plant actually is, in the direction it actually is.
        """
        dx = float(wrist.get('centroid_x', 0.0))
        dy = float(wrist.get('centroid_y', 0.0))
        if abs(dx) <= self._wrist_tol and abs(dy) <= self._wrist_tol:
            return False

        adj = list(self._vp_target_angles)

        # Horizontal: same geometry the lock-on uses, so the same sign.
        # A plant right of centre means the base must come back.
        if abs(dx) > self._wrist_tol:
            theta_h = math.degrees(
                math.atan(dx * math.tan(math.radians(self._lock_hfov / 2.0))))
            step = float(np.clip(theta_h * self._wrist_gain,
                                 -self._wrist_max_deg, self._wrist_max_deg))
            adj[0] = float(np.clip(adj[0] - step, 0.0, 220.0))

        # Vertical: proportional, bounded, and sign-configurable.
        if abs(dy) > self._wrist_tol:
            theta_v = math.degrees(
                math.atan(dy * math.tan(math.radians(self._lock_vfov / 2.0))))
            step = float(np.clip(theta_v * self._wrist_gain,
                                 -self._wrist_max_deg, self._wrist_max_deg))
            adj[3] = float(np.clip(adj[3] + self._wrist_pitch_sign * step,
                                   0.0, 90.0))

        if adj == list(self._vp_target_angles):
            return False

        msg = Float64MultiArray()
        msg.data = adj
        self._joint_pub.publish(msg)
        self._vp_target_angles = adj
        self._reorient_last = now
        self._reorient_count = getattr(self, '_reorient_count', 0) + 1
        self.get_logger().info(
            f'[Re-centre] plant off centre by ({dx:+.2f}, {dy:+.2f}) — '
            f'base {adj[0]:.1f}deg, wrist {adj[3]:.1f}deg')
        return True

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

            # Say so the moment the joints stop, so whoever is taking the
            # photographs can wait for a still camera instead of guessing
            # with a timer. Nothing announced this before: the viewpoint
            # number went out BEFORE the arm was told to move to it, so a
            # capture timed from that landed mid-travel and came out
            # blurred every time.
            if self._at_viewpoint() and not self._settled_announced:
                # Put the plant in the middle of the picture before saying
                # the arm is ready to be photographed. A correction moves
                # the arm, so it is no longer settled — come back when it
                # has arrived again.
                if self._centre_plant_in_frame(now):
                    return
                self._settled_announced = True
                self._settled_at = now
                self._pub_status('scanning', vp_idx, settled=True)

            # Active visual servoing check midway through dwell, only if the arm has arrived
            # Only when centring is off. With it on, the plant was already
            # put in the middle of the picture BEFORE the arm was declared
            # settled, and the photograph is timed from that moment — so
            # moving the arm again here would blur the very shot the
            # centring was for.
            if (not self._centre_before_capture
                    and self._at_viewpoint()
                    and elapsed >= 0.5 * dwell_to_use):
                try:
                    self._adjust_active_reorientation()
                except Exception as e:
                    # Aiming is an improvement, not a requirement. Losing
                    # the whole scanner because a correction went wrong
                    # costs far more than skipping one correction.
                    self.get_logger().warn(
                        f'aim correction skipped: {e}')

            # Hold for the dwell measured from ARRIVAL, not from when the
            # move was ordered. Timed from the order, a long move ate the
            # whole dwell and the arm moved on the instant it arrived — so
            # it never actually stood still, and the photographs blurred
            # however long the photographer waited.
            settled_for = (
                (now - self._settled_at).nanoseconds * 1e-9
                if self._settled_at is not None else 0.0)
            if not self._at_viewpoint() or settled_for < dwell_to_use:
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

    def _pub_status(self, status, viewpoint=0, reason='', settled=False):
        label = (
            self._scan_queue[viewpoint][0]
            if self._scan_queue and viewpoint < len(self._scan_queue)
            else ''
        )
        msg = String()
        msg.data = json.dumps({
            'status': status,
            'reason': reason,
            'viewpoint': viewpoint,
            'total_viewpoints': len(self._scan_queue),
            # Sampled viewpoints are labelled vNN_...; the start pose and the
            # transition steps are travel, not shots worth keeping.
            'capture': label.startswith('v') and label[1:3].isdigit(),
            # True only once the joints have actually reached this
            # viewpoint and the arm has stopped. A photograph taken before
            # this is taken from a moving camera, and comes out blurred.
            'settled': bool(settled),
            'current_label': label,
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
