import json
import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
import tf2_ros
import numpy as np
from cv_bridge import CvBridge


class DetectionGoto(Node):
    def __init__(self):
        super().__init__('detection_goto')

        self.declare_parameter('cmd_vel_topic', '/goto_cmd_vel')
        self.declare_parameter('detections_topic', '/ecobot/detections')
        self.declare_parameter('target_topic', '/ecobot/goto_target')
        self.declare_parameter('status_topic', '/ecobot/goto_status')
        self.declare_parameter('waypoints_topic', '/ecobot/waypoints')
        self.declare_parameter('odom_topic', '/odom')
        # 0.3m sits right at/inside the RealSense D415's minimum reliable
        # depth range — the tracked distance goes noisy (can even read as
        # increasing) exactly as the robot closes in, so it never cleanly
        # converges to "reached". 0.4m keeps the base's approach inside
        # depth range the camera can actually measure; the arm (own wrist
        # camera, unaffected by this) covers the remaining gap during its
        # scan. This may still need empirical tuning per-unit — override
        # with --ros-args -p stop_distance:=0.xx without a rebuild.
        # Nearest depth the camera can actually return; anything closer reads
        # as invalid and makes the target effectively invisible.
        self.declare_parameter('min_depth_range', 0.55)
        self.declare_parameter('stop_distance', 0.65)
        self.declare_parameter('wp_stop_distance', 0.15)
        self.declare_parameter('max_linear', 0.35)
        self.declare_parameter('max_angular', 1.2)
        self.declare_parameter('k_ang', 2.0)
        self.declare_parameter('k_lin', 1.0)
        self.declare_parameter('align_threshold', 0.35)
        # How centered the object must be (bearing angle in rad) before a
        # stop is allowed — the robot parks facing the object centrally.
        self.declare_parameter('stop_align_threshold', 0.09)
        # Within this depth (m) the approach starts trading forward speed
        # for heading alignment so it arrives dead-ahead of the plant.
        self.declare_parameter('final_align_dist', 1.0)
        self.declare_parameter('view_fill_fraction', 0.6)
        self.declare_parameter('view_image_width', 640)
        self.declare_parameter('view_image_height', 480)
        self.declare_parameter('fill_filter', 0.3)
        self.declare_parameter('min_stop_distance', 0.4)
        # Extra turn (rad) bias disabled to maintain dead-center alignment.
        self.declare_parameter('clip_turn_bias', 0.0)
        self.declare_parameter('lost_timeout', 2.0)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('obstacle_stop_distance', 0.3)
        self.declare_parameter('search_timeout', 20.0)
        # The boot-time 360 scan surveys the room and then drives to the
        # nearest plant it found. It ends only when ODOMETRY has measured a
        # full turn, so any under-count leaves the robot rotating for ever,
        # ignoring plants it can see perfectly well. Bound it by time too.
        self.declare_parameter('startup_scan_timeout_s', 45.0)
        # Switch the survey off to go straight to "see a plant, drive to it".
        self.declare_parameter('enable_startup_scan', True)
        self.declare_parameter('search_speed', 0.25)
        self.declare_parameter('avoid_distance', 0.4)
        self.declare_parameter('avoid_angle', 1.05)
        self.declare_parameter('use_map_frame', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('blind_approach_limit', 1.0)
        # Classes that get tracked automatically the moment they're seen,
        # with no dashboard 'select' click needed. Empty list (the
        # previous behavior) means fully manual selection only — this is
        # an explicit opt-in since it makes the robot start driving on
        # its own as soon as a matching object is detected.
        self.declare_parameter('auto_track_classes', ['potted plant'])
        # Handed off to plant_mission_node's 'scan_here' command (not
        # /arm/scanner_cmd directly) the moment an auto-tracked plant is
        # reached — that node already owns wrist-camera capture, the
        # Gemini health call, and dashboard status reporting, so this
        # reuses all of that instead of just waving the arm with nothing
        # watching. 'scan_here' skips navigation entirely since this
        # node's own controller already positioned the robot.
        self.declare_parameter('plant_scan_cmd_topic', '/ecobot/plant_scan_cmd')
        # Tells obstacle_avoidance.py (a separate, independent safety
        # layer downstream on /goto_cmd_vel -> /cmd_vel) to stand down
        # its own depth-based avoidance/escape maneuvers while true —
        # otherwise it sees the very plant being approached as something
        # to swerve/reverse away from once within its safe_distance
        # (default 0.9m), long before this node's own stop_distance is
        # reached. Only suppresses for auto-tracked targets — a
        # manually-selected chase target keeps normal avoidance.
        self.declare_parameter('suppress_avoidance_topic',
                                '/ecobot/goto_suppress_avoidance')
        self.declare_parameter('auto_scan_on_reach', True)
        # After stopping at a plant and scanning it, resume following the
        # saved waypoint track — turn around and continue to the next
        # point. resume_timeout bounds how long we wait for the arm scan
        # to finish before driving on anyway.
        self.declare_parameter('resume_track', True)
        self.declare_parameter('resume_timeout', 60.0)
        # How long one scan announcement holds the base. Short, because the
        # status republishes several times a second while a scan is live.
        self.declare_parameter('scan_hold_grace', 3.0)
        # Minimum depth (m) for a plant to be auto-selected while idle.
        # After inspecting a plant the robot is parked 0.4m from it;
        # requiring a farther target keeps auto-resume from immediately
        # re-picking the same plant and looping forever.
        self.declare_parameter('auto_resume_min_dist', 0.8)
        self.declare_parameter('plant_scan_status_topic',
                                '/ecobot/plant_scan_status')
        # The arm scanner's OWN status topic. Detection/goto must NOT rotate
        # away until the scanner reports idle — plant_mission_node abandons
        # a plant early on scan-settle timeout (status COMPLETE/WAITING/ERROR)
        # while the arm scanner is still mid-sweep, and trusting the mission
        # status made the robot drive off mid-scan.
        self.declare_parameter('scanner_status_topic',
                                '/arm/scanner_status')
        self.declare_parameter('accel_limit', 0.35)
        self.declare_parameter('decel_limit', 0.5)
        self.declare_parameter('omega_accel', 1.5)
        self.declare_parameter('z_filter', 0.7)
        self.declare_parameter('angle_filter', 0.85)
        self.declare_parameter('stop_confirm_ticks', 2)
        self.declare_parameter('min_creep', 0.02)

        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        target_topic = str(self.get_parameter('target_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)
        self.waypoints_topic = str(self.get_parameter('waypoints_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.min_depth_range = float(
            self.get_parameter('min_depth_range').value)
        self.stop_distance = float(self.get_parameter('stop_distance').value)
        # The depth camera cannot measure closer than about 0.5m — its nearest
        # valid reading on this robot is 0.50m. Stopping inside that blind zone
        # is impossible to detect: the plant's depth goes invalid on the final
        # approach, the tracker drops the detection for want of a z, and the
        # robot decides it has lost the target and rotates to search instead of
        # arriving. Keep the stop distance far enough out that the plant stays
        # measurable all the way in.
        self.stop_distance = max(self.min_depth_range + 0.1,
                                 min(self.stop_distance, 1.5))
        self.wp_stop_distance = float(
            self.get_parameter('wp_stop_distance').value)
        self.max_linear = float(self.get_parameter('max_linear').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.k_ang = float(self.get_parameter('k_ang').value)
        self.k_lin = float(self.get_parameter('k_lin').value)
        self.align_threshold = float(
            self.get_parameter('align_threshold').value)
        self.stop_align_threshold = float(
            self.get_parameter('stop_align_threshold').value)
        self.final_align_dist = float(
            self.get_parameter('final_align_dist').value)
        self.view_fill_fraction = float(
            self.get_parameter('view_fill_fraction').value)
        self.view_image_width = float(
            self.get_parameter('view_image_width').value)
        self.view_image_height = float(
            self.get_parameter('view_image_height').value)
        self.fill_filter = float(self.get_parameter('fill_filter').value)
        self.min_stop_distance = float(
            self.get_parameter('min_stop_distance').value)
        self.clip_turn_bias = float(
            self.get_parameter('clip_turn_bias').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.obstacle_stop_dist = float(
            self.get_parameter('obstacle_stop_distance').value)
        self.search_timeout = float(self.get_parameter('search_timeout').value)
        self.startup_scan_timeout_s = float(
            self.get_parameter('startup_scan_timeout_s').value)
        self._enable_startup_scan = bool(
            self.get_parameter('enable_startup_scan').value)
        self.search_speed = float(self.get_parameter('search_speed').value)
        self.avoid_distance = float(self.get_parameter('avoid_distance').value)
        self.avoid_angle = float(self.get_parameter('avoid_angle').value)
        self.use_map_frame = bool(self.get_parameter('use_map_frame').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self.blind_approach_limit = float(
            self.get_parameter('blind_approach_limit').value)
        self.auto_track_classes = {
            str(c).strip().lower()
            for c in self.get_parameter('auto_track_classes').value}
        self._auto_tracked = False
        # Once an auto-tracked episode ends (reached, lost, or cancelled),
        # auto-tracking stays paused indefinitely — it will NOT pick a
        # new (or the same) target on its own. Only an explicit
        # 'resume_auto_track' command on the target topic clears this.
        self._auto_track_paused = False
        # While true, obstacle_avoidance.py keeps its depth-based
        # wander/reverse-away maneuvers disabled so a robot parked right
        # in front of a reached plant doesn't get shoved backward by its
        # own safety layer. Set on a plant reach, cleared on a new target
        # selection or 'resume_auto_track'.
        self._parked_suppress = False
        self.auto_scan_on_reach = bool(
            self.get_parameter('auto_scan_on_reach').value)
        self.resume_track = bool(self.get_parameter('resume_track').value)
        self.resume_timeout = float(
            self.get_parameter('resume_timeout').value)
        self.scan_hold_grace = float(
            self.get_parameter('scan_hold_grace').value)
        self.auto_resume_min_dist = float(
            self.get_parameter('auto_resume_min_dist').value)
        # Keep real separation between where the robot parks and how far a
        # plant must be to count as a new one. With stop_distance at 0.65
        # the default 0.8 leaves only 15cm, so depth noise alone could push
        # the just-scanned plant back over the gate and it would re-pick
        # the plant it just finished, forever.
        self.auto_resume_min_dist = max(
            self.auto_resume_min_dist, self.stop_distance + 0.5)
        self.get_logger().info(
            f'auto_resume_min_dist={self.auto_resume_min_dist:.2f}m '
            f'(stop_distance={self.stop_distance:.2f}m)')
        self._scan_status_topic = str(
            self.get_parameter('plant_scan_status_topic').value)
        self._scanner_status_topic = str(
            self.get_parameter('scanner_status_topic').value)
        self.accel_limit = float(self.get_parameter('accel_limit').value)
        self.decel_limit = float(self.get_parameter('decel_limit').value)
        self.omega_accel = float(self.get_parameter('omega_accel').value)
        self.z_filter = float(self.get_parameter('z_filter').value)
        self.angle_filter = float(self.get_parameter('angle_filter').value)
        self.stop_confirm_ticks = int(
            self.get_parameter('stop_confirm_ticks').value)
        self.min_creep = float(self.get_parameter('min_creep').value)

        self.detections = []
        self.target_class = None
        self.target_pos = (0.0, 0.0, 0.0)
        # With the survey switched off the node stays idle until a plant
        # comes into view, and the ordinary auto-track path takes it. That
        # path is skipped entirely while the survey runs, because it only
        # gets called when the node is inactive.
        self.active = bool(self._enable_startup_scan)
        self._mode = 'startup_scan' if self._enable_startup_scan else 'idle'
        self._wp_target = (0.0, 0.0)
        self._wp_frame = 'odom'
        self.last_seen = None
        self.status = 'IDLE'
        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_lock = threading.Lock()
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._odom_set = False
        self._map_available = False
        self._map_tx = 0.0
        self._map_ty = 0.0
        self._map_tyaw = 0.0

        self.waypoints = []
        self._saved_class = None
        self._last_tracked_det = None
        self._blind_start = None
        self._blind_last_z = 0.0
        self._blind_heading = 0.0
        self._blind_start_yaw = 0.0

        self._avoid_state = 'none'
        self._avoid_dir = 1.0
        self._avoid_start = None

        self._fz = None
        self._fang = None
        self._ffill = None
        self._prev_lin = 0.0
        self._prev_ang = 0.0
        self._stop_confirm = 0
        self._last_ctrl = None

        # waypoint-track bookkeeping so the robot can turn around and
        # continue along the route after inspecting a plant
        self._wp_index = -1
        self._on_track = False
        self._resume_pending = False
        self._resume_since = None
        self._scan_done = False
        self._scan_requested = False
        self._scanner_scanning = False
        # Deadline-based hold covering the gap between a scan being
        # announced and the arm reporting that it is actually moving.
        # Refreshed by every SCANNING/ANALYZING status, so it lapses on its
        # own if the mission goes quiet rather than pinning the base.
        self._scan_hold_until = None
        # True once the arm has actually been seen sweeping for this request.
        # Both status sources sit at a terminal/idle value in the gap between
        # asking for a scan and the arm starting one, so without this the very
        # first message after the request reads as "already finished" and the
        # robot drives off while the arm is still working.
        self._scan_started = False
        # Set right after a plant is reached; while true, _maybe_auto_track
        # requires targets farther than auto_resume_min_dist so it doesn't
        # immediately re-pick the plant parked 0.4m in front. Cleared once
        # a new target is selected (or a fresh waypoint command).
        self._post_inspection_gate = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.plant_scan_cmd_pub = self.create_publisher(
            String, str(self.get_parameter('plant_scan_cmd_topic').value), 10)
        self.suppress_avoidance_pub = self.create_publisher(
            Bool, str(self.get_parameter('suppress_avoidance_topic').value), 10)
        wp_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)
        self.wp_pub = self.create_publisher(
            String, self.waypoints_topic, wp_qos)
        self.create_subscription(String, detections_topic,
                                 self.detections_cb, 10)
        self.create_subscription(String, target_topic, self.target_cb, 10)
        self.create_subscription(Image, depth_topic, self.depth_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(
            String, '/ecobot/tof_ranges', self.tof_cb, 10)
        self.latest_tof = None
        self.create_subscription(
            String, self._scan_status_topic, self._scan_status_cb, 10)
        self.create_subscription(
            String, self._scanner_status_topic, self._scanner_status_cb, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self._publish_waypoints()
        self.get_logger().info(
            f'detection_goto started — stop={self.stop_distance}m '
            f'wp_stop={self.wp_stop_distance}m '
            f'search={self.search_timeout}s@{self.search_speed}rad/s '
            f'avoid={self.avoid_distance}m@{math.degrees(self.avoid_angle):.0f}deg '
            f'map={self.use_map_frame}')

    def detections_cb(self, msg):
        try:
            dets = json.loads(msg.data)
            if isinstance(dets, list):
                self.detections = dets
        except Exception:
            pass

    def tof_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_tof = data.get('ranges_m')
        except Exception:
            pass

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._odom_x = p.x
        self._odom_y = p.y
        self._odom_yaw = math.atan2(
            2.0 * (o.w * o.z), 1.0 - 2.0 * (o.z * o.z))
        self._odom_set = True

    def _update_map_pose(self):
        if not self.use_map_frame or not self._odom_set:
            self._map_available = False
            return
        try:
            t = self.tf_buffer.lookup_transform(
                self._map_frame, 'odom',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
            self._map_tx = t.transform.translation.x
            self._map_ty = t.transform.translation.y
            self._map_tyaw = yaw
            self._map_available = True
        except Exception:
            self._map_available = False

    def _to_map(self, ox, oy):
        """Convert odom-frame (x, y) to map-frame (x, y)."""
        if not self._map_available:
            return ox, oy
        cos_m = math.cos(self._map_tyaw)
        sin_m = math.sin(self._map_tyaw)
        return (self._map_tx + cos_m * ox - sin_m * oy,
                self._map_ty + sin_m * ox + cos_m * oy)

    def _current_pose(self):
        """Return (x, y, yaw) in the navigation frame (map if available)."""
        if self._map_available:
            mx, my = self._to_map(self._odom_x, self._odom_y)
            myaw = self._odom_yaw + self._map_tyaw
            myaw = math.atan2(math.sin(myaw), math.cos(myaw))
            return mx, my, myaw
        return self._odom_x, self._odom_y, self._odom_yaw

    def target_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        action = data.get('action')
        if action == 'select':
            det = data.get('detection') or {}
            cls = det.get('class_name') or det.get('class')
            if cls:
                self._select_target(cls, det, auto=False)
        elif action == 'goto_waypoint':
            idx = data.get('index', 0)
            if 0 <= idx < len(self.waypoints):
                wp = self.waypoints[idx]
                self._mode = 'waypoint'
                self._wp_index = idx
                self._on_track = True
                self._resume_pending = False
                self._post_inspection_gate = False
                self._wp_target = (float(wp['x']), float(wp['y']))
                self._wp_frame = wp.get('frame', 'odom')
                self.target_class = wp.get('class_name', 'waypoint')
                self.active = True
                self._avoid_state = 'none'
                self._publish_status()
                self.get_logger().info(
                    f'goto waypoint {idx}: {self.target_class} '
                    f'({self._wp_target[0]:.3f},{self._wp_target[1]:.3f}) '
                    f'frame={self._wp_frame}')
        elif action == 'cancel':
            self.get_logger().info('goto cancelled')
            self._stop('IDLE')
        elif action == 'clear_waypoints':
            self.waypoints = []
            self._publish_waypoints()
            self.get_logger().info('waypoints cleared')
        elif action == 'pause_auto_track':
            # Stand down completely for the duration of a waypoint mission.
            # Nav2 owns the wheels then, and this node must neither pick a
            # target of its own nor rotate away while the arm is scanning.
            # A plain 'cancel' is not enough: with no target selected it
            # leaves auto-track unpaused, so the very next detection starts
            # a fresh chase. The control loop keeps publishing a zero
            # cmd_vel while stopped, which is what stops the downstream
            # safety layer reading "no command source" and creeping.
            self._stop('IDLE')
            self._auto_track_paused = True
            self._resume_pending = False
            self._parked_suppress = False
            self._post_inspection_gate = False
            self.get_logger().info('auto-track paused by mission')
        elif action == 'resume_auto_track':
            self._auto_track_paused = False
            self._parked_suppress = False
            self._resume_pending = False
            self._post_inspection_gate = False
            self.get_logger().info('auto-track resumed')

    def _select_target(self, cls, det, auto):
        self._mode = 'tracking'
        self.target_class = cls
        self.target_pos = (float(det.get('x', 0.0)),
                           float(det.get('y', 0.0)),
                           float(det.get('z', 0.0)))
        self.active = True
        self.last_seen = self.get_clock().now()
        self._saved_class = None
        self._avoid_state = 'none'
        self._auto_tracked = auto
        self._parked_suppress = False
        self._on_track = False
        self._resume_pending = False
        self._scan_requested = False
        self._scan_hold_until = None
        self._post_inspection_gate = False
        self._fz = None
        self._fang = None
        self._ffill = None
        self._stop_confirm = 0
        self._prev_lin = 0.0
        self._prev_ang = 0.0
        self._publish_status()
        self.get_logger().info(
            f'goto target {"auto-" if auto else ""}selected: {cls} '
            f'pos=({self.target_pos[0]:.3f},{self.target_pos[2]:.3f})')

    def _maybe_auto_track(self):
        """Start tracking the first detection matching auto_track_classes
        with no dashboard selection needed. Only fires while idle and not
        paused — once an auto-tracked episode ends, this stays off
        indefinitely (no new/repeat target on its own) until an explicit
        'resume_auto_track' command clears the pause."""
        if self.active or not self.auto_track_classes or self._auto_track_paused:
            return
        for d in self.detections:
            cls = str(d.get('class_name') or d.get('class') or '').strip().lower()
            z = d.get('z')
            # Right after inspecting a plant the robot is parked 0.4m
            # from it; while the post-inspection gate is active, only
            # pick targets beyond auto_resume_min_dist so it doesn't
            # immediately re-pick the same plant and loop forever.
            if (cls in self.auto_track_classes and z is not None
                    and (not self._post_inspection_gate
                         or z > self.auto_resume_min_dist)):
                self._select_target(cls, d, auto=True)
                return

    def _scan_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        # plant_mission_node reaches a terminal state once the arm scan +
        # analysis finishes (or errors). The arm scanner itself is the
        # authoritative source of "arm actually done moving" — a mission
        # ERROR/COMPLETE can fire on scan-settle timeout while the scanner
        # is still sweeping, so only treat this as scan-done when the
        # scanner also reports idle (see _scanner_status_cb).
        status = str(data.get('status', ''))
        if status in ('SCANNING', 'ANALYZING', 'ANALYSING', 'NAVIGATING'):
            # Deliberately does NOT set _scan_started. The mission node
            # reports SCANNING as soon as it accepts the request, well
            # before the arm moves; treating that as "started" let the
            # scanner's pre-scan 'idle' heartbeat immediately count as
            # "finished" and the robot drove off mid-sweep. It does hold
            # the base though: the arm can begin moving a control tick
            # before its own status arrives, and that tick was enough to
            # leak one search-rotation command into the sweep.
            self._scan_hold_until = (
                self.get_clock().now()
                + Duration(seconds=self.scan_hold_grace))
            return
        if status in ('COMPLETE', 'WAITING', 'IDLE', 'STOPPED', 'ERROR'):
            if self._scanner_scanning or not self._scan_started:
                return
            self._scan_done = True
            self._scan_hold_until = None

    def _scanner_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        # Arm scanner status: 'scanning' / 'idle' / 'recovering'. While it
        # reports scanning, the arm is mid-sweep — do not let the robot
        # drive away even if plant_mission_node has moved on.
        status = str(data.get('status', ''))
        self._scanner_scanning = (status == 'scanning')
        if status == 'scanning':
            self._scan_started = True
            return
        if status == 'failed':
            # Nothing is going to happen; stop waiting on it.
            self.get_logger().warn(
                f'arm scan failed: {data.get("reason", "no reason given")}')
            self._scan_done = True
            self._scan_hold_until = None
            return
        if status in ('idle', 'recovering'):
            # Only after the arm has been seen sweeping — this topic reads
            # idle in the gap before the scan begins as well.
            if self._scan_requested and self._scan_started:
                self._scan_done = True
                self._scan_hold_until = None

    def _trigger_arm_scan(self):
        """Hand off to ecobot_mission's plant_mission_node via its
        'scan_here' command — that node owns the actual wrist-camera
        capture + Gemini health call + dashboard status reporting, so
        this reuses all of that instead of just waving the arm with
        nothing watching. 'scan_here' skips navigation since this
        node's own controller already positioned the robot."""
        self._scan_done = False
        self._scan_requested = True
        self._scan_started = False
        self._scan_hold_until = (
            self.get_clock().now() + Duration(seconds=self.scan_hold_grace))
        self.plant_scan_cmd_pub.publish(
            String(data=json.dumps({'action': 'scan_here'})))
        self.get_logger().info(
            f'auto-reached {self.target_class} — requesting scan_here')

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.latest_depth = depth
        except Exception:
            pass

    def _obstacle_blocking(self, reference_z=None):
        """reference_z: current distance to the thing we're intentionally
        driving toward, if any. Without it, stop_distance == 0.3m and
        obstacle_stop_distance == 0.3m by default means the target itself
        (a plant right in front of the depth camera) reads as a "blocking
        obstacle" on the final approach, and the robot swerves into
        avoidance instead of stopping normally at it. When reference_z is
        given, only something meaningfully closer than the target counts
        as a foreign obstacle to avoid."""
        with self.depth_lock:
            depth = self.latest_depth
        if depth is None:
            return False
        h, w = depth.shape[:2]
        rows = slice(h // 2, 3 * h // 4)
        cols = slice(w // 3, 2 * w // 3)
        region = depth[rows, cols]
        valid = region[(region > 0) & (region < 5000)]
        if len(valid) < 50:
            return False
        sensed = float(np.median(valid)) * self.depth_scale
        threshold = self.obstacle_stop_dist
        if reference_z is not None:
            threshold = min(threshold, max(0.0, reference_z - 0.15))
        return sensed < threshold

    def _publish_status(self, distance=None):
        # 'mode' matters as much as 'status' when debugging: several
        # controllers report TRACKING, and 'distance' does not mean the same
        # thing in each — in tracking it is metres to the plant, in waypoint
        # mode it is metres left to the saved point. Without the mode there
        # is no way to tell which number you are reading.
        payload = {
            'status': self.status,
            'target_class': self.target_class,
            'mode': self._mode,
            'active': bool(self.active),
            'auto_track_paused': bool(self._auto_track_paused),
        }
        if distance is not None:
            payload['distance'] = round(distance, 2)
        self.status_pub.publish(String(data=json.dumps(payload)))

    def _publish_waypoints(self):
        self.wp_pub.publish(String(data=json.dumps(self.waypoints)))

    def _set_status(self, status, distance=None):
        self.status = status
        self._publish_status(distance)

    def _stop(self, status):
        self.active = False
        self._mode = 'idle'
        self._avoid_state = 'none'
        self._prev_lin = 0.0
        self._prev_ang = 0.0
        self.cmd_pub.publish(Twist())
        self._set_status(status)
        if status == 'REACHED':
            self._auto_tracked = False
            self._auto_track_paused = True
            self._parked_suppress = True
            self.get_logger().info('reached target plant — parked and ready for scan')
            if self.resume_track:
                self._resume_pending = True
                self._resume_since = self.get_clock().now()
                self._post_inspection_gate = True
                if (self._on_track
                        and 0 <= self._wp_index < len(self.waypoints)
                        and self._wp_index + 1 < len(self.waypoints)):
                    self.get_logger().info(
                        f'plant inspected — will resume track at waypoint '
                        f'{self._wp_index + 1} once scan completes')
                else:
                    self.get_logger().info(
                        'plant inspected — will rotate to search for next plant '
                        'once scan completes')
        elif (self.target_class and
              str(self.target_class).strip().lower()
              in self.auto_track_classes):
            self._auto_tracked = False
            self._auto_track_paused = True
            self.get_logger().info('auto-track paused')

    def _save_waypoint(self, det):
        cls = self.target_class
        if not cls or cls == self._saved_class:
            return
        if not self._odom_set:
            self.get_logger().warn('cannot save waypoint: no odom')
            return
        self._saved_class = cls
        # estimate object position in odom frame from detection + robot pose
        dz = float(det.get('z', 0.0))
        dx = float(det.get('x', 0.0))
        obj_odom_x = self._odom_x + dz * math.cos(self._odom_yaw) \
                     - dx * math.sin(self._odom_yaw)
        obj_odom_y = self._odom_y + dz * math.sin(self._odom_yaw) \
                     + dx * math.cos(self._odom_yaw)
        self._update_map_pose()
        wx, wy = self._to_map(obj_odom_x, obj_odom_y)
        frame = self._map_frame if self._map_available else 'odom'
        nearby = []
        for d in self.detections:
            name = d.get('class_name') or d.get('class')
            if name and d.get('z') is not None:
                nearby.append({
                    'class_name': name,
                    'confidence': d.get('confidence', 0),
                    'distance': round(float(d['z']), 2),
                })
        wp = {
            'class_name': cls,
            'frame': frame,
            'x': round(wx, 3),
            'y': round(wy, 3),
            'z': round(dz, 3),
            'time': self.get_clock().now().nanoseconds * 1e-9,
            'nearby': nearby,
        }
        self.waypoints.append(wp)
        self._publish_waypoints()
        self.get_logger().info(
            f'waypoint saved: {cls} ({wp["x"]:.3f},{wp["y"]:.3f}) '
            f'frame={frame} obj_odom=({obj_odom_x:.3f},{obj_odom_y:.3f}) '
            f'dist={dz:.3f}m nearby={len(nearby)} total={len(self.waypoints)}')

    def _get_detection_odom_coords(self, d):
        dz = float(d.get('z', 0.0))
        dx = float(d.get('x', 0.0))
        obj_odom_x = self._odom_x + dz * math.cos(self._odom_yaw) \
                     - dx * math.sin(self._odom_yaw)
        obj_odom_y = self._odom_y + dz * math.sin(self._odom_yaw) \
                     + dx * math.cos(self._odom_yaw)
        return obj_odom_x, obj_odom_y

    def _control_startup_scan(self, now):
        if not self._odom_set:
            self.get_logger().info('Waiting for odometry to start 360-degree scan...')
            return

        if not hasattr(self, '_startup_scan_initialized'):
            self._startup_scan_initialized = True
            self._startup_scan_start_yaw = self._odom_yaw
            self._startup_scan_last_yaw = self._startup_scan_start_yaw
            self._startup_scan_accumulated_yaw = 0.0
            self._startup_scan_started = self.get_clock().now()
            self._scanned_plants = []
            self.get_logger().info(
                f'Starting 360-degree startup scan '
                f'(giving up after {self.startup_scan_timeout_s:.0f}s)...')

        current_yaw = self._odom_yaw

        dyaw = current_yaw - self._startup_scan_last_yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        self._startup_scan_accumulated_yaw += abs(dyaw)
        self._startup_scan_last_yaw = current_yaw

        cmd = Twist()
        cmd.angular.z = self.search_speed
        self.cmd_pub.publish(cmd)
        self._set_status('SEARCHING')

        for d in self.detections:
            cls = str(d.get('class_name') or d.get('class') or '').strip().lower()
            z = d.get('z')
            if z is None:
                continue
            is_target_class = (
                cls in self.auto_track_classes or
                cls in ('potted plant', 'plant', 'pot', 'crop')
            )
            if is_target_class:
                ox, oy = self._get_detection_odom_coords(d)
                duplicate = False
                for p in self._scanned_plants:
                    if math.hypot(ox - p['x'], oy - p['y']) < 0.5:
                        duplicate = True
                        if z < p['z']:
                            p['x'] = ox
                            p['y'] = oy
                            p['z'] = z
                            p['det'] = d
                        break
                if not duplicate:
                    self._scanned_plants.append({
                        'x': ox,
                        'y': oy,
                        'z': z,
                        'class_name': cls,
                        'det': d
                    })

        turned = math.degrees(self._startup_scan_accumulated_yaw)
        elapsed = (now - self._startup_scan_started).nanoseconds * 1e-9
        timed_out = elapsed > self.startup_scan_timeout_s
        if timed_out and self._startup_scan_accumulated_yaw < 2 * math.pi - 0.1:
            # Odometry never reported a full turn. Stop anyway and use what
            # was seen, rather than rotating on the spot for ever with a
            # perfectly good plant in view.
            self.get_logger().warn(
                f'startup scan gave up after {elapsed:.0f}s: odometry only '
                f'measured {turned:.0f} of 360 degrees. Check that the wheel '
                f'encoders count while the robot turns. Using the '
                f'{len(self._scanned_plants)} plant(s) seen so far.')

        if self._startup_scan_accumulated_yaw >= 2 * math.pi - 0.1 or timed_out:
            if not timed_out:
                self.get_logger().info(
                    f'360-degree scan complete. '
                    f'Found {len(self._scanned_plants)} plants.')
            self.cmd_pub.publish(Twist())

            if timed_out:
                # The survey ended because odometry was not counting. Every
                # plant position collected during the spin was worked out
                # from that same frozen pose, so they are all wrong, and
                # waypoint mode steers by odometry too. Fall back to the
                # camera, which does not need odometry at all.
                self._mode = 'searching'
                self._search_ticks = 1
                self._publish_status()
                return

            if self._scanned_plants:
                nearest_plant = None
                min_dist = float('inf')
                for p in self._scanned_plants:
                    dist = math.hypot(p['x'] - self._odom_x, p['y'] - self._odom_y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_plant = p

                if nearest_plant:
                    self.get_logger().info(
                        f'Nearest plant located: {nearest_plant["class_name"]} '
                        f'at ({nearest_plant["x"]:.2f}, {nearest_plant["y"]:.2f}), '
                        f'distance: {min_dist:.2f}m. Navigating to it...')
                    self._wp_target = (nearest_plant['x'], nearest_plant['y'])
                    self._wp_frame = 'odom'
                    self.target_class = nearest_plant['class_name']
                    self._mode = 'waypoint'
                    self._on_track = True
                    self._auto_tracked = True
                    self._avoid_state = 'none'
                    self._search_ticks = 0
                    self._stop_confirm = 0
                    self._publish_status()
                    return

            self.get_logger().warn(
                'No plants found during 360-degree scan. Entering searching '
                'mode.')
            self._mode = 'searching'
            self._search_ticks = 1

    # ── main loop ────────────────────────────────────────────────

    def control_loop(self):
        self._publish_waypoints()
        self._update_map_pose()
        self.suppress_avoidance_pub.publish(
            Bool(data=bool(self.active or self._parked_suppress
                           or self._scanner_scanning
                           or self._scan_hold_pending())))
        # Unconditional hold while the arm is mid-sweep. Every other guard
        # here is a state machine that can be raced into the wrong branch;
        # this one reads the arm's own live status, so no ordering of
        # mission/scanner messages can drive the base while it is moving.
        if self._scanner_scanning or self._scan_hold_pending():
            self.cmd_pub.publish(Twist())
            self._prev_lin = 0.0
            self._prev_ang = 0.0
            self._publish_status()
            return
        if not self.active:
            self._maybe_auto_track()
            if not self.active:
                # Hold position: keep a zero cmd_vel flowing while parked
                # so the downstream safety layer never reads "no command
                # source" and starts its autonomous creep/wander.
                self.cmd_pub.publish(Twist())
                self._handle_resume_pending()
                self._publish_status()
                return
        now = self.get_clock().now()

        if self._avoid_state != 'none':
            self._control_avoid()
            return

        if self._mode == 'tracking':
            self._control_tracking(now)
        elif self._mode == 'searching':
            self._control_searching()
        elif self._mode == 'blind_drive':
            self._control_blind_drive(now)
        elif self._mode == 'waypoint':
            self._control_waypoint()
        elif self._mode == 'startup_scan':
            self._control_startup_scan(now)
        else:
            self._stop('IDLE')

    def _scan_hold_pending(self):
        """True while a scan has been announced but has not finished."""
        if self._scan_hold_until is None:
            return False
        if self.get_clock().now() >= self._scan_hold_until:
            self._scan_hold_until = None
            return False
        return True

    def _handle_resume_pending(self):
        """After a plant was inspected, wait for the arm scan to finish,
        then keep going automatically: resume the waypoint track if one
        is ahead, otherwise rotate to search for the next plant.
        The auto_resume_min_dist gate keeps it from re-picking the plant
        it just inspected."""
        if not self._resume_pending:
            return
        now = self.get_clock().now()
        waited = ((now - self._resume_since).nanoseconds * 1e-9
                  if self._resume_since is not None else 0.0)
        on_track_resume = (self._on_track
                           and 0 <= self._wp_index < len(self.waypoints)
                           and self._wp_index + 1 < len(self.waypoints))
        if (self._scan_done or not self.auto_scan_on_reach
                or not self._scan_requested
                or waited > self.resume_timeout):
            self._resume_pending = False
            if on_track_resume:
                next_idx = self._wp_index + 1
                wp = self.waypoints[next_idx]
                self._wp_index = next_idx
                self._on_track = True
                self._parked_suppress = False
                self._wp_target = (float(wp['x']), float(wp['y']))
                self._wp_frame = wp.get('frame', 'odom')
                self._mode = 'waypoint'
                self.active = True
                self.target_class = wp.get('class_name',
                                           self.target_class)
                self._avoid_state = 'none'
                self._search_ticks = 0
                self._stop_confirm = 0
                self._publish_status()
                self.get_logger().info(
                    f'resuming track — waypoint {next_idx}: '
                    f'{self.target_class} '
                    f'({self._wp_target[0]:.3f},{self._wp_target[1]:.3f})')
            else:
                self._auto_track_paused = False
                self._parked_suppress = False
                self._mode = 'searching'
                self.active = True
                self._search_ticks = 1
                self.get_logger().info(
                    'scan complete — rotating to search for the next plant')

    def _control_searching(self):
        """Rotate in place to search for the next plant regardless of orientation."""
        self._search_ticks += 1
        cmd = Twist()
        cmd.angular.z = self.search_speed
        self.cmd_pub.publish(cmd)
        self._set_status('SEARCHING')

        for d in self.detections:
            cls = str(d.get('class_name') or d.get('class') or '').strip().lower()
            z = d.get('z')
            if z is None:
                continue
            is_target_class = (
                cls in self.auto_track_classes or
                (self.target_class and cls == str(self.target_class).strip().lower()) or
                cls in ('potted plant', 'plant', 'pot', 'crop')
            )
            # The minimum distance exists only to stop the robot re-picking
            # the plant it has just inspected, which it is parked right in
            # front of. Applying it always — as this did — meant a plant in
            # plain view closer than auto_resume_min_dist never stopped the
            # rotation, and the robot searched for ever. Gate it the same way
            # _maybe_auto_track does: only just after an inspection.
            if is_target_class and (not self._post_inspection_gate
                                    or z > self.auto_resume_min_dist):
                self.get_logger().info(
                    f'found next plant ({cls}) at {z:.2f}m during rotation '
                    f'search — locking on!')
                self._search_ticks = 0
                self._select_target(cls, d, auto=True)
                return

        max_ticks = int(self.search_timeout * 10)
        if self._search_ticks > max_ticks:
            self._search_ticks = 1

    # ── AVOID state machine ──────────────────────────────────────

    def _control_avoid(self):
        px, py, _ = self._current_pose()
        if self._avoid_state == 'turn':
            elapsed = (self.get_clock().now() -
                       self._avoid_start).nanoseconds * 1e-9
            cmd = Twist()
            cmd.angular.z = self._avoid_dir * 0.35
            if elapsed >= 1.8:  # roughly 60° at 0.35 rad/s
                self._avoid_state = 'drive'
                self._avoid_start = self.get_clock().now()
            self.cmd_pub.publish(cmd)
            self._set_status('BLOCKED')
            return

        if self._avoid_state == 'drive':
            dist = math.hypot(px - self._avoid_base_x,
                              py - self._avoid_base_y)
            if dist >= self.avoid_distance or self._obstacle_blocking():
                self._avoid_state = 'none'
                self._avoid_dir *= -1.0
                self.get_logger().info('avoidance complete, resuming')
                return
            cmd = Twist()
            cmd.angular.z = 0.0
            cmd.linear.x = 0.2
            self.cmd_pub.publish(cmd)
            self._set_status('BLOCKED')
            return

    # ── blind drive (detection lost but close, just go straight) ──

    def _control_blind_drive(self, now):
        elapsed = (now - self._blind_start).nanoseconds * 1e-9
        est_traveled = elapsed * 0.15
        est_dist = self._blind_last_z - est_traveled

        # If TOF sensor range is active, use hardware TOF to measure exact
        # close-range distance to target. Do NOT let a spurious near TOF
        # reading (a leaf edge) trigger the stop while still misaligned —
        # TOF only refines distance, never makes the stop point closer
        # than stop_distance.
        if self.latest_tof:
            valid_tof = [v for v in self.latest_tof if v is not None and 0.05 < v < 1.5]
            if valid_tof:
                tof_min = min(valid_tof)
                est_dist = max(self.stop_distance, tof_min) \
                    if tof_min < self.stop_distance else tof_min

        # If the plant comes back into view while driving blind, hand back
        # to the tracking controller so the robot finishes aligned and
        # centered — not stopped off to the side at the blind estimate.
        for d in self.detections:
            name = d.get('class_name') or d.get('class')
            if name == self.target_class and d.get('z') is not None:
                self._switch_to_tracking(
                    d, f'blind re-acquired {self.target_class}, aligning')
                return

        # Compute dynamic heading error based on robot's rotation since entering blind drive
        _, _, current_yaw = self._current_pose()
        delta_yaw = current_yaw - self._blind_start_yaw
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
        # _blind_heading is a camera-frame bearing (atan2(x, z): positive
        # means the plant is to the RIGHT), while delta_yaw is odom yaw
        # (positive means the robot turned LEFT). Turning right to face a
        # right-hand plant makes delta_yaw negative, so the remaining
        # bearing is heading + delta_yaw. Subtracting instead made the
        # error grow every tick: the robot spun ~180 degrees past the
        # plant and timed out rather than arriving.
        current_heading_err = self._blind_heading + delta_yaw
        current_heading_err = math.atan2(math.sin(current_heading_err), math.cos(current_heading_err))
        heading_err = abs(current_heading_err)

        # Only stop once close AND the heading toward the plant's last
        # known bearing has been corrected (otherwise we park off to the
        # side at a 15°+ residual, as seen in live runs).
        if est_dist <= self.stop_distance and \
                heading_err <= self.stop_align_threshold:
            self.get_logger().info(
                f'blind reached {self.target_class} est={est_dist:.2f}m '
                f'heading={math.degrees(heading_err):.1f}°')
            self._mode = 'idle'
            reached_via_auto_track = self._auto_tracked
            self._stop('REACHED')
            if reached_via_auto_track and self.auto_scan_on_reach:
                self._trigger_arm_scan()
            return
        if est_dist <= self.stop_distance:
            # Close but not aligned yet: rotate in place to zero the
            # heading error instead of stopping crooked.
            dt = self._ctrl_dt()
            ang_cmd = self._rate_limit(
                self._prev_ang, -self.k_ang * current_heading_err,
                dt, self.omega_accel)
            self._prev_ang = ang_cmd
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = max(-self.max_angular,
                                min(self.max_angular, ang_cmd))
            self.cmd_pub.publish(cmd)
            self._set_status('TRACKING', distance=round(est_dist, 2))
            return
        if elapsed > 8.0:
            self.get_logger().info(
                f'blind timeout {self.target_class}')
            self._mode = 'idle'
            self._stop('LOST')
            return

        # if obstacle right ahead, slow down but keep going
        target = 0.15 if not self._obstacle_blocking() else 0.05
        dt = self._ctrl_dt()
        lin_cmd = self._rate_limit(
            self._prev_lin, target, dt,
            self.accel_limit, self.decel_limit)
        self._prev_lin = lin_cmd
        # keep steering toward the plant's last known bearing so we don't
        # drift further off to the side while the plant is out of view
        ang_cmd = self._rate_limit(
            self._prev_ang, -self.k_ang * current_heading_err,
            dt, self.omega_accel)
        self._prev_ang = ang_cmd
        cmd = Twist()
        cmd.linear.x = lin_cmd
        cmd.angular.z = max(-self.max_angular,
                            min(self.max_angular, ang_cmd))
        self.cmd_pub.publish(cmd)
        self._set_status('TRACKING', distance=round(est_dist, 2))

    def _smooth(self, prev, new, gain):
        if prev is None:
            return new
        return gain * new + (1.0 - gain) * prev

    def _switch_to_tracking(self, d, reason, on_track=None):
        """Begin a fresh tracking approach on the given detection, so the
        robot turns to face the plant and drives straight at it.
        on_track=True marks that we diverted from the waypoint route, so
        the robot resumes the track after the plant is inspected."""
        self._mode = 'tracking'
        if on_track is not None:
            self._on_track = on_track
            if on_track:
                # Diverted from the waypoint route to inspect a plant:
                # treat it like an auto-tracked plant so the arm scan
                # fires on reach and avoidance stays suppressed.
                self._auto_tracked = True
        self.target_pos = (float(d.get('x', 0.0)),
                           float(d.get('y', 0.0)),
                           float(d.get('z', 0.0)))
        self.last_seen = self.get_clock().now()
        self._saved_class = None
        self._fz = None
        self._fang = None
        self._ffill = None
        self._stop_confirm = 0
        self._prev_lin = 0.0
        self._prev_ang = 0.0
        self.get_logger().info(
            f'{reason} — approaching {self.target_class}')

    def _center_err(self, bbox, bearing):
        """Bearing toward the plant's 3D CENTER (rad). Returns bearing directly
        so the main camera faces the center of the plant centrally."""
        return bearing

    def _rate_limit(self, prev, target, dt, accel, decel=None):
        decel = accel if decel is None else decel
        limit = (accel if target >= prev else decel) * dt
        if abs(target - prev) <= limit:
            return target
        return prev + math.copysign(limit, target - prev)

    def _ctrl_dt(self):
        now = self.get_clock().now()
        if self._last_ctrl is None:
            self._last_ctrl = now
            return 0.1
        dt = (now - self._last_ctrl).nanoseconds * 1e-9
        self._last_ctrl = now
        return min(0.5, max(0.02, dt))

    def _is_target_match(self, cls_name):
        if not cls_name:
            return False
        c = str(cls_name).strip().lower()
        t = str(self.target_class).strip().lower() if self.target_class else ''
        if c == t:
            return True
        plant_aliases = {'potted plant', 'plant', 'pot', 'crop'}
        if (t in plant_aliases or t in self.auto_track_classes) and (c in plant_aliases or c in self.auto_track_classes):
            return True
        return False

    # ── tracking (camera-frame chase) ────────────────────────────

    def _control_tracking(self, now):
        best = None
        best_dist = float('inf')
        tx, tz = self.target_pos[0], self.target_pos[2]
        for d in self.detections:
            cls = d.get('class_name') or d.get('class')
            if not self._is_target_match(cls):
                continue
            if d.get('z') is None:
                continue
            dx = float(d.get('x', 0.0)) - tx
            dz = float(d.get('z', 0.0)) - tz
            d2 = dx * dx + dz * dz
            if d2 < best_dist:
                best_dist = d2
                best = d

        if best is None:
            if self.last_seen is not None:
                elapsed = (now - self.last_seen).nanoseconds * 1e-9
                if elapsed > self.lost_timeout:
                    if self._last_tracked_det is not None:
                        last_z = float(
                            self._last_tracked_det.get('z', 99.0))
                        if last_z < self.blind_approach_limit:
                            self._mode = 'blind_drive'
                            self._blind_start = now
                            self._blind_last_z = last_z
                            last_x = float(
                                self._last_tracked_det.get('x', 0.0))
                            self._blind_heading = math.atan2(last_x, last_z)
                            _, _, self._blind_start_yaw = self._current_pose()
                            self._save_waypoint(
                                self._last_tracked_det)
                            self.get_logger().info(
                                f'LOST close: {self.target_class} @'
                                f'{last_z:.2f}m — driving blind...')
                            return
                        # far: auto-waypoint navigation
                        self._save_waypoint(self._last_tracked_det)
                        if self.waypoints:
                            wp = self.waypoints[-1]
                            self._wp_target = (
                                float(wp['x']), float(wp['y']))
                            self._wp_frame = wp.get('frame',
                                                     'odom')
                            self._mode = 'waypoint'
                            self._wp_index = len(self.waypoints) - 1
                            self._on_track = False
                            self._avoid_state = 'none'
                            self._search_ticks = 0
                            self._saved_class = None
                            self.get_logger().info(
                                f'LOST far: {self.target_class}'
                                f' — waypoint navigation')
                            return
                    self._stop('LOST')
            return

        self.last_seen = now
        self._last_tracked_det = best
        x = float(best['x'])
        z = self._smooth(self._fz, float(best['z']), self.z_filter)

        # Refine close-range z measurement with hardware TOF sensors if valid
        if z < 1.0 and self.latest_tof:
            valid_tof = [v for v in self.latest_tof if v is not None and 0.05 < v < 1.5]
            if valid_tof:
                tof_min = min(valid_tof)
                z = 0.6 * tof_min + 0.4 * z

        self._fz = z
        raw_angle = math.atan2(x, z)
        angle = self._smooth(self._fang, raw_angle, self.angle_filter)
        self._fang = angle
        center_err = self._center_err(best.get('bbox'), angle)

        # Requirement: stop at a distance of exactly 0.4 meters between depth camera and plant
        z_target = self.stop_distance
        dist_err = z - z_target

        dt = self._ctrl_dt()

        # Proportional angular turn with braking speed limit to prevent overshoot
        target_ang = -self.k_ang * center_err
        brake_ang = math.sqrt(2.0 * self.omega_accel * max(0.0, abs(center_err)))
        target_ang = math.copysign(min(abs(target_ang), brake_ang), target_ang)
        ang_cmd = self._rate_limit(self._prev_ang, target_ang, dt, self.omega_accel)
        self._prev_ang = ang_cmd

        self._tick_log_count = getattr(self, '_tick_log_count', 0) + 1
        if self._tick_log_count % 30 == 0:
            self.get_logger().info(
                f'TRACKING {self.target_class}: dist={z:.2f}m '
                f'angle={math.degrees(angle):.0f}° '
                f'center={math.degrees(center_err):.0f}° '
                f'v_lin={min(self.max_linear, max(0.0, self.k_lin*dist_err)):.2f}')

        cmd = Twist()
        cmd.angular.z = max(-self.max_angular,
                            min(self.max_angular, ang_cmd))

        if dist_err <= 0.0:
            # Stop forward motion and ensure main camera faces center centrally with zero overshoot
            if abs(center_err) > self.stop_align_threshold:
                self._stop_confirm = 0
                cmd.linear.x = 0.0
                self._prev_lin = 0.0
                self.cmd_pub.publish(cmd)
                self._set_status('TRACKING', distance=z)
                return

            self._stop_confirm += 1
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self._prev_lin = 0.0
            self._prev_ang = 0.0
            if self._stop_confirm >= self.stop_confirm_ticks:
                self.get_logger().info(
                    f'reached target plant {self.target_class} at exactly {z:.2f}m facing center centrally (err={math.degrees(center_err):.2f}°)')
                if not self._on_track:
                    self._save_waypoint(best)
                self._stop('REACHED')
                if self.auto_scan_on_reach:
                    self._trigger_arm_scan()
            else:
                self.cmd_pub.publish(cmd)
                self._set_status('TRACKING', distance=z)
            return

        self._stop_confirm = 0

        # Linear speed control with smooth deceleration and alignment prioritization
        lin_target = min(self.max_linear, self.k_lin * dist_err)
        brake_v = math.sqrt(2.0 * self.decel_limit * max(0.0, dist_err))
        lin_target = min(lin_target, brake_v)
        if dist_err < 0.04:
            lin_target = max(0.0, self.k_lin * dist_err)
        else:
            lin_target = max(self.min_creep, lin_target)

        # Slow linear speed when approaching close (z < 0.8m) if heading error is present so heading aligns before 0.4m
        if z < 0.8 and abs(center_err) > 0.05:
            lin_target *= max(0.0, 1.0 - abs(center_err) / 0.2)

        lin_cmd = self._rate_limit(self._prev_lin, lin_target, dt,
                                   self.accel_limit, self.decel_limit)
        self._prev_lin = lin_cmd

        if abs(center_err) < self.align_threshold:
            if not self._is_target_match(self.target_class) and self._obstacle_blocking(reference_z=z):
                self._set_status('BLOCKED', distance=z)
                self._prev_lin = 0.0
                self._prev_ang = 0.0
                self._start_avoid()
                return
            cmd.linear.x = lin_cmd
        self.cmd_pub.publish(cmd)
        self._set_status('TRACKING', distance=z)

    # ── waypoint (odom/map frame) ────────────────────────────────

    def _control_waypoint(self):
        if not self._odom_set:
            return
        if self._wp_frame == 'odom':
            px, py, pyaw = self._odom_x, self._odom_y, self._odom_yaw
        else:
            px, py, pyaw = self._current_pose()
        wx, wy = self._wp_target
        if self._wp_frame != 'odom' and not self._map_available:
            self.get_logger().warn(
                'waypoint frame is map but map not available — '
                'waiting for map frame')
            return
        dx = wx - px
        dy = wy - py
        dist_err = math.hypot(dx, dy) - self.wp_stop_distance

        _search = getattr(self, '_search_ticks', 0)
        if dist_err <= 0.0 and _search == 0:
            # brake first — hold for a couple ticks so we coast to a
            # stop before the in-place search spin-up instead of slamming
            # from cruise into it.
            self._stop_confirm += 1
            self._prev_lin = 0.0
            if self._stop_confirm >= self.stop_confirm_ticks:
                self._stop_confirm = 0
                self.get_logger().info(
                    f'arrived near waypoint, searching for '
                    f'{self.target_class}...')
                self._search_ticks = 1
            else:
                self.cmd_pub.publish(Twist())
                self._set_status('TRACKING',
                                 distance=round(dist_err, 2))
            return

        if _search > 0:
            self._search_ticks = _search + 1
            cmd = Twist()
            cmd.angular.z = self.search_speed
            self.cmd_pub.publish(cmd)
            self._set_status('SEARCHING', distance=round(dist_err, 2))

            for d in self.detections:
                name = d.get('class_name') or d.get('class')
                if name == self.target_class and d.get('z') is not None:
                    self._search_ticks = 0
                    self._switch_to_tracking(
                        d, f'found {self.target_class} during search',
                        on_track=True)
                    return

            max_ticks = int(self.search_timeout * 10)
            if self._search_ticks > max_ticks:
                self._search_ticks = 0
                self.get_logger().info(
                    f'search complete — stopped at waypoint '
                    f'{self.target_class}')
                self._stop('REACHED')
            return
        self._search_ticks = 0
        self._stop_confirm = 0

        # If the plant is visible while driving along the path, turn to
        # face it right now instead of walking past it — the robot always
        # ends up facing the plant no matter where it is.
        for d in self.detections:
            name = d.get('class_name') or d.get('class')
            if name == self.target_class and d.get('z') is not None:
                self._switch_to_tracking(
                    d, f'{self.target_class} in view while driving',
                    on_track=True)
                return

        target_heading = math.atan2(dy, dx)
        angle_err = target_heading - pyaw
        angle_err = math.atan2(math.sin(angle_err), math.cos(angle_err))

        dt = self._ctrl_dt()

        ang_cmd = self._rate_limit(
            self._prev_ang, self.k_ang * angle_err, dt, self.omega_accel)
        self._prev_ang = ang_cmd

        self._tick_log_count = getattr(self, '_tick_log_count', 0) + 1
        if self._tick_log_count % 30 == 0:
            self.get_logger().info(
                f'WAYPOINT {self.target_class}: dist={dist_err:.2f}m '
                f'angle_err={math.degrees(angle_err):.0f}°')

        cmd = Twist()
        cmd.angular.z = max(-self.max_angular,
                            min(self.max_angular, ang_cmd))
        if abs(angle_err) < self.align_threshold:
            # Same reasoning as _control_tracking — an auto-tracked
            # plant's own foliage shouldn't trigger avoidance against
            # itself.
            if not self._auto_tracked and self._obstacle_blocking(
                    reference_z=dist_err + self.wp_stop_distance):
                self._set_status('BLOCKED',
                                 distance=round(dist_err, 2))
                self._prev_lin = 0.0
                self._prev_ang = 0.0
                self._start_avoid()
                return
            lin_target = max(self.min_creep,
                             min(self.max_linear,
                                 self.k_lin * dist_err))
            brake_v = math.sqrt(2.0 * self.decel_limit
                                * max(0.0, dist_err))
            lin_target = min(lin_target, brake_v)
            lin_cmd = self._rate_limit(self._prev_lin, lin_target, dt,
                                       self.accel_limit, self.decel_limit)
            self._prev_lin = lin_cmd
            cmd.linear.x = lin_cmd
        self.cmd_pub.publish(cmd)
        self._set_status('TRACKING', distance=round(dist_err, 2))

    def _start_avoid(self):
        self._avoid_state = 'turn'
        self._avoid_start = self.get_clock().now()
        self._avoid_base_x, self._avoid_base_y, _ = self._current_pose()
        self.get_logger().info(
            f'avoiding obstacle: {self._avoid_state}')


def main(args=None):
    rclpy.init(args=args)
    node = DetectionGoto()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
