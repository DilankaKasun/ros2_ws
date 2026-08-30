"""Plant survey-and-scan run: the one node that owns the wheels.

Two drivers share the robot and hand over cleanly between them:

  * the MAP driver (Nav2) does the long crossing of the room, and stops
    1.2m short of the plant — well outside the safety ring the obstacle
    layer draws around it;
  * the CAMERA driver does the last stretch, closing in with the plant
    centred and parking 0.65m out, square on.

Exactly one of them is publishing at any moment. The handover is
explicit: the map driver's goal is cancelled and this node waits for its
velocity messages to actually stop arriving before the camera driver
starts, because obstacle_avoidance.py picks its source by which topic is
still live.

Nothing here trusts the wheel counters for anything precise. This robot
is a skid-steer on tracks: the tracks scrub when it turns, so a reported
turn is about half the real one. That has a sharp consequence, and the
whole world model is shaped by it: a sighting taken part-way through a
sweep projects into the wheel frame at the wrong angle, so the SAME
plant lands somewhere different in every frame it appears in. Positions
collected while turning are worthless, and one plant becomes several.

So the survey does not build a map. It remembers each plant as the
heading the robot was reporting when it saw it, plus how far off it was.
A reported heading can be returned to even when it is badly wrong,
because the error is a consistent scaling: turning back until the wheels
report that heading again really does face the same way. Only then, with
the plant dead ahead and the robot no longer turning, is a position
worked out — and it is used immediately, to aim the one drive that
follows. Nothing keeps a position across a turn.

Every state has a deadline, every state change goes through the tick
(never straight from one handler into the next), and a state that runs
out of time says so rather than reporting an arrival it did not make.

The arm scan, the photographs, the health report and the dashboard
messages stay with plant_mission_node; this node asks it to scan where
the robot already is, and holds the wheels dead still until the ARM
itself reports it has finished.
"""
import json
import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

# Stamped on every command this node publishes onto the shared command
# topic, so its own subscription can tell them apart from the dashboard's
# and never answer itself.
_SELF_TAG = 'plant_run_node'


def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class Candidate:
    """A plant the survey saw, remembered as a direction rather than a place.

    ``heading`` is what the wheels were reporting when the plant was seen.
    It is not a true compass direction — the wheels under-count turns — but
    turning back until they report it again does face the same way, which
    is all it is ever used for. ``rng`` is how far off the plant was at
    that moment, used to decide which one to go to first.
    """

    __slots__ = ('heading', 'rng', 'name', 'seen', 'sightings',
                 'state', 'reason', 'x', 'y')

    def __init__(self, heading, rng, name):
        self.heading = heading
        self.rng = rng
        self.name = name
        self.seen = time.time()
        self.sightings = 1
        self.state = 'pending'      # pending | done | failed | duplicate
        self.reason = ''
        # Only ever filled in from a sighting taken while facing the plant
        # and not turning. Used to recognise a plant already finished, and
        # for nothing else.
        self.x = None
        self.y = None

    def merge(self, heading, rng):
        w = 1.0 / (self.sightings + 1)
        self.heading = _wrap(self.heading + _wrap(heading - self.heading) * w)
        self.rng += (rng - self.rng) * w
        self.sightings += 1
        self.seen = time.time()

    def as_dict(self):
        d = {'heading_deg': round(math.degrees(self.heading), 1),
             'range_m': round(self.rng, 2), 'name': self.name,
             'state': self.state, 'sightings': self.sightings}
        if self.x is not None:
            d['x'] = round(self.x, 2)
            d['y'] = round(self.y, 2)
        if self.reason:
            d['reason'] = self.reason
        return d


class PlantRunNode(Node):

    def __init__(self):
        super().__init__('plant_run_node')

        # -- what counts as a plant ------------------------------------
        self.declare_parameter('detections_topic', '/ecobot/detections')
        self.declare_parameter('plant_classes', ['potted plant', 'plant',
                                                 'vase', 'pot'])
        self.declare_parameter('min_confidence', 0.4)
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        # Fallback only, for a detection carrying no 3D point: the depth
        # camera sees a 57 degree wedge straight ahead and nothing else.
        self.declare_parameter('camera_hfov_deg', 57.0)
        self.declare_parameter('image_width', 640)

        # -- distances -------------------------------------------------
        self.declare_parameter('handover_distance_m', 1.2)
        self.declare_parameter('park_distance_m', 0.65)
        # The depth camera returns nothing usable closer than this, and a
        # plant it cannot measure is a plant it thinks it has lost.
        self.declare_parameter('min_depth_range_m', 0.5)
        self.declare_parameter('park_tolerance_m', 0.07)
        self.declare_parameter('park_bearing_tol_rad', 0.09)
        # Two sightings this close in direction and distance are the
        # same plant seen twice, not two plants.
        self.declare_parameter('merge_heading_rad', 0.30)
        self.declare_parameter('merge_range_m', 0.8)
        # A freshly sighted plant this close to one already finished is
        # that same plant again, not a new one to go and do.
        self.declare_parameter('revisit_radius_m', 0.9)
        self.declare_parameter('max_plant_range_m', 6.0)
        # How far off the remembered heading the plant may be when the
        # robot turns back to look for it again.
        self.declare_parameter('reacquire_window_rad', 0.45)
        self.declare_parameter('heading_tolerance_rad', 0.12)

        # -- speeds ----------------------------------------------------
        self.declare_parameter('survey_turn_speed', 0.5)
        # Exploring: creep forward and let the obstacle layer do the
        # swerving. It already reverses and turns away from walls, so the
        # run node does not steer here — it just keeps going and stops to
        # look round now and then.
        self.declare_parameter('explore_speed', 0.15)
        self.declare_parameter('approach_max_linear', 0.16)
        self.declare_parameter('approach_max_angular', 0.5)
        self.declare_parameter('approach_min_linear', 0.04)
        self.declare_parameter('back_off_speed', 0.06)
        self.declare_parameter('turn_away_speed', 0.5)
        self.declare_parameter('k_linear', 0.5)
        self.declare_parameter('k_angular', 1.2)

        # -- a deadline for every state --------------------------------
        self.declare_parameter('survey_timeout_s', 25.0)
        self.declare_parameter('drive_timeout_s', 90.0)
        self.declare_parameter('handover_timeout_s', 6.0)
        self.declare_parameter('approach_timeout_s', 60.0)
        self.declare_parameter('scan_timeout_s', 240.0)
        self.declare_parameter('report_timeout_s', 120.0)
        self.declare_parameter('turn_away_timeout_s', 8.0)
        self.declare_parameter('pick_timeout_s', 5.0)
        self.declare_parameter('reacquire_timeout_s', 35.0)
        # How long one push into the room lasts before stopping to look
        # round again, and how many of those a run may make without
        # finding anything. Reaching a plant resets the count.
        self.declare_parameter('explore_leg_s', 12.0)
        self.declare_parameter('max_explore_legs', 6)
        # How long the arm has to actually start moving after being asked.
        self.declare_parameter('scan_start_timeout_s', 25.0)
        self.declare_parameter('nav_server_wait_s', 5.0)
        # How long the camera driver keeps going on the last bearing it
        # had for a plant it can no longer see, before giving up.
        self.declare_parameter('lost_sight_timeout_s', 3.0)
        # A detection older than this is stale and is not steered on.
        self.declare_parameter('detection_max_age_s', 1.0)
        # The map driver's messages must have been absent this long before
        # the camera driver may take the wheels. Must exceed the 1.0s
        # source timeout in obstacle_avoidance.py, or both look live at
        # once and the safety layer picks the wrong one.
        self.declare_parameter('handover_quiet_s', 1.3)

        # -- odometry trim ---------------------------------------------
        # The scrubbing error is taken out at source, by
        # motor_control_node's turn_calibration. This is only a residual
        # trim on top of that, and should normally stay at 1.0. It is used
        # ONLY to judge roughly how far round the survey has looked; the
        # survey's real bound is its deadline, so a wrong value here slows
        # the robot down but can never hang it.
        self.declare_parameter('yaw_scale', 1.0)
        self.declare_parameter('survey_sweep_rad', 2.0 * math.pi)
        # Stop turning the moment a plant has been seen this many frames
        # running. The survey is not a mapping exercise — it is a look
        # round for something to go and do, and standing still spinning
        # past a plant that is plainly in view is the opposite of that.
        # Set act_on_sight false to always finish the full sweep first.
        self.declare_parameter('act_on_sight', True)
        self.declare_parameter('min_sightings_to_act', 3)
        # Enough of a turn to be facing somewhere new before moving on.
        self.declare_parameter('turn_away_rad', 1.2)

        # -- who this node talks to ------------------------------------
        self.declare_parameter('cmd_vel_topic', '/goto_cmd_vel')
        self.declare_parameter('nav_cmd_vel_topic', '/nav_cmd_vel')
        self.declare_parameter('nav_action_name', '/navigate_to_pose')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('suppress_topic',
                               '/ecobot/goto_suppress_avoidance')
        self.declare_parameter('run_cmd_topic', '/ecobot/plant_scan_cmd')
        self.declare_parameter('run_status_topic', '/ecobot/nav_status')
        self.declare_parameter('mission_status_topic',
                               '/ecobot/plant_scan_status')
        self.declare_parameter('scanner_status_topic', '/arm/scanner_status')
        self.declare_parameter('scan_samples', 6)
        # Where the arm's turning axis sits relative to base_footprint, so
        # a plant measured by the camera can be handed to the arm in the
        # frame the arm works in (x forward, y left, z up from the floor).
        self.declare_parameter('arm_base_offset_x', 0.0)
        self.declare_parameter('arm_base_offset_y', 0.0)
        # The arm refuses a target further than this from its own axis.
        self.declare_parameter('arm_max_reach_m', 0.85)
        # Stand still while no run is going. obstacle_avoidance.py drives
        # the robot forward by itself when no driver is publishing, so
        # silence here is not the same thing as stillness.
        self.declare_parameter('hold_still_when_idle', True)
        # How many times in a row the robot may look round again without
        # a plant getting scanned. Reaching one resets the count, because
        # a survey that leads to a scan was not a wasted one.
        self.declare_parameter('max_surveys_without_progress', 2)

        gp = self.get_parameter
        self._plant_classes = {str(c).strip().lower()
                               for c in gp('plant_classes').value}
        self._min_conf = float(gp('min_confidence').value)
        self._fixed_frame = str(gp('fixed_frame').value)
        self._base_frame = str(gp('base_frame').value)
        self._hfov = math.radians(float(gp('camera_hfov_deg').value))
        self._image_width = int(gp('image_width').value)

        self._handover_dist = float(gp('handover_distance_m').value)
        self._park_dist = float(gp('park_distance_m').value)
        self._min_depth = float(gp('min_depth_range_m').value)
        if self._park_dist < self._min_depth + 0.1:
            # Parking inside the camera's blind zone means the plant
            # cannot be measured, and a plant it cannot measure is one the
            # robot thinks it has lost. Refuse the setting rather than
            # drive to a spot the run can never succeed from.
            fixed = self._min_depth + 0.15
            self.get_logger().warning(
                f'park_distance_m of {self._park_dist}m is inside the '
                f'depth camera\'s blind zone (nothing reads closer than '
                f'{self._min_depth}m) — using {fixed}m instead')
            self._park_dist = fixed
        self._park_tol = float(gp('park_tolerance_m').value)
        self._park_bearing_tol = float(gp('park_bearing_tol_rad').value)
        self._merge_heading = float(gp('merge_heading_rad').value)
        self._merge_range = float(gp('merge_range_m').value)
        self._revisit_radius = float(gp('revisit_radius_m').value)
        self._max_range = float(gp('max_plant_range_m').value)
        self._reacquire_window = float(gp('reacquire_window_rad').value)
        self._heading_tol = float(gp('heading_tolerance_rad').value)

        self._survey_speed = float(gp('survey_turn_speed').value)
        self._explore_speed = float(gp('explore_speed').value)
        self._max_lin = float(gp('approach_max_linear').value)
        self._max_ang = float(gp('approach_max_angular').value)
        self._min_lin = float(gp('approach_min_linear').value)
        self._back_off = abs(float(gp('back_off_speed').value))
        self._turn_away_speed = float(gp('turn_away_speed').value)
        self._k_lin = float(gp('k_linear').value)
        self._k_ang = float(gp('k_angular').value)

        self._timeouts = {
            'SURVEY': float(gp('survey_timeout_s').value),
            'PICK': float(gp('pick_timeout_s').value),
            'REACQUIRE': float(gp('reacquire_timeout_s').value),
            'EXPLORE': float(gp('explore_leg_s').value),
            'DRIVE': float(gp('drive_timeout_s').value),
            'HANDOVER': float(gp('handover_timeout_s').value),
            'APPROACH': float(gp('approach_timeout_s').value),
            'SCAN': float(gp('scan_timeout_s').value),
            'REPORT': float(gp('report_timeout_s').value),
            'TURN_AWAY': float(gp('turn_away_timeout_s').value),
        }
        self._scan_start_timeout = float(gp('scan_start_timeout_s').value)
        self._nav_server_wait = float(gp('nav_server_wait_s').value)
        self._lost_timeout = float(gp('lost_sight_timeout_s').value)
        self._det_max_age = float(gp('detection_max_age_s').value)
        self._handover_quiet = float(gp('handover_quiet_s').value)

        self._yaw_scale = float(gp('yaw_scale').value)
        self._sweep_target = float(gp('survey_sweep_rad').value)
        self._act_on_sight = bool(gp('act_on_sight').value)
        self._min_sightings_to_act = int(gp('min_sightings_to_act').value)
        self._turn_away_rad = float(gp('turn_away_rad').value)
        self._scan_samples = int(gp('scan_samples').value)
        self._arm_off_x = float(gp('arm_base_offset_x').value)
        self._arm_off_y = float(gp('arm_base_offset_y').value)
        self._arm_max_reach = float(gp('arm_max_reach_m').value)
        self._hold_still_when_idle = bool(gp('hold_still_when_idle').value)
        self._max_surveys = int(gp('max_surveys_without_progress').value)
        self._max_explore_legs = int(gp('max_explore_legs').value)

        # -- run state --------------------------------------------------
        self._state = 'IDLE'
        self._state_since = time.time()
        self._driver = 'none'
        self._say = 'waiting for a run to start'
        self._plants = []             # Candidate objects
        self._target = None
        self._target_xy = None        # goal position, worked out fresh
        self._run_active = False
        self._surveys_done = 0        # since a plant was last scanned
        self._explore_legs = 0        # since a plant was last scanned
        self._pending = None          # a state change queued for the tick

        self._sweep_turned = 0.0
        self._sweep_last_yaw = None

        self._nav_goal_handle = None
        self._nav_result = None       # 'succeeded' | 'failed' | 'canceled'
        self._nav_cancel_sent = False
        self._nav_missing = False
        self._nav_goal_sent = False
        self._last_nav_vel_time = 0.0
        self._last_nav_move_time = 0.0

        self._last_seen_time = 0.0
        self._last_bearing = 0.0
        self._last_range = None
        # The whole detection behind the last good look, kept so the arm
        # can be told how tall and wide the plant actually is.
        self._last_det = None

        self._scan_seen_running = False
        self._scan_finished = False
        self._scan_reason = ''
        self._mission_status = ''

        self._turn_away_turned = 0.0
        self._turn_away_last_yaw = None

        self._odom_yaw = None
        self._detections = []
        self._detections_time = 0.0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._cmd_pub = self.create_publisher(
            Twist, str(gp('cmd_vel_topic').value), 10)
        self._suppress_pub = self.create_publisher(
            Bool, str(gp('suppress_topic').value), 10)
        self._status_pub = self.create_publisher(
            String, str(gp('run_status_topic').value), 10)
        self._scan_cmd_pub = self.create_publisher(
            String, str(gp('run_cmd_topic').value), 10)

        self._nav_client = ActionClient(
            self, NavigateToPose, str(gp('nav_action_name').value))

        self.create_subscription(
            String, str(gp('detections_topic').value), self._on_detections, 10)
        self.create_subscription(
            Odometry, str(gp('odom_topic').value), self._on_odom, 20)
        self.create_subscription(
            Twist, str(gp('nav_cmd_vel_topic').value), self._on_nav_vel, 10)
        self.create_subscription(
            String, str(gp('run_cmd_topic').value), self._on_cmd, 10)
        self.create_subscription(
            String, str(gp('scanner_status_topic').value),
            self._on_scanner_status, 10)
        self.create_subscription(
            String, str(gp('mission_status_topic').value),
            self._on_mission_status, 10)

        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            'plant run node ready — the map driver hands over at '
            f'{self._handover_dist}m and the camera driver parks at '
            f'{self._park_dist}m')

    # ================= incoming ==========================================

    def _on_cmd(self, msg):
        """Run-level commands only. The scan-level ones (scan_here, pause,
        resume, set_samples) belong to plant_mission_node and are ignored
        here, so one message is never acted on twice."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if data.get('from') == _SELF_TAG:
            return                      # our own message coming back
        action = data.get('action')
        if data.get('samples') is not None:
            try:
                self._scan_samples = max(1, min(int(data['samples']), 40))
            except (TypeError, ValueError):
                pass
        if action == 'start':
            self._start_run()
        elif action == 'stop':
            self._stop_run('stopped by the operator')
        elif action == 'next':
            self._queue_skip('skipped by the operator')

    def _send_scan_cmd(self, payload):
        payload = dict(payload)
        payload['from'] = _SELF_TAG
        self._scan_cmd_pub.publish(String(data=json.dumps(payload)))

    def _on_odom(self, msg):
        self._odom_yaw = _yaw_from_quat(msg.pose.pose.orientation)

    def _on_nav_vel(self, msg):
        # Only the ARRIVAL of a message matters, not what is in it: the
        # safety layer picks its source by which topic is still live, so
        # a map driver publishing nothing but zeros would still be the one
        # holding the wheels.
        self._last_nav_vel_time = time.time()
        if abs(msg.linear.x) > 1e-3 or abs(msg.angular.z) > 1e-3:
            self._last_nav_move_time = self._last_nav_vel_time

    def _on_detections(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(data, list):
            return
        self._detections = [d for d in data if self._is_plant(d)]
        self._detections_time = time.time()

    def _is_plant(self, det):
        if not isinstance(det, dict):
            return False
        try:
            if float(det.get('confidence') or 0.0) < self._min_conf:
                return False
        except (TypeError, ValueError):
            return False
        name = str(det.get('class_name', '')).strip().lower()
        return name in self._plant_classes

    def _on_scanner_status(self, msg):
        """The arm's own word on whether it has finished. Nothing else
        counts: a scan is over when the arm says it is idle, having first
        said it was scanning."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        status = data.get('status')
        if status in ('scanning', 'recovering'):
            self._scan_seen_running = True
        elif status == 'failed':
            self._scan_finished = True
            self._scan_reason = str(data.get('reason') or 'the arm reported a failure')
        elif status == 'idle' and self._scan_seen_running:
            self._scan_finished = True
            self._scan_reason = ''

    def _on_mission_status(self, msg):
        try:
            self._mission_status = str(json.loads(msg.data).get('status', ''))
        except Exception:
            pass

    # ================= geometry ==========================================

    def _bearing_and_range(self, det):
        """Where a detection sits relative to the robot: the angle left of
        straight ahead, and how far off it is. (None, None) when the frame
        gave no usable depth — the camera cannot measure closer than about
        half a metre, and reads nothing at all through glossy leaves."""
        rng = det.get('z')
        if rng is None:
            rng = det.get('distance')
        try:
            rng = float(rng)
        except (TypeError, ValueError):
            return None, None
        if rng <= 0.0:
            return None, None

        x = det.get('x')
        if x is not None:
            # Camera optical frame: x to the right, z straight ahead.
            try:
                bearing = math.atan2(-float(x), rng)
            except (TypeError, ValueError):
                return None, None
        else:
            bbox = det.get('bbox') or []
            if len(bbox) != 4:
                return None, None
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            frac = (cx / max(1.0, float(self._image_width))) - 0.5
            bearing = -frac * self._hfov
        return bearing, rng

    def _robot_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                self._fixed_frame, self._base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.0))
        except TransformException:
            return None
        return (tf.transform.translation.x, tf.transform.translation.y,
                _yaw_from_quat(tf.transform.rotation))

    def _to_fixed_frame(self, bearing, rng):
        """Turn an angle and a distance into a point in the fixed frame.
        Approximate by design — good enough to plan a drive that ends 1.2m
        out, and the camera driver corrects the rest."""
        pose = self._robot_pose()
        if pose is None:
            return None
        rx, ry, ryaw = pose
        angle = ryaw + bearing
        return rx + rng * math.cos(angle), ry + rng * math.sin(angle)

    def _record_sighting(self, det):
        """Remember a plant as a direction to come back to, and hand back
        the candidate it belongs to.

        No position is stored: the robot is turning while it surveys, and
        a position taken mid-turn on this robot is fiction."""
        bearing, rng = self._bearing_and_range(det)
        if bearing is None or rng > self._max_range:
            return None
        pose = self._robot_pose()
        if pose is None:
            return None
        heading = _wrap(pose[2] + bearing)
        for plant in self._plants:
            if abs(_wrap(plant.heading - heading)) <= self._merge_heading and \
                    abs(plant.rng - rng) <= self._merge_range:
                if plant.state == 'pending':
                    plant.merge(heading, rng)
                return plant
        self._plants.append(
            Candidate(heading, rng, str(det.get('class_name', 'plant'))))
        self.get_logger().info(
            f'saw a {self._plants[-1].name} {rng:.2f}m away, '
            f'{math.degrees(heading):.0f} degrees round — '
            f'{len(self._plants)} plant(s) found so far')
        return self._plants[-1]

    def _already_handled_near(self, x, y):
        """True when this spot is a plant the robot has already worked on,
        including the one it is still parked in front of. Only positions
        taken while facing a plant and standing still are compared, so
        this stays meaningful despite the wheel counters."""
        for plant in self._plants:
            if plant.state != 'done' or plant.x is None:
                continue
            if math.hypot(plant.x - x, plant.y - y) <= self._revisit_radius:
                return True
        return False

    def _pick_next_plant(self):
        """The plant needing the least turning to look at, out of those not
        already done. Least turn, not nearest: the remembered distances go
        stale as soon as the robot drives anywhere, but a heading can
        always be turned back to."""
        pose = self._robot_pose()
        if pose is None:
            return None
        best, best_turn = None, None
        for plant in self._plants:
            if plant.state != 'pending':
                continue
            turn = abs(_wrap(plant.heading - pose[2]))
            if best_turn is None or turn < best_turn:
                best, best_turn = plant, turn
        return best

    # ================= driving ===========================================

    def _drive(self, linear=0.0, angular=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self._cmd_pub.publish(cmd)

    def _map_driver_live(self):
        return (time.time() - self._last_nav_vel_time) < self._handover_quiet

    # ================= state machine =====================================
    #
    # Handlers never call one another. They queue the next state and the
    # tick performs it, so there is exactly one state change per tick and
    # no path where a chain of failures recurses through every plant.

    _DRIVERS = {
        'SURVEY': 'camera', 'REACQUIRE': 'camera', 'APPROACH': 'camera',
        'TURN_AWAY': 'camera', 'EXPLORE': 'camera',
        'DRIVE': 'map', 'HANDOVER': 'handover',
    }

    def _enter(self, state, say):
        self._state = state
        self._state_since = time.time()
        self._driver = self._DRIVERS.get(state, 'none')
        self._say = say
        self.get_logger().info(f'[{self._driver} driver] {state}: {say}')
        self._publish_status()

    def _queue(self, state, say):
        """Ask the tick to change state. First request in a tick wins, so
        a handler cannot be overridden by something it triggered."""
        if self._pending is None:
            self._pending = (state, say)

    def _elapsed(self):
        return time.time() - self._state_since

    def _deadline_passed(self):
        limit = self._timeouts.get(self._state)
        return limit is not None and self._elapsed() > limit

    # ---- run-level -------------------------------------------------------

    def _start_run(self):
        if self._run_active:
            self.get_logger().warning(
                f'ignoring start: a run is already going ({self._state})')
            return
        self._run_active = True
        self._plants = []
        self._target = None
        self._surveys_done = 0
        self._explore_legs = 0
        self._nav_missing = False
        self._queue('SURVEY', 'starting the run — looking around for plants')

    def _stop_run(self, why):
        self._cancel_nav_goal()
        self._send_scan_cmd({'action': 'stop'})
        self._run_active = False
        self._target = None
        self._drive(0.0, 0.0)
        self._pending = None
        self._enter('IDLE', why)

    def _queue_skip(self, why):
        if not self._run_active:
            return
        self._mark_target('failed', why)
        self._queue('TURN_AWAY', why)

    def _mark_target(self, state, reason=''):
        if self._target is not None:
            self._target.state = state
            self._target.reason = reason
            if state == 'done':
                # Where this plant stood, from the close-up look taken
                # while parked in front of it. The only position worth
                # keeping, and it is kept for one purpose: recognising
                # this plant if it is seen again.
                if self._target_xy is not None:
                    self._target.x, self._target.y = self._target_xy
                # A look round or a push into the room that led to a scan
                # was not a wasted one.
                self._surveys_done = 0
                self._explore_legs = 0
            elif state == 'failed':
                self.get_logger().warning(
                    f'giving up on the {self._target.name}: {reason}')
        self._target = None
        self._target_xy = None

    def _fail_target(self, why):
        """Give up on this plant honestly and go and find another."""
        self._cancel_nav_goal()
        self._mark_target('failed', why)
        self._queue('PICK', f'moving on — {why}')

    def _finish_run(self, why):
        self._run_active = False
        self._target = None
        self._drive(0.0, 0.0)
        self._queue('DONE', why)

    # ---- entering each state --------------------------------------------

    def _on_enter(self, state, say):
        if state == 'REACQUIRE':
            self._last_seen_time = 0.0
        elif state == 'EXPLORE':
            self._explore_legs += 1
            # A push into the room puts the robot somewhere new, so the
            # look-rounds that found nothing back there do not count
            # against the look-round from here.
            self._surveys_done = 0
        elif state == 'SURVEY':
            self._surveys_done += 1
            self._sweep_turned = 0.0
            self._sweep_last_yaw = self._odom_yaw
        elif state == 'DRIVE':
            self._nav_result = None
            self._nav_goal_handle = None
            self._nav_cancel_sent = False
            # The goal is sent from the tick, not from here. Waiting on the
            # action server blocks this node, and a blocked node publishes
            # no velocity — after one second of that, the safety layer
            # decides nobody is driving and starts moving the robot itself.
            self._nav_goal_sent = False
        elif state == 'HANDOVER':
            self._cancel_nav_goal()
            self._last_seen_time = 0.0
            self._last_range = None
        elif state == 'APPROACH':
            self._last_seen_time = 0.0
        elif state == 'SCAN':
            self._scan_seen_running = False
            self._scan_finished = False
            self._scan_reason = ''
            self._drive(0.0, 0.0)
            self._enter(state, say)
            # plant_mission_node owns the arm, the photographs and the
            # report. It is told to scan where the robot already is: this
            # node did the positioning, and nothing else may touch the
            # wheels until the arm says it has finished.
            payload = {'action': 'scan_here', 'samples': self._scan_samples}
            payload.update(self._plant_geometry())
            self._send_scan_cmd(payload)
            return
        elif state == 'TURN_AWAY':
            self._turn_away_turned = 0.0
            self._turn_away_last_yaw = self._odom_yaw
        self._enter(state, say)

    # ---- picking the next plant -----------------------------------------

    def _tick_pick(self):
        self._drive(0.0, 0.0)
        pose = self._robot_pose()
        if pose is None:
            self._say = 'waiting for the wheel frame before picking a plant'
            if self._deadline_passed():
                self._finish_run(
                    'no reading from the wheels — cannot pick a plant')
            return
        plant = self._pick_next_plant()
        if plant is not None:
            self._target = plant
            self._target_xy = None
            turn = math.degrees(_wrap(plant.heading - pose[2]))
            self._queue('REACQUIRE',
                        f'picked the {plant.name} that was {plant.rng:.2f}m '
                        f'away — turning {turn:+.0f} degrees to look at it '
                        'again')
            return
        if self._surveys_done < self._max_surveys:
            self._queue('SURVEY',
                        'no plant left that I know of — looking round again')
            return
        done = sum(1 for p in self._plants if p.state == 'done')
        failed = sum(1 for p in self._plants if p.state == 'failed')
        dup = sum(1 for p in self._plants if p.state == 'duplicate')
        say = f'run complete — {done} plant(s) scanned'
        if failed:
            say += f', {failed} could not be reached'
        if dup:
            say += f', {dup} turned out to be one already done'
        self._finish_run(say)

    # ---- looking at the chosen plant again -------------------------------

    def _tick_reacquire(self):
        """Turn back to the heading the plant was seen at, and get a fresh
        look before any driving happens.

        This is where the one usable position comes from. Facing the plant
        and not turning, the wheel counters' error does not enter the sum,
        so the point worked out here is good enough to aim a drive at. It
        is used at once and never stored across another turn.
        """
        pose = self._robot_pose()
        if pose is None or self._target is None:
            self._drive(0.0, 0.0)
            self._say = 'waiting for a reading from the wheels'
            if self._deadline_passed():
                self._fail_target('no reading from the wheels to turn by')
            return
        want = self._target.heading
        turn_err = _wrap(want - pose[2])

        bearing, rng = self._best_detection()
        facing = abs(turn_err) <= self._reacquire_window

        if bearing is not None and facing and rng is not None:
            pos = self._to_fixed_frame(bearing, rng)
            if pos is not None and self._already_handled_near(*pos):
                # This is a plant already scanned, seen from a new angle.
                self._drive(0.0, 0.0)
                self._mark_target(
                    'duplicate',
                    'already scanned — this is the same plant from a new '
                    'angle')
                self._queue('PICK',
                            'that is a plant already done, seen again — '
                            'not doing it twice')
                return
            if pos is not None and abs(bearing) <= self._reacquire_window:
                self._drive(0.0, 0.0)
                self._target_xy = pos
                self._target.rng = rng
                found = (f'found it again {rng:.2f}m off and '
                         f'{math.degrees(bearing):+.0f} degrees from '
                         'straight ahead')
                if rng <= self._handover_dist + self._park_tol:
                    # Already inside the handover ring. There is no room
                    # left to cross, so asking the map driver for a goal
                    # here would just be a goal on the spot the robot is
                    # standing on. The camera takes it straight away.
                    self._queue('APPROACH',
                                f'{found} — near enough already, no long '
                                'drive needed')
                else:
                    self._queue('DRIVE', found)
                return

        if abs(turn_err) > self._heading_tol:
            self._drive(0.0, _clamp(self._k_ang * turn_err,
                                    self._max_ang))
            self._say = (f'turning back to where the plant was — '
                         f'{math.degrees(turn_err):+.0f} degrees to go')
            return

        # On the remembered heading with nothing in view. The robot has
        # driven since, so the heading has gone stale. Keep turning the
        # same way and look right round, rather than flicking a few
        # degrees either side of a heading that is already known to be
        # wrong — that only ever re-checked the spot the plant is not in.
        self._drive(0.0, self._survey_speed)
        self._say = 'the plant is not where it was — looking round for it'

        if self._deadline_passed():
            self._drive(0.0, 0.0)
            self._fail_target(
                'could not find that plant again from here')

    def _watch_for_plants(self):
        """Record everything plant-like in view and hand back the candidate
        that is best established, or None. Used by both the look-round and
        the push into the room — either can act the moment a plant is
        plainly there."""
        best = None
        if (time.time() - self._detections_time) < self._det_max_age:
            for det in self._detections:
                hit = self._record_sighting(det)
                if hit is not None and hit.state == 'pending' and (
                        best is None or hit.sightings > best.sightings):
                    best = hit
        if best is not None and best.sightings >= self._min_sightings_to_act:
            return best
        return None

    # ---- exploring -------------------------------------------------------

    def _tick_explore(self):
        """Push into the room looking for a plant the look-round could not
        see from where it stood.

        The obstacle layer does the steering. It already slows for what is
        ahead, swerves toward the freer side and reverses out of a dead
        end, so this just keeps asking to go forward and lets that happen
        underneath. Avoidance stays fully on — this is the one state where
        the robot is deliberately driving at whatever is in front of it
        without knowing what it is.
        """
        sighted = self._watch_for_plants()
        if sighted is not None:
            self._drive(0.0, 0.0)
            self._queue('PICK',
                        f'a {sighted.name} came into view {sighted.rng:.2f}m '
                        'off while exploring')
            return

        if self._deadline_passed():
            self._drive(0.0, 0.0)
            self._queue('SURVEY', 'far enough — stopping to look round')
            return

        self._drive(self._explore_speed, 0.0)
        self._say = (f'exploring — pushing into the room, leg '
                     f'{self._explore_legs}/{self._max_explore_legs}, '
                     f'{self._elapsed():.0f}s of '
                     f'{self._timeouts["EXPLORE"]:.0f}s')

    # ---- the survey ------------------------------------------------------

    def _tick_survey(self):
        # Measure the turn from odometry, corrected for the tracks'
        # scrubbing. It is a hint about where the robot has looked, not a
        # promise: the deadline below is the real bound, so this never
        # waits for a full circle to be confirmed before acting.
        if self._odom_yaw is not None:
            if self._sweep_last_yaw is not None:
                self._sweep_turned += abs(
                    _wrap(self._odom_yaw - self._sweep_last_yaw)
                ) * self._yaw_scale
            self._sweep_last_yaw = self._odom_yaw

        seen_now = self._watch_for_plants()

        # A plant is in view and has held still enough frames to be real.
        # Stop turning and go to it — the aiming is REACQUIRE's job, and
        # it is better at it than a sweep that is already moving past.
        sighted = self._act_on_sight and seen_now is not None
        swept = self._sweep_turned >= self._sweep_target
        timed_out = self._deadline_passed()
        if not (sighted or swept or timed_out):
            self._drive(0.0, self._survey_speed)
            self._say = (
                f'turning to look — {len(self._plants)} plant(s) so far, '
                f'about {math.degrees(self._sweep_turned):.0f} degrees round')
            return

        self._drive(0.0, 0.0)
        if sighted:
            why = (f'a {seen_now.name} in view {seen_now.rng:.2f}m off, '
                   f'held for {seen_now.sightings} frames')
        elif swept:
            why = 'a full look round'
        else:
            why = 'the look-round time limit'
        pending = sum(1 for p in self._plants if p.state == 'pending')
        if pending:
            self._queue('PICK', f'{why} reached — '
                                f'{len(self._plants)} plant(s) known')
            return

        # Nothing left to go to from here. Before giving up, go and look
        # somewhere else: standing still and looking round again from the
        # same spot can only ever see the same nothing.
        if self._explore_legs < self._max_explore_legs:
            self._queue('EXPLORE',
                        f'{why} — no plant in sight from here, moving on '
                        'to look elsewhere')
            return
        self._finish_run(
            f'no plants found after {why} and '
            f'{self._explore_legs} push(es) into the room — nothing to do')

    # ---- the long drive, map driver -------------------------------------

    def _standoff_pose(self):
        """A pose on the line from the robot to the plant, stopping short
        of it by the handover distance and facing it."""
        pose = self._robot_pose()
        if pose is None or self._target_xy is None:
            return None
        rx, ry, ryaw = pose
        dx = self._target_xy[0] - rx
        dy = self._target_xy[1] - ry
        r = math.hypot(dx, dy)
        if r < 1e-3:
            ux, uy = math.cos(ryaw), math.sin(ryaw)
        else:
            ux, uy = dx / r, dy / r
        # Never ask for a goal behind us: if the robot is already inside
        # the handover ring, aim at where it stands.
        back = min(self._handover_dist, max(0.0, r))
        return (self._target_xy[0] - back * ux,
                self._target_xy[1] - back * uy,
                math.atan2(uy, ux))

    def _send_nav_goal(self):
        """Try to hand the map driver a goal. Never blocks: returns False
        if the server is not up yet, and the tick tries again."""
        if not self._nav_client.server_is_ready():
            return False

        target = self._standoff_pose()
        if target is None:
            self.get_logger().error(
                'no transform from the wheel frame — cannot plan a drive')
            self._nav_result = 'failed'
            return True
        gx, gy, gyaw = target

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self._fixed_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.orientation.z = math.sin(gyaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(gyaw / 2.0)

        self._nav_client.send_goal_async(goal).add_done_callback(
            self._on_nav_accepted)
        self.get_logger().info(
            f'map driver heading for ({gx:.2f}, {gy:.2f}) — '
            f'{self._handover_dist}m short of the plant, never at it')
        return True

    def _on_nav_accepted(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().error(f'the map driver refused the goal: {e}')
            self._nav_result = 'failed'
            return
        if not handle.accepted:
            self.get_logger().error('the map driver rejected the goal')
            self._nav_result = 'failed'
            return
        self._nav_goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        try:
            status = future.result().status
        except Exception as e:
            self.get_logger().error(f'the map driver failed: {e}')
            self._nav_result = 'failed'
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._nav_result = 'succeeded'
        elif status == GoalStatus.STATUS_CANCELED:
            self._nav_result = 'canceled'
        else:
            self._nav_result = 'failed'

    def _cancel_nav_goal(self):
        if self._nav_goal_handle is not None and not self._nav_cancel_sent:
            self._nav_cancel_sent = True
            self._nav_goal_handle.cancel_goal_async()
        self._nav_goal_handle = None

    def _distance_to_target(self):
        pose = self._robot_pose()
        if pose is None or self._target_xy is None:
            return None
        return math.hypot(self._target_xy[0] - pose[0],
                          self._target_xy[1] - pose[1])

    def _tick_drive(self):
        # The map driver owns the wheels. A zero is still published here so
        # that if Nav2 falls silent the safety layer stops the robot rather
        # than driving it forward on its own.
        self._drive(0.0, 0.0)

        if not self._nav_goal_sent:
            if self._send_nav_goal():
                self._nav_goal_sent = True
            elif self._elapsed() < self._nav_server_wait:
                self._say = 'waiting for the map driver (Nav2) to answer'
                return
            else:
                self.get_logger().error(
                    'the map driver (Nav2) is not running — it has to be up '
                    'to cross the room')
                self._nav_missing = True

        if self._nav_missing:
            self._finish_run(
                'the map driver (Nav2) is not running — start it with '
                'enable_navigation:=true, the run cannot cross the room '
                'without it')
            return

        dist = self._distance_to_target()
        if dist is not None:
            self._say = (f'crossing the room — about {dist:.2f}m from the '
                         f'plant, letting go at {self._handover_dist}m')
            # The stored position is rough, so arrival is judged by the
            # distance as well as by Nav2 saying it got there.
            if dist <= self._handover_dist + self._park_tol:
                self._queue('HANDOVER',
                            f'{dist:.2f}m from the plant — that is the '
                            'handover ring')
                return

        if self._nav_result == 'succeeded':
            self._queue('HANDOVER', 'the map driver reached its stopping point')
            return
        if self._nav_result in ('failed', 'canceled'):
            self._fail_target(f'the map driver {self._nav_result}')
            return
        if self._deadline_passed():
            self._fail_target(
                f'the map driver ran out of time after '
                f'{self._timeouts["DRIVE"]:.0f}s')

    # ---- the handover ----------------------------------------------------

    def _tick_handover(self):
        self._drive(0.0, 0.0)
        if not self._map_driver_live():
            self._queue('APPROACH',
                        'the map driver has let go — the camera has the '
                        'wheels now')
            return
        still_moving = (time.time() - self._last_nav_move_time) < \
            self._handover_quiet
        self._say = ('waiting for the map driver to fall silent — it is '
                     + ('still commanding movement' if still_moving
                        else 'idling but still holding the topic'))
        if self._deadline_passed():
            # Nav2 is still talking. Carrying on would mean two drivers at
            # once, which is the one thing this design will not do.
            self._fail_target('the map driver would not let go of the wheels')

    # ---- the last stretch, camera driver ---------------------------------

    def _best_detection(self):
        """Out of what the camera can see right now, the one being driven
        at: the closest in direction to the one tracked last time, or the
        most central if nothing is being tracked yet.

        Deliberately frame-free. Matching by a worked-out position would
        put the wheel counters' error back into the one part of the run
        that does not need them."""
        if (time.time() - self._detections_time) > self._det_max_age:
            return None, None
        reference = (self._last_bearing
                     if self._last_seen_time > 0.0 else 0.0)
        best = None
        for det in self._detections:
            bearing, rng = self._bearing_and_range(det)
            if bearing is None:
                continue
            score = abs(_wrap(bearing - reference))
            if best is None or score < best[0]:
                best = (score, bearing, rng, det)
        if best is None:
            return None, None
        self._last_det = best[3]
        return best[1], best[2]

    def _tick_approach(self):
        now = time.time()
        bearing, rng = self._best_detection()
        if bearing is not None:
            self._last_seen_time = now
            self._last_bearing = bearing
            self._last_range = rng
            self._update_target_position(bearing, rng)
        else:
            bearing = self._last_bearing
            rng = self._last_range

        if self._last_seen_time == 0.0:
            # Nothing seen since taking over. Turn towards where the plant
            # should be rather than driving at a guess.
            self._turn_toward_stored_target()
            if self._elapsed() > 2.0 * self._lost_timeout:
                self._fail_target(
                    'the camera never picked the plant up after the handover')
            return

        lost_for = now - self._last_seen_time
        if lost_for > self._lost_timeout:
            self._drive(0.0, 0.0)
            self._fail_target(f'lost sight of the plant for {lost_for:.1f}s')
            return

        if rng is None:
            self._drive(0.0, 0.0)
            self._say = 'the plant is in view but not measurable — holding'
            if self._deadline_passed():
                self._fail_target('never got a usable distance to the plant')
            return

        error = rng - self._park_dist
        centred = abs(bearing) <= self._park_bearing_tol

        if error < -self._park_tol:
            # Too close — including anything inside the camera's blind
            # zone, where a plant cannot be measured at all. Ease back
            # rather than press on blind.
            self._drive(-self._back_off, 0.0)
            self._say = (f'{rng:.2f}m is closer than the {self._park_dist}m '
                         'parking distance — easing back')
        elif error <= self._park_tol and centred:
            self._drive(0.0, 0.0)
            self._queue('SCAN',
                        f'parked {rng:.2f}m out and square on to the plant')
            return
        elif abs(bearing) > 3.0 * self._park_bearing_tol:
            # Well off centre: turn on the spot. The camera only sees a 57
            # degree wedge, so swinging while rolling loses the plant.
            self._drive(0.0, _clamp(self._k_ang * bearing, self._max_ang))
            self._say = (f'turning to centre the plant — '
                         f'{math.degrees(bearing):+.0f} degrees off')
        else:
            # Close in, slowing as the gap shrinks, still correcting the
            # last of the heading so it arrives square on.
            linear = _clamp(self._k_lin * max(0.0, error), self._max_lin)
            linear = max(linear, self._min_lin)
            self._drive(linear,
                        _clamp(self._k_ang * bearing, self._max_ang * 0.5))
            self._say = (f'closing in — {rng:.2f}m out, '
                         f'{math.degrees(bearing):+.0f} degrees off centre, '
                         f'parking at {self._park_dist}m')

        if self._deadline_passed():
            self._drive(0.0, 0.0)
            self._fail_target(
                f'could not park within {self._timeouts["APPROACH"]:.0f}s')

    def _update_target_position(self, bearing, rng):
        """Keep the working position fresh while closing in. On this last
        stretch the robot is driving nearly straight at the plant, so the
        point is as good as it ever gets — and it is what marks where this
        plant stood, so the robot can tell later that it has done it."""
        if rng is None:
            return
        pos = self._to_fixed_frame(bearing, rng)
        if pos is not None:
            self._target_xy = pos

    def _turn_toward_stored_target(self):
        pose = self._robot_pose()
        if pose is None or self._target_xy is None:
            self._drive(0.0, self._max_ang * 0.6)
            self._say = 'looking for the plant'
            return
        rx, ry, ryaw = pose
        err = _wrap(math.atan2(self._target_xy[1] - ry,
                               self._target_xy[0] - rx) - ryaw)
        self._drive(0.0, _clamp(self._k_ang * err, self._max_ang * 0.6))
        self._say = 'turning to where the plant should be'

    # ---- the scan --------------------------------------------------------

    def _plant_geometry(self):
        """Where the plant is and how big it is, in the frame the arm works
        in: x forward from the arm's axis, y left, z up from the floor.

        Without this the arm was told a fixed guess — 0.30m ahead, 0.50m
        up — while the robot was actually parked 0.65m away. It aimed a
        third of a metre short of the plant, which barely shows at the
        bottom of the sweep and sends the camera clean over the top of the
        plant into the ceiling at the top of it.

        Returns nothing at all when there is no measurement to give, so
        the arm falls back to its own defaults rather than being handed a
        number this node made up.
        """
        det = self._last_det
        rng = self._last_range
        if det is None or rng is None:
            self.get_logger().warning(
                'no measurement of the plant to give the arm — it will use '
                'its own defaults, and may not be aimed well')
            return {}

        bearing = self._last_bearing
        x = rng * math.cos(bearing) - self._arm_off_x
        y = rng * math.sin(bearing) - self._arm_off_y
        reach = math.hypot(x, y)
        if reach > self._arm_max_reach:
            # The arm refuses anything past its reach, and would simply
            # not scan. Pull the aim point in along the same bearing: the
            # plant is a body, not a point, so its near side is still a
            # fair thing to look at.
            scale = self._arm_max_reach / reach
            x *= scale
            y *= scale
            self.get_logger().warning(
                f'plant is {reach:.2f}m from the arm, past its '
                f'{self._arm_max_reach}m reach — aiming at its near side')

        geo = {'x': round(x, 3), 'y': round(y, 3)}

        centre = det.get('center_height')
        if centre is None:
            return geo
        geo['z'] = round(float(centre), 3)
        for src, dst in (('height', 'plant_height'), ('width', 'plant_width'),
                         ('top_height', 'z_top'),
                         ('bottom_height', 'z_bottom')):
            val = det.get(src)
            if val is not None:
                geo[dst] = round(float(val), 3)
        geo['plant_type'] = str(det.get('class_name', 'potted plant'))
        self.get_logger().info(
            f'telling the arm the plant is {geo["x"]:.2f}m ahead, '
            f'{geo["z"]:.2f}m up, '
            f'{geo.get("plant_height", 0.0):.2f}m tall')
        return geo

    def _tick_scan(self):
        # Dead still. A zero every tick, not silence: silence lets the
        # safety layer drive the robot forward by itself.
        self._drive(0.0, 0.0)
        if self._scan_finished:
            if self._scan_reason:
                self._fail_target(f'the arm scan failed: {self._scan_reason}')
                return
            self._queue('REPORT',
                        'the arm says it has finished — waiting for the '
                        'health report')
            return
        if self._scan_seen_running:
            self._say = 'holding still while the arm photographs the plant'
        elif self._elapsed() > self._scan_start_timeout:
            self._fail_target(
                f'the arm never started scanning within '
                f'{self._scan_start_timeout:.0f}s')
            return
        else:
            self._say = 'holding still, waiting for the arm to start'
        if self._deadline_passed():
            # Say what actually happened. A scan that ran out of time is
            # not a scan that succeeded.
            self._fail_target(
                'the arm never reported finishing within '
                f'{self._timeouts["SCAN"]:.0f}s')

    def _tick_report(self):
        self._drive(0.0, 0.0)
        self._say = 'waiting for the plant health report'
        if self._mission_status in ('COMPLETE', 'WAITING', 'IDLE', 'STOPPED'):
            self._mark_target('done')
            self._queue('TURN_AWAY', 'the report is in — turning away')
            return
        if self._mission_status == 'ERROR':
            self._fail_target('the report came back as an error')
            return
        if self._deadline_passed():
            # The plant was still scanned; only the write-up is missing, so
            # this counts as done with the reason recorded.
            self._mark_target('done', 'the report did not arrive in time')
            self._queue('TURN_AWAY',
                        'no report within the time limit — moving on anyway')

    # ---- turning away ----------------------------------------------------

    def _tick_turn_away(self):
        if self._odom_yaw is not None:
            if self._turn_away_last_yaw is not None:
                self._turn_away_turned += abs(
                    _wrap(self._odom_yaw - self._turn_away_last_yaw)
                ) * self._yaw_scale
            self._turn_away_last_yaw = self._odom_yaw

        if self._turn_away_turned >= self._turn_away_rad or \
                self._deadline_passed():
            self._drive(0.0, 0.0)
            self._queue('PICK', 'turned away — picking the next plant')
            return
        self._drive(0.0, self._turn_away_speed)
        self._say = 'turning away from the plant just scanned'

    # ================= tick ==============================================

    def _tick(self):
        # The safety layer stands down only for the last stretch and the
        # scan, where the thing filling the view is the goal rather than a
        # hazard. Everywhere else it stays fully on.
        self._suppress_pub.publish(Bool(
            data=self._state in ('APPROACH', 'SCAN', 'REPORT')))

        handler = {
            'SURVEY': self._tick_survey,
            'EXPLORE': self._tick_explore,
            'PICK': self._tick_pick,
            'REACQUIRE': self._tick_reacquire,
            'DRIVE': self._tick_drive,
            'HANDOVER': self._tick_handover,
            'APPROACH': self._tick_approach,
            'SCAN': self._tick_scan,
            'REPORT': self._tick_report,
            'TURN_AWAY': self._tick_turn_away,
        }.get(self._state)

        if handler is not None:
            handler()
        elif self._hold_still_when_idle:
            # IDLE or DONE: no driver has the wheels, and the robot stands
            # still rather than being nudged along by the safety layer.
            self._drive(0.0, 0.0)

        if self._pending is not None:
            state, say = self._pending
            self._pending = None
            self._on_enter(state, say)

        self._publish_status()

    def _publish_status(self):
        payload = {
            'state': self._state,
            'driver': self._driver,
            'saying': self._say,
            'running': self._run_active,
            'elapsed_s': round(self._elapsed(), 1),
            'deadline_s': self._timeouts.get(self._state),
            'plants': [p.as_dict() for p in self._plants],
            'done': sum(1 for p in self._plants if p.state == 'done'),
            'target': (self._target.as_dict() if self._target else None),
        }
        self._status_pub.publish(String(data=json.dumps(payload)))


def main(args=None):
    rclpy.init(args=args)
    node = PlantRunNode()
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
