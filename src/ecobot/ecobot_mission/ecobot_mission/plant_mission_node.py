"""Per-plant scan, photograph and health report.

This node does NOT drive. ecobot_navigation's plant_run_node owns the
wheels for the whole run — the survey, the long drive, the handover and
the last stretch — and asks this node to scan wherever it has parked the
robot. Two nodes both steering was what made the old runs impossible to
follow, so the wheel authority lives in exactly one place now.

What stays here: triggering arm_scanner_node's multi-viewpoint sweep,
grabbing a wrist-camera JPEG at each viewpoint, sending them to Gemini
for a health assessment, and speaking the dashboard's
/ecobot/plant_scan_cmd -> /ecobot/plant_scan_status / /ecobot/scan_capture
contract that the ecobot-ui dashboard consumes over rosbridge.

Run-level commands on /ecobot/plant_scan_cmd (start, stop, next) are for
plant_run_node; this node acts only on the scan-level ones (scan_here,
stop, pause, resume, set_samples). arm_scanner_node is still only ever
talked to over /arm/scanner_cmd + /arm/scanner_status.
"""
import json
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .gemini_client import GeminiClient

# Statuses a scan is actively running in — used to reject a fresh
# scan_here that would otherwise clobber one already in progress.
_ACTIVE_STATUSES = {'SCANNING', 'ANALYZING'}


class PlantMissionNode(Node):
    def __init__(self):
        super().__init__('plant_mission_node')

        self.declare_parameter('plant_scan_cmd_topic', '/ecobot/plant_scan_cmd')
        self.declare_parameter('plant_scan_status_topic', '/ecobot/plant_scan_status')
        self.declare_parameter('scan_capture_topic', '/ecobot/scan_capture')
        self.declare_parameter('scanner_cmd_topic', '/arm/scanner_cmd')
        self.declare_parameter('scanner_status_topic', '/arm/scanner_status')
        self.declare_parameter('wrist_camera_topic', '/arm/camera/image_raw')

        # Where to aim when scanning in place with no detected plant pose.
        # These sit inside the arm's workspace: the shoulder pivot is 38cm up
        # with a 35cm reach, so the old 0.18 target was unreachable and every
        # sampled viewpoint got filtered out, leaving a scan that never ran.
        self.declare_parameter('arm_scan_x', 0.30)
        self.declare_parameter('arm_scan_y', 0.0)
        self.declare_parameter('arm_scan_z', 0.50)
        # Photos per plant. The dashboard can change this between runs by
        # putting a 'samples' field on any /ecobot/plant_scan_cmd message.
        self.declare_parameter('scan_samples', 6)

        self.declare_parameter('capture_delay_s', 1.4)
        # Once the arm says it has stopped, how long to let the camera
        # settle before the shutter. Only exposure and the last of the
        # wobble — the arm is already still, so this is short.
        self.declare_parameter('settle_delay_s', 0.4)
        self.declare_parameter('frame_max_age_s', 2.0)
        # A scan is over when the ARM says so. These two only catch an arm
        # that has stopped talking altogether. The old single 20s budget
        # from the start of the scan abandoned plants mid-sweep and filed
        # them as finished, which is the one thing a run must never do.
        self.declare_parameter('scan_silence_timeout_s', 30.0)
        self.declare_parameter('scan_hard_timeout_s', 300.0)

        self.declare_parameter('gemini_model', '')
        self.declare_parameter('gemini_timeout_s', 20.0)
        self.declare_parameter('gemini_max_retries', 1)
        self.declare_parameter('jpeg_quality', 85)

        gp = self.get_parameter
        self._arm_scan_x = float(gp('arm_scan_x').value)
        self._arm_scan_y = float(gp('arm_scan_y').value)
        self._arm_scan_z = float(gp('arm_scan_z').value)
        self._scan_samples = int(gp('scan_samples').value)
        self._paused_from = None
        self._capture_delay_s = float(gp('capture_delay_s').value)
        self._settle_delay_s = float(gp('settle_delay_s').value)
        self._frame_max_age_s = float(gp('frame_max_age_s').value)
        self._scan_silence_timeout_s = float(gp('scan_silence_timeout_s').value)
        self._scan_hard_timeout_s = float(gp('scan_hard_timeout_s').value)
        self._jpeg_quality = int(gp('jpeg_quality').value)

        # -- scan state --
        # How many plants have been scanned this session. This node does
        # not know how many there are altogether — plant_run_node does,
        # and says so on /ecobot/nav_status — so the dashboard gets an
        # honest running tally rather than an invented total.
        self._idx = -1
        self._status = 'IDLE'
        self._error_msg = ''
        self._results = []
        self._current_result = None
        self._current_captures = []
        self._epoch = 0

        self._last_scanner_status = None
        self._last_scanner_msg_time = None
        self._last_geometry = {}
        # Whether this arm reports settling at all, and which viewpoint has
        # already been photographed, so one settle is one photograph.
        self._saw_settled = False
        self._captured_viewpoint = None
        self._scan_start_time = None
        self._capture_due_time = None
        self._pending_capture_label = None

        self._result_lock = threading.Lock()
        self._pending_gemini_result = None

        self._wrist_lock = threading.Lock()
        self._wrist_frame = None
        self._wrist_frame_stamp = 0.0
        self._bridge = CvBridge()

        try:
            self._gemini = GeminiClient(
                model=(gp('gemini_model').value or None),
                timeout_s=float(gp('gemini_timeout_s').value),
                max_retries=int(gp('gemini_max_retries').value))
        except RuntimeError as e:
            self.get_logger().warning(
                f'Gemini disabled: {e} — missions will still navigate/scan/'
                'capture, but plant-health results will be degraded')
            self._gemini = None

        self._scanner_cmd_pub = self.create_publisher(
            String, str(gp('scanner_cmd_topic').value), 10)
        self._status_pub = self.create_publisher(
            String, str(gp('plant_scan_status_topic').value), 10)

        self._scan_capture_pub = self.create_publisher(
            String, str(gp('scan_capture_topic').value), 10)

        self.create_subscription(
            String, str(gp('plant_scan_cmd_topic').value), self._on_cmd, 10)
        self.create_subscription(
            String, str(gp('scanner_status_topic').value),
            self._on_scanner_status, 10)
        self.create_subscription(
            Image, str(gp('wrist_camera_topic').value),
            self._on_wrist_frame, 10)

        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            'plant mission node ready — scanning and reporting only, '
            'plant_run_node owns the wheels')

    # ---- command dispatch --------------------------------------------

    def _on_cmd(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        action = data.get('action')

        # Any command may carry the shot count for the next scan.
        if data.get('samples') is not None:
            try:
                self._scan_samples = max(1, min(int(data['samples']), 40))
                self.get_logger().info(
                    f'scan sample count set to {self._scan_samples}')
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f'ignoring bad samples value: {data["samples"]!r}')

        if action == 'set_samples':
            self._publish_status()
            return
        # 'start' and 'next' are run-level: plant_run_node acts on those,
        # because it is the node that drives. Acting on them here too
        # would put two nodes in charge of one run.
        if action == 'scan_here':
            self._handle_scan_here(data)
        elif action == 'stop':
            self._handle_stop()
        elif action == 'pause':
            self._handle_pause()
        elif action == 'resume':
            self._handle_resume()

    # Geometry fields plant_run_node measures and passes straight through
    # to the arm. Aiming at a guess instead of these is what sent the top
    # of the sweep over the plant and into the ceiling.
    _GEOMETRY_FIELDS = ('x', 'y', 'z', 'plant_height', 'plant_width',
                        'z_top', 'z_bottom', 'plant_type')

    def _handle_scan_here(self, data=None):
        """Scan the plant the robot is already parked in front of.

        The only way a scan starts. plant_run_node has done the driving
        and is holding the wheels dead still; this node photographs and
        reports, and never moves the base."""
        if self._status in _ACTIVE_STATUSES:
            self.get_logger().warning(
                f'ignoring scan_here command: mission already {self._status}')
            return
        self._idx += 1
        self._error_msg = ''
        self._epoch += 1
        self._current_result = {
            'wp_idx': self._idx, 'captures': 0,
            'nav_status': 'parked by plant_run_node',
            'scan_status': None, 'health': None, 'confidence': None,
            'notes': None, 'timestamp': None,
        }
        self._start_arm_scan(data or {})

    def _handle_stop(self):
        self._epoch += 1
        self._scanner_cmd_pub.publish(String(data=json.dumps({'action': 'stop'})))
        self._capture_due_time = None
        self._pending_capture_label = None
        self._scan_start_time = None
        self._last_scanner_msg_time = None
        self._current_captures = []
        self._current_result = None
        self._status = 'STOPPED'
        self._publish_status()

    def _handle_pause(self):
        """Halt the arm where it is and hold, keeping what has been captured.

        Unlike stop, the captures and the waypoint position survive, so
        resume picks the plant back up rather than starting the run over.
        """
        if self._status not in _ACTIVE_STATUSES:
            self.get_logger().warning(
                f'ignoring pause: mission is {self._status}')
            return
        self._paused_from = self._status
        # Bump the epoch so any scan callback still in flight is discarded.
        self._epoch += 1
        self._scanner_cmd_pub.publish(String(data=json.dumps({'action': 'stop'})))
        self._capture_due_time = None
        self._pending_capture_label = None
        self._status = 'PAUSED'
        self.get_logger().info(
            f'paused ({len(self._current_captures)} captures held)')
        self._publish_status()

    def _handle_resume(self):
        """Carry on from a pause by rescanning the plant we were on."""
        if self._status != 'PAUSED':
            self.get_logger().warning(
                f'ignoring resume: mission is {self._status}')
            return
        self.get_logger().info('resuming — rescanning the current plant')
        self._start_arm_scan(self._last_geometry)

    # ---- finishing one plant -------------------------------------------

    def _finish_plant(self):
        """One plant is done with. plant_run_node is watching this status
        and takes the robot on to the next one; nothing is driven here."""
        if self._current_result is not None:
            self._current_result['timestamp'] = time.time()
            self._results.append(self._current_result)
            self._current_result = None
        self._current_captures = []
        self._status = 'COMPLETE'
        self._publish_status()

    # ---- arm scan ---------------------------------------------------------

    def _start_arm_scan(self, data=None):
        # Where the plant actually is, as measured by whoever parked the
        # robot in front of it. Only the fields that were sent are passed
        # on; anything missing falls back to this node's defaults, and the
        # arm falls back to its own beyond that.
        geometry = {k: (data or {})[k] for k in self._GEOMETRY_FIELDS
                    if (data or {}).get(k) is not None}
        self._last_geometry = geometry
        self._current_captures = []
        self._scan_start_time = self.get_clock().now()
        self._last_scanner_msg_time = None
        self._captured_viewpoint = None
        self._last_scanner_status = None
        # Defensive reset — arm_scanner_node can also auto-start a scan
        # from /ecobot/detections; this guards against one still running.
        self._scanner_cmd_pub.publish(String(data=json.dumps({'action': 'stop'})))
        cmd = {'action': 'scan', 'x': self._arm_scan_x,
               'y': self._arm_scan_y, 'z': self._arm_scan_z,
               'samples': self._scan_samples}
        cmd.update(geometry)
        if geometry:
            self.get_logger().info(
                f'scanning a measured plant: {geometry}')
        else:
            self.get_logger().warning(
                'no plant measurements given — aiming the arm at the '
                'fallback point, which may not be where the plant is')
        self._scanner_cmd_pub.publish(String(data=json.dumps(cmd)))
        self._status = 'SCANNING'
        self._publish_status()

    def _on_scanner_status(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        prev = self._last_scanner_status
        self._last_scanner_status = data
        self._last_scanner_msg_time = self.get_clock().now()
        if self._status != 'SCANNING':
            return

        if data.get('status') == 'scanning':
            # Only photograph the sampled viewpoints. Older scanners did not
            # send this flag, so treat its absence as "capture", keeping the
            # previous behaviour rather than silently taking no photos.
            wants_capture = data.get('capture', True)
            if not wants_capture:
                return

            if data.get('settled'):
                # The arm has reached this viewpoint and stopped. This is
                # the only moment worth photographing from.
                self._saw_settled = True
                if self._captured_viewpoint != data.get('viewpoint'):
                    self._captured_viewpoint = data.get('viewpoint')
                    self._pending_capture_label = data.get('current_label', '')
                    self._capture_due_time = (
                        self.get_clock().now()
                        + Duration(seconds=self._settle_delay_s))
                return

            # No settle signal has ever arrived, so this is an arm that
            # cannot say when it has stopped. Fall back to the old fixed
            # wait rather than never taking a photograph at all.
            if self._saw_settled:
                return
            if (prev is None or prev.get('status') != 'scanning'
                    or prev.get('viewpoint') != data.get('viewpoint')):
                self._pending_capture_label = data.get('current_label', '')
                self._capture_due_time = (
                    self.get_clock().now() + Duration(seconds=self._capture_delay_s))
        elif data.get('status') in ('idle', 'recovering') and prev is not None \
                and prev.get('status') in ('scanning', 'recovering'):
            self._on_scan_complete()

    def _on_scan_complete(self):
        self._scan_start_time = None
        self._capture_due_time = None
        if self._current_result is not None:
            self._current_result['captures'] = len(self._current_captures)
            self._current_result['scan_status'] = (
                'ok' if self._current_captures else 'no_captures')
            if self._last_scanner_status and 'parts_covered' in self._last_scanner_status:
                self._current_result['parts_covered'] = self._last_scanner_status['parts_covered']

        if self._gemini is None:
            if self._current_result is not None:
                self._current_result['health'] = 'unknown'
                self._current_result['notes'] = 'GOOGLE_API_KEY not set'
            self._finish_plant()
            return

        self._spawn_gemini_call()
        self._status = 'ANALYZING'
        self._publish_status()

    def _spawn_gemini_call(self):
        captures = list(self._current_captures)
        epoch = self._epoch
        idx = self._idx

        def _worker():
            labels = [label for label, _ in captures]
            images = [jpg for _, jpg in captures]
            # The Live streaming API only serves *-live-preview models; the
            # report runs on a pro model, so use the standard call.
            result = self._gemini.assess_plant(images, labels=labels)
            with self._result_lock:
                self._pending_gemini_result = (epoch, idx, result)

        threading.Thread(target=_worker, daemon=True).start()

    # ---- wrist camera capture -------------------------------------------

    def _on_wrist_frame(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f'wrist frame decode error: {e}')
            return
        with self._wrist_lock:
            self._wrist_frame = frame
            self._wrist_frame_stamp = time.time()

    def _get_wrist_jpeg(self):
        with self._wrist_lock:
            frame = self._wrist_frame
            stamp = self._wrist_frame_stamp
        if frame is None or (time.time() - stamp) > self._frame_max_age_s:
            return None
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return buf.tobytes() if ok else None

    def _do_capture(self, label):
        jpeg = self._get_wrist_jpeg()
        if jpeg is None:
            self.get_logger().warning(
                f'no fresh wrist frame for viewpoint "{label}", skipping capture')
            return
        self._current_captures.append((label, jpeg))
        self._publish_scan_capture(label, jpeg)

    # ---- publishing --------------------------------------------------------

    def _publish_status(self):
        payload = {
            'status': self._status,
            'idx': max(0, self._idx),
            # A running tally: the plants finished, plus the one in hand.
            'total': len(self._results) + (1 if self._current_result else 0),
            'results': self._results,
            # This node no longer holds positions — plant_run_node does,
            # and publishes them on /ecobot/nav_status.
            'waypoints': [],
            'samples': self._scan_samples,
            'captures': len(self._current_captures),
        }
        if self._error_msg:
            payload['error'] = self._error_msg
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _publish_scan_capture(self, label, jpeg_bytes):
        payload = {
            # The dashboard decodes this with bytes.fromhex(...) —
            # this MUST be a hex string, not base64.
            'image_jpeg': jpeg_bytes.hex(),
            'capture_count': len(self._current_captures),
            # Repurposed: no detected-object class applies here, so this
            # carries the scan viewpoint label (front/right/left/top).
            'class': label,
        }
        self._scan_capture_pub.publish(String(data=json.dumps(payload)))

    # ---- periodic tick ------------------------------------------------------

    def _tick(self):
        now = self.get_clock().now()
        self._publish_status()

        if self._capture_due_time is not None and now >= self._capture_due_time:
            label = self._pending_capture_label
            self._capture_due_time = None
            self._pending_capture_label = None
            self._do_capture(label)

        if self._status == 'SCANNING' and self._scan_start_time is not None:
            # A scan ends when the arm says it has ended. These two only
            # catch an arm that has stopped talking or has plainly hung:
            # while it keeps reporting, the wait carries on, however long
            # the sweep takes. The old fixed budget from the start of the
            # scan cut sweeps short and recorded them as finished.
            since_arm = (
                (now - self._last_scanner_msg_time).nanoseconds / 1e9
                if self._last_scanner_msg_time is not None
                else (now - self._scan_start_time).nanoseconds / 1e9)
            total = (now - self._scan_start_time).nanoseconds / 1e9
            reason = None
            if since_arm > self._scan_silence_timeout_s:
                reason = (f'the arm stopped reporting for '
                          f'{since_arm:.0f}s')
            elif total > self._scan_hard_timeout_s:
                reason = (f'the scan passed its hard limit of '
                          f'{self._scan_hard_timeout_s:.0f}s')
            if reason is not None:
                self.get_logger().warning(f'abandoning this plant: {reason}')
                self._scan_start_time = None
                self._last_scanner_msg_time = None
                if self._current_result is not None:
                    self._current_result['scan_status'] = f'timeout ({reason})'
                self._finish_plant()

        with self._result_lock:
            pending = self._pending_gemini_result
            self._pending_gemini_result = None
        if pending is not None:
            epoch, idx, result = pending
            if epoch == self._epoch and self._status == 'ANALYZING':
                if self._current_result is not None:
                    self._current_result['health'] = result['health']
                    self._current_result['confidence'] = result['confidence']
                    self._current_result['notes'] = result['notes']
                    if not self._current_result.get('scan_status'):
                        self._current_result['scan_status'] = (
                            'gemini_error' if result.get('error') else 'ok')
                self._finish_plant()


def main(args=None):
    rclpy.init(args=args)
    node = PlantMissionNode()
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
