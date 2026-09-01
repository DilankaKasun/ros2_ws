import asyncio
import json
import os
import threading
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge
from dotenv import find_dotenv, load_dotenv
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from livekit import api, rtc
from livekit.rtc import TrackPublishOptions, TrackSource, VideoBufferType, VideoFrame

load_dotenv(find_dotenv(usecwd=True))


class LiveKitStreamerNode(Node):
    """ROS 2 Node that streams robot camera feeds directly to LiveKit Cloud WebRTC SFU

    with ultra-low global conferencing latency (~150-250ms).
    """

    def __init__(self) -> None:
        super().__init__('livekit_streamer')

        self.declare_parameter(
            'livekit_url',
            os.environ.get(
                'LIVEKIT_URL', 'wss://dialog-project-ew7yzd0u.livekit.cloud'
            ),
        )
        self.declare_parameter(
            'livekit_api_key',
            os.environ.get('LIVEKIT_API_KEY', 'APIS7AD7a4oiJiw'),
        )
        self.declare_parameter(
            'livekit_api_secret',
            os.environ.get(
                'LIVEKIT_API_SECRET',
                '9CWunO7VdCqTyOP6QlwpQFb2lrUGM64puyWEBzmedBS',
            ),
        )
        self.declare_parameter(
            'livekit_room', os.environ.get('LIVEKIT_ROOM', 'ecobot-control')
        )
        self.declare_parameter(
            'color_topic',
            os.environ.get('ECOBOT_COLOR_TOPIC', '/camera/color/image_raw'),
        )
        self.declare_parameter(
            'arm_topic',
            os.environ.get('ECOBOT_ARM_TOPIC', '/arm/camera/image_raw'),
        )
        self.declare_parameter(
            'detection_topic',
            os.environ.get('ECOBOT_DETECTION_TOPIC', '/ecobot/detection_image'),
        )
        # Boxes are drawn onto the main colour track here rather than relying
        # on /ecobot/detection_image: that overlay is only republished at the
        # detector's inference rate (~2Hz), which is too choppy to watch a
        # live approach on. Drawing the newest boxes over the 30fps colour
        # frames gives smooth video with up-to-date boxes.
        self.declare_parameter('draw_detections', True)
        self.declare_parameter('detection_data_topic', '/ecobot/detections')
        # Boxes older than this stop being drawn, so a dead detector shows as
        # a clean picture instead of stale boxes frozen on screen.
        self.declare_parameter('detection_max_age_s', 2.0)
        # Classes detection_goto will actually drive to. Drawn in a different
        # colour so it is obvious at a glance whether the thing in view is a
        # class the robot chases, or one it will ignore.
        self.declare_parameter('highlight_classes', ['potted plant'])

        self._url = str(self.get_parameter('livekit_url').value)
        self._api_key = str(self.get_parameter('livekit_api_key').value)
        self._api_secret = str(self.get_parameter('livekit_api_secret').value)
        self._room_name = str(self.get_parameter('livekit_room').value)

        color_topic = str(self.get_parameter('color_topic').value)
        arm_topic = str(self.get_parameter('arm_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        self._draw_detections = bool(
            self.get_parameter('draw_detections').value)
        detection_data_topic = str(
            self.get_parameter('detection_data_topic').value)
        self._det_max_age_s = float(
            self.get_parameter('detection_max_age_s').value)
        self._highlight_classes = {
            str(c).strip().lower()
            for c in self.get_parameter('highlight_classes').value}

        self._bridge = CvBridge()
        self._room: Optional[rtc.Room] = None
        self._running = True

        self._det_lock = threading.Lock()
        self._latest_dets = []
        self._latest_dets_stamp = 0.0

        # Video Sources and Tracks (640x360 default resolution for fast VP8/H264 encoding)
        self._color_source = rtc.VideoSource(640, 360)
        self._color_track = rtc.LocalVideoTrack.create_video_track(
            'realsense_rgb', self._color_source
        )

        self._arm_source = rtc.VideoSource(640, 360)
        self._arm_track = rtc.LocalVideoTrack.create_video_track(
            'arm_camera', self._arm_source
        )

        self._det_source = rtc.VideoSource(640, 360)
        self._det_track = rtc.LocalVideoTrack.create_video_track(
            'detection_overlay', self._det_source
        )

        # Frame rate limiters
        self._last_color_time = 0.0
        self._last_arm_time = 0.0
        self._last_det_time = 0.0

        # ROS Subscriptions
        self._color_sub = self.create_subscription(
            Image, color_topic, self._color_cb, 10
        )
        self._arm_sub = self.create_subscription(
            Image, arm_topic, self._arm_cb, 10
        )
        self._det_sub = self.create_subscription(
            Image, detection_topic, self._det_cb, 10
        )
        if self._draw_detections:
            self._det_data_sub = self.create_subscription(
                String, detection_data_topic, self._det_data_cb, 10
            )

        # Start Asyncio LiveKit Connection in daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'LiveKit Streamer initialized. Room: {self._room_name}, Target: {self._url}'
        )

    def _mint_token(self) -> str:
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity('ecobot-video-publisher')
            .with_name('EcoBot Video Publisher')
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )
        return token

    def _run_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_publish())

    async def _connect_and_publish(self):
        while self._running:
            try:
                self.get_logger().info(
                    f'Connecting to LiveKit Room "{self._room_name}" at {self._url}...'
                )
                token = self._mint_token()
                self._room = rtc.Room()

                await self._room.connect(self._url, token)
                self.get_logger().info(
                    f'Successfully connected to LiveKit Room: {self._room.name}'
                )

                # Publish tracks with camera source options
                opts = TrackPublishOptions(source=TrackSource.SOURCE_CAMERA)
                await self._room.local_participant.publish_track(
                    self._color_track, opts
                )
                await self._room.local_participant.publish_track(
                    self._arm_track, opts
                )
                await self._room.local_participant.publish_track(
                    self._det_track, opts
                )
                self.get_logger().info(
                    'Published WebRTC video tracks: realsense_rgb, arm_camera, detection_overlay'
                )

                # Keep connected while running
                while self._running and self._room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    await asyncio.sleep(1.0)

            except Exception as e:
                self.get_logger().warn(
                    f'LiveKit connection error: {e}. Retrying in 4 seconds...'
                )
                await asyncio.sleep(4.0)

    def _det_data_cb(self, msg: String):
        try:
            dets = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(dets, list):
            return
        with self._det_lock:
            self._latest_dets = dets
            self._latest_dets_stamp = time.time()

    def _draw_boxes(self, img, src_w, src_h):
        """Draw the newest detections onto an already-resized frame.

        Boxes arrive in the detector's own image coordinates, so they are
        scaled to whatever size the frame was resized to before drawing.
        """
        with self._det_lock:
            dets = self._latest_dets
            stamp = self._latest_dets_stamp
        age = time.time() - stamp
        if not dets or age > self._det_max_age_s:
            return

        h, w = img.shape[:2]
        sx = w / float(max(1, src_w))
        sy = h / float(max(1, src_h))

        for d in dets:
            bbox = d.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            name = str(d.get('class_name') or d.get('class') or '?')
            tracked = name.strip().lower() in self._highlight_classes
            # Green for a class the robot will drive to, grey for one it
            # ignores — the difference matters more than the box itself.
            colour = (0, 220, 0) if tracked else (150, 150, 150)

            x1 = int(bbox[0] * sx)
            y1 = int(bbox[1] * sy)
            x2 = int(bbox[2] * sx)
            y2 = int(bbox[3] * sy)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2 if tracked else 1)

            label = name
            conf = d.get('confidence')
            if conf is not None:
                label += f' {float(conf):.2f}'
            z = d.get('z', d.get('distance'))
            if z is not None:
                label += f'  {float(z):.2f}m'

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty), colour, -1)
            cv2.putText(img, label, (x1 + 2, ty - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                        cv2.LINE_AA)

        # Age of the boxes, so a stalled detector is visible on the video
        # itself rather than only in the logs.
        cv2.putText(img, f'{len(dets)} det  {age * 1000:.0f}ms old',
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)

    def _process_and_capture(
        self,
        msg: Image,
        source: rtc.VideoSource,
        target_w: int = 640,
        target_h: int = 360,
        draw_boxes: bool = False,
    ):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_img.shape[:2]

            if w != target_w or h != target_h:
                cv_img = cv2.resize(
                    cv_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

            if draw_boxes and self._draw_detections:
                self._draw_boxes(cv_img, w, h)

            rgba_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
            raw_bytes = rgba_img.tobytes()

            frame = VideoFrame(
                target_w, target_h, VideoBufferType.RGBA, raw_bytes
            )
            source.capture_frame(frame)
        except Exception:
            pass

    def _color_cb(self, msg: Image):
        now = time.time()
        # Cap at 30 FPS (0.033s interval)
        if now - self._last_color_time < 0.030:
            return
        self._last_color_time = now
        self._process_and_capture(
            msg, self._color_source, 640, 360, draw_boxes=True)

    def _arm_cb(self, msg: Image):
        now = time.time()
        # Cap at 20 FPS (0.050s interval)
        if now - self._last_arm_time < 0.050:
            return
        self._last_arm_time = now
        self._process_and_capture(msg, self._arm_source, 640, 360)

    def _det_cb(self, msg: Image):
        now = time.time()
        # Cap at 20 FPS
        if now - self._last_det_time < 0.050:
            return
        self._last_det_time = now
        self._process_and_capture(msg, self._det_source, 640, 360)

    def destroy_node(self):
        self._running = False
        if self._room and self._loop:
            asyncio.run_coroutine_threadsafe(self._room.disconnect(), self._loop)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LiveKitStreamerNode()
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
