import asyncio
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

        self._url = str(self.get_parameter('livekit_url').value)
        self._api_key = str(self.get_parameter('livekit_api_key').value)
        self._api_secret = str(self.get_parameter('livekit_api_secret').value)
        self._room_name = str(self.get_parameter('livekit_room').value)

        color_topic = str(self.get_parameter('color_topic').value)
        arm_topic = str(self.get_parameter('arm_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)

        self._bridge = CvBridge()
        self._room: Optional[rtc.Room] = None
        self._running = True

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

    def _process_and_capture(
        self,
        msg: Image,
        source: rtc.VideoSource,
        target_w: int = 640,
        target_h: int = 360,
    ):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_img.shape[:2]

            if w != target_w or h != target_h:
                cv_img = cv2.resize(
                    cv_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

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
        self._process_and_capture(msg, self._color_source, 640, 360)

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
