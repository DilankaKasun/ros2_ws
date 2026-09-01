"""Bridge ROS 2 topics over LiveKit data messages.

This replaces rosbridge for remote clients. rosbridge listens on a plain
ws:// port, so an HTTPS-hosted dashboard cannot reach it without a public
TLS endpoint in front — which is what the Cloudflare tunnel used to
provide, at the cost of exposing an unauthenticated socket to the whole
internet.

Here both sides instead dial *out* to the LiveKit SFU, which is already
wss://, so nothing has to connect inbound to the robot and no tunnel is
needed. Access is gated by the same signed LiveKit token that already
gates video, and publishing is further restricted to an explicit topic
allowlist (see `writable_topics`) so a viewer token cannot drive the
robot just by knowing a topic name.

Wire format is JSON, deliberately shaped like the subset of the rosbridge
protocol the dashboard already speaks, so the client hook can mirror
useRos:

    inbound   {"op": "publish",   "topic": "/arm/joint_commands",
               "type": "std_msgs/msg/Float64MultiArray", "msg": {...}}
    inbound   {"op": "subscribe", "topic": "/arm/joint_angles"}
    outbound  {"op": "topic",     "topic": "/arm/joint_angles", "msg": {...}}

Telemetry is broadcast to the room on a fixed cadence per topic rather
than on every ROS callback: /arm/joint_angles alone ticks at ~66Hz while
the node is ramping, which is far more than a UI needs and enough to
congest the data channel.
"""

import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, Optional

from dotenv import find_dotenv, load_dotenv

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_message

from livekit import api, rtc

load_dotenv(find_dotenv(usecwd=True))

# Topics the robot streams out to the room, with the minimum seconds between
# sends for each. These are telemetry, so a dropped sample is harmless.
DEFAULT_READABLE = {
    '/arm/joint_angles': 0.05,
    '/arm/pose': 0.10,
    '/arm/status': 0.25,
    '/arm/pose_goal_result': 0.05,
    '/arm/scanner_status': 0.25,
    '/ecobot/plant_scan_status': 0.25,
    # Scan photos. Each is a whole JPEG as hex, so this is by far the
    # heaviest topic here; it only ticks while a scan is running.
    '/ecobot/scan_capture': 0.05,
    '/ecobot/hardware_status': 1.00,
    '/ecobot/detections': 0.25,
    '/ecobot/tof_ranges': 0.20,
    '/odom': 0.10,
    # The run: which driver has the wheels, what state it is in, why, and
    # how long before that state times out. The dashboard's whole run view
    # is built on this, and without it a remote client can START a run —
    # plant_scan_cmd is writable below — but never see that anything
    # happened, so the buttons sit dead while the robot drives.
    '/ecobot/nav_status': 0.25,
    # What each driver is asking for, and what the wheels were actually
    # told. The handover is decided by which of these two is still live, so
    # seeing them side by side is the only way to debug it from a distance.
    # Read-only here; the writable allowlist below governs commanding.
    '/goto_cmd_vel': 0.10,
    '/nav_cmd_vel': 0.10,
    '/cmd_vel': 0.10,
    # Whether the obstacle-avoidance layer has been asked to stand down. It
    # decides whether an approach can reach a plant at all, so it is worth
    # seeing from the dashboard when an approach stalls.
    '/ecobot/goto_suppress_avoidance': 0.25,
    # What the wrist camera can see, so a scan aiming at the wrong thing is
    # visible remotely rather than only in the robot's own logs.
    '/arm/detections': 0.25,
}

# Topics a remote client is allowed to publish to. Anything not listed is
# rejected — this is the actual authorization boundary for robot motion, so
# keep it as small as the dashboard genuinely needs.
DEFAULT_WRITABLE = {
    '/arm/joint_commands': 'std_msgs/msg/Float64MultiArray',
    '/arm/pose_goal': 'std_msgs/msg/Float64MultiArray',
    '/arm/enable': 'std_msgs/msg/String',
    '/arm/scanner_cmd': 'std_msgs/msg/String',
    '/arm/vla_prompt': 'std_msgs/msg/String',
    '/cmd_vel': 'geometry_msgs/msg/Twist',
    '/nav_cmd_vel': 'geometry_msgs/msg/Twist',
    '/ecobot/plant_scan_cmd': 'std_msgs/msg/String',
    '/ecobot/goto_target': 'std_msgs/msg/String',
}


class LiveKitBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('livekit_bridge')

        self.declare_parameter(
            'livekit_url',
            os.environ.get(
                'LIVEKIT_URL', 'wss://dialog-project-ew7yzd0u.livekit.cloud'
            ),
        )
        self.declare_parameter(
            'livekit_api_key', os.environ.get('LIVEKIT_API_KEY', ''))
        self.declare_parameter(
            'livekit_api_secret', os.environ.get('LIVEKIT_API_SECRET', ''))
        self.declare_parameter(
            'livekit_room', os.environ.get('LIVEKIT_ROOM', 'ecobot-control'))
        # Overridable so a deployment can widen or (more usefully) narrow the
        # command surface without editing code.
        self.declare_parameter('readable_topics', json.dumps(DEFAULT_READABLE))
        self.declare_parameter('writable_topics', json.dumps(DEFAULT_WRITABLE))

        self._url = str(self.get_parameter('livekit_url').value)
        self._api_key = str(self.get_parameter('livekit_api_key').value)
        self._api_secret = str(self.get_parameter('livekit_api_secret').value)
        self._room_name = str(self.get_parameter('livekit_room').value)

        self._readable: Dict[str, float] = self._load_param(
            'readable_topics', DEFAULT_READABLE)
        self._writable: Dict[str, str] = self._load_param(
            'writable_topics', DEFAULT_WRITABLE)

        self._room: Optional[rtc.Room] = None
        self._running = True
        self._connected = False

        self._topic_pubs: Dict[str, Any] = {}
        self._topic_subs: Dict[str, Any] = {}
        self._last_sent: Dict[str, float] = {}
        # Telemetry is only worth serialising and sending when somebody is
        # actually in the room to receive it.
        self._has_peers = False

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._subscribe_readable()

        self.get_logger().info(
            f'LiveKit bridge starting. room={self._room_name} '
            f'readable={len(self._readable)} writable={len(self._writable)}')

    def _load_param(self, name: str, fallback: dict) -> dict:
        raw = self.get_parameter(name).value
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception as e:
            self.get_logger().warn(f'{name} invalid ({e}); using defaults')
        return dict(fallback)

    # ---------------- ROS -> LiveKit ----------------

    def _subscribe_readable(self):
        for topic, interval in self._readable.items():
            msg_type = self._discover_type(topic)
            if msg_type is None:
                # The publisher may simply not be up yet; retry below.
                continue
            self._make_subscription(topic, msg_type, float(interval))

        if len(self._topic_subs) < len(self._readable):
            # Nodes come up in any order, so topics whose type could not be
            # resolved yet are retried instead of being lost for the session.
            self._retry_timer = self.create_timer(5.0, self._retry_subscribe)

    def _retry_subscribe(self):
        for topic, interval in self._readable.items():
            if topic in self._topic_subs:
                continue
            msg_type = self._discover_type(topic)
            if msg_type is not None:
                self._make_subscription(topic, msg_type, float(interval))

        if len(self._topic_subs) >= len(self._readable):
            self._retry_timer.cancel()
            self.get_logger().info('All readable topics subscribed')

    def _discover_type(self, topic: str) -> Optional[str]:
        for name, types in self.get_topic_names_and_types():
            if name == topic and types:
                return types[0]
        return None

    def _make_subscription(self, topic: str, type_str: str, interval: float):
        try:
            msg_class = get_message(type_str)
        except Exception as e:
            self.get_logger().warn(f'cannot resolve {type_str} for {topic}: {e}')
            return

        self._topic_subs[topic] = self.create_subscription(
            msg_class, topic,
            lambda msg, t=topic, i=interval: self._on_ros_msg(t, msg, i),
            10)
        self.get_logger().info(
            f'  -> forwarding {topic} ({type_str}) every {interval}s')

    def _on_ros_msg(self, topic: str, msg, interval: float):
        if not self._connected or not self._has_peers:
            return
        now = time.monotonic()
        if now - self._last_sent.get(topic, 0.0) < interval:
            return
        self._last_sent[topic] = now

        try:
            payload = json.dumps({
                'op': 'topic',
                'topic': topic,
                'msg': message_to_ordereddict(msg),
            }).encode('utf-8')
        except Exception as e:
            self.get_logger().warn(f'serialise failed for {topic}: {e}')
            return

        # Telemetry is periodic, so a lost packet is superseded by the next
        # one; lossy avoids head-of-line blocking on a congested channel.
        asyncio.run_coroutine_threadsafe(
            self._send(payload, reliable=False), self._loop)

    async def _send(self, payload: bytes, reliable: bool = True):
        if self._room is None:
            return
        try:
            await self._room.local_participant.publish_data(
                payload, reliable=reliable)
        except Exception as e:
            self.get_logger().debug(f'publish_data failed: {e}')

    # ---------------- LiveKit -> ROS ----------------

    def _on_data(self, packet: rtc.DataPacket):
        try:
            request = json.loads(packet.data.decode('utf-8'))
        except Exception:
            return
        if not isinstance(request, dict):
            return

        op = request.get('op')
        if op == 'publish':
            self._handle_publish(request)
        elif op in ('subscribe', 'unsubscribe'):
            # Subscriptions are driven by the readable allowlist, so these are
            # accepted and ignored; the client speaks them for parity with
            # rosbridge and should not be treated as an error.
            pass

    def _handle_publish(self, request: dict):
        topic = request.get('topic')
        payload = request.get('msg')
        if not isinstance(topic, str) or not isinstance(payload, dict):
            return

        expected = self._writable.get(topic)
        if expected is None:
            self.get_logger().warn(
                f'rejected publish to {topic}: not in writable_topics')
            return

        # A client-declared type is only honoured when it matches the
        # allowlist, so a caller cannot pick the message class being built.
        declared = request.get('type')
        if isinstance(declared, str) and declared and declared != expected:
            self.get_logger().warn(
                f'rejected publish to {topic}: type {declared} '
                f'does not match {expected}')
            return

        publisher = self._topic_pubs.get(topic)
        if publisher is None:
            try:
                publisher = self.create_publisher(get_message(expected), topic, 10)
            except Exception as e:
                self.get_logger().error(f'cannot create publisher {topic}: {e}')
                return
            self._topic_pubs[topic] = publisher

        try:
            msg = get_message(expected)()
            set_message_fields(msg, payload)
        except Exception as e:
            self.get_logger().warn(f'bad payload for {topic}: {e}')
            return

        publisher.publish(msg)

    # ---------------- connection ----------------

    def _mint_token(self) -> str:
        return (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity('ecobot-ros-bridge')
            .with_name('EcoBot ROS Bridge')
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=True,
                    # Without this the bridge connects and receives fine but
                    # every outbound telemetry send is refused by the SFU.
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())

    def _refresh_peers(self):
        if self._room is None:
            self._has_peers = False
            return
        self._has_peers = len(self._room.remote_participants) > 0

    async def _connect(self):
        if not self._api_key or not self._api_secret:
            self.get_logger().error(
                'LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set; '
                'the bridge cannot connect. Set them in '
                'ecobot_bringup/.env or pass them as parameters.')
            return

        while self._running:
            try:
                self.get_logger().info(f'Connecting to LiveKit {self._url}...')
                self._room = rtc.Room()

                self._room.on('data_received', self._on_data)
                self._room.on(
                    'participant_connected',
                    lambda *_: self._refresh_peers())
                self._room.on(
                    'participant_disconnected',
                    lambda *_: self._refresh_peers())

                await self._room.connect(self._url, self._mint_token())
                self._connected = True
                self._refresh_peers()
                self.get_logger().info(
                    f'Bridge connected to room "{self._room.name}"')

                while (self._running and self._room.connection_state
                       == rtc.ConnectionState.CONN_CONNECTED):
                    await asyncio.sleep(1.0)

                self._connected = False
                self.get_logger().warn('LiveKit disconnected; reconnecting...')

            except Exception as e:
                self._connected = False
                self.get_logger().warn(f'LiveKit error: {e}; retry in 4s')
                await asyncio.sleep(4.0)

    def destroy_node(self):
        self._running = False
        if self._room is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._room.disconnect(), self._loop).result(timeout=3)
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LiveKitBridgeNode()
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
