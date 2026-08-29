#!/bin/bash
set -e

echo "=========================================="
echo "          ECOBOT SYSTEM STARTUP           "
echo "=========================================="

# Parse command-line options
ENABLE_DETECTION=true
SERIAL_PORT="/dev/ttyACM0"

while [[ $# -gt 0 ]]; do
  case $1 in
    --check|-c)
      shift
      echo "=== Running EcoBot Hardware Check ==="
      python3 "$HOME/ros2_ws/src/ecobot/ecobot_bringup/scripts/hardware_check" "$@"
      exit $?
      ;;
    --no-detection)
      ENABLE_DETECTION=false
      shift
      ;;
    --port|-p)
      SERIAL_PORT="$2"
      shift 2
      ;;
    *)
      if [[ "$1" == /dev/* ]]; then
        SERIAL_PORT="$1"
      fi
      shift
      ;;
  esac
done

# Source ROS 2 environments
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash 2>/dev/null || true

# Kill any stale processes from a previous run. A prior start.sh may have died
# without running its exit trap (closed terminal, dropped SSH session, kill -9),
# which orphans "ros2 launch" and every node it started (reparented to init,
# never receiving the shutdown signal). Left alive, those orphans collide with
# the fresh instances below on I2C, cameras, and ports, and that contention is
# what crashes nodes like arm_manual_node on the new launch. Match everything
# this stack can start, not just the three names ecobot.launch.py used to run
# directly, so a stale generation never lingers into the next run.
ECOBOT_PROC_PATTERN="ros2_ws/install|ros2 launch ecobot_bringup|rosbridge_websocket|static_transform_publisher|lifecycle_manager|controller_server|planner_server|behavior_server|bt_navigator|depthimage_to_laserscan_node|robot_state_publisher|livekit_streamer"
pkill -f "$ECOBOT_PROC_PATTERN" 2>/dev/null || true

# Services this script no longer starts (the 8080 static UI, the 3001 Next.js
# server, the standalone WebRTC streamers) can still be running as orphans from
# an older start.sh, holding their ports. Sweep them once so an upgraded script
# does not leave the previous generation behind.
pkill -f "http.server 8080|next-server|next start|webrtc_streamer|camera_webserver|cloudflared" 2>/dev/null || true
sleep 2

# Source LiveKit env if present
if [ -f "$HOME/ros2_ws/src/ecobot/ecobot_bringup/.env" ]; then
  set -a
  . "$HOME/ros2_ws/src/ecobot/ecobot_bringup/.env"
  set +a
fi

# 1. Web dashboards are no longer served from the robot, and neither is a
# Cloudflare tunnel. The dashboard is the Vercel-hosted ecobot-ui: on the LAN
# it talks to rosbridge on 9090 directly, and from anywhere else it rides the
# LiveKit room that already carries the video, via livekit_bridge. Both ends
# dial out to LiveKit, so nothing needs to reach inbound to the robot and no
# public endpoint has to be opened.

# 2. Launch Full ROS 2 Stack
echo "  [ROS 2] Starting EcoBot nodes (sensors, arm, motor, diagnostics, detection, rosbridge)..."
ros2 launch ecobot_bringup ecobot.launch.py \
  serial_port:="$SERIAL_PORT" \
  use_sim_time:=false \
  enable_sensors:=true \
  enable_arm:=true \
  enable_navigation:=true \
  enable_detection:="$ENABLE_DETECTION" \
  enable_rosbridge:=true \
  enable_diagnostics:=true \
  enable_urdf:=false \
  enable_obstacle_avoidance:=true &
ROS_PID=$!

sleep 4

# Detect local IP
LOCAL_IP="$(hostname -I | awk '{print $1}')"

echo ""
echo "=========================================="
echo "          ECOBOT STACK IS ACTIVE          "
echo "=========================================="
echo "  Dashboard (remote): https://ecobot-ui.vercel.app   [via LiveKit]"
echo "  Dashboard (LAN):    http://$LOCAL_IP:3001/?robot=$LOCAL_IP"
echo "  Rosbridge WS:       ws://$LOCAL_IP:9090"
echo "  LiveKit Cloud:      wss://dialog-project-ew7yzd0u.livekit.cloud"
echo "  LiveKit Room:       ecobot-control"
echo "  Serial Port:        $SERIAL_PORT"
echo "  Hardware Diag:      Active (/diagnostics, /ecobot/hardware_status)"
echo "=========================================="
echo "Press Ctrl+C to stop all services."

# Cleanup trap on exit
cleanup() {
  echo ""
  echo "Shutting down EcoBot stack..."
  kill $ROS_PID 2>/dev/null || true
  sleep 1
  pkill -f "$ECOBOT_PROC_PATTERN" 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM
wait $ROS_PID
