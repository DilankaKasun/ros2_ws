#!/bin/bash
set -e

echo "=========================================="
echo "          ECOBOT SYSTEM STARTUP           "
echo "=========================================="

# Parse command-line options
ENABLE_TUNNEL=false
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
    --tunnel|-t)
      ENABLE_TUNNEL=true
      shift
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
pkill -f "http.server 8080|next-server|next start|webrtc_streamer|camera_webserver" 2>/dev/null || true
sleep 2

# Source LiveKit env if present
if [ -f "$HOME/ros2_ws/src/ecobot/ecobot_bringup/.env" ]; then
  set -a
  . "$HOME/ros2_ws/src/ecobot/ecobot_bringup/.env"
  set +a
fi

# 1. Web dashboards are no longer served from the robot. The dashboard is the
# Vercel-hosted ecobot-ui, which reaches the robot over rosbridge (port 9090)
# and LiveKit, so the local Next.js server (3001) and the static fallback UI
# (8080) were both redundant and have been removed. The ~/ecobot-ui and
# ~/remote_web_ui folders are left on disk; nothing here starts them.

# 2. Launch Cloudflare Tunnel if requested
TUNNEL_PID=""
TUNNEL_URL=""
if [ "$ENABLE_TUNNEL" = true ]; then
  echo "  [Tunnel] Launching Cloudflare Tunnel for Rosbridge (Port 9090)..."
  CLOUDFLARED_BIN="$(which cloudflared 2>/dev/null || echo "$HOME/.local/bin/cloudflared")"
  if [ -x "$CLOUDFLARED_BIN" ]; then
    LOG_FILE="/tmp/cloudflared_9090.log"
    rm -f "$LOG_FILE"
    "$CLOUDFLARED_BIN" tunnel --url "http://localhost:9090" > "$LOG_FILE" 2>&1 &
    TUNNEL_PID=$!
    
    # Wait for tunnel URL
    for i in {1..25}; do
      if [ -f "$LOG_FILE" ]; then
        TUNNEL_URL=$(grep -o "https://[a-zA-Z0-9.-]*\.trycloudflare\.com" "$LOG_FILE" | head -n 1 || true)
        if [ -n "$TUNNEL_URL" ]; then
          break
        fi
      fi
      sleep 0.5
    done

    if [ -n "$TUNNEL_URL" ]; then
      WSS_URL=$(echo "$TUNNEL_URL" | sed 's/https:/wss:/')
      cat <<EOF > "$HOME/remote_web_ui/tunnel_config.json"
{
  "updated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "port": 9090,
  "tunnel_https": "$TUNNEL_URL",
  "tunnel_wss": "$WSS_URL"
}
EOF
      echo "  [Tunnel] Public WSS URL: $WSS_URL"
    fi
  else
    echo "  [Tunnel] Warning: cloudflared binary not found."
  fi
fi

# 3. Launch Full ROS 2 Stack
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
echo "  Dashboard:          https://ecobot-ui.vercel.app"
echo "  Rosbridge WS:       ws://$LOCAL_IP:9090"
echo "  LiveKit Cloud:      wss://dialog-project-ew7yzd0u.livekit.cloud"
echo "  LiveKit Room:       ecobot-control"
echo "  Serial Port:        $SERIAL_PORT"
echo "  Hardware Diag:      Active (/diagnostics, /ecobot/hardware_status)"

if [ -n "$TUNNEL_URL" ]; then
  TUNNEL_HOST=$(echo "$TUNNEL_URL" | sed 's~https://~~')
  echo ""
  echo "  >>> CLOUDFLARE TUNNEL IS ONLINE <<<"
  echo "  Tunnel URL:         $TUNNEL_URL"
  echo "  Vercel Dashboard:   https://ecobot-ui.vercel.app/?robot=$TUNNEL_HOST"
fi
echo "=========================================="
echo "Press Ctrl+C to stop all services."

# Cleanup trap on exit
cleanup() {
  echo ""
  echo "Shutting down EcoBot stack..."
  kill $ROS_PID $TUNNEL_PID 2>/dev/null || true
  sleep 1
  pkill -f "$ECOBOT_PROC_PATTERN" 2>/dev/null || true
  if [ -n "$TUNNEL_PID" ]; then
    kill -9 $TUNNEL_PID 2>/dev/null || true
  fi
  exit 0
}

trap cleanup INT TERM
wait $ROS_PID
