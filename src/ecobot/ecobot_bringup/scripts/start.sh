#!/bin/bash
set -e

echo "=== ecobot startup ==="

# Source environments
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash 2>/dev/null || true

# Kill existing processes
pkill -f "motor_control_node|rosbridge|dashboard_server" 2>/dev/null || true
sleep 1

# Set serial port
SERIAL_PORT="${1:-/dev/ttyACM0}"
ENABLE_PORTAL="${2:-false}"
ENABLE_VOICE="${3:-false}"

# Source LiveKit env
if [ -f "$HOME/ros2_ws/src/ecobot/ecobot_bringup/.env" ]; then
  set -a
  . "$HOME/ros2_ws/src/ecobot/ecobot_bringup/.env"
  set +a
fi

# Launch the full stack
ros2 launch ecobot_bringup ecobot.launch.py \
  serial_port:="$SERIAL_PORT" \
  use_sim_time:=false \
  enable_sensors:=true \
  enable_teleop:=false \
  enable_navigation:=true \
  enable_dashboard:=true \
  enable_rosbridge:=true \
  enable_urdf:=false \
  enable_obstacle_avoidance:=true \
  enable_portal:="$ENABLE_PORTAL" \
  enable_voice_agent:="$ENABLE_VOICE" &

sleep 4

echo "=== ecobot running ==="
echo "  Dashboard:   http://localhost:8080"
echo "  Rosbridge:   ws://localhost:9090"
echo "  Serial port: $SERIAL_PORT"
echo "  Portal:      $ENABLE_PORTAL"
echo "  Voice agent: $ENABLE_VOICE"
echo "  Press Ctrl+C to stop"

trap "echo 'Shutting down...'; pkill -f 'motor_control_node|rosbridge|dashboard_server|robot_portal' 2>/dev/null; exit 0" INT TERM
wait
