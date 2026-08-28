# EcoBot Project Instructions & Overview

Welcome to the EcoBot ROS 2 project workspace. This document provides a comprehensive overview of the project structure, how to configure and set up each node, and methods to test individual components of the system.

## 1. Project Structure

The project is built within a standard ROS 2 workspace (`ros2_ws`). The core packages are located in `src/ecobot/`. The modular design allows different components (motors, sensors, arm, navigation, voice) to run independently or together via a master bringup launch file.

### Core Packages (`src/ecobot/`):

- **`ecobot_bringup`**: The main entry point. Contains `ecobot.launch.py` which conditionally launches all other necessary packages and nodes (motors, sensors, mapping, navigation, arm control).
- **`ecobot_motor_control`**: Interfaces with the mobile base motors over serial (usually `/dev/ttyACM0`). Handles velocity commands and robot kinematics.
- **`ecobot_sensors`**: Manages all sensor data inputs:
  - Intel RealSense camera (Color, Depth, PointClouds).
  - ESP32 Time-of-Flight (ToF) sensors (via Arduino firmware in `esp32_tof_sensors`).
  - WebRTC and LiveKit streaming for remote viewing.
  - Obstacle avoidance and object detection (YOLO).
  - Depth-to-LaserScan conversion for SLAM.
- **`ecobot_navigation`**: Integrates the standard ROS 2 `Nav2` stack for autonomous navigation, path planning, and obstacle avoidance.
- **`ecobot_arm_control`**: Controls the robotic arm via an I2C PCA9685 PWM driver. Supports manual joint control, inverse kinematics, and Vision-Language-Action (VLA) models (like Minicpm and OpenVLA).
- **`ecobot_voice`**: A LiveKit-based voice agent worker that provides an interactive voice assistant interface.
- **`ecobot_mission`**: Contains high-level behavior and mission scripts (e.g., `plant_mission_node`), acting as a state machine for autonomous operations.
- **`ecobot_dashboard`**: A web-based dashboard server (default port `8080`) providing a UI to monitor and control the robot.

---

## 2. Configuration & Setup

Most packages are configured using ROS 2 `launch` files and `yaml` parameters. The primary way to configure the system is by passing arguments to `ecobot.launch.py` in `ecobot_bringup`.

### General Bringup
To launch the entire system with specific components enabled:
```bash
ros2 launch ecobot_bringup ecobot.launch.py \
    enable_sensors:=true \
    enable_arm:=true \
    enable_navigation:=true \
    enable_diagnostics:=true \
    serial_port:=/dev/ttyACM0
```

### Sub-System Configurations:

- **Motor Control (`ecobot_motor_control`)**
  - **Node**: `motor_control_node`
  - **Parameters**: 
    - `serial_port` (default: `/dev/ttyACM0`)
    - `baudrate` (default: `115200`)
    - `max_rpm` (default: `130.0`)
  - **Setup**: Ensure the motor controller is plugged into USB and you have read/write permissions for `/dev/ttyACM0` (e.g., `sudo usermod -aG dialout $USER`).

- **Arm Control (`ecobot_arm_control`)**
  - **Node**: `arm_manual_node`
  - **Parameters**:
    - `i2c_bus` (default: `7`)
    - `pca9685_address` (default: `0x40` or 64)
  - **Setup**: Ensure I2C is enabled and the PCA9685 is wired correctly. Check I2C devices using `sudo i2cdetect -y -r 7`.

- **Sensors & Vision (`ecobot_sensors`)**
  - **Node Launch**: `sensors.launch.py`
  - **Arguments**: 
    - `enable_obstacle_avoidance` (true/false)
    - `enable_webrtc` / `enable_livekit` (true/false)
    - `enable_legacy_detection` (true/false for YOLO)
  - **Setup**: Ensure the RealSense camera is connected. The node uses `cv_bridge` and `image_transport` to process and publish compressed streams.

- **Dashboard (`ecobot_dashboard`)**
  - **Node**: `dashboard_server`
  - **Setup**: By default, it runs on port `8080`. Start it via the launch file and navigate to `http://localhost:8080` in your web browser.

- **Voice Agent (`ecobot_voice`)**
  - **Node Launch**: `voice_agent.launch.py`
  - **Setup**: Ensure you have LiveKit configured and environment variables set (like `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`).

---

## 3. Testing Methods for Each Component

Before testing, remember to build your workspace and source it:
```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Testing the Base Motors
If obstacle avoidance is **disabled**, you can publish directly to `/cmd_vel`:
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1
```
*(Use `Ctrl+C` to stop. Change `-1` to continuous publishing or hit enter multiple times if there's a timeout).*

If obstacle avoidance is **enabled**, publish to the navigation topic so the obstacle node processes it:
```bash
ros2 topic pub /nav_cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1, y: 0.0, z: 0.0}}" -1
```

### Testing Sensors (RealSense & Lidar)
1. **Camera Stream**:
   Open `rqt_image_view` to verify the video feeds:
   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```
   Select `/camera/color/image_raw` or `/camera/depth/image_raw` from the dropdown.

2. **Lidar/Scan (Depth-to-Scan)**:
   ```bash
   ros2 topic echo /scan
   ```
   Alternatively, open `rviz2` and add a `LaserScan` display listening to the `/scan` topic. Ensure your fixed frame is set to `base_footprint` or `odom`.

### Testing the Robotic Arm
Ensure the arm node is running (`ros2 launch ecobot_arm_control arm_control.launch.py`).
1. **Enable the Arm**:
   ```bash
   ros2 topic pub /arm/enable std_msgs/msg/String "{data: 'enable'}" -1
   ```
2. **Send Joint Commands**:
   Assuming it's a 6-DOF arm, publish an array of radians (e.g., move joint 2 and 3):
   ```bash
   ros2 topic pub /arm/joint_commands std_msgs/msg/Float64MultiArray "{data: [0.0, 1.57, -1.57, 0.0, 0.0, 0.0]}" -1
   ```
3. **Check Arm Status**:
   ```bash
   ros2 topic echo /arm/status
   ```

### Testing Navigation (Nav2 Stack)
1. Launch the full bringup with navigation:
   ```bash
   ros2 launch ecobot_bringup ecobot.launch.py enable_navigation:=true
   ```
2. Open RViz2:
   ```bash
   rviz2 -d $(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/rviz/nav2_default_view.rviz
   ```
3. Use the **"2D Pose Estimate"** tool to initialize the robot's location.
4. Use the **"Nav2 Goal"** tool to select a destination on the map. The robot should automatically compute a path and begin moving.

### Testing the Dashboard
1. Ensure the dashboard is running:
   ```bash
   ros2 launch ecobot_dashboard dashboard.launch.py
   ```
2. Open a web browser on your PC/network and go to `http://<ROBOT_IP>:8080/`. You should see the video feeds, robot pose, and control widgets.

### Testing Voice / WebRTC
- **MJPEG Streams**: Access directly via browser for debugging:
  - General Camera: `http://<ROBOT_IP>:8081/`
  - Obstacle Avoidance Camera: `http://<ROBOT_IP>:8086/obstacle.mjpg`
- **Voice Agent**: If LiveKit is set up, interact via the LiveKit web client frontend matching your LiveKit server configuration. Check logs using:
  ```bash
  ros2 run ecobot_voice voice_agent dev
  ```
