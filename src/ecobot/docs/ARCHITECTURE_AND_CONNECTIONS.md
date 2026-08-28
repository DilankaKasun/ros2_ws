# EcoBot Architecture & Node Connections

This document explains the full file structure, what each necessary file does, and how they connect with one another. It maps out the dependencies, imports, and ROS 2 topic communications across the workspace.

## 1. Starting Point: The Mission Orchestrator
If we look at `ecobot_mission/ecobot_mission/plant_mission_node.py` as our anchor point, here is how it connects to the rest of the workspace:

### Python Imports (Local to the package)
- `from .gemini_client import GeminiClient`: Imports the AI client helper located in the same directory (`gemini_client.py`). This is used to analyze the pictures captured by the arm.

### ROS 2 Topic Connections (Inter-package communication)
Instead of importing Python files from other packages directly, `plant_mission_node.py` communicates with other files by publishing and subscribing to ROS 2 topics:

**What it listens to (Subscribes):**
- `/arm/scanner_status`: Published by `ecobot_arm_control/arm_scanner_node.py`. Tells the mission node if the arm has finished moving to the next scanning viewpoint.
- `/arm/camera/image_raw`: Published by `ecobot_arm_control/arm_camera_server.py`. The mission node grabs frames from this topic to send to the Gemini API.
- `/ecobot/plant_scan_cmd`: Published by `ecobot_dashboard/dashboard_server.py` (when a user clicks UI buttons).
- `/ecobot/waypoints`: Published by `ecobot_dashboard/dashboard_server.py`.

**What it commands (Publishes):**
- `/arm/scanner_cmd`: Read by `ecobot_arm_control/arm_scanner_node.py` to trigger the multi-angle plant scanning sequence.
- `/ecobot/plant_scan_status`: Read by `ecobot_dashboard/dashboard_server.py` to update the web UI with the current mission state.
- `/ecobot/scan_capture`: Read by `ecobot_dashboard/dashboard_server.py` to display the AI's health assessment of the plant on the dashboard.

**Action Clients:**
- `/navigate_to_pose`: Sent to the `Nav2` stack (`ecobot_navigation`) to autonomously drive the robot base to the plant coordinates.

---

## 2. Complete File Structure & Purpose

Here is how the main functional files map out across the packages:

### `ecobot_bringup/` (The Root Launcher)
- **`launch/ecobot.launch.py`**: The absolute root of the project. If you launch this, it dynamically imports and executes all the other `.launch.py` files below based on your parameters. 

### `ecobot_mission/` (High-Level Behavior)
- **`launch/mission.launch.py`**: Boots the mission node.
- **`ecobot_mission/plant_mission_node.py`**: (Explained above). The brain of the operation.
- **`ecobot_mission/gemini_client.py`**: A helper class that formats images and makes HTTP requests to Google's Gemini API.

### `ecobot_motor_control/` (Base Movement)
- **`launch/motor_control.launch.py`**: Boots the motor node.
- **`ecobot_motor_control/motor_control_node.py`**: 
  - Subscribes to `/cmd_vel` (or `/nav_cmd_vel`). 
  - Converts twist messages into serial commands sent over USB (`/dev/ttyACM0`) to the microcontroller driving the wheels.
  - Reads serial wheel-tick data and publishes `/odom` (odometry).

### `ecobot_sensors/` (Perception & AI)
- **`launch/sensors.launch.py`**: Boots the camera, lidar, and detection nodes.
- **`ecobot_sensors/realsense_feed.py`**: Connects to the RealSense hardware and publishes raw video (`/camera/color/image_raw`, `/camera/depth/image_raw`).
- **`ecobot_sensors/obstacle_avoidance.py`**: 
  - Subscribes to `/nav_cmd_vel`, `/goto_cmd_vel`, and `/camera/depth/image_raw`.
  - Analyzes depth for immediate collisions. If safe, it forwards velocities to `/cmd_vel` for the motor node. If blocked, it publishes zero velocity to prevent crashing.
- **`ecobot_sensors/yolo_detection.py`**: Subscribes to the camera, runs a YOLOv8 neural network, and publishes bounding boxes (used by the dashboard and tracking).
- **`ecobot_sensors/livekit_streamer.py` / `webrtc_streamer.py`**: Grabs camera topics and pushes them to WebRTC servers so remote users can see the video feed.

### `ecobot_arm_control/` (Manipulation)
- **`launch/arm_control.launch.py`**: Boots the arm drivers and automation nodes.
- **`ecobot_arm_control/arm_manual_node.py`**: 
  - The core hardware interface. Subscribes to `/arm/pose_goal` and `/arm/joint_commands`. 
  - Uses `smbus2` to talk to the PCA9685 I2C PWM board to physically move the servos.
- **`ecobot_arm_control/arm_scanner_node.py`**: 
  - Subscribes to `/arm/scanner_cmd` (from `plant_mission_node.py`).
  - Calculates pre-defined scanning angles around a target.
  - Publishes `/arm/pose_goal` to command the `arm_manual_node.py`.
- **`ecobot_arm_control/minicpm_vla_node.py` / `openvla_node.py`**: Experimental Vision-Language-Action nodes. They subscribe to the camera feed and text prompts, run an AI model, and output `/arm/pose_goal` commands to move the arm intelligently.

### `ecobot_dashboard/` (User Interface)
- **`launch/dashboard.launch.py`**: Boots the web server.
- **`ecobot_dashboard/dashboard_server.py`**: 
  - Subscribes to *almost everything* (`/odom`, `/arm/status`, `/ecobot/plant_scan_status`, image topics).
  - Translates ROS 2 data into WebSockets JSON to send to the browser.
  - Listens for WebSocket commands from the browser and publishes them to ROS 2 topics (like `/cmd_vel` for manual driving or `/ecobot/plant_scan_cmd` to start a mission).
- **`www/index.html` & `www/main.js`**: The frontend UI.

---

## 3. How Data Flows Through the System (Example Scenario)

Let's trace a full execution flow when you click **"Start Mission"** on the Dashboard:

1. **User (Browser):** Clicks "Start Mission" in `index.html`.
2. **Dashboard (`dashboard_server.py`):** Receives the WebSocket event and publishes `{"data": "start"}` to `/ecobot/plant_scan_cmd`.
3. **Mission Node (`plant_mission_node.py`):** Receives the command. It pulls the first waypoint from its list and sends an Action Goal to `/navigate_to_pose`.
4. **Nav2 (`ecobot_navigation`):** Calculates the path and starts publishing velocity commands to `/nav_cmd_vel`.
5. **Obstacle Node (`obstacle_avoidance.py`):** Reads `/nav_cmd_vel` and checks `realsense_feed.py`'s depth map. If clear, it forwards the velocity to `/cmd_vel`.
6. **Motor Node (`motor_control_node.py`):** Reads `/cmd_vel`, translates it to serial bytes, and drives the hardware wheels.
7. *(Robot arrives at the plant)*
8. **Mission Node:** Sees navigation succeeded. Publishes `{"data": '{"command": "scan", "target": ...}'}` to `/arm/scanner_cmd`.
9. **Arm Scanner (`arm_scanner_node.py`):** Calculates angles and publishes to `/arm/pose_goal`.
10. **Arm Hardware (`arm_manual_node.py`):** Reads `/arm/pose_goal` and moves the servos via I2C.
11. **Arm Scanner:** Detects the move is complete and publishes `ready` to `/arm/scanner_status`.
12. **Mission Node:** Hears the `ready` status. Subscribes to `/arm/camera/image_raw`, grabs a frame, and passes it to `gemini_client.py`.
13. **Gemini Client:** Makes a cloud HTTP request to analyze the plant and returns the text result.
14. **Mission Node:** Publishes the AI response to `/ecobot/scan_capture`.
15. **Dashboard:** Reads `/ecobot/scan_capture` and pushes it over WebSockets to `index.html` to display to the user.
