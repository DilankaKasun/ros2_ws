# Ecobot Project Overview

## Current System Summary
The Ecobot project is a ROS 2 Humble based mobile robot platform.

- **Core Functionality:** Autonomous navigation using the Nav2 stack.
- **Hardware Interface:** Communicates with an RP2040 motor controller (Raspberry Pi Pico) via USB serial.
- **Sensors:** Utilizes Lidar (`/scan`) and Depth Camera (`/camera/depth/image_raw`) for obstacle detection and avoidance.
- **Navigation:** Uses `RegulatedPurePursuitController` and costmaps to plan paths and avoid obstacles.

## Development Goal: Enhanced Small Obstacle Avoidance

**Problem:** The current sensor configuration only detects obstacles between 0.05m and 1.0m height. Small objects lower than 5cm are missed.

**Task:** Modify the system to detect and avoid these smaller objects.

### Proposed Solutions:
1. Modify Existing Sensors: Adjust the angle of the existing depth camera to view the ground closer to the robot.
2. Optimize Software: Implement a lightweight object detection model (like YOLOv8) optimized for the Jetson Nano Orin.

## Development Goal: Real-time 3D Mapping on Dashboard

**Task:** Generate a real-time 3D map from the RealSense D415 depth data and display it on the web dashboard.

### Feasibility Analysis & Required Work:
- Sensor Data: The current `realsense_feed.py` publishes only raw Image messages. It must be modified to publish `PointCloud2` data, or switched to the official `realsense2_camera` package.
- Mapping Backend: A 3D SLAM or mapping package like `rtabmap_ros` or `octomap_ros` needs to be installed and integrated into the launch pipeline.
- Dashboard Backend: `dashboard_server.py` needs to subscribe to the 3D map data and stream it efficiently via WebSocket.
- Dashboard Frontend: `index.html` requires a 3D WebGL renderer (e.g., Three.js or ROS3D.js) to visualize the incoming map data.
