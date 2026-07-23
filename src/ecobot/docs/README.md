# Ecobot Project Overview

Ecobot is a sophisticated ROS 2 Humble differential-drive mobile robot platform designed for autonomous indoor navigation, obstacle avoidance, object detection, and web-based teleoperation.

## Hardware Architecture

*   **Compute:** Jetson Nano/Orin (Main processing unit)
*   **Motor Control:** RP2040 (Pi Pico) connected via USB serial, handling low-level motor commands.
*   **Sensors:** Intel RealSense D415 depth camera providing RGB and depth streams.

## Software Architecture

The system is organized into 7 distinct ROS 2 packages:

### `ecobot_motor_control`
Handles serial communication with the Pico, differential-drive kinematics, and publishes odometry and TF transforms (`base_footprint` → `odom`).

**Nodes:**
- `motor_control_node` — Subscribes to `/cmd_vel` (Twist), converts to per-motor RPM via differential-drive kinematics, sends COBS-encoded serial packets to the Pico, reads back encoder feedback, and publishes `/odom`, `/joint_states`, `/run_mode`, and the odom→base_footprint TF transform.

**Configuration parameters:**
- `serial_port` (default: `/dev/ttyACM0`) — USB serial device
- `baudrate` (default: `115200`) — Serial baud rate
- `control_frequency` (default: `10.0` Hz) — Control loop rate
- `cmd_vel_timeout` (default: `0.5` s) — Timeout after which commands are zeroed
- `max_rpm` (default: `130.0`) — Maximum motor RPM
- `product_id` (default: `1`) — Robot product identifier for serial protocol
- `odom_frame_id` (default: `odom`) — Odometry frame
- `base_frame_id` (default: `base_footprint`) — Base frame

**Published topics:** `/odom` (Odometry), `/joint_states` (JointState), `/run_mode` (UInt8)
**Subscribed topics:** `/cmd_vel` (Twist)
**TF:** `odom` → `base_footprint`

**Kinematics models** (defined in `kinematics.py`):
- `CUGOV4_PARAMS`: tread=0.376m, wheel_radius=0.03858m, reduction=20:1, encoder_resolution=30
- `CUGOV3I_PARAMS`: tread=0.380m, wheel_radius=0.03858m, reduction=15:1, encoder_resolution=24
- `UNAJU_PARAMS`: tread=0.145m, wheel_radius=0.0284m, reduction=74.83:1, encoder_resolution=48

**Serial protocol** (defined in `serial_protocol.py`): 72-byte packets (8-byte header + 64-byte body) with COBS encoding, 16-bit checksum, containing left/right RPM commands and encoder feedback.

### `ecobot_sensors`
Manages the RealSense camera feed, object detection, obstacle avoidance, ground detection, and chair navigation.

**Nodes:**

- `realsense_feed` — Publishes RGB, depth, and aligned depth images from the Intel RealSense D415 at 640×480, 30 FPS. Publishes CameraInfo with intrinsics. Supports optional OpenCV viewer.
  - **Published topics:** `/camera/color/image_raw`, `/camera/depth/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`, `/camera/depth/camera_info`
  - **Parameters:** `color_width` (640), `color_height` (480), `color_fps` (30), `depth_width` (640), `depth_height` (480), `depth_fps` (30), `show_viewer` (False)

- `camera_webserver` — MJPEG HTTP server on port 8081 serving `/camera/color/image_raw` as `/stream.mjpg`.
  - **Parameters:** `port` (8081), `quality` (70), `topic` (`/camera/color/image_raw`)

- `webrtc_streamer` — WebRTC signaling server on port 8082 with a `/offer` POST endpoint for low-latency video streaming via `aiortc`.
  - **Parameters:** `signaling_port` (8082)

- `object_detection` — Runs SSD MobileNet (Caffe) inference on color images at a configurable rate. Publishes bounding box overlays and JSON detection results with estimated distances from depth data.
  - **Published topics:** `/detection_overlay` (Image), `/detections` (String/JSON)
  - **Subscribed topics:** `/camera/color/image_raw`, `/camera/depth/image_raw`
  - **Parameters:** `conf_threshold` (0.5), `inference_rate` (4, run every Nth frame), `model_path` (auto-detected from package share)

- `obstacle_avoidance` — Reactive obstacle avoidance using depth image zone analysis. Divides the depth image into configurable horizontal zones, computes mean distances, and publishes velocity commands to `/cmd_vel`. Supports Nav2 integration by subscribing to `/nav_cmd_vel` and scaling velocities. Includes a multi-state escape behavior (REVERSE → TURN → FORWARD) when all zones are blocked. Serves a debug MJPEG stream on port 8083.
  - **Subscribed topics:** `/nav_cmd_vel` (Twist), `/camera/depth/image_raw` (Image)
  - **Published topics:** `/cmd_vel` (Twist)
  - **Parameters:** `safe_distance` (0.9m), `warn_distance` (1.1m), `zones` (5), `max_linear_speed` (0.3), `max_angular_speed` (0.5), `reverse_speed` (-0.2), `reverse_duration` (1.5s), `turn_duration` (2.0s), `mjpeg_port` (8083), `depth_scale` (0.001)

- `depth_ground_detection` — Projects depth points into `base_footprint` frame using TF, filters obstacles by height (above ground clearance, below max obstacle height), and publishes a PointCloud2 of detected ground-level obstacles. Serves a debug MJPEG stream on port 8084.
  - **Published topics:** `/ground_obstacle_points` (PointCloud2), `/ground_debug_overlay` (Image)
  - **Subscribed topics:** `/camera/depth/image_raw`, `/camera/depth/camera_info`
  - **Parameters:** `camera_height` (0.508m), `ground_clearance` (0.02m), `max_obstacle_height` (0.50m), `min_obstacle_height` (0.005m), `max_range` (3.0m), `min_range` (0.2m), `downsample` (2), `mjpeg_port` (8084), `depth_scale` (0.001)

- `chair_navigator` — Listens to `/detections` JSON for "chair" class detections, computes 3D position via depth + camera intrinsics, transforms to `odom` frame, offsets the goal by `goal_offset` meters, and sends a `NavigateToPose` action goal to Nav2. Includes cooldown and navigation-in-progress gating.
  - **Published topics:** `/chair_goal` (PoseStamped), `/chair_detected` (Bool)
  - **Subscribed topics:** `/detections` (String), `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`
  - **Parameters:** `conf_threshold` (0.6), `goal_offset` (0.5m), `nav_timeout` (60.0s), `cooldown_sec` (5.0s)

### `ecobot_navigation`
Integrates with the Nav2 stack for autonomous path planning and waypoint following.

**Nodes:**
- `waypoint_follower` — Reads waypoints from a CSV file, publishes `/nav_cmd_vel` using a proportional controller with configurable gains.
  - **Parameters:** `waypoint_file` (path), `linear_gain` (0.5), `angular_gain` (1.5), `goal_tolerance` (0.15m), `max_linear_speed` (0.4), `max_angular_speed` (0.8)

**Nav2 stack** (launched via `navigation.launch.py`):
- `map_server` — Loads a pre-built map (YAML) when a map path is provided
- `amcl` — Adaptive Monte Carlo localization (only with a map)
- `controller_server` — Regulated Pure Pursuit controller, outputs to `/nav_cmd_vel`
- `planner_server` — NavFn planner (GridBased)
- `behavior_server` — Spin, backup, and wait recovery behaviors
- `bt_navigator` — Behavior Tree navigator with replanning and recovery
- `lifecycle_manager` — Manages Nav2 node lifecycle

**Nav2 configuration files:**
- `nav2_params.yaml` — Full SLAM-based navigation with `map` frame, static map layer, AMCL localization
- `nav2_params_mapless.yaml` — Exploration mode with `odom` global frame, rolling-window global costmap, no static layer, no AMCL

**Key Nav2 parameters:**
- Controller: Regulated Pure Pursuit, `desired_linear_vel` 0.3 m/s (mapped) / 0.2 m/s (mapless), `lookahead_dist` 0.4m
- Local costmap: 3×3m rolling window, 0.05m resolution, obstacle + inflation layers
- Global costmap: 20×20m, static + obstacle + inflation layers (mapped) or rolling window (mapless)
- Footprint: `[[-0.15, -0.25], [-0.15, 0.25], [0.15, 0.25], [0.15, -0.25]]`
- Planner: NavFn (GridBased), tolerance 0.15m
- Goal checker: xy tolerance 0.15m, yaw tolerance 0.25 rad

### `ecobot_bringup`
Contains master launch files and utility nodes.

**Nodes:**
- `cmd_vel_mux` — Simple relay from `/nav_cmd_vel` to `/cmd_vel`, enabling Nav2 to drive the robot when navigation is active.
- `send_goal` — CLI utility to send a `NavigateToPose` goal: `ros2 run ecobot_bringup send_goal <x> <y> <yaw>`

**Launch files:**
- `ecobot.launch.py` — Master launch with conditional flags: `enable_sensors`, `enable_teleop`, `enable_navigation`, `enable_mapping`, `enable_detection`, `enable_chair_navigation`, `enable_obstacle_avoidance`, `enable_dashboard`, `enable_rosbridge`, `enable_urdf`, `serial_port`, `map`
- `mapping.launch.py` — Dedicated mapping launch with RTAB-Map, sensors, depth-to-scan, dashboard, and rosbridge
- `rtabmap_mapping.launch.py` — RTAB-Map RGB-D SLAM with configurable visual odometry, database path, and scan topic
- `slam.launch.py` — `slam_toolbox` async node for 2D SLAM

**SLAM configuration** (`slam_params.yaml`):
- `slam_toolbox` async mode, 0.05m resolution, 8.0m max laser range, Ceres optimization (huber_scale=0.1, max 10 iterations)

**RTAB-Map launch arguments:**
- `rgb_topic`: `/camera/color/image_raw`
- `depth_topic`: `/camera/aligned_depth_to_color/image_raw`
- `scan_topic`: `/scan`
- `odom_topic`: `/odom`
- `frame_id`: `base_footprint`
- `approx_sync_max_interval`: 0.05s
- `Grid/Sensor 1`, `Grid/RangeMax 3.0`

### `ecobot_bringup`
Contains master launch files, `cmd_vel` mux configuration, and integration with RTAB-Map or `slam_toolbox` for mapping.

### `ecobot_dashboard`
Provides an aiohttp web UI on port 8080 featuring live camera feeds, odometry status, 2D/3D maps, detection results, and obstacle zone visualization. Communicates with the frontend via WebSocket.

**Nodes:**
- `dashboard_server` — aiohttp web server serving static files from `www/`, WebSocket endpoint at `/ws`, 2D map binary at `/map2d/data`, 3D point cloud at `/map3d/data`.
  - **Subscribed topics:** `/odom`, `/run_mode`, `/camera/color/image_raw`, `/detection_overlay`, `/camera/depth/image_raw`, `/detections`, `/rtabmap/cloud_map`, `/rtabmap/map`, `/rtabmap/grid_prob_map`
  - **Parameters:** `port` (8080), `depth_topic`, `camera_topic`, `detection_topic`, `num_zones` (5), `safe_distance` (0.9), `warn_distance` (1.1), `depth_scale` (0.001), `map3d_max_points` (80000)

### `ecobot_teleop`
Enables keyboard-based teleoperation.

**Nodes:**
- `keyboard_teleop` — Publishes `/cmd_vel` from keyboard input (w/s/a/d/space/q).
  - **Parameters:** `max_linear_speed` (0.5), `max_angular_speed` (1.0), `linear_step` (0.05), `angular_step` (0.1)

## Data Flow & Navigation Logic

1. **Sensing:** RealSense D415 captures 640×480 RGB and depth streams at 30 FPS.
2. **Depth Processing:** Depth data is converted into a virtual laser scan (`/scan`) via `depthimage_to_laserscan` (range 0.3–8.0m, scan_height=240). Ground-level obstacles are also detected as `/ground_obstacle_points` (PointCloud2).
3. **Object Detection:** SSD MobileNet runs every Nth frame on the color image, publishing JSON detections with class, confidence, bounding box, and estimated distance.
4. **Navigation:** Nav2 uses the scan and costmaps for path planning and obstacle avoidance, generating velocity commands (`/nav_cmd_vel`). The Regulated Pure Pursuit controller tracks the planned path.
5. **Obstacle Avoidance:** The reactive obstacle avoidance node divides the depth image into 5 horizontal zones, computes mean distances, and overrides Nav2 commands when obstacles are too close. It implements a three-state escape behavior (reverse → turn → forward) when all zones are blocked.
6. **Control:** Commands pass through the mux (`/nav_cmd_vel` → `/cmd_vel`) to the `ecobot_motor_control` serial driver.
7. **Execution:** Pico receives COBS-encoded 72-byte packets at 115200 baud and drives the motors. Encoder feedback is read back to compute odometry via dead reckoning.

## Mapping & SLAM

Mapping can be performed using:
- **RTAB-Map** (RGB-D) for 3D mapping with loop closure, publishing `/rtabmap/cloud_map` (PointCloud2) and `/rtabmap/map` (OccupancyGrid). Database stored at `/home/ecobot/map_data/rtabmap.db`.
- **`slam_toolbox`** (2D) for generating 2D occupancy grids with Ceres-based scan matching.

## Web Interfaces

| Service | Port | Endpoint | Description |
|---------|------|----------|-------------|
| Dashboard | 8080 | `/` | Main web UI (aiohttp + WebSocket) |
| Camera MJPEG | 8081 | `/stream.mjpg` | Raw color camera stream |
| WebRTC signaling | 8082 | `/offer` | Low-latency video (POST SDP offer) |
| Obstacle MJPEG | 8083 | `/obstacle.mjpg` | Obstacle avoidance debug view |
| Ground MJPEG | 8084 | `/ground.mjpg` | Ground detection debug view |
| rosbridge | 9090 | WebSocket | ROS bridge for external clients |

## TF Tree

```
odom → base_footprint  (published by motor_control_node, updated from encoder dead reckoning)
base_footprint → camera_depth_optical_frame  (static: x=0, y=0, z=0.508, quaternion 0.5 -0.5 0.5 0.5)
base_footprint → camera_color_optical_frame  (static: x=0, y=0, z=0.508, quaternion 0.5 -0.5 0.5 0.5)
camera_depth_optical_frame → camera_color_optical_frame  (static: x=0.018, y=0, z=0)
```

## Topic Map

```
/camera/color/image_raw          ← realsense_feed → /camera/depth/image_raw
                                                                    ↓
                                                          depthimage_to_laserscan
                                                                    ↓
                                                                  /scan
                                                                    ↓
                                                          Nav2 costmaps & planner
                                                                    ↓
                                                              /nav_cmd_vel
                                                                    ↓
                                                          obstacle_avoidance (overrides if danger)
                                                                    ↓
                                                          cmd_vel_mux → /cmd_vel
                                                                    ↓
                                                          motor_control_node → serial → Pico
```

## ROS 2 Node Summary

| Node | Package | Executable | Purpose |
|------|---------|------------|---------|
| `motor_control_node` | `ecobot_motor_control` | `motor_control_node` | Serial motor driver, odometry, TF |
| `realsense_feed` | `ecobot_sensors` | `realsense_feed` | RealSense D415 RGB-D capture |
| `camera_webserver` | `ecobot_sensors` | `camera_webserver` | MJPEG HTTP camera stream |
| `webrtc_streamer` | `ecobot_sensors` | `webrtc_streamer` | WebRTC low-latency video |
| `object_detection` | `ecobot_sensors` | `object_detection` | SSD MobileNet object detection |
| `obstacle_avoidance` | `ecobot_sensors` | `obstacle_avoidance` | Reactive depth-based avoidance |
| `depth_ground_detection` | `ecobot_sensors` | `depth_ground_detection` | Ground-level obstacle point cloud |
| `chair_navigator` | `ecobot_sensors` | `chair_navigator` | Chair detection → Nav2 goal |
| `waypoint_follower` | `ecobot_navigation` | `waypoint_follower` | CSV waypoint following |
| `dashboard_server` | `ecobot_dashboard` | `dashboard_server` | Web UI (aiohttp + WebSocket) |
| `keyboard_teleop` | `ecobot_teleop` | `keyboard_teleop` | Keyboard velocity control |
| `cmd_vel_mux` | `ecobot_bringup` | `cmd_vel_mux` | Nav2 → cmd_vel relay |
| `send_goal` | `ecobot_bringup` | `send_goal` | CLI Nav2 goal sender |

## Launch Flags

| Flag | Default | Description |
|------|---------|-------------|
| `enable_sensors` | `true` | Start RealSense and camera streams |
| `enable_teleop` | `false` | Start keyboard teleop |
| `enable_navigation` | `false` | Start Nav2 stack (with optional map) |
| `enable_mapping` | `false` | Start RTAB-Map RGB-D SLAM |
| `enable_detection` | `false` | Start SSD MobileNet object detection |
| `enable_chair_navigation` | `false` | Start chair detection + Nav2 goal |
| `enable_obstacle_avoidance` | `false` | Start reactive obstacle avoidance |
| `enable_dashboard` | `true` | Start web dashboard |
| `enable_rosbridge` | `true` | Start rosbridge WebSocket server |
| `enable_urdf` | `false` | Start robot_state_publisher |
| `enable_teleop` | `false` | Start keyboard teleop |
| `serial_port` | `/dev/ttyACM0` | Motor controller serial device |
| `map` | `""` | Path to YAML map file for Nav2 |
