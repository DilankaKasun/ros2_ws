# EcoBot Architecture & Node Connections

This document explains the full file structure, what each necessary file does, and how they connect with one another. It maps out the dependencies, imports, and ROS 2 topic communications across the workspace.

## 1. Starting Point: Who Drives

One node owns the wheels for a whole run: `ecobot_navigation/ecobot_navigation/plant_run_node.py`.
Two nodes used to steer at once — a camera-driven one and a map-driven
one — each telling the other to stand down, which is why runs were
impossible to follow. That is gone.

`plant_run_node` runs the sequence and hands the wheels between two
drivers, one at a time:

- the **map driver** (Nav2) crosses the room and stops **1.2 m short** of
  the plant, never at it;
- the **camera driver** does the last stretch and parks **0.65 m** out,
  square on, because the depth camera reads nothing closer than 0.50 m.

The handover is explicit: the Nav2 goal is cancelled and the node waits
for `/nav_cmd_vel` to actually stop arriving before the camera driver
publishes anything, because `obstacle_avoidance.py` picks its source by
which topic is still live.

**States, each with its own deadline:** SURVEY → PICK → REACQUIRE →
DRIVE → HANDOVER → APPROACH → SCAN → REPORT → TURN_AWAY → (next plant) →
DONE. Whichever driver holds the wheels is named on `/ecobot/nav_status`,
along with what it is doing and why it stopped.

### Why the survey stores directions, not places

The tracks scrub, so the turn the wheels report is about half the real
one (`motor_control_node.py`'s `turn_calibration` takes this out — it
must be measured per robot). Any residual error has a sharp consequence:
a sighting taken part-way through a sweep projects into the wheel frame
at the wrong angle, so the same plant lands somewhere different in every
frame it appears in, and one plant becomes several.

So the survey stores each plant as **the heading the wheels were
reporting when it saw it**, plus how far off it was — never a position.
A reported heading can be turned back to even when it is wrong, because
the error is a consistent scaling. REACQUIRE turns back to that heading,
waits for a fresh look with the plant ahead and the robot not turning,
and only then works out a position — which is used immediately to aim the
one drive that follows, and never kept across another turn.

## 1b. The Scan and Report Node

`ecobot_mission/ecobot_mission/plant_mission_node.py` no longer drives.
It scans, photographs and reports where `plant_run_node` has parked the
robot.

**What it listens to (Subscribes):**
- `/arm/scanner_status`: from `ecobot_arm_control/arm_scanner_node.py`.
- `/arm/camera/image_raw`: from `ecobot_arm_control/arm_camera_server.py`.
- `/ecobot/plant_scan_cmd`: from the dashboard over rosbridge, and from
  `plant_run_node` (its own messages carry `"from": "plant_run_node"`).
  It acts only on the scan-level commands — `scan_here`, `stop`, `pause`,
  `resume`, `set_samples`. The run-level ones — `start`, `stop`, `next` —
  belong to `plant_run_node`.

**What it commands (Publishes):**
- `/arm/scanner_cmd`, `/ecobot/plant_scan_status`, `/ecobot/scan_capture`.

A scan is over when the **arm** says it is over. The node's timeouts
measure how long the arm has been *silent*, never how long a sweep has
taken, so a slow sweep is never cut short and filed as finished.

---

## 2. Complete File Structure & Purpose

Here is how the main functional files map out across the packages:

### `ecobot_bringup/` (The Root Launcher)
- **`launch/ecobot.launch.py`**: The absolute root of the project. If you launch this, it dynamically imports and executes all the other `.launch.py` files below based on your parameters. 

### `ecobot_navigation/` (The Run, and the Wheels)
- **`launch/navigation.launch.py`**: Boots Nav2 (mapless — it plans in the
  wheel frame against a rolling obstacle picture from the depth camera)
  and `plant_run_node`.
- **`ecobot_navigation/plant_run_node.py`**: (Explained above.) Owns the
  wheels for the whole run.
- **`config/nav2_params_mapless.yaml`**: Nav2's settings. Goal tolerances
  are deliberately loose — the map driver only has to finish roughly
  1.2 m out, and the camera driver does the precise part.

### `ecobot_mission/` (Scan, Photograph, Report)
- **`launch/mission.launch.py`**: Boots the mission node.
- **`ecobot_mission/plant_mission_node.py`**: (Explained above.) Scans and
  reports; it does not drive.
- **`ecobot_mission/gemini_client.py`**: A helper class that formats images and makes HTTP requests to Google's Gemini API.

### `ecobot_motor_control/` (Base Movement)
- **`launch/motor_control.launch.py`**: Boots the motor node.
- **`ecobot_motor_control/motor_control_node.py`**: 
  - Subscribes to `/cmd_vel` (or `/nav_cmd_vel`). 
  - Converts twist messages into serial commands sent over USB (`/dev/ttyACM0`) to the microcontroller driving the wheels.
  - Reads serial wheel-tick data and publishes `/odom` (odometry).
  - `turn_calibration` corrects the turn the encoders report, which the
    tracks' scrubbing makes about half the real one. **Measure it per
    robot:** turn the robot a known amount by hand — a full circle back to
    a floor mark is easiest — and divide the true turn by what `/odom`
    reported. Everything that steers closes its loop on this reading, so
    leaving the error in makes the wheel frame a differently shaped world
    that no planner can work in.

### `ecobot_sensors/` (Perception & AI)
- **`launch/sensors.launch.py`**: Boots the camera, lidar, and detection nodes.
- **`ecobot_sensors/realsense_feed.py`**: Connects to the RealSense hardware and publishes raw video (`/camera/color/image_raw`, `/camera/depth/image_raw`).
- **`ecobot_sensors/obstacle_avoidance.py`**: 
  - Subscribes to `/nav_cmd_vel`, `/goto_cmd_vel`, and `/camera/depth/image_raw`.
  - Analyzes depth for immediate collisions. If safe, it forwards velocities to `/cmd_vel` for the motor node. If blocked, it publishes zero velocity to prevent crashing.
- **`ecobot_sensors/yolo_detection.py`**: Subscribes to the camera, runs a YOLOv8 neural network, and publishes bounding boxes (used by the dashboard and tracking).
- **`ecobot_sensors/livekit_streamer.py`**: Grabs camera topics and publishes them to LiveKit so remote users can see the video feed.

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


---

## 3. How Data Flows Through the System (Example Scenario)

Let's trace a full execution flow when you click **"Start Mission"** on the Dashboard:

1. **User (Browser):** Clicks "Start Mission".
2. **Dashboard (ecobot-ui via rosbridge):** Publishes `{"action": "start"}` to `/ecobot/plant_scan_cmd`.
3. **Run Node (`plant_run_node.py`):** Takes the command. `plant_mission_node` ignores it — `start` is run-level.
4. **SURVEY:** The robot turns on the spot, listing every plant it sees on `/ecobot/detections` as a heading and a distance. It acts as soon as it has one and either the sweep looks complete or the time limit expires — it never waits for a confirmed full circle.
5. **PICK / REACQUIRE:** It picks the plant needing the least turning, turns back to that heading, and waits for a fresh look. Only now, facing the plant and not turning, does it work out where the plant actually is.
6. **DRIVE:** It sends `/navigate_to_pose` a goal **1.2 m short** of the plant. Nav2 publishes to `/nav_cmd_vel`. The camera driver publishes a zero throughout, so if Nav2 falls silent the robot stops rather than being nudged along.
7. **HANDOVER:** At 1.2 m the goal is cancelled and the node waits for `/nav_cmd_vel` to stop arriving. Until it does, the safety layer would ignore the camera driver.
8. **APPROACH:** The camera driver closes in on `/goto_cmd_vel`, keeping the plant centred and slowing as the gap shrinks, and parks 0.65 m out. This is the only stretch where `/ecobot/goto_suppress_avoidance` goes true — the thing filling the view here is the goal, not a hazard. Everywhere else avoidance stays fully on.
9. **Obstacle Node (`obstacle_avoidance.py`):** Reads whichever of `/nav_cmd_vel` and `/goto_cmd_vel` is live and forwards to `/cmd_vel`.
10. **Motor Node (`motor_control_node.py`):** Drives the wheels, and publishes the corrected `/odom`.
11. **SCAN:** The robot holds dead still — a zero every tick, not silence — and asks `plant_mission_node` for `scan_here`.
12. **Mission Node:** Publishes `scan` to `/arm/scanner_cmd`.
13. **Arm Scanner (`arm_scanner_node.py`):** Sweeps from level with the plant to looking down on it, pausing at each viewpoint, publishing `/arm/pose_goal` and its progress on `/arm/scanner_status`.
14. **Mission Node:** Grabs a wrist-camera frame at each viewpoint and publishes it to `/ecobot/scan_capture` for the dashboard.
15. **The robot does not move until the ARM reports idle.** Nothing else counts as finished.
16. **Mission Node:** Sends the photographs to `gemini_client.py`, and publishes the health result on `/ecobot/plant_scan_status`.
17. **REPORT → TURN_AWAY → next plant:** The run node waits for the report, turns away, and picks a plant it has not done — never re-picking the one it is parked in front of. When none are left it stops and says the run is complete.

At every step the run node says on `/ecobot/nav_status` which driver has
the wheels, what it is doing, and how long it has before its deadline. A
state that runs out of time says so; nothing reports success it did not
achieve.
