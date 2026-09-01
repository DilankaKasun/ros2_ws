# Instructions for Installing Cartographer from Source on ROS 2 Humble

Since official binary packages for Google Cartographer are not available for ROS 2 Humble, it must be built from source. Follow these steps to install it in your workspace.

## Prerequisites

Ensure you have your ROS 2 Humble environment sourced.

## Installation Steps

Execute the following commands in your terminal:

```bash
# 1. Create a new workspace or use existing. Here we use a dedicated one.
mkdir -p ~/cartographer_ws/src
cd ~/cartographer_ws

# 2. Download the Cartographer ROS repositories
vcs import src --input https://raw.githubusercontent.com/cartographer-project/cartographer_ros/master/cartographer_ros.repos

# 3. Install dependencies using rosdep
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y

# 4. Build the workspace
colcon build --symlink-install
```

## Sourcing the Workspace

After a successful build, source the overlay to use Cartographer:a

```bash
source ~/cartographer_ws/install/setup.bash
```

## Implementation Plan for Ecobot

To integrate Cartographer with the existing Ecobot platform, follow these steps:

1.  **Configure Topics:** Ensure Cartographer subscribes to the virtual laser scan topic (`/scan`) generated from the RealSense depth camera.
2.  **Update Transforms (TF):** Cartographer will provide the `map` -> `odom` transform. Ensure the existing `ecobot_motor_control` node continues to provide `odom` -> `base_footprint`.
3.  **Launch Integration:** Create a new launch file in `ecobot_bringup` that starts the Cartographer node with the appropriate parameters for indoor SLAM, replacing `slam_toolbox` or RTAB-Map when active.
4.  **Navigation Stack:** Update Nav2 configuration to use the published `/map` topic from Cartographer for global path planning.
5.  **Enhanced Obstacle Logic:** Modify the `ecobot_sensors/obstacle_avoidance` node. Instead of just reacting to depth zones, it should also subscribe to the real-time map published by Cartographer. This allows the robot to make decisions based on both immediate sensor data (the depth zones) and the accumulated map knowledge, preventing it from entering narrow spaces where it might get stuck due to camera FOV limitations.
