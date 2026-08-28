# EcoBot Object Detection & Dashboard Integration Plan

## Overview
This document outlines the plan to implement real-time object detection using YOLOv8 on the robot's Jetson Nano and integrate the results with the web dashboard.

## Phase 1: Object Detection (Jetson Nano)
1.  **Model:** Utilize YOLOv8n (Nano) for optimal performance on the Jetson Nano.
2.  **Acceleration:** Convert the model to TensorRT format to leverage GPU acceleration.
3.  **ROS 2 Node:** Enhance the existing `yolo_detection.py` node in `ecobot_sensors`.
    *   Subscribe to RealSense color and depth image topics.
    *   Perform inference using the TensorRT YOLO model.
    *   Calculate the 3D position (x, y, z) of each detected object relative to the camera using depth data and camera intrinsics.
    *   Publish detections as a custom ROS 2 message (e.g., `ecobot_msgs/Detections`) containing: class name, confidence, and 3D pose.

## Phase 2: Dashboard Integration
1.  **Data Bridge:** Use `rosbridge_suite` to expose the ROS 2 detection topic via WebSockets.
2.  **Dashboard Display:** Update the web dashboard to:
    *   Connect to the WebSocket.
    *   Subscribe to the detection topic.
    *   Display a live list of detected objects with their coordinates.
    *   (Optional) Visualize object positions on a 2D map.

## Phase 3: Goal Setting
1.  **Dashboard Input:** Allow the user to select an object from the list on the dashboard.
2.  **Navigation Goal:** Send the selected object's coordinates as a goal to the Nav2 stack to drive the robot to that location.
