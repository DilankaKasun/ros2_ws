# Task: Detect Chair and Navigate to It

1.  **Object Detection:**
    *   Use the existing camera feed.
    *   Implement or use a pre-trained object detection model (e.g., YOLO, SSD) trained to detect "chair".
    *   Publish the bounding box and class confidence of detected chairs.

2.  **Coordinate Transformation:**
    *   Determine the chair's position relative to the robot using depth data (if available) or bounding box size.
    *   Transform this relative position into the global map frame or `odom` frame using TF transforms.

3.  **Navigation:**
    *   Set the transformed chair position as a goal for Nav2.
    *   Command the robot to navigate to this goal location.
    *   Stop the robot once it reaches the vicinity of the chair.
