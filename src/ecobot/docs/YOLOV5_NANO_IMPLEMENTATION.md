# Task: Implement Light-weight Object Detection (YOLOv5 Nano)

1.  **Model Selection:**
    *   Replace the existing YOLOv8 model with YOLOv5 Nano.
    *   Ensure the model is optimized for Jetson Nano (e.g., using TensorRT if possible).

2.  **Integration:**
    *   Update the object detection node to load the YOLOv5 Nano model.
    *   Maintain the existing input camera feed and output topic structure (bounding boxes, confidence, "chair" class).

3.  **Performance Tuning:**
    *   Verify that the inference speed is acceptable on the Jetson Nano.
    *   If necessary, adjust input image resolution to balance speed and accuracy.
