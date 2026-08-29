from launch import LaunchDescription
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_obstacle_avoidance', default_value='false'),
        DeclareLaunchArgument('enable_livekit', default_value='true'),
        DeclareLaunchArgument('enable_detection', default_value='false'),
        DeclareLaunchArgument('enable_legacy_detection', default_value='false'),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_depth_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0.508',
                       '--qx', '0.5', '--qy', '-0.5', '--qz', '0.5', '--qw', '0.5',
                       '--frame-id', 'base_footprint', '--child-frame-id', 'camera_depth_optical_frame'],
            output='screen',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_color_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0.508',
                       '--qx', '0.5', '--qy', '-0.5', '--qz', '0.5', '--qw', '0.5',
                       '--frame-id', 'base_footprint', '--child-frame-id', 'camera_color_optical_frame'],
            output='screen',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_to_color_tf',
            arguments=['--x', '0.018', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                       '--frame-id', 'camera_depth_optical_frame', '--child-frame-id', 'camera_color_optical_frame'],
            output='screen',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_camera_link_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0.508',
                       '--qx', '0.5', '--qy', '-0.5', '--qz', '0.5', '--qw', '0.5',
                       '--frame-id', 'base_footprint', '--child-frame-id', 'camera_link'],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='realsense_feed',
            name='realsense_feed',
            parameters=[{
                'show_viewer': False,
            }],
            output='screen',
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='realsense_color_compressed_republish',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/camera/color/image_raw'),
                ('out/compressed', '/camera/color/image_raw/compressed'),
            ],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='livekit_streamer',
            name='livekit_streamer',
            condition=IfCondition(
                LaunchConfiguration('enable_livekit')),
            output='screen',
        ),
        # Carries control and telemetry over the same LiveKit room the video
        # uses, so a remote dashboard needs no inbound path to the robot.
        Node(
            package='ecobot_sensors',
            executable='livekit_bridge',
            name='livekit_bridge',
            condition=IfCondition(
                LaunchConfiguration('enable_livekit')),
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='obstacle_avoidance',
            name='obstacle_avoidance',
            condition=IfCondition(
                LaunchConfiguration('enable_obstacle_avoidance')),
            parameters=[{
                'safe_distance': 0.9,
                'show_viewer': False,
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='object_detection',
            name='object_detection',
            condition=IfCondition(
                LaunchConfiguration('enable_legacy_detection')),
            parameters=[{
                'conf_threshold': 0.5,
                'inference_rate': 4,
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='depth_ground_detection',
            name='depth_ground_detection',
            condition=IfCondition(
                LaunchConfiguration('enable_obstacle_avoidance')),
            parameters=[{
                'camera_height': 0.508,
                'ground_clearance': 0.02,
                'max_obstacle_height': 0.50,
                'downsample': 2,
                'max_range': 3.0,
                'mjpeg_port': 8086,
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='yolo_detection',
            name='yolo_detection',
            condition=IfCondition(
                LaunchConfiguration('enable_legacy_detection')),
            parameters=[{
                'model_path': '/home/ecobot/ros2_ws/src/ecobot/ecobot_sensors/models/yolov8n.engine',
                'conf_threshold': 0.5,
                'inference_rate': 2,
                'use_small_obstacle_filter': True,
                'mjpeg_port': 8087,
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='ecobot_detection_node',
            name='ecobot_detection_node',
            condition=IfCondition(
                LaunchConfiguration('enable_detection')),
            parameters=[{
                'model_path': '/home/ecobot/ros2_ws/src/ecobot/ecobot_sensors/models/yolov8n.engine',
                # ultralytics' AutoBackend fails this .engine at inference time
                # (TRT Error Code 1: Cask convolution execution on every frame)
                # even though the same file runs cleanly under trtexec and under
                # this node's own TrtBackend. Force the working path.
                'backend': 'tensorrt',
                'conf_threshold': 0.5,
                'inference_rate': 2,
                'depth_topic': '/camera/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'camera_frame': 'camera_color_optical_frame',
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='tof_sensors',
            name='tof_sensors',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baud': 115200,
                'http_url': '',
                'http_interval': 0.2,
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='detection_goto',
            name='detection_goto',
            condition=IfCondition(
                LaunchConfiguration('enable_detection')),
            parameters=[{
                'stop_distance': 0.4,
                'max_linear': 0.25,
                'max_angular': 0.8,
                'search_timeout': 20.0,
                'search_speed': 0.25,
                'avoid_distance': 0.4,
                'avoid_angle': 1.05,
                'use_map_frame': False,
                'blind_approach_limit': 1.0,
            }],
            output='screen',
        ),
    ])
