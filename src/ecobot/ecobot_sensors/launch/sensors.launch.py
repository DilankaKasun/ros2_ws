from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_obstacle_avoidance', default_value='false'),
        DeclareLaunchArgument('enable_webrtc', default_value='false'),
        DeclareLaunchArgument('enable_detection', default_value='false'),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_depth_tf',
            arguments=['0', '0', '0.508', '0.5', '-0.5', '0.5', '0.5',
                       'base_footprint', 'camera_depth_optical_frame'],
            output='screen',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_color_tf',
            arguments=['0', '0', '0.508', '0.5', '-0.5', '0.5', '0.5',
                       'base_footprint', 'camera_color_optical_frame'],
            output='screen',
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_to_color_tf',
            arguments=['0.018', '0', '0', '0', '0', '0', '1',
                       'camera_depth_optical_frame', 'camera_color_optical_frame'],
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
            package='ecobot_sensors',
            executable='webrtc_streamer',
            name='webrtc_streamer',
            condition=IfCondition(
                LaunchConfiguration('enable_webrtc')),
            parameters=[{
                'signaling_port': 8082,
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='camera_webserver',
            name='camera_webserver',
            condition=UnlessCondition(
                LaunchConfiguration('enable_webrtc')),
            parameters=[{
                'port': 8081,
                'quality': 70,
            }],
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
                LaunchConfiguration('enable_detection')),
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
            }],
            output='screen',
        ),
        Node(
            package='ecobot_sensors',
            executable='yolo_detection',
            name='yolo_detection',
            condition=IfCondition(
                LaunchConfiguration('enable_detection')),
            parameters=[{
                'model_path': '/home/ecobot/ros2_ws/src/ecobot/ecobot_sensors/models/yolov8n.engine',
                'conf_threshold': 0.5,
                'inference_rate': 2,
                'use_small_obstacle_filter': True,
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
                'conf_threshold': 0.5,
                'inference_rate': 2,
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
                'stop_distance': 0.6,
                'max_linear': 0.25,
                'max_angular': 0.8,
                'search_timeout': 20.0,
                'search_speed': 0.5,
                'avoid_distance': 0.4,
                'avoid_angle': 1.05,
                'use_map_frame': True,
                'blind_approach_limit': 1.0,
            }],
            output='screen',
        ),
    ])
