from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_obstacle_avoidance', default_value='false'),
        DeclareLaunchArgument('enable_webrtc', default_value='false'),
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
                'safe_distance': 0.8,
                'show_viewer': False,
            }],
            output='screen',
        ),
    ])
