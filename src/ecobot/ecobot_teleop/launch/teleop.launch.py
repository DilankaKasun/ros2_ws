from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ecobot_teleop',
            executable='keyboard_teleop',
            name='keyboard_teleop',
            parameters=[{
                'max_linear_speed': 0.5,
                'max_angular_speed': 1.0,
            }],
            output='screen',
        ),
    ])
