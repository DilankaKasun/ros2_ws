from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ecobot_dashboard',
            executable='dashboard_server',
            name='dashboard_server',
            output='screen',
        ),
    ])
