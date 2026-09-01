import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ecobot_mission',
            executable='plant_mission_node',
            name='plant_mission_node',
            output='screen',
            parameters=[os.path.join(
                get_package_share_directory('ecobot_mission'),
                'config', 'mission_params.yaml')],
        ),
    ])
