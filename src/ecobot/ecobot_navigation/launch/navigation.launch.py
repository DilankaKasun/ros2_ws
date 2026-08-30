"""The map driver (Nav2, mapless) plus the run node that owns the wheels.

Mapless: Nav2 plans in the robot's own wheel frame against a rolling
picture of obstacles built from the depth camera. Nothing has to be
recorded in advance, and the drifting frame does not matter because the
map driver only has to finish roughly 1.2m short of the plant — the
camera driver does the precise part.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    nav_share = get_package_share_directory('ecobot_navigation')
    params_file = os.path.join(nav_share, 'config', 'nav2_params_mapless.yaml')

    lifecycle_nodes = ['controller_server', 'planner_server',
                       'behavior_server', 'bt_navigator']

    return LaunchDescription([
        DeclareLaunchArgument('enable_run_node', default_value='true'),
        # Kept so callers that still pass map:=... do not fail. This stack
        # is mapless; a recorded map is not used here.
        DeclareLaunchArgument('map', default_value=''),

        # Mapless means nothing else writes map->odom, so a fixed identity
        # keeps the frame tree connected for anything that asks for 'map'.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                       '--frame-id', 'map', '--child-frame-id', 'odom'],
            output='screen',
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file],
            # The safety layer, not Nav2, is what finally writes /cmd_vel.
            # It picks whichever driver is still publishing, which is how
            # the handover works.
            remappings=[('/cmd_vel', '/nav_cmd_vel')],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': lifecycle_nodes,
            }],
        ),
        Node(
            package='ecobot_navigation',
            executable='plant_run_node',
            name='plant_run_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_run_node')),
        ),
    ])
