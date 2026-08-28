from launch import LaunchDescription
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context, *args, **kwargs):
    nav_share = get_package_share_directory('ecobot_navigation')
    map_path = LaunchConfiguration('map').perform(context)

    if map_path:
        params_file = os.path.join(nav_share, 'config', 'nav2_params.yaml')
    else:
        params_file = os.path.join(nav_share, 'config', 'nav2_params_mapless.yaml')

    node_names = []
    if map_path:
        node_names.extend(['map_server', 'amcl'])
    node_names.extend(['controller_server', 'planner_server', 'behavior_server', 'bt_navigator'])

    return [
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[params_file, {
                'yaml_filename': map_path,
            }],
            condition=IfCondition(PythonExpression(
                ['"', map_path, '" != ""'])),
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params_file],
            # AMCL has segfaulted repeatedly in live runs (exit code -11).
            # Respawn so localization self-recovers instead of leaving the
            # robot without a map frame until the whole system restarts.
            respawn=True,
            respawn_delay=3.0,
            condition=IfCondition(PythonExpression(
                ['"', map_path, '" != ""'])),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                       '--frame-id', 'map', '--child-frame-id', 'odom'],
            # AMCL publishes map->odom itself when a map is loaded. Publishing
            # a second (static identity) map->odom here is not just redundant —
            # two writers on the same TF pair cause intermittent extrapolation
            # failures in the costmaps ("unconnected trees") and the AMCL crash
            # seen in live runs. Only publish the identity in mapless mode.
            condition=IfCondition(PythonExpression(
                ['"', map_path, '" == ""'])),
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': node_names,
            }],
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file],
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
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('map', default_value=''),
        OpaqueFunction(function=launch_setup),
    ])
