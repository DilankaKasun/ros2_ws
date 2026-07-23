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

    return [
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[params_file],
            condition=IfCondition(PythonExpression(
                ['"', map_path, '" != ""'])),
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params_file],
            condition=IfCondition(PythonExpression(
                ['"', map_path, '" != ""'])),
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': LaunchConfiguration('autostart'),
                'node_names': PythonExpression([
                    '["controller_server", "planner_server", '
                    '"behavior_server", "bt_navigator"]'
                    ' + (["map_server", "amcl"] if "',
                    map_path, '" != "" else [])',
                ]),
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
