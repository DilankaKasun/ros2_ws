from launch import LaunchDescription
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    nav2_params = os.path.join(
        get_package_share_directory('ecobot_navigation'),
        'config', 'nav2_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=nav2_params),
        DeclareLaunchArgument('map', default_value=''),

        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
            condition=IfCondition(PythonExpression(
                ['"', LaunchConfiguration('map'), '" != ""'])),
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': LaunchConfiguration('autostart'),
                'node_names': PythonExpression([
                    '["amcl", "controller_server", "planner_server", '
                    '"behavior_server", "bt_navigator", "local_costmap", '
                    '"global_costmap"] + (["map_server"] if "',
                    LaunchConfiguration('map'), '" != "" else [])',
                ]),
            }],
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
            remappings=[('/cmd_vel', '/nav_cmd_vel')],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
        Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='local_costmap',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
        Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='global_costmap',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
