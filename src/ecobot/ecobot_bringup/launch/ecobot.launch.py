from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    motor_launch = os.path.join(
        get_package_share_directory('ecobot_motor_control'),
        'launch', 'motor_control.launch.py')
    sensors_launch = os.path.join(
        get_package_share_directory('ecobot_sensors'),
        'launch', 'sensors.launch.py')
    teleop_launch = os.path.join(
        get_package_share_directory('ecobot_teleop'),
        'launch', 'teleop.launch.py')
    nav_launch = os.path.join(
        get_package_share_directory('ecobot_navigation'),
        'launch', 'navigation.launch.py')
    dash_launch = os.path.join(
        get_package_share_directory('ecobot_dashboard'),
        'launch', 'dashboard.launch.py')

    depth_to_scan_launch = os.path.join(
        get_package_share_directory('ecobot_sensors'),
        'launch', 'depth_to_scan.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(motor_launch),
            launch_arguments={
                'serial_port': LaunchConfiguration('serial_port', default='/dev/ttyACM0'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sensors_launch),
            launch_arguments={
                'enable_obstacle_avoidance': PythonExpression([
                    '"', LaunchConfiguration('enable_obstacle_avoidance', default='false'),
                    '" == "true" or "', LaunchConfiguration('enable_navigation', default='false'),
                    '" == "true"']),
            }.items(),
            condition=IfCondition(
                LaunchConfiguration('enable_sensors', default='true')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depth_to_scan_launch),
            condition=IfCondition(
                LaunchConfiguration('enable_sensors', default='true')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(teleop_launch),
            condition=IfCondition(
                LaunchConfiguration('enable_teleop', default='false')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_launch),
            condition=IfCondition(
                LaunchConfiguration('enable_navigation', default='false')),
        ),
        Node(
            package='ecobot_bringup',
            executable='cmd_vel_mux',
            name='cmd_vel_mux',
            condition=IfCondition(
                LaunchConfiguration('enable_navigation', default='false')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(dash_launch),
            condition=IfCondition(
                LaunchConfiguration('enable_dashboard', default='true')),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            condition=IfCondition(
                LaunchConfiguration('enable_urdf', default='false')),
        ),
        ExecuteProcess(
            cmd=['ros2', 'run', 'rosbridge_server', 'rosbridge_websocket', '--port', '9090'],
            name='rosbridge_websocket',
            condition=IfCondition(
                LaunchConfiguration('enable_rosbridge', default='false')),
            shell=True,
        ),
    ])
