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
    nav_launch = os.path.join(
        get_package_share_directory('ecobot_navigation'),
        'launch', 'navigation.launch.py')
    localization_launch = os.path.join(
        get_package_share_directory('ecobot_bringup'),
        'launch', 'localization.launch.py')

    depth_to_scan_launch = os.path.join(
        get_package_share_directory('ecobot_sensors'),
        'launch', 'depth_to_scan.launch.py')

    mapping_launch = os.path.join(
        get_package_share_directory('ecobot_bringup'),
        'launch', 'rtabmap_mapping.launch.py')

    arm_launch = os.path.join(
        get_package_share_directory('ecobot_arm_control'),
        'launch', 'arm_control.launch.py')

    nav_share = get_package_share_directory('ecobot_navigation')
    default_map = os.path.join(nav_share, 'maps', 'default_map.yaml')

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
                    '" == "true" or "', LaunchConfiguration('enable_localization', default='false'),
                    '" == "true"']),
                'enable_detection': LaunchConfiguration('enable_detection', default='false'),
                'enable_webrtc': LaunchConfiguration('enable_webrtc', default='true'),
                'enable_livekit': LaunchConfiguration('enable_livekit', default='true'),
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
            PythonLaunchDescriptionSource(mapping_launch),
            launch_arguments={
                'visual_odometry': 'false',
                'database_path': '/home/ecobot/map_data/rtabmap.db',
            }.items(),
            condition=IfCondition(
                LaunchConfiguration('enable_mapping', default='false')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_launch),
            launch_arguments={
                'map': LaunchConfiguration('map', default=''),
            }.items(),
            condition=IfCondition(
                LaunchConfiguration('enable_navigation', default='false')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(localization_launch),
            launch_arguments={
                'scan_matcher': LaunchConfiguration('scan_matcher', default='slam_toolbox'),
                'map': LaunchConfiguration('map', default=default_map),
            }.items(),
            condition=IfCondition(
                LaunchConfiguration('enable_localization', default='false')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(arm_launch),
            launch_arguments={
                'i2c_bus': LaunchConfiguration('arm_i2c_bus', default='7'),
                'pca9685_address': LaunchConfiguration(
                    'arm_pca9685_address', default='0x40'),
                'enable_minicpm_vla': LaunchConfiguration('enable_minicpm_vla', default='false'),
                'enable_openvla': LaunchConfiguration('enable_openvla', default='false'),
                'vla_prompt': LaunchConfiguration('vla_prompt', default='reach forward'),
                'vla_dry_run': LaunchConfiguration('vla_dry_run', default='false'),
            }.items(),
            condition=IfCondition(
                LaunchConfiguration('enable_arm', default='true')),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            condition=IfCondition(
                LaunchConfiguration('enable_urdf', default='false')),
        ),
        Node(
            package='ecobot_bringup',
            executable='hardware_diagnostic_node',
            name='hardware_diagnostic_node',
            parameters=[{
                'check_interval': 5.0,
                'i2c_bus': 7,
                'pca9685_address': 0x40,
                'motor_serial_port': LaunchConfiguration('serial_port', default='/dev/ttyACM0'),
            }],
            condition=IfCondition(
                LaunchConfiguration('enable_diagnostics', default='true')),
            output='screen',
        ),
        ExecuteProcess(
            cmd=['ros2', 'run', 'rosbridge_server', 'rosbridge_websocket',
                 '--port', '9090', '--ros-args',
                 '-p', 'default_call_service_timeout:=5.0',
                 '-p', 'call_services_in_new_thread:=true',
                 '-p', 'send_action_goals_in_new_thread:=true'],
            name='rosbridge_websocket',
            condition=IfCondition(
                LaunchConfiguration('enable_rosbridge', default='true')),
            shell=True,
        ),
    ])
