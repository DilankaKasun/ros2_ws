from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    bringup_share = get_package_share_directory('ecobot_bringup')
    sensors_share = get_package_share_directory('ecobot_sensors')
    motor_share = get_package_share_directory('ecobot_motor_control')
    teleop_share = get_package_share_directory('ecobot_teleop')
    dash_share = get_package_share_directory('ecobot_dashboard')

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('enable_teleop', default_value='false'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(motor_share, 'launch', 'motor_control.launch.py')),
            launch_arguments={
                'serial_port': LaunchConfiguration('serial_port'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensors_share, 'launch', 'sensors.launch.py')),
            launch_arguments={
                'enable_obstacle_avoidance': 'false',
                'enable_detection': 'false',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensors_share, 'launch', 'depth_to_scan.launch.py')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_share, 'launch', 'rtabmap_mapping.launch.py')),
            launch_arguments={
                'database_path': '/home/ecobot/map_data/rtabmap.db',
                'rtabmap_viz': 'false',
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(teleop_share, 'launch', 'teleop.launch.py')),
            condition=IfCondition(LaunchConfiguration('enable_teleop')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(dash_share, 'launch', 'dashboard.launch.py')),
        ),
    ])
