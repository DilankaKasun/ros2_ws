from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
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

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(motor_share, 'launch', 'motor_control.launch.py')),
            launch_arguments={
                'serial_port': LaunchConfiguration('serial_port'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensors_share, 'launch', 'depth_to_scan.launch.py')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_share, 'launch', 'slam.launch.py')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(teleop_share, 'launch', 'teleop.launch.py')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(dash_share, 'launch', 'dashboard.launch.py')),
        ),
    ])
