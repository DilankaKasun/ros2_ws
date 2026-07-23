from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rtabmap_launch = get_package_share_directory('rtabmap_launch')

    return LaunchDescription([
        DeclareLaunchArgument('database_path',
                              default_value=os.path.expanduser('~/.ros/rtabmap.db')),
        DeclareLaunchArgument('rtabmap_viz', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('approx_sync', default_value='true'),
        DeclareLaunchArgument('use_action_for_goal', default_value='false'),
        DeclareLaunchArgument('visual_odometry', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rtabmap_launch, 'launch', 'rtabmap.launch.py')),
            launch_arguments={
                'rgb_topic': '/camera/color/image_raw',
                'depth_topic': '/camera/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'frame_id': 'base_footprint',
                'odom_frame_id': '',
                'database_path': LaunchConfiguration('database_path'),
                'visual_odometry': LaunchConfiguration('visual_odometry'),
                'rtabmap_viz': LaunchConfiguration('rtabmap_viz'),
                'rviz': LaunchConfiguration('rviz'),
                'approx_sync': LaunchConfiguration('approx_sync'),
                'approx_sync_max_interval': '0.05',
                'subscribe_scan': 'true',
                'scan_topic': '/scan',
                'odom_topic': '/odom',
                'use_action_for_goal': LaunchConfiguration('use_action_for_goal'),
            }.items(),
        ),
    ])
