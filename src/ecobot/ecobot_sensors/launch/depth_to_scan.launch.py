from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='depth_to_scan',
            output='screen',
            parameters=[{
                'output_frame': 'd415_depth_optical_frame',
                'range_min': 0.3,
                'range_max': 8.0,
                'scan_height': 0.1,
                'scan_time': 0.033,
            }],
            remappings=[
                ('/depth/image', '/camera/depth/image_raw'),
                ('/depth/points', '/camera/depth/color/points'),
            ],
        ),
    ])
