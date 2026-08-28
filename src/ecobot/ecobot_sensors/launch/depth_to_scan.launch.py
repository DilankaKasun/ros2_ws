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
                'output_frame': 'camera_depth_optical_frame',
                'range_min': 0.3,
                'range_max': 8.0,
                'scan_height': 240,
                'scan_time': 0.033,
            }],
            remappings=[
                ('depth', '/camera/depth/image_raw'),
                ('depth_camera_info', '/camera/depth/camera_info'),
            ],
        ),
        # Sanitize /scan (NaN/out-of-range -> inf) before Nav2 ingests it.
        # AMCL segfaults on invalid ranges; nav configs subscribe to
        # /scan_filtered, which this node publishes.
        Node(
            package='ecobot_sensors',
            executable='scan_filter_node',
            name='scan_filter',
            output='screen',
        ),
    ])
