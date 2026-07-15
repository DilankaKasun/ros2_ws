from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ecobot_motor_control',
            executable='motor_control_node',
            name='motor_control_node',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port', default='/dev/ttyACM0'),
                'baudrate': 115200,
                'control_frequency': 10.0,
                'cmd_vel_timeout': 0.5,
                'max_rpm': 130.0,
                'product_id': 1,
                'odom_frame_id': 'odom',
                'base_frame_id': 'base_footprint',
                'use_sim_time': LaunchConfiguration('use_sim_time', default='false'),
            }],
            output='screen',
        ),
    ])
