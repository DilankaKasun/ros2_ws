"""Launches the LiveKit voice agent worker.

This is an ExecuteProcess, not a launch_ros Node: livekit-agents' cli.run_app
owns the process's asyncio loop and its own start/dev/connect subcommands,
which doesn't fit launch_ros's Node/rclpy.spin() lifecycle. The agent creates
its own rclpy node internally (see ros_bridge.py) on a background thread.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('ecobot_voice'),
        'config', 'voice_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('enable_voice_agent', default_value='false'),
        DeclareLaunchArgument(
            'agent_mode', default_value='start',
            description="livekit-agents subcommand: 'start' (production) or 'dev'"),
        ExecuteProcess(
            cmd=['ros2', 'run', 'ecobot_voice', 'voice_agent',
                 LaunchConfiguration('agent_mode'),
                 '--ros-args', '--params-file', params_file],
            condition=IfCondition(LaunchConfiguration('enable_voice_agent')),
            output='screen',
        ),
    ])
