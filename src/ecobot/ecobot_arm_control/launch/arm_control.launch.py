import os
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_arm_camera', default_value='true'),
        DeclareLaunchArgument('enable_arm_scanner', default_value='true'),
        # The detector on the wrist camera, so the arm can confirm it is
        # looking at a plant rather than trusting a colour mask.
        DeclareLaunchArgument('enable_wrist_detection', default_value='true'),
        DeclareLaunchArgument('enable_rl_agent', default_value='false'),
        DeclareLaunchArgument('enable_minicpm_vla', default_value='false'),
        DeclareLaunchArgument('enable_openvla', default_value='false'),
        DeclareLaunchArgument('enable_adaptive_target', default_value='false'),
        DeclareLaunchArgument('vla_prompt', default_value='reach forward'),
        DeclareLaunchArgument('vla_dry_run', default_value='false'),
        DeclareLaunchArgument('peak_speed', default_value='20.0'),
        DeclareLaunchArgument('accel', default_value='90.0'),
        DeclareLaunchArgument('step_deg', default_value='1.0'),

        Node(
            package='ecobot_arm_control',
            executable='arm_manual_node',
            name='arm_control_node',
            output='screen',
            respawn=True,
            respawn_delay=2.0,
            parameters=[{
                'i2c_bus': LaunchConfiguration('i2c_bus', default=7),
                'pca9685_address': LaunchConfiguration(
                    'pca9685_address', default=0x40),
                'pwm_freq': LaunchConfiguration(
                    'pwm_freq', default=50.0),
                'cmd_timeout': LaunchConfiguration(
                    'cmd_timeout', default=0.0),
                'move_interval_ms': LaunchConfiguration(
                    'move_interval_ms', default=15),
                'home_ramp_steps': LaunchConfiguration(
                    'home_ramp_steps', default=50),
                'peak_speed': LaunchConfiguration('peak_speed'),
                'accel': LaunchConfiguration('accel'),
                'step_deg': LaunchConfiguration('step_deg'),
                'overrides': ParameterValue(
                    LaunchConfiguration('overrides', default='{}'),
                    value_type=str),
            }],
        ),
        Node(
            package='ecobot_arm_control',
            executable='arm_scanner_node',
            name='arm_scanner_node',
            condition=IfCondition(
                LaunchConfiguration('enable_arm_scanner')),
            parameters=[os.path.join(
                get_package_share_directory('ecobot_arm_control'),
                'config', 'arm_params.yaml')],
            output='screen',
        ),
        Node(
            package='ecobot_arm_control',
            executable='usb_camera_node',
            name='usb_camera_node',
            condition=IfCondition(
                LaunchConfiguration('enable_arm_camera')),
            parameters=[{
                'device': LaunchConfiguration('arm_camera_device',
                                              default='/dev/v4l/by-id/usb-HRY_YDL_lens_USB_Camera_20210616_720-video-index0'),
                'width': 640,
                'height': 480,
                'fps': 30,
                'topic': '/arm/camera/image_raw',
            }],
            output='screen',
        ),
        Node(
            package='ecobot_arm_control',
            executable='arm_camera_server',
            name='arm_camera_server',
            condition=IfCondition(
                LaunchConfiguration('enable_arm_camera')),
            parameters=[{
                'port': 8084,
                'topic': '/arm/camera/image_raw',
            }],
            output='screen',
        ),
        Node(
            package='image_transport',
            executable='republish',
            name='arm_camera_compressed_republish',
            condition=IfCondition(
                LaunchConfiguration('enable_arm_camera')),
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/arm/camera/image_raw'),
                ('out/compressed', '/arm/camera/image_raw/compressed'),
            ],
            output='screen',
        ),
        Node(
            package='ecobot_arm_control',
            executable='minicpm_vla_node',
            name='minicpm_vla_node',
            condition=IfCondition(
                LaunchConfiguration('enable_minicpm_vla')),
            parameters=[{
                'dry_run': LaunchConfiguration('vla_dry_run'),
                'prompt': LaunchConfiguration('vla_prompt'),
            }],
            output='screen',
        ),
        Node(
            package='ecobot_arm_control',
            executable='openvla_node',
            name='openvla_node',
            condition=IfCondition(
                LaunchConfiguration('enable_openvla')),
            parameters=[{
                'dry_run': LaunchConfiguration('vla_dry_run'),
                'prompt': LaunchConfiguration('vla_prompt'),
            }],
            output='screen',
        ),
        Node(
            package='ecobot_arm_control',
            executable='wrist_cv_analyzer',
            name='wrist_cv_analyzer',
            condition=IfCondition(
                LaunchConfiguration('enable_arm_camera')),
            output='screen',
        ),
        Node(
            package='ecobot_arm_control',
            executable='arm_scan_rl_agent',
            name='arm_scan_rl_agent',
            condition=IfCondition(
                LaunchConfiguration('enable_rl_agent')),
            output='screen',
        ),
        Node(
            package='ecobot_arm_control',
            executable='arm_target_tracker',
            name='arm_target_tracker',
            condition=IfCondition(
                LaunchConfiguration('enable_adaptive_target')),
            parameters=[os.path.join(
                get_package_share_directory('ecobot_arm_control'),
                'config', 'arm_tracking_params.yaml')],
            output='screen',
        ),
        # The object detector, pointed at the WRIST camera. The arm scanner
        # uses this to decide whether it is really looking at a plant and
        # to pull the aim back on when it drifts — colour masking alone
        # calls a green wall a plant and a beige pot nothing.
        #
        # No depth or camera info: this one is only asked WHETHER a plant
        # is in frame and WHERE in the frame, never how far away, so the
        # detector's distance fields simply come back empty.
        Node(
            package='ecobot_sensors',
            executable='ecobot_detection_node',
            name='wrist_detection_node',
            condition=IfCondition(LaunchConfiguration('enable_wrist_detection')),
            parameters=[{
                'model_path': '/home/ecobot/ros2_ws/src/ecobot/ecobot_sensors/models/yolov8n.engine',
                'backend': 'tensorrt',
                'conf_threshold': 0.35,
                # Slower than the forward detector: the arm holds still at
                # each viewpoint, so there is nothing to chase, and this
                # shares a GPU with it.
                'inference_rate': 4,
                'camera_topic': '/arm/camera/image_raw',
                'detections_topic': '/arm/detections',
                'overlay_topic': '/arm/detection_image',
                'depth_topic': '',
                'camera_info_topic': '',
            }],
            output='screen',
        ),
    ])
