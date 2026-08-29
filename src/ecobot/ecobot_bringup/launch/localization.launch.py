from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context, *args, **kwargs):
    nav_share = get_package_share_directory('ecobot_navigation')
    sensors_share = get_package_share_directory('ecobot_sensors')
    bringup_share = get_package_share_directory('ecobot_bringup')

    matcher = LaunchConfiguration('scan_matcher').perform(context)
    map_path = LaunchConfiguration('map').perform(context)

    localization_params = os.path.join(nav_share, 'config', 'localization_params.yaml')
    depth_to_scan_launch = os.path.join(sensors_share, 'launch', 'depth_to_scan.launch.py')

    nodes = []

    # 1) Depth -> LaserScan. Skippable: ecobot.launch.py already starts this
    # under enable_sensors, so including it again here would double-launch
    # depth_to_scan/scan_filter under identical node names and corrupt the
    # scan feed slam_toolbox/AMCL rely on.
    if LaunchConfiguration('enable_depth_to_scan').perform(context) == 'true':
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(depth_to_scan_launch),
            )
        )

    # 2) map_server (only if map path is given)
    if map_path:
        nodes.append(
            Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                output='screen',
                parameters=[localization_params, {
                    'yaml_filename': map_path,
                }],
            )
        )
        nodes.append(
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_map',
                output='screen',
                parameters=[{
                    'autostart': True,
                    'node_names': ['map_server'],
                }],
            )
        )

    # 3) Scan Matcher
    if matcher == 'slam_toolbox':
        # slam_toolbox in localization mode
        nodes.append(
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[localization_params],
            )
        )
    elif matcher == 'amcl':
        # AMCL (needs map)
        if map_path:
            nodes.append(
                Node(
                    package='nav2_amcl',
                    executable='amcl',
                    name='amcl',
                    output='screen',
                    parameters=[localization_params],
                )
            )
    elif matcher == 'rtabmap':
        # RTAB-Map in localization mode (RGB-D scan matching)
        rtabmap_launch = get_package_share_directory('rtabmap_launch')
        database_path = LaunchConfiguration('database_path').perform(context)
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(rtabmap_launch, 'launch', 'rtabmap.launch.py')),
                launch_arguments={
                    'rgb_topic': '/camera/color/image_raw',
                    'depth_topic': '/camera/aligned_depth_to_color/image_raw',
                    'camera_info_topic': '/camera/color/camera_info',
                    'frame_id': 'base_footprint',
                    'odom_frame_id': '',
                    'database_path': database_path,
                    'visual_odometry': 'false',
                    'rtabmap_viz': 'false',
                    'rviz': 'false',
                    'approx_sync': 'true',
                    'approx_sync_max_interval': '0.05',
                    'subscribe_scan': 'true',
                    'scan_topic': '/scan',
                    'odom_topic': '/odom',
                    'use_action_for_goal': 'false',
                    'localization': 'true',
                }.items(),
            )
        )

    # 4) Map -> Odom TF broadcaster (fallback if no scan matcher provides it)
    nodes.append(
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_fallback',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                       '--frame-id', 'map', '--child-frame-id', 'odom'],
            condition=IfCondition(PythonExpression([
                '"', matcher, '" == "none"'])),
        )
    )

    return nodes


def generate_launch_description():
    nav_share = get_package_share_directory('ecobot_navigation')
    default_map = os.path.join(nav_share, 'maps', 'default_map.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'scan_matcher',
            default_value='slam_toolbox',
            description='Scan matcher: slam_toolbox, amcl, rtabmap, or none'),
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Path to map YAML file (empty for mapless)'),
        DeclareLaunchArgument(
            'enable_depth_to_scan',
            default_value='true',
            description='Start depth_to_scan/scan_filter here. Set false '
                        'when the including launch file already starts it '
                        '(e.g. ecobot.launch.py with enable_sensors:=true).'),
        DeclareLaunchArgument(
            'database_path',
            default_value=os.path.expanduser('~/.ros/rtabmap.db'),
            description='RTAB-Map database path'),
        OpaqueFunction(function=launch_setup),
    ])
