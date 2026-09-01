from glob import glob
from setuptools import setup
import os

package_name = 'ecobot_sensors'

model_files = glob('models/**', recursive=True)
model_files = [f for f in model_files if os.path.isfile(f)]

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/models', model_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ecobot',
    maintainer_email='user@ecobot.local',
    description='RealSense D415 sensor driver and obstacle avoidance',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'realsense_feed = ecobot_sensors.realsense_feed:main',
            'obstacle_avoidance = ecobot_sensors.obstacle_avoidance:main',
            'object_detection = ecobot_sensors.object_detection:main',
            'depth_ground_detection = ecobot_sensors.depth_ground_detection:main',
            'yolo_detection = ecobot_sensors.yolo_detection:main',
            'ecobot_detection_node = ecobot_sensors.ecobot_detection_node:main',
            'tof_sensors = ecobot_sensors.tof_sensors:main',
            'scan_filter_node = ecobot_sensors.scan_filter_node:main',
            'livekit_streamer = ecobot_sensors.livekit_streamer:main',
            'livekit_bridge = ecobot_sensors.livekit_bridge:main',
        ],
    },
)
