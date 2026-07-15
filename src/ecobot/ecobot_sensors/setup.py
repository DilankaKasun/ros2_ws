from glob import glob
from setuptools import setup

package_name = 'ecobot_sensors'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
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
            'camera_webserver = ecobot_sensors.camera_webserver:main',
            'webrtc_streamer = ecobot_sensors.webrtc_streamer:main',
        ],
    },
)
