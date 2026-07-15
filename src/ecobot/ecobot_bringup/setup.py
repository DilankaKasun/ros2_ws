import os
from glob import glob
from setuptools import setup

package_name = 'ecobot_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
        (f'share/{package_name}/scripts', ['scripts/start.sh']),
    ],
    scripts=['scripts/start.sh'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ecobot',
    maintainer_email='user@ecobot.local',
    description='Launch files and startup for ecobot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmd_vel_mux = ecobot_bringup.cmd_vel_mux:main',
            'send_goal = ecobot_bringup.send_goal:main',
        ],
    },
)
