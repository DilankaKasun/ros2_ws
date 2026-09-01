from glob import glob
from setuptools import setup

package_name = 'ecobot_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ecobot',
    maintainer_email='user@ecobot.local',
    description='Autonomous plant-health mission orchestration for ecobot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plant_mission_node = ecobot_mission.plant_mission_node:main',
        ],
    },
)
