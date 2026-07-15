from setuptools import setup

package_name = 'ecobot_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/teleop.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ecobot',
    maintainer_email='user@ecobot.local',
    description='Keyboard teleoperation for ecobot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keyboard_teleop = ecobot_teleop.keyboard_teleop:main',
        ],
    },
)
