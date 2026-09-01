from setuptools import setup

package_name = 'ecobot_arm_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/arm_control.launch.py']),
        (f'share/{package_name}/config', ['config/arm_params.yaml']),
        (f'share/{package_name}/config', ['config/arm_tracking_params.yaml']),
        (f'share/{package_name}/urdf', ['urdf/ecobot_arm.urdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ecobot',
    maintainer_email='user@ecobot.local',
    description='Robot arm controller for ecobot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_manual_node = ecobot_arm_control.arm_manual_node:main',
            'arm_scanner_node = ecobot_arm_control.arm_scanner_node:main',
            'usb_camera_node = ecobot_arm_control.usb_camera_node:main',
            'arm_camera_server = ecobot_arm_control.arm_camera_server:main',
            'minicpm_vla_node = ecobot_arm_control.minicpm_vla_node:main',
            'openvla_node = ecobot_arm_control.openvla_node:main',
            'arm_target_tracker = ecobot_arm_control.arm_target_tracker:main',
            'arm_camera_calibrate = ecobot_arm_control.arm_camera_calibrate:main',
            'wrist_cv_analyzer = ecobot_arm_control.wrist_cv_analyzer:main',
            'arm_scan_rl_agent = ecobot_arm_control.arm_scan_rl_agent:main',
            'arm_teleop = ecobot_arm_control.arm_teleop:main',
        ],
    },
)
