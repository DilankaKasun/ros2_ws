from setuptools import setup

package_name = 'ecobot_dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', ['launch/dashboard.launch.py']),
        (f'share/{package_name}/www', ['www/index.html']),
    ],
    install_requires=['setuptools', 'aiohttp'],
    zip_safe=True,
    maintainer='ecobot',
    maintainer_email='user@ecobot.local',
    description='Web dashboard for ecobot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard_server = ecobot_dashboard.dashboard_server:main',
        ],
    },
)
