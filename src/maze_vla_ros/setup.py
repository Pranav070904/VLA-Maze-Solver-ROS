from setuptools import setup
import os
from glob import glob

package_name = 'maze_vla_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pranav',
    maintainer_email='you@example.com',
    description='VLA maze navigation in Gazebo Sim',
    license='MIT',
    entry_points={
        'console_scripts': [
            'maze_manager = maze_vla_ros.maze_manager_node:main',
            'image_to_maze = maze_vla_ros.image_to_maze_node:main',
            'vla_inference = maze_vla_ros.vla_inference_node:main',
            'motion_controller = maze_vla_ros.motion_controller_node:main',
        ],
    },
)