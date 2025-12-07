#!/usr/bin/env python3

"""
Basic Gazebo Launch Script

This script demonstrates how to launch Gazebo with ROS 2 integration programmatically.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate the launch description for Gazebo with ROS 2 integration.
    """
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='empty.sdf')

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': ['-r', '-v', '4', world]}.items()
    )

    # ROS - Gazebo Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Vec3d'
        ],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
            ('/odom', '/odom'),
            ('/scan', '/scan'),
            ('/tf', '/tf')
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add declared launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    ))

    ld.add_action(DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/usr/share/gazebo/worlds`'
    ))

    # Add nodes to launch
    ld.add_action(gazebo)
    ld.add_action(bridge)

    return ld


if __name__ == '__main__':
    # This script can be run directly to see the launch description
    ld = generate_launch_description()
    print("Launch description generated successfully")
    print(f"Number of actions: {len(ld.entities)}")