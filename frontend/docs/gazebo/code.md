---
title: Gazebo Code Examples
sidebar_position: 6.3
description: Code examples for Gazebo simulation integration
---

# Gazebo Code Examples

## Basic Gazebo Launch File

Here's a launch file to start Gazebo with ROS 2 integration:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    # Launch Arguments
    world = LaunchConfiguration('world')
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/usr/share/gazebo/worlds`'
    )

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
        arguments=['--ros-args', '--log-level', 'INFO'],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        bridge
    ])
```

## Robot State Publisher for Gazebo

Here's how to publish robot state in Gazebo:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    urdf_package = LaunchConfiguration('urdf_package', default='my_robot_description')
    urdf_file = LaunchConfiguration('urdf_file', default='robot.urdf')

    # Get URDF via xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindPackageShare(urdf_package), 'urdf', urdf_file])
    ])
    robot_description = {'robot_description': robot_description_content}

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': use_sim_time}]
    )

    # Joint State Publisher (for non-fixed joints)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.2'
        ],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add the declared launch arguments
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))
    ld.add_action(DeclareLaunchArgument('urdf_package', default_value='my_robot_description'))
    ld.add_action(DeclareLaunchArgument('urdf_file', default_value='robot.urdf'))

    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(spawn_entity)

    return ld
```

## Simple Controller for Gazebo Robot

Here's an example of a simple controller for a differential drive robot in Gazebo:

```python
#!/usr/bin/env python3

"""
Simple Differential Drive Controller for Gazebo Robot
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math


class SimpleController(Node):
    """
    A simple controller that navigates a robot in Gazebo based on laser scan data.
    """

    def __init__(self):
        super().__init__('simple_controller')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create subscriber for laser scan data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Create timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Initialize variables
        self.scan_data = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.obstacle_distance = float('inf')

        self.get_logger().info('Simple Controller initialized')

    def scan_callback(self, msg):
        """
        Callback function to process laser scan data.
        """
        self.scan_data = msg
        # Get the distance to the closest obstacle in front
        if len(msg.ranges) > 0:
            # Front is typically around the middle of the scan
            front_idx = len(msg.ranges) // 2
            self.obstacle_distance = msg.ranges[front_idx]

    def control_loop(self):
        """
        Main control loop to navigate the robot.
        """
        if self.scan_data is None:
            return

        msg = Twist()

        # Simple obstacle avoidance: if obstacle is close, turn
        if self.obstacle_distance < 1.0:  # 1 meter threshold
            msg.linear.x = 0.2  # Slow down
            msg.angular.z = 0.5  # Turn right
        else:
            msg.linear.x = 0.5  # Move forward
            msg.angular.z = 0.0  # No turning

        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    controller = SimpleController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        stop_msg = Twist()
        controller.cmd_vel_pub.publish(stop_msg)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Camera Image Processing in Gazebo

Here's an example of processing camera images from Gazebo:

```python
#!/usr/bin/env python3

"""
Camera Image Processor for Gazebo Simulation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraProcessor(Node):
    """
    Process images from Gazebo camera.
    """

    def __init__(self):
        super().__init__('camera_processor')

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        self.get_logger().info('Camera Processor initialized')

    def image_callback(self, msg):
        """
        Callback function to process camera images.
        """
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Example processing: detect edges
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Display the processed image
            cv2.imshow("Original", cv_image)
            cv2.imshow("Edges", edges)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)

    processor = CameraProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## URDF with Gazebo-Specific Tags

Here's a complete URDF example with Gazebo-specific tags:

```xml
<?xml version="1.0"?>
<robot name="gazebo_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Camera link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.15 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.05" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.15 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo-specific configurations -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Camera sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

  <!-- ROS 2 Control plugin -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_robot_package)/config/my_controllers.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

## Gazebo Controller Configuration

Here's a controller configuration file for the robot:

```yaml
# config/my_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    diff_drive_controller:
      type: diff_drive_controller/DiffDriveController

diff_drive_controller:
  ros__parameters:
    left_wheel_names: ["left_wheel_joint"]
    right_wheel_names: ["right_wheel_joint"]

    wheel_separation: 0.3
    wheel_radius: 0.1

    # Publish rates
    publish_rate: 50.0
    odom_publish_rate: 20.0

    # Frame names
    pose_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
    twist_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]

    # Velocity and acceleration limits
    # Note: The following are configured for a robot that can achieve
    # max velocity of 1.0 m/s and max acceleration of 2.0 m/s^2
    cmd_vel_timeout: 0.5
    allow_multiple_cmd_vel_publishers: false
    velocity_rolling_window_size: 10

    # Velocity limits
    linear.x.has_velocity_limits: true
    linear.x.max_velocity: 1.0  # m/s
    linear.x.has_acceleration_limits: true
    linear.x.max_acceleration: 2.0  # m/s^2
    linear.x.has_jerk_limits: false
    linear.x.max_jerk: 0.0  # m/s^3

    angular.z.has_velocity_limits: true
    angular.z.max_velocity: 1.0  # rad/s
    angular.z.has_acceleration_limits: true
    angular.z.max_acceleration: 2.0  # rad/s^2
    angular.z.has_jerk_limits: false
    angular.z.max_jerk: 0.0  # rad/s^3
```

## Launch File with Controllers

Here's a complete launch file that brings up the robot with controllers:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Get URDF via xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindPackageShare('my_robot_description'), 'urdf', 'robot.urdf'])
    ])
    robot_description = {'robot_description': robot_description_content}

    # Set parameters
    params = {'use_sim_time': use_sim_time}

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, params]
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.2'
        ],
        output='screen'
    )

    # Load controllers
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )

    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller'],
    )

    # Create launch description
    ld = LaunchDescription()

    # Add any necessary launch arguments
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))

    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_robot)

    # Register event handlers to start controllers after spawn
    ld.add_action(RegisterEventHandler(
        OnProcessStart(
            target_action=spawn_robot,
            on_start=[
                joint_state_broadcaster_spawner,
            ],
        )
    ))

    ld.add_action(RegisterEventHandler(
        OnProcessStart(
            target_action=joint_state_broadcaster_spawner,
            on_start=[
                diff_drive_controller_spawner,
            ],
        )
    ))

    return ld
```

## Summary

These code examples demonstrate how to integrate ROS 2 with Gazebo simulation, including:
- Launch files for starting Gazebo with ROS 2 bridge
- Robot model integration with Gazebo-specific tags
- Controller configuration for simulated robots
- Sensor processing from simulated environments
- Complete workflow for robot simulation

Each example is designed to be a starting point for your own Gazebo simulation projects.