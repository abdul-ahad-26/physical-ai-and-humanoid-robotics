---
title: URDF / SDF & Robot Description
sidebar_position: 6.1
description: Understanding Unified Robot Description Format (URDF) and Simulation Description Format (SDF) for robot modeling
---

# URDF / SDF & Robot Description

## Learning Objectives

- Understand the structure and purpose of URDF (Unified Robot Description Format)
- Learn how to create robot models using URDF
- Master the creation of links, joints, and materials in URDF
- Understand the relationship between URDF and SDF for simulation
- Implement realistic robot models with proper physical properties
- Validate and test robot models in simulation environments
- Troubleshoot common URDF/SDF issues

## Introduction to Robot Description Formats

Robot Description Format is essential for representing robots in ROS 2 and simulation environments. Two primary formats are used:

- **URDF (Unified Robot Description Format)**: Used primarily for ROS and MoveIt
- **SDF (Simulation Description Format)**: Used by Gazebo and other simulators

While URDF is the standard for ROS, SDF is required by simulation engines. Understanding both and how to convert between them is crucial for robotics development.

## URDF (Unified Robot Description Format)

### URDF Overview

URDF is an XML-based format that describes a robot's physical and kinematic properties. It defines:

- Links: Rigid bodies of the robot
- Joints: Connections between links
- Materials: Visual appearance
- Inertial properties: Mass, center of mass, inertia
- Visual and collision geometries

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base Link -->
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

  <!-- Example joint connecting to another link -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_link">
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
</robot>
```

## Links in URDF

### Link Definition

Links represent rigid bodies of the robot. Each link can have:

- Visual: How the link appears in simulation
- Collision: How the link interacts physically
- Inertial: Mass and inertia properties

### Visual Elements

```xml
<link name="visual_example">
  <visual>
    <!-- Geometry defines the shape -->
    <geometry>
      <!-- Box -->
      <box size="0.1 0.2 0.3"/>
      <!-- Cylinder -->
      <cylinder radius="0.1" length="0.2"/>
      <!-- Sphere -->
      <sphere radius="0.1"/>
      <!-- Mesh -->
      <mesh filename="package://my_robot/meshes/link.stl" scale="1 1 1"/>
    </geometry>

    <!-- Origin (position and orientation relative to link frame) -->
    <origin xyz="0 0 0" rpy="0 0 0"/>

    <!-- Material -->
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
</link>
```

### Collision Elements

```xml
<link name="collision_example">
  <collision>
    <!-- Similar to visual but for physics simulation -->
    <geometry>
      <box size="0.1 0.2 0.3"/>
    </geometry>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </collision>
</link>
```

### Inertial Properties

```xml
<link name="inertial_example">
  <inertial>
    <!-- Mass in kilograms -->
    <mass value="0.5"/>

    <!-- Inertia matrix (symmetric, only 6 values needed) -->
    <inertia
      ixx="0.001" ixy="0" ixz="0"
      iyy="0.002" iyz="0"
      izz="0.003"/>
  </inertial>
</link>
```

## Joints in URDF

### Joint Types

Joints connect links and define their relative motion:

- **fixed**: No motion between links
- **continuous**: Revolute joint with unlimited rotation
- **revolute**: Revolute joint with limited rotation
- **prismatic**: Linear sliding joint
- **floating**: 6DOF motion
- **planar**: Motion on a plane

### Joint Definition

```xml
<joint name="example_joint" type="revolute">
  <!-- Parent and child links -->
  <parent link="base_link"/>
  <child link="arm_link"/>

  <!-- Position and orientation of joint -->
  <origin xyz="0.1 0 0.2" rpy="0 0 0"/>

  <!-- Rotation axis -->
  <axis xyz="0 0 1"/>

  <!-- Joint limits (for revolute joints) -->
  <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>

  <!-- Joint safety limits -->
  <safety_controller k_position="20" k_velocity="500" soft_lower_limit="0.1" soft_upper_limit="1.4"/>
</joint>
```

## Complete Robot Model Example

Here's a complete example of a simple differential drive robot:

```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Constants -->
  <xacro:property name="base_width" value="0.3"/>
  <xacro:property name="base_length" value="0.4"/>
  <xacro:property name="base_height" value="0.1"/>
  <xacro:property name="wheel_radius" value="0.05"/>
  <xacro:property name="wheel_width" value="0.02"/>
  <xacro:property name="wheel_offset_x" value="0.1"/>
  <xacro:property name="wheel_offset_y" value="0.15"/>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
      <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia
        ixx="0.1" ixy="0" ixz="0"
        iyy="0.1" iyz="0"
        izz="0.2"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia
        ixx="0.001" ixy="0" ixz="0"
        iyy="0.001" iyz="0"
        izz="0.001"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia
        ixx="0.001" ixy="0" ixz="0"
        iyy="0.001" iyz="0"
        izz="0.001"/>
    </inertial>
  </link>

  <!-- Camera link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia
        ixx="0.0001" ixy="0" ixz="0"
        iyy="0.0001" iyz="0"
        izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="${wheel_offset_x} ${wheel_offset_y} 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="${wheel_offset_x} -${wheel_offset_y} 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="${base_length/2 - 0.02} 0 ${base_height}" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo-specific tags -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <!-- Camera sensor definition -->
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

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>${2 * wheel_offset_y}</wheel_separation>
      <wheel_radius>${wheel_radius}</wheel_radius>
      <odom_frame>odom</odom_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>

</robot>
```

## SDF (Simulation Description Format)

### SDF Overview

SDF is the native format for Gazebo and other simulation engines. While URDF is more common in ROS, SDF provides more detailed simulation properties.

### SDF to URDF Conversion

Gazebo can automatically convert URDF to SDF for simulation. However, you can also write native SDF files:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="diff_drive_robot">
    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0.05 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.2</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.4 0.3 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.4 0.3 0.1</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Wheels and joints would be defined similarly -->
  </model>
</sdf>
```

## Xacro Macros for Complex Models

Xacro is a macro language that simplifies complex URDF models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="complex_robot">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <!-- Macro for creating wheels -->
  <xacro:macro name="wheel" params="prefix *origin *axis">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <inertia
          ixx="0.001" ixy="0" ixz="0"
          iyy="0.001" iyz="0"
          izz="0.001"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <xacro:insert_block name="origin"/>
      <xacro:insert_block name="axis"/>
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia
        ixx="0.1" ixy="0" ixz="0"
        iyy="0.1" iyz="0"
        izz="0.2"/>
    </inertial>
  </link>

  <!-- Use the macro to create wheels -->
  <xacro:wheel prefix="front_left">
    <origin xyz="0.2 0.15 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </xacro:wheel>

  <xacro:wheel prefix="front_right">
    <origin xyz="0.2 -0.15 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </xacro:wheel>

  <xacro:wheel prefix="rear_left">
    <origin xyz="-0.2 0.15 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </xacro:wheel>

  <xacro:wheel prefix="rear_right">
    <origin xyz="-0.2 -0.15 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </xacro:wheel>

</robot>
```

## Validating URDF Models

### Using check_urdf

Validate your URDF models using ROS tools:

```bash
# Check URDF syntax and structure
check_urdf /path/to/robot.urdf

# Or if the URDF is in a ROS package
ros2 run urdf check_urdf $(ros2 pkg prefix my_robot_description)/share/my_robot_description/urdf/robot.urdf
```

### Using rviz for Visualization

Visualize your robot model in RViz:

```bash
# Launch RViz with robot model
ros2 run rviz2 rviz2

# Add RobotModel display and set the robot description parameter
```

### Using gazebo for Simulation

Test your robot in Gazebo:

```bash
# Launch Gazebo with your robot
gz sim -r -v 4 --gui-config-path ~/.gz/gui/config/server.config

# Or spawn your robot
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf
```

## Gazebo Integration

### Gazebo-Specific Tags

Add Gazebo-specific properties to your URDF:

```xml
<!-- Physics properties -->
<gazebo reference="link_name">
  <mu1>0.8</mu1>  <!-- Friction coefficient -->
  <mu2>0.8</mu2>  <!-- Secondary friction coefficient -->
  <kp>1000000.0</kp>  <!-- Spring stiffness -->
  <kd>100.0</kd>  <!-- Damping coefficient -->
  <material>Gazebo/Blue</material>
</gazebo>

<!-- Sensors -->
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

<!-- ROS Control plugin -->
<gazebo>
  <plugin name="ros_control" filename="libgazebo_ros_control.so">
    <parameters>$(find my_robot_config)/config/controllers.yaml</parameters>
  </plugin>
</gazebo>
```

## Troubleshooting Tips

- If the robot model doesn't appear in RViz, check that the robot_description parameter is properly set
- For simulation issues, verify that all joint limits and dynamics are properly defined
- If links fall through the ground, check collision geometries and physics properties
- Use `ros2 param list` to verify that robot_description is loaded
- Check that the URDF file is properly formatted with correct XML syntax
- Verify that mesh files are in the correct location and package is properly exported
- If joints don't move, check joint types and controller configurations
- Use `gz topic -l` to verify that sensor topics are being published correctly

## Cross-References

For related concepts, see:
- [ROS 2 Fundamentals](../ros2-core/index.md) - For ROS integration
- [Gazebo Simulation](../gazebo/index.md) - For simulation integration
- [Sensors](../sensors/index.md) - For sensor integration in URDF
- [Robotics Fundamentals](../humanoid-kinematics/index.md) - For kinematics

## Summary

URDF and SDF are fundamental to robotics development, providing the description of robot geometry, kinematics, and dynamics. Proper robot modeling is essential for successful simulation, visualization, and control. Understanding how to create accurate, efficient robot models is crucial for robotics development, as these models form the basis for simulation, motion planning, and control systems.