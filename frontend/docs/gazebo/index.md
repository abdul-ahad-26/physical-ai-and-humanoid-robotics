---
title: Gazebo Simulation
sidebar_position: 6.1
description: Introduction to Gazebo simulation environment for robotics development
---

# Gazebo Simulation

## Learning Objectives

- Understand the Gazebo simulation environment and its role in robotics
- Learn how to create and configure simulation worlds
- Master robot model integration in Gazebo
- Execute basic simulation workflows and experiments
- Understand physics engines and their impact on simulation accuracy
- Configure sensors and actuators in simulation
- Troubleshoot common simulation issues

## Introduction to Gazebo

Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. It provides:

- High-fidelity physics simulation
- Advanced 3D rendering capabilities
- Support for various robot sensors
- Realistic environmental conditions
- Integration with ROS/ROS 2

## Installing Gazebo

### Gazebo Garden (Recommended for ROS 2 Humble)

```bash
# Add Gazebo packages repository
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'

# Setup keys
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Update packages
sudo apt update

# Install Gazebo Garden
sudo apt install gz-garden
```

### Alternative: Install Gazebo Sim via ROS 2

```bash
# Install Gazebo Sim packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-dev
```

### Verify Installation

```bash
# Check Gazebo version
gz --version

# List available Gazebo commands
gz --list

# Test launch an empty world
gz sim -r -v 4 empty.sdf
```

### Environment Setup

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Optionally add to your .bashrc for automatic sourcing
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source /usr/share/gz/setup.sh" >> ~/.bashrc
```

## Basic Gazebo Concepts

### Worlds
Worlds in Gazebo define the environment where robots operate. They contain:
- Static and dynamic objects
- Lighting conditions
- Physics properties
- Terrain and surfaces

### Models
Models represent physical objects in the simulation:
- Robots
- Objects
- Sensors
- Actuators

### Plugins
Plugins extend Gazebo's functionality:
- Control plugins
- Sensor plugins
- GUI plugins
- Physics plugins

## Running Your First Simulation

### Launch Gazebo with an Empty World

```bash
gz sim -r -v 4 empty.sdf
```

### Launch with a Predefined World

```bash
gz sim -r -v 4 warehouse.sdf
```

## Creating a Custom World

### World File Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define physics -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Place objects -->
    <model name="my_robot">
      <pose>0 0 0.5 0 0 0</pose>
      <include>
        <uri>model://my_robot_model</uri>
      </include>
    </model>
  </world>
</sdf>
```

## Integrating with ROS 2

### Launch Gazebo with ROS 2 Bridge

```bash
# Terminal 1: Start Gazebo
gz sim -r my_world.sdf

# Terminal 2: Start ROS 2 bridge
source /opt/ros/humble/setup.bash
ros2 run ros_gz_bridge parameter_bridge
```

### Using Gazebo with ROS 2 Launch Files

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('ros_gz_sim'),
                    'launch',
                    'gz_sim.launch.py'
                ])
            ]),
            launch_arguments={'gz_args': '-r empty.sdf'}.items()
        )
    ])
```

## Physics Configuration

### Understanding Physics Engines

Gazebo supports multiple physics engines:
- **ODE**: Open Dynamics Engine (default)
- **Bullet**: Good balance of speed and accuracy
- **DART**: Advanced kinematic and dynamic analysis

### Physics Parameters

Key physics parameters to consider:
- **Max step size**: Smaller values for accuracy, larger for performance
- **Real-time factor**: Target simulation speed relative to real time
- **Update rate**: How often physics are updated

## Robot Modeling for Gazebo

### URDF to SDF Conversion

Gazebo works with SDF (Simulation Description Format), but can import URDF:

```xml
<!-- In your URDF, add Gazebo-specific tags -->
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>
```

### Adding Sensors

```xml
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
```

## Simulation Workflows

### Basic Workflow

1. **Design Robot Model**: Create URDF/SDF model of your robot
2. **Create World**: Design the simulation environment
3. **Configure Physics**: Set appropriate physics parameters
4. **Launch Simulation**: Start Gazebo with your world
5. **Control Robot**: Interface with ROS 2 to control the robot
6. **Collect Data**: Use sensors to gather simulation data
7. **Analyze Results**: Process and analyze simulation results

### Example: Mobile Robot Simulation

```bash
# 1. Launch simulation
gz sim -r -v 4 empty.sdf

# 2. Spawn robot (if not in world file)
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf

# 3. Control robot
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.0}, angular: {z: 0.5}}'
```

## Troubleshooting Tips

- If simulation runs slowly, increase max_step_size or decrease update rate
- If robot falls through the ground, check collision properties and physics parameters
- If sensors don't publish data, verify ROS 2 bridge is running
- Use `gz topic -l` to list available topics
- Check Gazebo logs for detailed error information
- Ensure your robot model has proper inertial properties
- Verify that all joint limits and dynamics are properly defined

## Performance Optimization

### Graphics Settings
- Adjust rendering quality based on your hardware
- Use simpler meshes for real-time simulation
- Limit the number of active sensors

### Physics Settings
- Balance accuracy with performance based on your needs
- Use appropriate solver parameters
- Consider fixed-step vs adaptive-step integration

## Troubleshooting Tips for Simulation Workflows

- If simulation runs slowly, increase max_step_size or decrease update rate in physics settings
- If robot falls through the ground, check collision properties and physics parameters
- If sensors don't publish data, verify ROS 2 bridge is running and topics are correctly mapped
- Use `gz topic -l` to list available topics and `gz service -l` to list available services
- Check Gazebo logs for detailed error information: `~/.gazebo/logs/`
- Ensure your robot model has proper inertial properties and collision geometries
- Verify that all joint limits and dynamics are properly defined in your URDF/SDF
- If controllers don't respond, check that the controller manager is running and properly configured
- For sensor issues, verify that sensor plugins are correctly configured and publishing to expected topics
- Use `ros2 run gazebo_ros spawn_entity.py` to debug robot spawning issues

## Performance Optimization

### Graphics Settings
- Adjust rendering quality based on your hardware capabilities
- Use simpler meshes for real-time simulation
- Limit the number of active sensors to reduce computational load

### Physics Settings
- Balance accuracy with performance based on your specific needs
- Use appropriate solver parameters (step size, update rate)
- Consider fixed-step vs adaptive-step integration for different use cases

## Cross-References

For integration with ROS 2, see:
- [ROS 2 Fundamentals](../ros2-core/index.md) - Core concepts for ROS 2 integration
- [ROS 2 Setup Instructions](../ros2-core/setup-instructions.md) - Complete setup guide
- [Unity Visualization](../unity/index.md) - Alternative simulation environment

## Summary

Gazebo provides a powerful simulation environment for robotics development, allowing you to test algorithms and robot designs before deploying to real hardware. Understanding how to configure and use Gazebo effectively is crucial for robotics development and testing.