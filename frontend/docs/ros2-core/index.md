---
title: ROS 2 Fundamentals
sidebar_position: 4.1
description: Introduction to ROS 2 concepts, architecture, and basic usage
---

# ROS 2 Fundamentals

## Learning Objectives

- Understand the core concepts and architecture of ROS 2
- Learn how to create and manage ROS 2 workspaces
- Master the fundamentals of nodes, topics, services, and actions
- Execute basic ROS 2 commands and tools
- Understand the differences between ROS 1 and ROS 2
- Implement simple ROS 2 nodes in Python
- Use Quality of Service (QoS) settings appropriately
- Create and use launch files for multi-node systems

## Introduction to ROS 2

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

ROS 2 addresses many of the limitations of ROS 1, including:
- Real-time support
- Multi-robot systems
- Non-ideal networks
- Commercial product support
- Platform diversity

## ROS 2 Architecture

ROS 2 is built on a DDS (Data Distribution Service) implementation, which provides a publish-subscribe communication pattern. This architecture allows for:

- Distributed computing
- Language independence
- Real-time capabilities
- Quality of Service (QoS) policies

<!-- ![ROS 2 Architecture Diagram](/assets/diagrams/ros2-architecture.png "ROS 2 Architecture Overview") -->

*Figure: ROS 2 Architecture showing nodes, topics, services, and the DDS middleware layer. (diagram to be added)*

### Key Components

- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Messages**: Data structures exchanged between nodes
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous goal-oriented communication
- **Parameters**: Configuration values that can be changed at runtime

## Setting Up Your First ROS 2 Workspace

### Creating a Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### Sourcing ROS 2 Environment

Before working with ROS 2, you need to source the setup script:

```bash
source /opt/ros/humble/setup.bash  # Replace 'humble' with your ROS 2 distribution
```

## Basic ROS 2 Commands

### Finding Available Packages

```bash
ros2 pkg list
```

### Creating a New Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_package
```

### Running Nodes

```bash
# List available nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>
```

### Working with Topics

```bash
# List all topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo <topic_name> <message_type>

# Publish a message to a topic
ros2 topic pub <topic_name> <message_type> <args>
```

### Working with Services

```bash
# List all services
ros2 service list

# Call a service
ros2 service call <service_name> <service_type> <request_args>
```

## Quality of Service (QoS) in ROS 2

QoS profiles allow you to tune the communication behavior between publishers and subscribers. Key QoS settings include:

- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local
- **History**: Keep last vs. keep all
- **Depth**: Number of messages to keep in history

## Setup Instructions

### Installing ROS 2

For Ubuntu 22.04 (Jammy) with ROS 2 Humble Hawksbill:

1. Set locale:
   ```bash
   locale  # check for UTF-8
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US.UTF-8
   ```

2. Add ROS 2 apt repository:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   ```

4. Install colcon build tool:
   ```bash
   sudo apt install python3-colcon-common-extensions
   ```

5. Source the ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

### Creating a Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build packages
colcon build

# Source the workspace
source install/setup.bash
```

## Command-Line Workflows

### Basic Navigation

```bash
# List available commands
ros2 --help

# Get help for specific command
ros2 node --help
```

### Package Management

```bash
# List all packages
ros2 pkg list

# Find a specific package
ros2 pkg executables <package_name>

# Create a new package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_package
```

### Node Operations

```bash
# List active nodes
ros2 node list

# Get information about a node
ros2 node info <node_name>

# Run a node
ros2 run <package_name> <executable_name>
```

### Topic Operations

```bash
# List topics
ros2 topic list

# Show topic information
ros2 topic info <topic_name>

# Echo messages from a topic
ros2 topic echo <topic_name> <msg_type>

# Publish to a topic
ros2 topic pub <topic_name> <msg_type> <args>
```

### Service Operations

```bash
# List services
ros2 service list

# Call a service
ros2 service call <service_name> <service_type> <request_args>

# Find services by type
ros2 service type <service_name>
```

## ROS 2 Launch Files

Launch files allow you to start multiple nodes with a single command:

```xml
<launch>
  <node pkg="my_robot_package" exec="robot_controller" name="controller"/>
  <node pkg="my_robot_package" exec="sensor_processor" name="processor"/>
</launch>
```

## Troubleshooting Tips

- Ensure your ROS 2 environment is properly sourced before running commands
- Check that the correct ROS_DOMAIN_ID is set if running multiple systems
- Use `ros2 doctor` to diagnose common configuration issues
- Verify that required packages are installed and built
- Make sure your Python scripts have execute permissions (`chmod +x script.py`)
- If nodes can't communicate, verify they're on the same network and using compatible QoS settings
- Check that your workspace is properly built with `colcon build`
- Ensure your firewall isn't blocking DDS communication ports

## Summary

ROS 2 provides a robust framework for developing robot applications with improved real-time capabilities, multi-robot support, and commercial-grade features. Understanding these fundamentals is crucial for building more complex robotic systems.