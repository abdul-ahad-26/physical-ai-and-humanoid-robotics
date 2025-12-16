---
title: ROS 2 Setup Instructions
sidebar_position: 4.4
description: Detailed instructions for setting up ROS 2 development environment
---

# ROS 2 Setup Instructions

## Learning Objectives

- Install ROS 2 on your development system
- Configure the ROS 2 environment
- Verify the installation works correctly
- Understand ROS 2 workspace structure
- Set up development tools and utilities

## System Requirements

### Supported Operating Systems
- Ubuntu 22.04 (Jammy Jellyfish) - Recommended
- Ubuntu 20.04 (Focal Fossa)
- Windows 10/11 (with WSL2)
- macOS (experimental support)

### Minimum Hardware Requirements
- 8GB RAM (16GB recommended)
- 100GB free disk space
- Multi-core processor
- Network connection for package installation

## Installing ROS 2 Humble Hawksbill

### On Ubuntu 22.04

#### 1. Set Locale
Ensure your locale is set to UTF-8:
```bash
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US.UTF-8
```

#### 2. Add ROS 2 APT Repository
```bash
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

#### 3. Install ROS 2 Packages
```bash
sudo apt update
sudo apt install ros-humble-desktop
```

#### 4. Install Additional Tools
```bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
```

#### 5. Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

## Environment Setup

### Sourcing the ROS 2 Environment

For temporary use in the current terminal:
```bash
source /opt/ros/humble/setup.bash
```

For persistent use across terminals, add to your `.bashrc`:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### Verify Installation
```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

You should see the ROS 2 version number.

## Creating a ROS 2 Workspace

### 1. Create Workspace Directory
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### 2. Build the Workspace
```bash
colcon build
```

### 3. Source the Workspace
```bash
source ~/ros2_ws/install/setup.bash
```

To source the workspace automatically, add to your `.bashrc`:
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Essential ROS 2 Tools

### Package Management
```bash
# List all packages
ros2 pkg list

# Create a new package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_package

# Build packages
cd ~/ros2_ws
colcon build --packages-select my_robot_package
```

### Node Management
```bash
# List active nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>
```

### Topic Management
```bash
# List topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo <topic_name> <msg_type>

# Publish to a topic
ros2 topic pub <topic_name> <msg_type> <args>
```

### Service Management
```bash
# List services
ros2 service list

# Call a service
ros2 service call <service_name> <service_type> <request_args>
```

## Development Tools

### ROS 2 Doctor
Check your ROS 2 installation:
```bash
ros2 doctor
```

### RQT Tools
Install visualization tools:
```bash
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins
```

Run RQT:
```bash
rqt
```

### RViz2
Install and run the 3D visualization tool:
```bash
sudo apt install ros-humble-rviz2
rviz2
```

## Python Development Setup

### Virtual Environment (Recommended)
```bash
# Install virtual environment tools
sudo apt install python3-venv

# Create virtual environment
python3 -m venv ~/ros2_env

# Activate virtual environment
source ~/ros2_env/bin/activate

# Install ROS 2 Python packages in virtual environment
pip install -U pip
pip install ros-humble-ros-base
```

### IDE Configuration

For VS Code, install the following extensions:
- Python
- ROS
- C/C++

## Troubleshooting Common Issues

### Environment Not Sourced
**Problem**: Commands like `ros2` are not found.
**Solution**: Ensure you've sourced the ROS 2 environment:
```bash
source /opt/ros/humble/setup.bash
```

### Permission Issues
**Problem**: Cannot access ROS topics/nodes from different terminals.
**Solution**: Set ROS_DOMAIN_ID consistently across terminals:
```bash
export ROS_DOMAIN_ID=0
```

### Package Installation Issues
**Problem**: APT cannot verify packages.
**Solution**: Update your package lists and keys:
```bash
sudo apt update
sudo apt upgrade
```

### Workspace Build Issues
**Problem**: `colcon build` fails.
**Solution**: Check dependencies and ensure proper workspace structure:
```bash
# Clean build artifacts
rm -rf build/ install/ log/

# Rebuild
colcon build
```

## Development Workflow

### 1. Create a New Package
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_package --dependencies rclpy std_msgs geometry_msgs
```

### 2. Develop Your Nodes
Add your Python scripts to `my_robot_package/my_robot_package/`

### 3. Update setup.py
Ensure your scripts are listed in the `entry_points` section of `setup.py`:
```python
entry_points={
    'console_scripts': [
        'my_node = my_robot_package.my_node:main',
    ],
},
```

### 4. Build and Test
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
ros2 run my_robot_package my_node
```

## Advanced Configuration

### Custom ROS_DOMAIN_ID
For multiple robots or isolated systems:
```bash
export ROS_DOMAIN_ID=42
```

### Network Configuration
For multi-machine setups:
```bash
export ROS_LOCALHOST_ONLY=0
export ROS_IP=192.168.1.100  # Your machine's IP
```

## Summary

This setup provides a complete ROS 2 development environment with all essential tools. Remember to always source your ROS 2 environment before working with ROS 2 commands, and consider using workspaces to organize your projects effectively.