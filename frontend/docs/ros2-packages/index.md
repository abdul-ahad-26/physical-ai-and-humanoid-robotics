---
title: ROS 2 Packages & Nodes
sidebar_position: 5.1
description: Understanding ROS 2 packages, nodes, and their role in robotics systems
---

# ROS 2 Packages & Nodes

## Learning Objectives

- Understand the structure and purpose of ROS 2 packages
- Learn how to create, build, and manage ROS 2 packages
- Master the creation and implementation of ROS 2 nodes
- Implement communication between nodes using topics, services, and actions
- Understand package dependencies and management
- Deploy and test ROS 2 packages in robotics applications
- Troubleshoot common package and node issues

## Introduction to ROS 2 Packages

ROS 2 packages are the fundamental building blocks of ROS 2 systems. They contain the source code, launch files, configuration files, and other resources needed to implement specific functionality. Packages provide modularity, reusability, and organization to complex robotics systems.

### Package Structure

A typical ROS 2 package follows this structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata and dependencies
├── setup.py                # Python build configuration
├── setup.cfg               # Installation configuration
├── my_robot_package/       # Python package directory
│   ├── __init__.py         # Makes it a Python package
│   ├── my_node.py          # Node implementation
│   └── my_module.py        # Supporting modules
├── launch/                 # Launch files
│   └── my_robot_launch.py
├── config/                 # Configuration files
│   └── my_params.yaml
├── src/                    # C++ source files
├── include/                # C++ header files
├── test/                   # Unit and integration tests
└── README.md               # Package documentation
```

## Creating ROS 2 Packages

### Using the ros2 pkg Command

The easiest way to create a new ROS 2 package is using the `ros2 pkg create` command:

```bash
# Create a Python-based package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_package --dependencies rclpy std_msgs sensor_msgs geometry_msgs

# Create a C++-based package
ros2 pkg create --build-type ament_cmake my_robot_cpp_package --dependencies rclcpp std_msgs sensor_msgs geometry_msgs
```

### Package.xml Configuration

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.1</version>
  <description>Example robot package for ROS 2</description>
  <maintainer email="maintainer@example.com">Maintainer Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Setup.py Configuration

For Python packages, the `setup.py` file defines the build configuration:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maintainer Name',
    maintainer_email='maintainer@example.com',
    description='Example robot package for ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_robot_node = my_robot_package.my_robot_node:main',
            'sensor_processor = my_robot_package.sensor_processor:main',
            'controller = my_robot_package.controller:main',
        ],
    },
)
```

## Creating ROS 2 Nodes

### Basic Node Structure

A ROS 2 node is a process that performs computation. Here's the basic structure:

```python
#!/usr/bin/env python3

"""
Basic ROS 2 Node Example

This example demonstrates the basic structure of a ROS 2 node.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class BasicRobotNode(Node):
    """
    A basic robot node demonstrating ROS 2 concepts
    """

    def __init__(self):
        # Initialize the node with a name
        super().__init__('basic_robot_node')

        # Create a publisher
        self.publisher = self.create_publisher(String, 'robot_status', 10)

        # Create a subscriber
        self.subscription = self.create_subscription(
            String,
            'robot_commands',
            self.command_callback,
            10
        )

        # Create a timer for periodic tasks
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize node-specific variables
        self.robot_state = 'idle'
        self.command_queue = []

        self.get_logger().info('Basic Robot Node initialized')

    def timer_callback(self):
        """
        Callback function called periodically by the timer
        """
        # Publish status
        msg = String()
        msg.data = f'Robot is {self.robot_state} at {self.get_clock().now().nanoseconds}'
        self.publisher.publish(msg)

        # Process queued commands
        if self.command_queue:
            command = self.command_queue.pop(0)
            self.execute_command(command)

    def command_callback(self, msg):
        """
        Callback function for processing incoming commands
        """
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Add command to queue for processing
        self.command_queue.append(command)

    def execute_command(self, command):
        """
        Execute a specific command
        """
        if command == 'start':
            self.robot_state = 'moving'
            self.get_logger().info('Starting robot movement')
        elif command == 'stop':
            self.robot_state = 'idle'
            self.get_logger().info('Stopping robot movement')
        else:
            self.get_logger().warn(f'Unknown command: {command}')


def main(args=None):
    """
    Main function to run the node
    """
    rclpy.init(args=args)

    basic_robot_node = BasicRobotNode()

    try:
        rclpy.spin(basic_robot_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        basic_robot_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Node with Multiple Communication Patterns

Here's a more advanced node that demonstrates different communication patterns:

```python
#!/usr/bin/env python3

"""
Advanced ROS 2 Node Example

This example demonstrates topics, services, and actions in a single node.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from example_interfaces.srv import SetBool
from example_interfaces.action import Fibonacci


class AdvancedRobotNode(Node):
    """
    Advanced robot node with multiple communication patterns
    """

    def __init__(self):
        super().__init__('advanced_robot_node')

        # QoS Profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)
        self.status_pub = self.create_publisher(String, '/robot_status', qos_profile)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            qos_profile
        )

        # Services
        self.emergency_stop_srv = self.create_service(
            SetBool,
            '/emergency_stop',
            self.emergency_stop_callback
        )

        # Action server
        self.fibonacci_action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci_sequence',
            self.execute_fibonacci_goal,
            goal_callback=self.fibonacci_goal_callback,
            cancel_callback=self.fibonacci_cancel_callback
        )

        # Internal state
        self.is_moving = False
        self.obstacle_detected = False
        self.last_laser_data = None

        self.get_logger().info('Advanced Robot Node initialized')

    def laser_callback(self, msg):
        """
        Process laser scan data
        """
        self.last_laser_data = msg

        # Check for obstacles in front
        if len(msg.ranges) > 0:
            front_range = min(msg.ranges[len(msg.ranges)//2 - 10:len(msg.ranges)//2 + 10])

            if front_range < 0.5:  # Obstacle within 0.5m
                self.obstacle_detected = True
                if self.is_moving:
                    self.get_logger().warn('Obstacle detected! Stopping robot.')
                    self.stop_robot()
            else:
                self.obstacle_detected = False

    def emergency_stop_callback(self, request, response):
        """
        Handle emergency stop service requests
        """
        if request.data:
            self.get_logger().warn('Emergency stop activated!')
            self.stop_robot()
            response.success = True
            response.message = 'Robot stopped due to emergency'
        else:
            response.success = False
            response.message = 'Emergency stop deactivation not supported'

        return response

    def fibonacci_goal_callback(self, goal_request):
        """
        Accept or reject goal requests
        """
        self.get_logger().info('Received Fibonacci goal request')

        # Accept all goals
        return GoalResponse.ACCEPT

    def fibonacci_cancel_callback(self, goal_handle):
        """
        Accept or reject goal cancel requests
        """
        self.get_logger().info('Received cancel request')

        # Accept all cancel requests
        return CancelResponse.ACCEPT

    async def execute_fibonacci_goal(self, goal_handle):
        """
        Execute the Fibonacci action goal
        """
        self.get_logger().info('Executing Fibonacci goal')

        # Notify goal accepted
        goal_handle.succeed()

        # Create feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()

        # Generate Fibonacci sequence
        sequence = [0, 1]
        feedback_msg.sequence = sequence

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Fibonacci goal canceled')
                goal_handle.canceled()

                result_msg.sequence = feedback_msg.sequence
                return result_msg

            # Update sequence
            sequence.append(sequence[i] + sequence[i-1])
            feedback_msg.sequence = sequence

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Sleep briefly to simulate work
            await rclpy.asyncio.sleep(0.5)

        # Complete goal
        goal_handle.succeed()
        result_msg.sequence = feedback_msg.sequence

        self.get_logger().info(f'Fibonacci goal completed: {result_msg.sequence}')
        return result_msg

    def stop_robot(self):
        """
        Stop the robot by publishing zero velocity
        """
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        self.is_moving = False
        self.publish_status('stopped')


    def publish_status(self, status):
        """
        Publish robot status
        """
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)


def main(args=None):
    """
    Main function to run the advanced node
    """
    rclpy.init(args=args)

    node = AdvancedRobotNode()

    # Use MultiThreadedExecutor to handle multiple callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch Files for Package Management

### Python Launch Files

Launch files allow you to start multiple nodes with a single command:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch Arguments
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='my_robot',
        description='Robot namespace'
    )

    # Robot controller node
    robot_controller = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='controller',
        namespace=namespace,
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_robot_package'),
                'config',
                'robot_params.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Sensor processor node
    sensor_processor = Node(
        package='my_robot_package',
        executable='sensor_processor',
        name='sensor_processor',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(namespace_arg)

    # Add nodes
    ld.add_action(robot_controller)
    ld.add_action(sensor_processor)

    return ld
```

### YAML Launch Files

For more complex launch configurations, you can also use YAML:

```yaml
# launch/my_robot_system.yaml
launch:
  - node:
      pkg: "my_robot_package"
      exec: "robot_controller"
      name: "controller"
      namespace: "my_robot"
      parameters:
        - "config/robot_params.yaml"
      remappings:
        - from: "/cmd_vel"
          to: "/my_robot/cmd_vel"

  - node:
      pkg: "my_robot_package"
      exec: "sensor_processor"
      name: "sensor_processor"
      namespace: "my_robot"
      parameters:
        - use_sim_time: false
```

## Package Dependencies and Management

### Managing Dependencies

Dependencies in ROS 2 are managed through the `package.xml` file. Common dependency types include:

- **build_depend**: Needed to build the package
- **exec_depend**: Needed to run the package
- **test_depend**: Needed for testing
- **depend**: Shorthand for both build and exec dependencies

### Common Package Dependencies

For robotics applications, common dependencies include:

```xml
<!-- Essential ROS 2 packages -->
<depend>rclpy</depend>  <!-- Python ROS client library -->
<depend>rclcpp</depend> <!-- C++ ROS client library -->

<!-- Message packages -->
<depend>std_msgs</depend>
<depend>geometry_msgs</depend>
<depend>sensor_msgs</depend>
<depend>nav_msgs</depend>
<depend>action_msgs</depend>

<!-- Common robotics packages -->
<depend>tf2_ros</depend>
<depend>robot_state_publisher</depend>
<depend>joint_state_publisher</depend>

<!-- Navigation -->
<depend>nav2_msgs</depend>
<depend>nav2_util</depend>

<!-- Simulation -->
<depend>gazebo_ros</depend>
<depend>ros_gz_bridge</depend>
```

## Building and Installing Packages

### Building Packages

To build a ROS 2 package:

```bash
# Navigate to workspace
cd ~/ros2_ws

# Build specific package
colcon build --packages-select my_robot_package

# Build with additional options
colcon build --packages-select my_robot_package --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build all packages
colcon build
```

### Sourcing the Workspace

After building, source the workspace to use the packages:

```bash
# Source the workspace
source install/setup.bash

# Or add to your .bashrc for automatic sourcing
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Testing and Debugging

### Unit Testing

Create unit tests for your nodes:

```python
# test/test_my_robot_node.py
import unittest
import rclpy
from my_robot_package.my_robot_node import BasicRobotNode


class TestBasicRobotNode(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.node = BasicRobotNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_initialization(self):
        """Test that node initializes correctly"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.robot_state, 'idle')

    def test_command_processing(self):
        """Test command processing"""
        self.node.command_queue.append('start')
        self.node.execute_command('start')
        self.assertEqual(self.node.robot_state, 'moving')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
```

### Debugging Tools

Common debugging approaches for ROS 2 packages:

```bash
# Check node status
ros2 node list
ros2 node info <node_name>

# Check topic status
ros2 topic list
ros2 topic echo <topic_name>
ros2 topic info <topic_name>

# Check service status
ros2 service list
ros2 service call <service_name> <service_type> <args>

# Check action status
ros2 action list
ros2 action info <action_name>

# Use ros2 doctor for diagnostics
ros2 doctor

# Use rqt tools for visualization
rqt
rqt_graph  # Shows node connections
```

## Best Practices for Package Development

### Code Organization

- **Separate concerns**: Keep different functionalities in separate modules
- **Use descriptive names**: Choose clear, meaningful names for packages, nodes, and topics
- **Follow naming conventions**: Use underscores for Python packages (snake_case)
- **Document everything**: Include docstrings and comments

### Configuration Management

- **Externalize parameters**: Use YAML files for configurable parameters
- **Use parameter declarations**: Declare parameters in your nodes
- **Provide defaults**: Always provide sensible default values

### Error Handling

- **Graceful degradation**: Handle errors without crashing the node
- **Log appropriately**: Use appropriate log levels (debug, info, warn, error)
- **Validate inputs**: Check message validity before processing

## Troubleshooting Tips

- If packages don't build, ensure dependencies are installed and properly declared
- If nodes don't communicate, check that namespaces and topic names match
- If parameters don't load, verify file paths and YAML syntax
- Use `ros2 doctor` to diagnose common configuration issues
- Check that the workspace is properly sourced
- Verify that nodes are in the same ROS domain if on different machines
- If actions seem stuck, check that goal callbacks are returning proper responses
- For memory issues, monitor node resource usage with `ros2 lifecycle` tools

## Cross-References

For related concepts, see:
- [ROS 2 Fundamentals](../ros2-core/index.md) - For basic ROS 2 concepts
- [Simulation Integration](../gazebo/index.md) - For simulation packages
- [Navigation Systems](../isaac-ros/index.md) - For navigation-specific packages
- [Development Tools](../ros2-core/setup-instructions.md) - For development setup

## Summary

ROS 2 packages and nodes form the backbone of robotics software development. Understanding how to properly structure, create, and manage packages is essential for building robust, maintainable robotics systems. The modular approach of packages allows for reusability and collaboration, while the various communication patterns (topics, services, actions) enable flexible system architectures.