---
title: ROS 2 Hands-On Exercises
sidebar_position: 4.2
description: Practical exercises to reinforce ROS 2 fundamentals
---

# ROS 2 Hands-On Exercises

## Learning Objectives

- Create and build a simple ROS 2 package
- Implement a publisher and subscriber
- Use ROS 2 command-line tools effectively
- Understand Quality of Service (QoS) settings
- Work with launch files

## Prerequisites

- ROS 2 environment properly sourced
- Basic understanding of Linux command line
- Python knowledge (for this example)

## Exercise 1: Creating Your First ROS 2 Package

### Task
Create a simple ROS 2 package that contains a publisher and subscriber.

### Steps

1. Create a new workspace directory:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

2. Create a new package:
   ```bash
   ros2 pkg create --build-type ament_python my_first_ros2_pkg
   ```

3. Navigate to the package directory:
   ```bash
   cd my_first_ros2_pkg
   ```

4. Create a publisher script in `my_first_ros2_pkg/my_first_ros2_pkg/`:
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String


   class MinimalPublisher(Node):

       def __init__(self):
           super().__init__('minimal_publisher')
           self.publisher_ = self.create_publisher(String, 'topic', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = 'Hello World: %d' % self.i
           self.publisher_.publish(msg)
           self.get_logger().info('Publishing: "%s"' % msg.data)
           self.i += 1


   def main(args=None):
       rclpy.init(args=args)

       minimal_publisher = MinimalPublisher()

       rclpy.spin(minimal_publisher)

       # Destroy the node explicitly
       minimal_publisher.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

5. Create a subscriber script in the same directory:
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String


   class MinimalSubscriber(Node):

       def __init__(self):
           super().__init__('minimal_subscriber')
           self.subscription = self.create_subscription(
               String,
               'topic',
               self.listener_callback,
               10)
           self.subscription  # prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info('I heard: "%s"' % msg.data)


   def main(args=None):
       rclpy.init(args=args)

       minimal_subscriber = MinimalSubscriber()

       rclpy.spin(minimal_subscriber)

       # Destroy the node explicitly
       minimal_subscriber.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

6. Make the scripts executable:
   ```bash
   chmod +x my_first_ros2_pkg/publisher_member_function.py
   chmod +x my_first_ros2_pkg/subscriber_member_function.py
   ```

7. Update the `setup.py` file to include entry points for your scripts:
   ```python
   entry_points={
       'console_scripts': [
           'talker = my_first_ros2_pkg.publisher_member_function:main',
           'listener = my_first_ros2_pkg.subscriber_member_function:main',
       ],
   },
   ```

8. Build the package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_first_ros2_pkg
   ```

9. Source the workspace:
   ```bash
   source install/setup.bash
   ```

### Expected Results
You should be able to run the publisher and subscriber nodes and see messages being published and received.

## Exercise 2: Using ROS 2 Tools

### Task
Explore ROS 2 command-line tools to understand your system's state.

### Steps

1. Start your publisher node from Exercise 1:
   ```bash
   ros2 run my_first_ros2_pkg talker
   ```

2. In a new terminal, use various ROS 2 tools:
   ```bash
   # List active nodes
   ros2 node list

   # List active topics
   ros2 topic list

   # Get info about a specific topic
   ros2 topic info /topic

   # Echo messages from the topic
   ros2 topic echo /topic std_msgs/msg/String
   ```

3. Start the subscriber node in another terminal:
   ```bash
   ros2 run my_first_ros2_pkg listener
   ```

### Expected Results
You should see the publisher sending messages and the subscriber receiving them. The various ROS 2 tools should show the active nodes and topics.

## Exercise 3: Quality of Service Settings

### Task
Experiment with different QoS settings to understand their impact.

### Steps

1. Modify the publisher to use a different QoS profile:
   ```python
   # In the publisher, change the QoS settings:
   from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

   qos_profile = QoSProfile(
       reliability=ReliabilityPolicy.RELIABLE,
       history=HistoryPolicy.KEEP_LAST,
       depth=10
   )

   self.publisher_ = self.create_publisher(String, 'topic', qos_profile)
   ```

2. Run experiments with different QoS settings and observe the behavior.

### Expected Results
Different QoS settings will affect message delivery and buffering behavior.

## Exercise 4: Launch Files

### Task
Create a launch file to start both nodes simultaneously.

### Steps

1. Create a `launch` directory in your package:
   ```bash
   mkdir my_first_ros2_pkg/launch
   ```

2. Create a launch file `my_first_ros2_pkg/launch/talker_listener.launch.py`:
   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node


   def generate_launch_description():
       return LaunchDescription([
           Node(
               package='my_first_ros2_pkg',
               executable='talker',
               name='talker',
           ),
           Node(
               package='my_first_ros2_pkg',
               executable='listener',
               name='listener',
           ),
       ])
   ```

3. Update the `setup.py` file to include the launch directory:
   ```python
   import os
   from glob import glob
   import setuptools

   # ... existing code ...

   package_data={'': ['package.xml']},
   data_files=[
       ('share/ament_index/resource_index/packages',
           ['resource/my_first_ros2_pkg']),
       ('share/my_first_ros2_pkg', ['package.xml']),
       # Include all launch files
       (os.path.join('share', 'my_first_ros2_pkg', 'launch'),
        glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
   ],
   ```

4. Rebuild the package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_first_ros2_pkg
   source install/setup.bash
   ```

5. Run the launch file:
   ```bash
   ros2 launch my_first_ros2_pkg talker_listener.launch.py
   ```

### Expected Results
Both nodes should start simultaneously, and you should see the publisher and subscriber communicating.

## Troubleshooting Tips

- If packages don't build, ensure you've sourced the ROS 2 environment
- If nodes can't communicate, check that they're on the same ROS_DOMAIN_ID
- Use `ros2 doctor` to diagnose common configuration issues
- Check file permissions on your Python scripts

## Summary

These exercises provided hands-on experience with core ROS 2 concepts including packages, nodes, topics, and launch files. Understanding these fundamentals is essential for developing more complex robotic applications.