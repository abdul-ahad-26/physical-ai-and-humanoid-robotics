---
title: VLA Systems Hands-On Exercises
sidebar_position: 9.2
description: Practical exercises for Vision-Language-Action system implementation
---

# VLA Systems Hands-On Exercises

## Learning Objectives

- Implement a basic VLA system architecture
- Integrate vision and language components
- Create action planning modules
- Evaluate VLA system performance
- Troubleshoot common VLA system issues
- Deploy VLA components on a simulated robot

## Prerequisites

- ROS 2 environment with Gazebo simulation
- Python knowledge for AI/ML frameworks
- Understanding of computer vision concepts
- Basic knowledge of natural language processing
- Robot simulation environment

## Exercise 1: Setting Up the VLA Framework

### Task
Create a basic VLA system framework that can receive language commands and process visual input.

### Steps

1. Create a new ROS 2 package for VLA components:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python vla_systems --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge
   ```

2. Create the VLA system structure in `vla_systems/vla_systems/`:
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, CameraInfo
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   import cv2
   from cv_bridge import CvBridge
   import numpy as np


   class VLASystem(Node):
       """
       Basic Vision-Language-Action system framework
       """

       def __init__(self):
           super().__init__('vla_system')

           # Initialize CV bridge
           self.bridge = CvBridge()

           # Create subscribers
           self.image_sub = self.create_subscription(
               Image,
               '/camera/image_raw',
               self.image_callback,
               10
           )

           self.command_sub = self.create_subscription(
               String,
               '/vla_command',
               self.command_callback,
               10
           )

           # Create publisher for robot commands
           self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

           # Initialize internal state
           self.current_image = None
           self.pending_command = None

           self.get_logger().info('VLA System initialized')

       def image_callback(self, msg):
           """
           Process incoming camera images
           """
           try:
               self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
               self.process_command_if_ready()
           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def command_callback(self, msg):
           """
           Process incoming language commands
           """
           self.pending_command = msg.data
           self.process_command_if_ready()

       def process_command_if_ready(self):
           """
           Process command when both image and command are available
           """
           if self.current_image is not None and self.pending_command is not None:
               self.get_logger().info(f'Processing command: {self.pending_command}')
               self.execute_command(self.pending_command, self.current_image)
               self.pending_command = None

       def execute_command(self, command, image):
           """
           Execute the given command based on the image
           """
           # Simple example: move forward if command contains "forward"
           if "forward" in command.lower():
               msg = Twist()
               msg.linear.x = 0.5  # Move forward at 0.5 m/s
               self.cmd_pub.publish(msg)
               self.get_logger().info('Moving forward')
           elif "stop" in command.lower():
               msg = Twist()
               self.cmd_pub.publish(msg)  # Stop
               self.get_logger().info('Stopping')

       def cleanup(self):
           """
           Cleanup function
           """
           stop_msg = Twist()
           self.cmd_pub.publish(stop_msg)


   def main(args=None):
       rclpy.init(args=args)

       vla_system = VLASystem()

       try:
           rclpy.spin(vla_system)
       except KeyboardInterrupt:
           pass
       finally:
           vla_system.cleanup()
           vla_system.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. Update the `setup.py` file to include the executable:
   ```python
   entry_points={
       'console_scripts': [
           'vla_system = vla_systems.vla_system:main',
       ],
   },
   ```

4. Build the package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select vla_systems
   source install/setup.bash
   ```

### Expected Results
You should have a basic VLA system that can receive both camera images and language commands, and respond accordingly.

## Exercise 2: Vision Component Integration

### Task
Enhance the VLA system with basic computer vision capabilities to detect objects in the environment.

### Steps

1. Install required dependencies:
   ```bash
   pip3 install opencv-python numpy
   ```

2. Create a vision processing module `vla_systems/vla_systems/vision_processor.py`:
   ```python
   import cv2
   import numpy as np


   class VisionProcessor:
       """
       Basic vision processing for VLA systems
       """

       def __init__(self):
           self.object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

       def detect_objects(self, image):
           """
           Detect objects in the image using basic techniques
           """
           gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           objects = self.object_cascade.detectMultiScale(gray, 1.1, 4)
           return objects

       def detect_color_objects(self, image, lower_color, upper_color):
           """
           Detect objects of a specific color range
           """
           hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
           mask = cv2.inRange(hsv, lower_color, upper_color)
           contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           return contours

       def get_object_position(self, image, object_contours):
           """
           Calculate the position of detected objects
           """
           positions = []
           for contour in object_contours:
               M = cv2.moments(contour)
               if M["m00"] != 0:
                   cX = int(M["m10"] / M["m00"])
                   cY = int(M["m01"] / M["m00"])
                   positions.append((cX, cY))
           return positions
   ```

3. Update the VLA system to use the vision processor:
   ```python
   # Add to imports
   from .vision_processor import VisionProcessor

   # In the VLASystem class constructor
   def __init__(self):
       # ... existing code ...
       self.vision_processor = VisionProcessor()

   # In the execute_command method
   def execute_command(self, command, image):
       if "find red" in command.lower():
           # Define red color range in HSV
           lower_red = np.array([0, 50, 50])
           upper_red = np.array([10, 255, 255])
           red_contours = self.vision_processor.detect_color_objects(image, lower_red, upper_red)

           if red_contours:
               positions = self.vision_processor.get_object_position(image, red_contours)
               self.get_logger().info(f'Found {len(positions)} red objects at positions: {positions}')

               # Move towards the first detected object
               if positions:
                   center_x, center_y = positions[0]
                   img_center_x = image.shape[1] / 2

                   msg = Twist()
                   msg.linear.x = 0.3  # Move forward slowly

                   # Turn towards the object if it's not centered
                   if center_x < img_center_x - 50:
                       msg.angular.z = 0.3
                   elif center_x > img_center_x + 50:
                       msg.angular.z = -0.3
                   else:
                       msg.angular.z = 0.0

                   self.cmd_pub.publish(msg)
   ```

### Expected Results
The robot should be able to detect objects of specific colors and move towards them based on language commands.

## Exercise 3: Language Processing Enhancement

### Task
Integrate a simple natural language processing component to better understand commands.

### Steps

1. Create a language processor module `vla_systems/vla_systems/language_processor.py`:
   ```python
   import re


   class LanguageProcessor:
       """
       Basic language processing for VLA systems
       """

       def __init__(self):
           # Define command patterns
           self.command_patterns = {
               'move_forward': [r'go forward', r'move forward', r'go ahead', r'move ahead'],
               'move_backward': [r'go backward', r'move backward', r'go back', r'move back'],
               'turn_left': [r'turn left', r'rotate left', r'go left'],
               'turn_right': [r'turn right', r'rotate right', r'go right'],
               'stop': [r'stop', r'halt', r'freeze'],
               'find_object': [r'find (.+)', r'look for (.+)', r'locate (.+)']
           }

       def parse_command(self, command):
           """
           Parse a natural language command and extract intent
           """
           command_lower = command.lower().strip()

           for intent, patterns in self.command_patterns.items():
               for pattern in patterns:
                   match = re.search(pattern, command_lower)
                   if match:
                       # Return the intent and any captured groups
                       return intent, match.groups() if match.groups() else None

           # If no pattern matches, return unknown
           return 'unknown', None

       def extract_object_name(self, command):
           """
           Extract object name from commands like "find the red ball"
           """
           patterns = [
               r'find (.+)',
               r'look for (.+)',
               r'locate (.+)',
               r'pick up (.+)',
               r'get (.+)'
           ]

           for pattern in patterns:
               match = re.search(pattern, command.lower())
               if match:
                   return match.group(1).strip()

           return None
   ```

2. Update the VLA system to use the language processor:
   ```python
   # Add to imports
   from .language_processor import LanguageProcessor

   # In the VLASystem class constructor
   def __init__(self):
       # ... existing code ...
       self.language_processor = LanguageProcessor()

   # Update the execute_command method
   def execute_command(self, command, image):
       intent, args = self.language_processor.parse_command(command)

       if intent == 'move_forward':
           msg = Twist()
           msg.linear.x = 0.5
           self.cmd_pub.publish(msg)
           self.get_logger().info('Moving forward')
       elif intent == 'move_backward':
           msg = Twist()
           msg.linear.x = -0.5
           self.cmd_pub.publish(msg)
           self.get_logger().info('Moving backward')
       elif intent == 'turn_left':
           msg = Twist()
           msg.angular.z = 0.5
           self.cmd_pub.publish(msg)
           self.get_logger().info('Turning left')
       elif intent == 'turn_right':
           msg = Twist()
           msg.angular.z = -0.5
           self.cmd_pub.publish(msg)
           self.get_logger().info('Turning right')
       elif intent == 'stop':
           msg = Twist()
           self.cmd_pub.publish(msg)
           self.get_logger().info('Stopping')
       elif intent == 'find_object':
           if args:
               object_name = args[0]
               self.get_logger().info(f'Trying to find: {object_name}')
               # Here you would implement object detection based on the name
               # For now, we'll just look for red objects as an example
               self.find_object_by_color(image, object_name)
       else:
           self.get_logger().info(f'Unknown command: {command}')

   def find_object_by_color(self, image, object_name):
       """
       Simple color-based object finding (example implementation)
       """
       # Define color ranges based on object name
       if 'red' in object_name.lower():
           lower_color = np.array([0, 50, 50])
           upper_color = np.array([10, 255, 255])
       elif 'blue' in object_name.lower():
           lower_color = np.array([100, 50, 50])
           upper_color = np.array([130, 255, 255])
       elif 'green' in object_name.lower():
           lower_color = np.array([40, 50, 50])
           upper_color = np.array([80, 255, 255])
       else:
           # Default to red if color not specified
           lower_color = np.array([0, 50, 50])
           upper_color = np.array([10, 255, 255])

       color_contours = self.vision_processor.detect_color_objects(image, lower_color, upper_color)

       if color_contours:
           positions = self.vision_processor.get_object_position(image, color_contours)
           self.get_logger().info(f'Found {len(positions)} {object_name} objects')

           # Move towards the first detected object
           if positions:
               center_x, center_y = positions[0]
               img_center_x = image.shape[1] / 2

               msg = Twist()
               msg.linear.x = 0.3  # Move forward slowly

               # Turn towards the object if it's not centered
               if center_x < img_center_x - 50:
                   msg.angular.z = 0.3
               elif center_x > img_center_x + 50:
                   msg.angular.z = -0.3
               else:
                   msg.angular.z = 0.0

               self.cmd_pub.publish(msg)
       else:
           self.get_logger().info(f'No {object_name} objects found')
   ```

### Expected Results
The VLA system should now better understand natural language commands and respond appropriately.

## Exercise 4: Integration with Gazebo Simulation

### Task
Connect the VLA system to a Gazebo simulation environment.

### Steps

1. Create a launch file `vla_systems/launch/vla_system.launch.py`:
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare


   def generate_launch_description():
       # Launch Arguments
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       # Launch Gazebo
       gazebo = IncludeLaunchDescription(
           PythonLaunchDescriptionSource([
               PathJoinSubstitution([
                   FindPackageShare('ros_gz_sim'),
                   'launch',
                   'gz_sim.launch.py'
               ])
           ]),
           launch_arguments={'gz_args': '-r empty.sdf'}.items()
       )

       # Spawn a simple robot (if not in the world file)
       spawn_robot = Node(
           package='ros_gz_sim',
           executable='create',
           arguments=[
               '-name', 'vla_robot',
               '-topic', 'robot_description',
               '-x', '0', '-y', '0', '-z', '0.2'
           ],
           output='screen'
       )

       # Launch the VLA system
       vla_system = Node(
           package='vla_systems',
           executable='vla_system',
           name='vla_system',
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       # ROS - Gazebo Bridge
       bridge = Node(
           package='ros_gz_bridge',
           executable='parameter_bridge',
           arguments=[
               '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
               '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
               '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
           ],
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       return LaunchDescription([
           DeclareLaunchArgument('use_sim_time', default_value='true'),
           gazebo,
           spawn_robot,
           bridge,
           vla_system
       ])
   ```

2. Test the integrated system:
   ```bash
   # Terminal 1: Launch the system
   ros2 launch vla_systems vla_system.launch.py

   # Terminal 2: Send commands
   ros2 topic pub /vla_command std_msgs/msg/String "data: 'find the red object'"
   ```

### Expected Results
The simulated robot should respond to language commands by processing camera images and executing appropriate actions.

## Exercise 5: Performance Evaluation

### Task
Evaluate the performance of your VLA system and identify areas for improvement.

### Steps

1. Create a simple evaluation script `vla_systems/vla_systems/evaluate_vla.py`:
   ```python
   #!/usr/bin/env python3

   """
   Simple evaluation script for VLA system
   """

   import time
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String, Float64
   from geometry_msgs.msg import Twist


   class VLAEvaluator(Node):
       """
       Evaluate VLA system performance
       """

       def __init__(self):
           super().__init__('vla_evaluator')

           # Publishers for commands
           self.command_pub = self.create_publisher(String, '/vla_command', 10)

           # Subscribers for metrics
           self.response_time_sub = self.create_subscription(
               Float64,
               '/vla_response_time',
               self.response_time_callback,
               10
           )

           self.success_sub = self.create_subscription(
               String,
               '/vla_task_success',
               self.success_callback,
               10
           )

           # Initialize metrics
           self.response_times = []
           self.success_count = 0
           self.total_tasks = 0

           self.get_logger().info('VLA Evaluator initialized')

       def run_evaluation(self):
           """
           Run a series of test commands to evaluate the VLA system
           """
           test_commands = [
               "move forward",
               "turn left",
               "find red object",
               "stop"
           ]

           for cmd in test_commands:
               self.get_logger().info(f'Sending command: {cmd}')
               msg = String()
               msg.data = cmd
               self.command_pub.publish(msg)

               # Wait for response
               time.sleep(3)  # Wait for command execution

           # Print evaluation results
           self.print_results()

       def response_time_callback(self, msg):
           """
           Record response time
           """
           self.response_times.append(msg.data)

       def success_callback(self, msg):
           """
           Record task success
           """
           if msg.data == "success":
               self.success_count += 1
           self.total_tasks += 1

       def print_results(self):
           """
           Print evaluation results
           """
           if self.response_times:
               avg_response_time = sum(self.response_times) / len(self.response_times)
               self.get_logger().info(f'Average response time: {avg_response_time:.2f}s')

           if self.total_tasks > 0:
               success_rate = (self.success_count / self.total_tasks) * 100
               self.get_logger().info(f'Success rate: {success_rate:.2f}% ({self.success_count}/{self.total_tasks})')


   def main(args=None):
       rclpy.init(args=args)

       evaluator = VLAEvaluator()

       # Run evaluation after a short delay
       timer = evaluator.create_timer(1.0, lambda: evaluator.run_evaluation())

       try:
           rclpy.spin(evaluator)
       except KeyboardInterrupt:
           pass
       finally:
           evaluator.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. Add the evaluator to setup.py:
   ```python
   entry_points={
       'console_scripts': [
           'vla_system = vla_systems.vla_system:main',
           'vla_evaluator = vla_systems.evaluate_vla:main',
       ],
   },
   ```

### Expected Results
You should be able to evaluate the performance of your VLA system in terms of response time and task success rate.

## Troubleshooting Tips

- If the system doesn't respond to commands, check that all nodes are properly connected
- If vision processing is slow, consider optimizing image processing algorithms
- If language understanding is poor, expand the command patterns in the language processor
- Ensure proper synchronization between camera images and commands
- Check that the robot's coordinate system matches the expected reference frame
- Verify that the camera is properly calibrated and positioned

## Summary

These exercises provided hands-on experience with implementing Vision-Language-Action systems, from basic framework setup to integration with simulated robotics platforms. Understanding these concepts is essential for developing advanced AI-integrated robotic systems that can understand and respond to natural language commands in real-world environments.