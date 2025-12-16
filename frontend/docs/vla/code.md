---
title: VLA Systems Code Examples
sidebar_position: 9.3
description: Code examples for Vision-Language-Action system implementation
---

# VLA Systems Code Examples

## Basic VLA System Architecture

Here's a complete example of a basic Vision-Language-Action system:

```python
#!/usr/bin/env python3

"""
Vision-Language-Action (VLA) System Example

This example demonstrates a complete VLA system that can receive natural language commands,
process visual input, and execute appropriate robotic actions.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import re


class VisionProcessor:
    """
    Basic vision processing for VLA systems
    """

    def __init__(self):
        # For this example, we'll use simple color detection
        pass

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


class VLASystem(Node):
    """
    Complete Vision-Language-Action system
    """

    def __init__(self):
        super().__init__('vla_system')

        # Initialize components
        self.bridge = CvBridge()
        self.vision_processor = VisionProcessor()
        self.language_processor = LanguageProcessor()

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
                self.find_object_by_color(image, object_name)
        else:
            self.get_logger().info(f'Unknown command: {command}')

    def find_object_by_color(self, image, object_name):
        """
        Simple color-based object finding
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

## Advanced VLA with Deep Learning Components

Here's an example that incorporates deep learning models for better vision and language understanding:

```python
#!/usr/bin/env python3

"""
Advanced VLA System with Deep Learning Components
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import cv2
import numpy as np


class DeepVisionProcessor(nn.Module):
    """
    Deep learning-based vision processor
    """

    def __init__(self):
        super().__init__()
        # Using a simple CNN for this example
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Linear(32 * 4 * 4, 10)  # 10 object classes for example

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeepLanguageProcessor:
    """
    Deep learning-based language processor using transformers
    """

    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_text(self, text):
        """
        Encode text using transformer model
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling


class AdvancedVLASystem(Node):
    """
    Advanced VLA system with deep learning components
    """

    def __init__(self):
        super().__init__('advanced_vla_system')

        # Initialize components
        self.bridge = CvBridge()
        self.vision_processor = DeepVisionProcessor()
        self.language_processor = DeepLanguageProcessor()

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
        self.command_embedding = None

        self.get_logger().info('Advanced VLA System initialized')

    def image_callback(self, msg):
        """
        Process incoming camera images using deep learning
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(cv_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                vision_features = self.vision_processor(image_tensor)

            self.current_image = cv_image
            self.current_vision_features = vision_features
            self.process_command_if_ready()
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """
        Process incoming language commands using deep learning
        """
        try:
            self.command_embedding = self.language_processor.encode_text(msg.data)
            self.pending_command = msg.data
            self.process_command_if_ready()
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    def process_command_if_ready(self):
        """
        Process command when both image and command embeddings are available
        """
        if self.current_vision_features is not None and self.command_embedding is not None:
            self.get_logger().info(f'Processing command: {self.pending_command}')
            self.execute_command_with_matching()
            self.command_embedding = None

    def execute_command_with_matching(self):
        """
        Execute command based on vision-language matching
        """
        # Simple example: if we have an object detection model, we could:
        # 1. Match the command embedding to object classes
        # 2. Find the corresponding object in the image
        # 3. Generate appropriate action

        # For this example, we'll use a simple rule-based approach
        # combined with the deep learning features
        command_text = self.pending_command.lower()

        if any(word in command_text for word in ['forward', 'ahead']):
            msg = Twist()
            msg.linear.x = 0.5
            self.cmd_pub.publish(msg)
            self.get_logger().info('Moving forward based on deep analysis')
        elif any(word in command_text for word in ['find', 'look', 'locate']):
            # In a real system, we would use the vision and language embeddings
            # to identify the target object in the scene
            self.get_logger().info('Looking for object based on deep analysis')
            self.search_for_object()
        else:
            self.get_logger().info('Command not recognized by deep analysis')

    def search_for_object(self):
        """
        Search for objects in the current image
        """
        # This is where we would use the deep vision model
        # to identify objects and match them with the command
        self.get_logger().info('Searching for objects using deep vision processing')


def main(args=None):
    rclpy.init(args=args)

    vla_system = AdvancedVLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## VLA System with Memory and Context

Here's an example that includes memory and context awareness:

```python
#!/usr/bin/env python3

"""
VLA System with Memory and Context Awareness
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import json


class MemoryBuffer:
    """
    Memory buffer to store context and previous interactions
    """

    def __init__(self, max_size=10):
        self.max_size = max_size
        self.interactions = deque(maxlen=max_size)
        self.objects_seen = {}
        self.locations = {}

    def add_interaction(self, command, result):
        """
        Add an interaction to memory
        """
        interaction = {
            'command': command,
            'result': result,
            'timestamp': rclpy.clock.Clock().now().nanoseconds
        }
        self.interactions.append(interaction)

    def get_recent_interactions(self, n=5):
        """
        Get recent interactions
        """
        return list(self.interactions)[-n:]

    def remember_object_location(self, obj_name, location):
        """
        Remember where an object was seen
        """
        self.objects_seen[obj_name] = {
            'location': location,
            'timestamp': rclpy.clock.Clock().now().nanoseconds
        }

    def get_object_location(self, obj_name):
        """
        Get remembered location of an object
        """
        return self.objects_seen.get(obj_name, None)


class ContextualVLASystem(Node):
    """
    VLA system with memory and context awareness
    """

    def __init__(self):
        super().__init__('contextual_vla_system')

        # Initialize components
        self.bridge = CvBridge()
        self.memory = MemoryBuffer()

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

        self.get_logger().info('Contextual VLA System initialized')

    def image_callback(self, msg):
        """
        Process incoming camera images and update memory
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # In a real system, we would detect objects and update memory
            # For this example, we'll just store the image timestamp
            self.last_image_time = rclpy.clock.Clock().now().nanoseconds

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """
        Process incoming language commands with context
        """
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Check if this is a follow-up command
        if self.is_follow_up_command(command):
            self.handle_follow_up_command(command)
        else:
            self.handle_new_command(command)

    def is_follow_up_command(self, command):
        """
        Check if command refers to previous context
        """
        follow_up_indicators = ['it', 'that', 'there', 'where', 'back', 'again']
        command_lower = command.lower()
        return any(indicator in command_lower for indicator in follow_up_indicators)

    def handle_follow_up_command(self, command):
        """
        Handle commands that refer to previous context
        """
        # Get recent interactions to understand context
        recent_interactions = self.memory.get_recent_interactions()

        if recent_interactions:
            last_interaction = recent_interactions[-1]
            self.get_logger().info(f'Following up on previous command: {last_interaction["command"]}')

            # Execute based on context
            if 'back' in command.lower():
                # Go back to previous location
                self.execute_go_back()
            elif 'it' in command.lower() or 'that' in command.lower():
                # Do something with the last object/action
                self.execute_with_previous_context(command)

        self.memory.add_interaction(command, "follow-up executed")

    def handle_new_command(self, command):
        """
        Handle new commands
        """
        # Parse and execute the new command
        if 'find' in command.lower() or 'look' in command.lower():
            obj_name = self.extract_object_name(command)
            if obj_name:
                remembered_location = self.memory.get_object_location(obj_name)
                if remembered_location:
                    self.get_logger().info(f'Remembered {obj_name} was at {remembered_location["location"]}')
                    # Go to remembered location
                    self.go_to_location(remembered_location['location'])
                else:
                    self.get_logger().info(f'Need to search for {obj_name}')
                    self.search_for_object(obj_name)

        self.memory.add_interaction(command, "new command executed")

    def extract_object_name(self, command):
        """
        Extract object name from command
        """
        import re
        patterns = [
            r'find (.+)',
            r'look for (.+)',
            r'locate (.+)',
            r'where is (.+)',
            r'get (.+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, command.lower())
            if match:
                return match.group(1).strip()

        return None

    def search_for_object(self, obj_name):
        """
        Search for a specific object
        """
        self.get_logger().info(f'Searching for {obj_name}')
        # In a real system, this would involve active vision and search patterns
        # For now, we'll just move forward
        msg = Twist()
        msg.linear.x = 0.3
        self.cmd_pub.publish(msg)

    def go_to_location(self, location):
        """
        Go to a remembered location
        """
        self.get_logger().info(f'Going to location: {location}')
        # In a real system, this would involve navigation
        # For now, we'll just move forward
        msg = Twist()
        msg.linear.x = 0.3
        self.cmd_pub.publish(msg)

    def execute_go_back(self):
        """
        Execute go back command
        """
        self.get_logger().info('Going back')
        msg = Twist()
        msg.linear.x = -0.3
        self.cmd_pub.publish(msg)

    def execute_with_previous_context(self, command):
        """
        Execute command using previous context
        """
        self.get_logger().info(f'Executing with previous context: {command}')
        # In a real system, this would use the context from previous interactions
        msg = Twist()
        msg.angular.z = 0.5  # Turn as an example
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    vla_system = ContextualVLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## VLA System Launch File

Here's a complete launch file for the VLA system:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
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

    # Spawn robot
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

    # ROS - Gazebo Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/robot_description@std_msgs/msg/String@gz.msgs.StringMsg'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # VLA System Node
    vla_system = Node(
        package='vla_systems',
        executable='vla_system',
        name='vla_system',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Command publisher for testing
    command_publisher = Node(
        package='rostopic',
        executable='rostopic',
        arguments=['pub', '/vla_command', 'std_msgs/msg/String', "{'data': 'move forward'}"],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='empty.sdf'),
        gazebo,
        spawn_robot,
        bridge,
        vla_system,
        # Note: The command_publisher is just for testing,
        # you'd typically send commands from another node or terminal
    ])
```

## Docker Configuration for VLA Systems

Here's a Dockerfile for deploying VLA systems:

```dockerfile
# Use ROS 2 Humble with GPU support
FROM osrf/ros:humble-desktop

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for VLA
RUN pip3 install torch torchvision torchaudio \
    transformers \
    opencv-python \
    numpy \
    scipy

# Set up ROS workspace
RUN mkdir -p /ws/src
WORKDIR /ws

# Copy VLA package
COPY vla_systems /ws/src/vla_systems

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select vla_systems

# Source the workspace
RUN echo "source /ws/install/setup.sh" >> ~/.bashrc
RUN echo "source /opt/ros/humble/setup.sh" >> ~/.bashrc

# Set up entrypoint
CMD ["bash", "-c", "source /opt/ros/humble/setup.sh && source /ws/install/setup.sh && exec \"$@\"", "--"]
```

## Configuration Files

Here's an example configuration file for the VLA system:

```yaml
# config/vla_config.yaml
vla_system:
  ros__parameters:
    # Vision processing parameters
    vision:
      image_topic: "/camera/image_raw"
      image_width: 640
      image_height: 480
      detection_threshold: 0.5

    # Language processing parameters
    language:
      command_topic: "/vla_command"
      response_timeout: 5.0
      max_command_history: 10

    # Action execution parameters
    action:
      cmd_vel_topic: "/cmd_vel"
      max_linear_velocity: 0.5
      max_angular_velocity: 1.0
      safety_timeout: 2.0

    # Memory parameters
    memory:
      max_interactions: 50
      object_memory_ttl: 300  # 5 minutes
      location_memory_ttl: 600  # 10 minutes
```

## Summary

These code examples demonstrate various approaches to implementing Vision-Language-Action systems:

1. **Basic VLA System**: A simple system that integrates vision, language, and action components
2. **Advanced VLA with Deep Learning**: Incorporates neural networks for better perception and understanding
3. **Contextual VLA System**: Includes memory and context awareness for more sophisticated interactions
4. **Launch Files**: Complete configuration for running the system with ROS 2
5. **Deployment Configuration**: Docker setup for easy deployment
6. **Parameter Configuration**: YAML files for system configuration

Each example builds upon the previous one, showing how to progressively add more sophisticated capabilities to VLA systems. These examples can be used as starting points for your own VLA system implementations.