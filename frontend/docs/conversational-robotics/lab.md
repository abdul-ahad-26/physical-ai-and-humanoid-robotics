---
title: Conversational Robotics Exercises
sidebar_position: 9.5
description: Hands-on exercises for implementing conversational robotics systems
---

# Conversational Robotics Exercises

## Learning Objectives

- Implement speech recognition and text-to-speech systems
- Create dialog management systems for robot conversations
- Integrate conversational systems with robot actions
- Evaluate conversational system performance
- Troubleshoot common conversational robotics issues
- Deploy conversational systems in simulated environments

## Prerequisites

- ROS 2 environment with Gazebo simulation
- Microphone and audio input capabilities
- Python knowledge for speech processing
- Understanding of ROS 2 messaging
- Basic knowledge of natural language processing

## Exercise 1: Basic Speech Recognition Setup

### Task
Set up a basic speech recognition system that can convert spoken commands to text.

### Steps

1. Install required dependencies:
   ```bash
   pip3 install speechrecognition pyaudio
   # For offline recognition, also install:
   pip3 install vosk
   ```

2. Create a new ROS 2 package for speech processing:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python speech_recognition_pkg --dependencies rclpy std_msgs
   ```

3. Create the speech recognition node `speech_recognition_pkg/speech_recognition_pkg/asr_node.py`:
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   import speech_recognition as sr
   from std_msgs.msg import String


   class ASRNode(Node):
       """
       Automatic Speech Recognition Node
       """

       def __init__(self):
           super().__init__('asr_node')

           # Initialize speech recognizer
           self.recognizer = sr.Recognizer()
           self.microphone = sr.Microphone()

           # Publisher for recognized text
           self.text_pub = self.create_publisher(String, '/recognized_text', 10)

           # Timer to continuously listen
           self.timer = self.create_timer(2.0, self.listen_callback)

           # Adjust for ambient noise
           with self.microphone as source:
               self.recognizer.adjust_for_ambient_noise(source)

           self.get_logger().info('ASR Node initialized')

       def listen_callback(self):
           """
           Listen for speech and convert to text
           """
           try:
               with self.microphone as source:
                   self.get_logger().info('Listening...')
                   audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)

               # Recognize speech using Google Web Speech API
               text = self.recognizer.recognize_google(audio)

               # Publish recognized text
               msg = String()
               msg.data = text
               self.text_pub.publish(msg)

               self.get_logger().info(f'Recognized: {text}')

           except sr.WaitTimeoutError:
               self.get_logger().info('No speech detected')
           except sr.UnknownValueError:
               self.get_logger().info('Could not understand audio')
           except sr.RequestError as e:
               self.get_logger().error(f'Could not request results from speech service; {e}')


   def main(args=None):
       rclpy.init(args=args)
       asr_node = ASRNode()

       try:
           rclpy.spin(asr_node)
       except KeyboardInterrupt:
           pass
       finally:
           asr_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. Update the `setup.py` file to include the executable:
   ```python
   entry_points={
       'console_scripts': [
           'asr_node = speech_recognition_pkg.asr_node:main',
       ],
   },
   ```

5. Build and run the node:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select speech_recognition_pkg
   source install/setup.bash
   ros2 run speech_recognition_pkg asr_node
   ```

6. In another terminal, listen to the recognized text:
   ```bash
   ros2 topic echo /recognized_text
   ```

### Expected Results
You should be able to speak into your microphone and see the recognized text published to the `/recognized_text` topic.

## Exercise 2: Text-to-Speech Implementation

### Task
Implement a text-to-speech system that can speak robot responses.

### Steps

1. Install required dependencies:
   ```bash
   pip3 install pyttsx3
   ```

2. Create the TTS node `speech_recognition_pkg/speech_recognition_pkg/tts_node.py`:
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   import pyttsx3
   from std_msgs.msg import String


   class TTSNode(Node):
       """
       Text-to-Speech Node
       """

       def __init__(self):
           super().__init__('tts_node')

           # Initialize TTS engine
           self.tts_engine = pyttsx3.init()

           # Set properties
           self.tts_engine.setProperty('rate', 150)  # Speed of speech
           self.tts_engine.setProperty('volume', 0.9)  # Volume level

           # Subscribe to robot responses
           self.response_sub = self.create_subscription(
               String, '/robot_response', self.response_callback, 10
           )

           self.get_logger().info('TTS Node initialized')

       def response_callback(self, msg):
           """
           Speak the received response text
           """
           text = msg.data
           self.get_logger().info(f'Speaking: {text}')

           # Speak the text
           self.tts_engine.say(text)
           self.tts_engine.runAndWait()


   def main(args=None):
       rclpy.init(args=args)
       tts_node = TTSNode()

       try:
           rclpy.spin(tts_node)
       except KeyboardInterrupt:
           pass
       finally:
           tts_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. Add the TTS node to setup.py:
   ```python
   entry_points={
       'console_scripts': [
           'asr_node = speech_recognition_pkg.asr_node:main',
           'tts_node = speech_recognition_pkg.tts_node:main',
       ],
   },
   ```

4. Build and run the TTS node:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select speech_recognition_pkg
   source install/setup.bash
   ros2 run speech_recognition_pkg tts_node
   ```

5. Test the TTS system by publishing messages:
   ```bash
   ros2 topic pub /robot_response std_msgs/msg/String "data: 'Hello, I am a robot'"
   ```

### Expected Results
The robot should speak the text you publish to the `/robot_response` topic.

## Exercise 3: Dialog Manager Implementation

### Task
Create a dialog manager that processes recognized text and generates appropriate responses.

### Steps

1. Create the dialog manager `speech_recognition_pkg/speech_recognition_pkg/dialog_manager.py`:
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   import re


   class IntentClassifier:
       """
       Simple intent classifier for conversational robotics
       """

       def __init__(self):
           self.intents = {
               'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
               'navigation': ['go forward', 'move forward', 'go back', 'move back', 'turn left', 'turn right', 'stop'],
               'object_interaction': ['pick up', 'grasp', 'take', 'get', 'find', 'look for'],
               'information_request': ['what is', 'tell me', 'describe', 'where is', 'how many'],
               'farewell': ['goodbye', 'bye', 'see you', 'good night']
           }

       def classify_intent(self, text):
           """
           Classify the intent of the given text
           """
           text_lower = text.lower()

           for intent, keywords in self.intents.items():
               for keyword in keywords:
                   if keyword in text_lower:
                       return intent

           return 'unknown'


   class DialogManager(Node):
       """
       Manages the conversation flow
       """

       def __init__(self):
           super().__init__('dialog_manager')

           # Subscribe to recognized text
           self.text_sub = self.create_subscription(
               String, '/recognized_text', self.text_callback, 10
           )

           # Publisher for robot responses
           self.response_pub = self.create_publisher(String, '/robot_response', 10)

           # Publisher for robot actions
           self.action_pub = self.create_publisher(String, '/robot_action', 10)

           # Initialize components
           self.intent_classifier = IntentClassifier()

           self.get_logger().info('Dialog Manager initialized')

       def text_callback(self, msg):
           """
           Process incoming recognized text
           """
           text = msg.data
           self.get_logger().info(f'Received text: {text}')

           # Classify intent
           intent = self.intent_classifier.classify_intent(text)

           # Generate response based on intent
           response = self.generate_response(intent, text)

           # Publish response
           if response:
               response_msg = String()
               response_msg.data = response
               self.response_pub.publish(response_msg)

           # Execute actions if needed
           self.execute_actions(intent, text)

       def generate_response(self, intent, text):
           """
           Generate a response based on the intent
           """
           if intent == 'greeting':
               return "Hello! How can I help you today?"
           elif intent == 'navigation':
               if 'forward' in text:
                   return "Moving forward."
               elif 'back' in text:
                   return "Moving backward."
               elif 'left' in text:
                   return "Turning left."
               elif 'right' in text:
                   return "Turning right."
               else:
                   return "I will move as requested."
           elif intent == 'object_interaction':
               return "I will look for that object."
           elif intent == 'information_request':
               return "I can help you with that information."
           elif intent == 'farewell':
               return "Goodbye! Have a great day!"
           else:
               return "I'm not sure how to respond to that. Can you rephrase?"

       def execute_actions(self, intent, text):
           """
           Execute robot actions based on intent
           """
           if intent == 'navigation':
               action_msg = String()
               if 'forward' in text:
                   action_msg.data = 'move_forward'
               elif 'back' in text:
                   action_msg.data = 'move_backward'
               elif 'left' in text:
                   action_msg.data = 'turn_left'
               elif 'right' in text:
                   action_msg.data = 'turn_right'
               elif 'stop' in text:
                   action_msg.data = 'stop'
               else:
                   return

               self.action_pub.publish(action_msg)
               self.get_logger().info(f'Published action: {action_msg.data}')


   def main(args=None):
       rclpy.init(args=args)
       dialog_manager = DialogManager()

       try:
           rclpy.spin(dialog_manager)
       except KeyboardInterrupt:
           pass
       finally:
           dialog_manager.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. Add the dialog manager to setup.py:
   ```python
   entry_points={
       'console_scripts': [
           'asr_node = speech_recognition_pkg.asr_node:main',
           'tts_node = speech_recognition_pkg.tts_node:main',
           'dialog_manager = speech_recognition_pkg.dialog_manager:main',
       ],
   },
   ```

3. Build and run the dialog manager:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select speech_recognition_pkg
   source install/setup.bash
   ros2 run speech_recognition_pkg dialog_manager
   ```

### Expected Results
The dialog manager should process text from the ASR node and generate appropriate responses for the TTS system.

## Exercise 4: Integration with Robot Actions

### Task
Connect the conversational system to actual robot actions in simulation.

### Steps

1. Create a robot action handler `speech_recognition_pkg/speech_recognition_pkg/action_handler.py`:
   ```python
   #!/usr/bin/env python3

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist


   class ActionHandler(Node):
       """
       Handles robot actions based on conversational commands
       """

       def __init__(self):
           super().__init__('action_handler')

           # Subscribe to robot actions
           self.action_sub = self.create_subscription(
               String, '/robot_action', self.action_callback, 10
           )

           # Publisher for robot movement
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

           self.get_logger().info('Action Handler initialized')

       def action_callback(self, msg):
           """
           Execute robot action based on command
           """
           action = msg.data
           self.get_logger().info(f'Executing action: {action}')

           twist_msg = Twist()

           if action == 'move_forward':
               twist_msg.linear.x = 0.5
           elif action == 'move_backward':
               twist_msg.linear.x = -0.5
           elif action == 'turn_left':
               twist_msg.angular.z = 0.5
           elif action == 'turn_right':
               twist_msg.angular.z = -0.5
           elif action == 'stop':
               # Already zero by default
               pass
           else:
               self.get_logger().info(f'Unknown action: {action}')
               return

           self.cmd_vel_pub.publish(twist_msg)


   def main(args=None):
       rclpy.init(args=args)
       action_handler = ActionHandler()

       try:
           rclpy.spin(action_handler)
       except KeyboardInterrupt:
           pass
       finally:
           # Stop the robot
           stop_msg = Twist()
           action_handler.cmd_vel_pub.publish(stop_msg)
           action_handler.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. Add the action handler to setup.py:
   ```python
   entry_points={
       'console_scripts': [
           'asr_node = speech_recognition_pkg.asr_node:main',
           'tts_node = speech_recognition_pkg.tts_node:main',
           'dialog_manager = speech_recognition_pkg.dialog_manager:main',
           'action_handler = speech_recognition_pkg.action_handler:main',
       ],
   },
   ```

3. Create a launch file `speech_recognition_pkg/launch/conversational_robot.launch.py`:
   ```python
   from launch import LaunchDescription
   from launch.actions import IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare


   def generate_launch_description():
       return LaunchDescription([
           # Launch Gazebo simulation
           IncludeLaunchDescription(
               PythonLaunchDescriptionSource([
                   PathJoinSubstitution([
                       FindPackageShare('ros_gz_sim'),
                       'launch',
                       'gz_sim.launch.py'
                   ])
               ]),
               launch_arguments={'gz_args': '-r empty.sdf'}.items()
           ),

           # ASR Node
           Node(
               package='speech_recognition_pkg',
               executable='asr_node',
               name='asr_node',
               output='screen'
           ),

           # TTS Node
           Node(
               package='speech_recognition_pkg',
               executable='tts_node',
               name='tts_node',
               output='screen'
           ),

           # Dialog Manager
           Node(
               package='speech_recognition_pkg',
               executable='dialog_manager',
               name='dialog_manager',
               output='screen'
           ),

           # Action Handler
           Node(
               package='speech_recognition_pkg',
               executable='action_handler',
               name='action_handler',
               output='screen'
           ),

           # Gazebo bridge
           Node(
               package='ros_gz_bridge',
               executable='parameter_bridge',
               arguments=[
                   '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                   '/robot_description@std_msgs/msg/String@gz.msgs.StringMsg'
               ],
               output='screen'
           )
       ])
   ```

4. Build and run the complete system:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select speech_recognition_pkg
   source install/setup.bash
   ros2 launch speech_recognition_pkg conversational_robot.launch.py
   ```

### Expected Results
You should have a complete conversational robotics system where speaking commands triggers robot actions in the simulation.

## Exercise 5: Performance Evaluation

### Task
Evaluate the performance of your conversational robotics system.

### Steps

1. Create an evaluation script `speech_recognition_pkg/speech_recognition_pkg/evaluate_conversation.py`:
   ```python
   #!/usr/bin/env python3

   """
   Evaluation script for conversational robotics system
   """

   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String, Float64
   import time


   class ConversationEvaluator(Node):
       """
       Evaluate conversational system performance
       """

       def __init__(self):
           super().__init__('conversation_evaluator')

           # Publishers for test commands
           self.command_pub = self.create_publisher(String, '/user_command_simulation', 10)

           # Track performance metrics
           self.start_time = None
           self.response_times = []
           self.correct_responses = 0
           self.total_responses = 0

           # Subscribe to system responses
           self.response_sub = self.create_subscription(
               String, '/robot_response', self.response_callback, 10
           )

           self.get_logger().info('Conversation Evaluator initialized')

       def run_evaluation(self):
           """
           Run a series of test commands to evaluate the system
           """
           test_commands = [
               ("Hello", "greeting"),
               ("Go forward", "navigation"),
               ("Turn left", "navigation"),
               ("Goodbye", "farewell")
           ]

           for command, expected_type in test_commands:
               self.get_logger().info(f'Testing: {command}')
               self.start_time = time.time()

               # Simulate command input
               cmd_msg = String()
               cmd_msg.data = command
               self.command_pub.publish(cmd_msg)

               # Wait for response
               time.sleep(5)  # Wait for response

           self.print_results()

       def response_callback(self, msg):
           """
           Process system response
           """
           if self.start_time:
               response_time = time.time() - self.start_time
               self.response_times.append(response_time)
               self.start_time = None

               self.total_responses += 1
               # In a real evaluation, you'd check if the response was correct
               self.correct_responses += 1

       def print_results(self):
           """
           Print evaluation results
           """
           if self.response_times:
               avg_response_time = sum(self.response_times) / len(self.response_times)
               self.get_logger().info(f'Average response time: {avg_response_time:.2f}s')

           if self.total_responses > 0:
               accuracy = (self.correct_responses / self.total_responses) * 100
               self.get_logger().info(f'Response accuracy: {accuracy:.2f}% ({self.correct_responses}/{self.total_responses})')

           self.get_logger().info(f'Total responses processed: {self.total_responses}')


   def main(args=None):
       rclpy.init(args=args)
       evaluator = ConversationEvaluator()

       # Run evaluation after a short delay
       timer = evaluator.create_timer(2.0, lambda: evaluator.run_evaluation())

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
           'asr_node = speech_recognition_pkg.asr_node:main',
           'tts_node = speech_recognition_pkg.tts_node:main',
           'dialog_manager = speech_recognition_pkg.dialog_manager:main',
           'action_handler = speech_recognition_pkg.action_handler:main',
           'conversation_evaluator = speech_recognition_pkg.evaluate_conversation:main',
       ],
   },
   ```

3. Build and run the evaluation:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select speech_recognition_pkg
   source install/setup.bash
   ros2 run speech_recognition_pkg conversation_evaluator
   ```

### Expected Results
You should see performance metrics for your conversational robotics system.

## Troubleshooting Tips

- If speech recognition doesn't work, check microphone permissions and audio input
- If the robot doesn't respond to speech, verify all nodes are connected and topics are properly matched
- If responses are slow, consider using smaller models or optimizing processing
- If the robot misunderstands commands, improve the intent classification patterns
- For audio feedback issues, check that the TTS engine is properly initialized
- If actions don't execute, verify that the action topics are correctly connected

## Summary

These exercises provided hands-on experience with implementing conversational robotics systems, from basic speech recognition to full integration with robot actions in simulation. Understanding these concepts is essential for creating robots that can interact naturally with humans through spoken language.