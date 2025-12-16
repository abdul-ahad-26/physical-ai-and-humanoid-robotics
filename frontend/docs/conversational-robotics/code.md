---
title: Conversational Robotics Code Examples
sidebar_position: 9.6
description: Code examples for implementing conversational robotics systems
---

# Conversational Robotics Code Examples

## Complete Conversational Robotics System

Here's a complete example of a conversational robotics system that integrates speech recognition, natural language processing, and robot action execution:

```python
#!/usr/bin/env python3

"""
Complete Conversational Robotics System

This example demonstrates a full conversational robotics system that can:
- Receive voice commands
- Process them with natural language understanding
- Execute appropriate robot actions
- Provide voice responses
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import speech_recognition as sr
import pyttsx3
import re
import threading
import time
import json


class SpeechRecognitionNode(Node):
    """
    Complete speech recognition node with multiple ASR options
    """

    def __init__(self):
        super().__init__('speech_recognition_node')

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

        # Confidence threshold for speech recognition
        self.confidence_threshold = 0.7

        self.get_logger().info('Speech Recognition Node initialized')

    def listen_callback(self):
        """
        Listen for speech and convert to text
        """
        try:
            with self.microphone as source:
                self.get_logger().info('Listening...')
                audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)

            # Try Google Web Speech API
            try:
                text = self.recognizer.recognize_google(audio)
                confidence = 1.0  # Google doesn't provide confidence scores
            except sr.UnknownValueError:
                self.get_logger().info('Google could not understand audio')
                return
            except sr.RequestError as e:
                self.get_logger().error(f'Could not request results from Google service; {e}')
                return

            # Publish recognized text
            if text and confidence >= self.confidence_threshold:
                msg = String()
                msg.data = text
                self.text_pub.publish(msg)
                self.get_logger().info(f'Recognized: {text} (confidence: {confidence:.2f})')
            else:
                self.get_logger().info(f'Discarded low-confidence recognition: {text} (confidence: {confidence:.2f})')

        except sr.WaitTimeoutError:
            self.get_logger().info('No speech detected')
        except Exception as e:
            self.get_logger().error(f'Error in speech recognition: {e}')


class IntentClassifier:
    """
    Advanced intent classifier for conversational robotics
    """

    def __init__(self):
        self.intents = {
            'navigation': {
                'patterns': [
                    r'go forward|move forward|go ahead|move ahead',
                    r'go back|move back|go backward|move backward',
                    r'turn left|rotate left|spin left',
                    r'turn right|rotate right|spin right',
                    r'stop|halt|freeze|pause',
                    r'go to (\w+)|move to (\w+)|navigate to (\w+)'
                ],
                'responses': [
                    'Moving forward.',
                    'Moving backward.',
                    'Turning left.',
                    'Turning right.',
                    'Stopping.',
                    'Navigating to {}.'
                ]
            },
            'object_interaction': {
                'patterns': [
                    r'pick up (\w+)|get (\w+)|take (\w+)|grasp (\w+)',
                    r'find (\w+)|look for (\w+)|locate (\w+)|search for (\w+)',
                    r'bring me (\w+)|hand me (\w+)',
                    r'put (\w+) on (\w+)|place (\w+) on (\w+)'
                ],
                'responses': [
                    'Picking up {}.',
                    'Looking for {}.',
                    'Getting {} for you.',
                    'Placing {} on {}.'
                ]
            },
            'information_request': {
                'patterns': [
                    r'what is (\w+)|what\'s (\w+)',
                    r'tell me about (\w+)|describe (\w+)',
                    r'how many (\w+)|count (\w+)',
                    r'where is (\w+)|location of (\w+)',
                    r'what color is (\w+)|color of (\w+)'
                ],
                'responses': [
                    'I can tell you about {}.',
                    'Let me describe {} for you.',
                    'There are {} {}.',
                    '{} is located at {}.',
                    '{} is {} in color.'
                ]
            },
            'social_interaction': {
                'patterns': [
                    r'hello|hi|hey|good morning|good afternoon|good evening',
                    r'goodbye|bye|see you|good night|farewell',
                    r'thank you|thanks|thank you so much',
                    r'please|excuse me|sorry'
                ],
                'responses': [
                    'Hello! How can I assist you?',
                    'Goodbye! Have a great day!',
                    'You\'re welcome!',
                    'Of course! How can I help?'
                ]
            }
        }

    def classify_intent(self, text):
        """
        Classify the intent of the given text with extracted entities
        """
        text_lower = text.lower()

        for intent_name, intent_data in self.intents.items():
            for i, pattern in enumerate(intent_data['patterns']):
                match = re.search(pattern, text_lower)
                if match:
                    # Extract entities from the match
                    entities = match.groups()
                    return intent_name, entities, intent_data['responses'][i]

        return 'unknown', [], 'I\'m not sure how to help with that. Can you rephrase?'


class DialogManager(Node):
    """
    Advanced dialog manager with state tracking and context awareness
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

        # Publisher for system state
        self.state_pub = self.create_publisher(String, '/system_state', 10)

        # Initialize components
        self.intent_classifier = IntentClassifier()
        self.conversation_history = []
        self.context = {}
        self.pending_action = None

        # Initialize state
        self.system_state = {
            'current_task': None,
            'last_intent': None,
            'conversation_turn': 0,
            'objects_seen': {},
            'locations_known': {}
        }

        self.get_logger().info('Dialog Manager initialized')

    def text_callback(self, msg):
        """
        Process incoming recognized text
        """
        text = msg.data
        self.get_logger().info(f'Received text: {text}')

        # Add to conversation history
        self.conversation_history.append({'speaker': 'user', 'text': text})
        if len(self.conversation_history) > 10:  # Keep only last 10 exchanges
            self.conversation_history.pop(0)

        # Classify intent
        intent, entities, response_template = self.intent_classifier.classify_intent(text)

        # Update system state
        self.system_state['last_intent'] = intent
        self.system_state['conversation_turn'] += 1

        # Process based on intent
        response = self.process_intent(intent, text, entities, response_template)

        # Publish response
        if response:
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

            # Add to conversation history
            self.conversation_history.append({'speaker': 'robot', 'text': response})

        # Update and publish system state
        self.update_system_state(intent, entities)
        state_msg = String()
        state_msg.data = json.dumps(self.system_state)
        self.state_pub.publish(state_msg)

    def process_intent(self, intent, text, entities, response_template):
        """
        Process the recognized intent and generate response
        """
        if intent == 'navigation':
            return self.handle_navigation_intent(text, entities, response_template)
        elif intent == 'object_interaction':
            return self.handle_object_intent(text, entities, response_template)
        elif intent == 'information_request':
            return self.handle_info_intent(text, entities, response_template)
        elif intent == 'social_interaction':
            return self.handle_social_intent(text, response_template)
        else:
            return self.handle_unknown_intent(text)

    def handle_navigation_intent(self, text, entities, response_template):
        """
        Handle navigation-related intents
        """
        # Determine navigation command
        if 'forward' in text:
            action_msg = String()
            action_msg.data = 'move_forward'
            self.action_pub.publish(action_msg)
            self.system_state['current_task'] = 'navigation'
            return response_template
        elif 'back' in text:
            action_msg = String()
            action_msg.data = 'move_backward'
            self.action_pub.publish(action_msg)
            self.system_state['current_task'] = 'navigation'
            return response_template
        elif 'left' in text:
            action_msg = String()
            action_msg.data = 'turn_left'
            self.action_pub.publish(action_msg)
            self.system_state['current_task'] = 'navigation'
            return response_template
        elif 'right' in text:
            action_msg = String()
            action_msg.data = 'turn_right'
            self.action_pub.publish(action_msg)
            self.system_state['current_task'] = 'navigation'
            return response_template
        elif 'stop' in text:
            action_msg = String()
            action_msg.data = 'stop'
            self.action_pub.publish(action_msg)
            self.system_state['current_task'] = None
            return response_template
        elif entities:
            # Handle navigation to specific location
            location = entities[0] if entities else 'destination'
            action_msg = String()
            action_msg.data = f'navigate_to:{location}'
            self.action_pub.publish(action_msg)
            self.system_state['current_task'] = f'navigating_to_{location}'
            return response_template.format(location)
        else:
            return "Where would you like me to go?"

    def handle_object_intent(self, text, entities, response_template):
        """
        Handle object interaction intents
        """
        if entities:
            obj = entities[0]
            action_msg = String()
            action_msg.data = f'find_object:{obj}'
            self.action_pub.publish(action_msg)
            self.system_state['current_task'] = f'finding_{obj}'

            # Fill in the response template with entities
            if '{}' in response_template:
                return response_template.format(obj)
            else:
                return response_template
        else:
            return "What object are you looking for?"

    def handle_info_intent(self, text, entities, response_template):
        """
        Handle information request intents
        """
        if entities:
            # Fill in the response template with entities
            filled_response = response_template
            for entity in entities:
                if '{}' in filled_response:
                    filled_response = filled_response.replace('{}', entity, 1)
            return filled_response
        else:
            return "I can help you find information. What are you looking for?"

    def handle_social_intent(self, text, response_template):
        """
        Handle social interaction intents
        """
        self.system_state['current_task'] = None
        return response_template

    def handle_unknown_intent(self, text):
        """
        Handle unknown intents
        """
        # Try to extract any useful information
        if 'please' in text.lower():
            return "I heard 'please' in your request. How can I assist you?"

        # Look for context clues
        for prev_exchange in reversed(self.conversation_history[-3:]):  # Look at last 3 exchanges
            if prev_exchange['speaker'] == 'robot' and 'what' in prev_exchange['text'].lower():
                return "I'm still waiting for more information to help you."

        return "I'm not sure how to help with that. Can you rephrase your request?"

    def update_system_state(self, intent, entities):
        """
        Update the system state based on current interaction
        """
        if intent == 'object_interaction' and entities:
            obj_name = entities[0]
            if 'find' in self.conversation_history[-1]['text']:
                self.system_state['objects_seen'][obj_name] = time.time()

        if intent == 'navigation' and entities:
            location = entities[0] if entities else 'unknown'
            self.system_state['locations_known'][location] = time.time()


class TextToSpeechNode(Node):
    """
    Text-to-Speech Node with emotion and prosody control
    """

    def __init__(self):
        super().__init__('tts_node')

        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()

        # Set properties
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level

        # Get available voices
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Set to female voice if available, otherwise first voice
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            else:
                self.tts_engine.setProperty('voice', voices[0].id)

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


class ActionExecutor(Node):
    """
    Executes robot actions based on conversational commands
    """

    def __init__(self):
        super().__init__('action_executor')

        # Subscribe to robot actions
        self.action_sub = self.create_subscription(
            String, '/robot_action', self.action_callback, 10
        )

        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher for robot status
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # Initialize state
        self.current_action = None
        self.action_queue = []

        self.get_logger().info('Action Executor initialized')

    def action_callback(self, msg):
        """
        Execute robot action based on command
        """
        action = msg.data
        self.get_logger().info(f'Queueing action: {action}')

        # Add to action queue
        self.action_queue.append(action)

        # Process queue if not currently executing
        if self.current_action is None:
            self.process_next_action()

    def process_next_action(self):
        """
        Process the next action in the queue
        """
        if self.action_queue:
            action = self.action_queue.pop(0)
            self.current_action = action

            self.get_logger().info(f'Executing action: {action}')

            # Publish status
            status_msg = String()
            status_msg.data = f'executing:{action}'
            self.status_pub.publish(status_msg)

            # Execute the action
            self.execute_action(action)

            # Clear current action when done
            self.current_action = None

            # Process next action if available
            if self.action_queue:
                self.process_next_action()

    def execute_action(self, action):
        """
        Execute a specific action
        """
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
        elif action.startswith('navigate_to:'):
            # In a real system, this would trigger navigation
            location = action.split(':')[1]
            self.get_logger().info(f'Navigating to {location}')
            # For simulation, just move forward
            twist_msg.linear.x = 0.3
        elif action.startswith('find_object:'):
            # In a real system, this would trigger object detection
            obj = action.split(':')[1]
            self.get_logger().info(f'Looking for {obj}')
            # For simulation, just turn slowly
            twist_msg.angular.z = 0.2
        else:
            self.get_logger().info(f'Unknown action: {action}')
            return

        self.cmd_vel_pub.publish(twist_msg)

        # Stop after a moment (for simulation)
        time.sleep(2)
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)


def main(args=None):
    """
    Main function to run the complete conversational robotics system
    """
    rclpy.init(args=args)

    # Create nodes
    asr_node = SpeechRecognitionNode()
    dialog_node = DialogManager()
    tts_node = TextToSpeechNode()
    action_node = ActionExecutor()

    # Use MultiThreadedExecutor to run all nodes concurrently
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(asr_node)
    executor.add_node(dialog_node)
    executor.add_node(tts_node)
    executor.add_node(action_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        stop_msg = Twist()
        action_node.cmd_vel_pub.publish(stop_msg)

        asr_node.destroy_node()
        dialog_node.destroy_node()
        tts_node.destroy_node()
        action_node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Whisper + GPT Integration Example

Here's an example that integrates Whisper for speech recognition and GPT for natural language understanding:

```python
#!/usr/bin/env python3

"""
Advanced Conversational System with Whisper and GPT

This example demonstrates integration with Whisper for speech recognition
and GPT for natural language understanding and response generation.
"""

import openai
import whisper
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import threading
import queue
import time
import tempfile
import wave
import pyaudio


class WhisperGPTNode(Node):
    """
    Advanced conversational node using Whisper and GPT
    """

    def __init__(self):
        super().__init__('whisper_gpt_node')

        # Initialize Whisper model
        self.whisper_model = whisper.load_model("base")

        # Set OpenAI API key (you'll need to set this)
        # openai.api_key = "your-openai-api-key"

        # Initialize PyAudio for audio recording
        self.pyaudio_instance = pyaudio.PyAudio()

        # Queues for processing
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()

        # Publishers
        self.response_pub = self.create_publisher(String, '/robot_response', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)

        # Audio recording parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5

        # Start audio recording thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Whisper-GPT Node initialized')

    def record_audio(self):
        """
        Continuously record audio from microphone
        """
        stream = self.pyaudio_instance.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        while rclpy.ok():
            try:
                frames = []

                # Record for specified duration
                for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
                    data = stream.read(self.chunk)
                    frames.append(data)

                # Save to temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    wf = wave.open(temp_wav.name, 'wb')
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.audio_format))
                    wf.setframerate(self.rate)
                    wf.writeframes(b''.join(frames))
                    wf.close()

                    # Add to processing queue
                    self.audio_queue.put(temp_wav.name)

            except Exception as e:
                self.get_logger().error(f'Error recording audio: {e}')

        stream.stop_stream()
        stream.close()

    def process_loop(self):
        """
        Main processing loop
        """
        while rclpy.ok():
            try:
                # Process audio files
                if not self.audio_queue.empty():
                    audio_file = self.audio_queue.get()

                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(audio_file)
                    text = result['text']

                    if text.strip():  # Only process non-empty transcriptions
                        self.get_logger().info(f'Whisper transcribed: {text}')

                        # Process with GPT
                        response = self.process_with_gpt(text)

                        if response:
                            # Publish response
                            response_msg = String()
                            response_msg.data = response
                            self.response_pub.publish(response_msg)

                            # Extract and execute actions
                            self.extract_and_execute_actions(response)

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {e}')

    def process_with_gpt(self, user_text):
        """
        Process user input with GPT to generate response
        """
        try:
            # Create a conversation with GPT
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful robot assistant. Respond concisely and provide "
                        "clear information. If the user requests an action, include action "
                        "tags like [MOVE_FORWARD] or [TURN_LEFT]. Only use the following "
                        "action tags: [MOVE_FORWARD], [MOVE_BACKWARD], [TURN_LEFT], "
                        "[TURN_RIGHT], [STOP], [FIND_OBJECT], [NAVIGATE_TO]."
                    )
                },
                {"role": "user", "content": user_text}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )

            return response.choices[0].message['content'].strip()

        except Exception as e:
            self.get_logger().error(f'Error calling GPT: {e}')
            return "I'm having trouble processing your request right now."

    def extract_and_execute_actions(self, response):
        """
        Extract action commands from GPT response and execute them
        """
        import re

        # Define action patterns
        action_patterns = {
            'move_forward': r'\[MOVE_FORWARD\]',
            'move_backward': r'\[MOVE_BACKWARD\]',
            'turn_left': r'\[TURN_LEFT\]',
            'turn_right': r'\[TURN_RIGHT\]',
            'stop': r'\[STOP\]',
            'find_object': r'\[FIND_OBJECT\]',
            'navigate_to': r'\[NAVIGATE_TO\]'
        }

        for action, pattern in action_patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                action_msg = String()
                action_msg.data = action
                self.action_pub.publish(action_msg)
                self.get_logger().info(f'Executed action: {action}')


def main(args=None):
    rclpy.init(args=args)
    node = WhisperGPTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## ROS 2 Launch File for Conversational System

Here's a complete launch file that brings up the entire conversational robotics system:

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

    # Launch Gazebo simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': '-r -v 4 empty.sdf'}.items()
    )

    # Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'conversational_robot',
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
            '/robot_description@std_msgs/msg/String@gz.msgs.StringMsg'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Speech Recognition Node
    speech_recognition = Node(
        package='conversational_robot',
        executable='speech_recognition',
        name='speech_recognition_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Dialog Manager Node
    dialog_manager = Node(
        package='conversational_robot',
        executable='dialog_manager',
        name='dialog_manager',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Text-to-Speech Node
    tts_node = Node(
        package='conversational_robot',
        executable='tts_node',
        name='tts_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Action Executor Node
    action_executor = Node(
        package='conversational_robot',
        executable='action_executor',
        name='action_executor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        gazebo,
        spawn_robot,
        bridge,
        speech_recognition,
        dialog_manager,
        tts_node,
        action_executor
    ])
```

## Docker Configuration for Conversational Robotics

Here's a Dockerfile for deploying conversational robotics systems:

```dockerfile
# Use ROS 2 Humble with audio support
FROM osrf/ros:humble-desktop

# Install audio and speech processing dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    pulseaudio \
    alsa-utils \
    portaudio19-dev \
    python3-pyaudio \
    espeak \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for speech processing
RUN pip3 install torch torchvision torchaudio \
    transformers \
    openai \
    openai-whisper \
    speechrecognition \
    pyttsx3 \
    pyaudio \
    numpy \
    scipy

# Set up ROS workspace
RUN mkdir -p /ws/src
WORKDIR /ws

# Copy conversational robotics package
COPY conversational_robot /ws/src/conversational_robot

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select conversational_robot

# Source the workspace
RUN echo "source /ws/install/setup.sh" >> ~/.bashrc
RUN echo "source /opt/ros/humble/setup.sh" >> ~/.bashrc

# Enable audio access
RUN groupadd -r audio && usermod -a -G audio ros

# Set up entrypoint
CMD ["bash", "-c", "source /opt/ros/humble/setup.sh && source /ws/install/setup.sh && exec \"$@\"", "--"]
```

## Configuration Files

Here's an example configuration file for the conversational robotics system:

```yaml
# config/conversational_config.yaml
conversational_system:
  ros__parameters:
    # Speech recognition parameters
    speech_recognition:
      confidence_threshold: 0.7
      listening_timeout: 5.0
      phrase_time_limit: 10.0
      sample_rate: 16000
      chunk_size: 1024

    # TTS parameters
    text_to_speech:
      rate: 150
      volume: 0.9
      voice_selection: "automatic"  # female, male, or automatic

    # Dialog management parameters
    dialog_manager:
      max_conversation_history: 10
      context_retention_time: 300  # 5 minutes
      intent_confidence_threshold: 0.6

    # Action execution parameters
    action_execution:
      default_linear_speed: 0.5
      default_angular_speed: 0.5
      action_timeout: 10.0
      safety_stop_distance: 0.5
```

## Summary

These code examples demonstrate various approaches to implementing conversational robotics systems:

1. **Complete Conversational System**: A full system integrating speech recognition, natural language understanding, dialog management, and action execution
2. **Whisper + GPT Integration**: Advanced system using state-of-the-art models for speech recognition and language understanding
3. **Launch Files**: Complete configuration for running the system with ROS 2
4. **Docker Configuration**: Deployment setup for easy distribution
5. **Parameter Configuration**: YAML files for system configuration

Each example builds upon the previous one, showing how to progressively add more sophisticated capabilities to conversational robotics systems. These examples can be used as starting points for your own conversational robotics implementations.