#!/usr/bin/env python3

"""
Whisper + GPT + ROS 2 Integration Example

This module demonstrates how to integrate OpenAI Whisper for speech recognition,
GPT for natural language understanding, and ROS 2 for robot control in a
conversational robotics system.
"""

import openai
import whisper
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import queue
import time
import tempfile
import wave
import pyaudio
import json
import re


class WhisperGPTROSNode(Node):
    """
    Complete conversational robotics node integrating Whisper, GPT, and ROS 2
    """

    def __init__(self):
        super().__init__('whisper_gpt_ros_node')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.whisper_model = whisper.load_model("base")
        self.get_logger().info('Whisper model loaded successfully')

        # Initialize OpenAI GPT (ensure you set your API key)
        # openai.api_key = "your-openai-api-key-here"

        # Initialize components
        self.bridge = CvBridge()

        # Initialize PyAudio for audio recording
        self.pyaudio_instance = pyaudio.PyAudio()

        # Queues for processing
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.image_queue = queue.Queue()

        # Publishers
        self.response_pub = self.create_publisher(String, '/robot_response', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Audio recording parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5

        # System state
        self.system_state = {
            'current_task': None,
            'conversation_history': [],
            'objects_detected': [],
            'locations_visited': [],
            'last_interaction_time': time.time()
        }

        # Start processing threads
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.gpt_processing_thread = threading.Thread(target=self.gpt_process_loop, daemon=True)

        self.recording_thread.start()
        self.processing_thread.start()
        self.gpt_processing_thread.start()

        self.get_logger().info('Whisper-GPT-ROS Node initialized successfully')

    def image_callback(self, msg):
        """
        Process incoming camera images
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Add image to processing queue
            self.image_queue.put(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def record_audio(self):
        """
        Continuously record audio from microphone
        """
        self.get_logger().info('Starting audio recording...')

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
        Main processing loop for audio transcription
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

                        # Add to text processing queue for GPT
                        self.text_queue.put(text)

                        # Update system state
                        self.system_state['last_interaction_time'] = time.time()
                        self.system_state['conversation_history'].append({
                            'speaker': 'user',
                            'text': text,
                            'timestamp': time.time()
                        })

                        # Keep only last 10 exchanges
                        if len(self.system_state['conversation_history']) > 10:
                            self.system_state['conversation_history'] = self.system_state['conversation_history'][-10:]

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                self.get_logger().error(f'Error in audio processing loop: {e}')

    def gpt_process_loop(self):
        """
        Loop for processing text with GPT
        """
        while rclpy.ok():
            try:
                if not self.text_queue.empty():
                    user_text = self.text_queue.get()

                    # Process with GPT
                    response = self.process_with_gpt(user_text)

                    if response:
                        # Publish response
                        response_msg = String()
                        response_msg.data = response
                        self.response_pub.publish(response_msg)

                        # Add to conversation history
                        self.system_state['conversation_history'].append({
                            'speaker': 'robot',
                            'text': response,
                            'timestamp': time.time()
                        })

                        # Extract and execute actions
                        self.extract_and_execute_actions(response)

                        # Update and publish system status
                        self.update_system_status()

                time.sleep(0.1)

            except Exception as e:
                self.get_logger().error(f'Error in GPT processing loop: {e}')

    def process_with_gpt(self, user_text):
        """
        Process user input with GPT to generate response
        """
        try:
            # Prepare context for GPT
            context = self.build_context()

            # Create a conversation with GPT
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful robot assistant in a conversational robotics system. "
                        "Respond concisely and provide clear information. If the user requests an action, "
                        "include action tags like [MOVE_FORWARD] or [TURN_LEFT]. Only use the following "
                        "action tags: [MOVE_FORWARD], [MOVE_BACKWARD], [TURN_LEFT], [TURN_RIGHT], [STOP], "
                        "[FIND_OBJECT], [NAVIGATE_TO], [TAKE_PICTURE], [DESCRIBE_SCENE]. "
                        f"Context: {context}"
                    )
                },
                {"role": "user", "content": user_text}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )

            return response.choices[0].message['content'].strip()

        except Exception as e:
            self.get_logger().error(f'Error calling GPT: {e}')
            return "I'm having trouble processing your request right now. Could you please repeat that?"

    def build_context(self):
        """
        Build context for GPT based on current system state
        """
        context_parts = []

        # Add recent conversation history
        recent_conv = self.system_state['conversation_history'][-3:]  # Last 3 exchanges
        if recent_conv:
            conv_context = "Recent conversation: "
            for exchange in recent_conv:
                speaker = "User" if exchange['speaker'] == 'user' else "Robot"
                conv_context += f"{speaker}: {exchange['text']}; "
            context_parts.append(conv_context)

        # Add detected objects (if any)
        if self.system_state['objects_detected']:
            objects = ", ".join(self.system_state['objects_detected'][:5])  # Limit to 5
            context_parts.append(f"Objects detected: {objects}")

        # Add visited locations
        if self.system_state['locations_visited']:
            locations = ", ".join(self.system_state['locations_visited'][-3:])  # Last 3
            context_parts.append(f"Locations visited: {locations}")

        # Add current task
        if self.system_state['current_task']:
            context_parts.append(f"Current task: {self.system_state['current_task']}")

        return " ".join(context_parts)

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
            'navigate_to': r'\[NAVIGATE_TO\]',
            'take_picture': r'\[TAKE_PICTURE\]',
            'describe_scene': r'\[DESCRIBE_SCENE\]'
        }

        for action, pattern in action_patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                action_msg = String()
                action_msg.data = action
                self.action_pub.publish(action_msg)

                self.system_state['current_task'] = action
                self.get_logger().info(f'Executed action: {action}')

                # Execute the specific action
                self.execute_specific_action(action)

    def execute_specific_action(self, action):
        """
        Execute a specific robot action
        """
        twist_msg = Twist()

        if action == 'move_forward':
            twist_msg.linear.x = 0.5
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(2)  # Move for 2 seconds
        elif action == 'move_backward':
            twist_msg.linear.x = -0.5
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(2)
        elif action == 'turn_left':
            twist_msg.angular.z = 0.5
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(1)  # Turn for 1 second
        elif action == 'turn_right':
            twist_msg.angular.z = -0.5
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(1)
        elif action == 'stop':
            # Already zero by default, just publish
            self.cmd_vel_pub.publish(twist_msg)
        elif action == 'find_object':
            self.get_logger().info('Looking for objects in the environment')
            # In a real system, this would trigger object detection
        elif action == 'navigate_to':
            self.get_logger().info('Preparing to navigate to a location')
            # In a real system, this would trigger navigation
        elif action == 'take_picture':
            self.get_logger().info('Taking a picture')
            # In a real system, this would trigger camera capture
        elif action == 'describe_scene':
            self.get_logger().info('Describing the current scene')
            # In a real system, this would analyze the current image

        # Stop robot after action
        time.sleep(0.1)
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

    def update_system_status(self):
        """
        Update and publish system status
        """
        status_msg = String()
        status_msg.data = json.dumps(self.system_state)
        self.status_pub.publish(status_msg)

    def cleanup(self):
        """
        Cleanup function
        """
        # Stop audio stream
        self.pyaudio_instance.terminate()

        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        self.get_logger().info('Whisper-GPT-ROS Node cleaned up')


def main(args=None):
    """
    Main function to run the Whisper + GPT + ROS 2 integration
    """
    rclpy.init(args=args)

    node = WhisperGPTROSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Now let me create a more specific example for voice command processing:
