#!/usr/bin/env python3

"""
Voice Command Processor for ROS 2

This module processes voice commands using Whisper for speech recognition
and executes appropriate ROS 2 actions based on the recognized commands.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import whisper
import threading
import queue
import tempfile
import wave
import pyaudio
import time
import re


class VoiceCommandProcessor(Node):
    """
    Process voice commands and execute ROS 2 robot actions
    """

    def __init__(self):
        super().__init__('voice_command_processor')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.whisper_model = whisper.load_model("base")
        self.get_logger().info('Whisper model loaded')

        # Initialize components
        self.bridge = CvBridge()

        # Initialize PyAudio for audio recording
        self.pyaudio_instance = pyaudio.PyAudio()

        # Queues for processing
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/voice_status', 10)

        # Audio recording parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 4  # Shorter for voice commands

        # Command mapping
        self.command_mapping = {
            'move forward': 'forward',
            'go forward': 'forward',
            'move backward': 'backward',
            'go backward': 'backward',
            'turn left': 'left',
            'turn right': 'right',
            'stop': 'stop',
            'halt': 'stop',
            'freeze': 'stop',
            'look around': 'look_around',
            'take picture': 'take_picture',
            'find object': 'find_object',
            'hello': 'greet',
            'hi': 'greet',
            'goodbye': 'farewell',
            'bye': 'farewell'
        }

        # Start processing threads
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.processing_thread = threading.Thread(target=self.process_loop, daemon=True)

        self.recording_thread.start()
        self.processing_thread.start()

        self.get_logger().info('Voice Command Processor initialized')

    def record_audio(self):
        """
        Record audio from microphone
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
        Main processing loop for voice commands
        """
        while rclpy.ok():
            try:
                # Process audio files
                if not self.audio_queue.empty():
                    audio_file = self.audio_queue.get()

                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(audio_file)
                    text = result['text'].lower()

                    if text.strip():  # Only process non-empty transcriptions
                        self.get_logger().info(f'Heard: {text}')

                        # Map to command
                        command = self.map_voice_to_command(text)

                        if command:
                            # Add to command queue
                            self.command_queue.put(command)

                            # Publish status
                            status_msg = String()
                            status_msg.data = f'heard:{text}|mapped_to:{command}'
                            self.status_pub.publish(status_msg)

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {e}')

    def map_voice_to_command(self, text):
        """
        Map recognized text to robot commands
        """
        # Check for exact matches first
        for voice_command, robot_command in self.command_mapping.items():
            if voice_command in text:
                return robot_command

        # Check for partial matches
        for voice_command, robot_command in self.command_mapping.items():
            # Use regex for more flexible matching
            if re.search(rf'\b{re.escape(voice_command)}\b', text):
                return robot_command

        # If no match found, return None
        return None

    def execute_command(self, command):
        """
        Execute the mapped robot command
        """
        twist_msg = Twist()

        if command == 'forward':
            twist_msg.linear.x = 0.5
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info('Moving forward')
            time.sleep(2)  # Move for 2 seconds
        elif command == 'backward':
            twist_msg.linear.x = -0.5
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info('Moving backward')
            time.sleep(2)
        elif command == 'left':
            twist_msg.angular.z = 0.5
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info('Turning left')
            time.sleep(1)  # Turn for 1 second
        elif command == 'right':
            twist_msg.angular.z = -0.5
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info('Turning right')
            time.sleep(1)
        elif command == 'stop':
            # Already zero by default, just publish
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info('Stopping')
        elif command == 'look_around':
            # Turn slowly to look around
            twist_msg.angular.z = 0.2
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info('Looking around')
            time.sleep(5)  # Look for 5 seconds
        elif command == 'greet':
            self.get_logger().info('Greeting user')
        elif command == 'farewell':
            self.get_logger().info('Saying goodbye')
        else:
            self.get_logger().info(f'Unknown command: {command}')

        # Stop robot after action
        time.sleep(0.1)
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

    def command_processing_loop(self):
        """
        Process commands from the queue
        """
        while rclpy.ok():
            try:
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    self.execute_command(command)

                time.sleep(0.1)

            except Exception as e:
                self.get_logger().error(f'Error in command processing: {e}')

    def cleanup(self):
        """
        Cleanup function
        """
        # Stop audio stream
        self.pyaudio_instance.terminate()

        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        self.get_logger().info('Voice Command Processor cleaned up')


def main(args=None):
    """
    Main function to run the voice command processor
    """
    rclpy.init(args=args)

    node = VoiceCommandProcessor()

    try:
        # Start command processing in a separate thread
        command_thread = threading.Thread(target=node.command_processing_loop, daemon=True)
        command_thread.start()

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

Now let me update the sidebars.js to include the conversational robotics chapter navigation (T037):
