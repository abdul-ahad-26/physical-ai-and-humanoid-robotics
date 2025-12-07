---
title: Conversational Robotics
sidebar_position: 9.4
description: Implementing conversational interfaces for human-robot interaction
---

# Conversational Robotics

## Learning Objectives

- Understand the principles of conversational AI for robotics
- Learn how to integrate speech recognition and natural language processing
- Implement voice-based interaction systems for robots
- Design effective human-robot conversation flows
- Integrate conversational systems with robot actions
- Evaluate conversational system performance and user experience

## Introduction to Conversational Robotics

Conversational robotics combines artificial intelligence, natural language processing, and robotics to enable natural human-robot interaction through spoken language. This field encompasses:

- **Automatic Speech Recognition (ASR)**: Converting speech to text
- **Natural Language Understanding (NLU)**: Interpreting the meaning of text
- **Dialog Management**: Maintaining coherent conversation flow
- **Natural Language Generation (NLG)**: Creating appropriate responses
- **Text-to-Speech (TTS)**: Converting text responses to speech
- **Action Integration**: Connecting conversation to robot behaviors

## Architecture of Conversational Robotics Systems

### High-Level Architecture

```
Human Speech → ASR → NLU → Dialog Manager → NLG → TTS → Robot Response
     ↑                                           ↓
     ←--------- Action Execution Layer ←---------
```

### Component Integration with ROS 2

Conversational robotics systems in ROS 2 typically involve:

- **Audio Input Nodes**: Capture and process microphone input
- **Speech Recognition Nodes**: Convert speech to text
- **Language Understanding Nodes**: Interpret user intent
- **Dialog Management Nodes**: Manage conversation state
- **Response Generation Nodes**: Create appropriate responses
- **Speech Synthesis Nodes**: Convert text to speech
- **Action Execution Nodes**: Perform robot actions based on conversation

## Automatic Speech Recognition (ASR)

### Speech Recognition Options

#### Online Services
- **Google Speech-to-Text API**: High accuracy, requires internet
- **Microsoft Azure Speech Service**: Enterprise-grade features
- **AWS Transcribe**: Scalable cloud-based solution

#### Offline Solutions
- **Vosk**: Lightweight, runs locally, good for privacy
- **Coqui STT**: Open-source, based on DeepSpeech
- **Kaldi**: Traditional but powerful speech recognition toolkit

### Example ASR Implementation

```python
import speech_recognition as sr
import rclpy
from rclpy.node import Node
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
        self.timer = self.create_timer(1.0, self.listen_callback)

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
```

## Natural Language Understanding (NLU)

### Intent Recognition

Intent recognition determines what the user wants to do:

```python
class IntentClassifier:
    """
    Simple intent classifier for conversational robotics
    """

    def __init__(self):
        self.intents = {
            'navigation': [
                'go to', 'move to', 'navigate to', 'go forward',
                'go back', 'turn left', 'turn right', 'stop'
            ],
            'object_interaction': [
                'pick up', 'grasp', 'take', 'get', 'find',
                'look for', 'locate', 'show me'
            ],
            'information_request': [
                'what is', 'tell me about', 'describe',
                'how many', 'where is', 'what color'
            ],
            'social_interaction': [
                'hello', 'hi', 'good morning', 'goodbye',
                'thank you', 'please', 'sorry'
            ]
        }

    def classify_intent(self, text):
        """
        Classify the intent of the given text
        """
        text_lower = text.lower()

        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent, keyword

        return 'unknown', None
```

### Entity Extraction

Entity extraction identifies specific objects, locations, or values:

```python
import re


class EntityExtractor:
    """
    Extract named entities from text
    """

    def __init__(self):
        # Define patterns for common entities
        self.patterns = {
            'location': [
                r'the (\w+) room',
                r'(\w+) area',
                r'(\w+) table',
                r'(\w+) shelf'
            ],
            'object': [
                r'the (\w+) (?:object|item|thing)',
                r'(\w+) (?:box|cup|ball|toy)',
                r'(\w+) color(?:ed|) (?:object|item)'
            ],
            'number': [
                r'(\d+) of',
                r'(\d+) (?:times|minutes|seconds)',
                r'number (\d+)'
            ]
        }

    def extract_entities(self, text):
        """
        Extract entities from the given text
        """
        entities = {}

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].extend(matches)

        return entities
```

## Dialog Management

### State-Based Dialog Manager

```python
class DialogState:
    """
    Represents the current state of the dialog
    """
    def __init__(self):
        self.context = {}
        self.history = []
        self.current_intent = None
        self.waiting_for_response = False
        self.response_callback = None


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
        self.entity_extractor = EntityExtractor()
        self.dialog_state = DialogState()

        self.get_logger().info('Dialog Manager initialized')

    def text_callback(self, msg):
        """
        Process incoming recognized text
        """
        text = msg.data
        self.get_logger().info(f'Received text: {text}')

        # Classify intent
        intent, matched_keyword = self.intent_classifier.classify_intent(text)

        # Extract entities
        entities = self.entity_extractor.extract_entities(text)

        # Process based on intent
        response = self.process_intent(intent, text, entities)

        # Publish response
        if response:
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

    def process_intent(self, intent, text, entities):
        """
        Process the recognized intent and generate response
        """
        if intent == 'navigation':
            return self.handle_navigation_intent(text, entities)
        elif intent == 'object_interaction':
            return self.handle_object_intent(text, entities)
        elif intent == 'information_request':
            return self.handle_info_intent(text, entities)
        elif intent == 'social_interaction':
            return self.handle_social_intent(text)
        else:
            return "I'm not sure how to help with that. Can you rephrase?"

    def handle_navigation_intent(self, text, entities):
        """
        Handle navigation-related intents
        """
        # Determine navigation command
        if 'forward' in text:
            action_msg = String()
            action_msg.data = 'move_forward'
            self.action_pub.publish(action_msg)
            return "Moving forward."
        elif 'back' in text:
            action_msg = String()
            action_msg.data = 'move_backward'
            self.action_pub.publish(action_msg)
            return "Moving backward."
        elif 'left' in text:
            action_msg = String()
            action_msg.data = 'turn_left'
            self.action_pub.publish(action_msg)
            return "Turning left."
        elif 'right' in text:
            action_msg = String()
            action_msg.data = 'turn_right'
            self.action_pub.publish(action_msg)
            return "Turning right."
        else:
            return "Where would you like me to go?"

    def handle_object_intent(self, text, entities):
        """
        Handle object interaction intents
        """
        if entities.get('object'):
            obj = entities['object'][0]
            return f"I'll look for the {obj}."
        else:
            return "What object are you looking for?"

    def handle_info_intent(self, text, entities):
        """
        Handle information request intents
        """
        if 'color' in text:
            return "I can identify colors. What would you like to know?"
        else:
            return "I can help you find information. What are you looking for?"

    def handle_social_intent(self, text):
        """
        Handle social interaction intents
        """
        if any(greeting in text.lower() for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I assist you today?"
        elif any(farewell in text.lower() for farewell in ['goodbye', 'bye', 'see you']):
            return "Goodbye! Have a great day!"
        else:
            return "Hello! How can I help you?"
```

## Text-to-Speech (TTS)

### TTS Integration Example

```python
import pyttsx3
import rclpy
from rclpy.node import Node
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
```

## Integration with Whisper + GPT + ROS 2

### Advanced Conversational System

```python
import openai
import whisper
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import threading
import queue


class AdvancedConversationalRobot(Node):
    """
    Advanced conversational robot using Whisper, GPT, and ROS 2
    """

    def __init__(self):
        super().__init__('advanced_conversational_robot')

        # Initialize Whisper for speech recognition
        self.whisper_model = whisper.load_model("base")

        # Initialize GPT (set your API key)
        # openai.api_key = "your-api-key-here"

        # Queues for processing
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()

        # Publishers and subscribers
        self.response_pub = self.create_publisher(String, '/robot_response', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)

        # Audio input subscription
        # In a real system, you'd have an audio input node
        # For now, we'll simulate with text input
        self.text_input_sub = self.create_subscription(
            String, '/user_input', self.text_input_callback, 10
        )

        # Start processing threads
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Advanced Conversational Robot initialized')

    def text_input_callback(self, msg):
        """
        Handle text input (simulating speech recognition)
        """
        self.text_queue.put(msg.data)

    def process_loop(self):
        """
        Main processing loop for the conversational system
        """
        while rclpy.ok():
            try:
                # Get text from queue
                if not self.text_queue.empty():
                    user_text = self.text_queue.get()

                    # Process with GPT
                    response = self.process_with_gpt(user_text)

                    # Publish response
                    if response:
                        response_msg = String()
                        response_msg.data = response
                        self.response_pub.publish(response_msg)

                        # Extract actions from response
                        self.extract_and_execute_actions(response)

                # Small delay to prevent busy waiting
                time.sleep(0.1)

            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {e}')

    def process_with_gpt(self, user_text):
        """
        Process user input with GPT to generate response
        """
        try:
            # Create a conversation with GPT
            messages = [
                {"role": "system", "content": "You are a helpful robot assistant. Respond concisely and provide clear information. If the user requests an action, include action tags like [MOVE_FORWARD] or [TURN_LEFT]."},
                {"role": "user", "content": user_text}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150
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
            'pick_up': r'\[PICK_UP\]',
            'find_object': r'\[FIND (.+?)\]'
        }

        for action, pattern in action_patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                # Extract object name if applicable
                matches = re.findall(pattern, response, re.IGNORECASE)

                action_msg = String()
                if matches:
                    action_msg.data = f"{action}:{matches[0]}"
                else:
                    action_msg.data = action
                self.action_pub.publish(action_msg)


def main(args=None):
    rclpy.init(args=args)
    robot = AdvancedConversationalRobot()

    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        pass
    finally:
        robot.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Conversational Robotics

### Design Principles

1. **Clear Communication**: Use simple, unambiguous language
2. **Feedback Mechanisms**: Provide audio/visual feedback for understanding
3. **Error Handling**: Gracefully handle misunderstood commands
4. **Context Awareness**: Maintain conversation context across turns
5. **Privacy Considerations**: Handle sensitive information appropriately

### Performance Optimization

- **Latency**: Minimize response time for natural interaction
- **Accuracy**: Balance speed with recognition accuracy
- **Resource Usage**: Optimize for robot's computational constraints
- **Robustness**: Handle noisy environments and varied speakers

## Troubleshooting Tips

- If speech recognition fails frequently, check microphone quality and ambient noise levels
- For slow responses, consider using smaller models or edge computing
- If the robot misunderstands commands, improve the intent classification training data
- For poor conversation flow, implement better context management
- If TTS sounds unnatural, experiment with different voices or services
- For privacy concerns, implement local processing when possible

## Cross-References

For related concepts, see:
- [ROS 2 Fundamentals](../ros2-core/index.md) - For ROS 2 integration
- [VLA Systems](../vla/index.md) - For AI integration concepts
- [Human-Robot Interaction](../hri/index.md) - For interaction design principles
- [Isaac ROS](../isaac-ros/index.md) - For NVIDIA Isaac ROS integration
- [NVIDIA Isaac Sim](../isaac/index.md) - For simulation environment

## Isaac ROS Integration

For conversational robotics systems using NVIDIA Isaac ROS, the following integration patterns are common:

### Isaac ROS Perception Integration

Isaac ROS provides specialized perception nodes that can enhance conversational robotics:

- **Visual Perception**: Use Isaac ROS perception pipelines for object detection and scene understanding
- **Audio Processing**: Leverage Isaac ROS audio processing for enhanced speech recognition in noisy environments
- **Sensor Fusion**: Combine multiple sensor modalities for robust understanding

### Example Isaac ROS Integration

```python
# Example of integrating Isaac ROS perception with conversational system
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from audio_common_msgs.msg import AudioData
from std_msgs.msg import String


class IsaacROSConversationalNode(Node):
    """
    Conversational node integrated with Isaac ROS perception
    """

    def __init__(self):
        super().__init__('isaac_ros_conversational_node')

        # Subscribe to Isaac ROS perception outputs
        self.perception_sub = self.create_subscription(
            Detection2DArray,
            '/isaac_ros/detections',
            self.perception_callback,
            10
        )

        # Subscribe to Isaac ROS audio
        self.audio_sub = self.create_subscription(
            AudioData,
            '/isaac_ros/audio',
            self.audio_callback,
            10
        )

        # Initialize other components as before
        # ... rest of initialization

    def perception_callback(self, msg):
        """
        Process detections from Isaac ROS perception pipeline
        """
        detected_objects = []
        for detection in msg.detections:
            label = detection.results[0].hypothesis.name if detection.results else "unknown"
            confidence = detection.results[0].hypothesis.score if detection.results else 0.0

            if confidence > 0.5:  # Threshold for valid detection
                detected_objects.append(label)

        # Update system context with detected objects
        self.update_context_with_objects(detected_objects)

    def audio_callback(self, msg):
        """
        Process audio from Isaac ROS audio pipeline
        """
        # In a real system, this would interface with Isaac ROS audio processing
        # For now, we'll just log the audio data size
        self.get_logger().info(f'Received Isaac ROS audio data: {len(msg.data)} bytes')
```

## Summary

Conversational robotics enables natural human-robot interaction through speech and language. Success requires careful integration of speech recognition, natural language processing, dialog management, and action execution. Modern approaches leverage advanced AI models like Whisper and GPT for more sophisticated interactions, while traditional approaches use rule-based systems for more predictable behavior.