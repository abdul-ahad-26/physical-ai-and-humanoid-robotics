---
title: Human-Robot Interaction (HRI)
sidebar_position: 10.1
description: Principles and implementation of Human-Robot Interaction systems
---

# Human-Robot Interaction (HRI)

## Learning Objectives

- Understand the fundamental principles of Human-Robot Interaction (HRI)
- Implement multimodal interaction systems (speech, gestures, touch)
- Design intuitive robot interfaces and communication protocols
- Integrate social robotics principles into robot behavior
- Evaluate HRI system usability and effectiveness
- Create accessible interaction systems for diverse users
- Troubleshoot common HRI implementation issues

## Introduction to Human-Robot Interaction

Human-Robot Interaction (HRI) is an interdisciplinary field focused on understanding, designing, and evaluating robotic systems for human use. HRI encompasses the study of how humans and robots communicate, collaborate, and interact in various contexts.

### Key HRI Principles

1. **Intuitive Communication**: Natural interaction methods that don't require special training
2. **Trust and Transparency**: Clear robot intentions and reliable behavior
3. **Social Acceptance**: Robot behaviors that are culturally appropriate and comfortable
4. **Accessibility**: Systems usable by people with diverse abilities
5. **Safety**: Physical and psychological safety during interaction
6. **Adaptability**: Systems that learn and adapt to individual users

### HRI Domains

- **Service Robotics**: Customer service, healthcare assistance, domestic help
- **Industrial Collaboration**: Cobots working alongside humans
- **Educational Robotics**: Teaching and learning applications
- **Entertainment**: Interactive games and experiences
- **Assistive Robotics**: Support for elderly, disabled, or special needs

## HRI Communication Modalities

### Speech-Based Interaction

Speech provides a natural communication channel for HRI systems:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import speech_recognition as sr
import pyttsx3
import threading
import queue


class SpeechInteractionNode(Node):
    """
    Speech-based interaction for HRI systems
    """

    def __init__(self):
        super().__init__('speech_interaction_node')

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

        # Publishers
        self.response_pub = self.create_publisher(String, '/hri_response', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/hri_command', self.command_callback, 10
        )

        # Internal state
        self.speech_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.user_profile = {}  # Store user preferences and history

        # Start speech recognition thread
        self.speech_thread = threading.Thread(target=self.speech_recognition_loop, daemon=True)
        self.speech_thread.start()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.get_logger().info('Speech Interaction Node initialized')

    def speech_recognition_loop(self):
        """
        Continuous speech recognition loop
        """
        while rclpy.ok():
            try:
                with self.microphone as source:
                    self.get_logger().info('Listening...')
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)

                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                self.get_logger().info(f'Recognized: {text}')

                # Add to processing queue
                self.speech_queue.put(text)

            except sr.WaitTimeoutError:
                continue  # Just keep listening
            except sr.UnknownValueError:
                self.get_logger().info('Could not understand audio')
            except sr.RequestError as e:
                self.get_logger().error(f'Error with speech recognition service: {e}')

    def command_callback(self, msg):
        """
        Process commands from other nodes
        """
        command = msg.data
        self.get_logger().info(f'Received command: {command}')
        self.command_queue.put(command)

    def process_speech_command(self, text):
        """
        Process speech command and generate response
        """
        text_lower = text.lower()

        # Intent recognition
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = self.handle_greeting(text)
        elif any(word in text_lower for word in ['help', 'assist', 'what can you do']):
            response = self.handle_help_request(text)
        elif any(word in text_lower for word in ['move', 'go', 'navigate', 'forward', 'backward', 'left', 'right']):
            response = self.handle_navigation_command(text)
        elif any(word in text_lower for word in ['stop', 'halt', 'pause']):
            response = self.handle_stop_command(text)
        elif any(word in text_lower for word in ['who are you', 'what are you']):
            response = self.handle_identity_query(text)
        else:
            response = self.handle_unknown_command(text)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        # Speak response
        self.speak_response(response)

        return response

    def handle_greeting(self, command):
        """
        Handle greeting commands
        """
        # Personalize greeting based on user profile
        user_name = self.user_profile.get('name', 'friend')
        responses = [
            f"Hello {user_name}! How can I assist you today?",
            f"Hi there {user_name}! Ready to help!",
            f"Greetings {user_name}! What would you like to do?"
        ]

        import random
        return random.choice(responses)

    def handle_help_request(self, command):
        """
        Handle help requests
        """
        capabilities = [
            "I can navigate to different locations",
            "I can provide information about my environment",
            "I can follow basic movement commands",
            "I can answer questions about robotics"
        ]

        help_text = "I can help with several things: " + "; ".join(capabilities) + ". What would you like me to do?"
        return help_text

    def handle_navigation_command(self, command):
        """
        Handle navigation commands
        """
        # Parse navigation command
        if 'forward' in command.lower() or 'ahead' in command.lower():
            self.move_robot('forward')
            return "Moving forward as requested."
        elif 'backward' in command.lower() or 'back' in command.lower():
            self.move_robot('backward')
            return "Moving backward as requested."
        elif 'left' in command.lower():
            self.move_robot('left')
            return "Turning left as requested."
        elif 'right' in command.lower():
            self.move_robot('right')
            return "Turning right as requested."
        elif 'to' in command.lower():
            # Extract destination
            destination = self.extract_destination(command)
            if destination:
                self.navigate_to_location(destination)
                return f"Navigating to {destination}."
            else:
                return "I didn't catch where you want me to go. Could you repeat that?"
        else:
            return "I can move forward, backward, left, or right. Where would you like me to go?"

    def handle_stop_command(self, command):
        """
        Handle stop commands
        """
        self.stop_robot()
        return "Stopping as requested."

    def handle_identity_query(self, command):
        """
        Handle identity queries
        """
        identity_info = (
            "I am an AI-powered robot designed to assist with various tasks. "
            "I can navigate, recognize objects, understand speech, and interact socially. "
            "My purpose is to help make your life easier and more convenient."
        )
        return identity_info

    def handle_unknown_command(self, command):
        """
        Handle unrecognized commands
        """
        return f"I'm not sure how to help with '{command}'. Could you try rephrasing your request?"

    def move_robot(self, direction):
        """
        Move robot in specified direction
        """
        cmd_vel = Twist()

        if direction == 'forward':
            cmd_vel.linear.x = 0.3
        elif direction == 'backward':
            cmd_vel.linear.x = -0.3
        elif direction == 'left':
            cmd_vel.angular.z = 0.3
        elif direction == 'right':
            cmd_vel.angular.z = -0.3

        self.cmd_vel_pub.publish(cmd_vel)

    def stop_robot(self):
        """
        Stop robot movement
        """
        cmd_vel = Twist()  # Zero velocities
        self.cmd_vel_pub.publish(cmd_vel)

    def navigate_to_location(self, location):
        """
        Navigate to specified location
        """
        # This would interface with navigation stack
        # For now, just move forward as an example
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2
        self.cmd_vel_pub.publish(cmd_vel)

    def extract_destination(self, command):
        """
        Extract destination from command (simplified)
        """
        # In a real implementation, this would use NLP
        # For this example, we'll look for common location words
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'door', 'window']

        for loc in locations:
            if loc in command.lower():
                return loc

        return None

    def speak_response(self, response):
        """
        Speak the response using TTS
        """
        def speak_in_thread():
            self.tts_engine.say(response)
            self.tts_engine.runAndWait()

        # Speak in separate thread to avoid blocking
        speak_thread = threading.Thread(target=speak_in_thread)
        speak_thread.start()

    def process_queues(self):
        """
        Process command and speech queues
        """
        # Process speech commands
        while not self.speech_queue.empty():
            speech_text = self.speech_queue.get()
            self.process_speech_command(speech_text)

        # Process other commands
        while not self.command_queue.empty():
            command = self.command_queue.get()
            # Process command (could be from other nodes)
            pass
```

### Gesture Recognition

Gesture recognition enables non-verbal communication:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import numpy as np


class GestureRecognitionNode(Node):
    """
    Gesture recognition for HRI systems
    """

    def __init__(self):
        super().__init__('gesture_recognition_node')

        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Publishers
        self.gesture_pub = self.create_publisher(String, '/hri_gesture', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # CV Bridge for image conversion
        from cv_bridge import CvBridge
        self.bridge = CvBridge()

        # Gesture state
        self.previous_gesture = None
        self.gesture_history = []
        self.gesture_cooldown = 0

        self.get_logger().info('Gesture Recognition Node initialized')

    def image_callback(self, msg):
        """
        Process camera image for gesture recognition
        """
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process image for hand landmarks
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    # Analyze gesture
                    gesture = self.analyze_gesture(hand_landmarks, cv_image.shape)

                    if gesture and gesture != self.previous_gesture:
                        self.handle_gesture(gesture)
                        self.previous_gesture = gesture
                        self.gesture_history.append(gesture)

                        # Publish gesture
                        gesture_msg = String()
                        gesture_msg.data = gesture
                        self.gesture_pub.publish(gesture_msg)

                        # Set cooldown to prevent spam
                        self.gesture_cooldown = 10  # 10 cycles cooldown

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def analyze_gesture(self, landmarks, image_shape):
        """
        Analyze hand landmarks to determine gesture
        """
        image_height, image_width, _ = image_shape

        # Get landmark positions
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # Convert normalized coordinates to pixel coordinates
        wrist_x = int(wrist.x * image_width)
        wrist_y = int(wrist.y * image_height)
        thumb_x = int(thumb_tip.x * image_width)
        thumb_y = int(thumb_tip.y * image_height)
        index_x = int(index_tip.x * image_width)
        index_y = int(index_tip.y * image_height)
        middle_x = int(middle_tip.x * image_width)
        middle_y = int(middle_tip.y * image_height)
        ring_x = int(ring_tip.x * image_width)
        ring_y = int(ring_tip.y * image_height)
        pinky_x = int(pinky_tip.x * image_width)
        pinky_y = int(pinky_tip.y * image_height)

        # Calculate distances between fingertips and palm
        index_palm_dist = np.sqrt((index_x - wrist_x)**2 + (index_y - wrist_y)**2)
        middle_palm_dist = np.sqrt((middle_x - wrist_x)**2 + (middle_y - wrist_y)**2)
        ring_palm_dist = np.sqrt((ring_x - wrist_x)**2 + (ring_y - wrist_y)**2)
        pinky_palm_dist = np.sqrt((pinky_x - wrist_x)**2 + (pinky_y - wrist_y)**2)

        # Define gesture detection logic
        if self.is_fist(landmarks, image_shape):
            return 'stop'
        elif self.is_open_hand(landmarks, image_shape):
            return 'go'
        elif self.is_pointing_gesture(landmarks, image_shape):
            return 'pointing'
        elif self.is_victory_sign(landmarks, image_shape):
            return 'victory'
        elif self.is_thumb_up(landmarks, image_shape):
            return 'thumb_up'
        elif self.is_thumbs_up(landmarks, image_shape):
            return 'thumbs_up'
        elif self.is_wave_gesture(landmarks, image_shape):
            return 'wave'
        else:
            return None

    def is_fist(self, landmarks, image_shape):
        """
        Detect fist gesture
        """
        # Check if all fingers are bent (tips near palm)
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        finger_tips = [
            landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]

        image_height, image_width, _ = image_shape
        wrist_x = int(wrist.x * image_width)
        wrist_y = int(wrist.y * image_height)

        for tip in finger_tips:
            tip_x = int(tip.x * image_width)
            tip_y = int(tip.y * image_height)
            distance = np.sqrt((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2)

            # If fingertips are too far from palm, it's not a fist
            if distance > 100:  # Adjust threshold as needed
                return False

        return True

    def is_open_hand(self, landmarks, image_shape):
        """
        Detect open hand gesture
        """
        # Check if all fingers are extended (tips far from palm)
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        finger_tips = [
            landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]

        image_height, image_width, _ = image_shape
        wrist_x = int(wrist.x * image_width)
        wrist_y = int(wrist.y * image_height)

        extended_count = 0
        for tip in finger_tips:
            tip_x = int(tip.x * image_width)
            tip_y = int(tip.y * image_height)
            distance = np.sqrt((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2)

            # If fingertip is far enough from palm, consider it extended
            if distance > 150:  # Adjust threshold as needed
                extended_count += 1

        # If most fingers are extended, it's an open hand
        return extended_count >= 3

    def is_pointing_gesture(self, landmarks, image_shape):
        """
        Detect pointing gesture (index finger extended, others bent)
        """
        # Index finger should be extended
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        image_height, image_width, _ = image_shape
        wrist_x = int(wrist.x * image_width)
        wrist_y = int(wrist.y * image_height)
        index_x = int(index_tip.x * image_width)
        index_y = int(index_tip.y * image_height)

        index_distance = np.sqrt((index_x - wrist_x)**2 + (index_y - wrist_y)**2)

        # Index finger should be extended (far from palm)
        if index_distance < 150:
            return False

        # Other fingers should be bent (close to palm)
        other_fingers = [middle_tip, ring_tip, pinky_tip]
        for finger in other_fingers:
            finger_x = int(finger.x * image_width)
            finger_y = int(finger.y * image_height)
            finger_distance = np.sqrt((finger_x - wrist_x)**2 + (finger_y - wrist_y)**2)

            if finger_distance > 100:  # Too far, not bent
                return False

        return True

    def is_victory_sign(self, landmarks, image_shape):
        """
        Detect victory sign (index and middle fingers extended)
        """
        # Index and middle fingers extended, others bent
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        image_height, image_width, _ = image_shape
        wrist_x = int(wrist.x * image_width)
        wrist_y = int(wrist.y * image_height)

        # Get distances
        index_dist = np.sqrt((int(index_tip.x * image_width) - wrist_x)**2 + (int(index_tip.y * image_height) - wrist_y)**2)
        middle_dist = np.sqrt((int(middle_tip.x * image_width) - wrist_x)**2 + (int(middle_tip.y * image_height) - wrist_y)**2)
        ring_dist = np.sqrt((int(ring_tip.x * image_width) - wrist_x)**2 + (int(ring_tip.y * image_height) - wrist_y)**2)
        pinky_dist = np.sqrt((int(pinky_tip.x * image_width) - wrist_x)**2 + (int(pinky_tip.y * image_height) - wrist_y)**2)

        # Index and middle extended, ring and pinky bent
        return (index_dist > 150 and middle_dist > 150 and
                ring_dist < 100 and pinky_dist < 100)

    def is_thumb_up(self, landmarks, image_shape):
        """
        Detect thumb up gesture
        """
        # Thumb extended up, other fingers bent
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        image_height, image_width, _ = image_shape
        wrist_x = int(wrist.x * image_width)
        wrist_y = int(wrist.y * image_height)

        # Get distances
        thumb_dist = np.sqrt((int(thumb_tip.x * image_width) - wrist_x)**2 + (int(thumb_tip.y * image_height) - wrist_y)**2)
        index_dist = np.sqrt((int(index_tip.x * image_width) - wrist_x)**2 + (int(index_tip.y * image_height) - wrist_y)**2)
        middle_dist = np.sqrt((int(middle_tip.x * image_width) - wrist_x)**2 + (int(middle_tip.y * image_height) - wrist_y)**2)
        ring_dist = np.sqrt((int(ring_tip.x * image_width) - wrist_x)**2 + (int(ring_tip.y * image_height) - wrist_y)**2)
        pinky_dist = np.sqrt((int(pinky_tip.x * image_width) - wrist_x)**2 + (int(pinky_tip.y * image_height) - wrist_y)**2)

        # Thumb extended, others bent
        return (thumb_dist > 150 and
                index_dist < 100 and middle_dist < 100 and
                ring_dist < 100 and pinky_dist < 100)

    def handle_gesture(self, gesture):
        """
        Handle recognized gesture
        """
        self.get_logger().info(f'Recognized gesture: {gesture}')

        if gesture == 'stop':
            self.stop_robot()
            self.speak_response("Stopping as requested.")
        elif gesture == 'go':
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.3
            self.cmd_vel_pub.publish(cmd_vel)
            self.speak_response("Moving forward.")
        elif gesture == 'pointing':
            self.speak_response("I see you're pointing. How can I help?")
        elif gesture == 'victory':
            self.speak_response("Peace sign detected! What would you like to do?")
        elif gesture == 'thumb_up':
            self.speak_response("Thumbs up! I appreciate the positive feedback.")
        elif gesture == 'wave':
            self.speak_response("Wave detected! Hello there!")

    def speak_response(self, response):
        """
        Publish response for TTS system to handle
        """
        from std_msgs.msg import String
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)
```

### Touch-Based Interaction

Touch interfaces provide direct physical interaction:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import pygame
import math


class TouchInteractionNode(Node):
    """
    Touch-based interaction for HRI systems
    """

    def __init__(self):
        super().__init__('touch_interaction_node')

        # Initialize pygame for touch interface
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("HRI Touch Interface")

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)

        # Touch interface elements
        self.interface_elements = self.create_interface_elements()

        # Publishers
        self.touch_event_pub = self.create_publisher(String, '/hri_touch_event', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.ui_state_pub = self.create_publisher(String, '/hri_ui_state', 10)

        # Subscribers
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joystick_callback, 10)

        # Touch state
        self.active_touches = {}
        self.ui_state = {
            'selected_element': None,
            'gesture_in_progress': False,
            'last_touch_position': (0, 0)
        }

        # Start touch processing loop
        self.touch_processing_timer = self.create_timer(0.033, self.process_touch_events)  # ~30 FPS

        self.get_logger().info('Touch Interaction Node initialized')

    def create_interface_elements(self):
        """
        Create touch interface elements
        """
        elements = []

        # Navigation controls (virtual joystick)
        nav_area = {
            'id': 'nav_joystick',
            'type': 'joystick',
            'rect': pygame.Rect(50, 500, 200, 200),
            'center': (150, 600),
            'radius': 100,
            'active': False,
            'value': (0.0, 0.0)
        }
        elements.append(nav_area)

        # Command buttons
        button_positions = [
            (300, 500, 'Move Forward', 'forward'),
            (300, 550, 'Turn Left', 'left'),
            (450, 550, 'Turn Right', 'right'),
            (300, 600, 'Move Back', 'backward'),
            (450, 600, 'Stop', 'stop'),
            (300, 650, 'Help', 'help'),
            (450, 650, 'Reset', 'reset')
        ]

        for x, y, label, command in button_positions:
            button = {
                'id': f'button_{command}',
                'type': 'button',
                'rect': pygame.Rect(x, y, 120, 40),
                'label': label,
                'command': command,
                'active': False
            }
            elements.append(button)

        # Status display
        status_display = {
            'id': 'status_display',
            'type': 'display',
            'rect': pygame.Rect(600, 500, 200, 140),
            'text': 'Ready',
            'color': self.GREEN
        }
        elements.append(status_display)

        return elements

    def process_touch_events(self):
        """
        Process touch events and update interface
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_touch_start(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.handle_touch_end(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_touch_move(event.pos)

        # Update display
        self.update_display()

    def handle_touch_start(self, pos):
        """
        Handle touch start event
        """
        x, y = pos

        # Check if touch is in any interface element
        for element in self.interface_elements:
            if element['rect'].collidepoint(x, y):
                element['active'] = True
                self.ui_state['selected_element'] = element['id']

                # Handle different element types
                if element['type'] == 'button':
                    self.execute_button_command(element['command'])
                elif element['type'] == 'joystick':
                    self.ui_state['last_touch_position'] = pos
                    self.handle_joystick_touch(pos, element)

                # Publish touch event
                touch_msg = String()
                touch_msg.data = f"touch_start:{element['id']}"
                self.touch_event_pub.publish(touch_msg)

    def handle_touch_move(self, pos):
        """
        Handle touch move event
        """
        x, y = pos
        self.ui_state['last_touch_position'] = pos

        # Update active joystick
        for element in self.interface_elements:
            if element['type'] == 'joystick' and element['active']:
                self.handle_joystick_touch(pos, element)

    def handle_touch_end(self, pos):
        """
        Handle touch end event
        """
        # Deactivate all elements
        for element in self.interface_elements:
            element['active'] = False

        self.ui_state['selected_element'] = None
        self.ui_state['gesture_in_progress'] = False

        # Publish touch end event
        touch_msg = String()
        touch_msg.data = f"touch_end"
        self.touch_event_pub.publish(touch_msg)

    def handle_joystick_touch(self, pos, joystick_element):
        """
        Handle joystick touch for navigation
        """
        center_x, center_y = joystick_element['center']
        dx = pos[0] - center_x
        dy = pos[1] - center_y

        # Calculate distance from center
        distance = math.sqrt(dx*dx + dy*dy)

        # Normalize to get direction
        if distance > 0:
            norm_dx = dx / distance
            norm_dy = dy / distance
        else:
            norm_dx = 0
            norm_dy = 0

        # Clamp distance to joystick radius
        clamped_distance = min(distance, joystick_element['radius'])
        normalized_distance = clamped_distance / joystick_element['radius']

        # Calculate joystick values (-1 to 1)
        joystick_x = norm_dx * normalized_distance
        joystick_y = norm_dy * normalized_distance

        joystick_element['value'] = (joystick_x, joystick_y)

        # Publish movement command
        cmd_vel = Twist()
        cmd_vel.linear.x = joystick_y * 0.5  # Scale for robot speed
        cmd_vel.angular.z = -joystick_x * 1.0  # Negative for natural rotation
        self.cmd_vel_pub.publish(cmd_vel)

    def execute_button_command(self, command):
        """
        Execute command from button press
        """
        cmd_vel = Twist()

        if command == 'forward':
            cmd_vel.linear.x = 0.3
        elif command == 'backward':
            cmd_vel.linear.x = -0.3
        elif command == 'left':
            cmd_vel.angular.z = 0.5
        elif command == 'right':
            cmd_vel.angular.z = -0.5
        elif command == 'stop':
            # Velocities are already zero
            pass
        elif command == 'help':
            self.speak_response("I can help with navigation and basic commands. What would you like to do?")
        elif command == 'reset':
            self.reset_interface()

        if command in ['forward', 'backward', 'left', 'right', 'stop']:
            self.cmd_vel_pub.publish(cmd_vel)

        # Update status display
        status_elem = next((elem for elem in self.interface_elements if elem['id'] == 'status_display'), None)
        if status_elem:
            status_elem['text'] = f"Command: {command}"
            status_elem['color'] = self.BLUE

    def joystick_callback(self, msg):
        """
        Handle joystick input from physical controller
        """
        # Map joystick axes to robot movement
        linear_x = msg.axes[1] * 0.5  # Left stick vertical
        angular_z = msg.axes[0] * 1.0  # Left stick horizontal

        cmd_vel = Twist()
        cmd_vel.linear.x = linear_x
        cmd_vel.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel)

        # Update interface to reflect joystick input
        for element in self.interface_elements:
            if element['type'] == 'joystick':
                element['value'] = (msg.axes[0], msg.axes[1])
                break

    def reset_interface(self):
        """
        Reset interface to default state
        """
        for element in self.interface_elements:
            if element['type'] == 'joystick':
                element['value'] = (0.0, 0.0)
            elif element['type'] == 'button':
                element['active'] = False
            elif element['type'] == 'display':
                element['text'] = 'Ready'
                element['color'] = self.GREEN

    def update_display(self):
        """
        Update the touch interface display
        """
        # Clear screen
        self.screen.fill(self.WHITE)

        # Draw interface elements
        for element in self.interface_elements:
            if element['type'] == 'joystick':
                # Draw joystick area
                pygame.draw.circle(self.screen, self.BLUE, element['center'], element['radius'], 2)

                # Draw joystick position
                center_x, center_y = element['center']
                joy_x, joy_y = element['value']
                pos_x = center_x + int(joy_x * element['radius'])
                pos_y = center_y + int(joy_y * element['radius'])

                pygame.draw.circle(self.screen, self.RED, (pos_x, pos_y), 15)
                pygame.draw.circle(self.screen, self.GREEN, (pos_x, pos_y), 15, 2)

            elif element['type'] == 'button':
                color = self.GREEN if element['active'] else self.BLUE
                pygame.draw.rect(self.screen, color, element['rect'], border_radius=10)

                # Draw button text
                font = pygame.font.Font(None, 24)
                text_surface = font.render(element['label'], True, self.WHITE)
                text_rect = text_surface.get_rect(center=element['rect'].center)
                self.screen.blit(text_surface, text_rect)

            elif element['type'] == 'display':
                pygame.draw.rect(self.screen, element['color'], element['rect'], border_radius=5)

                # Draw status text
                font = pygame.font.Font(None, 24)
                text_surface = font.render(element['text'], True, self.BLACK)
                text_rect = text_surface.get_rect(center=element['rect'].center)
                self.screen.blit(text_surface, text_rect)

        # Update display
        pygame.display.flip()

    def speak_response(self, response):
        """
        Publish response for TTS system
        """
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)
```

## Social Robotics Principles

### Theory of Mind Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import json
import time


class SocialRobotNode(Node):
    """
    Social robotics implementation with Theory of Mind
    """

    def __init__(self):
        super().__init__('social_robot_node')

        # Publishers
        self.social_behavior_pub = self.create_publisher(String, '/social_behavior', 10)
        self.explanation_pub = self.create_publisher(String, '/robot_explanation', 10)
        self.attention_pub = self.create_publisher(PoseStamped, '/robot_attention_target', 10)

        # Subscribers
        self.user_state_sub = self.create_subscription(
            String, '/user_state', self.user_state_callback, 10
        )
        self.robot_state_sub = self.create_subscription(
            Odometry, '/odom', self.robot_state_callback, 10
        )

        # Social state management
        self.users = {}  # Track multiple users
        self.social_context = {
            'current_activity': 'idle',
            'group_size': 0,
            'attention_focus': None,
            'social_rules': self.initialize_social_rules()
        }

        # Theory of Mind state
        self.belief_state = {
            'user_knowledge': {},  # What users know about robot capabilities
            'user_intentions': {},  # What users want to do
            'user_preferences': {},  # User preferences and habits
            'user_emotions': {}     # Detected user emotions
        }

        # Timer for social behavior updates
        self.social_timer = self.create_timer(1.0, self.update_social_behavior)

        self.get_logger().info('Social Robot Node initialized with Theory of Mind')

    def initialize_social_rules(self):
        """
        Initialize social behavior rules
        """
        return {
            'personal_space': 1.0,  # meters
            'greeting_distance': 2.0,  # meters
            'attention_shift_delay': 2.0,  # seconds
            'politeness_level': 0.8,  # 0-1 scale
            'engagement_threshold': 0.6  # minimum engagement score
        }

    def user_state_callback(self, msg):
        """
        Process user state information
        """
        try:
            user_data = json.loads(msg.data)
            user_id = user_data.get('user_id')

            if user_id:
                # Update user tracking
                self.users[user_id] = {
                    'position': user_data.get('position', {}),
                    'timestamp': time.time(),
                    'greeted': user_data.get('greeted', False),
                    'engagement_score': user_data.get('engagement_score', 0.0)
                }

                # Update belief state about this user
                self.update_user_beliefs(user_id, user_data)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid user state JSON')

    def robot_state_callback(self, msg):
        """
        Process robot state
        """
        self.robot_position = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z
        }

    def update_user_beliefs(self, user_id, user_data):
        """
        Update Theory of Mind beliefs about a user
        """
        # Update knowledge beliefs
        if 'capabilities_requested' in user_data:
            if user_id not in self.belief_state['user_knowledge']:
                self.belief_state['user_knowledge'][user_id] = set()
            self.belief_state['user_knowledge'][user_id].add(user_data['capabilities_requested'])

        # Update intention beliefs
        if 'intended_action' in user_data:
            self.belief_state['user_intentions'][user_id] = user_data['intended_action']

        # Update preference beliefs
        if 'interaction_style' in user_data:
            if user_id not in self.belief_state['user_preferences']:
                self.belief_state['user_preferences'][user_id] = {}
            self.belief_state['user_preferences'][user_id]['style'] = user_data['interaction_style']

        # Update emotion beliefs
        if 'emotion_detected' in user_data:
            self.belief_state['user_emotions'][user_id] = user_data['emotion_detected']

    def update_social_behavior(self):
        """
        Update social behavior based on Theory of Mind
        """
        # Determine current social context
        self.social_context['group_size'] = len(self.users)
        self.social_context['attention_focus'] = self.determine_attention_focus()

        # Generate appropriate social behavior
        behavior = self.generate_social_behavior()
        if behavior:
            behavior_msg = String()
            behavior_msg.data = json.dumps(behavior)
            self.social_behavior_pub.publish(behavior_msg)

    def determine_attention_focus(self):
        """
        Determine who to focus attention on based on Theory of Mind
        """
        if not self.users:
            return None

        best_candidate = None
        best_score = -1

        for user_id, user_info in self.users.items():
            # Calculate attention score based on multiple factors
            engagement_score = user_info.get('engagement_score', 0.0)
            time_since_interaction = time.time() - user_info.get('timestamp', time.time())

            # Prefer users who have engaged recently
            recency_factor = max(0, 1 - (time_since_interaction / 30))  # 30-second decay

            # Prefer users with higher engagement
            engagement_factor = engagement_score

            # Consider Theory of Mind beliefs
            intention_factor = 0
            if user_id in self.belief_state['user_intentions']:
                # Higher score if user has expressed an intention
                intention_factor = 0.5

            # Calculate total score
            total_score = (engagement_factor * 0.5 +
                          recency_factor * 0.3 +
                          intention_factor * 0.2)

            if total_score > best_score:
                best_score = total_score
                best_candidate = user_id

        return best_candidate

    def generate_social_behavior(self):
        """
        Generate appropriate social behavior based on context and beliefs
        """
        if not self.social_context['attention_focus']:
            # No one to focus on, maybe patrol or wait
            return {
                'behavior': 'idle_scan',
                'target': None,
                'explanation': 'Scanning for users to interact with'
            }

        focus_user = self.social_context['attention_focus']
        user_info = self.users[focus_user]

        # Check if we should greet this user
        if not user_info.get('greeted', False):
            return self.generate_greeting_behavior(focus_user)

        # Check user's apparent intention
        user_intention = self.belief_state['user_intentions'].get(focus_user)
        if user_intention:
            return self.generate_intention_response_behavior(focus_user, user_intention)

        # Check user's emotion
        user_emotion = self.belief_state['user_emotions'].get(focus_user)
        if user_emotion:
            return self.generate_emotion_response_behavior(focus_user, user_emotion)

        # Default engagement behavior
        return {
            'behavior': 'engage_user',
            'target': focus_user,
            'explanation': f'Engaging with user {focus_user} to maintain interaction'
        }

    def generate_greeting_behavior(self, user_id):
        """
        Generate greeting behavior for new user
        """
        # Update that user has been greeted
        self.users[user_id]['greeted'] = True

        # Calculate distance to user
        user_pos = self.users[user_id].get('position', {})
        if user_pos and self.robot_position:
            distance = self.calculate_distance(user_pos, self.robot_position)

            if distance > self.social_rules['greeting_distance']:
                # Too far, move closer
                return {
                    'behavior': 'approach_user',
                    'target': user_id,
                    'target_position': user_pos,
                    'explanation': f'Moving closer to greet user {user_id}'
                }
            else:
                # Appropriate distance, greet
                return {
                    'behavior': 'greet_user',
                    'target': user_id,
                    'explanation': f'Greeting user {user_id}',
                    'greeting_type': 'polite' if self.social_rules['politeness_level'] > 0.5 else 'casual'
                }

        return {
            'behavior': 'attempt_greeting',
            'target': user_id,
            'explanation': f'Attempting to greet user {user_id}'
        }

    def generate_intention_response_behavior(self, user_id, intention):
        """
        Generate behavior based on user's apparent intention
        """
        if intention == 'navigation_help':
            return {
                'behavior': 'offer_navigation_assistance',
                'target': user_id,
                'explanation': f'Offering navigation assistance to user {user_id} based on detected intention',
                'intention': intention
            }
        elif intention == 'information_request':
            return {
                'behavior': 'provide_information',
                'target': user_id,
                'explanation': f'Providing information to user {user_id} based on detected intention',
                'intention': intention
            }
        elif intention == 'follow_me':
            return {
                'behavior': 'follow_user',
                'target': user_id,
                'explanation': f'Following user {user_id} based on detected intention',
                'intention': intention
            }
        else:
            return {
                'behavior': 'acknowledge_intention',
                'target': user_id,
                'explanation': f'Acknowledging user {user_id} intention: {intention}',
                'intention': intention
            }

    def generate_emotion_response_behavior(self, user_id, emotion):
        """
        Generate behavior based on detected user emotion
        """
        if emotion == 'happy':
            return {
                'behavior': 'positive_engagement',
                'target': user_id,
                'explanation': f'Responding positively to user {user_id} happy emotion',
                'emotion': emotion
            }
        elif emotion == 'confused':
            return {
                'behavior': 'clarify_interaction',
                'target': user_id,
                'explanation': f'Clarifying interaction for user {user_id} showing confusion',
                'emotion': emotion
            }
        elif emotion == 'frustrated':
            return {
                'behavior': 'deescalate_interaction',
                'target': user_id,
                'explanation': f'De-escalating interaction with user {user_id} showing frustration',
                'emotion': emotion
            }
        elif emotion == 'sad':
            return {
                'behavior': 'empathetic_response',
                'target': user_id,
                'explanation': f'Providing empathetic response to user {user_id} sad emotion',
                'emotion': emotion
            }
        else:
            return {
                'behavior': 'monitor_emotion',
                'target': user_id,
                'explanation': f'Monitoring user {user_id} emotion: {emotion}',
                'emotion': emotion
            }

    def calculate_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions
        """
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)
        return (dx*dx + dy*dy + dz*dz)**0.5

    def explain_robot_behavior(self, explanation):
        """
        Explain robot's behavior to user
        """
        explanation_msg = String()
        explanation_msg.data = explanation
        self.explanation_pub.publish(explanation_msg)

    def focus_attention(self, target_user_id):
        """
        Focus robot's attention on a specific user
        """
        target_msg = PoseStamped()
        # In a real system, this would point to the user's position
        # For now, we'll just publish a placeholder
        target_msg.header.frame_id = "map"
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.pose.position.x = self.users[target_user_id]['position'].get('x', 0)
        target_msg.pose.position.y = self.users[target_user_id]['position'].get('y', 0)
        target_msg.pose.position.z = 1.5  # Head height
        target_msg.pose.orientation.w = 1.0

        self.attention_pub.publish(target_msg)
```

## Accessibility Features

### Inclusive Design Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
import numpy as np


class AccessibleHRIInterfaceNode(Node):
    """
    Accessible HRI interface supporting diverse user needs
    """

    def __init__(self):
        super().__init__('accessible_hri_interface')

        # Publishers
        self.accessible_command_pub = self.create_publisher(Twist, '/accessible_cmd_vel', 10)
        self.accessibility_status_pub = self.create_publisher(String, '/accessibility_status', 10)
        self.feedback_pub = self.create_publisher(String, '/hri_feedback', 10)

        # Subscribers
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joystick_callback, 10)
        self.voice_command_sub = self.create_subscription(
            String, '/voice_command', self.voice_command_callback, 10
        )

        # Accessibility settings
        self.accessibility_configs = {
            'motor_impairment': {
                'enabled': False,
                'control_sensitivity': 1.0,
                'alternative_inputs': ['voice', 'switch', 'eyetracking']
            },
            'visual_impairment': {
                'enabled': False,
                'audio_feedback': True,
                'haptic_feedback': True,
                'high_contrast_ui': True
            },
            'hearing_impairment': {
                'enabled': False,
                'visual_feedback': True,
                'haptic_feedback': True,
                'text_output': True
            },
            'cognitive_impairment': {
                'enabled': False,
                'simplified_interface': True,
                'extended_wait_times': True,
                'step_by_step_guidance': True
            }
        }

        # Current user profile
        self.current_user_profile = 'standard'  # Will be updated based on detection
        self.active_accessibility_modes = set()

        # Control parameters for different impairments
        self.control_sensitivity = 1.0
        self.response_delay = 0.0
        self.confirmation_required = False

        # Timer for accessibility status updates
        self.status_timer = self.create_timer(5.0, self.update_accessibility_status)

        self.get_logger().info('Accessible HRI Interface initialized')

    def joystick_callback(self, msg):
        """
        Process joystick input with accessibility considerations
        """
        # Apply sensitivity adjustments based on user profile
        adjusted_axes = [axis * self.control_sensitivity for axis in msg.axes]
        adjusted_buttons = msg.buttons

        # Generate movement command
        cmd_vel = Twist()

        # Apply accessibility modifications
        if self.accessibility_configs['motor_impairment']['enabled']:
            # Reduce sensitivity and smooth movements
            cmd_vel.linear.x = self.smooth_input(adjusted_axes[1]) * 0.3  # Left stick vertical
            cmd_vel.angular.z = self.smooth_input(-adjusted_axes[0]) * 0.5  # Left stick horizontal, inverted
        else:
            # Standard controls
            cmd_vel.linear.x = adjusted_axes[1] * 0.5
            cmd_vel.angular.z = -adjusted_axes[0] * 1.0

        # Apply response delay if needed
        if self.response_delay > 0:
            time.sleep(self.response_delay)

        # Require confirmation for safety-critical movements
        if self.confirmation_required:
            confirmed = self.request_confirmation(f"Move with linear:{cmd_vel.linear.x}, angular:{cmd_vel.angular.z}")
            if not confirmed:
                cmd_vel = Twist()  # Cancel movement

        self.accessible_cmd_vel_pub.publish(cmd_vel)

    def voice_command_callback(self, msg):
        """
        Process voice commands with accessibility considerations
        """
        command = msg.data.lower()

        # For hearing-impaired users, provide visual feedback
        if self.accessibility_configs['hearing_impairment']['enabled']:
            self.provide_visual_feedback(f"Heard command: {command}")

        # For cognitive impairment, provide confirmation
        if self.accessibility_configs['cognitive_impairment']['enabled']:
            self.provide_step_by_step_feedback(f"Processing command: {command}")

        # Parse and execute command
        cmd_vel = self.parse_voice_command(command)
        if cmd_vel:
            self.accessible_cmd_vel_pub.publish(cmd_vel)

    def parse_voice_command(self, command):
        """
        Parse voice command and return appropriate Twist command
        """
        cmd_vel = Twist()

        if any(word in command for word in ['forward', 'ahead', 'go', 'move']):
            cmd_vel.linear.x = 0.3
        elif any(word in command for word in ['backward', 'back', 'reverse']):
            cmd_vel.linear.x = -0.3
        elif any(word in command for word in ['left', 'turn left', 'rotate left']):
            cmd_vel.angular.z = 0.5
        elif any(word in command for word in ['right', 'turn right', 'rotate right']):
            cmd_vel.angular.z = -0.5
        elif any(word in command for word in ['stop', 'halt', 'pause']):
            # Keep as zero (stop)
            pass
        elif any(word in command for word in ['help', 'assist', 'what can you do']):
            self.provide_help()
            return None
        else:
            # For cognitive impairment, provide guidance
            if self.accessibility_configs['cognitive_impairment']['enabled']:
                self.provide_step_by_step_feedback(f"Unknown command: {command}. Try saying 'move forward' or 'turn left'.")
            return None

        return cmd_vel

    def smooth_input(self, raw_input, smoothing_factor=0.1):
        """
        Apply smoothing to input for users with motor impairments
        """
        # Simple exponential smoothing
        if not hasattr(self, 'smoothed_input'):
            self.smoothed_input = 0.0

        self.smoothed_input = (smoothing_factor * raw_input +
                              (1 - smoothing_factor) * self.smoothed_input)
        return self.smoothed_input

    def request_confirmation(self, action_description):
        """
        Request confirmation for safety-critical actions
        """
        # In a real system, this would use a confirmation interface
        # For now, we'll just log and return True
        self.get_logger().info(f"Confirmation requested for: {action_description}")

        # For cognitive impairment, provide clear feedback
        if self.accessibility_configs['cognitive_impairment']['enabled']:
            self.provide_step_by_step_feedback(f"About to execute: {action_description}. Press button to confirm.")

        return True  # In a real system, this would wait for user confirmation

    def provide_visual_feedback(self, message):
        """
        Provide visual feedback for hearing-impaired users
        """
        feedback_msg = String()
        feedback_msg.data = f"VISUAL FEEDBACK: {message}"
        self.feedback_pub.publish(feedback_msg)

    def provide_step_by_step_feedback(self, message):
        """
        Provide step-by-step feedback for cognitive impairment
        """
        feedback_msg = String()
        feedback_msg.data = f"STEP_GUIDANCE: {message}"
        self.feedback_pub.publish(feedback_msg)

    def provide_help(self):
        """
        Provide help information
        """
        help_text = (
            "I can help with basic movement commands. "
            "Try saying 'move forward', 'turn left', 'turn right', or 'stop'. "
            "For accessibility options, say 'accessibility help'."
        )

        feedback_msg = String()
        feedback_msg.data = help_text
        self.feedback_pub.publish(feedback_msg)

    def update_accessibility_status(self):
        """
        Update and publish accessibility status
        """
        status = {
            'active_modes': list(self.active_accessibility_modes),
            'control_sensitivity': self.control_sensitivity,
            'response_delay': self.response_delay,
            'confirmation_required': self.confirmation_required
        }

        status_msg = String()
        status_msg.data = json.dumps(status)
        self.accessibility_status_pub.publish(status_msg)

    def detect_user_accessibility_needs(self, user_data):
        """
        Detect user's accessibility needs from sensor data
        """
        # This would analyze user behavior, physical capabilities, etc.
        # For this example, we'll use a simplified detection
        detected_needs = set()

        # Example: If user moves joystick very slowly, they might have motor impairment
        if user_data.get('slow_movements', False):
            detected_needs.add('motor_impairment')
            self.accessibility_configs['motor_impairment']['enabled'] = True
            self.control_sensitivity = 0.5  # Reduce sensitivity

        # Example: If user requests audio feedback, they might have visual impairment
        if user_data.get('request_audio_feedback', False):
            detected_needs.add('visual_impairment')
            self.accessibility_configs['visual_impairment']['enabled'] = True

        # Update active modes
        self.active_accessibility_modes = detected_needs

        # Adjust interface based on detected needs
        self.adjust_interface_for_accessibility()

    def adjust_interface_for_accessibility(self):
        """
        Adjust interface based on active accessibility needs
        """
        # Adjust control parameters
        if 'motor_impairment' in self.active_accessibility_modes:
            self.control_sensitivity = 0.7
            self.response_delay = 0.1
        else:
            self.control_sensitivity = 1.0
            self.response_delay = 0.0

        # Set confirmation requirements
        if 'cognitive_impairment' in self.active_accessibility_modes:
            self.confirmation_required = True
        else:
            self.confirmation_required = False

        self.get_logger().info(f"Interface adjusted for accessibility modes: {self.active_accessibility_modes}")
```

## HRI Evaluation Metrics

### Interaction Quality Assessment

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import time
import statistics


class HRIEvaluationNode(Node):
    """
    Evaluate HRI system quality and effectiveness
    """

    def __init__(self):
        super().__init__('hri_evaluation_node')

        # Publishers
        self.engagement_score_pub = self.create_publisher(Float32, '/hri_engagement_score', 10)
        self.task_success_pub = self.create_publisher(Float32, '/task_success_rate', 10)
        self.evaluation_report_pub = self.create_publisher(String, '/hri_evaluation_report', 10)

        # Subscribers
        self.user_interaction_sub = self.create_subscription(
            String, '/user_interaction_log', self.interaction_callback, 10
        )
        self.task_completion_sub = self.create_subscription(
            String, '/task_completion_status', self.task_completion_callback, 10
        )
        self.social_behavior_sub = self.create_subscription(
            String, '/social_behavior', self.social_behavior_callback, 10
        )

        # Evaluation metrics
        self.interaction_history = []
        self.task_completions = []
        self.engagement_scores = []
        self.satisfaction_ratings = []

        # Evaluation timers
        self.metrics_timer = self.create_timer(10.0, self.calculate_metrics)
        self.report_timer = self.create_timer(60.0, self.generate_evaluation_report)

        self.get_logger().info('HRI Evaluation Node initialized')

    def interaction_callback(self, msg):
        """
        Log user interactions for evaluation
        """
        try:
            interaction_data = json.loads(msg.data)
            interaction_data['timestamp'] = time.time()
            self.interaction_history.append(interaction_data)

            # Calculate engagement based on interaction
            engagement = self.calculate_engagement_score(interaction_data)
            self.engagement_scores.append(engagement)

            # Publish current engagement score
            engagement_msg = Float32()
            engagement_msg.data = engagement
            self.engagement_score_pub.publish(engagement_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid interaction log JSON')

    def task_completion_callback(self, msg):
        """
        Log task completion events
        """
        try:
            task_data = json.loads(msg.data)
            task_data['timestamp'] = time.time()
            self.task_completions.append(task_data)

            # Calculate success rate
            success_rate = self.calculate_task_success_rate()

            # Publish success rate
            success_msg = Float32()
            success_msg.data = success_rate
            self.task_success_pub.publish(success_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid task completion JSON')

    def social_behavior_callback(self, msg):
        """
        Log social behavior events
        """
        try:
            behavior_data = json.loads(msg.data)
            behavior_data['timestamp'] = time.time()

            # Evaluate appropriateness of social behavior
            appropriateness_score = self.evaluate_social_appropriateness(behavior_data)

            # Log for evaluation
            self.log_social_evaluation(behavior_data, appropriateness_score)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid social behavior JSON')

    def calculate_engagement_score(self, interaction_data):
        """
        Calculate engagement score based on interaction
        """
        score = 0.0

        # Duration of interaction
        if 'duration' in interaction_data:
            duration_score = min(1.0, interaction_data['duration'] / 30.0)  # 30 seconds = max
            score += duration_score * 0.3

        # Type of interaction (more complex interactions get higher scores)
        interaction_type = interaction_data.get('type', 'passive')
        if interaction_type in ['command', 'request', 'conversation']:
            score += 0.4
        elif interaction_type in ['greeting', 'acknowledgment']:
            score += 0.2
        else:
            score += 0.1  # passive observation

        # Frequency of interactions
        recent_interactions = [
            i for i in self.interaction_history
            if time.time() - i.get('timestamp', 0) < 60  # Last minute
        ]
        frequency_score = min(1.0, len(recent_interactions) / 10)  # 10 interactions = max
        score += frequency_score * 0.3

        return min(1.0, score)

    def calculate_task_success_rate(self):
        """
        Calculate task success rate
        """
        if not self.task_completions:
            return 0.0

        successful_tasks = sum(1 for task in self.task_completions if task.get('success', False))
        return successful_tasks / len(self.task_completions)

    def evaluate_social_appropriateness(self, behavior_data):
        """
        Evaluate how appropriate the social behavior was
        """
        # This would compare behavior against social norms and context
        # For this example, we'll use a simplified evaluation
        behavior_type = behavior_data.get('behavior', 'unknown')
        context = behavior_data.get('context', {})

        # Define appropriateness rules
        appropriateness_rules = {
            'greet_user': 0.9,  # Generally appropriate
            'approach_user': 0.8,  # Appropriate if not violating personal space
            'ignore_user': 0.1,   # Generally inappropriate
            'follow_user': 0.3,   # Only appropriate in specific contexts
            'engage_user': 0.7    # Appropriate if user is receptive
        }

        base_score = appropriateness_rules.get(behavior_type, 0.5)

        # Adjust based on context
        if context.get('user_engagement', 0) < 0.3:
            # User not engaged, reduce score for engaging behaviors
            if behavior_type in ['greet_user', 'engage_user']:
                base_score *= 0.5

        return base_score

    def log_social_evaluation(self, behavior_data, appropriateness_score):
        """
        Log social behavior evaluation
        """
        evaluation_entry = {
            'behavior': behavior_data,
            'appropriateness_score': appropriateness_score,
            'timestamp': time.time()
        }
        # We could store these for trend analysis
        pass

    def calculate_metrics(self):
        """
        Calculate and publish HRI metrics
        """
        if not self.engagement_scores:
            return

        # Engagement metrics
        avg_engagement = statistics.mean(self.engagement_scores[-20:])  # Last 20 scores
        engagement_trend = self.calculate_trend(self.engagement_scores[-10:])

        # Task success metrics
        success_rate = self.calculate_task_success_rate()

        # Interaction quality metrics
        interaction_count = len(self.interaction_history[-10:])  # Last 10 interactions
        avg_interaction_duration = self.calculate_avg_interaction_duration()

        # Publish metrics
        engagement_msg = Float32()
        engagement_msg.data = avg_engagement
        self.engagement_score_pub.publish(engagement_msg)

        success_msg = Float32()
        success_msg.data = success_rate
        self.task_success_pub.publish(success_msg)

        self.get_logger().info(
            f'HRI Metrics - Engagement: {avg_engagement:.2f}, '
            f'Success Rate: {success_rate:.2f}, '
            f'Trend: {"increasing" if engagement_trend > 0 else "decreasing"}'
        )

    def calculate_trend(self, scores):
        """
        Calculate trend of engagement scores
        """
        if len(scores) < 2:
            return 0

        # Simple linear regression slope
        n = len(scores)
        x = list(range(n))
        y = scores

        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Calculate slope
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x)**2 for i in range(n))

        if denominator == 0:
            return 0

        return numerator / denominator

    def calculate_avg_interaction_duration(self):
        """
        Calculate average duration of interactions
        """
        recent_interactions = [
            i for i in self.interaction_history
            if time.time() - i.get('timestamp', 0) < 300  # Last 5 minutes
        ]

        durations = [i.get('duration', 0) for i in recent_interactions if 'duration' in i]
        return statistics.mean(durations) if durations else 0

    def generate_evaluation_report(self):
        """
        Generate comprehensive HRI evaluation report
        """
        if not self.interaction_history:
            return

        report = {
            'timestamp': time.time(),
            'period': 'last_minute',  # Could be configurable
            'engagement_metrics': {
                'average_score': statistics.mean(self.engagement_scores[-20:]) if self.engagement_scores else 0,
                'median_score': statistics.median(self.engagement_scores[-20:]) if self.engagement_scores else 0,
                'trend': self.calculate_trend(self.engagement_scores[-10:]) if len(self.engagement_scores) >= 10 else 0
            },
            'task_metrics': {
                'success_rate': self.calculate_task_success_rate(),
                'total_attempts': len(self.task_completions),
                'successful_completions': sum(1 for task in self.task_completions if task.get('success', False))
            },
            'interaction_metrics': {
                'total_interactions': len(self.interaction_history[-60:]),  # Last minute
                'interaction_types': self.get_interaction_type_distribution(),
                'average_duration': self.calculate_avg_interaction_duration()
            },
            'social_metrics': {
                'average_appropriateness': self.calculate_avg_social_appropriateness()
            }
        }

        # Publish report
        report_msg = String()
        report_msg.data = json.dumps(report, indent=2)
        self.evaluation_report_pub.publish(report_msg)

        self.get_logger().info(f'HRI Evaluation Report: {report_msg.data}')

    def get_interaction_type_distribution(self):
        """
        Get distribution of interaction types
        """
        interactions = self.interaction_history[-60:]  # Last minute
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction.get('type', 'unknown')
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        return type_counts

    def calculate_avg_social_appropriateness(self):
        """
        Calculate average social appropriateness score
        """
        # In a real implementation, we would have stored social evaluations
        # For this example, we'll return a placeholder
        return 0.75  # Placeholder average
```

## Troubleshooting Common Issues

### Connection Problems

- If ROS connections fail, verify that Isaac Sim and ROS nodes are on the same network
- Check that the ROS bridge is properly configured and running
- Verify that topic names match between Isaac Sim and ROS nodes
- Ensure that message types are compatible between systems

### Performance Issues

- For slow simulation, reduce the number of active sensors or lower their update rates
- If graphics are lagging, reduce rendering quality or disable advanced effects
- For high CPU usage, optimize robot control loops and reduce unnecessary computations
- If GPU memory is exhausted, reduce texture resolutions or model complexities

### Interaction Problems

- If speech recognition fails frequently, check microphone quality and ambient noise
- For poor gesture recognition, ensure proper lighting and camera positioning
- If touch interface is unresponsive, verify touch screen calibration
- For inappropriate social behavior, review and adjust Theory of Mind parameters

### Accessibility Issues

- If accessibility features don't activate, verify user profile detection
- For unclear audio feedback, improve speech synthesis quality
- If visual feedback is insufficient, enhance contrast and size
- For cognitive overload, simplify interface and reduce options

## Cross-References

For related concepts, see:
- [ROS 2 Fundamentals](../ros2-core/index.md) - For ROS communication
- [Sensors](../sensors/index.md) - For sensor integration in HRI
- [Conversational Robotics](../conversational-robotics/index.md) - For speech interaction
- [Humanoid Kinematics](../humanoid-kinematics/index.md) - For robot movement
- [Unity Visualization](../unity/index.md) - For alternative visualization

## Summary

Human-Robot Interaction is a critical component of modern robotics systems, requiring careful consideration of communication modalities, social behavior, accessibility, and evaluation. Successful HRI implementations combine multiple interaction methods (speech, gesture, touch) with appropriate social behaviors and inclusive design principles. The evaluation of HRI systems is equally important, requiring metrics for engagement, task success, and social appropriateness. When properly implemented, HRI systems create more natural and effective interactions between humans and robots, enhancing both user experience and task performance.