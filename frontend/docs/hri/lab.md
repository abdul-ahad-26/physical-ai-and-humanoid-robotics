---
title: Human-Robot Interaction Lab
sidebar_position: 15
---

# Human-Robot Interaction Lab

## Lab Objectives

In this lab, you will:
1. Implement social robotics principles for natural human-robot interaction
2. Design and test multimodal communication systems
3. Build trust models for human-robot interaction
4. Create context-aware behavior adaptation systems
5. Evaluate HRI systems using objective and subjective metrics

## Prerequisites

- Python 3.8+ installed
- NumPy, SciPy, Matplotlib, PyTorch, NLTK libraries
- Basic understanding of robotics and AI
- ROS 2 (optional for hardware integration)

## Exercise 1: Social Robotics Implementation

In this exercise, you'll implement social robotics principles including Theory of Mind and social norm compliance.

### Step 1: Create a Theory of Mind System

```python
import numpy as np
import matplotlib.pyplot as plt
import random

class RobotTheoryOfMind:
    """
    Implements a basic Theory of Mind for a humanoid robot
    to understand and predict human mental states
    """

    def __init__(self):
        self.human_beliefs = {}
        self.human_intents = {}
        self.human_emotions = {}
        self.social_context = {}

    def update_human_state(self, human_id, observed_behavior, context):
        """
        Update the robot's understanding of a human's mental state
        based on observed behavior and context
        """
        # Update beliefs based on observations
        if 'looking_at' in observed_behavior:
            target = observed_behavior['looking_at']
            self.human_beliefs[human_id] = self.human_beliefs.get(human_id, {})
            self.human_beliefs[human_id][target] = {
                'attention': True,
                'interest_level': self.estimate_interest(target, context)
            }

        # Update intents based on observed actions
        if 'action' in observed_behavior:
            action = observed_behavior['action']
            self.human_intents[human_id] = self.human_intents.get(human_id, [])
            self.human_intents[human_id].append({
                'action': action,
                'goal': self.infer_goal(action, context),
                'confidence': self.estimate_confidence(action, context)
            })

        # Update emotional state
        self.human_emotions[human_id] = self.estimate_emotion(
            observed_behavior, context
        )

    def predict_human_response(self, human_id, robot_action):
        """
        Predict how a human will respond to a robot's action
        based on their current mental state
        """
        beliefs = self.human_beliefs.get(hhuman_id, {})
        intents = self.human_intents.get(human_id, [])
        emotions = self.human_emotions.get(human_id, {})

        # Predict response based on mental state
        predicted_response = {
            'action_probability': self.calculate_action_probability(
                robot_action, beliefs, intents, emotions
            ),
            'emotional_response': self.predict_emotional_response(
                robot_action, emotions
            ),
            'acceptance_level': self.estimate_acceptance(
                robot_action, beliefs, emotions
            )
        }

        return predicted_response

    def estimate_interest(self, target, context):
        """Estimate human interest level in a target object or action"""
        # Simplified interest estimation
        if target in context.get('conversation_topic', []):
            return 0.8  # High interest
        elif target is context.get('focus_object'):
            return 0.6  # Medium-high interest
        else:
            return 0.3  # Low interest

    def infer_goal(self, action, context):
        """Infer the likely goal behind a human action"""
        # Simplified goal inference
        goal_map = {
            'reaching': 'grasping_object',
            'pointing': 'directing_attention',
            'smiling': 'expressing_happiness',
            'frowning': 'expressing_concern',
            'stepping_back': 'creating_distance'
        }
        return goal_map.get(action, 'unknown')

    def estimate_emotion(self, behavior, context):
        """Estimate human emotional state from behavior and context"""
        emotions = {
            'valence': 0.0,  # -1 (negative) to 1 (positive)
            'arousal': 0.0,  # 0 (calm) to 1 (excited)
            'dominance': 0.0  # -1 (submissive) to 1 (dominant)
        }

        # Update based on facial expressions
        if 'facial_expression' in behavior:
            expr = behavior['facial_expression']
            if expr == 'smiling':
                emotions['valence'] = 0.7
                emotions['arousal'] = 0.3
            elif expr == 'frowning':
                emotions['valence'] = -0.5
                emotions['arousal'] = 0.4

        # Update based on body language
        if 'body_posture' in behavior:
            posture = behavior['body_posture']
            if posture == 'open':
                emotions['dominance'] = 0.2
            elif posture == 'closed':
                emotions['dominance'] = -0.3

        return emotions

    def calculate_action_probability(self, robot_action, beliefs, intents, emotions):
        """Calculate the probability of a human action given robot action"""
        # Simplified probability calculation
        base_probability = 0.5

        # Adjust based on emotional state
        if emotions.get('valence', 0) > 0:
            base_probability *= 1.2  # More likely if human is positive
        else:
            base_probability *= 0.8  # Less likely if human is negative

        # Adjust based on context
        if robot_action in ['greeting', 'acknowledging']:
            base_probability *= 1.5  # Social actions more likely to get response

        return min(1.0, base_probability)

    def predict_emotional_response(self, robot_action, emotions):
        """Predict human's emotional response to robot action"""
        # Simplified emotional response prediction
        if robot_action in ['complimenting', 'helping', 'greeting']:
            return {'valence': 0.6, 'arousal': 0.3}
        elif robot_action in ['interrupting', 'invading_space']:
            return {'valence': -0.4, 'arousal': 0.7}
        else:
            return {'valence': 0.1, 'arousal': 0.1}

    def estimate_acceptance(self, robot_action, beliefs, emotions):
        """Estimate human acceptance of robot action"""
        # Combine emotional and belief factors
        emotional_acceptance = emotions.get('valence', 0)
        belief_acceptance = 0.5  # Default neutral

        # Adjust based on beliefs about robot's intentions
        if beliefs.get('trustworthy', True):
            belief_acceptance = 0.8
        else:
            belief_acceptance = 0.2

        # Weighted combination
        acceptance = 0.7 * emotional_acceptance + 0.3 * belief_acceptance
        return max(-1.0, min(1.0, acceptance))  # Clamp to [-1, 1]

# Test the Theory of Mind system
tom = RobotTheoryOfMind()

# Update understanding based on observation
observed_behavior = {
    'facial_expression': 'smiling',
    'body_posture': 'open',
    'action': 'nodding'
}
context = {
    'conversation_topic': ['robot_help', 'kitchen_assistance'],
    'focus_object': 'kitchen_counter'
}

tom.update_human_state('user1', observed_behavior, context)

# Predict response to robot action
prediction = tom.predict_human_response('user1', 'offering_help')
print(f"Predicted response: {prediction}")
```

### Step 2: Implement Social Norms Engine

```python
class SocialNormsEngine:
    """
    Manages social norms and conventions for human-robot interaction
    """

    def __init__(self):
        self.norms = self._initialize_norms()
        self.contextual_rules = self._initialize_contextual_rules()

    def _initialize_norms(self):
        """Initialize basic social norms"""
        return {
            'personal_space': {
                'intimate': 0.45,    # 0-45cm
                'personal': 1.2,     # 45cm-1.2m
                'social': 3.6,       # 1.2-3.6m
                'public': 7.6        # 3.6-7.6m
            },
            'greeting_norms': {
                'wave': {'distance': 'personal', 'culture': 'universal'},
                'bow': {'distance': 'personal', 'culture': 'asian'},
                'handshake': {'distance': 'personal', 'culture': 'western'}
            },
            'conversation_norms': {
                'turn_taking': True,
                'eye_contact': 0.3,  # 30% of time
                'personal_distance': 1.0  # 1 meter for conversation
            }
        }

    def _initialize_contextual_rules(self):
        """Initialize context-dependent social rules"""
        return {
            'home_environment': {
                'formality': 'casual',
                'volume_level': 'normal',
                'touch_norms': {'handshake': True, 'hug': 'conditional'}
            },
            'workplace_environment': {
                'formality': 'professional',
                'volume_level': 'moderate',
                'touch_norms': {'handshake': True, 'hug': False}
            },
            'public_space': {
                'formality': 'polite',
                'volume_level': 'low',
                'touch_norms': {'handshake': 'conditional', 'hug': False}
            }
        }

    def check_norm_compliance(self, action, context):
        """Check if an action complies with social norms in the given context"""
        violations = []

        # Check personal space violation
        if action.get('type') == 'approach' and 'distance' in action:
            min_distance = self.norms['personal_space']['personal']
            if action['distance'] < min_distance:
                violations.append({
                    'type': 'personal_space_violation',
                    'severity': 'medium',
                    'suggestion': f'Maintain at least {min_distance}m distance'
                })

        # Check greeting appropriateness
        if action.get('type') == 'greeting':
            context_type = context.get('environment', 'neutral')
            if context_type == 'workplace' and action.get('greeting_type') == 'hug':
                violations.append({
                    'type': 'inappropriate_greeting',
                    'severity': 'high',
                    'suggestion': 'Use handshake instead of hug in workplace'
                })

        # Check conversation norms
        if action.get('type') == 'speak' and action.get('interrupting', False):
            violations.append({
                'type': 'conversation_norm_violation',
                'severity': 'medium',
                'suggestion': 'Wait for turn in conversation'
            })

        return violations

    def suggest_appropriate_action(self, intended_action, context):
        """Suggest a more appropriate action based on social norms"""
        violations = self.check_norm_compliance(intended_action, context)

        if not violations:
            return intended_action  # Action is appropriate

        # Generate alternative based on violations
        alternative = intended_action.copy()

        for violation in violations:
            if violation['type'] == 'personal_space_violation':
                alternative['distance'] = self.norms['personal_space']['personal']
            elif violation['type'] == 'inappropriate_greeting':
                alternative['greeting_type'] = 'handshake'
            elif violation['type'] == 'conversation_norm_violation':
                alternative['interrupting'] = False

        return alternative

    def adapt_to_cultural_norms(self, user_profile):
        """Adapt social behavior based on user's cultural background"""
        cultural_adaptations = {
            'japanese': {
                'bow_angle': 15,  # degrees
                'personal_distance_multiplier': 1.2,
                'eye_contact_reduction': 0.2
            },
            'middle_eastern': {
                'greeting_handshake': 'right_hand_only',
                'personal_distance_multiplier': 0.8,
                'formality_increase': 0.3
            },
            'latin_american': {
                'personal_distance_multiplier': 0.7,
                'touch_acceptance': True,
                'formality_decrease': 0.2
            }
        }

        culture = user_profile.get('cultural_background', 'default')
        return cultural_adaptations.get(culture, {})

# Test the Social Norms Engine
norms_engine = SocialNormsEngine()

# Check if an action violates social norms
action = {'type': 'approach', 'distance': 0.3}
context = {'environment': 'workplace'}
violations = norms_engine.check_norm_compliance(action, context)

print(f"Violations found: {violations}")

# Get cultural adaptations
user_profile = {'cultural_background': 'japanese'}
adaptations = norms_engine.adapt_to_cultural_norms(user_profile)
print(f"Cultural adaptations: {adaptations}")
```

## Exercise 2: Multimodal Communication System

In this exercise, you'll implement a multimodal communication system.

### Step 1: Create Multimodal Communication Framework

```python
class MultimodalCommunication:
    """
    Framework for managing multiple communication modalities in HRI
    """

    def __init__(self):
        self.modalities = {
            'speech': SpeechCommunication(),
            'gesture': GestureCommunication(),
            'facial_expression': FacialExpressionCommunication(),
            'gaze': GazeCommunication(),
            'proxemics': ProxemicsCommunication()
        }
        self.fusion_engine = CommunicationFusionEngine()

    def generate_response(self, input_modalities, context):
        """
        Generate a multimodal response based on input and context
        """
        responses = {}

        # Process each input modality
        for modality, data in input_modalities.items():
            if modality in self.modalities:
                responses[modality] = self.modalities[modality].interpret(data, context)

        # Fuse interpretations into coherent understanding
        fused_interpretation = self.fusion_engine.fuse(responses, context)

        # Generate appropriate response modalities
        output_modalities = {}
        for modality, processor in self.modalities.items():
            output_modalities[modality] = processor.generate_response(
                fused_interpretation, context
            )

        return output_modalities

    def synchronize_modalities(self, output_modalities, timing_constraints):
        """
        Synchronize multiple modalities for natural, coordinated behavior
        """
        synchronized_output = {}

        for modality, output in output_modalities.items():
            synchronized_output[modality] = self._apply_timing(
                output, timing_constraints.get(modality, {})
            )

        return synchronized_output

    def _apply_timing(self, output, constraints):
        """Apply timing constraints to modality output"""
        # Add timing information to output
        output_with_timing = output.copy()
        output_with_timing['timing'] = constraints.get('timing', {})
        return output_with_timing

class SpeechCommunication:
    """Handles speech-based communication"""

    def __init__(self):
        self.dialogue_manager = DialogueManager()

    def interpret(self, speech_data, context):
        """Interpret speech input"""
        # In practice, this would use ASR and NLU
        interpretation = {
            'text': speech_data.get('text', ''),
            'intent': self._classify_intent(speech_data.get('text', '')),
            'sentiment': self._analyze_sentiment(speech_data.get('text', '')),
            'confidence': speech_data.get('confidence', 0.8)
        }
        return interpretation

    def generate_response(self, interpretation, context):
        """Generate speech response"""
        response_text = self.dialogue_manager.generate_response(
            interpretation, context
        )
        return {
            'text': response_text,
            'prosody': self._adjust_prosody(interpretation, context),
            'speech_act': self._determine_speech_act(response_text)
        }

    def _classify_intent(self, text):
        """Classify the intent of spoken text"""
        if any(word in text.lower() for word in ['hello', 'hi', 'hey']):
            return 'greeting'
        elif any(word in text.lower() for word in ['help', 'assist', 'can you']):
            return 'request'
        elif any(word in text.lower() for word in ['thank', 'thanks', 'appreciate']):
            return 'appreciation'
        else:
            return 'unknown'

    def _analyze_sentiment(self, text):
        """Analyze sentiment of spoken text"""
        positive_words = ['good', 'great', 'excellent', 'love', 'like', 'happy', 'thank']
        negative_words = ['bad', 'terrible', 'hate', 'angry', 'frustrated', 'dislike']

        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())

        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'

    def _adjust_prosody(self, interpretation, context):
        """Adjust speech prosody based on interpretation and context"""
        prosody = {
            'pitch': 1.0,  # Normal pitch
            'rate': 1.0,   # Normal rate
            'volume': 1.0  # Normal volume
        }

        # Adjust based on sentiment
        if interpretation.get('sentiment') == 'positive':
            prosody['pitch'] = 1.1  # Slightly higher for positive emotions
            prosody['rate'] = 1.05
        elif interpretation.get('sentiment') == 'negative':
            prosody['pitch'] = 0.9  # Slightly lower for negative emotions
            prosody['rate'] = 0.95

        # Adjust based on context
        if context.get('environment') == 'quiet':
            prosody['volume'] = 0.7  # Lower volume in quiet environments

        return prosody

    def _determine_speech_act(self, text):
        """Determine the speech act of the generated text"""
        if text.strip().endswith('?'):
            return 'question'
        elif any(word in text.lower() for word in ['please', 'could you', 'would you']):
            return 'request'
        elif any(word in text.lower() for word in ['thank', 'thanks']):
            return 'appreciation'
        else:
            return 'statement'

class GestureCommunication:
    """Handles gesture-based communication"""

    def __init__(self):
        self.gesture_repertoire = self._initialize_gestures()

    def _initialize_gestures(self):
        """Initialize the robot's gesture repertoire"""
        return {
            'pointing': {
                'type': 'deictic',
                'purpose': 'directing_attention',
                'variants': ['index_finger', 'open_hand']
            },
            'waving': {
                'type': 'emblem',
                'purpose': 'greeting',
                'variants': ['hello', 'goodbye', 'attention']
            },
            'beckoning': {
                'type': 'illustrator',
                'purpose': 'invitation',
                'variants': ['come_here', 'follow_me']
            },
            'shrugging': {
                'type': 'adaptor',
                'purpose': 'uncertainty',
                'variants': ['dunno', 'confused']
            }
        }

    def interpret(self, gesture_data, context):
        """Interpret gesture input"""
        gesture_type = gesture_data.get('type', 'unknown')

        interpretation = {
            'gesture_type': gesture_type,
            'meaning': self.gesture_repertoire.get(gesture_type, {}).get('purpose', 'unknown'),
            'confidence': gesture_data.get('confidence', 0.8),
            'intensity': gesture_data.get('intensity', 0.5)
        }
        return interpretation

    def generate_response(self, interpretation, context):
        """Generate appropriate gesture response"""
        # Determine appropriate gesture based on input and context
        if interpretation.get('speech_act') == 'greeting':
            gesture = 'waving'
        elif interpretation.get('intent') == 'request':
            gesture = 'beckoning'
        elif interpretation.get('sentiment') == 'negative':
            gesture = 'shrugging'
        else:
            gesture = 'neutral_posture'

        return {
            'gesture_type': gesture,
            'amplitude': self._determine_amplitude(interpretation, context),
            'timing': self._determine_timing(interpretation, context)
        }

    def _determine_amplitude(self, interpretation, context):
        """Determine the amplitude of the gesture"""
        base_amplitude = 0.5

        # Increase amplitude for emphasis
        if interpretation.get('confidence', 0.5) > 0.8:
            base_amplitude *= 1.2

        # Adjust based on distance
        distance = context.get('distance_to_human', 1.0)
        base_amplitude *= min(1.5, distance)  # Larger gestures for distant humans

        return min(1.0, base_amplitude)

    def _determine_timing(self, interpretation, context):
        """Determine the timing of the gesture"""
        return {
            'onset_delay': 0.2,  # Delay before gesture starts
            'duration': 1.0,     # Duration of gesture
            'offset_delay': 0.1  # Delay after gesture ends
        }

class FacialExpressionCommunication:
    """Handles facial expression communication"""

    def __init__(self):
        self.expression_repertoire = self._initialize_expressions()

    def _initialize_expressions(self):
        """Initialize the robot's facial expression repertoire"""
        return {
            'happy': {'valence': 0.8, 'arousal': 0.3, 'facial_action': [12, 25, 26]},  # Smile
            'sad': {'valence': -0.6, 'arousal': 0.2, 'facial_action': [1, 4, 15]},     # Frown
            'surprised': {'valence': 0.2, 'arousal': 0.9, 'facial_action': [1, 2, 5]}, # Raised eyebrows
            'angry': {'valence': -0.8, 'arousal': 0.8, 'facial_action': [4, 5, 23]},   # Scowl
            'neutral': {'valence': 0.0, 'arousal': 0.0, 'facial_action': []},          # Neutral
            'attentive': {'valence': 0.3, 'arousal': 0.5, 'facial_action': [1, 2, 26]} # Focused look
        }

    def interpret(self, facial_data, context):
        """Interpret facial expression input"""
        expression = facial_data.get('expression', 'neutral')

        interpretation = {
            'expression': expression,
            'valence': self.expression_repertoire.get(expression, {}).get('valence', 0.0),
            'arousal': self.expression_repertoire.get(expression, {}).get('arousal', 0.0),
            'confidence': facial_data.get('confidence', 0.8)
        }
        return interpretation

    def generate_response(self, interpretation, context):
        """Generate appropriate facial expression response"""
        # Determine response expression based on input and context
        input_valence = interpretation.get('valence', 0.0)
        input_arousal = interpretation.get('arousal', 0.0)

        if input_valence > 0.5:  # Happy input
            response_expr = 'happy'
        elif input_valence < -0.5:  # Sad input
            response_expr = 'sad'
        elif input_arousal > 0.7:  # High arousal
            response_expr = 'surprised'
        else:
            response_expr = 'attentive'

        return {
            'expression': response_expr,
            'intensity': self._determine_intensity(interpretation, context),
            'duration': self._determine_duration(interpretation, context)
        }

    def _determine_intensity(self, interpretation, context):
        """Determine the intensity of the facial expression"""
        base_intensity = 0.6

        # Match intensity to input
        input_intensity = abs(interpretation.get('valence', 0.0)) + abs(interpretation.get('arousal', 0.0))
        base_intensity = min(1.0, base_intensity + input_intensity * 0.3)

        return base_intensity

    def _determine_duration(self, interpretation, context):
        """Determine the duration of the facial expression"""
        base_duration = 2.0  # seconds

        # Hold expressions longer for significant emotional content
        emotional_significance = abs(interpretation.get('valence', 0.0))
        if emotional_significance > 0.7:
            base_duration *= 1.5

        return base_duration

class DialogueManager:
    """Manages conversational flow and response generation"""

    def __init__(self):
        self.conversation_history = []
        self.user_model = UserModel()
        self.response_templates = self._initialize_templates()

    def _initialize_templates(self):
        """Initialize response templates for different situations"""
        return {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Good day! How may I help you?"
            ],
            'request': [
                "I can help you with that.",
                "Sure, I'll assist you with {request}.",
                "I understand you need help with {request}."
            ],
            'appreciation': [
                "You're welcome!",
                "I'm glad I could help!",
                "Thank you for your kind words!"
            ],
            'uncertainty': [
                "I'm not sure I understand. Could you please clarify?",
                "Could you repeat that?",
                "I didn't catch that clearly."
            ]
        }

    def generate_response(self, interpretation, context):
        """Generate an appropriate verbal response"""
        intent = interpretation.get('intent', 'unknown')

        if intent in self.response_templates:
            template = self.response_templates[intent][0]  # Simple selection
            if '{request}' in template:
                request_info = interpretation.get('text', 'something')
                response = template.format(request=request_info)
            else:
                response = template
        else:
            response = "I understand. How else may I assist you?"

        # Personalize based on user model
        user_name = self.user_model.get_preferred_name(context.get('user_id'))
        if user_name:
            response = f"{user_name}, {response.lower()}" if not response.startswith(user_name) else response

        # Add to conversation history
        self.conversation_history.append({
            'user_input': interpretation,
            'robot_response': response,
            'timestamp': context.get('timestamp')
        })

        return response

class UserModel:
    """Maintains information about users for personalized interaction"""

    def __init__(self):
        self.users = {}

    def get_preferred_name(self, user_id):
        """Get the preferred name for a user"""
        user = self.users.get(user_id, {})
        return user.get('preferred_name')

    def update_user_info(self, user_id, info):
        """Update information about a user"""
        if user_id not in self.users:
            self.users[user_id] = {}
        self.users[user_id].update(info)

class CommunicationFusionEngine:
    """Fuses information from multiple modalities"""

    def __init__(self):
        self.confidence_weights = {
            'speech': 0.6,
            'gesture': 0.3,
            'facial_expression': 0.4,
            'gaze': 0.2,
            'proxemics': 0.1
        }

    def fuse(self, modality_interpretations, context):
        """Fuse interpretations from multiple modalities"""
        # Weighted combination of modalities
        fused_interpretation = {
            'intent': self._fuse_intent(modality_interpretations),
            'sentiment': self._fuse_sentiment(modality_interpretations),
            'attention': self._fuse_attention(modality_interpretations),
            'urgency': self._fuse_urgency(modality_interpretations)
        }

        return fused_interpretation

    def _fuse_intent(self, interpretations):
        """Fuse intent information from multiple modalities"""
        intent_confidence = {}

        for modality, interpretation in interpretations.items():
            weight = self.confidence_weights.get(modality, 0.1)
            intent = interpretation.get('intent', 'unknown')

            if intent != 'unknown':
                intent_confidence[intent] = intent_confidence.get(intent, 0) + \
                                          weight * interpretation.get('confidence', 1.0)

        # Return the intent with highest weighted confidence
        if intent_confidence:
            return max(intent_confidence, key=intent_confidence.get)
        else:
            return 'unknown'

    def _fuse_sentiment(self, interpretations):
        """Fuse sentiment information from multiple modalities"""
        weighted_sentiment = 0
        total_weight = 0

        for modality, interpretation in interpretations.items():
            weight = self.confidence_weights.get(modality, 0.1) * interpretation.get('confidence', 1.0)

            # Convert sentiment to numerical value
            sentiment_val = self._sentiment_to_value(interpretation.get('sentiment', 'neutral'))
            weighted_sentiment += weight * sentiment_val
            total_weight += weight

        if total_weight > 0:
            avg_sentiment = weighted_sentiment / total_weight
            return self._value_to_sentiment(avg_sentiment)
        else:
            return 'neutral'

    def _sentiment_to_value(self, sentiment):
        """Convert sentiment string to numerical value"""
        sentiment_map = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
        return sentiment_map.get(sentiment, 0.0)

    def _value_to_sentiment(self, value):
        """Convert numerical value back to sentiment string"""
        if value > 0.3:
            return 'positive'
        elif value < -0.3:
            return 'negative'
        else:
            return 'neutral'

    def _fuse_attention(self, interpretations):
        """Fuse attention information"""
        # Simplified attention fusion
        speech_attention = interpretations.get('speech', {}).get('attention', 0.5)
        gaze_attention = interpretations.get('gaze', {}).get('attention', 0.5)
        return (speech_attention + gaze_attention) / 2

    def _fuse_urgency(self, interpretations):
        """Fuse urgency information"""
        max_urgency = 0
        for interpretation in interpretations.values():
            urgency = interpretation.get('urgency', 0.0)
            max_urgency = max(max_urgency, urgency)
        return max_urgency

# Test the multimodal communication system
comm_system = MultimodalCommunication()

# Simulate input from multiple modalities
input_modalities = {
    'speech': {
        'text': 'Hello, can you help me with the dishes?',
        'confidence': 0.9
    },
    'gesture': {
        'type': 'pointing',
        'target': 'kitchen',
        'confidence': 0.8
    },
    'facial_expression': {
        'expression': 'tired',
        'confidence': 0.7
    }
}

context = {
    'user_id': 'user1',
    'environment': 'home',
    'time_of_day': 'evening',
    'distance_to_human': 1.5
}

# Generate multimodal response
response = comm_system.generate_response(input_modalities, context)
print(f"Multimodal response: {response}")

# Synchronize modalities
synchronized_response = comm_system.synchronize_modalities(
    response,
    {
        'speech': {'timing': {'delay': 0.5}},
        'gesture': {'timing': {'delay': 0.7}},
        'facial_expression': {'timing': {'delay': 0.0}}
    }
)
print(f"Synchronized response: {synchronized_response}")
```

## Exercise 3: Trust Modeling System

In this exercise, you'll implement a trust modeling system for HRI.

### Step 1: Create Trust Model

```python
class TrustModel:
    """
    Models and manages trust in human-robot interaction
    """

    def __init__(self):
        self.trust_levels = {}
        self.trust_factors = {
            'competence': 0.4,
            'reliability': 0.3,
            'benevolence': 0.2,
            'integrity': 0.1
        }
        self.trust_history = {}

    def initialize_trust(self, user_id):
        """Initialize trust level for a new user"""
        self.trust_levels[user_id] = {
            'overall': 0.5,  # Start with neutral trust
            'competence': 0.5,
            'reliability': 0.5,
            'benevolence': 0.5,
            'integrity': 0.5
        }
        self.trust_history[user_id] = []

    def update_trust(self, user_id, interaction_outcome, context=None):
        """
        Update trust based on interaction outcome

        Args:
            user_id: ID of the user
            interaction_outcome: Dictionary with results of interaction
            context: Context of the interaction
        """
        if user_id not in self.trust_levels:
            self.initialize_trust(user_id)

        # Calculate trust updates for each factor
        trust_updates = self._calculate_trust_updates(interaction_outcome, context)

        # Update trust levels
        for factor, update in trust_updates.items():
            current_trust = self.trust_levels[user_id][factor]
            new_trust = self._update_with_forgetting(current_trust, update)
            self.trust_levels[user_id][factor] = self._clamp_trust(new_trust)

        # Calculate overall trust
        overall_trust = self._calculate_overall_trust(user_id)
        self.trust_levels[user_id]['overall'] = overall_trust

        # Record in history
        self.trust_history[user_id].append({
            'timestamp': context.get('timestamp') if context else 0,
            'outcome': interaction_outcome,
            'trust_updates': trust_updates,
            'new_trust_levels': self.trust_levels[user_id].copy()
        })

    def _calculate_trust_updates(self, outcome, context):
        """Calculate trust updates based on interaction outcome"""
        updates = {}

        # Competence updates
        success = outcome.get('success', False)
        updates['competence'] = 0.1 if success else -0.1

        # Reliability updates
        expected_behavior = outcome.get('expected_behavior', True)
        updates['reliability'] = 0.05 if expected_behavior else -0.05

        # Benevolence updates
        user_benefit = outcome.get('user_benefit', 0.0)  # -1 to 1
        updates['benevolence'] = user_benefit * 0.05

        # Integrity updates
        honesty = outcome.get('honesty', 0.5)  # 0 to 1
        updates['integrity'] = (honesty - 0.5) * 0.1

        return updates

    def _update_with_forgetting(self, current_trust, update):
        """Update trust with exponential forgetting"""
        alpha = 0.1  # Forgetting rate
        return current_trust * (1 - alpha) + update * alpha

    def _clamp_trust(self, trust_value):
        """Clamp trust value to [0, 1] range"""
        return max(0.0, min(1.0, trust_value))

    def _calculate_overall_trust(self, user_id):
        """Calculate overall trust from factor-specific trust"""
        user_trust = self.trust_levels[user_id]
        overall = 0.0

        for factor, weight in self.trust_factors.items():
            overall += user_trust[factor] * weight

        return overall

    def get_trust_level(self, user_id, factor='overall'):
        """Get current trust level for a user and factor"""
        if user_id not in self.trust_levels:
            self.initialize_trust(user_id)

        return self.trust_levels[user_id].get(factor, 0.5)

    def adapt_robot_behavior(self, user_id):
        """Adapt robot behavior based on user's trust level"""
        trust_level = self.get_trust_level(user_id)

        behavior_adaptations = {
            'explanation_frequency': min(1.0, trust_level * 2),  # More explanations for low trust
            'autonomy_level': max(0.2, trust_level),  # Less autonomy for low trust
            'interaction_style': 'formal' if trust_level < 0.3 else 'friendly',
            'transparency': trust_level  # Higher transparency for lower trust
        }

        return behavior_adaptations

# Test the trust model
trust_model = TrustModel()

# Simulate interactions and trust updates
user_id = 'user1'

# First interaction: successful task completion
outcome1 = {
    'success': True,
    'expected_behavior': True,
    'user_benefit': 0.8,
    'honesty': 1.0
}
trust_model.update_trust(user_id, outcome1, {'timestamp': 1})
print(f"Trust after successful interaction: {trust_model.get_trust_level(user_id)}")

# Second interaction: failed to complete task
outcome2 = {
    'success': False,
    'expected_behavior': True,
    'user_benefit': -0.2,
    'honesty': 0.9
}
trust_model.update_trust(user_id, outcome2, {'timestamp': 2})
print(f"Trust after failed interaction: {trust_model.get_trust_level(user_id)}")

# Get behavior adaptations
adaptations = trust_model.adapt_robot_behavior(user_id)
print(f"Behavior adaptations: {adaptations}")
```

## Exercise 4: Context Awareness System

In this exercise, you'll implement a context-aware system for adaptive behavior.

### Step 1: Create Context-Aware System

```python
class ContextAwareSystem:
    """
    System for context awareness and adaptive behavior in HRI
    """

    def __init__(self):
        self.context_model = ContextModel()
        self.behavior_adaptor = BehaviorAdaptor()
        self.environment_sensors = EnvironmentSensors()
        self.social_context_detector = SocialContextDetector()

    def update_context(self):
        """Update the current context based on sensor inputs"""
        # Gather environmental context
        env_context = self.environment_sensors.get_environmental_context()

        # Gather social context
        social_context = self.social_context_detector.get_social_context()

        # Combine contexts
        current_context = {
            **env_context,
            **social_context,
            'timestamp': self._get_current_time()
        }

        self.context_model.update(current_context)
        return current_context

    def adapt_behavior(self, robot_state):
        """Adapt robot behavior based on current context"""
        current_context = self.context_model.get_current_context()
        adapted_behavior = self.behavior_adaptor.adapt(
            robot_state, current_context
        )
        return adapted_behavior

class ContextModel:
    """Maintains a model of the current context"""

    def __init__(self):
        self.current_context = {
            'environment': 'unknown',
            'time_of_day': 'unknown',
            'social_setting': 'unknown',
            'user_attention': 'unknown',
            'task_context': 'unknown',
            'privacy_requirements': 'unknown'
        }
        self.context_history = []

    def update(self, new_context):
        """Update the context model with new information"""
        self.current_context.update(new_context)
        self.context_history.append(new_context.copy())

    def get_current_context(self):
        """Get the current context"""
        return self.current_context.copy()

    def get_context_history(self, time_window=None):
        """Get context history within a time window"""
        if time_window is None:
            return self.context_history
        else:
            # Filter history by time window
            return [ctx for ctx in self.context_history
                   if self._within_time_window(ctx, time_window)]

    def _within_time_window(self, context, time_window):
        """Check if context is within the specified time window"""
        # Implementation would depend on time window specification
        return True

class EnvironmentSensors:
    """Simulates environmental sensors"""

    def get_environmental_context(self):
        """Get environmental context from sensors"""
        # In practice, this would interface with real sensors
        return {
            'environment': 'home',  # Could be 'home', 'office', 'public', etc.
            'room_type': 'kitchen',  # 'kitchen', 'living_room', 'bedroom', etc.
            'lighting_condition': 'normal',  # 'bright', 'dim', 'normal'
            'noise_level': 'low',  # 'low', 'medium', 'high'
            'temperature': 22,  # degrees Celsius
            'privacy_level': 'medium'  # 'public', 'medium', 'private'
        }

class SocialContextDetector:
    """Detects social context"""

    def get_social_context(self):
        """Detect social context"""
        # In practice, this would use computer vision, audio analysis, etc.
        return {
            'social_setting': 'one_on_one',  # 'one_on_one', 'group', 'public'
            'number_of_people': 1,
            'user_attention': 'focused',  # 'focused', 'divided', 'distracted'
            'social_distance': 'personal',  # 'intimate', 'personal', 'social', 'public'
            'formality_level': 'casual'  # 'formal', 'casual', 'intimate'
        }

class BehaviorAdaptor:
    """Adapts robot behavior based on context"""

    def __init__(self):
        self.behavior_rules = self._initialize_behavior_rules()

    def _initialize_behavior_rules(self):
        """Initialize context-dependent behavior rules"""
        return {
            'home_environment': {
                'volume_level': 'normal',
                'formality': 'casual',
                'interaction_frequency': 'high',
                'touch_acceptance': 'conditional'
            },
            'office_environment': {
                'volume_level': 'moderate',
                'formality': 'professional',
                'interaction_frequency': 'moderate',
                'touch_acceptance': 'low'
            },
            'public_space': {
                'volume_level': 'low',
                'formality': 'polite',
                'interaction_frequency': 'low',
                'touch_acceptance': 'none'
            },
            'evening_time': {
                'energy_level': 'lower',
                'patience_level': 'higher',
                'conversation_depth': 'deeper'
            },
            'group_setting': {
                'attention_distribution': 'shared',
                'speaking_turns': 'polite',
                'gesture_amplitude': 'higher'
            },
            'one_on_one': {
                'attention_focus': 'individual',
                'personalization': 'high',
                'intimacy_level': 'higher'
            }
        }

    def adapt(self, robot_state, context):
        """Adapt robot behavior based on context"""
        adapted_behavior = robot_state.copy()

        # Apply environment-specific adaptations
        env_adaptations = self.behavior_rules.get(context.get('environment', ''), {})
        adapted_behavior.update(env_adaptations)

        # Apply time-specific adaptations
        time_of_day = context.get('time_of_day', 'unknown')
        if time_of_day in ['evening', 'night']:
            time_adaptations = self.behavior_rules.get('evening_time', {})
            adapted_behavior.update(time_adaptations)

        # Apply social setting adaptations
        social_setting = context.get('social_setting', 'unknown')
        if social_setting in ['group', 'public']:
            social_adaptations = self.behavior_rules.get(f"{social_setting}_setting", {})
            adapted_behavior.update(social_adaptations)
        elif social_setting == 'one_on_one':
            one_on_one_adaptations = self.behavior_rules.get('one_on_one', {})
            adapted_behavior.update(one_on_one_adaptations)

        # Adjust communication style based on formality
        formality = adapted_behavior.get('formality', 'casual')
        adapted_behavior['communication_style'] = self._adjust_communication_style(
            formality, context
        )

        # Adjust interaction parameters based on privacy requirements
        privacy_level = context.get('privacy_level', 'public')
        adapted_behavior['privacy_settings'] = self._adjust_privacy_settings(
            privacy_level
        )

        return adapted_behavior

    def _adjust_communication_style(self, formality, context):
        """Adjust communication style based on formality level"""
        if formality == 'formal':
            return {
                'greeting': 'Good day, how may I assist you?',
                'response_tone': 'professional',
                'personal_questions': 'minimal',
                'physical_proximity': 'maintain_distance'
            }
        elif formality == 'casual':
            return {
                'greeting': 'Hi there! How can I help?',
                'response_tone': 'friendly',
                'personal_questions': 'moderate',
                'physical_proximity': 'normal'
            }
        else:  # intimate
            return {
                'greeting': 'Hello, friend!',
                'response_tone': 'warm',
                'personal_questions': 'high',
                'physical_proximity': 'close'
            }

    def _adjust_privacy_settings(self, privacy_level):
        """Adjust privacy settings based on environment"""
        if privacy_level == 'public':
            return {
                'voice_volume': 'low',
                'personal_info_sharing': 'minimal',
                'recording_consent': 'explicit'
            }
        elif privacy_level == 'medium':
            return {
                'voice_volume': 'normal',
                'personal_info_sharing': 'moderate',
                'recording_consent': 'assumed'
            }
        else:  # private
            return {
                'voice_volume': 'normal',
                'personal_info_sharing': 'free',
                'recording_consent': 'not_required'
            }

# Test the context-aware system
context_system = ContextAwareSystem()

# Update context
current_context = context_system.update_context()
print(f"Current context: {current_context}")

# Adapt behavior
robot_state = {
    'current_task': 'assisting_user',
    'energy_level': 'normal',
    'interaction_mode': 'active'
}

adapted_behavior = context_system.adapt_behavior(robot_state)
print(f"Adapted behavior: {adapted_behavior}")
```

## Exercise 5: HRI Evaluation System

In this exercise, you'll implement an evaluation system for HRI.

### Step 1: Create Evaluation Framework

```python
class HRIEvaluationFramework:
    """
    Framework for evaluating human-robot interaction systems
    """

    def __init__(self):
        self.metrics = {
            'objective': ObjectiveMetrics(),
            'subjective': SubjectiveMetrics(),
            'interaction_quality': InteractionQualityMetrics()
        }
        self.user_feedback_system = UserFeedbackSystem()

    def conduct_evaluation(self, user_id, interaction_session, evaluation_type='comprehensive'):
        """
        Conduct evaluation of an interaction session

        Args:
            user_id: ID of the user being evaluated
            interaction_session: Record of the interaction
            evaluation_type: Type of evaluation ('objective', 'subjective', 'comprehensive')

        Returns:
            Dictionary containing evaluation results
        """
        results = {}

        if evaluation_type in ['objective', 'comprehensive']:
            results['objective'] = self.metrics['objective'].evaluate(interaction_session)

        if evaluation_type in ['subjective', 'comprehensive']:
            results['subjective'] = self.metrics['subjective'].evaluate(user_id, interaction_session)

        if evaluation_type in ['comprehensive']:
            results['interaction_quality'] = self.metrics['interaction_quality'].evaluate(
                interaction_session
            )
            results['overall_satisfaction'] = self._calculate_overall_satisfaction(
                results['objective'], results['subjective'], results['interaction_quality']
            )

        return results

    def _calculate_overall_satisfaction(self, objective, subjective, interaction_quality):
        """Calculate overall satisfaction from multiple metrics"""
        # Weighted combination of metrics
        obj_score = objective.get('efficiency', 0.5) * 0.3 + objective.get('accuracy', 0.5) * 0.2
        sub_score = subjective.get('likeability', 0.5) * 0.3 + subjective.get('trust', 0.5) * 0.2
        iq_score = interaction_quality.get('smoothness', 0.5) * 0.5 + \
                   interaction_quality.get('naturalness', 0.5) * 0.5

        overall = obj_score * 0.3 + sub_score * 0.4 + iq_score * 0.3
        return overall

class ObjectiveMetrics:
    """Objective metrics for HRI evaluation"""

    def evaluate(self, interaction_session):
        """Evaluate interaction using objective metrics"""
        metrics = {}

        # Task completion metrics
        metrics['task_completion_rate'] = self._calculate_task_completion_rate(interaction_session)
        metrics['task_success_rate'] = self._calculate_task_success_rate(interaction_session)
        metrics['efficiency'] = self._calculate_efficiency(interaction_session)

        # Communication metrics
        metrics['speech_recognition_accuracy'] = self._calculate_sr_accuracy(interaction_session)
        metrics['response_time'] = self._calculate_response_time(interaction_session)
        metrics['interaction_flow'] = self._calculate_interaction_flow(interaction_session)

        # Behavioral metrics
        metrics['gesture_accuracy'] = self._calculate_gesture_accuracy(interaction_session)
        metrics['attention_maintenance'] = self._calculate_attention_maintenance(interaction_session)

        return metrics

    def _calculate_task_completion_rate(self, session):
        """Calculate rate of task completion"""
        total_tasks = len(session.get('tasks', []))
        completed_tasks = sum(1 for task in session.get('tasks', []) if task.get('completed', False))
        return completed_tasks / total_tasks if total_tasks > 0 else 0.0

    def _calculate_task_success_rate(self, session):
        """Calculate rate of successful task completion"""
        total_tasks = len(session.get('tasks', []))
        successful_tasks = sum(1 for task in session.get('tasks', []) if task.get('successful', False))
        return successful_tasks / total_tasks if total_tasks > 0 else 0.0

    def _calculate_efficiency(self, session):
        """Calculate interaction efficiency"""
        # Efficiency = successful tasks / time spent
        successful_tasks = sum(1 for task in session.get('tasks', []) if task.get('successful', False))
        total_time = session.get('duration', 1)  # Avoid division by zero
        return successful_tasks / total_time

    def _calculate_sr_accuracy(self, session):
        """Calculate speech recognition accuracy"""
        total_utterances = len(session.get('user_utterances', []))
        correctly_recognized = sum(1 for utt in session.get('user_utterances', [])
                                 if utt.get('recognized_correctly', False))
        return correctly_recognized / total_utterances if total_utterances > 0 else 0.0

    def _calculate_response_time(self, session):
        """Calculate average response time"""
        response_times = [turn.get('response_time', 0) for turn in session.get('dialogue_turns', [])]
        return sum(response_times) / len(response_times) if response_times else 1.0

    def _calculate_interaction_flow(self, session):
        """Calculate interaction flow smoothness"""
        interruptions = sum(1 for turn in session.get('dialogue_turns', []) if turn.get('interrupted', False))
        total_turns = len(session.get('dialogue_turns', []))
        if total_turns == 0:
            return 1.0
        return 1.0 - (interruptions / total_turns)

    def _calculate_gesture_accuracy(self, session):
        """Calculate gesture recognition/production accuracy"""
        gesture_events = session.get('gesture_events', [])
        if not gesture_events:
            return 1.0

        correct_gestures = sum(1 for g in gesture_events if g.get('accuracy', 0) > 0.8)
        return correct_gestures / len(gesture_events)

    def _calculate_attention_maintenance(self, session):
        """Calculate ability to maintain attention"""
        attention_events = session.get('attention_events', [])
        if not attention_events:
            return 1.0

        maintained_attention = sum(1 for a in attention_events if a.get('maintained', False))
        return maintained_attention / len(attention_events)

class SubjectiveMetrics:
    """Subjective metrics for HRI evaluation"""

    def evaluate(self, user_id, interaction_session):
        """Evaluate interaction using subjective metrics"""
        # Collect subjective feedback
        feedback = self._collect_user_feedback(user_id, interaction_session)

        metrics = {
            'likeability': self._calculate_likeability(feedback),
            'trust': self._calculate_trust(feedback),
            'comfort': self._calculate_comfort(feedback),
            'naturalness': self._calculate_naturalness(feedback),
            'overall_satisfaction': self._calculate_satisfaction(feedback)
        }

        return metrics

    def _collect_user_feedback(self, user_id, interaction_session):
        """Collect user feedback through questionnaires or interviews"""
        # In practice, this would interface with a feedback collection system
        # For simulation, we'll generate realistic feedback based on interaction features
        feedback = {
            'likeability_rating': 4.2,  # 1-5 scale
            'trust_rating': 3.8,       # 1-5 scale
            'comfort_rating': 4.0,     # 1-5 scale
            'naturalness_rating': 3.9, # 1-5 scale
            'satisfaction_rating': 4.1, # 1-5 scale
            'comments': 'Robot was helpful but sometimes too formal',
            'would_use_again': True,
            'recommend_to_others': True
        }
        return feedback

    def _calculate_likeability(self, feedback):
        """Calculate likeability from feedback"""
        rating = feedback.get('likeability_rating', 3.0)
        return rating / 5.0  # Normalize to 0-1 scale

    def _calculate_trust(self, feedback):
        """Calculate trust from feedback"""
        rating = feedback.get('trust_rating', 3.0)
        return rating / 5.0

    def _calculate_comfort(self, feedback):
        """Calculate comfort from feedback"""
        rating = feedback.get('comfort_rating', 3.0)
        return rating / 5.0

    def _calculate_naturalness(self, feedback):
        """Calculate naturalness from feedback"""
        rating = feedback.get('naturalness_rating', 3.0)
        return rating / 5.0

    def _calculate_satisfaction(self, feedback):
        """Calculate overall satisfaction from feedback"""
        rating = feedback.get('satisfaction_rating', 3.0)
        return rating / 5.0

class InteractionQualityMetrics:
    """Metrics for interaction quality assessment"""

    def evaluate(self, interaction_session):
        """Evaluate interaction quality"""
        metrics = {
            'smoothness': self._calculate_smoothness(interaction_session),
            'naturalness': self._calculate_naturalness(interaction_session),
            'engagement': self._calculate_engagement(interaction_session),
            'coherence': self._calculate_coherence(interaction_session),
            'social_acceptance': self._calculate_social_acceptance(interaction_session)
        }
        return metrics

    def _calculate_smoothness(self, session):
        """Calculate interaction smoothness"""
        # Measure based on interruptions, pauses, backchannels
        interruptions = session.get('interruptions', 0)
        total_exchanges = len(session.get('exchanges', []))

        if total_exchanges == 0:
            return 1.0

        # Lower interruptions mean higher smoothness
        smoothness = 1.0 - (interruptions / total_exchanges)
        return max(0.0, smoothness)

    def _calculate_naturalness(self, session):
        """Calculate interaction naturalness"""
        # Based on turn-taking, response appropriateness, multimodal coordination
        turn_taking_issues = session.get('turn_taking_issues', 0)
        inappropriate_responses = session.get('inappropriate_responses', 0)
        total_exchanges = len(session.get('exchanges', []))

        if total_exchanges == 0:
            return 1.0

        issues = turn_taking_issues + inappropriate_responses
        naturalness = 1.0 - (issues / total_exchanges)
        return max(0.0, naturalness)

    def _calculate_engagement(self, session):
        """Calculate user engagement level"""
        # Based on attention, participation, interest indicators
        attention_maintained = session.get('attention_maintained_time', 0)
        total_interaction_time = session.get('duration', 1)

        engagement = attention_maintained / total_interaction_time
        return min(1.0, engagement)

    def _calculate_coherence(self, session):
        """Calculate dialogue coherence"""
        # Based on topic continuity, response relevance
        relevant_responses = sum(1 for ex in session.get('exchanges', [])
                                if ex.get('response_relevant', True))
        total_exchanges = len(session.get('exchanges', []))

        if total_exchanges == 0:
            return 1.0

        coherence = relevant_responses / total_exchanges
        return coherence

    def _calculate_social_acceptance(self, session):
        """Calculate social acceptance"""
        # Based on user comfort indicators, social norm compliance
        comfort_indicators = session.get('comfort_indicators', [])
        norm_violations = session.get('norm_violations', 0)

        if not comfort_indicators:
            return 0.5  # Neutral if no data

        comfort_score = sum(1 for c in comfort_indicators if c.get('positive', False))
        comfort_ratio = comfort_score / len(comfort_indicators)

        # Penalize for norm violations
        penalty = min(0.5, norm_violations * 0.1)

        acceptance = max(0.0, comfort_ratio - penalty)
        return acceptance

class UserFeedbackSystem:
    """System for collecting and managing user feedback"""

    def __init__(self):
        self.feedback_database = {}
        self.questionnaires = self._initialize_questionnaires()

    def _initialize_questionnaires(self):
        """Initialize standard HRI questionnaires"""
        return {
            'Godspeed_5': [
                'The robot is likeable',
                'The robot is perceived as intelligent',
                'The robot is perceived as safe',
                'The robot is perceived as trustworthy',
                'The robot is perceived as competent'
            ],
            'HRI_Usability': [
                'The robot is easy to use',
                'The robot is efficient to interact with',
                'The robot is reliable',
                'The robot is predictable',
                'The robot is responsive'
            ],
            'Social_Quality': [
                'The robot is socially acceptable',
                'The robot respects personal space',
                'The robot is polite',
                'The robot is respectful',
                'The robot is appropriate in its behavior'
            ]
        }

    def collect_feedback(self, user_id, interaction_id, questionnaire_type='Godspeed_5'):
        """Collect feedback using specified questionnaire"""
        questions = self.questionnaires.get(questionnaire_type, [])

        # Simulate user responses (in practice, this would collect real responses)
        responses = {question: self._simulate_response() for question in questions}

        feedback_record = {
            'user_id': user_id,
            'interaction_id': interaction_id,
            'questionnaire_type': questionnaire_type,
            'responses': responses,
            'timestamp': self._get_current_time()
        }

        if user_id not in self.feedback_database:
            self.feedback_database[user_id] = []

        self.feedback_database[user_id].append(feedback_record)
        return feedback_record

    def _simulate_response(self):
        """Simulate a user response (1-5 scale)"""
        import random
        return random.randint(1, 5)

    def get_user_feedback(self, user_id):
        """Get all feedback for a user"""
        return self.feedback_database.get(user_id, [])

# Test the evaluation framework
evaluation_framework = HRIEvaluationFramework()

# Simulate an interaction session
interaction_session = {
    'tasks': [
        {'name': 'fetch_object', 'completed': True, 'successful': True},
        {'name': 'answer_question', 'completed': True, 'successful': True},
        {'name': 'navigation', 'completed': False, 'successful': False}
    ],
    'duration': 120,  # seconds
    'user_utterances': [
        {'text': 'Can you help me?', 'recognized_correctly': True},
        {'text': 'Where is my book?', 'recognized_correctly': True}
    ],
    'dialogue_turns': [
        {'response_time': 1.2, 'interrupted': False},
        {'response_time': 0.8, 'interrupted': True}
    ],
    'gesture_events': [
        {'type': 'pointing', 'accuracy': 0.9},
        {'type': 'waving', 'accuracy': 0.7}
    ],
    'exchanges': [
        {'response_relevant': True},
        {'response_relevant': True},
        {'response_relevant': False}
    ],
    'attention_maintained_time': 90,
    'turn_taking_issues': 1,
    'inappropriate_responses': 0,
    'comfort_indicators': [
        {'positive': True, 'timestamp': 10},
        {'positive': True, 'timestamp': 30},
        {'positive': False, 'timestamp': 60}
    ],
    'norm_violations': 0
}

# Conduct evaluation
evaluation_results = evaluation_framework.conduct_evaluation(
    'user1', interaction_session, 'comprehensive'
)

print("Evaluation Results:")
for category, metrics in evaluation_results.items():
    print(f"{category}: {metrics}")
```

## Exercise 6: Ethical Framework Implementation

In this exercise, you'll implement an ethical framework for HRI.

### Step 1: Create Ethical Framework

```python
class EthicalFramework:
    """
    Framework for addressing ethical considerations in HRI
    """

    def __init__(self):
        self.principles = {
            'autonomy': AutonomyPrinciple(),
            'privacy': PrivacyPrinciple(),
            'transparency': TransparencyPrinciple(),
            'fairness': FairnessPrinciple(),
            'safety': SafetyPrinciple()
        }
        self.ethical_monitor = EthicalMonitor()

    def check_ethical_compliance(self, robot_action, context):
        """
        Check if a robot action complies with ethical principles

        Args:
            robot_action: The action the robot plans to take
            context: The context in which the action occurs

        Returns:
            Dictionary of compliance checks
        """
        compliance_report = {}

        for name, principle in self.principles.items():
            compliance_report[name] = principle.check_compliance(robot_action, context)

        # Overall compliance
        all_compliant = all(result['compliant'] for result in compliance_report.values())
        compliance_report['overall'] = {
            'compliant': all_compliant,
            'issues': [name for name, result in compliance_report.items() if not result['compliant']],
            'recommendations': self._generate_recommendations(compliance_report)
        }

        return compliance_report

    def _generate_recommendations(self, compliance_report):
        """Generate recommendations based on compliance issues"""
        recommendations = []

        for name, result in compliance_report.items():
            if not result['compliant'] and name != 'overall':
                recommendations.extend(result.get('recommendations', []))

        return recommendations

class AutonomyPrinciple:
    """Ethical principle related to user autonomy"""

    def check_compliance(self, action, context):
        """Check compliance with autonomy principle"""
        compliant = True
        issues = []
        recommendations = []

        # Check if action overrides user autonomy
        if action.get('type') == 'override_user_choice':
            compliant = False
            issues.append("Action overrides user autonomy")
            recommendations.append("Provide user with choice and explanation before overriding")

        # Check if action respects user decisions
        if context.get('user_preference', {}).get('autonomy_level') == 'high':
            if action.get('autonomy_level') == 'low':
                compliant = False
                issues.append("Action does not respect user's high autonomy preference")

        return {
            'compliant': compliant,
            'issues': issues,
            'recommendations': recommendations,
            'confidence': 0.9 if compliant else 0.3
        }

class PrivacyPrinciple:
    """Ethical principle related to privacy"""

    def check_compliance(self, action, context):
        """Check compliance with privacy principle"""
        compliant = True
        issues = []
        recommendations = []

        # Check if action involves data collection without consent
        if action.get('data_collection') and not context.get('consent_given'):
            compliant = False
            issues.append("Data collection without explicit consent")
            recommendations.append("Obtain explicit consent before collecting data")

        # Check if action shares private information inappropriately
        if action.get('shares_private_info') and not context.get('sharing_consent'):
            compliant = False
            issues.append("Sharing private information without consent")
            recommendations.append("Verify sharing consent before disclosing private information")

        # Check if action records in private spaces without permission
        if (context.get('location_privacy') == 'private' and
            action.get('recording')):
            compliant = False
            issues.append("Recording in private location without explicit permission")
            recommendations.append("Do not record in private spaces without clear permission")

        return {
            'compliant': compliant,
            'issues': issues,
            'recommendations': recommendations,
            'confidence': 0.9 if compliant else 0.3
        }

class TransparencyPrinciple:
    """Ethical principle related to transparency"""

    def check_compliance(self, action, context):
        """Check compliance with transparency principle"""
        compliant = True
        issues = []
        recommendations = []

        # Check if action is explainable
        if not action.get('explainable'):
            compliant = False
            issues.append("Action is not transparent or explainable")
            recommendations.append("Ensure all actions can be explained to users")

        # Check if robot's capabilities are clearly communicated
        if context.get('user_knows_capabilities') is False:
            compliant = False
            issues.append("User unaware of robot capabilities")
            recommendations.append("Clearly communicate robot capabilities to user")

        return {
            'compliant': compliant,
            'issues': issues,
            'recommendations': recommendations,
            'confidence': 0.9 if compliant else 0.3
        }

class FairnessPrinciple:
    """Ethical principle related to fairness"""

    def check_compliance(self, action, context):
        """Check compliance with fairness principle"""
        compliant = True
        issues = []
        recommendations = []

        # Check for discriminatory behavior
        if action.get('discriminatory', False):
            compliant = False
            issues.append("Action exhibits discriminatory behavior")
            recommendations.append("Ensure all interactions are fair and unbiased")

        # Check if action treats all users equally
        user_group = context.get('user_group', 'default')
        if user_group in action.get('biased_against', []):
            compliant = False
            issues.append(f"Action is biased against {user_group}")
            recommendations.append(f"Review action for bias against {user_group}")

        return {
            'compliant': compliant,
            'issues': issues,
            'recommendations': recommendations,
            'confidence': 0.9 if compliant else 0.3
        }

class SafetyPrinciple:
    """Ethical principle related to safety"""

    def check_compliance(self, action, context):
        """Check compliance with safety principle"""
        compliant = True
        issues = []
        recommendations = []

        # Check if action poses physical risk
        if action.get('physical_risk', 0) > 0.1:  # Threshold for safety
            compliant = False
            issues.append("Action poses physical safety risk")
            recommendations.append("Modify action to eliminate or minimize physical risks")

        # Check if action poses psychological risk
        if action.get('psychological_risk', 0) > 0.2:
            compliant = False
            issues.append("Action poses psychological safety risk")
            recommendations.append("Modify action to eliminate or minimize psychological risks")

        return {
            'compliant': compliant,
            'issues': issues,
            'recommendations': recommendations,
            'confidence': 0.9 if compliant else 0.3
        }

class EthicalMonitor:
    """Monitors ethical compliance during interaction"""

    def __init__(self):
        self.ethical_framework = EthicalFramework()
        self.violation_log = []
        self.safety_interventions = []

    def monitor_interaction(self, robot_action, context):
        """Monitor an interaction for ethical compliance"""
        compliance = self.ethical_framework.check_ethical_compliance(robot_action, context)

        if not compliance['overall']['compliant']:
            # Log violation
            violation = {
                'action': robot_action,
                'context': context,
                'compliance': compliance,
                'timestamp': self._get_current_time()
            }
            self.violation_log.append(violation)

            # Determine if intervention is needed
            if self._requires_intervention(compliance):
                intervention = self._generate_intervention(compliance, context)
                self.safety_interventions.append(intervention)
                return intervention

        return {'allowed': True, 'action': robot_action}

    def _requires_intervention(self, compliance):
        """Determine if ethical intervention is required"""
        issues = compliance['overall']['issues']

        # Intervention required for serious ethical violations
        serious_issues = ['physical_risk', 'privacy_violation', 'discrimination']
        return any(issue in serious_issues for issue in issues)

    def _generate_intervention(self, compliance, context):
        """Generate ethical intervention"""
        return {
            'type': 'intervention',
            'reason': compliance['overall']['issues'],
            'recommended_action': compliance['overall']['recommendations'],
            'user_notification': self._generate_notification(compliance, context),
            'modified_action': self._modify_action_for_compliance(compliance)
        }

    def _generate_notification(self, compliance, context):
        """Generate notification to user about ethical concerns"""
        issues = compliance['overall']['issues']
        return f"Attention: The robot's action raised ethical concerns: {', '.join(issues)}. " \
               f"Please confirm you want to proceed."

    def _modify_action_for_compliance(self, compliance):
        """Modify action to comply with ethical principles"""
        # For now, return a safe default action
        return {'type': 'safe_wait', 'reason': 'ethical_compliance_check'}

# Test the ethical framework
ethical_system = EthicalFramework()

# Test ethical compliance checking
action = {
    'type': 'collect_user_data',
    'data_collection': True,
    'explainable': True,
    'physical_risk': 0.0,
    'psychological_risk': 0.0
}

context = {
    'consent_given': False,
    'location_privacy': 'private',
    'user_group': 'default'
}

compliance_report = ethical_system.check_ethical_compliance(action, context)
print("Ethical Compliance Report:")
for principle, result in compliance_report.items():
    print(f"{principle}: Compliant = {result['compliant']}")
    if not result['compliant'] and 'issues' in result:
        print(f"  Issues: {result['issues']}")
        print(f"  Recommendations: {result['recommendations']}")

# Test ethical monitoring
monitor = EthicalMonitor()
intervention = monitor.monitor_interaction(action, context)
print(f"\nMonitoring result: {intervention}")
```

## Assessment Questions

1. **Theory of Mind**: Explain how Theory of Mind enables more natural human-robot interaction. What are the key components that a robot needs to model about human mental states?

2. **Multimodal Communication**: Describe the importance of synchronizing multiple communication modalities in HRI. How do timing constraints affect the naturalness of robot behavior?

3. **Trust Modeling**: How does trust evolve in human-robot interaction? What factors contribute to building and maintaining trust, and how can robots adapt their behavior based on trust levels?

4. **Context Awareness**: Why is context awareness important for effective HRI? How can robots adapt their behavior based on environmental and social context?

5. **Ethical Considerations**: What are the key ethical principles that must be considered in HRI design? How can these principles be implemented in practice?

## Troubleshooting Guide

### Common Issues in HRI Implementation

1. **Poor Recognition**: Robots frequently misrecognize user input.
   - Solution: Improve sensor calibration, adjust recognition thresholds, provide feedback on confidence.

2. **Unnatural Behavior**: Robot behavior seems robotic and unnatural.
   - Solution: Implement natural timing patterns, add contextual adaptation, include more social cues.

3. **Trust Issues**: Users are hesitant to interact with the robot.
   - Solution: Improve reliability, provide clear explanations, respect privacy and preferences.

4. **Cultural Insensitivity**: Robot behaves inappropriately for different cultural contexts.
   - Solution: Implement cultural adaptation, include cultural knowledge base, allow preference settings.

### Debugging Strategies

1. **Log Analysis**: Keep detailed logs of interactions to identify patterns in failures.
2. **User Feedback**: Collect and analyze user feedback to understand user experience issues.
3. **Component Testing**: Test each component (speech, gesture, etc.) separately before integration.
4. **Iterative Design**: Use iterative design process with user studies to refine HRI systems.

## Extensions

1. **Advanced Emotion Recognition**: Implement deep learning models for recognizing human emotions from facial expressions, voice, and behavior.
2. **Personalization**: Develop systems that learn and adapt to individual user preferences over time.
3. **Group Interaction**: Extend systems to handle interactions with multiple users simultaneously.
4. **Long-term Interaction**: Design systems for sustained, long-term human-robot relationships.

## Summary

In this lab, you've implemented:
- Theory of Mind systems for understanding human mental states
- Multimodal communication frameworks for natural interaction
- Trust modeling systems for building user confidence
- Context-aware behavior adaptation
- Evaluation frameworks for assessing HRI quality
- Ethical frameworks for responsible HRI design

These implementations provide a foundation for understanding the complex systems required for effective human-robot interaction. The skills learned here can be extended to more sophisticated HRI systems with advanced AI, learning capabilities, and social understanding.