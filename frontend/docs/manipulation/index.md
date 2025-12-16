---
title: Manipulation & Grasping
sidebar_position: 14
---

# Manipulation & Grasping

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental principles of robotic manipulation and grasping
- Analyze different types of robotic grippers and their applications
- Implement grasp planning algorithms for various objects
- Design force control strategies for stable grasping
- Apply machine learning techniques to improve manipulation capabilities

## Introduction to Robotic Manipulation

Robotic manipulation is a critical capability for humanoid robots, enabling them to interact with objects in their environment. Unlike fixed-base industrial robots, humanoid robots must perform manipulation tasks while maintaining balance and coordinating with locomotion. This adds significant complexity to manipulation planning and control.

Robotic manipulation involves several key components:
- **Perception**: Identifying and localizing objects in the environment
- **Planning**: Determining how to approach and grasp objects
- **Control**: Executing the manipulation task with appropriate forces and motions
- **Learning**: Improving manipulation skills through experience

### Key Challenges in Humanoid Manipulation

1. **Dynamic Balance**: Maintaining stability while extending arms to manipulate objects
2. **Limited Workspace**: Working within the physical constraints of the humanoid form
3. **Force Control**: Managing contact forces to prevent object damage or robot instability
4. **Coordination**: Synchronizing manipulation with other robot behaviors

## Types of Robotic Grippers

The choice of gripper significantly impacts the manipulation capabilities of a humanoid robot. Different gripper types are suited for different tasks and objects.

### Parallel Jaw Grippers

Parallel jaw grippers are the most common type of robotic gripper, consisting of two opposing fingers that move in parallel. They are simple, reliable, and suitable for a wide range of objects.

**Advantages:**
- Simple control mechanism
- Reliable grasping for objects with parallel surfaces
- Easy to maintain and repair

**Disadvantages:**
- Limited to objects with graspable surfaces
- Difficulty with irregularly shaped objects
- Requires precise positioning

### Multi-Fingered Hands

Multi-fingered hands provide greater dexterity and can manipulate objects with more complex shapes. They can perform both power grasps and precision grasps, similar to human hands.

```python
class MultiFingeredHand:
    def __init__(self, num_fingers=5, joints_per_finger=3):
        self.num_fingers = num_fingers
        self.joints_per_finger = joints_per_finger
        self.finger_positions = [[0.0] * joints_per_finger for _ in range(num_fingers)]

    def power_grasp(self, object_diameter):
        """Perform a power grasp suitable for large objects"""
        # Close fingers around the object
        for i in range(self.num_fingers):
            for j in range(self.joints_per_finger):
                self.finger_positions[i][j] = min(1.57, object_diameter * 0.5)

    def precision_grasp(self):
        """Perform a precision grasp using fingertips"""
        # Position fingertips to grasp small objects
        # Thumb opposes other fingers
        self.finger_positions[0] = [1.0, 0.5, 0.3]  # Thumb
        for i in range(1, self.num_fingers):
            self.finger_positions[i] = [0.2, 0.8, 1.2]  # Other fingers
```

### Suction Grippers

Suction grippers use vacuum pressure to pick up objects. They are particularly effective for flat, smooth objects.

**Applications:**
- Handling flat objects like papers or sheets
- Picking up objects with smooth surfaces
- Delicate object handling where minimal contact force is required

### Adaptive/Underactuated Hands

Adaptive hands can conform to the shape of objects, providing stable grasps without requiring precise positioning. They often use underactuation principles where a single motor can control multiple joints.

## Grasp Planning

Grasp planning involves determining the optimal configuration of a gripper to securely grasp an object. This includes selecting contact points, determining grasp forces, and planning the approach trajectory.

### Grasp Quality Metrics

Several metrics are used to evaluate the quality of a potential grasp:

1. **Force Closure**: The ability to resist arbitrary external forces and torques
2. **Grasp Wrench Space**: The set of forces and torques that can be applied to the object
3. **Grasp Stability**: The robustness of the grasp to perturbations
4. **Approach Feasibility**: Whether the gripper can physically reach the grasp configuration

### Grasp Synthesis Approaches

#### Analytical Methods

Analytical methods use geometric and physical models to determine stable grasp configurations:

```python
import numpy as np
from scipy.spatial.distance import cdist

def find_grasp_points(object_mesh, gripper_width):
    """Find potential grasp points on an object mesh"""
    # Find surface points on the object
    surface_points = get_surface_points(object_mesh)

    # Calculate pairwise distances between points
    distances = cdist(surface_points, surface_points)

    # Find pairs of points that match gripper width
    grasp_candidates = []
    for i in range(len(surface_points)):
        for j in range(i+1, len(surface_points)):
            if abs(distances[i, j] - gripper_width) < 0.01:  # 1cm tolerance
                # Check if points face opposite directions (potential grasp)
                normal_i = get_surface_normal(object_mesh, surface_points[i])
                normal_j = get_surface_normal(object_mesh, surface_points[j])

                if np.dot(normal_i, normal_j) < -0.8:  # Opposing normals
                    grasp_candidates.append((surface_points[i], surface_points[j]))

    return grasp_candidates

def evaluate_grasp_quality(grasp_points, object_properties):
    """Evaluate the quality of a potential grasp"""
    point1, point2 = grasp_points

    # Calculate grasp width
    grasp_width = np.linalg.norm(point1 - point2)

    # Calculate grasp stability based on friction coefficients
    mu = object_properties['friction_coefficient']
    normal1 = get_surface_normal_at_point(point1)
    normal2 = get_surface_normal_at_point(point2)

    # Evaluate force closure conditions
    # This is a simplified evaluation
    force_closure_score = calculate_force_closure(normal1, normal2, mu)

    # Consider object weight and center of mass
    object_weight = object_properties['weight']
    com = object_properties['center_of_mass']

    # Calculate torque around grasp points
    torque1 = np.cross(com - point1, [0, 0, -object_weight])
    torque2 = np.cross(com - point2, [0, 0, -object_weight])

    # Return quality score
    return force_closure_score * (1.0 / (1.0 + np.linalg.norm(torque1) + np.linalg.norm(torque2)))
```

#### Learning-Based Methods

Modern approaches use machine learning to improve grasp planning:

```python
import torch
import torch.nn as nn

class GraspQualityNetwork(nn.Module):
    """Neural network to predict grasp quality from visual input"""

    def __init__(self, input_channels=3, num_classes=1):
        super(GraspQualityNetwork, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU()
        )

        # Calculate the size of the flattened features
        conv_output_size = 128 * 6 * 6  # Assuming input is 64x64

        # Fully connected layers for grasp quality prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + 2, 256),  # +2 for grasp position
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Output between 0 and 1 (quality score)
        )

    def forward(self, image, grasp_position):
        # Process image through convolutional layers
        features = self.conv_layers(image)
        features = features.view(features.size(0), -1)  # Flatten

        # Concatenate with grasp position
        grasp_pos_tensor = torch.tensor(grasp_position, dtype=torch.float32)
        combined_features = torch.cat((features, grasp_pos_tensor), dim=1)

        # Predict grasp quality
        quality_score = self.fc_layers(combined_features)
        return quality_score

# Example usage
def predict_grasp_quality(robot_vision, grasp_position):
    """Use learned model to predict grasp quality"""
    # Load pre-trained model
    model = GraspQualityNetwork()
    # model.load_state_dict(torch.load('grasp_quality_model.pth'))

    # Preprocess image from robot vision
    image_tensor = preprocess_image(robot_vision)

    # Predict quality
    quality = model(image_tensor, grasp_position)
    return quality.item()
```

## Force Control in Manipulation

Force control is crucial for successful manipulation, especially when interacting with objects of varying compliance or when performing tasks that require specific contact forces.

### Impedance Control

Impedance control treats the robot as a mechanical impedance (mass, damping, stiffness) that can be modulated to achieve desired interaction behavior:

```python
class ImpedanceController:
    def __init__(self, mass=1.0, damping=10.0, stiffness=100.0):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness

    def update(self, desired_pos, current_pos, desired_force, current_force, dt):
        """Update impedance controller"""
        # Position error
        pos_error = desired_pos - current_pos

        # Force error
        force_error = desired_force - current_force

        # Calculate control force based on impedance model
        control_force = (self.stiffness * pos_error +
                        self.damping * (pos_error / dt if dt > 0 else 0) +
                        self.mass * (pos_error / (dt**2) if dt > 0 else 0) +
                        force_error)

        return control_force

def compliant_grasp_approach(object_pos, gripper_pos, controller, dt):
    """Approach an object with compliant control"""
    # Set desired position to be slightly before the object
    desired_pos = object_pos - np.array([0.05, 0, 0])  # 5cm before object

    # Set desired force to be low during approach
    desired_force = 0.5  # 0.5N

    # Get current measurements
    current_pos = gripper_pos
    current_force = measure_contact_force()  # Simulated function

    # Calculate control force
    control_force = controller.update(
        desired_pos, current_pos, desired_force, current_force, dt
    )

    # Apply control force to gripper
    apply_force_to_gripper(control_force)
```

### Hybrid Position/Force Control

For complex manipulation tasks, it's often necessary to control both position and force simultaneously in different coordinate directions:

```python
def hybrid_position_force_control(desired_pos, desired_force, current_pos, current_force,
                                force_control_axes, dt):
    """
    Hybrid position/force control
    force_control_axes: list of axes (0,1,2) where force control is applied
    """
    # Initialize control command
    control_cmd = np.zeros(6)  # 3 position + 3 orientation

    # Position control for non-force-controlled axes
    for i in range(3):
        if i not in force_control_axes:
            control_cmd[i] = desired_pos[i] - current_pos[i]

    # Force control for specified axes
    for i in force_control_axes:
        force_error = desired_force[i] - current_force[i]
        control_cmd[i] = force_error  # Convert to position adjustment

    return control_cmd
```

## Grasping Strategies

Different objects require different grasping strategies based on their shape, weight, and surface properties.

### Power Grasps

Power grasps are used for heavy objects or when high grip forces are needed:

```python
def plan_power_grasp(object_properties):
    """Plan a power grasp for heavy objects"""
    grasp_config = {
        'gripper_type': 'parallel_jaw',
        'jaw_aperture': object_properties['diameter'] * 1.1,  # 10% larger
        'grasp_force': min(50.0, object_properties['weight'] * 5),  # 5x weight
        'approach_direction': object_properties['major_axis'],
        'grasp_orientation': 'horizontal'  # For stability
    }
    return grasp_config
```

### Precision Grasps

Precision grasps are used for delicate objects or when fine manipulation is required:

```python
def plan_precision_grasp(object_properties):
    """Plan a precision grasp for delicate objects"""
    grasp_config = {
        'gripper_type': 'multi_fingered',
        'finger_positions': calculate_finger_poses(object_properties),
        'grasp_force': min(5.0, object_properties['weight'] * 2),  # Lower force
        'contact_points': identify_delicate_areas(object_properties),
        'approach_direction': 'top_down'  # Minimize contact area
    }
    return grasp_config
```

## Manipulation Planning

Manipulation planning involves generating trajectories that allow the robot to achieve its manipulation goals while avoiding obstacles and maintaining balance.

### Task and Motion Planning (TAMP)

Task and Motion Planning integrates high-level task planning with low-level motion planning:

```python
class ManipulationPlanner:
    def __init__(self, robot_model, environment):
        self.robot = robot_model
        self.env = environment
        self.kinematics = robot_model.kinematics

    def plan_manipulation_task(self, task_description):
        """
        Plan a manipulation task considering both task-level and motion-level constraints
        """
        # Decompose task into subtasks
        subtasks = self.decompose_task(task_description)

        # Plan motion for each subtask
        trajectories = []
        for subtask in subtasks:
            trajectory = self.plan_subtask_motion(subtask)
            if trajectory is None:
                return None  # Planning failed
            trajectories.append(trajectory)

        return self.concatenate_trajectories(trajectories)

    def decompose_task(self, task_description):
        """Decompose high-level task into motion primitives"""
        # Example: "Pick up cup and place on table"
        # Subtasks: approach cup, grasp cup, lift cup, approach table, place cup
        return [
            {'action': 'approach', 'target': task_description['object']},
            {'action': 'grasp', 'target': task_description['object']},
            {'action': 'lift', 'target': task_description['object']},
            {'action': 'approach', 'target': task_description['destination']},
            {'action': 'place', 'target': task_description['destination']}
        ]

    def plan_subtask_motion(self, subtask):
        """Plan motion for a specific subtask"""
        if subtask['action'] == 'approach':
            return self.plan_approach_motion(subtask['target'])
        elif subtask['action'] == 'grasp':
            return self.plan_grasp_motion(subtask['target'])
        elif subtask['action'] == 'lift':
            return self.plan_lift_motion(subtask['target'])
        elif subtask['action'] == 'place':
            return self.plan_place_motion(subtask['target'])
        else:
            return None
```

### Grasp-Dependent Planning

The grasp configuration affects the robot's reachable workspace and the feasible manipulation paths:

```python
def plan_with_grasp_constraints(robot_config, object_pose, grasp_config):
    """
    Plan manipulation motion considering the current grasp configuration
    """
    # Calculate reachable workspace based on grasp
    reachable_workspace = calculate_reachable_workspace(
        robot_config, grasp_config
    )

    # Check if destination is reachable
    if not is_in_workspace(reachable_workspace, object_pose):
        # Plan to regrasp the object
        regrasp_plan = plan_regrasp(robot_config, grasp_config)
        if regrasp_plan is None:
            return None  # Cannot reach destination with current grasp

        # Plan to destination after regrasp
        destination_plan = plan_to_destination(
            robot_config, object_pose, regrasp_plan['new_grasp']
        )

        return concatenate_plans(regrasp_plan, destination_plan)
    else:
        # Direct plan to destination is possible
        return plan_to_destination(robot_config, object_pose, grasp_config)
```

## Learning-Based Manipulation

Modern manipulation systems increasingly use machine learning to improve performance and adapt to new situations.

### Reinforcement Learning for Grasping

Reinforcement learning can be used to learn optimal grasping policies:

```python
import torch
import torch.nn as nn
import numpy as np

class GraspPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GraspPolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # Output probability distribution over actions
        )

    def forward(self, state):
        return self.network(state)

class GraspLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):
        self.policy_net = GraspPolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.gamma = 0.99  # Discount factor

    def select_action(self, state, explore=True):
        """Select action using the policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)

        if explore:
            # Sample from distribution during training
            action = torch.multinomial(action_probs, 1).item()
        else:
            # Take best action during execution
            action = torch.argmax(action_probs).item()

        return action, action_probs[0][action].item()

    def update_policy(self, states, actions, rewards, log_probs):
        """Update policy using REINFORCE algorithm"""
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
```

### Imitation Learning

Imitation learning allows robots to learn manipulation skills by observing human demonstrations:

```python
def train_from_demonstrations(demonstration_data):
    """
    Train a manipulation policy using imitation learning
    demonstration_data: list of (state, action) pairs from human demonstrations
    """
    # Separate states and actions
    states = torch.FloatTensor([d[0] for d in demonstration_data])
    actions = torch.LongTensor([d[1] for d in demonstration_data])

    # Create policy network
    policy_net = GraspPolicyNetwork(state_dim=states.shape[1], action_dim=actions.max().item() + 1)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

    # Supervised learning - minimize cross-entropy loss
    for epoch in range(1000):
        optimizer.zero_grad()
        predicted_actions = policy_net(states)
        loss = nn.CrossEntropyLoss()(predicted_actions, actions)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return policy_net
```

## Humanoid-Specific Manipulation Considerations

Humanoid robots have unique challenges in manipulation due to their bipedal form and need to maintain balance.

### Balance-Constraint Manipulation

When a humanoid robot manipulates objects, it must maintain its center of mass within the support polygon:

```python
def plan_balance_constrained_manipulation(robot_state, target_object, desired_grasp):
    """
    Plan manipulation while maintaining balance
    """
    # Calculate current center of mass
    current_com = calculate_center_of_mass(robot_state)

    # Determine support polygon based on foot positions
    support_polygon = calculate_support_polygon(robot_state['foot_positions'])

    # Plan manipulation trajectory that keeps CoM in support polygon
    manipulation_trajectory = []

    for t in range(len(desired_trajectory)):
        # Predict CoM position after manipulation action
        predicted_com = predict_com_position_after_action(
            current_com, desired_trajectory[t]
        )

        # Check if CoM will be in support polygon
        if not is_in_polygon(predicted_com[:2], support_polygon):
            # Adjust manipulation plan to maintain balance
            adjusted_action = adjust_for_balance(
                desired_trajectory[t], current_com, support_polygon
            )
            manipulation_trajectory.append(adjusted_action)
        else:
            manipulation_trajectory.append(desired_trajectory[t])

    return manipulation_trajectory
```

### Dual-Arm Coordination

Humanoid robots with two arms can perform more complex manipulation tasks through coordination:

```python
def coordinate_dual_arm_manipulation(object_pose, task_description):
    """
    Coordinate two arms for complex manipulation tasks
    """
    # Determine roles for each arm based on task
    if task_description['type'] == 'lifting_heavy_object':
        left_arm_role = 'support'
        right_arm_role = 'manipulate'
    elif task_description['type'] == 'assembling_parts':
        left_arm_role = 'hold_part1'
        right_arm_role = 'manipulate_part2'
    else:
        # Default: one arm manipulates while other provides balance
        left_arm_role = 'balance'
        right_arm_role = 'manipulate'

    # Plan coordinated motion
    left_arm_plan = plan_arm_motion_for_role(
        'left', left_arm_role, object_pose, task_description
    )
    right_arm_plan = plan_arm_motion_for_role(
        'right', right_arm_role, object_pose, task_description
    )

    # Synchronize the plans
    synchronized_plan = synchronize_arm_plans(left_arm_plan, right_arm_plan)

    return synchronized_plan
```

## Advanced Manipulation Techniques

### Tactile Feedback Integration

Tactile sensors provide crucial feedback for manipulation:

```python
class TactileFeedbackController:
    def __init__(self):
        self.slip_detection_threshold = 0.1
        self.force_distribution_threshold = 0.8

    def process_tactile_data(self, tactile_sensors):
        """
        Process tactile sensor data to detect slip and adjust grasp
        """
        slip_detected = False
        force_distribution = []

        for sensor in tactile_sensors:
            # Check for slip based on tactile patterns
            if sensor['slip_detector'] > self.slip_detection_threshold:
                slip_detected = True

            # Record force distribution
            force_distribution.append(sensor['force'])

        # Check if force distribution is balanced
        force_balance = np.std(force_distribution) / np.mean(force_distribution)
        force_distribution_ok = force_balance < self.force_distribution_threshold

        return {
            'slip_detected': slip_detected,
            'force_distribution_ok': force_distribution_ok,
            'force_values': force_distribution
        }

    def adjust_grasp_based_on_tactile(self, grasp_state, tactile_data):
        """
        Adjust grasp based on tactile feedback
        """
        if tactile_data['slip_detected']:
            # Increase grasp force gradually
            new_grasp_force = min(grasp_state['force'] * 1.1, grasp_state['max_force'])
            return {'force': new_grasp_force}
        elif not tactile_data['force_distribution_ok']:
            # Adjust finger positions for better force distribution
            adjustment = calculate_force_distribution_adjustment(
                tactile_data['force_values']
            )
            return {'finger_positions': adjustment}
        else:
            # No adjustment needed
            return {}
```

### Visual Servoing

Visual servoing uses visual feedback to guide manipulation:

```python
def visual_servoing_control(current_image, target_image, current_pose):
    """
    Use visual feedback to control manipulation motion
    """
    # Extract visual features
    current_features = extract_visual_features(current_image)
    target_features = extract_visual_features(target_image)

    # Calculate visual error
    visual_error = calculate_visual_error(current_features, target_features)

    # Map visual error to motion commands
    motion_command = map_visual_error_to_motion(visual_error, current_pose)

    return motion_command

def extract_visual_features(image):
    """Extract relevant visual features for servoing"""
    # This could include edge detection, corner detection, etc.
    # For simplicity, return object center and orientation
    object_center = find_object_center(image)
    object_orientation = estimate_object_orientation(image)

    return {'center': object_center, 'orientation': object_orientation}

def calculate_visual_error(current_features, target_features):
    """Calculate error between current and target visual features"""
    center_error = target_features['center'] - current_features['center']
    orientation_error = target_features['orientation'] - current_features['orientation']

    return np.concatenate([center_error, [orientation_error]])
```

## Troubleshooting and Safety

### Common Manipulation Issues

1. **Object Slippage**: Increase grasp force or use a different grasp strategy
2. **Collision During Approach**: Improve collision checking and planning
3. **Force Limit Exceeded**: Reduce approach speed or adjust force control parameters
4. **Balance Loss**: Restrict manipulation workspace or use balance controller

### Safety Considerations

```python
def safe_manipulation_wrapper(manipulation_function, safety_limits):
    """
    Wrapper to ensure safe manipulation execution
    """
    def safe_execute(*args, **kwargs):
        # Check safety conditions before execution
        if not check_safety_conditions(safety_limits):
            raise SafetyError("Safety conditions not met")

        try:
            # Execute manipulation
            result = manipulation_function(*args, **kwargs)

            # Monitor during execution
            while not is_execution_complete(result):
                if not check_safety_conditions(safety_limits):
                    emergency_stop()
                    raise SafetyError("Safety violation during execution")

            return result
        except Exception as e:
            emergency_stop()
            raise e

    return safe_execute
```

## Summary

Robotic manipulation and grasping represent one of the most challenging aspects of humanoid robotics. Success requires:

1. **Understanding of Grasp Types**: Selecting appropriate grippers and grasp strategies
2. **Force Control**: Managing contact forces for stable manipulation
3. **Planning**: Integrating task and motion planning for complex manipulation
4. **Learning**: Using machine learning to improve manipulation skills
5. **Balance Considerations**: Maintaining stability during manipulation
6. **Safety**: Ensuring safe and reliable operation

The field continues to evolve with advances in machine learning, tactile sensing, and control theory, enabling increasingly sophisticated manipulation capabilities in humanoid robots.