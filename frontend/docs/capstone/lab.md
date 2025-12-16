---
title: Capstone Lab - Autonomous Humanoid Robot Project
sidebar_position: 16
---

# Capstone Lab - Autonomous Humanoid Robot Project

## Lab Objectives

In this capstone lab, you will:
1. Integrate multiple humanoid robot subsystems into a cohesive autonomous system
2. Implement and test system-level control architectures
3. Develop and execute complex multi-modal tasks
4. Evaluate system performance and reliability in realistic scenarios
5. Optimize system performance for real-time operation

## Prerequisites

- Python 3.8+ installed
- NumPy, SciPy, Matplotlib, PyTorch, ROS 2 libraries
- Understanding of all previous chapters (kinematics, manipulation, HRI, etc.)
- Access to humanoid robot simulation or hardware (optional for full testing)

## Exercise 1: System Integration Framework

In this exercise, you'll build the foundational integration framework for the autonomous humanoid robot.

### Step 1: Create the System Architecture

```python
import numpy as np
import time
import threading
import queue
from datetime import datetime
import json

class RobotSubsystem:
    """Base class for robot subsystems"""

    def __init__(self, name):
        self.name = name
        self.status = "initialized"
        self.last_update = None
        self.data_queue = queue.Queue()
        self.command_queue = queue.Queue()

    def update(self):
        """Update subsystem state"""
        self.last_update = time.time()

    def get_status(self):
        """Get subsystem status"""
        return {
            'name': self.name,
            'status': self.status,
            'last_update': self.last_update
        }

class PerceptionSubsystem(RobotSubsystem):
    """Perception subsystem for sensing and understanding environment"""

    def __init__(self):
        super().__init__("Perception")
        self.sensors = {
            'camera': None,
            'lidar': None,
            'imu': None,
            'microphone': None
        }
        self.world_model = {}

    def update(self):
        """Update perception with sensor data"""
        super().update()

        # Simulate sensor data collection
        sensor_data = self._collect_sensor_data()

        # Process sensor data
        processed_data = self._process_sensor_data(sensor_data)

        # Update world model
        self._update_world_model(processed_data)

        # Add processed data to queue for other subsystems
        self.data_queue.put({
            'timestamp': time.time(),
            'data': processed_data,
            'world_model': self.world_model.copy()
        })

        self.status = "operational"

    def _collect_sensor_data(self):
        """Simulate collecting sensor data"""
        return {
            'camera': np.random.rand(480, 640, 3),  # Simulated camera image
            'lidar': np.random.rand(360) * 10,      # Simulated LIDAR ranges
            'imu': [0.1, 0.05, 9.8, 0.01, 0.02, 0.03],  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            'microphone': np.random.rand(1000)      # Simulated audio data
        }

    def _process_sensor_data(self, raw_data):
        """Process raw sensor data"""
        processed = {}

        # Process camera data (simplified)
        processed['objects'] = self._detect_objects(raw_data['camera'])
        processed['faces'] = self._detect_faces(raw_data['camera'])

        # Process LIDAR data (simplified)
        processed['obstacles'] = self._detect_obstacles(raw_data['lidar'])
        processed['free_space'] = self._detect_free_space(raw_data['lidar'])

        # Process IMU data
        processed['orientation'] = self._calculate_orientation(raw_data['imu'])
        processed['acceleration'] = raw_data['imu'][:3]

        # Process audio data
        processed['speech_detected'] = self._detect_speech(raw_data['microphone'])

        return processed

    def _detect_objects(self, camera_data):
        """Detect objects in camera image"""
        # Simplified object detection
        # In reality, this would use deep learning models
        if np.random.random() > 0.7:  # 30% chance of detecting an object
            return [{'type': 'box', 'position': [1.0, 0.5, 0.0], 'confidence': 0.8}]
        return []

    def _detect_faces(self, camera_data):
        """Detect faces in camera image"""
        # Simplified face detection
        if np.random.random() > 0.8:  # 20% chance of detecting a face
            return [{'position': [2.0, 0.0, 1.5], 'confidence': 0.9}]
        return []

    def _detect_obstacles(self, lidar_data):
        """Detect obstacles from LIDAR data"""
        # Simplified obstacle detection
        min_distance = 0.5  # meters
        obstacles = []

        for angle, distance in enumerate(lidar_data):
            if distance < min_distance:
                obstacles.append({
                    'angle': angle,
                    'distance': distance,
                    'position': [distance * np.cos(angle * np.pi / 180),
                                distance * np.sin(angle * np.pi / 180)]
                })

        return obstacles

    def _detect_free_space(self, lidar_data):
        """Detect free space from LIDAR data"""
        free_angles = []
        min_distance = 2.0  # meters

        for angle, distance in enumerate(lidar_data):
            if distance > min_distance:
                free_angles.append(angle)

        return free_angles

    def _calculate_orientation(self, imu_data):
        """Calculate orientation from IMU data"""
        # Simplified orientation calculation
        # In reality, this would use sensor fusion algorithms
        return [0.0, 0.0, 0.1]  # [roll, pitch, yaw]

    def _detect_speech(self, audio_data):
        """Detect speech in audio data"""
        # Simplified speech detection
        energy = np.mean(audio_data ** 2)
        return energy > 0.01  # Threshold for speech detection

    def _update_world_model(self, processed_data):
        """Update world model with processed data"""
        self.world_model['timestamp'] = time.time()
        self.world_model['objects'] = processed_data['objects']
        self.world_model['obstacles'] = processed_data['obstacles']
        self.world_model['faces'] = processed_data['faces']
        self.world_model['free_space'] = processed_data['free_space']

class NavigationSubsystem(RobotSubsystem):
    """Navigation subsystem for movement planning and execution"""

    def __init__(self):
        super().__init__("Navigation")
        self.current_pose = [0.0, 0.0, 0.0]  # [x, y, theta]
        self.path = []
        self.velocity = [0.0, 0.0, 0.0]  # [v_x, v_y, omega]
        self.goal = None

    def update(self):
        """Update navigation state"""
        super().update()

        # Execute navigation if there's a path
        if self.path:
            self._execute_path()

        # Update current pose based on velocity
        dt = 0.1  # 100ms time step
        self.current_pose[0] += self.velocity[0] * dt
        self.current_pose[1] += self.velocity[1] * dt
        self.current_pose[2] += self.velocity[2] * dt

        # Keep angle in [-pi, pi]
        self.current_pose[2] = ((self.current_pose[2] + np.pi) % (2 * np.pi)) - np.pi

        self.status = "operational"

    def set_goal(self, goal_pose):
        """Set navigation goal"""
        self.goal = goal_pose
        self._plan_path()

    def _plan_path(self):
        """Plan path to goal"""
        if self.goal is None:
            self.path = []
            return

        # Simplified path planning - in reality this would use A*, RRT, etc.
        start = self.current_pose[:2]
        goal = self.goal[:2]

        # Create a straight-line path (simplified)
        steps = 10
        dx = (goal[0] - start[0]) / steps
        dy = (goal[1] - start[1]) / steps

        self.path = []
        for i in range(steps + 1):
            x = start[0] + i * dx
            y = start[1] + i * dy
            self.path.append([x, y])

    def _execute_path(self):
        """Execute the planned path"""
        if not self.path:
            self.velocity = [0.0, 0.0, 0.0]
            return

        # Simple path following
        target = self.path[0]
        current = self.current_pose[:2]

        # Calculate direction to target
        direction = [target[0] - current[0], target[1] - current[1]]
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # Close enough to target point
            if len(self.path) > 1:
                self.path.pop(0)  # Move to next point
            else:
                self.path = []  # Reached goal
                self.velocity = [0.0, 0.0, 0.0]
                return

        # Normalize direction and set velocity
        if distance > 0:
            direction = [d / distance for d in direction]
            speed = min(0.5, distance)  # Max speed 0.5 m/s, slower when close
            self.velocity[0] = direction[0] * speed
            self.velocity[1] = direction[1] * speed
        else:
            self.velocity[0] = 0.0
            self.velocity[1] = 0.0

        # Calculate angular velocity to face direction of movement
        target_angle = np.arctan2(direction[1], direction[0])
        angle_diff = target_angle - self.current_pose[2]
        # Keep angle difference in [-pi, pi]
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        self.velocity[2] = angle_diff * 2.0  # Angular gain

class ManipulationSubsystem(RobotSubsystem):
    """Manipulation subsystem for object interaction"""

    def __init__(self):
        super().__init__("Manipulation")
        self.arm_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 7-DOF arm
        self.gripper_position = 0.0  # 0 (closed) to 1 (open)
        self.end_effector_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, roll, pitch, yaw]
        self.grasped_object = None

    def update(self):
        """Update manipulation state"""
        super().update()

        # Update end-effector pose based on joint angles (simplified)
        self._update_end_effector_pose()

        self.status = "operational"

    def _update_end_effector_pose(self):
        """Update end-effector pose based on joint angles"""
        # Simplified forward kinematics
        # In reality, this would use DH parameters or kinematics library
        self.end_effector_pose[0] = 0.5 + 0.3 * np.cos(self.arm_joints[0])  # x
        self.end_effector_pose[1] = 0.3 * np.sin(self.arm_joints[0])       # y
        self.end_effector_pose[2] = 0.8 + 0.2 * np.sin(self.arm_joints[1]) # z

    def move_arm_to(self, target_pose):
        """Move arm to target pose"""
        # Simplified inverse kinematics
        # In reality, this would use numerical methods or analytical solutions
        target_x, target_y, target_z = target_pose[:3]

        # Update joint angles to move toward target (simplified)
        current_x, current_y, current_z = self.end_effector_pose[:3]

        self.arm_joints[0] += 0.1 * np.arctan2(target_y - current_y, target_x - current_x)
        self.arm_joints[1] += 0.1 * (target_z - current_z)

        # Limit joint angles
        for i in range(len(self.arm_joints)):
            self.arm_joints[i] = np.clip(self.arm_joints[i], -np.pi, np.pi)

    def grasp_object(self, object_pose):
        """Grasp an object at given pose"""
        # Move to object location
        self.move_arm_to(object_pose)

        # Wait for arm to reach position (simplified)
        time.sleep(0.5)

        # Close gripper
        self.gripper_position = 0.0
        self.grasped_object = object_pose
        return True

    def release_object(self):
        """Release grasped object"""
        self.gripper_position = 1.0
        self.grasped_object = None

class HRISubsystem(RobotSubsystem):
    """Human-Robot Interaction subsystem"""

    def __init__(self):
        super().__init__("HRI")
        self.spoken_commands = []
        self.understood_intents = []
        self.social_behaviors = []
        self.current_interaction_state = "idle"

    def update(self):
        """Update HRI state"""
        super().update()

        # Process any new commands or interactions
        self._process_interactions()

        self.status = "operational"

    def _process_interactions(self):
        """Process human interactions"""
        # Check for new commands in queue
        while not self.command_queue.empty():
            command = self.command_queue.get()
            self._handle_command(command)

    def _handle_command(self, command):
        """Handle a command from external source"""
        if command['type'] == 'speech':
            intent = self._understand_speech(command['content'])
            self.understood_intents.append(intent)
            self.current_interaction_state = "processing_command"
        elif command['type'] == 'gesture':
            behavior = self._interpret_gesture(command['content'])
            self.social_behaviors.append(behavior)

    def _understand_speech(self, speech_text):
        """Understand speech command (simplified NLP)"""
        # Simplified natural language understanding
        speech_lower = speech_text.lower()

        if 'move to' in speech_lower or 'go to' in speech_lower:
            return {'action': 'navigate', 'target': self._extract_location(speech_text)}
        elif 'pick up' in speech_lower or 'grasp' in speech_lower:
            return {'action': 'manipulate', 'target': self._extract_object(speech_text)}
        elif 'hello' in speech_lower or 'hi' in speech_lower:
            return {'action': 'greet', 'target': 'user'}
        else:
            return {'action': 'unknown', 'content': speech_text}

    def _extract_location(self, text):
        """Extract location from speech text"""
        # Simplified location extraction
        if 'kitchen' in text:
            return [2.0, 1.0, 0.0]
        elif 'living room' in text:
            return [0.0, 2.0, 0.0]
        elif 'bedroom' in text:
            return [-1.0, -1.0, 0.0]
        else:
            return [1.0, 1.0, 0.0]  # Default location

    def _extract_object(self, text):
        """Extract object from speech text"""
        # Simplified object extraction
        if 'cup' in text:
            return {'type': 'cup', 'name': 'cup'}
        elif 'box' in text:
            return {'type': 'box', 'name': 'box'}
        else:
            return {'type': 'object', 'name': 'unknown_object'}

    def _interpret_gesture(self, gesture_data):
        """Interpret human gesture"""
        # Simplified gesture interpretation
        return {'type': 'acknowledged', 'confidence': 0.8}

# Test the subsystems
print("Testing individual subsystems...")

# Test Perception Subsystem
perception = PerceptionSubsystem()
perception.update()
perception_data = perception.data_queue.get()
print(f"Perception update: Found {len(perception_data['data']['objects'])} objects, "
      f"{len(perception_data['data']['obstacles'])} obstacles")

# Test Navigation Subsystem
navigation = NavigationSubsystem()
navigation.set_goal([2.0, 2.0, 0.0])
navigation.update()
print(f"Navigation: Current pose = {navigation.current_pose}, "
      f"Velocity = {navigation.velocity}")

# Test Manipulation Subsystem
manipulation = ManipulationSubsystem()
manipulation.move_arm_to([0.5, 0.3, 0.8])
print(f"Manipulation: End-effector pose = {manipulation.end_effector_pose}")

# Test HRI Subsystem
hri = HRISubsystem()
hri.command_queue.put({'type': 'speech', 'content': 'Please go to the kitchen'})
hri.update()
print(f"HRI: Understood intent = {hri.understood_intents[-1] if hri.understood_intents else 'None'}")
```

### Step 2: Create the Integration Framework

```python
class SystemIntegrationFramework:
    """Framework for integrating all robot subsystems"""

    def __init__(self):
        self.perception = PerceptionSubsystem()
        self.navigation = NavigationSubsystem()
        self.manipulation = ManipulationSubsystem()
        self.hri = HRISubsystem()

        self.world_model = {}
        self.task_queue = queue.Queue()
        self.event_bus = queue.Queue()
        self.system_state = "idle"

        # Communication channels between subsystems
        self.shared_data = {
            'world_model': {},
            'tasks': [],
            'events': []
        }

    def initialize_system(self):
        """Initialize the complete system"""
        print("Initializing autonomous humanoid robot system...")

        # Initialize all subsystems
        self.perception.status = "operational"
        self.navigation.status = "operational"
        self.manipulation.status = "operational"
        self.hri.status = "operational"

        # Set initial system state
        self.system_state = "ready"

        print("System initialization complete!")

    def run_system_cycle(self):
        """Run one complete cycle of the integrated system"""
        # Update all subsystems
        self._update_all_subsystems()

        # Process events and communications
        self._process_communications()

        # Update shared data
        self._update_shared_data()

        # Check system status
        self._check_system_status()

    def _update_all_subsystems(self):
        """Update all subsystems in the proper order"""
        # Update perception first (sensors)
        self.perception.update()

        # Process perception data
        if not self.perception.data_queue.empty():
            perception_data = self.perception.data_queue.get()
            self.world_model.update(perception_data['world_model'])

        # Update other subsystems
        self.navigation.update()
        self.manipulation.update()
        self.hri.update()

    def _process_communications(self):
        """Process communications between subsystems"""
        # Handle events from subsystems
        while not self.event_bus.empty():
            event = self.event_bus.get()
            self._handle_system_event(event)

        # Process any tasks
        while not self.task_queue.empty():
            task = self.task_queue.get()
            self._execute_task(task)

    def _update_shared_data(self):
        """Update shared data structures"""
        self.shared_data['world_model'] = self.world_model.copy()
        self.shared_data['system_state'] = self.system_state
        self.shared_data['subsystem_statuses'] = {
            'perception': self.perception.get_status(),
            'navigation': self.navigation.get_status(),
            'manipulation': self.manipulation.get_status(),
            'hri': self.hri.get_status()
        }

    def _check_system_status(self):
        """Check overall system status"""
        statuses = [
            self.perception.status,
            self.navigation.status,
            self.manipulation.status,
            self.hri.status
        ]

        if all(status == "operational" for status in statuses):
            self.system_state = "operational"
        elif any(status == "error" for status in statuses):
            self.system_state = "degraded"
        else:
            self.system_state = "ready"

    def _handle_system_event(self, event):
        """Handle a system event"""
        event_type = event.get('type', 'unknown')

        if event_type == 'object_detected':
            # Add object to world model
            obj = event.get('object', {})
            if 'objects' not in self.world_model:
                self.world_model['objects'] = []
            self.world_model['objects'].append(obj)

        elif event_type == 'goal_reached':
            print("Navigation goal reached!")

        elif event_type == 'grasp_success':
            print("Object grasped successfully!")

    def _execute_task(self, task):
        """Execute a high-level task"""
        task_type = task.get('type', 'unknown')

        if task_type == 'navigate_to':
            target = task.get('target', [0, 0, 0])
            self.navigation.set_goal(target)

        elif task_type == 'grasp_object':
            object_pose = task.get('object_pose', [0.5, 0.3, 0.8])
            success = self.manipulation.grasp_object(object_pose)
            event = {
                'type': 'grasp_success' if success else 'grasp_failed',
                'timestamp': time.time()
            }
            self.event_bus.put(event)

        elif task_type == 'process_command':
            command = task.get('command', {})
            self.hri.command_queue.put(command)

    def add_task(self, task):
        """Add a task to the system"""
        self.task_queue.put(task)

    def get_system_status(self):
        """Get overall system status"""
        return {
            'system_state': self.system_state,
            'world_model': self.world_model,
            'subsystem_statuses': self.shared_data['subsystem_statuses'],
            'task_queue_size': self.task_queue.qsize(),
            'event_queue_size': self.event_bus.qsize()
        }

# Test the integration framework
print("\nTesting system integration framework...")

framework = SystemIntegrationFramework()
framework.initialize_system()

# Add some tasks
framework.add_task({
    'type': 'navigate_to',
    'target': [1.0, 1.0, 0.0],
    'priority': 'high'
})

framework.add_task({
    'type': 'process_command',
    'command': {'type': 'speech', 'content': 'Please pick up the red cup'},
    'priority': 'medium'
})

# Run a few system cycles
for i in range(5):
    framework.run_system_cycle()
    status = framework.get_system_status()
    print(f"Cycle {i+1}: System state = {status['system_state']}, "
          f"Objects detected = {len(status['world_model'].get('objects', []))}")

print("Integration framework test completed!")
```

## Exercise 2: Task Planning and Execution

In this exercise, you'll implement a task planning and execution system that coordinates multiple subsystems.

### Step 1: Create Task Planning System

```python
class TaskPlanner:
    """System for planning and scheduling robot tasks"""

    def __init__(self):
        self.tasks = []
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_task = None

    def add_task(self, task_description, priority=5):
        """Add a task to the planning system"""
        task = {
            'id': len(self.tasks),
            'description': task_description,
            'priority': priority,
            'status': 'pending',
            'created_time': time.time(),
            'dependencies': task_description.get('dependencies', []),
            'resources_required': self._analyze_resource_requirements(task_description),
            'estimated_duration': self._estimate_duration(task_description)
        }

        self.tasks.append(task)
        self.task_queue.put((-priority, task['id'], task))  # Negative for max-heap

        print(f"Added task {task['id']}: {task_description['type']} with priority {priority}")

    def _analyze_resource_requirements(self, task_description):
        """Analyze what resources a task requires"""
        requirements = {
            'navigation': False,
            'manipulation': False,
            'perception': False,
            'hri': False,
            'computation': 'low'  # low, medium, high
        }

        task_type = task_description.get('type', '')

        if task_type in ['navigate', 'move_to', 'go_to']:
            requirements['navigation'] = True
            requirements['perception'] = True
            requirements['computation'] = 'medium'

        elif task_type in ['grasp', 'pick_up', 'manipulate', 'place']:
            requirements['manipulation'] = True
            requirements['perception'] = True
            requirements['navigation'] = True  # May need to navigate to object
            requirements['computation'] = 'high'

        elif task_type in ['greet', 'respond', 'converse']:
            requirements['hri'] = True
            requirements['perception'] = True
            requirements['computation'] = 'medium'

        return requirements

    def _estimate_duration(self, task_description):
        """Estimate how long a task will take"""
        task_type = task_description.get('type', '')

        if task_type == 'navigate':
            distance = np.linalg.norm(
                np.array(task_description.get('target', [0, 0, 0]))[:2]
            )
            return max(5, distance * 2)  # 2 seconds per meter, minimum 5 seconds

        elif task_type == 'grasp':
            return 10  # 10 seconds for grasping

        elif task_type == 'greet':
            return 5  # 5 seconds for greeting

        else:
            return 15  # Default estimate

    def get_next_task(self):
        """Get the next task to execute based on priority and dependencies"""
        if self.task_queue.empty():
            return None

        # Check if current task is still running
        if self.current_task and self.current_task['status'] == 'executing':
            return self.current_task

        # Get highest priority task that has satisfied dependencies
        try:
            priority, task_id, task = self.task_queue.get_nowait()

            # Check dependencies
            if self._dependencies_satisfied(task):
                self.current_task = task
                self.current_task['status'] = 'executing'
                self.current_task['start_time'] = time.time()
                return self.current_task
            else:
                # Put task back if dependencies not satisfied
                self.task_queue.put((priority, task_id, task))
                return None
        except queue.Empty:
            return None

    def _dependencies_satisfied(self, task):
        """Check if task dependencies are satisfied"""
        for dep_id in task['dependencies']:
            dep_task = next((t for t in self.completed_tasks if t['id'] == dep_id), None)
            if dep_task is None:
                return False
        return True

    def mark_task_completed(self, task_id, success=True, result=None):
        """Mark a task as completed"""
        task = next((t for t in self.tasks if t['id'] == task_id), None)
        if task:
            task['status'] = 'completed' if success else 'failed'
            task['end_time'] = time.time()
            task['result'] = result
            task['duration'] = task['end_time'] - task['start_time']

            if success:
                self.completed_tasks.append(task)
            else:
                self.failed_tasks.append(task)

            # Clear current task if it matches
            if self.current_task and self.current_task['id'] == task_id:
                self.current_task = None

            print(f"Task {task_id} marked as {'completed' if success else 'failed'}")

    def get_system_load(self):
        """Get current system load estimate"""
        active_tasks = [t for t in self.tasks if t['status'] == 'executing']
        total_priority = sum(t['priority'] for t in active_tasks)

        return {
            'active_tasks': len(active_tasks),
            'total_priority': total_priority,
            'queue_size': self.task_queue.qsize(),
            'completed_today': len(self.completed_tasks)
        }

class TaskExecutor:
    """Execute tasks using robot subsystems"""

    def __init__(self, integration_framework):
        self.framework = integration_framework
        self.planner = TaskPlanner()
        self.active_task = None

    def execute_task(self, task):
        """Execute a single task"""
        if not task:
            return False

        task_type = task['description'].get('type', 'unknown')

        print(f"Executing task {task['id']}: {task_type}")

        if task_type == 'navigate':
            return self._execute_navigation_task(task)
        elif task_type == 'grasp':
            return self._execute_grasp_task(task)
        elif task_type == 'greet':
            return self._execute_greet_task(task)
        elif task_type == 'complex_sequence':
            return self._execute_complex_task(task)
        else:
            print(f"Unknown task type: {task_type}")
            return False

    def _execute_navigation_task(self, task):
        """Execute navigation task"""
        target = task['description'].get('target', [0, 0, 0])

        # Add navigation goal to framework
        nav_task = {
            'type': 'navigate_to',
            'target': target
        }
        self.framework.add_task(nav_task)

        # Simulate navigation execution
        start_time = time.time()
        timeout = task['estimated_duration']

        while time.time() - start_time < timeout:
            self.framework.run_system_cycle()

            # Check if navigation is complete
            current_pos = self.framework.navigation.current_pose
            distance_to_goal = np.linalg.norm(
                np.array(current_pos[:2]) - np.array(target[:2])
            )

            if distance_to_goal < 0.2:  # Within 20cm of goal
                print(f"Navigation task completed, reached {target}")
                return True

            time.sleep(0.1)  # 100ms update rate

        print("Navigation task timed out")
        return False

    def _execute_grasp_task(self, task):
        """Execute grasp task"""
        object_pose = task['description'].get('object_pose', [0.5, 0.3, 0.8])

        # Navigate to object first
        nav_to_obj = {
            'type': 'navigate_to',
            'target': [object_pose[0], object_pose[1], 0.0]  # Navigate to object xy, z=0
        }
        self.framework.add_task(nav_to_obj)

        # Wait for navigation to complete (simplified)
        time.sleep(2)

        # Execute grasp
        success = self.framework.manipulation.grasp_object(object_pose)

        if success:
            print(f"Grasp task completed, object at {object_pose}")
        else:
            print("Grasp task failed")

        return success

    def _execute_greet_task(self, task):
        """Execute greeting task"""
        user_location = task['description'].get('user_location', [1.0, 0.0, 1.5])

        # Navigate toward user
        nav_task = {
            'type': 'navigate_to',
            'target': [user_location[0], user_location[1], 0.0]
        }
        self.framework.add_task(nav_task)

        # Wait for navigation (simplified)
        time.sleep(1)

        # Add HRI command
        hri_task = {
            'type': 'process_command',
            'command': {
                'type': 'speech',
                'content': 'Hello! How can I assist you today?'
            }
        }
        self.framework.add_task(hri_task)

        print("Greeting task completed")
        return True

    def _execute_complex_task(self, task):
        """Execute a complex task with multiple steps"""
        steps = task['description'].get('steps', [])

        for step in steps:
            step_task = {
                'type': step['type'],
                'description': step
            }

            # Create temporary task for step
            temp_task = {
                'id': -1,  # Temporary ID
                'description': step,
                'status': 'pending',
                'estimated_duration': 10  # Default duration
            }

            step_success = self.execute_task(temp_task)
            if not step_success:
                print(f"Complex task failed at step: {step['type']}")
                return False

        print("Complex task completed successfully")
        return True

# Test the task planning and execution system
print("\nTesting task planning and execution...")

task_executor = TaskExecutor(framework)

# Add some tasks to the planner
task_executor.planner.add_task({
    'type': 'navigate',
    'target': [2.0, 1.0, 0.0],
    'description': 'Move to kitchen area'
}, priority=8)

task_executor.planner.add_task({
    'type': 'greet',
    'user_location': [1.5, 0.5, 1.5],
    'description': 'Greet the person in the living room'
}, priority=6)

task_executor.planner.add_task({
    'type': 'grasp',
    'object_pose': [0.8, 0.6, 0.8],
    'description': 'Pick up the cup on the table'
}, priority=10)

# Execute tasks
for i in range(3):
    task = task_executor.planner.get_next_task()
    if task:
        success = task_executor.execute_task(task)
        task_executor.planner.mark_task_completed(task['id'], success)
    else:
        print("No tasks available")
        break

print(f"Task execution completed. System load: {task_executor.planner.get_system_load()}")
```

## Exercise 3: Real-time Control and Coordination

In this exercise, you'll implement real-time control and coordination mechanisms for the integrated system.

### Step 1: Create Real-time Control System

```python
import threading
import time
from collections import deque

class RealTimeController:
    """Real-time controller for coordinated robot operation"""

    def __init__(self, integration_framework):
        self.framework = integration_framework
        self.control_frequency = 100  # Hz
        self.dt = 1.0 / self.control_frequency
        self.running = False
        self.control_thread = None

        # Control loops with different frequencies
        self.control_loops = {
            'high_freq': {'frequency': 100, 'last_run': 0},      # Joint control, balance
            'medium_freq': {'frequency': 50, 'last_run': 0},     # Trajectory following
            'low_freq': {'frequency': 10, 'last_run': 0}         # Task planning, high-level decisions
        }

        # Safety and emergency systems
        self.safety_monitor = SafetyMonitor()
        self.emergency_stop = False

    def start_control_loop(self):
        """Start the real-time control loop in a separate thread"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()
        print("Real-time control loop started")

    def stop_control_loop(self):
        """Stop the real-time control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()
        print("Real-time control loop stopped")

    def _control_loop(self):
        """Main real-time control loop"""
        last_time = time.time()

        while self.running and not self.emergency_stop:
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= self.dt:
                # Run the integrated system cycle
                self._run_control_cycle(current_time)

                # Update timing
                last_time = current_time
            else:
                # Sleep to maintain timing (in a real system, this might use more precise timing)
                time.sleep(max(0, self.dt - elapsed))

    def _run_control_cycle(self, current_time):
        """Run one control cycle"""
        # Check for emergency conditions
        if self.safety_monitor.check_emergency_conditions():
            self.emergency_stop = True
            print("EMERGENCY STOP ACTIVATED")
            return

        # Run different control loops based on their frequencies
        self._run_high_frequency_controls(current_time)
        self._run_medium_frequency_controls(current_time)
        self._run_low_frequency_controls(current_time)

        # Update the integration framework
        self.framework.run_system_cycle()

    def _run_high_frequency_controls(self, current_time):
        """Run high-frequency control (100Hz)"""
        loop_info = self.control_loops['high_freq']
        if (current_time - loop_info['last_run']) >= (1.0 / loop_info['frequency']):
            # Joint position/velocity control
            self._update_joint_control()

            # Balance control
            self._update_balance_control()

            # Collision avoidance
            self._update_collision_avoidance()

            loop_info['last_run'] = current_time

    def _run_medium_frequency_controls(self, current_time):
        """Run medium-frequency control (50Hz)"""
        loop_info = self.control_loops['medium_freq']
        if (current_time - loop_info['last_run']) >= (1.0 / loop_info['frequency']):
            # Trajectory following
            self._update_trajectory_following()

            # Manipulation control
            self._update_manipulation_control()

            loop_info['last_run'] = current_time

    def _run_low_frequency_controls(self, current_time):
        """Run low-frequency control (10Hz)"""
        loop_info = self.control_loops['low_freq']
        if (current_time - loop_info['last_run']) >= (1.0 / loop_info['frequency']):
            # Task planning and scheduling
            self._update_task_planning()

            # Path planning
            self._update_path_planning()

            # System monitoring
            self._update_system_monitoring()

            loop_info['last_run'] = current_time

    def _update_joint_control(self):
        """Update joint position/velocity control"""
        # In a real system, this would interface with robot hardware
        # For simulation, we'll just update joint positions based on desired trajectories
        pass

    def _update_balance_control(self):
        """Update balance control"""
        # Monitor center of mass and adjust for stability
        # This would interface with balance control algorithms
        pass

    def _update_collision_avoidance(self):
        """Update collision avoidance"""
        # Check for potential collisions using sensor data
        # This would use distance sensors, LIDAR, etc.
        pass

    def _update_trajectory_following(self):
        """Update trajectory following"""
        # Follow planned trajectories for navigation and manipulation
        pass

    def _update_manipulation_control(self):
        """Update manipulation control"""
        # Control arm movements and gripper actions
        pass

    def _update_task_planning(self):
        """Update task planning"""
        # Check for new tasks and schedule them
        pass

    def _update_path_planning(self):
        """Update path planning"""
        # Recalculate paths based on updated world model
        pass

    def _update_system_monitoring(self):
        """Update system monitoring"""
        # Monitor system health and performance
        pass

class SafetyMonitor:
    """Monitor system for safety conditions"""

    def __init__(self):
        self.safety_limits = {
            'joint_position': {'min': -3.0, 'max': 3.0},  # radians
            'joint_velocity': {'max': 5.0},               # rad/s
            'torque': {'max': 100.0},                     # Nm
            'acceleration': {'max': 9.81},                # m/s²
            'temperature': {'max': 80.0},                 # °C
            'current': {'max': 10.0}                      # A
        }
        self.emergency_conditions = []
        self.safety_history = deque(maxlen=100)

    def check_emergency_conditions(self):
        """Check for emergency safety conditions"""
        # In a real system, this would check actual sensor readings
        # For simulation, we'll create some conditions randomly

        # Simulate some sensor readings
        joint_angles = np.random.uniform(-2, 2, 7)  # 7 DOF arm
        joint_velocities = np.random.uniform(-3, 3, 7)
        temperatures = np.random.uniform(20, 70, 5)  # 5 motors

        # Check limits
        emergency = False

        # Check joint positions
        if np.any(joint_angles < self.safety_limits['joint_position']['min']) or \
           np.any(joint_angles > self.safety_limits['joint_position']['max']):
            self.emergency_conditions.append('joint_position_violation')
            emergency = True

        # Check joint velocities
        if np.any(np.abs(joint_velocities) > self.safety_limits['joint_velocity']['max']):
            self.emergency_conditions.append('joint_velocity_violation')
            emergency = True

        # Check temperatures
        if np.any(temperatures > self.safety_limits['temperature']['max']):
            self.emergency_conditions.append('overheating')
            emergency = True

        # Record safety status
        safety_status = {
            'timestamp': time.time(),
            'emergency': emergency,
            'conditions': self.emergency_conditions.copy()
        }
        self.safety_history.append(safety_status)

        return emergency

    def get_safety_status(self):
        """Get current safety status"""
        return {
            'emergency_conditions': self.emergency_conditions,
            'safety_history': list(self.safety_history),
            'last_check': time.time()
        }

# Test the real-time control system
print("\nTesting real-time control system...")

real_time_ctrl = RealTimeController(framework)

# Start the control loop
real_time_ctrl.start_control_loop()

# Let it run for a few seconds
time.sleep(2)

# Check safety status
safety_status = real_time_ctrl.safety_monitor.get_safety_status()
print(f"Safety status: Emergency conditions = {len(safety_status['emergency_conditions'])}")

# Stop the control loop
real_time_ctrl.stop_control_loop()
print("Real-time control test completed!")
```

## Exercise 4: Performance Optimization and Testing

In this exercise, you'll implement performance optimization and testing for the integrated system.

### Step 1: Create Performance Analysis Tools

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import defaultdict
import time

class PerformanceAnalyzer:
    """Analyze performance of the integrated robot system"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.end_times = {}
        self.benchmark_results = {}

    def start_measurement(self, metric_name):
        """Start measuring a performance metric"""
        self.start_times[metric_name] = time.time()

    def end_measurement(self, metric_name):
        """End measuring a performance metric"""
        if metric_name in self.start_times:
            duration = time.time() - self.start_times[metric_name]
            self.metrics[metric_name].append(duration)
            return duration
        return None

    def record_metric(self, metric_name, value):
        """Record a specific metric value"""
        self.metrics[metric_name].append(value)

    def get_statistics(self, metric_name):
        """Get statistics for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None

        values = self.metrics[metric_name]
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }

    def benchmark_subsystem(self, subsystem, test_iterations=100):
        """Benchmark a specific subsystem"""
        test_name = f"{subsystem.name}_benchmark"

        self.start_measurement(test_name)

        start_time = time.time()
        for i in range(test_iterations):
            subsystem.update()

        total_time = time.time() - start_time
        avg_time = total_time / test_iterations

        self.benchmark_results[test_name] = {
            'total_time': total_time,
            'avg_time': avg_time,
            'iterations': test_iterations,
            'rate': test_iterations / total_time  # iterations per second
        }

        return self.benchmark_results[test_name]

    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        report = {
            'timestamp': time.time(),
            'system_metrics': {},
            'subsystem_benchmarks': self.benchmark_results,
            'resource_usage': self._get_resource_usage()
        }

        # Calculate statistics for each metric
        for metric_name in self.metrics:
            report['system_metrics'][metric_name] = self.get_statistics(metric_name)

        return report

    def _get_resource_usage(self):
        """Get current resource usage (simulated)"""
        return {
            'cpu_usage': np.random.uniform(30, 70),      # %
            'memory_usage': np.random.uniform(40, 80),   # %
            'disk_io': np.random.uniform(1, 10),         # MB/s
            'network_usage': np.random.uniform(0, 5)     # MB/s
        }

    def plot_performance_metrics(self):
        """Plot performance metrics"""
        if not self.metrics:
            print("No metrics to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Robot System Performance Metrics')

        # Plot 1: Execution times
        ax1 = axes[0, 0]
        for metric_name, values in self.metrics.items():
            if values:
                ax1.plot(values[:50], label=metric_name, alpha=0.7)  # Show first 50 values
        ax1.set_title('Execution Times')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time (s)')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Metric distributions
        ax2 = axes[0, 1]
        for metric_name, values in self.metrics.items():
            if values and len(values) > 1:
                ax2.hist(values, bins=20, alpha=0.5, label=metric_name, density=True)
        ax2.set_title('Metric Distributions')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Resource usage (simulated)
        ax3 = axes[1, 0]
        resources = ['cpu', 'memory', 'disk_io', 'network']
        usage = [np.random.uniform(30, 80) for _ in resources]
        ax3.bar(resources, usage)
        ax3.set_title('Resource Usage (%)')
        ax3.set_ylabel('Usage (%)')
        ax3.grid(True, axis='y')

        # Plot 4: Benchmark results
        ax4 = axes[1, 1]
        if self.benchmark_results:
            subsystems = list(self.benchmark_results.keys())
            rates = [self.benchmark_results[sub]['rate'] for sub in subsystems]
            ax4.bar(subsystems, rates)
            ax4.set_title('Subsystem Performance (ops/sec)')
            ax4.set_ylabel('Operations per Second')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No benchmark data',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
        ax4.grid(True, axis='y')

        plt.tight_layout()
        plt.show()

class SystemTester:
    """Comprehensive testing framework for the integrated system"""

    def __init__(self, integration_framework, performance_analyzer):
        self.framework = integration_framework
        self.analyzer = performance_analyzer
        self.test_results = {}

    def run_comprehensive_test(self, duration=30):
        """Run comprehensive system test"""
        print(f"Starting comprehensive system test for {duration} seconds...")

        start_time = time.time()
        test_start = time.time()

        # Initialize test
        self.framework.initialize_system()

        # Run test loop
        iteration_count = 0
        while time.time() - start_time < duration:
            # Start performance measurement
            self.analyzer.start_measurement('system_cycle_time')

            # Run system cycle
            self.framework.run_system_cycle()

            # End performance measurement
            cycle_time = self.analyzer.end_measurement('system_cycle_time')

            # Record additional metrics
            status = self.framework.get_system_status()
            self.analyzer.record_metric('system_stability',
                                      1.0 if status['system_state'] == 'operational' else 0.0)

            # Add some tasks periodically
            if iteration_count % 50 == 0:  # Every 50 iterations
                self._add_test_task()

            iteration_count += 1

            # Small delay to prevent overwhelming the system
            time.sleep(0.01)

        test_duration = time.time() - test_start
        iterations_per_second = iteration_count / test_duration

        self.analyzer.record_metric('iterations_per_second', iterations_per_second)

        print(f"Test completed: {iteration_count} iterations in {test_duration:.2f}s "
              f"({iterations_per_second:.2f} ips)")

        # Generate test results
        self.test_results = {
            'duration': test_duration,
            'iterations': iteration_count,
            'ips': iterations_per_second,
            'performance_report': self.analyzer.generate_performance_report()
        }

        return self.test_results

    def _add_test_task(self):
        """Add a random test task"""
        task_types = ['navigate', 'greet', 'monitor']
        task_type = np.random.choice(task_types)

        if task_type == 'navigate':
            target = [np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0]
            self.framework.add_task({
                'type': 'navigate_to',
                'target': target
            })
        elif task_type == 'greet':
            self.framework.add_task({
                'type': 'process_command',
                'command': {
                    'type': 'speech',
                    'content': 'System status report: All systems nominal.'
                }
            })
        # 'monitor' tasks just let the system continue monitoring

    def run_stress_test(self, duration=60):
        """Run stress test with high load"""
        print(f"Starting stress test for {duration} seconds...")

        start_time = time.time()
        task_counter = 0

        while time.time() - start_time < duration:
            # Add multiple tasks per cycle to stress the system
            for _ in range(3):  # Add 3 tasks per cycle
                self._add_stress_task()
                task_counter += 1

            # Run system cycle
            self.analyzer.start_measurement('stress_cycle_time')
            self.framework.run_system_cycle()
            self.analyzer.end_measurement('stress_cycle_time')

            # Small delay
            time.sleep(0.005)

        print(f"Stress test completed: {task_counter} tasks processed")

        stress_results = {
            'tasks_processed': task_counter,
            'stress_performance': self.analyzer.get_statistics('stress_cycle_time')
        }

        return stress_results

    def _add_stress_task(self):
        """Add a task for stress testing"""
        task_type = np.random.choice(['navigate', 'grasp', 'hri'])

        if task_type == 'navigate':
            self.framework.add_task({
                'type': 'navigate_to',
                'target': [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0]
            })
        elif task_type == 'grasp':
            self.framework.add_task({
                'type': 'process_command',
                'command': {
                    'type': 'speech',
                    'content': f'Grasp object {np.random.randint(1, 100)}'
                }
            })
        elif task_type == 'hri':
            self.framework.add_task({
                'type': 'process_command',
                'command': {
                    'type': 'speech',
                    'content': f'Hello user {np.random.randint(1, 10)}, how are you?'
                }
            })

    def run_reliability_test(self, duration=300):  # 5 minutes
        """Run long-term reliability test"""
        print(f"Starting reliability test for {duration} seconds...")

        start_time = time.time()
        error_count = 0
        successful_cycles = 0

        while time.time() - start_time < duration:
            try:
                # Run system cycle
                self.framework.run_system_cycle()

                # Check for errors in system status
                status = self.framework.get_system_status()
                if status['system_state'] != 'operational':
                    error_count += 1
                else:
                    successful_cycles += 1

            except Exception as e:
                error_count += 1
                print(f"Error during reliability test: {e}")

            # Record reliability metric
            self.analyzer.record_metric('reliability',
                                      1.0 if error_count == 0 else 0.0)

            time.sleep(0.1)  # 10Hz update rate

        reliability_rate = successful_cycles / (successful_cycles + error_count) if (successful_cycles + error_count) > 0 else 0

        print(f"Reliability test completed: {reliability_rate:.2%} success rate")

        return {
            'total_cycles': successful_cycles + error_count,
            'successful_cycles': successful_cycles,
            'error_count': error_count,
            'reliability_rate': reliability_rate
        }

# Test the performance analysis and testing framework
print("\nTesting performance analysis and system testing...")

analyzer = PerformanceAnalyzer()
tester = SystemTester(framework, analyzer)

# Run comprehensive test
comprehensive_results = tester.run_comprehensive_test(duration=15)  # 15 seconds for demo
print(f"Comprehensive test completed. IPS: {comprehensive_results['ips']:.2f}")

# Run stress test
stress_results = tester.run_stress_test(duration=10)  # 10 seconds for demo
print(f"Stress test completed. Tasks processed: {stress_results['tasks_processed']}")

# Run reliability test (shortened for demo)
reliability_results = tester.run_reliability_test(duration=20)  # 20 seconds for demo
print(f"Reliability test completed. Success rate: {reliability_results['reliability_rate']:.2%}")

# Generate performance report
report = analyzer.generate_performance_report()
print(f"Performance report generated with {len(report['system_metrics'])} metrics")

# Show some statistics
cycle_stats = analyzer.get_statistics('system_cycle_time')
if cycle_stats:
    print(f"System cycle time: mean={cycle_stats['mean']:.4f}s, "
          f"std={cycle_stats['std']:.4f}s, "
          f"min={cycle_stats['min']:.4f}s, "
          f"max={cycle_stats['max']:.4f}s")
```

## Exercise 5: Scenario-Based Integration Testing

In this exercise, you'll create and execute scenario-based tests that demonstrate the complete integrated system.

### Step 1: Create Scenario Testing Framework

```python
class ScenarioBasedTester:
    """Framework for scenario-based testing of integrated systems"""

    def __init__(self, integration_framework, performance_analyzer):
        self.framework = integration_framework
        self.analyzer = performance_analyzer
        self.scenarios = {}

    def register_scenario(self, name, scenario_function):
        """Register a scenario testing function"""
        self.scenarios[name] = scenario_function

    def run_scenario(self, scenario_name, **kwargs):
        """Run a specific scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not registered")

        print(f"Running scenario: {scenario_name}")
        start_time = time.time()

        # Run the scenario
        results = self.scenarios[scenario_name](self.framework, self.analyzer, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        scenario_results = {
            'name': scenario_name,
            'execution_time': execution_time,
            'results': results,
            'timestamp': start_time
        }

        print(f"Scenario {scenario_name} completed in {execution_time:.2f}s")
        return scenario_results

    def run_all_scenarios(self):
        """Run all registered scenarios"""
        all_results = {}
        for scenario_name in self.scenarios:
            try:
                all_results[scenario_name] = self.run_scenario(scenario_name)
            except Exception as e:
                print(f"Error running scenario {scenario_name}: {e}")
                all_results[scenario_name] = {
                    'name': scenario_name,
                    'error': str(e),
                    'timestamp': time.time()
                }

        return all_results

# Create specific scenarios
def create_assistive_living_scenario():
    """Create assistive living scenario"""
    def scenario(framework, analyzer, **kwargs):
        results = {'tasks_completed': 0, 'tasks_failed': 0, 'details': []}

        # Task 1: Greet user
        analyzer.start_measurement('greeting_task')
        framework.add_task({
            'type': 'process_command',
            'command': {'type': 'speech', 'content': 'Good morning! How can I assist you today?'}
        })

        # Simulate task execution
        time.sleep(1)
        results['tasks_completed'] += 1
        results['details'].append('Greeting task completed')
        greeting_time = analyzer.end_measurement('greeting_task')

        # Task 2: Navigate to kitchen
        analyzer.start_measurement('navigation_task')
        framework.add_task({
            'type': 'navigate_to',
            'target': [2.0, 1.0, 0.0]
        })

        # Simulate navigation
        for _ in range(20):  # Simulate 2 seconds of navigation
            framework.run_system_cycle()
            time.sleep(0.1)

        results['tasks_completed'] += 1
        results['details'].append('Navigation task completed')
        navigation_time = analyzer.end_measurement('navigation_task')

        # Task 3: Fetch water
        analyzer.start_measurement('manipulation_task')
        framework.add_task({
            'type': 'process_command',
            'command': {'type': 'speech', 'content': 'Fetching water for you.'}
        })

        time.sleep(1)
        results['tasks_completed'] += 1
        results['details'].append('Water fetching task completed')
        manipulation_time = analyzer.end_measurement('manipulation_task')

        return {
            'greeting_time': greeting_time,
            'navigation_time': navigation_time,
            'manipulation_time': manipulation_time,
            **results
        }

    return scenario

def create_warehouse_assistant_scenario():
    """Create warehouse assistant scenario"""
    def scenario(framework, analyzer, **kwargs):
        results = {'tasks_completed': 0, 'tasks_failed': 0, 'details': []}

        # Task 1: Scan inventory
        analyzer.start_measurement('inventory_scan')
        for i in range(5):  # Simulate scanning 5 items
            framework.run_system_cycle()
            time.sleep(0.2)

        results['tasks_completed'] += 1
        results['details'].append('Inventory scanning completed')
        scan_time = analyzer.end_measurement('inventory_scan')

        # Task 2: Transport item
        analyzer.start_measurement('transport_task')
        framework.add_task({
            'type': 'navigate_to',
            'target': [5.0, 3.0, 0.0]
        })

        # Simulate transport
        for _ in range(30):  # Simulate 3 seconds of transport
            framework.run_system_cycle()
            time.sleep(0.1)

        results['tasks_completed'] += 1
        results['details'].append('Item transport completed')
        transport_time = analyzer.end_measurement('transport_task')

        # Task 3: Update inventory system
        analyzer.start_measurement('inventory_update')
        framework.add_task({
            'type': 'process_command',
            'command': {'type': 'speech', 'content': 'Inventory updated successfully.'}
        })

        time.sleep(0.5)
        results['tasks_completed'] += 1
        results['details'].append('Inventory update completed')
        update_time = analyzer.end_measurement('inventory_update')

        return {
            'scan_time': scan_time,
            'transport_time': transport_time,
            'update_time': update_time,
            **results
        }

    return scenario

def create_search_rescue_scenario():
    """Create search and rescue scenario"""
    def scenario(framework, analyzer, **kwargs):
        results = {'tasks_completed': 0, 'tasks_failed': 0, 'details': []}

        # Task 1: Area mapping
        analyzer.start_measurement('mapping_task')
        for i in range(10):  # Simulate mapping area
            framework.run_system_cycle()
            time.sleep(0.3)

        results['tasks_completed'] += 1
        results['details'].append('Area mapping completed')
        mapping_time = analyzer.end_measurement('mapping_task')

        # Task 2: Victim detection
        analyzer.start_measurement('detection_task')
        framework.add_task({
            'type': 'process_command',
            'command': {'type': 'speech', 'content': 'Person detected. Establishing communication.'}
        })

        time.sleep(1)
        results['tasks_completed'] += 1
        results['details'].append('Victim detection completed')
        detection_time = analyzer.end_measurement('detection_task')

        # Task 3: Navigate to victim
        analyzer.start_measurement('rescue_navigation')
        framework.add_task({
            'type': 'navigate_to',
            'target': [-1.0, 2.0, 0.0]
        })

        # Simulate navigation to victim
        for _ in range(25):  # Simulate 2.5 seconds of navigation
            framework.run_system_cycle()
            time.sleep(0.1)

        results['tasks_completed'] += 1
        results['details'].append('Navigation to victim completed')
        rescue_nav_time = analyzer.end_measurement('rescue_navigation')

        # Task 4: Provide assistance
        analyzer.start_measurement('assistance_task')
        framework.add_task({
            'type': 'process_command',
            'command': {'type': 'speech', 'content': 'Help is on the way. Stay calm.'}
        })

        time.sleep(1.5)
        results['tasks_completed'] += 1
        results['details'].append('Assistance provided')
        assistance_time = analyzer.end_measurement('assistance_task')

        return {
            'mapping_time': mapping_time,
            'detection_time': detection_time,
            'rescue_navigation_time': rescue_nav_time,
            'assistance_time': assistance_time,
            **results
        }

    return scenario

# Create and register scenarios
scenario_tester = ScenarioBasedTester(framework, analyzer)

# Register scenarios
scenario_tester.register_scenario('assistive_living', create_assistive_living_scenario())
scenario_tester.register_scenario('warehouse_assistant', create_warehouse_assistant_scenario())
scenario_tester.register_scenario('search_rescue', create_search_rescue_scenario())

# Run scenarios
print("\nRunning scenario-based integration tests...")

# Run assistive living scenario
assistive_results = scenario_tester.run_scenario('assistive_living')
print(f"Assistive living scenario: {assistive_results['results']['tasks_completed']} tasks completed")

# Run warehouse assistant scenario
warehouse_results = scenario_tester.run_scenario('warehouse_assistant')
print(f"Warehouse assistant scenario: {warehouse_results['results']['tasks_completed']} tasks completed")

# Run search and rescue scenario
rescue_results = scenario_tester.run_scenario('search_rescue')
print(f"Search and rescue scenario: {rescue_results['results']['tasks_completed']} tasks completed")

# Run all scenarios
all_scenario_results = scenario_tester.run_all_scenarios()
print(f"All scenarios completed: {len(all_scenario_results)} total")

# Generate final performance report
final_report = analyzer.generate_performance_report()
print(f"\nFinal Performance Report:")
print(f"- Total metrics collected: {len(final_report['system_metrics'])}")
print(f"- Subsystem benchmarks: {len(final_report['subsystem_benchmarks'])}")
print(f"- Test duration: {final_report['timestamp'] - all_scenario_results['assistive_living']['timestamp']:.2f}s")

# Display some key metrics
if 'system_cycle_time' in final_report['system_metrics']:
    cycle_stats = final_report['system_metrics']['system_cycle_time']
    print(f"- Average system cycle time: {cycle_stats['mean']:.4f}s")
    print(f"- System cycle time std: {cycle_stats['std']:.4f}s")
```

## Assessment Questions

1. **System Integration**: Explain how the different subsystems (perception, navigation, manipulation, HRI) coordinate to achieve complex tasks. What are the key interfaces and data flows between them?

2. **Task Planning**: Describe the task planning and execution architecture. How does the system handle task dependencies and resource conflicts?

3. **Real-time Control**: What challenges arise when implementing real-time control for integrated systems? How does the multi-frequency control approach address these challenges?

4. **Performance Optimization**: What metrics are most important for evaluating autonomous humanoid robot performance? How would you prioritize optimization efforts?

5. **Scenario Testing**: Why is scenario-based testing important for autonomous systems? What types of scenarios would you add to better test the system?

## Troubleshooting Guide

### Common Integration Issues

1. **Timing Problems**: Subsystems not updating at appropriate frequencies
   - Solution: Implement proper scheduling and timing mechanisms

2. **Resource Conflicts**: Multiple subsystems competing for same resources
   - Solution: Implement resource allocation and conflict resolution

3. **Data Synchronization**: Different subsystems operating on outdated information
   - Solution: Implement proper data synchronization and versioning

4. **Communication Failures**: Subsystems unable to communicate effectively
   - Solution: Implement robust communication protocols and error handling

### Debugging Strategies

1. **Modular Testing**: Test each subsystem independently before integration
2. **Logging**: Implement comprehensive logging for debugging
3. **Visualization**: Use visualization tools to understand system state
4. **Performance Monitoring**: Continuously monitor system performance metrics

## Extensions

1. **Machine Learning Integration**: Add learning capabilities to improve system performance over time
2. **Multi-Robot Coordination**: Extend system for coordination between multiple robots
3. **Advanced Perception**: Implement more sophisticated perception algorithms
4. **Adaptive Control**: Create systems that adapt to changing conditions

## Summary

In this capstone lab, you've implemented:
- System integration framework connecting all major subsystems
- Task planning and execution architecture
- Real-time control and coordination mechanisms
- Performance analysis and optimization tools
- Comprehensive testing framework with scenario-based tests

These implementations demonstrate the complexity and integration challenges of autonomous humanoid robot systems, providing a foundation for real-world deployment and operation.