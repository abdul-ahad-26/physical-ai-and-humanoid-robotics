---
title: Manipulation Lab
sidebar_position: 14
---

# Manipulation Lab

## Lab Objectives

In this lab, you will:
1. Implement grasp planning algorithms for different object types
2. Design force control strategies for stable grasping
3. Create manipulation trajectories that maintain robot balance
4. Experiment with learning-based manipulation approaches

## Prerequisites

- Python 3.8+ installed
- NumPy, SciPy, Matplotlib, PyTorch libraries
- Basic understanding of robotics kinematics
- ROS 2 (optional for hardware integration)

## Exercise 1: Grasp Planning Implementation

In this exercise, you'll implement grasp planning algorithms for different types of objects.

### Step 1: Create Object Representation

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

class ObjectRepresentation:
    """Represents an object for grasp planning"""

    def __init__(self, points=None, shape_type="custom", **kwargs):
        """
        Initialize object representation

        Args:
            points: Array of 3D points representing the object
            shape_type: Type of object ("box", "cylinder", "sphere", "custom")
            **kwargs: Additional parameters based on shape type
        """
        if shape_type == "box":
            # Create a box with given dimensions
            length, width, height = kwargs.get('dimensions', [0.1, 0.05, 0.05])
            self.points = self._create_box_points(length, width, height)
        elif shape_type == "cylinder":
            # Create a cylinder with given dimensions
            radius, height = kwargs.get('dimensions', [0.05, 0.1])
            self.points = self._create_cylinder_points(radius, height)
        elif shape_type == "sphere":
            # Create a sphere with given radius
            radius = kwargs.get('radius', 0.05)
            self.points = self._create_sphere_points(radius)
        else:
            # Use provided points
            self.points = points if points is not None else np.random.rand(100, 3)

    def _create_box_points(self, length, width, height, num_points=1000):
        """Create points for a box shape"""
        points = []
        # Generate points on the surface of the box
        for _ in range(num_points):
            # Random face selection
            face = np.random.randint(0, 6)
            if face == 0:  # Front face
                x = np.random.uniform(-length/2, length/2)
                y = np.random.uniform(-width/2, width/2)
                z = height/2
            elif face == 1:  # Back face
                x = np.random.uniform(-length/2, length/2)
                y = np.random.uniform(-width/2, width/2)
                z = -height/2
            elif face == 2:  # Top face
                x = np.random.uniform(-length/2, length/2)
                y = width/2
                z = np.random.uniform(-height/2, height/2)
            elif face == 3:  # Bottom face
                x = np.random.uniform(-length/2, length/2)
                y = -width/2
                z = np.random.uniform(-height/2, height/2)
            elif face == 4:  # Right face
                x = length/2
                y = np.random.uniform(-width/2, width/2)
                z = np.random.uniform(-height/2, height/2)
            else:  # Left face
                x = -length/2
                y = np.random.uniform(-width/2, width/2)
                z = np.random.uniform(-height/2, height/2)
            points.append([x, y, z])

        return np.array(points)

    def _create_cylinder_points(self, radius, height, num_points=1000):
        """Create points for a cylinder shape"""
        points = []
        for _ in range(num_points):
            # Random point on cylinder surface
            if np.random.rand() < 0.3:  # Top/bottom cap
                r = np.random.uniform(0, radius)
                theta = np.random.uniform(0, 2*np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = height/2 if np.random.rand() > 0.5 else -height/2
            else:  # Side surface
                theta = np.random.uniform(0, 2*np.pi)
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                z = np.random.uniform(-height/2, height/2)
            points.append([x, y, z])

        return np.array(points)

    def _create_sphere_points(self, radius, num_points=1000):
        """Create points for a sphere shape"""
        points = []
        for _ in range(num_points):
            # Random point on sphere surface
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            points.append([x, y, z])

        return np.array(points)

    def get_surface_points(self, num_samples=100):
        """Get a sample of surface points"""
        indices = np.random.choice(len(self.points),
                                 size=min(num_samples, len(self.points)),
                                 replace=False)
        return self.points[indices]

    def get_convex_hull(self):
        """Get the convex hull of the object"""
        return ConvexHull(self.points)

# Test object creation
box_obj = ObjectRepresentation(shape_type="box", dimensions=[0.1, 0.05, 0.08])
cylinder_obj = ObjectRepresentation(shape_type="cylinder", dimensions=[0.04, 0.12])
sphere_obj = ObjectRepresentation(shape_type="sphere", radius=0.06)

print(f"Box object points shape: {box_obj.points.shape}")
print(f"Cylinder object points shape: {cylinder_obj.points.shape}")
print(f"Sphere object points shape: {sphere_obj.points.shape}")
```

### Step 2: Implement Grasp Planning Algorithm

```python
class GraspPlanner:
    """Grasp planning for different gripper types"""

    def __init__(self, gripper_type="parallel_jaw", finger_width=0.01):
        self.gripper_type = gripper_type
        self.finger_width = finger_width

    def find_antipodal_grasps(self, object_points, gripper_width=0.08):
        """
        Find antipodal grasps - pairs of points with opposing surface normals
        """
        # Calculate surface normals (simplified approach)
        surface_points = self._sample_surface_points(object_points)

        # Find pairs of points that are approximately gripper width apart
        distances = cdist(surface_points, surface_points)

        grasp_candidates = []
        for i in range(len(surface_points)):
            for j in range(i+1, len(surface_points)):
                if abs(distances[i, j] - gripper_width) < 0.01:  # 1cm tolerance
                    # Calculate approximate normals (for this example, simplified)
                    normal_i = self._estimate_normal(surface_points, i)
                    normal_j = self._estimate_normal(surface_points, j)

                    # Check if normals are roughly opposing (dot product close to -1)
                    if np.dot(normal_i, normal_j) < -0.7:  # Opposing normals
                        grasp_center = (surface_points[i] + surface_points[j]) / 2
                        grasp_approach = self._calculate_approach_vector(
                            surface_points[i], surface_points[j], normal_i, normal_j
                        )

                        grasp_candidates.append({
                            'position': grasp_center,
                            'points': (surface_points[i], surface_points[j]),
                            'approach': grasp_approach,
                            'quality': self._evaluate_grasp_quality(
                                surface_points[i], surface_points[j],
                                normal_i, normal_j
                            )
                        })

        # Sort by quality
        grasp_candidates.sort(key=lambda x: x['quality'], reverse=True)
        return grasp_candidates

    def _sample_surface_points(self, points, num_samples=200):
        """Sample points from the object surface"""
        indices = np.random.choice(len(points),
                                  size=min(num_samples, len(points)),
                                  replace=False)
        return points[indices]

    def _estimate_normal(self, points, idx, neighborhood_radius=0.02):
        """Estimate surface normal at a point"""
        # Find neighboring points
        point = points[idx]
        distances = np.linalg.norm(points - point, axis=1)
        neighbors = points[distances < neighborhood_radius]

        if len(neighbors) < 3:
            return np.array([0, 0, 1])  # Default normal

        # Calculate covariance matrix
        cov_matrix = np.cov(neighbors.T)

        # Get the eigenvector corresponding to the smallest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # Ensure normal points outward
        if normal[2] < 0:  # Assuming z is up
            normal = -normal

        return normal / np.linalg.norm(normal)

    def _calculate_approach_vector(self, point1, point2, normal1, normal2):
        """Calculate the approach vector for the grasp"""
        # Approach along the line connecting the grasp points
        approach = point2 - point1
        return approach / np.linalg.norm(approach)

    def _evaluate_grasp_quality(self, point1, point2, normal1, normal2):
        """Evaluate the quality of a potential grasp"""
        # Calculate grasp width
        grasp_width = np.linalg.norm(point1 - point2)

        # Calculate how opposing the normals are
        normal_alignment = np.dot(normal1, normal2)

        # Calculate friction cone constraint (simplified)
        friction_coeff = 0.5  # Assume friction coefficient
        friction_quality = 1.0 / (1.0 + abs(normal_alignment))

        # Combine factors for overall quality
        quality = friction_quality * (1.0 / (1.0 + abs(grasp_width - 0.08)))  # Target width of 8cm

        return quality

# Test grasp planning
planner = GraspPlanner()
object_points = box_obj.points
grasp_candidates = planner.find_antipodal_grasps(object_points, gripper_width=0.08)

print(f"Found {len(grasp_candidates)} grasp candidates")
if grasp_candidates:
    best_grasp = grasp_candidates[0]
    print(f"Best grasp position: {best_grasp['position']}")
    print(f"Best grasp quality: {best_grasp['quality']:.3f}")
```

### Step 3: Visualize Grasp Candidates

```python
def visualize_grasps(object_points, grasp_candidates, num_to_show=5):
    """Visualize grasp candidates on the object"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot object points
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2],
               c='lightblue', alpha=0.5, s=1)

    # Plot grasp points and approach vectors
    for i, grasp in enumerate(grasp_candidates[:num_to_show]):
        p1, p2 = grasp['points']

        # Plot grasp points
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   c=['red', 'blue'], s=50)

        # Plot approach vector (from center of grasp to show direction)
        center = grasp['position']
        approach = grasp['approach'] * 0.02  # Scale for visibility
        ax.quiver(center[0], center[1], center[2],
                  approach[0], approach[1], approach[2],
                  color='green', arrow_length_ratio=0.3, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Grasp Candidates Visualization')
    plt.show()

# Visualize grasps
visualize_grasps(object_points, grasp_candidates)
```

## Exercise 2: Force Control Implementation

In this exercise, you'll implement force control strategies for stable grasping.

### Step 1: Implement Impedance Controller

```python
class ImpedanceController:
    """Impedance controller for manipulation"""

    def __init__(self, mass=1.0, damping=10.0, stiffness=100.0):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness

        # State variables
        self.prev_error = 0
        self.integrated_error = 0

    def update(self, desired_pos, current_pos, desired_force, current_force, dt):
        """
        Update impedance controller

        Args:
            desired_pos: Desired position
            current_pos: Current position
            desired_force: Desired force
            current_force: Current force measurement
            dt: Time step

        Returns:
            Control command (position adjustment)
        """
        # Position error
        pos_error = desired_pos - current_pos

        # Force error
        force_error = desired_force - current_force

        # Calculate control output
        # Position-based control
        pos_control = (self.stiffness * pos_error +
                      self.damping * (pos_error - self.prev_error) / dt if dt > 0 else 0)

        # Force-based control
        force_control = force_error  # Adjust based on force feedback

        # Combine position and force control
        control_output = pos_control * 0.7 + force_control * 0.3  # Weighted combination

        # Update state variables
        self.prev_error = pos_error
        self.integrated_error += pos_error * dt

        return control_output

class ForceController:
    """Force controller for grasp force regulation"""

    def __init__(self, kp=1.0, ki=0.1, kd=0.05):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        # State variables
        self.prev_error = 0
        self.integral = 0

    def update(self, desired_force, current_force, dt):
        """
        Update force controller

        Args:
            desired_force: Desired grasp force
            current_force: Current measured force
            dt: Time step

        Returns:
            Force adjustment command
        """
        error = desired_force - current_force

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = (self.kp * error +
                 self.ki * self.integral +
                 self.kd * derivative)

        self.prev_error = error

        return output

# Test force controllers
impedance_ctrl = ImpedanceController(mass=0.5, damping=5.0, stiffness=50.0)
force_ctrl = ForceController(kp=2.0, ki=0.2, kd=0.1)

# Simulate a grasping scenario
dt = 0.01  # 100 Hz
time_steps = np.arange(0, 2.0, dt)  # 2 seconds

# Simulate object approach and grasp
desired_positions = np.zeros(len(time_steps))
desired_forces = np.zeros(len(time_steps))

# Set up desired trajectory
for i, t in enumerate(time_steps):
    if t < 0.5:  # Approach phase
        desired_positions[i] = -0.1  # Move towards object
        desired_forces[i] = 0.1      # Low force during approach
    elif t < 1.0:  # Grasp phase
        desired_positions[i] = -0.05  # Grasp position
        desired_forces[i] = 5.0       # Increase grasp force
    else:  # Hold phase
        desired_positions[i] = -0.05  # Maintain position
        desired_forces[i] = 5.0       # Maintain grasp force

# Simulate the control process
current_pos = -0.2  # Start position
current_force = 0.0  # Initial force
pos_history = [current_pos]
force_history = [current_force]
control_history = []

for i, t in enumerate(time_steps):
    # Simulate sensor readings with noise
    measured_pos = current_pos + np.random.normal(0, 0.001)
    measured_force = current_force + np.random.normal(0, 0.01)

    # Update controllers
    pos_control = impedance_ctrl.update(
        desired_positions[i], measured_pos,
        desired_forces[i], measured_force, dt
    )

    force_control = force_ctrl.update(
        desired_forces[i], measured_force, dt
    )

    # Combine controls (simplified)
    total_control = pos_control * 0.8 + force_control * 0.2

    # Apply control (simplified dynamics)
    current_pos += total_control * dt
    current_force = min(10.0, max(0.0, desired_forces[i] + force_control * dt))

    # Store history
    pos_history.append(current_pos)
    force_history.append(current_force)
    control_history.append(total_control)

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(time_steps, desired_positions, label='Desired Position', linestyle='--')
plt.plot(time_steps, pos_history[:-1], label='Actual Position')
plt.title('Position Control')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(time_steps, desired_forces, label='Desired Force', linestyle='--')
plt.plot(time_steps, force_history[:-1], label='Actual Force')
plt.title('Force Control')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(time_steps, control_history)
plt.title('Control Output')
plt.xlabel('Time (s)')
plt.ylabel('Control Signal')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Exercise 3: Balance-Constrained Manipulation

In this exercise, you'll implement manipulation planning that maintains robot balance.

### Step 1: Implement Balance Constraint Functions

```python
class BalanceConstraintPlanner:
    """Plan manipulation while maintaining robot balance"""

    def __init__(self, com_height=0.8, foot_separation=0.2):
        self.com_height = com_height
        self.foot_separation = foot_separation
        self.gravity = 9.81

    def calculate_support_polygon(self, left_foot_pos, right_foot_pos):
        """Calculate the support polygon from foot positions"""
        # For simplicity, assume rectangular support polygon
        # In reality, this would be more complex
        center_x = (left_foot_pos[0] + right_foot_pos[0]) / 2
        center_y = (left_foot_pos[1] + right_foot_pos[1]) / 2

        # Support polygon bounds (simplified as rectangle)
        half_width = abs(left_foot_pos[1] - right_foot_pos[1]) / 2
        half_length = 0.1  # Approximate foot length

        return {
            'center': [center_x, center_y],
            'bounds': [half_length, half_width]
        }

    def is_balance_safe(self, com_pos, support_polygon):
        """Check if center of mass is within support polygon"""
        com_x, com_y = com_pos[:2]
        center_x, center_y = support_polygon['center']
        half_length, half_width = support_polygon['bounds']

        # Check if CoM is within bounds
        x_safe = abs(com_x - center_x) <= half_length
        y_safe = abs(com_y - center_y) <= half_width

        return x_safe and y_safe

    def calculate_zmp(self, com_pos, com_vel, com_acc):
        """Calculate Zero Moment Point"""
        zmp_x = com_pos[0] - (self.com_height / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (self.com_height / self.gravity) * com_acc[1]
        return np.array([zmp_x, zmp_y])

    def plan_safe_manipulation(self, initial_com, target_object_pos,
                              manipulator_pos, dt=0.01):
        """Plan manipulation trajectory that maintains balance"""
        # Simplified approach: adjust manipulator position to keep CoM in support polygon

        # Define trajectory parameters
        duration = 2.0  # 2 seconds for manipulation
        steps = int(duration / dt)

        # Initialize trajectory
        com_trajectory = []
        manipulator_trajectory = []
        zmp_trajectory = []

        current_com = initial_com.copy()
        current_manipulator = manipulator_pos.copy()

        # Calculate initial support polygon
        left_foot = np.array([0.0, self.foot_separation/2, 0.0])
        right_foot = np.array([0.0, -self.foot_separation/2, 0.0])
        support_polygon = self.calculate_support_polygon(left_foot, right_foot)

        for i in range(steps):
            # Calculate desired manipulator position (approach target)
            approach_progress = min(1.0, i * dt / (duration * 0.5))  # 50% for approach
            desired_manipulator = (1 - approach_progress) * current_manipulator + \
                                 approach_progress * target_object_pos

            # Calculate effect of manipulator position on CoM
            # Simplified model: CoM shifts based on manipulator position
            com_shift = 0.1 * (desired_manipulator[:2] - current_manipulator[:2])
            predicted_com = current_com.copy()
            predicted_com[:2] += com_shift

            # Check if predicted CoM is safe
            if not self.is_balance_safe(predicted_com, support_polygon):
                # Adjust manipulator position to maintain balance
                # Move more conservatively toward target
                safe_manipulator = current_manipulator + 0.5 * (target_object_pos - current_manipulator) * dt
            else:
                safe_manipulator = desired_manipulator

            # Update positions
            current_manipulator = safe_manipulator
            current_com[:2] += com_shift * 0.8  # Dampen the effect

            # Calculate ZMP
            com_vel = np.zeros(3)  # Simplified
            com_acc = np.zeros(3)  # Simplified
            zmp = self.calculate_zmp(current_com, com_vel, com_acc)

            # Store trajectory
            com_trajectory.append(current_com.copy())
            manipulator_trajectory.append(current_manipulator.copy())
            zmp_trajectory.append(zmp.copy())

        return {
            'com_trajectory': np.array(com_trajectory),
            'manipulator_trajectory': np.array(manipulator_trajectory),
            'zmp_trajectory': np.array(zmp_trajectory)
        }

# Test balance-constrained manipulation
balance_planner = BalanceConstraintPlanner(com_height=0.8, foot_separation=0.2)

# Initial conditions
initial_com = np.array([0.0, 0.0, 0.8])
target_object = np.array([0.3, 0.1, 0.2])  # Object at x=0.3, y=0.1, z=0.2
initial_manipulator = np.array([0.1, 0.0, 0.5])  # Manipulator start position

# Plan manipulation
result = balance_planner.plan_safe_manipulation(
    initial_com, target_object, initial_manipulator
)

print(f"Planned trajectory with {len(result['com_trajectory'])} steps")
print(f"Final CoM position: {result['com_trajectory'][-1]}")
print(f"Final manipulator position: {result['manipulator_trajectory'][-1]}")

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
com_x = [pos[0] for pos in result['com_trajectory']]
com_y = [pos[1] for pos in result['com_trajectory']]
plt.plot(com_x, com_y, label='CoM Path', linewidth=2)
plt.title('CoM Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.subplot(2, 3, 2)
manip_x = [pos[0] for pos in result['manipulator_trajectory']]
manip_y = [pos[1] for pos in result['manipulator_trajectory']]
plt.plot(manip_x, manip_y, label='Manipulator Path', linewidth=2)
plt.title('Manipulator Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.subplot(2, 3, 3)
zmp_x = [pos[0] for pos in result['zmp_trajectory']]
zmp_y = [pos[1] for pos in result['zmp_trajectory']]
plt.plot(zmp_x, zmp_y, label='ZMP Path', linewidth=2)
plt.title('ZMP Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.subplot(2, 3, 4)
time_steps = np.linspace(0, 2.0, len(com_x))
plt.plot(time_steps, com_x, label='CoM X')
plt.plot(time_steps, manip_x, label='Manipulator X')
plt.title('X Position Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(time_steps, com_y, label='CoM Y')
plt.plot(time_steps, manip_y, label='Manipulator Y')
plt.title('Y Position Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(time_steps, [pos[2] for pos in result['com_trajectory']], label='CoM Height')
plt.title('CoM Height Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

## Exercise 4: Learning-Based Grasp Evaluation

In this exercise, you'll implement a simple learning approach to evaluate grasp quality.

### Step 1: Create a Grasp Quality Evaluation Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GraspQualityNet(nn.Module):
    """Neural network to evaluate grasp quality"""

    def __init__(self, input_size=10, hidden_size=64):
        super(GraspQualityNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1),
            nn.Sigmoid()  # Output between 0 and 1 (quality score)
        )

    def forward(self, x):
        return self.network(x)

def create_grasp_features(object_dims, grasp_width, grasp_approach, grasp_quality=None):
    """
    Create feature vector for grasp evaluation

    Args:
        object_dims: [length, width, height] of object
        grasp_width: Distance between gripper fingers
        grasp_approach: [x, y, z] approach vector
        grasp_quality: Optional ground truth quality (for training)

    Returns:
        Feature vector and optionally the quality label
    """
    # Normalize object dimensions
    norm_dims = np.array(object_dims) / np.linalg.norm(object_dims)

    # Normalize approach vector
    norm_approach = grasp_approach / np.linalg.norm(grasp_approach)

    # Create feature vector
    features = np.concatenate([
        norm_dims,                    # 3: normalized object dimensions
        [grasp_width],                # 1: grasp width
        norm_approach,                # 3: normalized approach vector
        [grasp_width / max(object_dims)]  # 1: relative grasp width
    ])

    if grasp_quality is not None:
        return features, grasp_quality
    else:
        return features

# Generate training data (simulated)
def generate_training_data(num_samples=1000):
    """Generate simulated training data for grasp quality evaluation"""
    X = []
    y = []

    for _ in range(num_samples):
        # Random object dimensions
        dims = np.random.uniform(0.02, 0.2, 3)

        # Random grasp width (relative to object size)
        grasp_width = np.random.uniform(0.02, max(dims) * 1.5)

        # Random approach vector
        approach = np.random.uniform(-1, 1, 3)
        approach = approach / np.linalg.norm(approach)

        # Simulate quality based on geometric factors
        # Objects with good aspect ratios and appropriate grasp width get higher scores
        aspect_ratio = min(dims) / max(dims)
        width_ratio = grasp_width / max(dims)

        # Quality calculation (simplified physics model)
        quality = 0.5  # Base quality

        # Higher quality for objects with good aspect ratios
        quality += 0.3 * aspect_ratio

        # Higher quality for appropriate grasp width (not too wide or narrow)
        width_score = 1.0 - abs(width_ratio - 1.0)  # Peak at width_ratio = 1.0
        quality += 0.2 * max(0, width_score)

        # Higher quality for approach aligned with object principal axes
        # For simplicity, assume principal axes are aligned with dimensions
        alignment_score = max(abs(approach[0]), abs(approach[1]), abs(approach[2]))
        quality += 0.2 * alignment_score

        # Clamp quality to [0, 1]
        quality = np.clip(quality, 0.0, 1.0)

        features, label = create_grasp_features(dims, grasp_width, approach, quality)
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Generate training data
X_train, y_train = generate_training_data(2000)
X_val, y_val = generate_training_data(500)

print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, labels shape: {y_val.shape}")

# Create and train the network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraspQualityNet(input_size=8)  # 8 features
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

# Training loop
epochs = 200
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    train_loss = criterion(outputs, y_train_tensor)
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plot training progress
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Test the trained model
model.eval()
with torch.no_grad():
    test_outputs = model(X_val_tensor[:10])
    test_predictions = test_outputs.squeeze().numpy()
    test_actual = y_val[:10]

plt.subplot(1, 2, 2)
plt.scatter(test_actual, test_predictions, alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Prediction vs Actual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Sample predictions vs actual:")
for i in range(5):
    print(f"  Actual: {test_actual[i]:.3f}, Predicted: {test_predictions[i]:.3f}")
```

## Exercise 5: Integration Challenge - Complete Manipulation System

Combine all the concepts learned to create a complete manipulation system.

### Step 1: Create Integrated Manipulation System

```python
class IntegratedManipulationSystem:
    """Complete manipulation system integrating planning, control, and learning"""

    def __init__(self):
        self.grasp_planner = GraspPlanner()
        self.impedance_controller = ImpedanceController()
        self.force_controller = ForceController()
        self.balance_planner = BalanceConstraintPlanner()
        self.grasp_quality_model = GraspQualityNet(input_size=8)

        # State variables
        self.current_object = None
        self.current_grasp = None
        self.current_manipulator_pos = np.array([0.1, 0.0, 0.5])
        self.current_com = np.array([0.0, 0.0, 0.8])

    def perceive_object(self, object_data):
        """Perceive and model an object in the environment"""
        # In a real system, this would use sensors like cameras or lidar
        # For this simulation, we'll use the ObjectRepresentation class
        if isinstance(object_data, dict):
            self.current_object = ObjectRepresentation(**object_data)
        else:
            self.current_object = object_data

    def plan_grasp(self, gripper_width=0.08):
        """Plan a grasp for the current object"""
        if self.current_object is None:
            raise ValueError("No object to grasp")

        # Find grasp candidates
        grasp_candidates = self.grasp_planner.find_antipodal_grasps(
            self.current_object.points, gripper_width
        )

        if not grasp_candidates:
            return None

        # Evaluate grasps using the learned model
        best_grasp = None
        best_quality = -1

        for grasp in grasp_candidates[:10]:  # Evaluate top 10 candidates
            # Extract features for this grasp
            object_dims = [0.1, 0.05, 0.08]  # Placeholder - would come from perception
            grasp_width = gripper_width
            approach = grasp['approach']

            features = create_grasp_features(object_dims, grasp_width, approach)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # Get quality prediction
            with torch.no_grad():
                quality = self.grasp_quality_model(features_tensor).item()

            if quality > best_quality:
                best_quality = quality
                best_grasp = grasp
                best_grasp['predicted_quality'] = quality

        return best_grasp

    def execute_grasp(self, grasp_plan, dt=0.01):
        """Execute a planned grasp with force control"""
        if grasp_plan is None:
            return False

        # Simulate grasp execution
        grasp_position = grasp_plan['position']
        approach_vector = grasp_plan['approach']

        # Approach phase
        approach_steps = int(0.5 / dt)  # 0.5 seconds for approach
        for i in range(approach_steps):
            # Calculate intermediate position
            progress = i / approach_steps
            target_pos = (1 - progress) * self.current_manipulator_pos + \
                        progress * (grasp_position - 0.05 * approach_vector)  # Stop 5cm before

            # Apply impedance control
            pos_error = target_pos - self.current_manipulator_pos
            control_output = self.impedance_controller.update(
                target_pos, self.current_manipulator_pos,
                0.5, 0.1, dt  # Low force during approach
            )

            # Update manipulator position (simplified)
            self.current_manipulator_pos += control_output * dt * 0.1

        # Grasp phase
        grasp_steps = int(0.3 / dt)  # 0.3 seconds for grasping
        for i in range(grasp_steps):
            # Move to grasp position with force control
            target_pos = grasp_position
            desired_force = 5.0  # Grasp force

            # Apply combined position and force control
            pos_control = self.impedance_controller.update(
                target_pos, self.current_manipulator_pos,
                desired_force, 2.0, dt  # Actual force starts low
            )

            force_control = self.force_controller.update(
                desired_force, 2.0, dt  # Actual force
            )

            # Combine controls
            total_control = pos_control * 0.7 + force_control * 0.3
            self.current_manipulator_pos += total_control * dt * 0.05

        # Update current grasp
        self.current_grasp = grasp_plan
        return True

    def move_object(self, target_position, dt=0.01):
        """Move the grasped object to a target position"""
        if self.current_grasp is None:
            raise ValueError("No object is grasped")

        # Plan trajectory considering balance constraints
        trajectory_plan = self.balance_planner.plan_safe_manipulation(
            self.current_com, target_position, self.current_manipulator_pos, dt
        )

        # Execute trajectory
        for i in range(len(trajectory_plan['manipulator_trajectory'])):
            target_pos = trajectory_plan['manipulator_trajectory'][i]

            # Apply impedance control to follow trajectory
            pos_control = self.impedance_controller.update(
                target_pos, self.current_manipulator_pos,
                5.0, 5.0, dt  # Maintain grasp force
            )

            # Update positions
            self.current_manipulator_pos += pos_control * dt * 0.1
            self.current_com = trajectory_plan['com_trajectory'][i]

        return True

# Test the integrated system
manipulation_system = IntegratedManipulationSystem()

# Perceive an object
object_config = {
    'shape_type': 'box',
    'dimensions': [0.1, 0.05, 0.08]
}
manipulation_system.perceive_object(object_config)

print("Object perceived successfully")

# Plan a grasp
grasp_plan = manipulation_system.plan_grasp(gripper_width=0.08)
if grasp_plan:
    print(f"Grasp planned successfully with predicted quality: {grasp_plan.get('predicted_quality', 0):.3f}")
    print(f"Grasp position: {grasp_plan['position']}")
else:
    print("No valid grasp found")

# Execute the grasp
grasp_success = manipulation_system.execute_grasp(grasp_plan)
print(f"Grasp execution: {'Success' if grasp_success else 'Failed'}")

# Move object to new position
if grasp_success:
    target_pos = np.array([0.2, 0.1, 0.4])
    move_success = manipulation_system.move_object(target_pos)
    print(f"Object move: {'Success' if move_success else 'Failed'}")
    print(f"Final manipulator position: {manipulation_system.current_manipulator_pos}")

# Print system status
print(f"\nSystem Status:")
print(f"  Current manipulator position: {manipulation_system.current_manipulator_pos}")
print(f"  Current CoM: {manipulation_system.current_com}")
print(f"  Current grasp: {'Yes' if manipulation_system.current_grasp else 'No'}")
```

## Assessment Questions

1. **Grasp Planning**: Explain the difference between antipodal grasps and caging grasps. When would you use each approach?

2. **Force Control**: Describe the advantages and disadvantages of impedance control versus admittance control for robotic manipulation.

3. **Balance Constraints**: How does the center of mass trajectory affect the stability of a humanoid robot during manipulation tasks?

4. **Learning Approaches**: Compare analytical grasp evaluation methods with learning-based approaches. What are the trade-offs?

5. **Integration Challenge**: How would you modify the integrated system to handle multiple objects in a cluttered environment?

## Troubleshooting Guide

### Common Issues in Manipulation

1. **Grasp Failure**: Objects slipping or being dropped during manipulation.
   - Solution: Increase grasp force, improve grasp planning, or use tactile feedback.

2. **Balance Loss**: Robot becoming unstable during manipulation.
   - Solution: Implement stricter balance constraints, use balance recovery strategies.

3. **Force Control Instability**: Oscillations or instability in force control.
   - Solution: Tune controller parameters, add filtering, or use adaptive control.

### Debugging Strategies

1. **Visualization**: Always visualize grasp candidates and planned trajectories.
2. **Sensor Calibration**: Ensure force/torque and position sensors are properly calibrated.
3. **Gradual Testing**: Start with simple objects and tasks before complex ones.
4. **Safety Limits**: Implement strict safety limits to prevent damage.

## Extensions

1. **Advanced Grippers**: Implement multi-fingered hand control with individual finger control.
2. **Tactile Feedback**: Integrate tactile sensor data for more robust grasping.
3. **Visual Servoing**: Add visual feedback to guide manipulation in real-time.
4. **Learning from Failure**: Implement systems that learn from grasp failures.

## Summary

In this lab, you've implemented:
- Grasp planning algorithms for different object types
- Force control strategies for stable grasping
- Balance-constrained manipulation planning
- Learning-based grasp quality evaluation
- An integrated manipulation system

These implementations provide a foundation for understanding the complex control systems required for robotic manipulation. The skills learned here can be extended to more sophisticated manipulation systems with advanced sensors, learning algorithms, and control strategies.