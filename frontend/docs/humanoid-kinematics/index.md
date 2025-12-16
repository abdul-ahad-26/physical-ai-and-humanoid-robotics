---
title: Humanoid Kinematics & Locomotion
sidebar_position: 13
---

# Humanoid Kinematics & Locomotion

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental principles of humanoid kinematics and joint configurations
- Apply forward and inverse kinematics to humanoid robot systems
- Implement basic locomotion patterns for bipedal robots
- Analyze stability and balance in humanoid locomotion
- Design control strategies for humanoid walking and movement

## Introduction to Humanoid Kinematics

Humanoid robots represent one of the most challenging areas in robotics, requiring sophisticated understanding of human-like movement patterns, balance, and coordination. Unlike wheeled or tracked robots, humanoid robots must navigate complex terrains while maintaining dynamic balance through multiple degrees of freedom.

The kinematics of humanoid robots involves the study of motion without considering the forces that cause the motion. This includes understanding how joints and links work together to achieve desired end-effector positions and orientations. Humanoid robots typically have 20-30 degrees of freedom distributed across their body structure.

### Key Components of Humanoid Robots

Humanoid robots are typically composed of:
- **Torso**: The central body structure housing the main computing units and power systems
- **Head**: Contains cameras, microphones, and sometimes displays for human-robot interaction
- **Arms**: Usually 2 arms with 6-7 degrees of freedom each, including shoulder, elbow, and wrist joints
- **Legs**: 2 legs with 6-7 degrees of freedom each, including hip, knee, and ankle joints
- **Hands**: Dextrous hands with multiple degrees of freedom for manipulation tasks

### Degrees of Freedom in Humanoid Systems

The number of degrees of freedom (DOF) in a humanoid robot determines its flexibility and capability to perform various tasks. A typical humanoid robot might have:

- **Head**: 2-3 DOF (pitch, yaw, roll)
- **Each Arm**: 6-7 DOF (shoulder: 3 DOF, elbow: 1 DOF, wrist: 2-3 DOF)
- **Each Leg**: 6-7 DOF (hip: 3 DOF, knee: 1 DOF, ankle: 2-3 DOF)
- **Total**: 26-32 DOF for a complete humanoid system

## Forward Kinematics for Humanoid Systems

Forward kinematics involves calculating the position and orientation of the end-effector given the joint angles. For humanoid robots, this is particularly complex due to the multiple limbs and the need to maintain balance while performing tasks.

### Mathematical Representation

The forward kinematics for each limb of a humanoid robot can be represented using transformation matrices. For a humanoid arm with n joints:

```
T = T1(θ1) * T2(θ2) * ... * Tn(θn)
```

Where T is the final transformation matrix and Ti(θi) represents the transformation due to joint i with angle θi.

### Denavit-Hartenberg Convention

The Denavit-Hartenberg (DH) convention is commonly used to define the coordinate frames for each joint in a humanoid robot. This systematic approach allows for consistent representation of the kinematic chain.

```python
import numpy as np

def dh_transform(a, alpha, d, theta):
    """Calculate DH transformation matrix"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])
    return T
```

## Inverse Kinematics for Humanoid Systems

Inverse kinematics is more complex than forward kinematics, as it involves finding the joint angles required to achieve a desired end-effector position and orientation. For humanoid robots, this often involves solving multiple simultaneous inverse kinematics problems while maintaining balance constraints.

### Analytical vs Numerical Solutions

For humanoid robots, both analytical and numerical approaches are used:

- **Analytical Solutions**: Fast but only possible for simple kinematic chains
- **Numerical Solutions**: More general but computationally intensive

### Jacobian-Based Methods

The Jacobian matrix relates joint velocities to end-effector velocities and is crucial for solving inverse kinematics problems:

```python
def compute_jacobian(robot_config, joint_angles):
    """Compute the Jacobian matrix for the robot"""
    # Implementation would depend on specific robot configuration
    # This is a simplified representation
    pass

def inverse_kinematics_jacobian(target_position, current_joints, robot_config):
    """Solve inverse kinematics using Jacobian transpose method"""
    # Iteratively update joint angles to reach target position
    # Implementation would include convergence criteria and limits
    pass
```

### Optimization-Based Approaches

For complex humanoid systems, optimization-based approaches are often used to solve inverse kinematics while considering multiple constraints:

- Joint angle limits
- Balance constraints
- Collision avoidance
- Energy efficiency

## Locomotion Principles

Humanoid locomotion involves complex control strategies to maintain balance while moving. Unlike humans who can rely on learned motor patterns, humanoid robots require sophisticated algorithms to achieve stable walking.

### Types of Locomotion

1. **Static Walking**: The robot maintains balance by ensuring the center of mass is always over the support polygon
2. **Dynamic Walking**: The robot uses momentum and dynamic effects to maintain balance
3. **Running**: High-speed locomotion with aerial phases
4. **Jumping**: Controlled flight phases for obstacle negotiation

### Zero Moment Point (ZMP)

The Zero Moment Point (ZMP) is a crucial concept in humanoid locomotion. It represents the point on the ground where the net moment of the ground reaction force is zero. For stable walking, the ZMP must remain within the support polygon defined by the feet.

```python
def calculate_zmp(center_of_mass, ground_reaction_force, moment):
    """Calculate Zero Moment Point"""
    # ZMP = (M_y/F_z, -M_x/F_z) where M is moment and F is force
    zmp_x = moment[1] / ground_reaction_force[2]  # M_y / F_z
    zmp_y = -moment[0] / ground_reaction_force[2]  # -M_x / F_z
    return np.array([zmp_x, zmp_y])
```

### Capture Point

The Capture Point is the location where a biped can step to stop walking and come to a complete stop. It's calculated based on the current velocity and center of mass position.

## Walking Pattern Generation

Generating stable walking patterns for humanoid robots involves several key components:

### Center of Mass Trajectory

The center of mass trajectory is planned to ensure stability during walking. Common approaches include:

- **Linear Inverted Pendulum Model (LIPM)**
- **Cart-Table model**
- **Preview control**

### Footstep Planning

Footstep planning determines where and when the feet should be placed to maintain balance:

```python
def plan_footsteps(start_pos, goal_pos, step_length, step_width):
    """Plan a sequence of footsteps from start to goal"""
    footsteps = []
    current_pos = start_pos.copy()

    # Calculate number of steps needed
    distance = np.linalg.norm(goal_pos - start_pos)
    num_steps = int(np.ceil(distance / step_length))

    # Generate intermediate steps
    direction = (goal_pos - start_pos) / distance
    for i in range(num_steps):
        # Alternate feet
        foot_offset = np.array([0, step_width/2 if i % 2 == 0 else -step_width/2, 0])
        step_pos = start_pos + direction * min(i * step_length, distance) + foot_offset
        footsteps.append(step_pos)

    return footsteps
```

### Gait Generation

Gait generation involves creating the joint angle trajectories for each step:

- **Double Support Phase**: Both feet on the ground
- **Single Support Phase**: One foot on the ground
- **Transition Phase**: Foot lift and placement

## Balance Control

Maintaining balance is one of the most critical challenges in humanoid robotics. Balance control strategies include:

### Feedback Control

- **PID controllers** for joint position control
- **Balance feedback** based on IMU readings
- **Foot pressure sensors** for balance adjustment

### Feedforward Control

- **Precomputed trajectories** for known movements
- **Model-based predictions** for balance maintenance

### Balance Strategies

1. **Ankle Strategy**: Small balance adjustments using ankle joints
2. **Hip Strategy**: Larger adjustments using hip joints
3. **Stepping Strategy**: Taking a step to restore balance
4. **Suspension Strategy**: Using arm movements to counteract disturbances

## Control Architecture

Humanoid robots typically employ a hierarchical control architecture:

### High-Level Planning

- Path planning and navigation
- Task sequencing
- Gait selection based on terrain

### Mid-Level Control

- Trajectory generation
- Balance control
- Footstep planning

### Low-Level Control

- Joint servo control
- Motor control
- Sensor feedback processing

## Simulation and Testing

Before deploying on real hardware, humanoid kinematics and locomotion algorithms are typically tested in simulation:

### Simulation Platforms

- **Gazebo**: Physics-based simulation with ROS integration
- **Webots**: Robot simulation with realistic physics
- **V-REP/CoppeliaSim**: Multi-robot simulation platform
- **NVIDIA Isaac Sim**: GPU-accelerated simulation for robotics

### Testing Scenarios

- Flat ground walking
- Inclined surfaces
- Obstacle negotiation
- Stair climbing
- Disturbance rejection

## Practical Implementation Example

Here's a simplified example of implementing basic humanoid walking:

```python
import numpy as np
import math

class HumanoidWalker:
    def __init__(self, robot_config):
        self.config = robot_config
        self.current_step = 0
        self.support_foot = "left"

    def generate_walking_trajectory(self, step_length, step_height, step_time):
        """Generate joint trajectories for a single step"""
        # Generate trajectories for center of mass
        com_trajectory = self.generate_com_trajectory(step_length, step_time)

        # Generate swing foot trajectory
        swing_trajectory = self.generate_swing_trajectory(step_length, step_height, step_time)

        # Generate joint angles using inverse kinematics
        joint_trajectories = self.inverse_kinematics_trajectories(
            com_trajectory, swing_trajectory
        )

        return joint_trajectories

    def generate_com_trajectory(self, step_length, step_time):
        """Generate center of mass trajectory for walking"""
        # Simplified LIPM-based trajectory
        omega = math.sqrt(9.81 / self.config.com_height)  # Natural frequency
        t = np.linspace(0, step_time, int(step_time * 100))  # 100 Hz control rate

        # Generate CoM trajectory
        x_com = np.zeros(len(t))
        y_com = np.zeros(len(t))
        z_com = np.full(len(t), self.config.com_height)

        # Apply CoM movement for balance
        for i, time in enumerate(t):
            x_com[i] = step_length/2 * (1 - math.cosh(omega * (time - step_time/2)) / math.cosh(omega * step_time/2))
            y_com[i] = self.config.step_width/2 * math.sin(math.pi * time / step_time)

        return np.column_stack((x_com, y_com, z_com))

    def generate_swing_trajectory(self, step_length, step_height, step_time):
        """Generate swing foot trajectory"""
        t = np.linspace(0, step_time, int(step_time * 100))

        # Generate 3D trajectory for swing foot
        x_foot = step_length * t / step_time
        y_foot = np.zeros(len(t))  # Maintain constant lateral position
        z_foot = np.zeros(len(t))

        # Add parabolic trajectory for foot lift
        for i, time in enumerate(t):
            if time < step_time/2:
                z_foot[i] = step_height * (4 * time / step_time)
            else:
                z_foot[i] = step_height * (4 * (step_time - time) / step_time)

        return np.column_stack((x_foot, y_foot, z_foot))

# Example usage
robot_config = {
    'com_height': 0.8,  # Center of mass height in meters
    'step_width': 0.2,  # Lateral distance between feet
    'max_step_length': 0.3  # Maximum step length
}

walker = HumanoidWalker(robot_config)
trajectory = walker.generate_walking_trajectory(
    step_length=0.25,
    step_height=0.05,
    step_time=1.0
)
```

## Advanced Topics

### Whole-Body Control

Advanced humanoid robots employ whole-body control frameworks that coordinate all joints simultaneously to achieve multiple tasks:

- Maintain balance
- Execute manipulation tasks
- Avoid collisions
- Optimize energy consumption

### Learning-Based Approaches

Recent advances in machine learning have enabled new approaches to humanoid locomotion:

- **Reinforcement Learning**: Learning walking patterns through trial and error
- **Imitation Learning**: Learning from human demonstrations
- **Neural Networks**: Learning complex control policies

### Terrain Adaptation

Modern humanoid robots can adapt their walking patterns to different terrains:

- Uneven ground
- Slopes and stairs
- Slippery surfaces
- Obstacle negotiation

## Troubleshooting Tips

### Common Issues in Humanoid Locomotion

1. **Instability during walking**: Check ZMP calculations and balance control parameters
2. **Joint limit violations**: Verify joint angle ranges and implement soft limits
3. **Foot slippage**: Improve foot-ground contact models and adjust walking parameters
4. **Energy inefficiency**: Optimize trajectories for minimal energy consumption

### Debugging Strategies

- **Visualization**: Use simulation tools to visualize CoM, ZMP, and foot trajectories
- **Sensor monitoring**: Monitor IMU, force/torque, and joint encoder data
- **Incremental testing**: Test on simple terrains before complex environments
- **Parameter tuning**: Systematically adjust control parameters

## Summary

Humanoid kinematics and locomotion represent one of the most challenging areas in robotics, requiring sophisticated understanding of mechanics, control theory, and human-like movement patterns. Success in this field requires:

1. Solid understanding of kinematics principles (forward and inverse)
2. Advanced control strategies for balance and locomotion
3. Appropriate simulation and testing methodologies
4. Careful consideration of hardware constraints and capabilities

The field continues to evolve with advances in machine learning, sensor technology, and computational power, promising more capable and robust humanoid robots in the future.