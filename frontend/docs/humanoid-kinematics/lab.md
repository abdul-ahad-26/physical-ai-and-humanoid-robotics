---
title: Humanoid Kinematics Lab
sidebar_position: 13
---

# Humanoid Kinematics Lab

## Lab Objectives

In this lab, you will:
1. Implement forward and inverse kinematics for a simple humanoid arm
2. Simulate basic walking patterns for a bipedal robot
3. Analyze the center of mass and Zero Moment Point (ZMP) for stability
4. Experiment with balance control strategies

## Prerequisites

- Python 3.8+ installed
- NumPy and Matplotlib libraries
- Basic understanding of robotics kinematics
- ROS 2 (optional for hardware integration)

## Exercise 1: Forward Kinematics Implementation

In this exercise, you'll implement forward kinematics for a simple humanoid arm model.

### Step 1: Create the DH Parameter Table

First, let's define the Denavit-Hartenberg parameters for a 3-DOF arm:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Define DH parameters for a 3-DOF arm
dh_params = [
    {'a': 0, 'alpha': np.pi/2, 'd': 0.1, 'theta': 0},      # Joint 1 (shoulder)
    {'a': 0.3, 'alpha': 0, 'd': 0, 'theta': 0},            # Joint 2 (elbow)
    {'a': 0.25, 'alpha': 0, 'd': 0, 'theta': 0}            # Joint 3 (wrist)
]

def forward_kinematics(joint_angles, dh_params):
    """Calculate forward kinematics for the arm"""
    T_total = np.eye(4)  # Identity matrix

    for i, (angle, params) in enumerate(zip(joint_angles, dh_params)):
        # Update theta with the joint angle
        params_copy = params.copy()
        params_copy['theta'] = angle

        # Calculate transformation matrix for this joint
        T_joint = dh_transform(
            params_copy['a'],
            params_copy['alpha'],
            params_copy['d'],
            params_copy['theta']
        )

        # Multiply to get total transformation
        T_total = T_total @ T_joint

        # Extract position for visualization
        if i == 0:
            positions = T_total[:3, 3:4]
        else:
            positions = np.hstack([positions, T_total[:3, 3:4]])

    # Extract end-effector position
    end_effector_pos = T_total[:3, 3]

    return end_effector_pos, positions

# Test the forward kinematics
joint_angles = [np.pi/4, -np.pi/3, np.pi/6]  # 45°, -60°, 30°
end_pos, all_positions = forward_kinematics(joint_angles, dh_params)

print(f"End-effector position: {end_pos}")
print(f"All joint positions: {all_positions}")
```

### Step 2: Visualize the Arm Configuration

```python
def plot_arm(positions):
    """Plot the arm configuration in 3D"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the arm links
    ax.plot(positions[0, :], positions[1, :], positions[2, :], 'bo-', linewidth=3, markersize=8, label='Arm links')

    # Mark the base and end-effector
    ax.scatter([0], [0], [0], color='red', s=100, label='Base')
    ax.scatter([positions[0, -1]], [positions[1, -1]], [positions[2, -1]], color='green', s=100, label='End-effector')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3-DOF Arm Configuration')
    ax.legend()
    ax.grid(True)

    plt.show()

# Visualize the arm
plot_arm(all_positions)
```

## Exercise 2: Inverse Kinematics Implementation

Now let's implement inverse kinematics to find joint angles for a desired end-effector position.

### Step 1: Implement Analytical Inverse Kinematics

```python
def inverse_kinematics_3dof(x, y, z):
    """Solve inverse kinematics for 3-DOF arm analytically"""
    # Arm lengths
    l1 = dh_params[1]['a']  # 0.3
    l2 = dh_params[2]['a']  # 0.25

    # Calculate joint angles
    # theta1 is in the x-y plane
    theta1 = np.arctan2(y, x)

    # Project to x-z plane
    r = np.sqrt(x**2 + y**2)

    # Use law of cosines to find theta3
    cos_theta3 = (r**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
    # Clamp to avoid numerical errors
    cos_theta3 = np.clip(cos_theta3, -1, 1)
    theta3 = np.arccos(cos_theta3)

    # Find theta2
    k1 = l1 + l2 * np.cos(theta3)
    k2 = l2 * np.sin(theta3)
    theta2 = np.arctan2(z, r) - np.arctan2(k2, k1)

    return np.array([theta1, theta2, theta3])

# Test inverse kinematics
target_pos = [0.3, 0.2, 0.2]
joint_angles_solution = inverse_kinematics_3dof(target_pos[0], target_pos[1], target_pos[2])

print(f"Target position: {target_pos}")
print(f"Joint angles: {np.degrees(joint_angles_solution)} degrees")

# Verify by running forward kinematics
end_pos_verification, _ = forward_kinematics(joint_angles_solution, dh_params)
print(f"Verification - Forward kinematics result: {end_pos_verification}")
print(f"Error: {np.linalg.norm(np.array(target_pos) - end_pos_verification)}")
```

### Step 2: Implement Numerical Inverse Kinematics

```python
def jacobian_3dof(joint_angles):
    """Calculate the Jacobian matrix for the 3-DOF arm"""
    l1 = dh_params[1]['a']
    l2 = dh_params[2]['a']

    theta1, theta2, theta3 = joint_angles

    # Calculate Jacobian elements
    J = np.zeros((3, 3))

    # Partial derivatives of end-effector position w.r.t. joint angles
    J[0, 0] = -l1 * np.sin(theta1) * np.cos(theta2) - l2 * np.sin(theta1) * np.cos(theta2 + theta3)
    J[1, 0] = l1 * np.cos(theta1) * np.cos(theta2) + l2 * np.cos(theta1) * np.cos(theta2 + theta3)
    J[2, 0] = 0

    J[0, 1] = -l1 * np.cos(theta1) * np.sin(theta2) - l2 * np.cos(theta1) * np.sin(theta2 + theta3)
    J[1, 1] = -l1 * np.sin(theta1) * np.sin(theta2) - l2 * np.sin(theta1) * np.sin(theta2 + theta3)
    J[2, 1] = l1 * np.cos(theta2) + l2 * np.cos(theta2 + theta3)

    J[0, 2] = -l2 * np.cos(theta1) * np.sin(theta2 + theta3)
    J[1, 2] = -l2 * np.sin(theta1) * np.sin(theta2 + theta3)
    J[2, 2] = -l2 * np.cos(theta2 + theta3)

    return J

def numerical_inverse_kinematics(target_pos, initial_angles, max_iterations=100, tolerance=1e-6):
    """Solve inverse kinematics using numerical method (Jacobian transpose)"""
    current_angles = initial_angles.copy()

    for i in range(max_iterations):
        # Calculate current position
        current_pos, _ = forward_kinematics(current_angles, dh_params)

        # Calculate error
        error = target_pos - current_pos

        # Check if we're close enough
        if np.linalg.norm(error) < tolerance:
            print(f"Converged after {i+1} iterations")
            break

        # Calculate Jacobian
        J = jacobian_3dof(current_angles)

        # Update angles using Jacobian transpose method
        angle_change = 0.1 * J.T @ error  # Learning rate of 0.1
        current_angles += angle_change

    return current_angles

# Test numerical inverse kinematics
initial_guess = [0, 0, 0]
numerical_solution = numerical_inverse_kinematics(target_pos, initial_guess)

print(f"Numerical solution: {np.degrees(numerical_solution)} degrees")
end_pos_numerical, _ = forward_kinematics(numerical_solution, dh_params)
print(f"Numerical verification: {end_pos_numerical}")
print(f"Numerical error: {np.linalg.norm(np.array(target_pos) - end_pos_numerical)}")
```

## Exercise 3: Walking Pattern Generation

In this exercise, we'll implement basic walking pattern generation for a bipedal robot.

### Step 1: Implement ZMP-based Walking

```python
def generate_com_trajectory(step_length, step_time, com_height):
    """Generate CoM trajectory using Linear Inverted Pendulum Model"""
    omega = np.sqrt(9.81 / com_height)
    t = np.linspace(0, step_time, int(step_time * 100))

    # Calculate CoM trajectory
    x_com = step_length/2 * (1 - np.cosh(omega * (t - step_time/2)) / np.cosh(omega * step_time/2))
    y_com = np.zeros(len(t))  # Keep CoM centered laterally
    z_com = np.full(len(t), com_height)

    return np.column_stack((x_com, y_com, z_com)), t

def calculate_zmp(com_pos, com_vel, com_acc, com_height):
    """Calculate Zero Moment Point"""
    zmp_x = com_pos[0] - (com_height / 9.81) * com_acc[0]
    zmp_y = com_pos[1] - (com_height / 9.81) * com_acc[1]
    return np.array([zmp_x, zmp_y])

# Test walking pattern generation
step_length = 0.3  # 30 cm step
step_time = 1.0    # 1 second per step
com_height = 0.8   # 80 cm CoM height

com_trajectory, time_vector = generate_com_trajectory(step_length, step_time, com_height)

# Plot CoM trajectory
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(time_vector, com_trajectory[:, 0])
plt.title('CoM X Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(time_vector, com_trajectory[:, 1])
plt.title('CoM Y Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(time_vector, com_trajectory[:, 2])
plt.title('CoM Z Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Step 2: Implement Footstep Planning

```python
def plan_footsteps(start_pos, goal_pos, step_length, step_width):
    """Plan a sequence of footsteps from start to goal"""
    footsteps = []
    current_pos = np.array(start_pos)
    goal_pos = np.array(goal_pos)

    # Calculate number of steps needed
    distance = np.linalg.norm(goal_pos - current_pos)
    num_steps = int(np.ceil(distance / step_length))

    # Calculate direction vector
    direction = (goal_pos - current_pos) / distance if distance > 0 else np.array([1, 0, 0])

    # Generate intermediate steps
    for i in range(num_steps + 1):
        # Alternate feet (left foot on even steps, right foot on odd steps)
        foot_offset_y = step_width/2 if i % 2 == 0 else -step_width/2
        foot_offset = np.array([0, foot_offset_y, 0])

        # Calculate step position
        step_pos = current_pos + direction * min(i * step_length, distance) + foot_offset
        footsteps.append(step_pos.copy())

    return np.array(footsteps)

# Test footstep planning
start_pos = [0, 0, 0]
goal_pos = [2, 0, 0]  # 2 meters forward
step_length = 0.3
step_width = 0.2

footsteps = plan_footsteps(start_pos, goal_pos, step_length, step_width)

print(f"Number of footsteps: {len(footsteps)}")
print(f"Footstep positions:\n{footsteps}")

# Plot footsteps
plt.figure(figsize=(10, 6))
plt.plot(footsteps[:, 0], footsteps[:, 1], 'ro-', markersize=8, linewidth=2, label='Footsteps')
plt.scatter([start_pos[0]], [start_pos[1]], color='green', s=100, label='Start', zorder=5)
plt.scatter([goal_pos[0]], [goal_pos[1]], color='red', s=100, label='Goal', zorder=5)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Footstep Planning')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Exercise 4: Balance Control Simulation

In this exercise, we'll implement a simple balance control simulation.

### Step 1: Implement PID Balance Controller

```python
class BalanceController:
    def __init__(self, kp=10.0, ki=0.1, kd=0.5):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        """Update PID controller"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output

def simulate_balance(initial_com_pos, initial_com_vel, target_zmp, dt=0.01):
    """Simulate balance control for a humanoid"""
    controller = BalanceController()

    # Initialize state
    com_pos = initial_com_pos.copy()
    com_vel = initial_com_vel.copy()
    com_acc = np.array([0.0, 0.0])

    # Simulation parameters
    com_height = 0.8  # 80 cm
    gravity = 9.81

    # Store history for plotting
    pos_history = [com_pos.copy()]
    vel_history = [com_vel.copy()]
    acc_history = [com_acc.copy()]
    time_history = [0.0]

    # Simulate for 5 seconds
    simulation_time = 5.0
    t = 0.0

    while t < simulation_time:
        # Calculate current ZMP
        current_zmp = com_pos - (com_height / gravity) * com_acc

        # Calculate error from target ZMP
        zmp_error = target_zmp - current_zmp

        # Use PID controller to calculate corrective acceleration
        # For simplicity, we'll just use the x-component for balance
        corrective_acc = controller.update(zmp_error[0], dt)

        # Update state
        com_acc[0] = corrective_acc
        com_vel += com_acc * dt
        com_pos += com_vel * dt

        # Store history
        pos_history.append(com_pos.copy())
        vel_history.append(com_vel.copy())
        acc_history.append(com_acc.copy())
        time_history.append(t)

        t += dt

    return np.array(pos_history), np.array(vel_history), np.array(acc_history), np.array(time_history)

# Test balance control
initial_com_pos = np.array([0.0, 0.0])  # Initial CoM position
initial_com_vel = np.array([0.0, 0.0])  # Initial CoM velocity
target_zmp = np.array([0.0, 0.0])       # Target ZMP (centered)

pos_history, vel_history, acc_history, time_history = simulate_balance(
    initial_com_pos, initial_com_vel, target_zmp
)

# Plot balance control results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(time_history, pos_history[:, 0], label='CoM X Position')
plt.title('CoM X Position Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(time_history, vel_history[:, 0], label='CoM X Velocity')
plt.title('CoM X Velocity Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(time_history, acc_history[:, 0], label='CoM X Acceleration')
plt.title('CoM X Acceleration Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

## Exercise 5: Integration Challenge

Combine all the concepts learned to implement a simple walking controller.

### Step 1: Create a Complete Walking Controller

```python
class SimpleWalkingController:
    def __init__(self, com_height=0.8, step_length=0.3, step_width=0.2, step_time=1.0):
        self.com_height = com_height
        self.step_length = step_length
        self.step_width = step_width
        self.step_time = step_time
        self.balance_controller = BalanceController()

        # Initialize robot state
        self.com_pos = np.array([0.0, 0.0, com_height])
        self.com_vel = np.array([0.0, 0.0, 0.0])
        self.current_step = 0
        self.step_phase = 0.0  # 0.0 to 1.0, representing phase of current step

    def update(self, dt):
        """Update the walking controller"""
        # Update step phase
        self.step_phase += dt / self.step_time
        if self.step_phase >= 1.0:
            self.step_phase = 0.0
            self.current_step += 1

        # Calculate desired CoM position based on walking pattern
        # This is a simplified model
        desired_com_x = self.current_step * self.step_length + self.step_length * self.step_phase
        desired_com_y = self.step_width/2 * np.sin(np.pi * self.step_phase)  # Sway to maintain balance

        # Calculate desired ZMP based on CoM position
        desired_zmp = np.array([desired_com_x, desired_com_y])

        # Calculate current ZMP
        current_zmp = np.array([
            self.com_pos[0] - (self.com_height / 9.81) * self.com_vel[0] / dt,
            self.com_pos[1] - (self.com_height / 9.81) * self.com_vel[1] / dt
        ])

        # Calculate error
        zmp_error = desired_zmp[:2] - current_zmp

        # Apply balance control
        corrective_force_x = self.balance_controller.update(zmp_error[0], dt)
        corrective_force_y = self.balance_controller.update(zmp_error[1], dt)

        # Update CoM dynamics (simplified)
        self.com_vel[0] += corrective_force_x * dt
        self.com_vel[1] += corrective_force_y * dt
        self.com_pos[:2] += self.com_vel[:2] * dt

        return self.com_pos.copy(), self.com_vel.copy(), desired_zmp, current_zmp

# Test the walking controller
walker = SimpleWalkingController()
dt = 0.01  # 100 Hz control rate

# Run simulation
com_positions = []
com_velocities = []
desired_zmps = []
current_zmps = []
time_steps = []

for i in range(500):  # 5 seconds of walking
    com_pos, com_vel, desired_zmp, current_zmp = walker.update(dt)

    com_positions.append(com_pos.copy())
    com_velocities.append(com_vel.copy())
    desired_zmps.append(desired_zmp.copy())
    current_zmps.append(current_zmp.copy())
    time_steps.append(i * dt)

com_positions = np.array(com_positions)
desired_zmps = np.array(desired_zmps)
current_zmps = np.array(current_zmps)

# Plot walking results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(time_steps, com_positions[:, 0], label='CoM X Position')
plt.plot(time_steps, desired_zmps[:, 0], '--', label='Desired ZMP X')
plt.plot(time_steps, current_zmps[:, 0], ':', label='Current ZMP X')
plt.title('X Position/CoM and ZMP')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(time_steps, com_positions[:, 1], label='CoM Y Position')
plt.plot(time_steps, desired_zmps[:, 1], '--', label='Desired ZMP Y')
plt.plot(time_steps, current_zmps[:, 1], ':', label='Current ZMP Y')
plt.title('Y Position/CoM and ZMP')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(com_positions[:, 0], com_positions[:, 1], label='CoM Path')
plt.plot(desired_zmps[:, 0], desired_zmps[:, 1], '--', label='Desired ZMP Path')
plt.title('CoM Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(time_steps, np.sqrt((desired_zmps[:, 0] - current_zmps[:, 0])**2 +
                            (desired_zmps[:, 1] - current_zmps[:, 1])**2))
plt.title('ZMP Tracking Error')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(com_positions[:, 0], com_positions[:, 2])
plt.title('CoM Height Over X')
plt.xlabel('X Position (m)')
plt.ylabel('Z Position (m)')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(time_steps, com_positions[:, 2])
plt.title('CoM Height Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Z Position (m)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Total distance walked: {com_positions[-1, 0]:.2f} meters")
print(f"Final CoM position: {com_positions[-1, :2]}")
```

## Assessment Questions

1. **Forward Kinematics**: Explain how the Denavit-Hartenberg convention helps in defining the kinematic structure of a humanoid robot. What are the advantages of using DH parameters?

2. **Inverse Kinematics**: Compare the analytical and numerical methods for solving inverse kinematics. When would you choose one over the other?

3. **Walking Patterns**: Why is the Zero Moment Point (ZMP) important for humanoid locomotion? How does it relate to the stability of the robot?

4. **Balance Control**: Describe the different balance strategies a humanoid robot can use (ankle, hip, stepping). When would each strategy be most appropriate?

5. **Implementation Challenge**: How would you modify the walking controller to handle uneven terrain or obstacles?

## Troubleshooting Guide

### Common Issues in Implementation

1. **Kinematic Singularities**: When the Jacobian becomes singular, the inverse becomes undefined. This typically happens when joints are aligned in certain configurations.

2. **ZMP Outside Support Polygon**: If the calculated ZMP goes outside the support polygon (the area covered by the feet), the robot will be unstable.

3. **Numerical Instability**: Small errors in calculations can accumulate over time, leading to unstable behavior.

### Debugging Strategies

1. **Visualization**: Always visualize your kinematic solutions to verify they are correct.

2. **Incremental Testing**: Test each component separately before integrating them.

3. **Parameter Tuning**: Start with conservative parameters and gradually adjust for performance.

## Extensions

1. **Add More DOF**: Extend the kinematic model to include more joints in the humanoid robot.

2. **Implement Whole-Body Control**: Coordinate multiple limbs simultaneously for more complex tasks.

3. **Terrain Adaptation**: Modify the walking controller to handle different types of terrain.

4. **Hardware Integration**: Connect your simulation to a real humanoid robot platform if available.

## Summary

In this lab, you've implemented:
- Forward and inverse kinematics for a simplified humanoid arm
- Walking pattern generation using ZMP principles
- Balance control using PID controllers
- Integration of all components into a walking controller

These implementations provide a foundation for understanding the complex control systems required for humanoid robots. The skills learned here can be extended to more complex humanoid systems with additional degrees of freedom and more sophisticated control strategies.