---
title: Isaac Orbit (RL Training)
sidebar_position: 8.5
description: NVIDIA Isaac Orbit for reinforcement learning training and robot skill acquisition
---

# Isaac Orbit (RL Training)

## Learning Objectives

- Understand Isaac Orbit's role in reinforcement learning for robotics
- Set up Isaac Orbit for robot skill training
- Implement RL environments and training configurations
- Train robotic policies using Isaac Orbit
- Deploy trained policies to real robots
- Optimize training performance and hyperparameters
- Troubleshoot common Isaac Orbit issues

## Introduction to Isaac Orbit

Isaac Orbit is NVIDIA's comprehensive framework for reinforcement learning in robotics. It provides a unified platform for training robot policies in simulation with the ability to transfer them to real robots. Isaac Orbit combines high-fidelity physics simulation with efficient RL algorithms to enable rapid skill acquisition for complex robotic tasks.

### Key Isaac Orbit Capabilities

- **Modular Environment Design**: Flexible environment creation for diverse robotic tasks
- **GPU-Accelerated Training**: Leverage GPU parallelism for faster training
- **Domain Randomization**: Techniques for sim-to-real transfer
- **Multi-Task Learning**: Train policies for multiple tasks simultaneously
- **Real-time Policy Execution**: Deploy trained policies with low latency
- **Physics Simulation**: Accurate physics modeling with PhysX integration

### Isaac Orbit vs Traditional RL Frameworks

| Aspect | Traditional RL | Isaac Orbit |
|--------|----------------|-------------|
| Physics | Basic or simplified | PhysX-based high-fidelity |
| Robotics Focus | General purpose | Robotics-specific |
| Simulation Quality | Standard | Photo-realistic with RTX |
| Hardware Acceleration | CPU/GPU | GPU-optimized with CUDA |
| Environment Complexity | Limited | Highly configurable |
| Transfer Learning | Manual | Built-in domain randomization |

## Isaac Orbit Architecture

### Core Components

Isaac Orbit consists of several key components:

- **Environment Manager**: Handles environment creation and management
- **Task Manager**: Defines specific tasks and rewards
- **Policy Manager**: Manages policy training and execution
- **Simulation Engine**: PhysX-based physics simulation
- **Observation Manager**: Processes sensor observations
- **Action Manager**: Maps actions to robot commands

### Package Structure

```
isaac_orbit/
├── orbit/
│   ├── core/
│   │   ├── envs/           # Environment definitions
│   │   ├── tasks/          # Task definitions
│   │   ├── agents/         # Agent implementations
│   │   └── utils/          # Utility functions
│   ├── assets/             # Robot and environment models
│   │   ├── robots/
│   │   ├── environments/
│   │   └── objects/
│   ├── configs/            # Training configurations
│   │   ├── rl_games/
│   │   ├── skrl/
│   │   └── rsl_rl/
│   ├── extensions/         # Custom extensions
│   └── scripts/            # Training and deployment scripts
```

## Installation and Setup

### System Requirements

- **GPU**: NVIDIA GPU with compute capability 6.0+ (RTX series recommended)
- **VRAM**: 16GB+ for complex environments, 32GB+ for large-scale training
- **CUDA**: CUDA 11.8+ with compatible drivers
- **OS**: Ubuntu 20.04/22.04 or Windows 10/11
- **Python**: Python 3.8-3.11

### Installation Methods

#### Docker Installation (Recommended)

```bash
# Pull Isaac Orbit Docker image
docker pull nvcr.io/nvidia/isaac-orbit:latest

# Run Isaac Orbit container with GPU access
docker run --gpus all -it --rm \
  --network=host \
  --env "ACCEPT_EULA=Y" \
  --env "NVIDIA_VISIBLE_DEVICES=all" \
  --volume $(pwd):/workspace/current \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume $HOME/.Xauthority:/root/.Xauthority \
  --runtime=nvidia \
  --privileged \
  --device=/dev/snd \
  -e DISPLAY=$DISPLAY \
  nvcr.io/nvidia/isaac-orbit:latest
```

#### Pip Installation

```bash
# Create virtual environment
python -m venv orbit_env
source orbit_env/bin/activate  # On Windows: orbit_env\Scripts\activate

# Install Isaac Orbit
pip install orbit-starter

# Or install from source for latest features
git clone https://github.com/NVIDIA-Omniverse/orbit.git
cd orbit
pip install -e .
```

### Verification and Testing

```bash
# Verify installation
python -c "import omni.isaac.orbit as orbit; print(orbit.__version__)"

# Run a simple test environment
python -m orbit.examples.hello_world

# Check available environments
python -m orbit.core.envs.list_envs
```

## Environment Configuration

### Basic Environment Setup

```python
"""Basic Isaac Orbit Environment Configuration"""

import gymnasium as gym
from omni.isaac.orbit_assets import CARTPOLE_CFG
from omni.isaac.orbit.envs import RLTaskEnv, RLTaskEnvCfg
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.managers import EventTermCfg, SceneEntityCfg
from omni.isaac.orbit.utils import configclass


@configclass
class CartpoleEnvCfg(RLTaskEnvCfg):
    """Configuration for the cart-pole balancing environment."""

    def __post_init__(self):
        # General settings
        self.scene.num_envs = 2048
        self.scene.env_spacing = 2.0

        # Simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.cfg.physics_material


def main():
    """Run a simple cart-pole environment."""
    # Create environment
    env_cfg = CartpoleEnvCfg()
    env = gym.make("Isaac-Cartpole-v0", cfg=env_cfg)

    # Reset environment
    obs, _ = env.reset()

    # Run simulation
    num_steps = 1000
    for step in range(num_steps):
        # Random actions
        action = env.action_space.sample()

        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)

        # Reset if terminated
        if terminated.any() or truncated.any():
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
```

### Advanced Environment with Custom Robot

```python
"""Advanced Isaac Orbit Environment with Custom Robot"""

import gymnasium as gym
import torch
import numpy as np

from omni.isaac.orbit.assets import ArticulationCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg
from omni.isaac.orbit.utils import configclass


@configclass
class FrankaCubeLiftEnvCfg(RLTaskEnvCfg):
    """Configuration for the Franka picking and lifting environment."""

    def __post_init__(self):
        # General settings
        self.scene.num_envs = 2048
        self.scene.env_spacing = 2.0

        # Simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4

        # Scene
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn_func="omni.isaac.orbit_assets.franka.franka_panda",
            init_state={
                "joint_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.571,
                    "panda_joint7": 0.785,
                    "panda_finger_joint.*": 0.04,
                },
                "joint_vel": {".*": 0.0},
            },
            actuators={
                "panda_shoulder": {
                    "joint_names": ["panda_joint.*"],
                    "actuator_type": "position",
                    "stiffness": 800.0,
                    "damping": 40.0,
                    "a_range": (-85.0, 85.0),
                },
                "panda_finger": {
                    "joint_names": ["panda_finger_joint.*"],
                    "actuator_type": "position",
                    "stiffness": 200.0,
                    "damping": 20.0,
                    "a_range": (0.0, 0.04),
                },
            },
        )


def train_policy():
    """Train a policy using Isaac Orbit."""
    from omni.isaac.orbit_tasks.locomotion.velocity import mdp
    from omni.isaac.orbit_tasks import RL_TASK_ENVS

    # Create environment
    env_cfg = FrankaCubeLiftEnvCfg()
    env = gym.make("Isaac-Franka-Cube-Lift-v0", cfg=env_cfg)

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}")

    # Initialize RL agent (example with a simple random agent)
    num_steps = 10000

    obs, _ = env.reset()
    ep_returns = []
    ep_lengths = []
    current_ep_return = 0
    current_ep_length = 0

    for step in range(num_steps):
        # Sample random action
        action = torch.randn_like(env.action_space.sample())

        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)

        # Update episode statistics
        current_ep_return += reward.mean().item()
        current_ep_length += 1

        if terminated.any() or truncated.any():
            ep_returns.append(current_ep_return)
            ep_lengths.append(current_ep_length)
            current_ep_return = 0
            current_ep_length = 0
            env.reset()

        # Print progress
        if step % 1000 == 0:
            avg_return = np.mean(ep_returns[-10:]) if ep_returns else 0
            avg_length = np.mean(ep_lengths[-10:]) if ep_lengths else 0
            print(f"Step {step}, Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.2f}")

    env.close()


if __name__ == "__main__":
    train_policy()
```

## RL Training Configuration

### Training Algorithm Configuration

Isaac Orbit supports multiple RL algorithms through different frameworks:

```yaml
# configs/rl_games/franka_lift.yaml
params:
  seed: 42

  algo:
    name: 'rl_games'

  model:
    name: 'ActorCritic'
    separate: False
    actor_mlp:
      units: [512, 256, 128]
      activation: 'elu'
      d2rl: False
      layer_norm: False
    critic_mlp:
      units: [512, 256, 128]
      activation: 'elu'
      d2rl: False
      layer_norm: False
    init_noise_std: 1.0

  network:
    name: 'ActorCritic'
    separate: False

  load_checkpoint: False
  path: ''

  config:
    name: franka_lift
    env_name: 'franka_lift'
    device: 'cuda:0'
    device_name: 'cuda:0'
    multi_gpu: False
    ppo: True
    mixed_precision: False
    unscale_loss: False
    normalize_input: True
    normalize_value: True
    value_bootstrap: False

    gamma: 0.99
    tau: 0.95
    learning_rate: 3e-4
    lr_schedule: 'adaptive'
    schedule_type: 'legacy'
    kl_threshold: 0.01
    score_to_win: 20000
    max_epochs: 1000
    save_best_after: 100
    print_stats: True
    save_frequency: 50

    minibatch_size: 8192
    mini_epochs: 8
    critic_coef: 2
    clip_value: False

    clip_coef: 0.2
    entropy_coef: 0.0
    truncate_grads: True
    grad_norm: 1.0

    e_clip: 0.2
    horizon_length: 32
    max_agent_step: 2048
    mini_batch_trace: False

    no_terminal: False
    bounds_loss_coef: 0.0005
    normalize_advantage: True
    asymmetric_observations: False
    normalize_rms_obs: False
    discounted_rewards: True

    normalize_input: True
    normalize_value: True
    value_bootstrap: False

    experiment_name: 'franka_lift_ppo'
    run_name: 'ppo_franka_lift'
    model_dir: 'models/franka_lift/'
    log_dir: 'logs/franka_lift/'
```

### Curriculum Learning Configuration

```python
"""Curriculum learning configuration for progressive skill development"""

from omni.isaac.orbit.managers import CurriculumTermCfg


@configclass
class CuriculumEnvCfg:
    """Environment configuration with curriculum learning."""

    def __post_init__(self):
        # Define curriculum terms
        self.curriculum = {
            "object_distance": CurriculumTermCfg(
                initial_value=0.1,
                num_terms=10,
                schedule="linear",
                parameters={
                    "min_value": 0.1,
                    "max_value": 1.0,
                }
            ),
            "lift_height": CurriculumTermCfg(
                initial_value=0.05,
                num_terms=10,
                schedule="linear",
                parameters={
                    "min_value": 0.05,
                    "max_value": 0.5,
                }
            ),
            "noise_scale": CurriculumTermCfg(
                initial_value=0.01,
                num_terms=5,
                schedule="exponential",
                parameters={
                    "factor": 1.2,
                }
            )
        }


def implement_curriculum_learning():
    """Implement curriculum learning for skill acquisition."""

    # Start with simple tasks
    # Gradually increase difficulty based on performance
    # Use curriculum terms to modify environment parameters

    pass  # Implementation would depend on specific task requirements
```

## Domain Randomization

### Randomization Configuration

```python
"""Domain randomization for sim-to-real transfer"""

from omni.isaac.orbit.managers import EventTermCfg


@configclass
class DomainRandomizationEnvCfg:
    """Environment configuration with domain randomization."""

    def __post_init__(self):
        # Randomize physics properties
        self.events.physics_params = EventTermCfg(
            func="randomize_rigid_body_materials",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.7, 1.3),
                "dynamic_friction_range": (0.7, 1.3),
                "restitution_range": (0.7, 1.3),
                "num_buckets": 64,
            },
        )

        # Randomize joint properties
        self.events.joint_dynamics = EventTermCfg(
            func="randomize_joint_parameters",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_range": (0.9, 1.1),
                "damping_range": (0.9, 1.1),
            },
        )

        # Randomize mass properties
        self.events.mass_props = EventTermCfg(
            func="randomize_mass_and_inertia",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_ratio_range": (0.8, 1.2),
                "inertia_ratio_range": (0.8, 1.2),
            },
        )

        # Randomize actuator properties
        self.events.actuator_params = EventTermCfg(
            func="randomize_actuator_parameters",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "gear_ratio_range": (0.95, 1.05),
            },
        )

        # Randomize sensor noise
        self.events.sensor_noise = EventTermCfg(
            func="randomize_sensor_noise",
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "accelerometer_noise_std": (0.001, 0.001, 0.001),
                "gyroscope_noise_std": (0.001, 0.001, 0.001),
            },
        )

        # Randomize gravity
        self.events.gravity = EventTermCfg(
            func="randomize_rigid_body_gravity",
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "range": (-0.1, 0.1),
            },
        )
```

## Policy Deployment

### Converting Policies for Deployment

```python
"""Policy deployment and conversion utilities"""

import torch
import numpy as np


def export_policy_to_onnx(policy, filepath, input_shape):
    """Export trained policy to ONNX format for deployment."""

    policy.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    torch.onnx.export(
        policy,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )


def convert_policy_for_real_robot(policy_path, robot_config):
    """Convert Isaac Orbit policy for real robot deployment."""

    # Load trained policy
    policy = torch.load(policy_path)

    # Adjust for real robot differences
    # - Action scaling
    # - Observation normalization
    # - Control frequency adaptation

    # Create deployment wrapper
    class PolicyWrapper:
        def __init__(self, policy, robot_config):
            self.policy = policy
            self.robot_config = robot_config

        def __call__(self, observation):
            # Preprocess observation for real robot
            obs_tensor = torch.tensor(observation, dtype=torch.float32)

            # Get action from policy
            with torch.no_grad():
                action = self.policy(obs_tensor.unsqueeze(0))

            # Post-process action for real robot
            action = self.post_process_action(action.squeeze(0).numpy())

            return action

        def post_process_action(self, action):
            """Convert normalized action to robot-specific commands."""
            # Scale action to robot joint limits
            scaled_action = self.scale_action_to_robot_limits(action)

            # Apply any robot-specific transformations
            robot_command = self.apply_robot_transformations(scaled_action)

            return robot_command

        def scale_action_to_robot_limits(self, action):
            """Scale action to robot joint limits."""
            # Implementation would depend on robot configuration
            return action

        def apply_robot_transformations(self, action):
            """Apply robot-specific transformations."""
            # Implementation would depend on robot type
            return action

    return PolicyWrapper(policy, robot_config)


def deploy_policy_on_robot(policy_wrapper, robot_interface):
    """Deploy policy on real robot."""

    # Initialize robot
    robot_interface.initialize()

    # Main control loop
    while True:
        # Get observation from robot
        observation = robot_interface.get_observation()

        # Get action from policy
        action = policy_wrapper(observation)

        # Apply action to robot
        robot_interface.apply_action(action)

        # Small delay to maintain control frequency
        time.sleep(1.0 / robot_interface.control_frequency)
```

## Isaac Orbit Examples

### Mobile Manipulation Example

```python
"""Mobile manipulation environment example"""

import gymnasium as gym
import torch
import numpy as np

from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass


@configclass
class MobileManipulationEnvCfg(RLTaskEnvCfg):
    """Configuration for mobile manipulation environment."""

    def __post_init__(self):
        # General settings
        self.scene.num_envs = 1024
        self.scene.env_spacing = 3.0

        # Simulation settings
        self.sim.dt = 1.0 / 60.0  # Lower for more stable mobile base
        self.sim.render_interval = 4

        # Mobile manipulator configuration
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn_func="omni.isaac.orbit_assets.mobile_manipulator.mobile_manipulator",
            init_state={
                "joint_pos": {
                    ".*": 0.0,  # Initialize all joints to 0
                },
                "joint_vel": {".*": 0.0},
            },
            actuators={
                "mobile_base": {
                    "joint_names": ["base_wheel_.*"],
                    "actuator_type": "velocity",
                    "stiffness": 0.0,
                    "damping": 10.0,
                },
                "manipulator": {
                    "joint_names": ["arm_joint_.*"],
                    "actuator_type": "position",
                    "stiffness": 800.0,
                    "damping": 40.0,
                },
            },
        )

        # Target object to manipulate
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn_func="omni.isaac.orbit_assets.cube.cube",
            init_state={
                "pos": (0.5, 0.0, 0.1),
                "rot": (1.0, 0.0, 0.0, 0.0),
                "vel": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            },
        )

        # Define goals
        self.commands.base_command = SceneEntityCfg("robot", body_names="base_link")
        self.commands.arm_command = SceneEntityCfg("robot", joint_names="arm_joint_.*")


def train_mobile_manipulation_policy():
    """Train policy for mobile manipulation task."""

    # Create environment
    env_cfg = MobileManipulationEnvCfg()
    env = gym.make("Isaac-Mobile-Manipulation-v0", cfg=env_cfg)

    # Get dimensions
    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print(f"Observation dimension: {num_obs}")
    print(f"Action dimension: {num_actions}")

    # Initialize policy (simplified example)
    class SimplePolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, action_dim),
                torch.nn.Tanh()
            )

        def forward(self, obs):
            return self.network(obs)

    policy = SimplePolicy(num_obs, num_actions)

    # Training loop
    max_steps = 500000
    obs, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Get action from policy
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action = policy(obs_tensor).detach().numpy()

        # Apply action
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update statistics
        total_reward += reward.mean()

        # Reset if needed
        if terminated.any() or truncated.any():
            env.reset()
            print(f"Episode reward: {total_reward:.2f}")
            total_reward = 0

        # Update policy (simplified - in real training you'd use proper RL algorithm)
        obs = next_obs

        # Log progress
        if step % 10000 == 0:
            print(f"Training step: {step}")

    env.close()

    return policy


if __name__ == "__main__":
    trained_policy = train_mobile_manipulation_policy()

    # Export policy
    export_policy_to_onnx(trained_policy, "mobile_manipulation_policy.onnx",
                         (1, env.observation_space.shape[0]))
```

### Locomotion Example

```python
"""Legged locomotion environment example"""

from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.managers import TerminationTermCfg, RewardTermCfg


@configclass
class LocomotionEnvCfg:
    """Configuration for legged locomotion environment."""

    def __post_init__(self):
        # Termination conditions
        self.terminations.base_contact = TerminationTermCfg(
            func="check_base_contact",
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base.*")},
        )

        self.terminations.joint_torques = TerminationTermCfg(
            func="check_joint_torques",
            params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 1000.0},
        )

        # Reward terms
        self.rewards.track_lin_vel_xy_exp = RewardTermCfg(
            func="compute_track_lin_vel_xy_exp",
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.1},
        )

        self.rewards.track_ang_vel_z_exp = RewardTermCfg(
            func="compute_track_ang_vel_z_exp",
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.2},
        )

        self.rewards.lin_vel_z_l2 = RewardTermCfg(
            func="compute_lin_vel_z_l2",
            weight=-2.0,
        )

        self.rewards.ang_vel_xy_l2 = RewardTermCfg(
            func="compute_ang_vel_xy_l2",
            weight=-0.05,
        )

        self.rewards.dof_torques_l2 = RewardTermCfg(
            func="compute_dof_torques_l2",
            weight=-1e-5,
        )

        self.rewards.dof_acc_l2 = RewardTermCfg(
            func="compute_dof_acc_l2",
            weight=-1e-7,
        )

        self.rewards.action_rate_l2 = RewardTermCfg(
            func="compute_action_rate_l2",
            weight=-0.01,
        )

        self.rewards.energy = RewardTermCfg(
            func="compute_energy",
            weight=-5e-5,
        )

        self.rewards.feet_air_time = RewardTermCfg(
            func="compute_feet_air_time",
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )

        self.rewards.undesired_contacts = RewardTermCfg(
            func="compute_contacts",
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
                "threshold": 1.0,
            },
        )

        self.rewards.flat_orientation_l2 = RewardTermCfg(
            func="compute_flat_orientation_l2",
            weight=-5.0,
        )

        self.rewards.dof_pos_limits = RewardTermCfg(
            func="compute_dof_pos_limits",
            weight=-1.0,
        )
```

## Performance Optimization

### GPU Memory Management

```python
"""GPU memory optimization utilities"""

import torch
import gc


def optimize_training_memory():
    """Optimize GPU memory usage during training."""

    # Enable gradient checkpointing
    torch.utils.checkpoint = True

    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Batch size optimization
    def find_optimal_batch_size(env, policy):
        """Find optimal batch size for available GPU memory."""
        batch_sizes = [256, 512, 1024, 2048]

        for batch_size in batch_sizes:
            try:
                # Test with this batch size
                test_obs = torch.randn(batch_size, env.observation_space.shape[0])
                test_action = policy(test_obs)

                # If successful, this batch size works
                return batch_size

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch size {batch_size} too large, trying smaller...")
                    continue
                else:
                    raise e

        return 128  # Minimum fallback

    # Clear cache periodically
    def clear_gpu_cache():
        """Clear GPU cache to free memory."""
        torch.cuda.empty_cache()
        gc.collect()


def optimize_inference_performance():
    """Optimize policy inference for deployment."""

    # Use TensorRT for optimized inference
    import torch_tensorrt

    def optimize_policy_with_tensorrt(policy, input_shape):
        """Optimize policy with TensorRT."""
        policy.eval()

        # Compile with TensorRT
        optimized_policy = torch_tensorrt.compile(
            policy,
            inputs=[torch_tensorrt.Input(input_shape)],
            enabled_precisions={torch.float, torch.half},
            workspace_size=1 << 25
        )

        return optimized_policy
```

## Troubleshooting Tips

- If Isaac Orbit fails to initialize, verify GPU drivers and CUDA compatibility
- For poor training performance, check that GPU acceleration is enabled and batch sizes are optimized
- If policies don't transfer to real robots, implement proper domain randomization and sim-to-real techniques
- For unstable locomotion policies, adjust reward weights and termination conditions
- If environments crash frequently, reduce complexity or increase simulation stability parameters
- For memory issues, reduce batch sizes or use gradient checkpointing
- If training diverges, adjust learning rates and normalize observations/actions
- For contact instability, tune PhysX parameters and material properties
- If policies overfit to simulation, increase domain randomization range
- For deployment issues, verify that action/observation scaling is properly handled

## Cross-References

For related concepts, see:
- [ROS 2 Integration](../ros2-core/index.md) - For ROS communication
- [Simulation Environments](../gazebo/index.md) - For alternative simulation
- [Isaac ROS](../isaac-ros/index.md) - For perception integration
- [Unity Visualization](../unity/index.md) - For alternative visualization
- [Humanoid Robotics](../humanoid-kinematics/index.md) - For robot kinematics

## Summary

Isaac Orbit provides a powerful platform for reinforcement learning in robotics with GPU acceleration, domain randomization, and real-world deployment capabilities. When properly configured with appropriate environments and training parameters, Isaac Orbit enables complex robot skill acquisition that can be transferred to real hardware. Success requires understanding both the underlying RL algorithms and the specific simulation requirements for effective training.