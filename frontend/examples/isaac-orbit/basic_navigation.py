#!/usr/bin/env python3
"""
Basic Isaac Orbit Navigation Example

This example demonstrates fundamental navigation concepts using Isaac Orbit's
GPU-accelerated navigation stack.
"""

import numpy as np
import torch
import gymnasium as gym
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass


@configclass
class IsaacOrbitNavigationEnvCfg(RLTaskEnvCfg):
    """Configuration for the Isaac Orbit navigation environment."""

    def __post_init__(self):
        # Environment settings
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.0

        # Simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4

        # Episode length
        self.terminations.episode_length = True
        self.terminations.episode_length.time_range = (5.0, 10.0)

        # Rewards
        self.rewards.progress = True
        self.rewards.progress.weight = 1.0
        self.rewards.termination_penalty = True
        self.rewards.termination_penalty.weight = -10.0


def main():
    """Run the basic navigation example."""
    # Create environment
    env_cfg = IsaacOrbitNavigationEnvCfg()

    # In a real Isaac Orbit setup, you would create the environment like:
    # env = gym.make("IsaacOrbit-Navigation-v0", cfg=env_cfg)

    print("Isaac Orbit Navigation Example:")
    print("- Demonstrates GPU-accelerated navigation")
    print("- Shows environment configuration")
    print("- Illustrates reward and termination design")

    # For this example, we'll just show the configuration structure
    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Environment spacing: {env_cfg.scene.env_spacing}")
    print(f"Simulation timestep: {env_cfg.sim.dt}")


if __name__ == "__main__":
    main()