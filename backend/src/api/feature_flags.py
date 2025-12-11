"""
Feature Flags Module for the RAG + Agentic Backend for AI-Textbook Chatbot.

This module provides feature flagging capabilities for gradual rollouts including:
- Dynamic feature toggles
- Percentage-based rollouts
- User/group-based feature access
- Environment-specific configurations
- Real-time feature updates
"""

import json
import os
import threading
import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import fnmatch
import logging
from contextvars import ContextVar


class FeatureFlagType(Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_BASED = "user_based"
    ENVIRONMENT = "environment"
    GRADUAL_ROLLOUT = "gradual_rollout"


class RolloutStrategy(Enum):
    """Strategies for gradual rollouts"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM_SCHEDULE = "custom_schedule"
    BY_USER_ID = "by_user_id"
    BY_USER_GROUP = "by_user_group"


@dataclass
class FeatureFlag:
    """Representation of a feature flag"""
    name: str
    enabled: bool
    flag_type: FeatureFlagType
    rollout_percentage: float = 0.0
    user_ids: List[str] = None
    user_groups: List[str] = None
    environment: str = None
    rollout_strategy: RolloutStrategy = RolloutStrategy.LINEAR
    created_at: str = None
    updated_at: str = None
    description: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.user_ids is None:
            self.user_ids = []
        if self.user_groups is None:
            self.user_groups = []
        if self.metadata is None:
            self.metadata = {}


class FeatureFlagStore:
    """Storage for feature flags with hot reloading capabilities"""

    def __init__(self, config_file: str = None):
        self.flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.Lock()
        self.config_file = config_file or os.getenv("FEATURE_FLAGS_CONFIG", "feature_flags.json")
        self.last_modified = 0
        self.auto_reload = True

        # Load initial configuration
        self.load_flags()

        # Start auto-reload thread if enabled
        if self.auto_reload:
            self._start_auto_reload()

    def load_flags(self):
        """Load feature flags from configuration file"""
        try:
            if os.path.exists(self.config_file):
                stat = os.stat(self.config_file)
                if stat.st_mtime <= self.last_modified:
                    return  # No changes

                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)

                with self._lock:
                    self.flags.clear()
                    for flag_name, flag_data in config_data.items():
                        flag = self._dict_to_feature_flag(flag_name, flag_data)
                        self.flags[flag_name] = flag

                self.last_modified = stat.st_mtime
        except Exception as e:
            print(f"Error loading feature flags from {self.config_file}: {e}")

    def save_flags(self):
        """Save feature flags to configuration file"""
        try:
            with self._lock:
                config_data = {}
                for flag_name, flag in self.flags.items():
                    config_data[flag_name] = self._feature_flag_to_dict(flag)

            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.last_modified = os.stat(self.config_file).st_mtime
        except Exception as e:
            print(f"Error saving feature flags to {self.config_file}: {e}")

    def _dict_to_feature_flag(self, name: str, data: Dict[str, Any]) -> FeatureFlag:
        """Convert dictionary to FeatureFlag object"""
        return FeatureFlag(
            name=name,
            enabled=data.get('enabled', False),
            flag_type=FeatureFlagType(data.get('type', 'boolean')),
            rollout_percentage=data.get('rollout_percentage', 0.0),
            user_ids=data.get('user_ids', []),
            user_groups=data.get('user_groups', []),
            environment=data.get('environment'),
            rollout_strategy=RolloutStrategy(data.get('rollout_strategy', 'linear')),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )

    def _feature_flag_to_dict(self, flag: FeatureFlag) -> Dict[str, Any]:
        """Convert FeatureFlag object to dictionary"""
        return {
            'enabled': flag.enabled,
            'type': flag.flag_type.value,
            'rollout_percentage': flag.rollout_percentage,
            'user_ids': flag.user_ids,
            'user_groups': flag.user_groups,
            'environment': flag.environment,
            'rollout_strategy': flag.rollout_strategy.value,
            'created_at': flag.created_at,
            'updated_at': flag.updated_at,
            'description': flag.description,
            'metadata': flag.metadata
        }

    def _start_auto_reload(self):
        """Start auto-reload thread"""
        def reload_worker():
            while True:
                time.sleep(5)  # Check every 5 seconds
                if self.auto_reload:
                    self.load_flags()

        thread = threading.Thread(target=reload_worker, daemon=True)
        thread.start()

    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get a feature flag by name"""
        self.load_flags()  # Check for updates
        return self.flags.get(name)

    def set_flag(self, flag: FeatureFlag):
        """Set or update a feature flag"""
        with self._lock:
            flag.updated_at = datetime.utcnow().isoformat()
            self.flags[flag.name] = flag
        self.save_flags()

    def delete_flag(self, name: str):
        """Delete a feature flag"""
        with self._lock:
            if name in self.flags:
                del self.flags[name]
        self.save_flags()

    def list_flags(self) -> List[FeatureFlag]:
        """List all feature flags"""
        self.load_flags()  # Check for updates
        return list(self.flags.values())


class FeatureFlagEvaluator:
    """Evaluates feature flags based on various conditions"""

    def __init__(self, store: FeatureFlagStore):
        self.store = store

    def is_enabled(self, flag_name: str, user_id: str = None, user_group: str = None,
                   context: Dict[str, Any] = None) -> bool:
        """Evaluate if a feature flag is enabled for the given context"""
        flag = self.store.get_flag(flag_name)
        if not flag:
            return False

        # If flag is disabled globally, return False regardless of other conditions
        if not flag.enabled:
            return False

        # Check environment restriction
        if flag.environment:
            current_env = os.getenv("ENVIRONMENT", "development")
            if flag.environment != current_env:
                return False

        # Evaluate based on flag type
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            return flag.enabled
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            return self._evaluate_percentage_rollout(flag, user_id)
        elif flag.flag_type == FeatureFlagType.USER_BASED:
            return self._evaluate_user_based(flag, user_id, user_group)
        elif flag.flag_type == FeatureFlagType.GRADUAL_ROLLOUT:
            return self._evaluate_gradual_rollout(flag, user_id, user_group)
        else:
            return False

    def _evaluate_percentage_rollout(self, flag: FeatureFlag, user_id: str = None) -> bool:
        """Evaluate percentage-based rollout"""
        if flag.rollout_percentage <= 0:
            return False
        if flag.rollout_percentage >= 100:
            return True

        # If user_id is provided, use it for consistent hashing
        if user_id:
            hash_input = f"{flag.name}:{user_id}"
        else:
            # If no user_id, generate a random value for this evaluation
            import random
            hash_input = f"{flag.name}:{random.random()}"

        # Create a hash and convert to percentage
        hash_value = hash(hash_input) % 10000  # 0-9999
        percentage = hash_value / 100.0  # 0-99.99%

        return percentage < flag.rollout_percentage

    def _evaluate_user_based(self, flag: FeatureFlag, user_id: str = None, user_group: str = None) -> bool:
        """Evaluate user-based feature flag"""
        # Check if user is specifically enabled
        if user_id and user_id in flag.user_ids:
            return True

        # Check if user group is enabled
        if user_group and user_group in flag.user_groups:
            return True

        # If no specific users or groups are defined, check rollout percentage
        if not flag.user_ids and not flag.user_groups:
            return self._evaluate_percentage_rollout(flag, user_id)

        return False

    def _evaluate_gradual_rollout(self, flag: FeatureFlag, user_id: str = None, user_group: str = None) -> bool:
        """Evaluate gradual rollout feature flag"""
        if flag.rollout_strategy == RolloutStrategy.BY_USER_ID:
            return self._evaluate_percentage_rollout(flag, user_id)
        elif flag.rollout_strategy == RolloutStrategy.BY_USER_GROUP:
            # Use group name for consistent hashing if available
            if user_group:
                hash_input = f"{flag.name}:{user_group}"
            else:
                hash_input = f"{flag.name}:default"
            hash_value = hash(hash_input) % 10000
            percentage = hash_value / 100.0
            return percentage < flag.rollout_percentage
        else:
            # Default to linear rollout
            return self._evaluate_percentage_rollout(flag, user_id)

    def get_rollout_percentage(self, flag_name: str) -> float:
        """Get the current rollout percentage for a flag"""
        flag = self.store.get_flag(flag_name)
        if not flag:
            return 0.0
        return flag.rollout_percentage

    def update_rollout_percentage(self, flag_name: str, percentage: float):
        """Update the rollout percentage for a flag"""
        flag = self.store.get_flag(flag_name)
        if flag:
            flag.rollout_percentage = max(0.0, min(100.0, percentage))  # Clamp between 0-100
            flag.updated_at = datetime.utcnow().isoformat()
            self.store.set_flag(flag)


class FeatureFlagsManager:
    """Main interface for feature flags functionality"""

    def __init__(self, config_file: str = None):
        self.store = FeatureFlagStore(config_file)
        self.evaluator = FeatureFlagEvaluator(self.store)
        self._default_context = {}

    def is_enabled(self, flag_name: str, user_id: str = None, user_group: str = None,
                   context: Dict[str, Any] = None) -> bool:
        """Check if a feature is enabled for the given context"""
        return self.evaluator.is_enabled(flag_name, user_id, user_group, context)

    def enable_feature(self, flag_name: str, rollout_percentage: float = 100.0):
        """Enable a feature with optional rollout percentage"""
        flag = self.store.get_flag(flag_name)
        if flag:
            flag.enabled = True
            flag.rollout_percentage = rollout_percentage
            self.store.set_flag(flag)
        else:
            # Create new flag if it doesn't exist
            new_flag = FeatureFlag(
                name=flag_name,
                enabled=True,
                flag_type=FeatureFlagType.PERCENTAGE,
                rollout_percentage=rollout_percentage
            )
            self.store.set_flag(new_flag)

    def disable_feature(self, flag_name: str):
        """Disable a feature"""
        flag = self.store.get_flag(flag_name)
        if flag:
            flag.enabled = False
            self.store.set_flag(flag)

    def set_rollout_percentage(self, flag_name: str, percentage: float):
        """Set rollout percentage for a feature"""
        self.evaluator.update_rollout_percentage(flag_name, percentage)

    def create_flag(self, name: str, flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN,
                    enabled: bool = False, rollout_percentage: float = 0.0,
                    user_ids: List[str] = None, user_groups: List[str] = None,
                    description: str = ""):
        """Create a new feature flag"""
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            flag_type=flag_type,
            rollout_percentage=rollout_percentage,
            user_ids=user_ids or [],
            user_groups=user_groups or [],
            description=description
        )
        self.store.set_flag(flag)

    def list_features(self) -> List[FeatureFlag]:
        """List all feature flags"""
        return self.store.list_flags()

    def get_feature_status(self, flag_name: str) -> Dict[str, Any]:
        """Get detailed status of a feature flag"""
        flag = self.store.get_flag(flag_name)
        if not flag:
            return {"error": f"Feature flag '{flag_name}' not found"}

        return {
            "name": flag.name,
            "enabled": flag.enabled,
            "type": flag.flag_type.value,
            "rollout_percentage": flag.rollout_percentage,
            "user_ids": flag.user_ids,
            "user_groups": flag.user_groups,
            "environment": flag.environment,
            "created_at": flag.created_at,
            "updated_at": flag.updated_at,
            "description": flag.description
        }

    def gradual_rollout(self, flag_name: str, target_percentage: float, steps: int = 10,
                        interval: int = 60, callback: callable = None):
        """Gradually increase rollout percentage over time"""
        import asyncio

        async def rollout_process():
            current_percentage = self.evaluator.get_rollout_percentage(flag_name)
            step_increase = (target_percentage - current_percentage) / steps
            step_interval = interval / steps

            for i in range(steps):
                new_percentage = current_percentage + (step_increase * (i + 1))
                new_percentage = max(0.0, min(100.0, new_percentage))  # Clamp between 0-100

                self.set_rollout_percentage(flag_name, new_percentage)

                if callback:
                    callback(flag_name, new_percentage)

                await asyncio.sleep(step_interval)

            # Ensure we reach the target
            self.set_rollout_percentage(flag_name, target_percentage)
            if callback:
                callback(flag_name, target_percentage)

        # In a real implementation, you'd run this asynchronously
        # For now, we'll just set to target percentage directly
        self.set_rollout_percentage(flag_name, target_percentage)


# Global feature flags manager instance
feature_flags = FeatureFlagsManager()


# Decorator for conditional feature execution
def conditional_feature(feature_name: str, user_id_param: str = "user_id",
                      user_group_param: str = "user_group"):
    """Decorator to conditionally execute functions based on feature flags"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract user_id and user_group from kwargs or args
            user_id = kwargs.get(user_id_param)
            user_group = kwargs.get(user_group_param)

            # If not found in kwargs, try to find in args
            if not user_id and len(args) > 0:
                # This is a simplification - in practice you'd need to inspect the function signature
                pass

            if feature_flags.is_enabled(feature_name, user_id=user_id, user_group=user_group):
                return func(*args, **kwargs)
            else:
                # Return default value or raise an exception based on the function's purpose
                # For now, we'll return a default response
                return {"error": f"Feature '{feature_name}' is not enabled", "status": "feature_disabled"}
        return wrapper
    return decorator


# Context manager for feature flag evaluation
class FeatureContext:
    """Context manager for feature flag evaluation with additional context"""

    def __init__(self, feature_name: str, user_id: str = None, user_group: str = None,
                 context: Dict[str, Any] = None):
        self.feature_name = feature_name
        self.user_id = user_id
        self.user_group = user_group
        self.context = context or {}

    def __enter__(self):
        self.is_active = feature_flags.is_enabled(self.feature_name, self.user_id, self.user_group, self.context)
        return self.is_active

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Could log feature usage here
        pass


# Helper functions for common use cases
def can_use_new_ui(user_id: str = None, user_group: str = None) -> bool:
    """Check if user can use the new UI"""
    return feature_flags.is_enabled("new_ui", user_id, user_group)


def can_access_beta_features(user_id: str = None, user_group: str = None) -> bool:
    """Check if user can access beta features"""
    return feature_flags.is_enabled("beta_features", user_id, user_group)


def is_performance_optimization_enabled(user_id: str = None, user_group: str = None) -> bool:
    """Check if performance optimizations are enabled"""
    return feature_flags.is_enabled("performance_optimization", user_id, user_group)


# Example configuration for feature flags
EXAMPLE_FEATURE_FLAGS_CONFIG = {
    "new_ui": {
        "enabled": True,
        "type": "gradual_rollout",
        "rollout_percentage": 10.0,
        "rollout_strategy": "by_user_id",
        "user_ids": ["admin_user"],
        "user_groups": ["beta_testers"],
        "description": "New user interface with improved UX",
        "metadata": {
            "target_completion": "2024-12-31",
            "depends_on": []
        }
    },
    "advanced_search": {
        "enabled": False,
        "type": "percentage",
        "rollout_percentage": 0.0,
        "description": "Advanced search functionality with filters and facets",
        "metadata": {
            "target_completion": "2025-01-15",
            "depends_on": ["search_index_v2"]
        }
    },
    "beta_features": {
        "enabled": True,
        "type": "user_based",
        "user_groups": ["beta_testers", "premium_users"],
        "description": "Access to experimental features",
        "metadata": {
            "target_completion": "ongoing",
            "depends_on": []
        }
    },
    "rate_limit_relaxation": {
        "enabled": False,
        "type": "percentage",
        "rollout_percentage": 0.0,
        "description": "Relaxed rate limits for premium users",
        "metadata": {
            "target_completion": "2024-12-20",
            "depends_on": ["billing_integration"]
        }
    }
}


def initialize_feature_flags():
    """Initialize feature flags with default configuration"""
    # Check if config file exists, if not create with defaults
    config_file = os.getenv("FEATURE_FLAGS_CONFIG", "feature_flags.json")

    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump(EXAMPLE_FEATURE_FLAGS_CONFIG, f, indent=2)
        print(f"Created default feature flags configuration at {config_file}")


# Initialize feature flags
initialize_feature_flags()


if __name__ == "__main__":
    # Example usage
    print("Feature Flags Module Initialized")

    # Create some example flags
    feature_flags.create_flag(
        name="experimental_chat",
        flag_type=FeatureFlagType.GRADUAL_ROLLOUT,
        enabled=True,
        rollout_percentage=5.0,
        user_groups=["early_adopters"],
        description="Experimental chat interface"
    )

    # Check if features are enabled
    print(f"New UI enabled: {feature_flags.is_enabled('new_ui', user_id='user123', user_group='regular_users')}")
    print(f"Beta features enabled: {feature_flags.is_enabled('beta_features', user_id='user456', user_group='beta_testers')}")
    print(f"Experimental chat enabled: {feature_flags.is_enabled('experimental_chat', user_id='user789', user_group='early_adopters')}")

    # Get feature status
    print("\nFeature Status:")
    print(feature_flags.get_feature_status('new_ui'))

    # List all features
    print("\nAll Features:")
    for feature in feature_flags.list_features():
        print(f"- {feature.name}: {feature.enabled} ({feature.rollout_percentage}%)")

    # Test gradual rollout
    print(f"\nBefore rollout - Advanced search enabled: {feature_flags.is_enabled('advanced_search')}")
    feature_flags.enable_feature('advanced_search', 100.0)
    print(f"After enabling - Advanced search enabled: {feature_flags.is_enabled('advanced_search')}")