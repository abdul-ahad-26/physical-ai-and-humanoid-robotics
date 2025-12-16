---
title: Hardware Labs - Code Examples
sidebar_label: Hardware Labs Code Examples
description: Code examples for hardware requirements and lab architecture in Physical AI and Humanoid Robotics
---

# Hardware Labs - Code Examples

## Overview

This chapter provides code examples for implementing hardware requirements assessment, lab infrastructure setup, and component integration for Physical AI and Humanoid Robotics applications. The examples demonstrate best practices for hardware abstraction, configuration management, and system validation.

## Hardware Configuration Management

### Hardware Specification Schema

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class HardwareType(Enum):
    COMPUTE = "compute"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COMMUNICATION = "communication"
    POWER = "power"

class ComputePlatform(Enum):
    WORKSTATION = "workstation"
    EDGE_DEVICE = "edge_device"
    SINGLE_BOARD = "single_board"
    CLOUD_INSTANCE = "cloud_instance"

class SensorType(Enum):
    RGB_CAMERA = "rgb_camera"
    RGBD_CAMERA = "rgbd_camera"
    LIDAR = "lidar"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"

@dataclass
class ComputeSpec:
    platform: ComputePlatform
    cpu_cores: int
    cpu_model: str
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[int] = None
    ram_gb: int = 16
    storage_type: str = "SSD"
    storage_gb: int = 512

@dataclass
class SensorSpec:
    sensor_type: SensorType
    model: str
    resolution: Optional[str] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    fov_horizontal: Optional[float] = None
    fov_vertical: Optional[float] = None
    update_rate_hz: float = 30.0

@dataclass
class HardwareConfiguration:
    name: str
    description: str
    compute: ComputeSpec
    sensors: List[SensorSpec]
    power_budget_watts: float
    cost_estimate_usd: float
    weight_kg: float
    dimensions_m: Dict[str, float]  # width, height, depth
```

### Hardware Validation Class

```python
import json
from typing import List, Dict, Any
from dataclasses import asdict

class HardwareValidator:
    def __init__(self):
        self.specification_templates = {
            "mobile_robot": {
                "min_gpu_memory": 8,
                "min_ram": 16,
                "min_compute_units": 1024,
                "max_power_consumption": 500,
                "required_sensors": ["lidar", "camera", "imu"]
            },
            "humanoid_robot": {
                "min_gpu_memory": 16,
                "min_ram": 32,
                "min_compute_units": 2048,
                "max_power_consumption": 1000,
                "required_sensors": ["rgbd_camera", "imu", "force_torque"]
            },
            "research_platform": {
                "min_gpu_memory": 24,
                "min_ram": 64,
                "min_compute_units": 4096,
                "max_power_consumption": 2000,
                "required_sensors": ["lidar", "rgbd_camera", "imu", "force_torque"]
            }
        }

    def validate_configuration(self, config: HardwareConfiguration, platform_type: str) -> Dict[str, Any]:
        """Validate hardware configuration against platform requirements."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "score": 0.0
        }

        template = self.specification_templates.get(platform_type)
        if not template:
            results["errors"].append(f"Unknown platform type: {platform_type}")
            results["valid"] = False
            return results

        # Validate compute requirements
        if config.compute.gpu_memory_gb and config.compute.gpu_memory_gb < template["min_gpu_memory"]:
            results["errors"].append(
                f"GPU memory ({config.compute.gpu_memory_gb}GB) below minimum ({template['min_gpu_memory']}GB)"
            )
            results["valid"] = False

        if config.compute.ram_gb < template["min_ram"]:
            results["errors"].append(
                f"RAM ({config.compute.ram_gb}GB) below minimum ({template['min_ram']}GB)"
            )
            results["valid"] = False

        # Validate power consumption
        if config.power_budget_watts > template["max_power_consumption"]:
            results["warnings"].append(
                f"Power consumption ({config.power_budget_watts}W) exceeds recommended ({template['max_power_consumption']}W)"
            )

        # Validate required sensors
        sensor_types = [sensor.sensor_type.value for sensor in config.sensors]
        for required_sensor in template["required_sensors"]:
            if not any(required_sensor in sensor_type for sensor_type in sensor_types):
                results["errors"].append(f"Missing required sensor: {required_sensor}")
                results["valid"] = False

        # Calculate validation score
        total_checks = 4  # compute, ram, power, sensors
        passed_checks = total_checks - len(results["errors"])
        results["score"] = passed_checks / total_checks if total_checks > 0 else 0.0

        return results

    def generate_cost_estimate(self, config: HardwareConfiguration) -> Dict[str, float]:
        """Generate detailed cost breakdown for hardware configuration."""
        cost_breakdown = {
            "compute": 0.0,
            "sensors": 0.0,
            "actuators": 0.0,
            "power_system": 0.0,
            "communication": 0.0,
            "miscellaneous": 0.0,
            "total": 0.0
        }

        # Compute cost estimation
        if config.compute.gpu_model:
            gpu_costs = {
                "RTX 4080": 1200,
                "RTX 4090": 1600,
                "RTX 6000 Ada": 4000,
                "A6000": 4500,
                "H100": 25000
            }
            cost_breakdown["compute"] = gpu_costs.get(config.compute.gpu_model, 1000)

        # Add CPU and RAM costs
        cost_breakdown["compute"] += config.compute.cpu_cores * 50  # Approximate cost per core
        cost_breakdown["compute"] += config.compute.ram_gb * 10    # Approximate cost per GB RAM

        # Sensor costs
        sensor_costs = {
            SensorType.RGB_CAMERA: 100,
            SensorType.RGBD_CAMERA: 300,
            SensorType.LIDAR: 2000,
            SensorType.IMU: 150,
            SensorType.FORCE_TORQUE: 5000
        }

        for sensor in config.sensors:
            cost_breakdown["sensors"] += sensor_costs.get(sensor.sensor_type, 200)

        cost_breakdown["total"] = sum(v for v in cost_breakdown.values())
        return cost_breakdown
```

## Lab Infrastructure Setup

### Network Configuration

```python
import subprocess
import json
from typing import Dict, List, Optional

class NetworkConfiguration:
    def __init__(self, config_file: str = "network_config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load network configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.default_config()

    def default_config(self) -> Dict[str, Any]:
        """Return default network configuration."""
        return {
            "subnet": "192.168.1.0/24",
            "gateway": "192.168.1.1",
            "dns_servers": ["8.8.8.8", "1.1.1.1"],
            "reserved_ips": {
                "robot1": "192.168.1.100",
                "robot2": "192.168.1.101",
                "workstation": "192.168.1.50"
            },
            "security": {
                "firewall_enabled": True,
                "ports": {
                    "ssh": 22,
                    "ros": 11311,
                    "http": 80,
                    "https": 443
                }
            }
        }

    def setup_network_segmentation(self):
        """Setup network segmentation for robotics lab."""
        # Create network namespaces for different robot types
        robot_types = ["mobile", "manipulator", "humanoid"]

        for robot_type in robot_types:
            namespace = f"robot_{robot_type}"
            subprocess.run(["ip", "netns", "add", namespace], check=True)

            # Configure namespace network settings
            subprocess.run([
                "ip", "netns", "exec", namespace,
                "ip", "addr", "add", f"10.0.{robot_types.index(robot_type)+1}.1/24", "dev", "lo"
            ], check=True)

            print(f"Created network namespace: {namespace}")

    def configure_firewall_rules(self):
        """Configure firewall rules for robotics lab."""
        # Allow ROS communication
        subprocess.run([
            "ufw", "allow", "from", self.config["subnet"], "to", "any", "port",
            str(self.config["security"]["ports"]["ros"])
        ], check=True)

        # Allow SSH access
        subprocess.run([
            "ufw", "allow", "from", self.config["subnet"], "to", "any", "port",
            str(self.config["security"]["ports"]["ssh"])
        ], check=True)

        # Enable firewall
        subprocess.run(["ufw", "--force", "enable"], check=True)

    def setup_vpn_access(self):
        """Setup VPN access for remote lab access."""
        # This is a simplified example - in practice, you'd use OpenVPN or similar
        vpn_config = {
            "server": "lab-vpn.example.com",
            "port": 1194,
            "protocol": "udp",
            "certificates": {
                "ca": "/etc/vpn/ca.crt",
                "server": "/etc/vpn/server.crt",
                "key": "/etc/vpn/server.key"
            },
            "network": "10.8.0.0/24"
        }

        print("VPN configuration prepared:")
        print(json.dumps(vpn_config, indent=2))
```

### Safety System Implementation

```python
import threading
import time
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class SafetyZone:
    name: str
    boundaries: List[float]  # [x_min, x_max, y_min, y_max, z_min, z_max]
    priority: int  # Higher number = higher priority
    active: bool = True

class SafetySystem:
    def __init__(self):
        self.safety_zones = []
        self.emergency_stop_active = False
        self.safety_callbacks = []
        self.monitoring_thread = None
        self.running = False

    def add_safety_zone(self, zone: SafetyZone):
        """Add a safety zone to the system."""
        self.safety_zones.append(zone)
        self.safety_zones.sort(key=lambda z: z.priority, reverse=True)

    def add_safety_callback(self, callback: Callable):
        """Add a callback function to be called when safety event occurs."""
        self.safety_callbacks.append(callback)

    def emergency_stop(self):
        """Activate emergency stop."""
        self.emergency_stop_active = True
        self._trigger_callbacks("EMERGENCY_STOP")

    def reset_emergency_stop(self):
        """Reset emergency stop."""
        self.emergency_stop_active = False
        self._trigger_callbacks("EMERGENCY_STOP_RESET")

    def _trigger_callbacks(self, event_type: str):
        """Trigger all registered safety callbacks."""
        for callback in self.safety_callbacks:
            try:
                callback(event_type)
            except Exception as e:
                print(f"Error in safety callback: {e}")

    def check_safety_conditions(self, robot_position: List[float]) -> bool:
        """Check if robot position violates any safety zones."""
        if self.emergency_stop_active:
            return False

        for zone in self.safety_zones:
            if not zone.active:
                continue

            x, y, z = robot_position
            x_min, x_max, y_min, y_max, z_min, z_max = zone.boundaries

            if (x_min <= x <= x_max and
                y_min <= y <= y_max and
                z_min <= z <= z_max):
                # Violation detected
                self._trigger_callbacks(f"ZONE_VIOLATION_{zone.name}")
                return False

        return True

    def start_monitoring(self):
        """Start safety monitoring thread."""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            # In a real implementation, this would get robot positions
            # from the robot's localization system
            robot_positions = self._get_robot_positions()

            for robot_id, position in robot_positions.items():
                if not self.check_safety_conditions(position):
                    print(f"SAFETY VIOLATION: Robot {robot_id} in restricted area")
                    self.emergency_stop()
                    break

            time.sleep(0.1)  # 100ms monitoring interval

    def _get_robot_positions(self) -> Dict[str, List[float]]:
        """Get current robot positions (mock implementation)."""
        # In practice, this would interface with localization systems
        return {
            "robot1": [1.0, 2.0, 0.0],
            "robot2": [3.0, 4.0, 0.0]
        }
```

## Sensor Integration and Calibration

### Sensor Abstraction Layer

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class SensorInterface(ABC):
    """Abstract base class for sensor interfaces."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the sensor."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the sensor."""
        pass

    @abstractmethod
    def read_data(self) -> Any:
        """Read data from the sensor."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get sensor metadata."""
        pass

class CameraInterface(SensorInterface):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.connection = None
        self.calibration_data = None

    def connect(self) -> bool:
        """Connect to camera using OpenCV."""
        import cv2
        self.connection = cv2.VideoCapture(self.device_id)
        return self.connection.isOpened()

    def disconnect(self):
        """Disconnect from camera."""
        if self.connection:
            self.connection.release()
            self.connection = None

    def read_data(self) -> np.ndarray:
        """Read image data from camera."""
        if self.connection:
            ret, frame = self.connection.read()
            if ret:
                return frame
        return np.array([])

    def get_metadata(self) -> Dict[str, Any]:
        """Get camera metadata."""
        if self.connection:
            return {
                "width": int(self.connection.get(3)),
                "height": int(self.connection.get(4)),
                "fps": self.connection.get(5),
                "format": self.connection.get(8)
            }
        return {}

    def calibrate_camera(self, calibration_images: List[np.ndarray]) -> Dict[str, Any]:
        """Calibrate camera using chessboard pattern."""
        import cv2
        import glob

        # Prepare object points
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            self.calibration_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist()
            }

            return self.calibration_data

        return {}

class LIDARInterface(SensorInterface):
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.connection = None

    def connect(self) -> bool:
        """Connect to LIDAR device."""
        try:
            import serial
            self.connection = serial.Serial(self.port, self.baudrate)
            return True
        except Exception as e:
            print(f"LIDAR connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from LIDAR."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def read_data(self) -> Dict[str, Any]:
        """Read LIDAR data."""
        if self.connection and self.connection.in_waiting > 0:
            data = self.connection.readline().decode().strip()
            # Parse LIDAR data (simplified example)
            try:
                # Assuming data format: "angle,distance,intensity"
                parts = data.split(',')
                return {
                    "angle": float(parts[0]),
                    "distance": float(parts[1]),
                    "intensity": float(parts[2]) if len(parts) > 2 else 1.0
                }
            except:
                return {}
        return {}

    def get_metadata(self) -> Dict[str, Any]:
        """Get LIDAR metadata."""
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "type": "2D LIDAR"
        }

    def get_scan(self, num_points: int = 360) -> List[Dict[str, float]]:
        """Get complete LIDAR scan."""
        scan_data = []
        for _ in range(num_points):
            point = self.read_data()
            if point:
                scan_data.append(point)
        return scan_data
```

## Power System Management

### Power Monitoring Class

```python
import time
import threading
from typing import Dict, List, Callable
from dataclasses import dataclass

@dataclass
class PowerReading:
    timestamp: float
    voltage: float
    current: float
    power: float
    temperature: float

class PowerSystemMonitor:
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.readings: List[PowerReading] = []
        self.callbacks: List[Callable] = []
        self.monitoring_thread = None
        self.running = False
        self.battery_capacity_wh = 100.0  # Example: 100Wh battery
        self.current_charge_wh = 100.0

    def add_callback(self, callback: Callable[[PowerReading], None]):
        """Add callback for power events."""
        self.callbacks.append(callback)

    def simulate_power_reading(self) -> PowerReading:
        """Simulate power reading (in real system, this would interface with hardware)."""
        import random
        return PowerReading(
            timestamp=time.time(),
            voltage=12.0 + random.uniform(-0.1, 0.1),  # 12V ± 0.1V
            current=5.0 + random.uniform(-1.0, 1.0),   # 5A ± 1A
            power=60.0 + random.uniform(-10.0, 10.0),  # 60W ± 10W
            temperature=25.0 + random.uniform(0, 10.0) # 25°C + heat
        )

    def start_monitoring(self):
        """Start power monitoring thread."""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop power monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            reading = self.simulate_power_reading()
            self.readings.append(reading)

            # Update battery charge (discharge simulation)
            power_consumption_w = reading.power
            time_elapsed = self.update_interval
            energy_consumed_wh = (power_consumption_w * time_elapsed) / 3600.0
            self.current_charge_wh -= energy_consumption_wh

            # Trigger callbacks
            for callback in self.callbacks:
                try:
                    callback(reading)
                except Exception as e:
                    print(f"Error in power callback: {e}")

            time.sleep(self.update_interval)

    def get_average_power(self, window_seconds: int = 60) -> float:
        """Get average power consumption over specified window."""
        current_time = time.time()
        recent_readings = [
            r for r in self.readings
            if current_time - r.timestamp <= window_seconds
        ]

        if recent_readings:
            return sum(r.power for r in recent_readings) / len(recent_readings)
        return 0.0

    def get_battery_level_percent(self) -> float:
        """Get current battery level as percentage."""
        return max(0.0, (self.current_charge_wh / self.battery_capacity_wh) * 100.0)

    def get_estimated_runtime_minutes(self) -> float:
        """Get estimated remaining runtime."""
        avg_power = self.get_average_power(300)  # 5-minute average
        if avg_power > 0:
            remaining_energy = self.current_charge_wh
            return (remaining_energy / avg_power) * 60  # Convert to minutes
        return float('inf')
```

## System Integration and Validation

### Hardware Integration Test Suite

```python
import unittest
from typing import Dict, Any

class HardwareIntegrationTest(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.validator = HardwareValidator()
        self.safety_system = SafetySystem()

        # Create test hardware configuration
        self.test_config = HardwareConfiguration(
            name="Test Robot",
            description="Test configuration for validation",
            compute=ComputeSpec(
                platform=ComputePlatform.WORKSTATION,
                cpu_cores=16,
                cpu_model="AMD Ryzen 9 5900X",
                gpu_model="RTX 4080",
                gpu_memory_gb=16,
                ram_gb=32,
                storage_gb=1000
            ),
            sensors=[
                SensorSpec(
                    sensor_type=SensorType.LIDAR,
                    model="Velodyne VLP-16",
                    range_min=0.2,
                    range_max=100.0,
                    update_rate_hz=10.0
                ),
                SensorSpec(
                    sensor_type=SensorType.RGBD_CAMERA,
                    model="Intel RealSense D435i",
                    resolution="1920x1080",
                    fov_horizontal=69.4,
                    fov_vertical=42.5,
                    update_rate_hz=30.0
                )
            ],
            power_budget_watts=300.0,
            cost_estimate_usd=5000.0,
            weight_kg=10.0,
            dimensions_m={"width": 0.5, "height": 0.5, "depth": 0.5}
        )

    def test_hardware_validation(self):
        """Test hardware configuration validation."""
        results = self.validator.validate_configuration(
            self.test_config, "mobile_robot"
        )

        self.assertTrue(results["valid"],
                       f"Configuration should be valid: {results['errors']}")
        self.assertGreater(results["score"], 0.7,
                          "Validation score should be above threshold")

    def test_cost_estimation(self):
        """Test cost estimation functionality."""
        cost_breakdown = self.validator.generate_cost_estimate(self.test_config)

        self.assertIn("total", cost_breakdown)
        self.assertGreater(cost_breakdown["total"], 0)

        # The estimated cost should be reasonably close to the provided estimate
        estimated_total = sum(v for v in cost_breakdown.values() if isinstance(v, (int, float)))
        self.assertAlmostEqual(
            estimated_total,
            self.test_config.cost_estimate_usd,
            delta=self.test_config.cost_estimate_usd * 0.3  # 30% tolerance
        )

    def test_safety_zone_violation(self):
        """Test safety zone violation detection."""
        # Add a safety zone
        test_zone = SafetyZone(
            name="restricted_area",
            boundaries=[0.0, 2.0, 0.0, 2.0, 0.0, 1.0],  # x: 0-2, y: 0-2, z: 0-1
            priority=1
        )
        self.safety_system.add_safety_zone(test_zone)

        # Test position inside zone (should violate)
        inside_position = [1.0, 1.0, 0.5]
        self.assertFalse(
            self.safety_system.check_safety_conditions(inside_position),
            "Position inside zone should violate safety"
        )

        # Test position outside zone (should be safe)
        outside_position = [3.0, 3.0, 0.5]
        self.assertTrue(
            self.safety_system.check_safety_conditions(outside_position),
            "Position outside zone should be safe"
        )

    def test_power_monitoring(self):
        """Test power system monitoring."""
        monitor = PowerSystemMonitor(update_interval=0.1)

        readings_collected = []

        def capture_reading(reading: PowerReading):
            readings_collected.append(reading)

        monitor.add_callback(capture_reading)
        monitor.start_monitoring()

        # Collect readings for a short period
        time.sleep(0.5)
        monitor.stop_monitoring()

        self.assertGreater(len(readings_collected), 0,
                          "Should have collected power readings")

        # Check that readings have reasonable values
        for reading in readings_collected:
            self.assertGreater(reading.voltage, 0, "Voltage should be positive")
            self.assertGreaterEqual(reading.current, 0, "Current should be non-negative")
            self.assertGreaterEqual(reading.power, 0, "Power should be non-negative")

    def tearDown(self):
        """Clean up after tests."""
        pass

def run_hardware_tests():
    """Run all hardware integration tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)

# Example usage
if __name__ == "__main__":
    # Run tests
    run_hardware_tests()

    # Example of creating and validating a hardware configuration
    validator = HardwareValidator()

    # Create a configuration for a humanoid robot
    humanoid_config = HardwareConfiguration(
        name="Humanoid Robot Platform",
        description="Advanced humanoid robot for research",
        compute=ComputeSpec(
            platform=ComputePlatform.WORKSTATION,
            cpu_cores=32,
            cpu_model="AMD EPYC",
            gpu_model="RTX 6000 Ada",
            gpu_memory_gb=48,
            ram_gb=64,
            storage_gb=2000
        ),
        sensors=[
            SensorSpec(
                sensor_type=SensorType.RGBD_CAMERA,
                model="Azure Kinect",
                resolution="2160x1208",
                fov_horizontal=75.0,
                fov_vertical=45.0,
                update_rate_hz=30.0
            ),
            SensorSpec(
                sensor_type=SensorType.IMU,
                model="VectorNav VN-100",
                update_rate_hz=1000.0
            ),
            SensorSpec(
                sensor_type=SensorType.FORCE_TORQUE,
                model="ATI Gamma",
                update_rate_hz=1000.0
            )
        ],
        power_budget_watts=800.0,
        cost_estimate_usd=15000.0,
        weight_kg=25.0,
        dimensions_m={"width": 0.8, "height": 1.5, "depth": 0.6}
    )

    # Validate the configuration
    results = validator.validate_configuration(humanoid_config, "humanoid_robot")
    print(f"Validation Results: {results}")

    # Generate cost estimate
    cost_breakdown = validator.generate_cost_estimate(humanoid_config)
    print(f"Cost Breakdown: {json.dumps(cost_breakdown, indent=2)}")
```

## Lab Setup Automation Script

```bash
#!/bin/bash

# Hardware Lab Setup Automation Script
# This script automates the setup of a robotics lab environment

set -e  # Exit on any error

echo "Starting Robotics Lab Setup..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."
if ! command_exists docker; then
    echo "Docker is required but not installed. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
fi

if ! command_exists git; then
    echo "Git is required but not installed. Installing..."
    sudo apt update
    sudo apt install -y git
fi

# Create lab directory structure
echo "Creating lab directory structure..."
mkdir -p ~/robotics-lab/{config,data,scripts,logs,backups}

# Clone common robotics repositories
echo "Cloning common robotics repositories..."
cd ~/robotics-lab
git clone https://github.com/ros2/ros2.git --branch humble --depth 1
git clone https://github.com/RobotLocomotion/drake.git --depth 1
git clone https://github.com/erdos-project/erdos.git --depth 1

# Install common dependencies
echo "Installing common dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libbullet-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    wget \
    python3-pip \
    libasio-dev \
    libtinyxml2-dev \
    libcunit1-dev

# Install Python packages
pip3 install \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    transforms3d \
    pyquaternion \
    requests \
    flask \
    plotly

# Setup ROS 2 environment
echo "Setting up ROS 2 environment..."
source /opt/ros/humble/setup.bash

# Create network configuration
echo "Creating network configuration..."
cat > ~/robotics-lab/config/network.conf << EOF
# Robotics Lab Network Configuration
SUBNET=192.168.1.0/24
GATEWAY=192.168.1.1
DNS_SERVERS="8.8.8.8 1.1.1.1"
RESERVED_IPS="192.168.1.100 192.168.1.101 192.168.1.102"
EOF

# Setup firewall rules
echo "Setting up firewall rules..."
sudo ufw allow 22    # SSH
sudo ufw allow 11311 # ROS master
sudo ufw allow 8080  # Web interfaces
sudo ufw --force enable

# Create safety monitoring service
echo "Creating safety monitoring service..."
sudo tee /etc/systemd/system/safety-monitor.service > /dev/null << EOF
[Unit]
Description=Robotics Lab Safety Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/robotics-lab
ExecStart=/usr/bin/python3 /home/$USER/robotics-lab/scripts/safety_monitor.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl enable safety-monitor.service
sudo systemctl start safety-monitor.service

# Setup data backup
echo "Setting up data backup..."
mkdir -p ~/robotics-lab/backups/daily
crontab -l | { cat; echo "0 2 * * * /home/$USER/robotics-lab/scripts/backup_data.sh"; } | crontab -

echo "Robotics Lab setup completed successfully!"
echo "Please log out and log back in to apply Docker group membership."
echo "Lab directory: ~/robotics-lab"
```

## Summary

This chapter provided comprehensive code examples for hardware requirements assessment, lab infrastructure setup, and system validation for Physical AI and Humanoid Robotics applications. The examples include:

1. Hardware configuration management with validation and cost estimation
2. Network configuration and safety system implementation
3. Sensor abstraction layer with camera and LIDAR interfaces
4. Power system monitoring with battery management
5. Hardware integration test suite for validation
6. Lab setup automation script for deployment

These examples demonstrate best practices for hardware abstraction, configuration management, and system validation in robotics applications. The modular design allows for easy extension and customization based on specific project requirements.