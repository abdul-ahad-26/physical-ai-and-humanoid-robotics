---
title: Cloud vs. On-Premise Workflows - Code Examples
sidebar_label: Cloud vs. On-Premise Code Examples
description: Code examples for comparing cloud and on-premise workflows in Physical AI and Humanoid Robotics
---

# Cloud vs. On-Premise Workflows - Code Examples

## Overview

This chapter provides code examples for implementing, comparing, and managing cloud and on-premise workflows in Physical AI and Humanoid Robotics applications. The examples demonstrate hybrid architectures, cost optimization, performance benchmarking, and deployment strategies for different computing environments.

## Cloud vs. On-Premise Architecture Patterns

### Hybrid Computing Abstraction Layer

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import threading
from dataclasses import dataclass

@dataclass
class ComputeResource:
    resource_id: str
    location: str  # "cloud" or "onprem"
    gpu_model: Optional[str] = None
    cpu_cores: int = 4
    memory_gb: int = 16
    bandwidth_mbps: float = 100.0
    cost_per_hour: float = 0.0
    availability: float = 0.99

class ComputingBackend(ABC):
    """Abstract base class for computing backends."""

    @abstractmethod
    def execute_task(self, task_name: str, data: Any, timeout: float = 30.0) -> Dict[str, Any]:
        """Execute a task on the computing backend."""
        pass

    @abstractmethod
    def get_resource_info(self) -> ComputeResource:
        """Get information about the computing resource."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the computing backend is available."""
        pass

class CloudBackend(ComputingBackend):
    """Cloud computing backend implementation."""

    def __init__(self, provider: str = "aws", instance_type: str = "g4dn.xlarge"):
        self.provider = provider
        self.instance_type = instance_type
        self.resource_info = self._get_resource_spec()

    def _get_resource_spec(self) -> ComputeResource:
        """Get resource specifications based on instance type."""
        if "g4dn" in self.instance_type:
            return ComputeResource(
                resource_id=f"{self.provider}-{self.instance_type}",
                location="cloud",
                gpu_model="NVIDIA T4",
                cpu_cores=4,
                memory_gb=16,
                bandwidth_mbps=2500.0,
                cost_per_hour=0.526,
                availability=0.999
            )
        elif "p3" in self.instance_type:
            return ComputeResource(
                resource_id=f"{self.provider}-{self.instance_type}",
                location="cloud",
                gpu_model="NVIDIA V100",
                cpu_cores=8,
                memory_gb=61,
                bandwidth_mbps=10000.0,
                cost_per_hour=3.06,
                availability=0.999
            )
        else:
            return ComputeResource(
                resource_id=f"{self.provider}-{self.instance_type}",
                location="cloud",
                cpu_cores=4,
                memory_gb=16,
                bandwidth_mbps=1000.0,
                cost_per_hour=0.192,
                availability=0.999
            )

    def execute_task(self, task_name: str, data: Any, timeout: float = 30.0) -> Dict[str, Any]:
        """Execute a task in the cloud."""
        start_time = time.time()

        # Simulate network latency
        network_latency = 0.05  # 50ms average
        time.sleep(network_latency)

        # Simulate task execution
        if "perception" in task_name.lower():
            # Simulate perception task (object detection, etc.)
            execution_time = 0.5  # 500ms for perception
        elif "control" in task_name.lower():
            # Simulate control task
            execution_time = 0.02  # 20ms for control
        else:
            # General task
            execution_time = 0.1  # 100ms default

        time.sleep(execution_time)

        end_time = time.time()

        return {
            "task_id": f"cloud-{task_name}-{int(time.time())}",
            "result": f"Processed {len(str(data))} bytes in cloud",
            "execution_time": end_time - start_time,
            "network_latency": network_latency,
            "cost": self.resource_info.cost_per_hour * (end_time - start_time) / 3600,
            "location": "cloud"
        }

    def get_resource_info(self) -> ComputeResource:
        return self.resource_info

    def is_available(self) -> bool:
        # In a real implementation, this would check actual availability
        return True

class OnPremBackend(ComputingBackend):
    """On-premise computing backend implementation."""

    def __init__(self, resource_id: str, gpu_model: Optional[str] = None):
        self.resource_id = resource_id
        self.gpu_model = gpu_model
        self.resource_info = self._get_resource_spec()

    def _get_resource_spec(self) -> ComputeResource:
        """Get resource specifications for on-premise system."""
        if self.gpu_model and "RTX 4090" in self.gpu_model:
            return ComputeResource(
                resource_id=self.resource_id,
                location="onprem",
                gpu_model=self.gpu_model,
                cpu_cores=16,
                memory_gb=64,
                bandwidth_mbps=1000.0,  # Limited by local network
                cost_per_hour=0.0,  # Fixed cost, no per-hour billing
                availability=0.95  # Dependent on local maintenance
            )
        else:
            return ComputeResource(
                resource_id=self.resource_id,
                location="onprem",
                cpu_cores=8,
                memory_gb=32,
                bandwidth_mbps=1000.0,
                cost_per_hour=0.0,
                availability=0.95
            )

    def execute_task(self, task_name: str, data: Any, timeout: float = 30.0) -> Dict[str, Any]:
        """Execute a task on-premise."""
        start_time = time.time()

        # Simulate task execution (faster due to no network latency)
        if "perception" in task_name.lower():
            # Simulate perception task
            execution_time = 0.3  # 300ms for perception (faster than cloud)
        elif "control" in task_name.lower():
            # Simulate control task (very fast on local system)
            execution_time = 0.005  # 5ms for control
        else:
            # General task
            execution_time = 0.05  # 50ms default

        time.sleep(execution_time)

        end_time = time.time()

        return {
            "task_id": f"onprem-{task_name}-{int(time.time())}",
            "result": f"Processed {len(str(data))} bytes on-premise",
            "execution_time": end_time - start_time,
            "network_latency": 0.0,  # No network latency for on-premise
            "cost": 0.0,  # No per-execution cost for on-premise
            "location": "onpremise"
        }

    def get_resource_info(self) -> ComputeResource:
        return self.resource_info

    def is_available(self) -> bool:
        # In a real implementation, this would check actual availability
        return True

class HybridOrchestrator:
    """Orchestrates tasks between cloud and on-premise backends."""

    def __init__(self, cloud_backend: CloudBackend, onprem_backend: OnPremBackend):
        self.cloud_backend = cloud_backend
        self.onprem_backend = onprem_backend
        self.task_history = []

    def route_task(self, task_name: str, data: Any, task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Route task to appropriate backend based on requirements."""

        # Determine routing based on task characteristics
        if task_requirements.get("real_time", False):
            # Real-time tasks go to on-premise to minimize latency
            backend = self.onprem_backend
        elif task_requirements.get("high_compute", False):
            # High-compute tasks may benefit from cloud resources
            backend = self.cloud_backend
        elif task_requirements.get("sensitive_data", False):
            # Sensitive data processing goes on-premise
            backend = self.onprem_backend
        elif len(str(data)) > 1000000:  # 1MB threshold
            # Large data processing might be better in cloud
            backend = self.cloud_backend
        else:
            # Default to on-premise for general tasks
            backend = self.onprem_backend

        # Execute the task
        result = backend.execute_task(task_name, data)
        result["routed_to"] = backend.get_resource_info().location

        # Store in history
        self.task_history.append(result)

        return result

    def get_cost_analysis(self) -> Dict[str, float]:
        """Calculate cost analysis for executed tasks."""
        cloud_cost = sum(r.get("cost", 0) for r in self.task_history if r.get("location") == "cloud")
        total_tasks = len(self.task_history)

        return {
            "total_tasks": total_tasks,
            "cloud_cost": cloud_cost,
            "avg_cost_per_task": cloud_cost / total_tasks if total_tasks > 0 else 0
        }
```

### Performance Benchmarking Framework

```python
import time
import statistics
from typing import List, Dict, Any, Callable
import matplotlib.pyplot as plt

class PerformanceBenchmark:
    """Framework for benchmarking cloud vs on-premise performance."""

    def __init__(self, cloud_backend: CloudBackend, onprem_backend: OnPremBackend):
        self.cloud_backend = cloud_backend
        self.onprem_backend = onprem_backend
        self.results = {
            "cloud": [],
            "onprem": []
        }

    def benchmark_task(self, task_name: str, data: Any, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark a specific task on both backends."""
        cloud_times = []
        onprem_times = []

        print(f"Benchmarking {task_name} with {iterations} iterations...")

        # Benchmark cloud backend
        for i in range(iterations):
            start_time = time.time()
            result = self.cloud_backend.execute_task(task_name, data)
            end_time = time.time()
            cloud_times.append(end_time - start_time)

        # Benchmark on-premise backend
        for i in range(iterations):
            start_time = time.time()
            result = self.onprem_backend.execute_task(task_name, data)
            end_time = time.time()
            onprem_times.append(end_time - start_time)

        # Calculate statistics
        cloud_stats = {
            "mean": statistics.mean(cloud_times),
            "median": statistics.median(cloud_times),
            "std_dev": statistics.stdev(cloud_times) if len(cloud_times) > 1 else 0,
            "min": min(cloud_times),
            "max": max(cloud_times),
            "percentile_95": sorted(cloud_times)[int(0.95 * len(cloud_times))]
        }

        onprem_stats = {
            "mean": statistics.mean(onprem_times),
            "median": statistics.median(onprem_times),
            "std_dev": statistics.stdev(onprem_times) if len(onprem_times) > 1 else 0,
            "min": min(onprem_times),
            "max": max(onprem_times),
            "percentile_95": sorted(onprem_times)[int(0.95 * len(onprem_times))]
        }

        results = {
            "task_name": task_name,
            "cloud_performance": cloud_stats,
            "onprem_performance": onprem_stats,
            "improvement_factor": cloud_stats["mean"] / onprem_stats["mean"] if onprem_stats["mean"] > 0 else float('inf')
        }

        self.results["cloud"].extend(cloud_times)
        self.results["onprem"].extend(onprem_times)

        return results

    def compare_workloads(self, workloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare performance across multiple workloads."""
        comparison_results = []

        for workload in workloads:
            task_name = workload["name"]
            data = workload["data"]
            iterations = workload.get("iterations", 10)

            result = self.benchmark_task(task_name, data, iterations)
            comparison_results.append(result)

        return comparison_results

    def generate_report(self) -> str:
        """Generate a performance comparison report."""
        if not self.results["cloud"] or not self.results["onprem"]:
            return "No benchmark data available."

        cloud_mean = statistics.mean(self.results["cloud"])
        onprem_mean = statistics.mean(self.results["onprem"])
        improvement_factor = cloud_mean / onprem_mean if onprem_mean > 0 else float('inf')

        report = f"""
Performance Comparison Report
=============================

Overall Performance:
- Cloud Average Time: {cloud_mean:.4f}s
- On-Premise Average Time: {onprem_mean:.4f}s
- Performance Ratio (Cloud/On-Prem): {improvement_factor:.2f}x

Interpretation:
- If ratio > 1: On-premise is faster
- If ratio < 1: Cloud is faster
- Ratio of 2.0 means on-premise is 2x faster than cloud
        """

        return report

    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """Plot performance comparison between cloud and on-premise."""
        if not self.results["cloud"] or not self.results["onprem"]:
            print("No data to plot.")
            return

        plt.figure(figsize=(12, 6))

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot execution time distributions
        ax1.hist(self.results["cloud"], alpha=0.7, label='Cloud', bins=20)
        ax1.hist(self.results["onprem"], alpha=0.7, label='On-Premise', bins=20)
        ax1.set_xlabel('Execution Time (s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Execution Time Distribution')
        ax1.legend()

        # Plot box plots for comparison
        data_to_plot = [self.results["cloud"], self.results["onprem"]]
        ax2.boxplot(data_to_plot, labels=['Cloud', 'On-Premise'])
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Execution Time Comparison (Box Plot)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
```

## Cost Analysis and Modeling

### Cost Calculator

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class CostCalculator:
    """Calculate and compare costs for cloud vs on-premise deployments."""

    def __init__(self):
        self.cloud_pricing = {
            "aws": {
                "g4dn.xlarge": {"hourly": 0.526, "monthly": 384.0},
                "p3.2xlarge": {"hourly": 3.06, "monthly": 2234.0},
                "c5.large": {"hourly": 0.085, "monthly": 62.1},
            },
            "gcp": {
                "n1-standard-4": {"hourly": 0.1912, "monthly": 139.6},
                "n1-standard-8": {"hourly": 0.3824, "monthly": 279.2},
            },
            "azure": {
                "Standard_NC6s_v3": {"hourly": 1.208, "monthly": 882.0},
                "Standard_D4s_v3": {"hourly": 0.203, "monthly": 148.2},
            }
        }

        self.onprem_costs = {
            "workstation": {
                "initial_cost": 5000,
                "annual_maintenance": 500,
                "power_consumption_kw": 0.5,
                "lifespan_years": 5
            },
            "server": {
                "initial_cost": 10000,
                "annual_maintenance": 1000,
                "power_consumption_kw": 1.0,
                "lifespan_years": 5
            }
        }

    def calculate_cloud_cost(self, provider: str, instance_type: str, hours: float) -> float:
        """Calculate cloud cost for specific instance."""
        if provider not in self.cloud_pricing:
            raise ValueError(f"Provider {provider} not supported")

        if instance_type not in self.cloud_pricing[provider]:
            raise ValueError(f"Instance type {instance_type} not found for {provider}")

        hourly_rate = self.cloud_pricing[provider][instance_type]["hourly"]
        return hourly_rate * hours

    def calculate_onprem_cost(self, system_type: str, hours: float, electricity_cost_per_kwh: float = 0.12) -> float:
        """Calculate on-premise cost including depreciation and power."""
        if system_type not in self.onprem_costs:
            raise ValueError(f"System type {system_type} not supported")

        config = self.onprem_costs[system_type]

        # Calculate depreciation cost per hour
        total_cost_over_lifespan = config["initial_cost"] + (config["annual_maintenance"] * config["lifespan_years"])
        hours_in_lifespan = config["lifespan_years"] * 365 * 24
        depreciation_per_hour = total_cost_over_lifespan / hours_in_lifespan

        # Calculate power cost per hour
        power_cost_per_hour = config["power_consumption_kw"] * electricity_cost_per_kwh

        # Total cost
        total_cost = hours * (depreciation_per_hour + power_cost_per_hour)
        return total_cost

    def compare_deployment_costs(
        self,
        cloud_config: Dict[str, Any],
        onprem_config: Dict[str, Any],
        usage_hours: float
    ) -> Dict[str, Any]:
        """Compare costs between cloud and on-premise deployments."""
        cloud_cost = self.calculate_cloud_cost(
            cloud_config["provider"],
            cloud_config["instance_type"],
            usage_hours
        )

        onprem_cost = self.calculate_onprem_cost(
            onprem_config["system_type"],
            usage_hours,
            onprem_config.get("electricity_cost", 0.12)
        )

        return {
            "cloud": {
                "cost": cloud_cost,
                "configuration": cloud_config
            },
            "onprem": {
                "cost": onprem_cost,
                "configuration": onprem_config
            },
            "total_hours": usage_hours,
            "difference": cloud_cost - onprem_cost,
            "is_onprem_cheaper": onprem_cost < cloud_cost
        }

    def find_break_even_point(
        self,
        cloud_config: Dict[str, Any],
        onprem_config: Dict[str, Any],
        max_hours: int = 8760  # One year
    ) -> Optional[float]:
        """Find the break-even point where on-premise becomes cheaper."""
        for hours in range(1, max_hours + 1):
            comparison = self.compare_deployment_costs(cloud_config, onprem_config, hours)
            if comparison["is_onprem_cheaper"]:
                return float(hours)
        return None

    def generate_cost_model(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate cost models for multiple scenarios."""
        results = []
        for scenario in scenarios:
            comparison = self.compare_deployment_costs(
                scenario["cloud_config"],
                scenario["onprem_config"],
                scenario["usage_hours"]
            )
            comparison["scenario_name"] = scenario.get("name", "unnamed")
            results.append(comparison)
        return results
```

## Security and Compliance Management

### Security Policy Manager

```python
from typing import Dict, List, Any, Optional
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurityPolicyManager:
    """Manage security policies for cloud and on-premise environments."""

    def __init__(self):
        self.policies = {
            "cloud": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "access_control": "IAM",
                "logging": True,
                "compliance": ["SOC2", "GDPR", "HIPAA"]
            },
            "onprem": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "access_control": "LDAP/Kerberos",
                "logging": True,
                "compliance": ["ISO 27001", "NIST"]
            }
        }

    def generate_encryption_key(self, password: Optional[str] = None) -> bytes:
        """Generate a secure encryption key."""
        if password:
            # Derive key from password
            salt = secrets.token_bytes(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        else:
            # Generate random key
            key = Fernet.generate_key()

        return key

    def encrypt_data(self, data: str, key: bytes) -> str:
        """Encrypt data using provided key."""
        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data.decode()

    def decrypt_data(self, encrypted_data: str, key: bytes) -> str:
        """Decrypt data using provided key."""
        f = Fernet(key)
        decrypted_data = f.decrypt(encrypted_data.encode())
        return decrypted_data.decode()

    def validate_compliance(self, environment: str, data_type: str) -> Dict[str, Any]:
        """Validate compliance for specific data type in environment."""
        if environment not in self.policies:
            raise ValueError(f"Environment {environment} not supported")

        policy = self.policies[environment]
        compliance_status = {
            "environment": environment,
            "data_type": data_type,
            "encryption_at_rest": policy["encryption_at_rest"],
            "encryption_in_transit": policy["encryption_in_transit"],
            "access_control_compliant": True,  # Simplified
            "logging_enabled": policy["logging"],
            "compliance_standards": policy["compliance"],
            "is_compliant": True  # Simplified validation
        }

        # Specific checks based on data type
        if data_type.lower() in ["personal", "medical", "financial"]:
            # More stringent requirements for sensitive data
            compliance_status["requires_additional_protection"] = True
            compliance_status["recommended_environment"] = "onprem" if environment == "cloud" else "onprem"

        return compliance_status

    def assess_security_risk(self, environment: str, data_sensitivity: str = "standard") -> Dict[str, float]:
        """Assess security risk for environment and data sensitivity."""
        base_risks = {
            "cloud": {
                "data_breach_risk": 0.05,  # 5% annual probability
                "compliance_violation": 0.02,  # 2% annual probability
                "service_disruption": 0.01  # 1% annual probability
            },
            "onprem": {
                "data_breach_risk": 0.08,  # 8% annual probability
                "compliance_violation": 0.03,  # 3% annual probability
                "service_disruption": 0.05  # 5% annual probability
            }
        }

        sensitivity_multipliers = {
            "low": 0.5,
            "standard": 1.0,
            "high": 2.0,
            "critical": 5.0
        }

        if environment not in base_risks:
            raise ValueError(f"Environment {environment} not supported")

        multiplier = sensitivity_multipliers.get(data_sensitivity.lower(), 1.0)
        base_risk = base_risks[environment]

        risk_assessment = {
            "data_breach_risk": base_risk["data_breach_risk"] * multiplier,
            "compliance_violation": base_risk["compliance_violation"] * multiplier,
            "service_disruption": base_risk["service_disruption"] * multiplier,
            "total_risk_score": sum(base_risk.values()) * multiplier
        }

        return risk_assessment
```

## Deployment Automation

### Multi-Environment Deployment Manager

```python
import subprocess
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import yaml

@dataclass
class DeploymentConfig:
    environment: str  # "cloud" or "onprem"
    provider: Optional[str] = None  # "aws", "gcp", "azure" for cloud
    region: Optional[str] = None
    instance_type: Optional[str] = None
    docker_image: str = ""
    environment_vars: Dict[str, str] = None
    ports: List[int] = None

class DeploymentManager:
    """Manage deployments across cloud and on-premise environments."""

    def __init__(self):
        self.active_deployments = {}

    def deploy_to_cloud(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to cloud environment."""
        if config.provider == "aws":
            return self._deploy_aws(config)
        elif config.provider == "gcp":
            return self._deploy_gcp(config)
        elif config.provider == "azure":
            return self._deploy_azure(config)
        else:
            raise ValueError(f"Unsupported cloud provider: {config.provider}")

    def deploy_to_onprem(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to on-premise environment."""
        # For on-premise, we'll use Docker
        try:
            # Pull the Docker image
            subprocess.run(["docker", "pull", config.docker_image], check=True)

            # Run the container
            cmd = ["docker", "run", "-d"]

            # Add environment variables
            for key, value in (config.environment_vars or {}).items():
                cmd.extend(["-e", f"{key}={value}"])

            # Add port mappings
            for port in (config.ports or []):
                cmd.extend(["-p", f"{port}:{port}"])

            # Add image name
            cmd.append(config.docker_image)

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            container_id = result.stdout.strip()

            return {
                "deployment_id": container_id,
                "status": "success",
                "environment": "onpremise",
                "provider": "docker",
                "details": f"Container {container_id} started successfully"
            }
        except subprocess.CalledProcessError as e:
            return {
                "deployment_id": None,
                "status": "failed",
                "error": str(e),
                "details": e.stderr
            }

    def _deploy_aws(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to AWS."""
        try:
            # This would use boto3 in a real implementation
            # For demonstration, we'll simulate the deployment

            # Create deployment command (simplified)
            cmd = [
                "aws", "ec2", "run-instances",
                "--image-id", "ami-12345678",  # Example AMI
                "--instance-type", config.instance_type or "t3.micro",
                "--key-name", "robotics-key",
                "--security-group-ids", "sg-12345678"
            ]

            # In a real implementation, you would:
            # 1. Create EC2 instance
            # 2. Install Docker
            # 3. Run your container
            # 4. Configure networking

            # Simulate successful deployment
            return {
                "deployment_id": f"i-{secrets.token_hex(8)}",
                "status": "success",
                "environment": "cloud",
                "provider": "aws",
                "region": config.region or "us-east-1",
                "instance_type": config.instance_type or "t3.micro",
                "details": "Instance created and configured"
            }
        except Exception as e:
            return {
                "deployment_id": None,
                "status": "failed",
                "error": str(e),
                "details": "Deployment failed"
            }

    def _deploy_gcp(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to GCP."""
        try:
            # Simulate GCP deployment
            return {
                "deployment_id": f"gcp-{secrets.token_hex(8)}",
                "status": "success",
                "environment": "cloud",
                "provider": "gcp",
                "region": config.region or "us-central1",
                "instance_type": config.instance_type or "e2-medium",
                "details": "GCP instance created and configured"
            }
        except Exception as e:
            return {
                "deployment_id": None,
                "status": "failed",
                "error": str(e),
                "details": "GCP deployment failed"
            }

    def _deploy_azure(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to Azure."""
        try:
            # Simulate Azure deployment
            return {
                "deployment_id": f"azure-{secrets.token_hex(8)}",
                "status": "success",
                "environment": "cloud",
                "provider": "azure",
                "region": config.region or "eastus",
                "instance_type": config.instance_type or "Standard_B2s",
                "details": "Azure VM created and configured"
            }
        except Exception as e:
            return {
                "deployment_id": None,
                "status": "failed",
                "error": str(e),
                "details": "Azure deployment failed"
            }

    def deploy_application(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy application based on configuration."""
        if config.environment == "cloud":
            if not config.provider:
                raise ValueError("Cloud provider must be specified")
            result = self.deploy_to_cloud(config)
        elif config.environment == "onprem":
            result = self.deploy_to_onprem(config)
        else:
            raise ValueError(f"Unsupported environment: {config.environment}")

        # Store deployment info
        if result.get("deployment_id"):
            self.active_deployments[result["deployment_id"]] = {
                "config": config,
                "result": result,
                "timestamp": time.time()
            }

        return result

    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a specific deployment."""
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]["result"]
        else:
            return {"status": "not_found", "error": f"Deployment {deployment_id} not found"}

    def teardown_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Teardown a specific deployment."""
        if deployment_id not in self.active_deployments:
            return {"status": "not_found", "error": f"Deployment {deployment_id} not found"}

        deployment_info = self.active_deployments[deployment_id]
        config = deployment_info["config"]

        try:
            if config.environment == "onprem":
                # Stop and remove Docker container
                subprocess.run(["docker", "stop", deployment_id], check=True)
                subprocess.run(["docker", "rm", deployment_id], check=True)
            else:
                # For cloud environments, terminate the instance
                if config.provider == "aws":
                    subprocess.run(["aws", "ec2", "terminate-instances", "--instance-ids", deployment_id], check=True)

            # Remove from active deployments
            del self.active_deployments[deployment_id]

            return {
                "status": "success",
                "message": f"Deployment {deployment_id} terminated successfully"
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": f"Failed to terminate deployment {deployment_id}"
            }
```

## Monitoring and Observability

### Multi-Environment Monitoring System

```python
import psutil
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import threading

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: float

class MultiEnvironmentMonitor:
    """Monitor systems across cloud and on-premise environments."""

    def __init__(self):
        self.metrics_history = {
            "cloud": [],
            "onprem": []
        }
        self.alerts = []
        self.monitoring_thread = None
        self.running = False

    def collect_local_metrics(self) -> SystemMetrics:
        """Collect metrics from the local system."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        network_info = psutil.net_io_counters()

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_info.percent,
            disk_usage_percent=disk_usage.percent,
            network_bytes_sent=network_info.bytes_sent,
            network_bytes_recv=network_info.bytes_recv,
            timestamp=time.time()
        )

    def collect_cloud_metrics(self, cloud_provider: str, resource_id: str) -> Optional[Dict[str, Any]]:
        """Collect metrics from cloud provider."""
        # In a real implementation, this would call cloud provider APIs
        # For example, AWS CloudWatch, GCP Operations, Azure Monitor

        # Simulate cloud metrics collection
        if cloud_provider == "aws":
            # This would use boto3 to get CloudWatch metrics
            return {
                "cpu_utilization": 45.0,
                "memory_utilization": 60.0,
                "network_in": 1024000,
                "network_out": 512000,
                "disk_read_ops": 100,
                "disk_write_ops": 50
            }
        elif cloud_provider == "gcp":
            # This would use google-cloud-monitoring
            return {
                "cpu_utilization": 35.0,
                "memory_utilization": 55.0,
                "network_bytes": 768000,
                "disk_io_bytes": 256000
            }
        else:
            return None

    def check_alert_conditions(self, metrics: SystemMetrics) -> List[str]:
        """Check if any alert conditions are met."""
        alerts = []

        if metrics.cpu_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_percent}%")

        if metrics.memory_percent > 90:
            alerts.append(f"High memory usage: {metrics.memory_percent}%")

        if metrics.disk_usage_percent > 95:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent}%")

        return alerts

    def start_monitoring(self, interval: float = 30.0):
        """Start continuous monitoring."""
        self.running = True
        self.monitoring_interval = interval

        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect local metrics
                local_metrics = self.collect_local_metrics()
                self.metrics_history["onprem"].append(local_metrics)

                # Check for alerts
                alerts = self.check_alert_conditions(local_metrics)
                for alert in alerts:
                    self.alerts.append({
                        "timestamp": time.time(),
                        "source": "onprem",
                        "message": alert
                    })
                    print(f"ALERT: {alert}")

                # In a real implementation, you would also collect cloud metrics
                # cloud_metrics = self.collect_cloud_metrics("aws", "i-123456789")
                # if cloud_metrics:
                #     self.metrics_history["cloud"].append(cloud_metrics)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between environments."""
        onprem_data = self.metrics_history["onprem"]

        if not onprem_data:
            return {"error": "No monitoring data available"}

        # Calculate averages for on-premise
        avg_cpu = sum(m.cpu_percent for m in onprem_data) / len(onprem_data)
        avg_memory = sum(m.memory_percent for m in onprem_data) / len(onprem_data)

        return {
            "onprem": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "sample_count": len(onprem_data),
                "time_range": {
                    "start": min(m.timestamp for m in onprem_data),
                    "end": max(m.timestamp for m in onprem_data)
                }
            },
            "cloud": {
                # Would be populated when cloud metrics are collected
                "avg_cpu_percent": None,
                "avg_memory_percent": None,
                "sample_count": 0
            }
        }

    def generate_monitoring_report(self) -> str:
        """Generate a monitoring report."""
        comparison = self.get_performance_comparison()

        if "error" in comparison:
            return "No monitoring data available."

        report = f"""
Monitoring Report
===============

On-Premise System:
- Average CPU Usage: {comparison['onprem']['avg_cpu_percent']:.2f}%
- Average Memory Usage: {comparison['onprem']['avg_memory_percent']:.2f}%
- Samples Collected: {comparison['onprem']['sample_count']}
- Monitoring Period: {time.ctime(comparison['onprem']['time_range']['start'])} to {time.ctime(comparison['onprem']['time_range']['end'])}

Alerts Generated: {len(self.alerts)}
        """

        return report
```

## Complete Example: Hybrid Robotics Application

```python
def main():
    """Main example demonstrating cloud-on-premise hybrid architecture."""

    print("Setting up Hybrid Cloud-On-Premise Architecture for Robotics...")

    # Initialize cloud and on-premise backends
    cloud_backend = CloudBackend(provider="aws", instance_type="g4dn.xlarge")
    onprem_backend = OnPremBackend(resource_id="robotics-workstation-01", gpu_model="RTX 4090")

    # Create hybrid orchestrator
    orchestrator = HybridOrchestrator(cloud_backend, onprem_backend)

    # Define tasks with different requirements
    tasks = [
        {
            "name": "real_time_control",
            "data": "control_commands",
            "requirements": {"real_time": True, "sensitive_data": False}
        },
        {
            "name": "data_processing",
            "data": "large_dataset" * 10000,  # Simulate large data
            "requirements": {"high_compute": True, "sensitive_data": False}
        },
        {
            "name": "perception_processing",
            "data": "sensor_data",
            "requirements": {"real_time": True, "sensitive_data": True}
        }
    ]

    print("\nExecuting tasks with intelligent routing...")
    for task in tasks:
        result = orchestrator.route_task(
            task["name"],
            task["data"],
            task["requirements"]
        )
        print(f"Task '{task['name']}' executed on {result['routed_to']}: {result['execution_time']:.4f}s")

    # Perform performance benchmarking
    print("\nPerforming performance benchmarking...")
    benchmark = PerformanceBenchmark(cloud_backend, onprem_backend)

    workloads = [
        {"name": "perception_task", "data": "camera_feed", "iterations": 5},
        {"name": "control_task", "data": "sensor_data", "iterations": 5},
        {"name": "planning_task", "data": "environment_map", "iterations": 5}
    ]

    benchmark_results = benchmark.compare_workloads(workloads)

    for result in benchmark_results:
        print(f"Task: {result['task_name']}")
        print(f"  Cloud mean time: {result['cloud_performance']['mean']:.4f}s")
        print(f"  On-premise mean time: {result['onprem_performance']['mean']:.4f}s")
        print(f"  Improvement factor: {result['improvement_factor']:.2f}x")
        print()

    # Generate performance report
    print(benchmark.generate_report())

    # Cost analysis
    print("\nPerforming cost analysis...")
    cost_calculator = CostCalculator()

    cloud_config = {
        "provider": "aws",
        "instance_type": "g4dn.xlarge"
    }

    onprem_config = {
        "system_type": "workstation",
        "electricity_cost": 0.15
    }

    # Compare costs for 24/7 operation for one month
    usage_hours = 30 * 24  # 30 days
    cost_comparison = cost_calculator.compare_deployment_costs(
        cloud_config, onprem_config, usage_hours
    )

    print(f"Monthly cost comparison for {usage_hours} hours:")
    print(f"  Cloud: ${cost_comparison['cloud']['cost']:.2f}")
    print(f"  On-premise: ${cost_comparison['onprem']['cost']:.2f}")
    print(f"  Difference: ${cost_comparison['difference']:.2f}")
    print(f"  On-premise cheaper: {cost_comparison['is_onprem_cheaper']}")

    # Security assessment
    print("\nPerforming security assessment...")
    security_manager = SecurityPolicyManager()

    cloud_compliance = security_manager.validate_compliance("cloud", "standard")
    onprem_compliance = security_manager.validate_compliance("onprem", "standard")

    print(f"Cloud compliance: {cloud_compliance['is_compliant']}")
    print(f"On-premise compliance: {onprem_compliance['is_compliant']}")

    cloud_risk = security_manager.assess_security_risk("cloud", "standard")
    onprem_risk = security_manager.assess_security_risk("onprem", "standard")

    print(f"Cloud total risk score: {cloud_risk['total_risk_score']:.4f}")
    print(f"On-premise total risk score: {onprem_risk['total_risk_score']:.4f}")

    # Deployment example
    print("\nPerforming deployment...")
    deployer = DeploymentManager()

    onprem_config = DeploymentConfig(
        environment="onprem",
        docker_image="robotics-app:latest",
        environment_vars={"ENV": "onprem", "DEBUG": "false"},
        ports=[8080, 9090]
    )

    deployment_result = deployer.deploy_application(onprem_config)
    print(f"Deployment result: {deployment_result['status']}")

    # Start monitoring
    print("\nStarting monitoring...")
    monitor = MultiEnvironmentMonitor()
    monitor.start_monitoring(interval=10)  # Monitor every 10 seconds

    # Let it run for a bit to collect metrics
    time.sleep(30)

    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    print(report)

    # Stop monitoring
    monitor.stop_monitoring()

    # Print final cost analysis from orchestrator
    cost_analysis = orchestrator.get_cost_analysis()
    print(f"\nTask execution cost analysis:")
    print(f"  Total tasks executed: {cost_analysis['total_tasks']}")
    print(f"  Cloud execution cost: ${cost_analysis['cloud_cost']:.4f}")
    print(f"  Average cost per task: ${cost_analysis['avg_cost_per_task']:.4f}")

if __name__ == "__main__":
    main()
```

## Summary

This chapter provided comprehensive code examples for comparing and implementing cloud vs. on-premise workflows in Physical AI and Humanoid Robotics applications. The examples include:

1. **Hybrid Computing Abstraction**: A flexible architecture that can route tasks to appropriate computing backends based on requirements
2. **Performance Benchmarking**: Framework for comparing performance characteristics of different deployment options
3. **Cost Analysis**: Tools for calculating and comparing costs of cloud vs. on-premise deployments
4. **Security Management**: Security policy and compliance validation for different environments
5. **Deployment Automation**: Multi-environment deployment manager for consistent deployments
6. **Monitoring System**: Cross-environment monitoring for performance and security

These examples demonstrate practical approaches to implementing hybrid architectures that leverage the strengths of both cloud and on-premise computing for robotics applications. The modular design allows for easy extension and customization based on specific project requirements and constraints.