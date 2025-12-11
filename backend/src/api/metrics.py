import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from enum import Enum


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Represents a single metric measurement."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Holds performance metrics for a specific operation."""
    name: str
    metric_type: MetricType
    description: str
    points: List[MetricPoint] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and manages performance metrics for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self, retention_minutes: int = 60):
        """
        Initialize the metrics collector.

        Args:
            retention_minutes: Number of minutes to retain metrics (default 60)
        """
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.Lock()
        self.retention_time = timedelta(minutes=retention_minutes)
        self.start_time = time.time()

    def _cleanup_old_metrics(self):
        """Remove metrics that are older than the retention time."""
        cutoff_time = time.time() - self.retention_time.total_seconds()

        with self.lock:
            for metric_name, metric in self.metrics.items():
                # Filter out old points
                metric.points = [p for p in metric.points if p.timestamp >= cutoff_time]

    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     labels: Optional[Dict[str, str]] = None, description: str = ""):
        """
        Record a metric value.

        Args:
            name: Name of the metric
            value: Value to record
            metric_type: Type of metric
            labels: Optional labels to attach to the metric
            description: Optional description of the metric
        """
        if labels is None:
            labels = {}

        with self.lock:
            # Create or get the metric
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetrics(
                    name=name,
                    metric_type=metric_type,
                    description=description or f"Performance metric for {name}",
                    labels=labels
                )

            metric = self.metrics[name]

            # Add the new point
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels
            )
            metric.points.append(point)

        # Clean up old metrics periodically
        if len(self.metrics.get(name, []).points) % 100 == 0:  # Every 100 points
            self._cleanup_old_metrics()

    def record_timing(self, name: str, labels: Optional[Dict[str, str]] = None,
                     description: str = "Operation execution time in seconds"):
        """
        Context manager to record timing for an operation.

        Args:
            name: Name of the metric (e.g., "query_processing_time")
            labels: Optional labels to attach to the metric
            description: Optional description of the metric

        Yields:
            None
        """
        class Timer:
            def __enter__(timer_self):
                self.start_time = time.time()
                return timer_self

            def __exit__(timer_self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.record_metric(
                    name=name,
                    value=duration,
                    metric_type=MetricType.HISTOGRAM,
                    labels=labels,
                    description=description
                )

        return Timer()

    def increment_counter(self, name: str, value: float = 1.0,
                         labels: Optional[Dict[str, str]] = None,
                         description: str = "Counter metric"):
        """
        Increment a counter metric.

        Args:
            name: Name of the metric
            value: Value to increment by (default 1.0)
            labels: Optional labels to attach to the metric
            description: Optional description of the metric
        """
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetrics(
                    name=name,
                    metric_type=MetricType.COUNTER,
                    description=description,
                    labels=labels or {}
                )

            # For counters, we keep a running total
            current_value = self.metrics[name].points[-1].value if self.metrics[name].points else 0
            new_value = current_value + value

            point = MetricPoint(
                timestamp=time.time(),
                value=new_value,
                labels=labels or {}
            )
            self.metrics[name].points.append(point)

    def set_gauge(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None,
                  description: str = "Gauge metric"):
        """
        Set a gauge metric to a specific value.

        Args:
            name: Name of the metric
            value: Value to set
            labels: Optional labels to attach to the metric
            description: Optional description of the metric
        """
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetrics(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    description=description,
                    labels=labels or {}
                )

            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self.metrics[name].points.append(point)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary with metrics summary
        """
        self._cleanup_old_metrics()

        summary = {
            "start_time": self.start_time,
            "uptime_seconds": time.time() - self.start_time,
            "metrics": {}
        }

        with self.lock:
            for name, metric in self.metrics.items():
                if metric.points:
                    # Calculate basic statistics
                    values = [p.value for p in metric.points]
                    count = len(values)
                    total = sum(values)
                    avg = total / count if count > 0 else 0
                    min_val = min(values) if values else 0
                    max_val = max(values) if values else 0

                    # Calculate percentiles (simplified)
                    sorted_values = sorted(values)
                    p50_idx = int(0.5 * len(sorted_values))
                    p95_idx = int(0.95 * len(sorted_values))
                    p99_idx = int(0.99 * len(sorted_values))

                    p50 = sorted_values[p50_idx] if sorted_values else 0
                    p95 = sorted_values[min(p95_idx, len(sorted_values) - 1)] if sorted_values else 0
                    p99 = sorted_values[min(p99_idx, len(sorted_values) - 1)] if sorted_values else 0

                    summary["metrics"][name] = {
                        "type": metric.metric_type.value,
                        "description": metric.description,
                        "count": count,
                        "total": total,
                        "average": avg,
                        "min": min_val,
                        "max": max_val,
                        "p50": p50,
                        "p95": p95,
                        "p99": p99,
                        "latest_value": values[-1] if values else 0
                    }

        return summary

    def get_metric_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific metric by name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            Metric data or None if not found
        """
        self._cleanup_old_metrics()

        with self.lock:
            if name not in self.metrics:
                return None

            metric = self.metrics[name]
            if not metric.points:
                return None

            values = [p.value for p in metric.points]
            count = len(values)
            total = sum(values)
            avg = total / count if count > 0 else 0
            min_val = min(values) if values else 0
            max_val = max(values) if values else 0

            # Calculate percentiles
            sorted_values = sorted(values)
            p50_idx = int(0.5 * len(sorted_values))
            p95_idx = int(0.95 * len(sorted_values))
            p99_idx = int(0.99 * len(sorted_values))

            p50 = sorted_values[p50_idx] if sorted_values else 0
            p95 = sorted_values[min(p95_idx, len(sorted_values) - 1)] if sorted_values else 0
            p99 = sorted_values[min(p99_idx, len(sorted_values) - 1)] if sorted_values else 0

            return {
                "name": metric.name,
                "type": metric.metric_type.value,
                "description": metric.description,
                "labels": metric.labels,
                "count": count,
                "total": total,
                "average": avg,
                "min": min_val,
                "max": max_val,
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "latest_value": values[-1] if values else 0,
                "points_count": len(values)
            }

    def reset_metrics(self):
        """Reset all collected metrics."""
        with self.lock:
            self.metrics = {}
            self.start_time = time.time()

    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus text format
        """
        self._cleanup_old_metrics()
        output_lines = []

        with self.lock:
            for name, metric in self.metrics.items():
                if not metric.points:
                    continue

                # Add metric help text
                output_lines.append(f"# HELP {name} {metric.description}")
                output_lines.append(f"# TYPE {name} {metric.metric_type.value}")

                # Add metric values
                for point in metric.points:
                    labels_str = ""
                    if point.labels:
                        labels_str = "{" + ",".join([f'{k}="{v}"' for k, v in point.labels.items()]) + "}"

                    output_lines.append(f"{name}{labels_str} {point.value} {int(point.timestamp * 1000)}")

        return "\n".join(output_lines)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    return metrics_collector


def record_timing(name: str, labels: Optional[Dict[str, str]] = None,
                 description: str = "Operation execution time in seconds"):
    """
    Context manager to record timing for an operation.

    Args:
        name: Name of the metric (e.g., "query_processing_time")
        labels: Optional labels to attach to the metric
        description: Optional description of the metric

    Returns:
        Context manager for timing
    """
    return metrics_collector.record_timing(name, labels, description)


def increment_counter(name: str, value: float = 1.0,
                     labels: Optional[Dict[str, str]] = None,
                     description: str = "Counter metric"):
    """
    Increment a counter metric.

    Args:
        name: Name of the metric
        value: Value to increment by (default 1.0)
        labels: Optional labels to attach to the metric
        description: Optional description of the metric
    """
    metrics_collector.increment_counter(name, value, labels, description)


def set_gauge(name: str, value: float,
              labels: Optional[Dict[str, str]] = None,
              description: str = "Gauge metric"):
    """
    Set a gauge metric to a specific value.

    Args:
        name: Name of the metric
        value: Value to set
        labels: Optional labels to attach to the metric
        description: Optional description of the metric
    """
    metrics_collector.set_gauge(name, value, labels, description)


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of all collected metrics.

    Returns:
        Dictionary with metrics summary
    """
    return metrics_collector.get_metrics_summary()


def get_metric_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific metric by name.

    Args:
        name: Name of the metric to retrieve

    Returns:
        Metric data or None if not found
    """
    return metrics_collector.get_metric_by_name(name)


# Predefined metrics for common operations
class PredefinedMetrics:
    """Predefined metric names for common operations."""

    # API metrics
    API_RESPONSE_TIME = "api_response_time_seconds"
    API_REQUEST_COUNT = "api_requests_total"
    API_ERROR_COUNT = "api_errors_total"

    # Agent metrics
    AGENT_EXECUTION_TIME = "agent_execution_time_seconds"
    AGENT_EXECUTION_COUNT = "agent_executions_total"
    AGENT_ERROR_COUNT = "agent_errors_total"

    # Retrieval metrics
    RETRIEVAL_TIME = "retrieval_time_seconds"
    RETRIEVAL_COUNT = "retrievals_total"
    RETRIEVAL_ERROR_COUNT = "retrieval_errors_total"
    RETRIEVAL_TOP_K_ACCURACY = "retrieval_top_k_accuracy"

    # Database metrics
    DB_QUERY_TIME = "db_query_time_seconds"
    DB_CONNECTION_POOL_USAGE = "db_connection_pool_usage_ratio"

    # System metrics
    ACTIVE_SESSIONS = "active_sessions"
    REQUEST_QUEUE_SIZE = "request_queue_size"


# Example usage
if __name__ == "__main__":
    print("Performance metrics example:")

    # Record some sample metrics
    with record_timing(PredefinedMetrics.API_RESPONSE_TIME, {"endpoint": "/query"}):
        time.sleep(0.1)  # Simulate work
        print("Simulated API call completed")

    increment_counter(PredefinedMetrics.API_REQUEST_COUNT, 1, {"endpoint": "/query", "method": "POST"})
    set_gauge(PredefinedMetrics.ACTIVE_SESSIONS, 5)

    # Record agent execution time
    with record_timing(PredefinedMetrics.AGENT_EXECUTION_TIME, {"agent": "RAGAgent", "operation": "process_query"}):
        time.sleep(0.05)  # Simulate agent work
        print("Simulated agent execution completed")

    increment_counter(PredefinedMetrics.AGENT_EXECUTION_COUNT, 1, {"agent": "RAGAgent"})

    # Get metrics summary
    summary = get_metrics_summary()
    print(f"\nMetrics summary: {json.dumps(summary, indent=2)}")

    # Get specific metric
    response_time_metric = get_metric_by_name(PredefinedMetrics.API_RESPONSE_TIME)
    print(f"\nAPI response time metric: {json.dumps(response_time_metric, indent=2)}")

    # Export in Prometheus format
    prometheus_output = metrics_collector.export_prometheus_format()
    print(f"\nPrometheus format:\n{prometheus_output}")