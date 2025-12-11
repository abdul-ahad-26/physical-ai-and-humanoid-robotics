"""
Security Audit Logger for the RAG + Agentic Backend for AI-Textbook Chatbot.

This module provides comprehensive security audit logging capabilities including:
- Authentication attempts and results
- Authorization checks and failures
- Suspicious activity detection
- Security event logging
- Compliance reporting
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import hashlib
import ipaddress
from dataclasses import dataclass
from contextvars import ContextVar
import re
import threading
from collections import defaultdict, deque

# Import the structured logger we already have
from .logging import get_logger, logger as structured_logger


class SecurityEventType(Enum):
    """Types of security events that can be audited"""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_SUCCESS = "authz_success"
    AUTHORIZATION_FAILURE = "authz_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    ADMIN_ACTION = "admin_action"
    SECURITY_BREACH = "security_breach"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    SESSION_CREATED = "session_created"
    SESSION_DESTROYED = "session_destroyed"
    API_KEY_USED = "api_key_used"
    API_KEY_REVOKED = "api_key_revoked"


@dataclass
class SecurityEvent:
    """Represents a security audit event"""
    event_type: SecurityEventType
    user_id: Optional[str]
    ip_address: Optional[str]
    endpoint: str
    method: str
    timestamp: float
    details: Dict[str, Any]
    severity: str  # info, warning, critical
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


class SecurityAuditLogger:
    """Security audit logger with comprehensive logging capabilities"""

    def __init__(self, logger=None):
        self.logger = logger or structured_logger
        self.audit_events = deque(maxlen=10000)  # Keep last 10k events in memory
        self._lock = threading.Lock()

        # Track suspicious patterns
        self.failed_auth_attempts = defaultdict(list)
        self.suspicious_ips = set()
        self.abuse_patterns = defaultdict(list)

        # Configuration for anomaly detection
        self.max_failed_attempts = 5
        self.attempt_window = 300  # 5 minutes
        self.max_requests_per_minute = 100

    def log_security_event(self, event: SecurityEvent):
        """Log a security event with structured format"""
        with self._lock:
            # Add to in-memory event store
            self.audit_events.append(event)

            # Log using structured logger
            extra_data = {
                "event_type": event.event_type.value,
                "user_id": event.user_id,
                "ip_address": event.ip_address,
                "endpoint": event.endpoint,
                "method": event.method,
                "details": event.details,
                "severity": event.severity,
                "session_id": event.session_id,
                "correlation_id": event.correlation_id
            }

            if event.severity == "critical":
                self.logger.critical(f"Security event: {event.event_type.value}", extra_data)
            elif event.severity == "warning":
                self.logger.warning(f"Security event: {event.event_type.value}", extra_data)
            else:
                self.logger.info(f"Security event: {event.event_type.value}", extra_data)

    def log_authentication_attempt(self, user_id: str, ip_address: str, success: bool,
                                  endpoint: str = "", method: str = "", details: Dict[str, Any] = None):
        """Log an authentication attempt"""
        event_type = SecurityEventType.AUTHENTICATION_SUCCESS if success else SecurityEventType.AUTHENTICATION_FAILURE
        severity = "info" if success else "warning"

        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=time.time(),
            details=details or {},
            severity=severity
        )

        self.log_security_event(event)

        # Track failed attempts for anomaly detection
        if not success:
            self._track_failed_auth_attempt(ip_address, user_id)

    def log_authorization_check(self, user_id: str, ip_address: str, endpoint: str,
                               method: str, resource: str, allowed: bool,
                               details: Dict[str, Any] = None):
        """Log an authorization check"""
        event_type = SecurityEventType.AUTHORIZATION_SUCCESS if allowed else SecurityEventType.AUTHORIZATION_FAILURE
        severity = "info" if allowed else "warning"

        event_details = details or {}
        event_details["resource"] = resource

        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=time.time(),
            details=event_details,
            severity=severity
        )

        self.log_security_event(event)

    def log_rate_limit_violation(self, user_id: str, ip_address: str, endpoint: str,
                                method: str, limit_type: str, details: Dict[str, Any] = None):
        """Log a rate limit violation"""
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=time.time(),
            details=details or {"limit_type": limit_type},
            severity="warning"
        )

        self.log_security_event(event)

        # Track potential abuse
        self._track_abuse_pattern(ip_address, user_id)

    def log_suspicious_activity(self, user_id: str, ip_address: str, endpoint: str,
                               method: str, activity_type: str, details: Dict[str, Any] = None):
        """Log suspicious activity"""
        event = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=time.time(),
            details=details or {"activity_type": activity_type},
            severity="critical"
        )

        self.log_security_event(event)

    def log_input_validation_failure(self, user_id: str, ip_address: str, endpoint: str,
                                    method: str, field: str, value: str, rule: str):
        """Log input validation failure"""
        # Don't log the actual value to avoid logging sensitive data
        sanitized_value = self._sanitize_log_value(value)

        event = SecurityEvent(
            event_type=SecurityEventType.INPUT_VALIDATION_FAILURE,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=time.time(),
            details={
                "field": field,
                "sanitized_value": sanitized_value,
                "validation_rule": rule
            },
            severity="warning"
        )

        self.log_security_event(event)

    def log_data_access(self, user_id: str, ip_address: str, endpoint: str,
                       method: str, resource_id: str, details: Dict[str, Any] = None):
        """Log data access"""
        event = SecurityEvent(
            event_type=SecurityEventType.DATA_ACCESS,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=time.time(),
            details=details or {"resource_id": resource_id},
            severity="info"
        )

        self.log_security_event(event)

    def log_admin_action(self, admin_user_id: str, ip_address: str, action: str,
                        target_resource: str, details: Dict[str, Any] = None):
        """Log administrative action"""
        event = SecurityEvent(
            event_type=SecurityEventType.ADMIN_ACTION,
            user_id=admin_user_id,
            ip_address=ip_address,
            endpoint="admin",
            method="ADMIN",
            timestamp=time.time(),
            details=details or {"action": action, "target_resource": target_resource},
            severity="info"
        )

        self.log_security_event(event)

    def detect_anomalies(self) -> List[SecurityEvent]:
        """Detect security anomalies based on tracked patterns"""
        anomalies = []

        # Check for IPs with too many failed auth attempts
        current_time = time.time()
        for ip, attempts in self.failed_auth_attempts.items():
            recent_attempts = [t for t in attempts if current_time - t < self.attempt_window]
            if len(recent_attempts) > self.max_failed_attempts:
                event = SecurityEvent(
                    event_type=SecurityEventType.SECURITY_BREACH,
                    user_id=None,
                    ip_address=ip,
                    endpoint="security",
                    method="ANOMALY_DETECTION",
                    timestamp=current_time,
                    details={
                        "anomaly_type": "brute_force_attack",
                        "failed_attempts_count": len(recent_attempts),
                        "window_seconds": self.attempt_window
                    },
                    severity="critical"
                )
                anomalies.append(event)

        # Check for IPs with potential abuse patterns
        for ip, requests in self.abuse_patterns.items():
            recent_requests = [t for t in requests if current_time - t < 60]  # Last minute
            if len(recent_requests) > self.max_requests_per_minute:
                event = SecurityEvent(
                    event_type=SecurityEventType.SECURITY_BREACH,
                    user_id=None,
                    ip_address=ip,
                    endpoint="security",
                    method="ANOMALY_DETECTION",
                    timestamp=current_time,
                    details={
                        "anomaly_type": "rate_limit_abuse",
                        "requests_count": len(recent_requests),
                        "window_seconds": 60
                    },
                    severity="warning"
                )
                anomalies.append(event)

        # Log detected anomalies
        for anomaly in anomalies:
            self.log_security_event(anomaly)

        return anomalies

    def _track_failed_auth_attempt(self, ip_address: str, user_id: str):
        """Track failed authentication attempts for anomaly detection"""
        current_time = time.time()
        self.failed_auth_attempts[ip_address].append(current_time)

        # Clean up old attempts
        cutoff = current_time - self.attempt_window
        self.failed_auth_attempts[ip_address] = [
            t for t in self.failed_auth_attempts[ip_address] if t > cutoff
        ]

    def _track_abuse_pattern(self, ip_address: str, user_id: str):
        """Track potential abuse patterns"""
        current_time = time.time()
        self.abuse_patterns[ip_address].append(current_time)

        # Clean up old requests
        cutoff = current_time - 60  # Last minute
        self.abuse_patterns[ip_address] = [
            t for t in self.abuse_patterns[ip_address] if t > cutoff
        ]

    def _sanitize_log_value(self, value: str) -> str:
        """Sanitize potentially sensitive values for logging"""
        if not value:
            return value

        # Hash sensitive values
        if self._is_sensitive_field(value):
            return f"[HASHED:{hashlib.sha256(value.encode()).hexdigest()[:16]}]"

        # Sanitize common sensitive patterns
        value = re.sub(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b', '[CREDIT_CARD]', value)  # Credit cards
        value = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', value)  # Emails
        value = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', value)  # SSNs

        return value

    def _is_sensitive_field(self, value: str) -> bool:
        """Check if a value looks like it might be sensitive"""
        # This is a simple check - in a real system you'd have more sophisticated detection
        sensitive_patterns = [
            r'password',
            r'token',
            r'key',
            r'secret',
            r'auth',
            r'credential',
            r'api.*key',
            r'access.*token'
        ]

        value_lower = value.lower()
        return any(re.search(pattern, value_lower) for pattern in sensitive_patterns)

    def get_audit_trail(self, start_time: float = None, end_time: float = None,
                       event_types: List[SecurityEventType] = None,
                       ip_address: str = None, user_id: str = None,
                       limit: int = 100) -> List[SecurityEvent]:
        """Retrieve audit trail based on filters"""
        with self._lock:
            filtered_events = []

            for event in reversed(self.audit_events):  # Most recent first
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if event_types and event.event_type not in event_types:
                    continue
                if ip_address and event.ip_address != ip_address:
                    continue
                if user_id and event.user_id != user_id:
                    continue

                filtered_events.append(event)

                if len(filtered_events) >= limit:
                    break

            return filtered_events

    def generate_compliance_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate a compliance report for a date range"""
        start_timestamp = datetime.fromisoformat(start_date.replace('Z', '+00:00')).timestamp()
        end_timestamp = datetime.fromisoformat(end_date.replace('Z', '+00:00')).timestamp()

        with self._lock:
            events_in_range = [
                event for event in self.audit_events
                if start_timestamp <= event.timestamp <= end_timestamp
            ]

            # Count events by type
            event_counts = defaultdict(int)
            severity_counts = defaultdict(int)

            for event in events_in_range:
                event_counts[event.event_type.value] += 1
                severity_counts[event.severity] += 1

            return {
                "report_period": {"start": start_date, "end": end_date},
                "total_events": len(events_in_range),
                "events_by_type": dict(event_counts),
                "events_by_severity": dict(severity_counts),
                "sample_events": [self._event_to_dict(event) for event in events_in_range[:10]]
            }

    def _event_to_dict(self, event: SecurityEvent) -> Dict[str, Any]:
        """Convert SecurityEvent to dictionary for reporting"""
        return {
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "ip_address": event.ip_address,
            "endpoint": event.endpoint,
            "method": event.method,
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "details": event.details,
            "severity": event.severity,
            "session_id": event.session_id
        }


# Global security audit logger instance
security_audit_logger = SecurityAuditLogger()


# Decorators for easy integration
def audit_authenticated_access(event_type: SecurityEventType = SecurityEventType.DATA_ACCESS):
    """Decorator to audit authenticated access to endpoints"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request information
            request = kwargs.get('request') or (args[0] if args else None)

            user_id = "unknown"
            ip_address = "unknown"
            endpoint = f"{func.__module__}.{func.__name__}"
            method = "UNKNOWN"

            if hasattr(request, 'client') and request.client:
                ip_address = request.client.host
            if hasattr(request, 'method'):
                method = request.method

            # Extract user ID if available (this depends on your auth implementation)
            # For now, we'll just log as "authenticated_user" or similar
            user_id = getattr(request.state, 'user_id', 'anonymous')

            try:
                result = await func(*args, **kwargs)

                # Log successful access
                security_audit_logger.log_data_access(
                    user_id=user_id,
                    ip_address=ip_address,
                    endpoint=endpoint,
                    method=method,
                    resource_id=endpoint,
                    details={"result": "success"}
                )

                return result
            except Exception as e:
                # Log failed access attempt
                security_audit_logger.log_security_event(SecurityEvent(
                    event_type=event_type,
                    user_id=user_id,
                    ip_address=ip_address,
                    endpoint=endpoint,
                    method=method,
                    timestamp=time.time(),
                    details={"error": str(e), "result": "failure"},
                    severity="warning"
                ))
                raise

        return wrapper
    return decorator


def audit_api_key_usage():
    """Decorator to audit API key usage"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or (args[0] if args else None)

            user_id = "unknown"
            ip_address = "unknown"
            endpoint = f"{func.__module__}.{func.__name__}"
            method = "UNKNOWN"

            if hasattr(request, 'client') and request.client:
                ip_address = request.client.host
            if hasattr(request, 'method'):
                method = request.method

            # Extract API key info if available
            auth_header = request.headers.get('authorization', '')
            if auth_header.startswith('Bearer ') or auth_header.startswith('ApiKey '):
                api_key_hash = hashlib.sha256(auth_header[7:].encode()).hexdigest()[:16]
                user_id = f"api_key_{api_key_hash}"

            try:
                result = await func(*args, **kwargs)

                # Log API key usage
                security_audit_logger.log_security_event(SecurityEvent(
                    event_type=SecurityEventType.API_KEY_USED,
                    user_id=user_id,
                    ip_address=ip_address,
                    endpoint=endpoint,
                    method=method,
                    timestamp=time.time(),
                    details={"result": "success"},
                    severity="info"
                ))

                return result
            except Exception as e:
                security_audit_logger.log_security_event(SecurityEvent(
                    event_type=SecurityEventType.API_KEY_USED,
                    user_id=user_id,
                    ip_address=ip_address,
                    endpoint=endpoint,
                    method=method,
                    timestamp=time.time(),
                    details={"error": str(e), "result": "failure"},
                    severity="warning"
                ))
                raise

        return wrapper
    return decorator


# Example usage functions
def run_security_audit_cycle():
    """Run periodic security audit checks"""
    # Detect anomalies
    anomalies = security_audit_logger.detect_anomalies()

    # You could add additional security checks here

    return anomalies


if __name__ == "__main__":
    # Example usage
    print("Security audit logging module initialized")

    # Example of logging different security events
    security_audit_logger.log_authentication_attempt(
        user_id="user123",
        ip_address="192.168.1.100",
        success=True,
        endpoint="/api/v1/query",
        method="POST",
        details={"auth_method": "bearer_token"}
    )

    security_audit_logger.log_authorization_check(
        user_id="user123",
        ip_address="192.168.1.100",
        endpoint="/api/v1/admin",
        method="DELETE",
        resource="user_data",
        allowed=False,
        details={"reason": "insufficient_permissions"}
    )

    security_audit_logger.log_input_validation_failure(
        user_id="user123",
        ip_address="192.168.1.100",
        endpoint="/api/v1/query",
        method="POST",
        field="question",
        value="<script>alert('xss')</script>",
        rule="no_script_tags"
    )

    # Generate a sample compliance report
    import datetime
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)

    report = security_audit_logger.generate_compliance_report(
        start_date=yesterday.isoformat(),
        end_date=now.isoformat()
    )

    print("Sample compliance report:", json.dumps(report, indent=2))