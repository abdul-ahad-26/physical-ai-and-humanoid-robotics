"""
Security middleware for the RAG + Agentic AI-Textbook Chatbot API.
Implements security headers, request sanitization, and abuse detection.
"""
from typing import Callable, Awaitable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import re
import html
import urllib.parse
from datetime import datetime
import hashlib
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from urllib.parse import unquote


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all responses to protect against common web vulnerabilities.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # Set security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"  # or "SAMEORIGIN" if needed
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        # Allow Swagger UI and ReDoc assets by being more permissive for docs/redoc endpoints
        if request.url.path in ["/docs", "/redoc"]:
            # More permissive policy for documentation UIs
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' data: https://cdn.jsdelivr.net https://fonts.googleapis.com https://fonts.gstatic.com; "
                "connect-src 'self' https://cdn.jsdelivr.net; "
                "frame-ancestors 'none';"
            )
        else:
            # Stricter policy for all other endpoints
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://www.google-analytics.com; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RequestSanitizationMiddleware(BaseHTTPMiddleware):
    """
    Sanitizes incoming requests to prevent common injection attacks.
    """

    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_request_size = max_request_size

        # Patterns for detecting potential malicious content
        self.malicious_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>', re.IGNORECASE),
            re.compile(r'<object[^>]*>.*?</object>', re.IGNORECASE),
            re.compile(r'<embed[^>]*>.*?</embed>', re.IGNORECASE),
        ]

    def sanitize_input(self, value: str) -> str:
        """
        Sanitize a single input value.
        """
        if not value:
            return value

        # URL decode the value first
        try:
            value = unquote(value)
        except:
            pass  # If URL decoding fails, continue with original value

        # Remove null bytes
        value = value.replace('\x00', '')

        # HTML encode the value
        value = html.escape(value)

        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if pattern.search(value):
                raise HTTPException(
                    status_code=400,
                    detail="Malicious content detected in input"
                )

        return value

    def sanitize_dict(self, data: dict) -> dict:
        """
        Recursively sanitize a dictionary.
        """
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_input(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self.sanitize_item(item) for item in value]
            else:
                sanitized[key] = value
        return sanitized

    def sanitize_item(self, item):
        """
        Sanitize a single item (string, dict, or other).
        """
        if isinstance(item, str):
            return self.sanitize_input(item)
        elif isinstance(item, dict):
            return self.sanitize_dict(item)
        elif isinstance(item, list):
            return [self.sanitize_item(subitem) for subitem in item]
        else:
            return item

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check request size (for non-streaming requests)
        if request.headers.get("content-length"):
            content_length = int(request.headers.get("content-length", 0))
            if content_length > self.max_request_size:
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request too large", "details": "Request exceeds maximum allowed size"}
                )

        # For now, we'll proceed with the request as the sanitization would need to happen
        # at the route level due to FastAPI's request parsing. We'll implement validation
        # at the Pydantic model level instead.

        response = await call_next(request)
        return response


class AbuseDetectionMiddleware(BaseHTTPMiddleware):
    """
    Detects and prevents abuse patterns like DoS attacks, brute force, etc.
    """

    def __init__(self, app, max_requests_per_minute: int = 100,
                 suspicious_request_threshold: int = 10):
        super().__init__(app)
        self.max_requests_per_minute = max_requests_per_minute
        self.suspicious_request_threshold = suspicious_request_threshold
        self.request_log = {}  # In production, use Redis or database
        self.suspicious_patterns = [
            re.compile(r'\.\./'),  # Directory traversal
            re.compile(r'union\s+select', re.IGNORECASE),  # SQL injection
            re.compile(r'drop\s+table', re.IGNORECASE),  # SQL injection
            re.compile(r'exec\s*\(', re.IGNORECASE),  # Command injection
            re.compile(r';\s*drop', re.IGNORECASE),  # SQL injection
            re.compile(r'waitfor\s+delay', re.IGNORECASE),  # SQL injection
        ]

    def get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request, considering potential proxy headers.
        """
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        if request.client and request.client.host:
            return request.client.host

        return "127.0.0.1"

    def check_suspicious_content(self, content: str) -> bool:
        """
        Check if content contains suspicious patterns.
        """
        if not content:
            return False

        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                return True
        return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = self.get_client_ip(request)
        current_time = time.time()

        # Clean old entries (older than 1 minute)
        self.request_log = {
            ip: [(t, c) for t, c in times if current_time - t < 60]
            for ip, times in self.request_log.items()
        }

        # Add current request
        if client_ip not in self.request_log:
            self.request_log[client_ip] = []

        # Check for suspicious content in the request path and query params
        suspicious_content = (
            self.check_suspicious_content(request.url.path) or
            any(self.check_suspicious_content(str(v)) for v in request.query_params.values())
        )

        if suspicious_content:
            # Log suspicious activity
            logging.warning(f"Suspicious content detected from IP {client_ip}: {request.url.path}")
            return JSONResponse(
                status_code=400,
                content={"error": "Suspicious content detected", "details": "Request blocked by security filter"}
            )

        # Add request to log
        self.request_log[client_ip].append((current_time, request.method))

        # Check rate limit
        recent_requests = len(self.request_log[client_ip])
        if recent_requests > self.max_requests_per_minute:
            logging.warning(f"Rate limit exceeded for IP {client_ip}: {recent_requests} requests in last minute")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "details": "Too many requests, please try again later"}
            )

        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log the error but don't expose internal details
            logging.error(f"Error processing request from {client_ip}: {str(e)}")
            raise


# Example usage in main app:
"""
from fastapi import FastAPI

app = FastAPI()

# Add security middlewares
app.add_middleware(AbuseDetectionMiddleware)
app.add_middleware(RequestSanitizationMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
"""