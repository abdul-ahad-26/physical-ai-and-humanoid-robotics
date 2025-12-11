import time
import asyncio
from typing import Dict, Optional
from fastapi import Request, HTTPException
from starlette.responses import Response
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib
import redis
import os
from urllib.parse import urlparse


class RateLimiter:
    """
    Implements hybrid rate limiting with multiple strategies:
    - Request-based limiting
    - Token-based limiting (simulated)
    - Concurrent request limiting
    - Sliding window algorithm for accurate rate limiting
    """

    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 10000, max_concurrent: int = 10):
        """
        Initialize the rate limiter with limits.

        Args:
            requests_per_minute: Max requests per minute per IP
            tokens_per_minute: Max tokens (simulated) per minute per IP
            max_concurrent: Max concurrent requests per IP
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_concurrent = max_concurrent

        # Try to connect to Redis for distributed rate limiting
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test the connection
            self.redis_client.ping()
            self.use_redis = True
            print("Connected to Redis for distributed rate limiting")
        except:
            print("Redis not available, falling back to in-memory rate limiting")
            self.redis_client = None
            self.use_redis = False

        # Track requests per IP
        self.requests = defaultdict(deque)
        self.tokens_used = defaultdict(deque)
        self.concurrent_requests = defaultdict(int)

    def is_allowed(self, ip: str, tokens: int = 1) -> tuple[bool, str]:
        """
        Check if a request is allowed based on rate limits.

        Args:
            ip: Client IP address
            tokens: Number of tokens the request consumes (simulated)

        Returns:
            Tuple of (allowed, reason_for_rejection)
        """
        if self.use_redis and self.redis_client is not None:
            try:
                return self._is_allowed_redis(ip, tokens)
            except Exception as e:
                # If Redis fails during runtime, fall back to memory
                print(f"Redis error, falling back to memory: {e}")
                return self._is_allowed_memory(ip, tokens)
        else:
            return self._is_allowed_memory(ip, tokens)

    def _is_allowed_memory(self, ip: str, tokens: int = 1) -> tuple[bool, str]:
        """
        Check if a request is allowed based on rate limits using in-memory storage.

        Args:
            ip: Client IP address
            tokens: Number of tokens the request consumes (simulated)

        Returns:
            Tuple of (allowed, reason_for_rejection)
        """
        now = time.time()
        minute_ago = now - 60

        # Clean old requests (older than 1 minute)
        while self.requests[ip] and self.requests[ip][0] < minute_ago:
            self.requests[ip].popleft()

        # Clean old token usage
        while self.tokens_used[ip] and self.tokens_used[ip][0][0] < minute_ago:
            self.tokens_used[ip].popleft()

        # Check request limit
        if len(self.requests[ip]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

        # Check token limit
        total_tokens_used = sum(item[1] for item in self.tokens_used[ip])
        if total_tokens_used + tokens > self.tokens_per_minute:
            return False, f"Token limit exceeded: {self.tokens_per_minute} tokens per minute"

        # Check concurrent request limit
        if self.concurrent_requests[ip] >= self.max_concurrent:
            return False, f"Concurrent request limit exceeded: {self.max_concurrent} concurrent requests"

        # If allowed, record the request and token usage
        self.requests[ip].append(now)
        self.tokens_used[ip].append((now, tokens))

        # Increment concurrent requests (should be decremented when request completes)
        self.concurrent_requests[ip] += 1

        return True, ""

    def _is_allowed_redis(self, ip: str, tokens: int = 1) -> tuple[bool, str]:
        """
        Check if a request is allowed based on rate limits using Redis for distributed storage.

        Args:
            ip: Client IP address
            tokens: Number of tokens the request consumes (simulated)

        Returns:
            Tuple of (allowed, reason_for_rejection)
        """
        now = time.time()
        pipeline = self.redis_client.pipeline()

        # Use sliding window algorithm with Redis
        # Clean old requests (older than 1 minute)
        pipeline.zremrangebyscore(f"rate_limit:requests:{ip}", "-inf", now - 60)

        # Get current request count
        pipeline.zcard(f"rate_limit:requests:{ip}")

        # Clean old token usage
        pipeline.zremrangebyscore(f"rate_limit:tokens:{ip}", "-inf", now - 60)

        # Get current token usage
        pipeline.zrange(f"rate_limit:tokens:{ip}", 0, -1, withscores=True)

        # Get current concurrent count
        pipeline.get(f"rate_limit:concurrent:{ip}")

        results = pipeline.execute()
        current_requests = results[1]
        current_token_records = results[3]
        current_concurrent_str = results[4]

        current_concurrent = int(current_concurrent_str) if current_concurrent_str else 0

        # Check request limit
        if current_requests >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

        # Check token limit
        total_tokens_used = sum(int(record[1]) for record in current_token_records)
        if total_tokens_used + tokens > self.tokens_per_minute:
            return False, f"Token limit exceeded: {self.tokens_per_minute} tokens per minute"

        # Check concurrent request limit
        if current_concurrent >= self.max_concurrent:
            return False, f"Concurrent request limit exceeded: {self.max_concurrent} concurrent requests"

        # If allowed, record the request and token usage
        pipeline.multi()
        pipeline.zadd(f"rate_limit:requests:{ip}", {str(now): now})
        pipeline.expire(f"rate_limit:requests:{ip}", 61)  # Expire after 61 seconds
        pipeline.zadd(f"rate_limit:tokens:{ip}", {str(now): tokens})
        pipeline.expire(f"rate_limit:tokens:{ip}", 61)  # Expire after 61 seconds
        pipeline.incr(f"rate_limit:concurrent:{ip}")
        pipeline.expire(f"rate_limit:concurrent:{ip}", 300)  # Expire after 5 minutes
        pipeline.execute()

        return True, ""

    def request_completed(self, ip: str):
        """
        Call this when a request completes to decrement concurrent counter.

        Args:
            ip: Client IP address
        """
        if self.use_redis:
            try:
                # Decrement concurrent counter in Redis
                current_count = self.redis_client.decr(f"rate_limit:concurrent:{ip}")
                # Ensure it doesn't go below 0
                if current_count and current_count < 0:
                    self.redis_client.set(f"rate_limit:concurrent:{ip}", 0)
                    self.redis_client.expire(f"rate_limit:concurrent:{ip}", 300)
            except:
                # If Redis fails, skip decrementing
                pass
        else:
            if self.concurrent_requests[ip] > 0:
                self.concurrent_requests[ip] -= 1


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request, considering potential proxy headers.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address
    """
    # Check for forwarded-for header (common with proxies/load balancers)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the first IP if multiple are provided
        return forwarded_for.split(",")[0].strip()

    # Check for real IP header
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    # Fall back to client host
    if request.client and request.client.host:
        return request.client.host

    # Default to localhost if no IP found
    return "127.0.0.1"


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.

    Args:
        request: FastAPI request object
        call_next: Next middleware/function in the chain

    Returns:
        Response from the next function in the chain
    """
    # Determine the endpoint to apply different limits if needed
    endpoint = request.url.path

    # For now, use a simple token count based on request size
    # In a real implementation, this would be based on actual token usage
    content_length = request.headers.get("content-length", "0")
    try:
        tokens = min(int(content_length) // 100, 100)  # Simulate token usage based on content size
    except ValueError:
        tokens = 1

    # Get client IP
    client_ip = get_client_ip(request)

    # Check if request is allowed
    allowed, reason = rate_limiter.is_allowed(client_ip, tokens)

    if not allowed:
        # Clean up the concurrent request counter if rate limited
        rate_limiter.request_completed(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "details": reason
            }
        )

    try:
        # Process the request
        response = await call_next(request)
        return response
    finally:
        # Decrement concurrent request counter when request is done
        rate_limiter.request_completed(client_ip)


# Alternative: Decorator-based rate limiting for specific endpoints
def rate_limit(requests_per_minute: int = None, tokens_per_minute: int = None, max_concurrent: int = None):
    """
    Decorator for applying rate limits to specific endpoints.

    Args:
        requests_per_minute: Max requests per minute
        tokens_per_minute: Max tokens per minute
        max_concurrent: Max concurrent requests
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from function arguments (assumes request is passed)
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                # If no request object found, proceed without rate limiting
                return await func(*args, **kwargs)

            client_ip = get_client_ip(request)

            # Use provided limits or defaults
            rps = requests_per_minute or rate_limiter.requests_per_minute
            tps = tokens_per_minute or rate_limiter.tokens_per_minute
            max_concurrent_local = max_concurrent or rate_limiter.max_concurrent

            # Create a temporary rate limiter with custom limits
            temp_limiter = RateLimiter(rps, tps, max_concurrent_local)
            temp_limiter.requests = rate_limiter.requests  # Share request tracking
            temp_limiter.tokens_used = rate_limiter.tokens_used  # Share token tracking
            temp_limiter.concurrent_requests = rate_limiter.concurrent_requests  # Share concurrent tracking

            # For this simplified version, we'll use the global limiter
            # but with custom parameters if needed in the future
            allowed, reason = rate_limiter.is_allowed(client_ip)

            if not allowed:
                rate_limiter.request_completed(client_ip)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "details": reason
                    }
                )

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                rate_limiter.request_completed(client_ip)

        return wrapper
    return decorator


# Example usage in FastAPI app:
"""
from fastapi import FastAPI
from .middleware.rate_limiter import rate_limit_middleware

app = FastAPI()
app.middleware("http")(rate_limit_middleware)
"""