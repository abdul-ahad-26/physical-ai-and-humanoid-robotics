"""Better Auth middleware for session validation."""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

from src.config import get_settings
from src.db.connection import get_db_pool


class BetterAuthSession(BaseModel):
    """Session data from Better Auth."""

    user_id: str
    email: str
    display_name: Optional[str] = None


class AuthenticatedUser(BaseModel):
    """Authenticated user context for request handlers."""

    id: UUID
    email: str
    display_name: Optional[str] = None


async def validate_session(request: Request) -> AuthenticatedUser:
    """Validate Better Auth session from request cookies.

    This middleware validates the session cookie by querying the database
    directly instead of calling an external Better Auth service.

    Args:
        request: The FastAPI request object.

    Returns:
        AuthenticatedUser with validated user information.

    Raises:
        HTTPException: 401 if session is invalid or missing.
    """
    # Get session cookie
    session_cookie = request.cookies.get("session")

    if not session_cookie:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Cookie"},
        )

    try:
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            # Query session with user join
            session = await conn.fetchrow(
                """
                SELECT s.expires_at, u.id, u.email, u.display_name
                FROM auth_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = $1 AND s.expires_at > NOW()
                """,
                session_cookie,
            )

            if not session:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired session",
                )

            return AuthenticatedUser(
                id=UUID(str(session["id"])),
                email=session["email"],
                display_name=session["display_name"],
            )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid session data: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Authentication error: {str(e)}",
        )


async def optional_session(request: Request) -> Optional[AuthenticatedUser]:
    """Optionally validate session, returning None if not authenticated.

    Use this for endpoints that work with or without authentication.

    Args:
        request: The FastAPI request object.

    Returns:
        AuthenticatedUser if valid session exists, None otherwise.
    """
    try:
        return await validate_session(request)
    except HTTPException:
        return None


# Dependency for protected endpoints
def get_current_user(
    user: AuthenticatedUser = Depends(validate_session),
) -> AuthenticatedUser:
    """Dependency that requires authenticated user."""
    return user


# Dependency for optional authentication
def get_optional_user(
    user: Optional[AuthenticatedUser] = Depends(optional_session),
) -> Optional[AuthenticatedUser]:
    """Dependency that optionally authenticates user."""
    return user


class RateLimiter:
    """Simple in-memory rate limiter for API endpoints."""

    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}

    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limit.

        Args:
            user_id: The user identifier to check.

        Returns:
            True if within limit, False if exceeded.
        """
        import time

        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        # Get or create user's request history
        if user_id not in self._requests:
            self._requests[user_id] = []

        # Clean old requests outside window
        self._requests[user_id] = [
            req_time
            for req_time in self._requests[user_id]
            if req_time > window_start
        ]

        # Check if within limit
        if len(self._requests[user_id]) >= self.requests_per_minute:
            return False

        # Record new request
        self._requests[user_id].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


class AIRateLimiter:
    """
    Dedicated rate limiter for AI endpoints (personalization/translation).

    Uses per-user token bucket with configurable limits.
    Returns retry_after time for better client handling.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed for user.

        Args:
            user_id: The user identifier to check.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int)
        """
        import time

        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Get or create user's request history
        if user_id not in self._requests:
            self._requests[user_id] = []

        # Clean old requests outside window
        self._requests[user_id] = [
            req_time
            for req_time in self._requests[user_id]
            if req_time > window_start
        ]

        # Check if within limit
        if len(self._requests[user_id]) >= self.max_requests:
            # Calculate retry_after based on oldest request in window
            oldest = min(self._requests[user_id])
            retry_after = int(oldest + self.window_seconds - current_time) + 1
            return False, max(retry_after, 1)

        # Record new request
        self._requests[user_id].append(current_time)
        return True, 0

    def get_remaining(self, user_id: str) -> int:
        """Get remaining requests for user in current window."""
        import time

        current_time = time.time()
        window_start = current_time - self.window_seconds

        if user_id not in self._requests:
            return self.max_requests

        # Count requests in window
        recent_requests = [
            req_time
            for req_time in self._requests[user_id]
            if req_time > window_start
        ]
        return max(0, self.max_requests - len(recent_requests))


# Global AI rate limiter instance (10 requests per minute)
ai_rate_limiter = AIRateLimiter(max_requests=10, window_seconds=60)


async def check_rate_limit(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """Dependency that checks rate limit for authenticated user.

    Args:
        user: The authenticated user.

    Returns:
        The authenticated user if within rate limit.

    Raises:
        HTTPException: 429 if rate limit exceeded.
    """
    settings = get_settings()
    rate_limiter.requests_per_minute = settings.rate_limit_requests

    if not await rate_limiter.check_rate_limit(str(user.id)):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {settings.rate_limit_requests} requests per minute.",
            headers={"Retry-After": "60"},
        )

    return user
