"""Better Auth middleware for session validation."""

from typing import Optional
from uuid import UUID

import httpx
from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

from src.config import get_settings


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

    This middleware calls Better Auth's session endpoint to validate
    the session cookie and retrieve user information.

    Args:
        request: The FastAPI request object.

    Returns:
        AuthenticatedUser with validated user information.

    Raises:
        HTTPException: 401 if session is invalid or missing.
    """
    settings = get_settings()

    # Get session cookie
    session_cookie = request.cookies.get("better-auth.session_token")

    if not session_cookie:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Cookie"},
        )

    try:
        # Call Better Auth API to validate session
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.better_auth_url}/api/auth/get-session",
                cookies={"better-auth.session_token": session_cookie},
                timeout=10.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired session",
                )

            session_data = response.json()

            # Extract user info from session
            user = session_data.get("user", {})
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid session data",
                )

            return AuthenticatedUser(
                id=UUID(user.get("id")),
                email=user.get("email"),
                display_name=user.get("name"),
            )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Authentication service error: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid session data: {str(e)}",
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
