"""
Authentication Router

Provides endpoints for user authentication using Better Auth pattern:
- POST /api/auth/signup - Create new user account
- POST /api/auth/login - Authenticate existing user
- POST /api/auth/logout - Terminate session
- GET /api/auth/session - Validate current session

Session management uses HTTP-only cookies stored in Neon Postgres.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel, EmailStr, Field

from src.db.connection import get_db_pool
from src.config import get_settings

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
ph = PasswordHasher()


# ============================================================================
# Request/Response Models
# ============================================================================


class SignupRequest(BaseModel):
    """Signup request with email and password."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Login request with email and password."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    display_name: Optional[str]
    auth_provider: Optional[str] = "email"
    created_at: Optional[str] = None
    last_login: Optional[str] = None


class SessionInfo(BaseModel):
    """Session information."""

    token: str
    expires_at: str


class AuthSuccessResponse(BaseModel):
    """Successful authentication response."""

    user: UserResponse
    session: SessionInfo


class SessionResponse(BaseModel):
    """Current session response."""

    user: UserResponse
    session: dict


class LogoutResponse(BaseModel):
    """Logout response."""

    message: str


class ErrorDetail(BaseModel):
    """Structured error response."""

    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: ErrorDetail


# ============================================================================
# Authentication Endpoints
# ============================================================================


@router.post("/signup", response_model=AuthSuccessResponse, status_code=201)
async def signup(request: SignupRequest, response: Response):
    """
    Create a new user account.

    On success:
    - Creates user in database with hashed password
    - Creates session with 7-day expiration
    - Sets HTTP-only session cookie
    - Returns user and session information
    """
    pool = await get_db_pool()

    # Validate password length (Pydantic handles this, but explicit check for clarity)
    if len(request.password) < 8:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "AUTH_005",
                    "message": "Password must be at least 8 characters",
                    "field": "password",
                }
            },
        )

    async with pool.acquire() as conn:
        # Check if user already exists
        existing_user = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1", request.email
        )

        if existing_user:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": {
                        "code": "AUTH_002",
                        "message": "Email already registered",
                        "field": "email",
                    }
                },
            )

        # Hash password
        password_hash = ph.hash(request.password)

        # Create user
        try:
            user = await conn.fetchrow(
                """
                INSERT INTO users (email, password_hash, created_at)
                VALUES ($1, $2, NOW())
                RETURNING id, email, display_name, created_at
                """,
                request.email,
                password_hash,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "code": "DB_002",
                        "message": "Failed to create user",
                        "field": None,
                    }
                },
            )

        # Create session
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.utcnow() + timedelta(days=7)

        await conn.execute(
            """
            INSERT INTO auth_sessions (user_id, session_token, expires_at)
            VALUES ($1, $2, $3)
            """,
            user["id"],
            session_token,
            expires_at,
        )

    # Set session cookie with environment-aware security settings
    settings = get_settings()
    response.set_cookie(
        key="session",
        value=session_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=604800,  # 7 days in seconds
    )

    return AuthSuccessResponse(
        user=UserResponse(
            id=str(user["id"]),
            email=user["email"],
            display_name=user["display_name"],
            created_at=user["created_at"].isoformat() if user["created_at"] else None,
        ),
        session=SessionInfo(token=session_token, expires_at=expires_at.isoformat()),
    )


@router.post("/login", response_model=AuthSuccessResponse)
async def login(request: LoginRequest, response: Response):
    """
    Authenticate an existing user.

    On success:
    - Verifies password against stored hash
    - Creates new session with 7-day expiration
    - Sets HTTP-only session cookie
    - Updates last_login timestamp
    - Returns user and session information
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Get user by email
        user = await conn.fetchrow(
            "SELECT * FROM users WHERE email = $1", request.email
        )

        if not user:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": "AUTH_001",
                        "message": "Invalid email or password",
                        "field": None,
                    }
                },
            )

        # Verify password
        try:
            ph.verify(user["password_hash"], request.password)
        except VerifyMismatchError:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": "AUTH_001",
                        "message": "Invalid email or password",
                        "field": None,
                    }
                },
            )

        # Create session
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.utcnow() + timedelta(days=7)

        await conn.execute(
            """
            INSERT INTO auth_sessions (user_id, session_token, expires_at)
            VALUES ($1, $2, $3)
            """,
            user["id"],
            session_token,
            expires_at,
        )

        # Update last_login
        await conn.execute(
            "UPDATE users SET last_login = NOW() WHERE id = $1", user["id"]
        )

    # Set session cookie with environment-aware security settings
    settings = get_settings()
    response.set_cookie(
        key="session",
        value=session_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=604800,
    )

    return AuthSuccessResponse(
        user=UserResponse(
            id=str(user["id"]),
            email=user["email"],
            display_name=user["display_name"],
            last_login=datetime.utcnow().isoformat(),
        ),
        session=SessionInfo(token=session_token, expires_at=expires_at.isoformat()),
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    response: Response,
    session_token: Optional[str] = Cookie(None, alias="session"),
):
    """
    Terminate the current session.

    Deletes session from database and clears cookie.
    """

    if not session_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "AUTH_003",
                    "message": "No active session",
                    "field": None,
                }
            },
        )

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Delete session from database
        await conn.execute(
            "DELETE FROM auth_sessions WHERE session_token = $1", session_token
        )

    # Clear cookie
    response.delete_cookie("session")

    return LogoutResponse(message="Logged out successfully")


@router.get("/session", response_model=SessionResponse)
async def get_session(session_token: Optional[str] = Cookie(None, alias="session")):
    """
    Validate current session and return user information.

    Used by frontend to check authentication status.
    Includes profile_completed flag for 005-user-personalization feature.
    """

    if not session_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "AUTH_003",
                    "message": "No active session",
                    "field": None,
                }
            },
        )

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Query session with user join - include profile_completed
        session = await conn.fetchrow(
            """
            SELECT s.expires_at, u.id, u.email, u.display_name, u.auth_provider,
                   COALESCE(u.profile_completed, FALSE) as profile_completed
            FROM auth_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = $1 AND s.expires_at > NOW()
            """,
            session_token,
        )

        if not session:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": "AUTH_003",
                        "message": "Session expired or invalid",
                        "field": None,
                    }
                },
            )

    return SessionResponse(
        user=UserResponse(
            id=str(session["id"]),
            email=session["email"],
            display_name=session["display_name"],
            auth_provider=session["auth_provider"],
        ),
        session={
            "expires_at": session["expires_at"].isoformat(),
            "profile_completed": session["profile_completed"],
        },
    )


@router.get("/get-session", response_model=SessionResponse)
async def get_session_alias(session_token: Optional[str] = Cookie(None, alias="session")):
    """
    Alias for /session endpoint - Better Auth React compatibility.
    """
    return await get_session(session_token)


# Better Auth compatibility endpoints
@router.post("/sign-up/email", response_model=AuthSuccessResponse, status_code=201)
async def better_auth_signup(request: SignupRequest, response: Response):
    """
    Better Auth-compatible signup endpoint.
    Alias for /signup - matches Better Auth client expectations.
    """
    return await signup(request, response)


@router.post("/sign-in/email", response_model=AuthSuccessResponse)
async def better_auth_login(request: LoginRequest, response: Response):
    """
    Better Auth-compatible login endpoint.
    Alias for /login - matches Better Auth client expectations.
    """
    return await login(request, response)


@router.post("/sign-out", response_model=LogoutResponse)
async def better_auth_logout(
    response: Response,
    session_token: Optional[str] = Cookie(None, alias="session"),
):
    """
    Better Auth-compatible logout endpoint.
    Alias for /logout - matches Better Auth client expectations.
    """
    return await logout(response, session_token)


# ============================================================================
# Profile Management Endpoints
# ============================================================================


class UpdateProfileRequest(BaseModel):
    """Profile update request."""

    display_name: Optional[str] = Field(None, min_length=1, max_length=100)


class ProfileResponse(BaseModel):
    """Profile update response."""

    user: UserResponse
    message: str


@router.patch("/profile", response_model=ProfileResponse)
async def update_profile(
    request: UpdateProfileRequest,
    session_token: Optional[str] = Cookie(None, alias="session"),
):
    """
    Update current user's profile.

    Allows users to update their display name.
    """
    if not session_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "AUTH_003",
                    "message": "No active session",
                    "field": None,
                }
            },
        )

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Verify session and get user
        session = await conn.fetchrow(
            """
            SELECT s.expires_at, u.id, u.email, u.display_name, u.auth_provider
            FROM auth_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = $1 AND s.expires_at > NOW()
            """,
            session_token,
        )

        if not session:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": "AUTH_003",
                        "message": "Session expired or invalid",
                        "field": None,
                    }
                },
            )

        # Update display_name if provided
        if request.display_name is not None:
            await conn.execute(
                "UPDATE users SET display_name = $1 WHERE id = $2",
                request.display_name.strip(),
                session["id"],
            )
            new_display_name = request.display_name.strip()
        else:
            new_display_name = session["display_name"]

    return ProfileResponse(
        user=UserResponse(
            id=str(session["id"]),
            email=session["email"],
            display_name=new_display_name,
            auth_provider=session["auth_provider"],
        ),
        message="Profile updated successfully",
    )


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    session_token: Optional[str] = Cookie(None, alias="session"),
):
    """
    Get current user's profile.
    """
    if not session_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "AUTH_003",
                    "message": "No active session",
                    "field": None,
                }
            },
        )

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        session = await conn.fetchrow(
            """
            SELECT s.expires_at, u.id, u.email, u.display_name, u.auth_provider
            FROM auth_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = $1 AND s.expires_at > NOW()
            """,
            session_token,
        )

        if not session:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": "AUTH_003",
                        "message": "Session expired or invalid",
                        "field": None,
                    }
                },
            )

    return ProfileResponse(
        user=UserResponse(
            id=str(session["id"]),
            email=session["email"],
            display_name=session["display_name"],
            auth_provider=session["auth_provider"],
        ),
        message="Profile retrieved successfully",
    )
