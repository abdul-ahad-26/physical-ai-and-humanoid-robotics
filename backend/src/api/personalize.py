"""
Personalization API Router

Provides endpoint for AI-powered content personalization:
- POST /api/personalize - Generate personalized chapter content

Uses OpenAI Agents SDK to adapt content based on user's technical background.
Rate limited to 10 requests per minute per user.
"""

import asyncio
import json
import time
from typing import List
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel, Field

# Timeout for AI requests (60 seconds for long content)
AI_TIMEOUT_SECONDS = 60

from src.config import get_settings
from src.db.connection import get_db_pool
from src.db.queries import get_user_profile, log_personalization
from src.api.middleware import ai_rate_limiter
from src.services.cache import (
    get_cached_personalization,
    set_cached_personalization,
)
from src.services.code_extractor import extract_code, restore_code


router = APIRouter(prefix="/api", tags=["Personalization"])


# ============================================================================
# Request/Response Models
# ============================================================================


class PersonalizeRequest(BaseModel):
    """Request body for content personalization."""

    chapter_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=500000)  # ~500KB max
    title: str = Field(..., min_length=1, max_length=500)


class PersonalizationMetadata(BaseModel):
    """Metadata about the personalization operation."""

    user_level: str
    adaptations_made: List[str]
    cached: bool
    response_time_ms: int


class PersonalizeResponse(BaseModel):
    """Response from personalization endpoint."""

    success: bool
    personalized_content: str
    metadata: PersonalizationMetadata


class ErrorDetail(BaseModel):
    """Structured error response."""

    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: ErrorDetail


# ============================================================================
# Helper Functions
# ============================================================================


async def get_current_user_for_personalize(
    session_token: str | None = Cookie(None, alias="session")
) -> tuple[UUID, dict]:
    """Validate session and return user_id and profile data.

    Returns:
        Tuple of (user_id, profile_dict) with profile_completed status.

    Raises:
        HTTPException: If not authenticated or profile not completed.
    """
    if not session_token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "AUTH_003",
                    "message": "Authentication required",
                    "field": None,
                }
            },
        )

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        session = await conn.fetchrow(
            """
            SELECT s.user_id, u.profile_completed,
                   u.software_background, u.hardware_background
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

        return session["user_id"], dict(session)


# ============================================================================
# Personalization Endpoint
# ============================================================================


@router.post("/personalize", response_model=PersonalizeResponse)
async def personalize_content(
    request: PersonalizeRequest,
    response: Response,
    session_token: str | None = Cookie(None, alias="session"),
):
    """
    Generate personalized chapter content based on user's profile.

    Requires:
    - Valid authentication session
    - Completed user profile (profile_completed=true)
    - Rate limit: 10 requests per minute per user

    Process:
    1. Validate authentication and profile completion
    2. Check cache for existing personalization
    3. Extract code blocks from content
    4. Run Personalization Agent
    5. Restore code blocks
    6. Cache result and log to database

    Returns personalized content with metadata about adaptations made.
    """
    start_time = time.time()

    # Step 1: Validate authentication
    user_id, session_data = await get_current_user_for_personalize(session_token)
    user_id_str = str(user_id)

    # Step 2: Check profile completion
    if not session_data.get("profile_completed"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": "PROFILE_001",
                    "message": "Profile must be completed before personalizing content",
                    "field": None,
                }
            },
        )

    # Step 3: Parse JSONB fields if they're strings (do this early for all code paths)
    if session_data.get("software_background") and isinstance(session_data["software_background"], str):
        session_data["software_background"] = json.loads(session_data["software_background"])
    if session_data.get("hardware_background") and isinstance(session_data["hardware_background"], str):
        session_data["hardware_background"] = json.loads(session_data["hardware_background"])

    # Step 4: Check rate limit
    allowed, retry_after = ai_rate_limiter.is_allowed(user_id_str)
    if not allowed:
        response.headers["Retry-After"] = str(retry_after)
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "RATE_001",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    "field": None,
                }
            },
        )

    # Step 5: Check cache
    cached_content = get_cached_personalization(user_id_str, request.chapter_id)
    if cached_content:
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log cache hit
        await log_personalization(
            user_id=user_id,
            chapter_id=request.chapter_id,
            status="success",
            response_time_ms=response_time_ms,
            error_message=None,
        )

        return PersonalizeResponse(
            success=True,
            personalized_content=cached_content,
            metadata=PersonalizationMetadata(
                user_level=_get_user_level(session_data),
                adaptations_made=["(cached result)"],
                cached=True,
                response_time_ms=response_time_ms,
            ),
        )

    # Step 6: Extract code blocks before AI processing
    extracted = extract_code(request.content)

    # Step 7: Run Personalization Agent with timeout
    try:
        # Import here to avoid circular imports and defer loading
        from src.agents.personalization import personalize_content as run_personalization
        from src.agents.personalization import UserProfileData

        # Build profile data for agent (already parsed in step 3)
        software_bg = session_data.get("software_background") or {}
        hardware_bg = session_data.get("hardware_background") or {}

        profile = UserProfileData(
            software_level=software_bg.get("level", "beginner"),
            software_languages=software_bg.get("languages", []),
            software_frameworks=software_bg.get("frameworks", []),
            hardware_level=hardware_bg.get("level", "none"),
            hardware_domains=hardware_bg.get("domains", []),
        )

        # Run the personalization agent with timeout
        result = await asyncio.wait_for(
            run_personalization(
                content=extracted.text,
                profile=profile,
                chapter_title=request.title,
            ),
            timeout=AI_TIMEOUT_SECONDS,
        )

        personalized_text = result.content
        adaptations_made = result.adaptations_made

    except asyncio.TimeoutError:
        # Log timeout error
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_personalization(
            user_id=user_id,
            chapter_id=request.chapter_id,
            status="timeout",
            response_time_ms=response_time_ms,
            error_message=f"AI request timed out after {AI_TIMEOUT_SECONDS} seconds",
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "code": "AI_TIMEOUT",
                    "message": f"Personalization request timed out after {AI_TIMEOUT_SECONDS} seconds. Please try again.",
                    "field": None,
                }
            },
        )
    except Exception as e:
        # Log error and return failure
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_personalization(
            user_id=user_id,
            chapter_id=request.chapter_id,
            status="error",
            response_time_ms=response_time_ms,
            error_message=str(e)[:500],
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "AI_001",
                    "message": "Failed to personalize content. Please try again.",
                    "field": None,
                }
            },
        )

    # Step 7: Restore code blocks
    final_content = restore_code(personalized_text, extracted)

    # Step 8: Cache the result
    set_cached_personalization(user_id_str, request.chapter_id, final_content)

    # Step 9: Log success
    response_time_ms = int((time.time() - start_time) * 1000)
    await log_personalization(
        user_id=user_id,
        chapter_id=request.chapter_id,
        status="success",
        response_time_ms=response_time_ms,
        error_message=None,
    )

    return PersonalizeResponse(
        success=True,
        personalized_content=final_content,
        metadata=PersonalizationMetadata(
            user_level=_get_user_level(session_data),
            adaptations_made=adaptations_made,
            cached=False,
            response_time_ms=response_time_ms,
        ),
    )


def _get_user_level(session_data: dict) -> str:
    """Extract user's software level from session data.

    Note: Expects software_background to already be parsed (done in step 3 of endpoint).
    """
    software_bg = session_data.get("software_background") or {}
    return software_bg.get("level", "beginner")
