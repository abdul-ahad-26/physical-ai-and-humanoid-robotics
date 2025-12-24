"""
Translation API Router

Provides endpoint for AI-powered content translation:
- POST /api/translate - Translate chapter content to Urdu

Uses OpenAI Agents SDK to translate content while preserving code blocks.
Rate limited to 10 requests per minute per user.
"""

import asyncio
import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel, Field

# Timeout for AI requests (60 seconds for long translations)
AI_TIMEOUT_SECONDS = 60

from src.config import get_settings
from src.db.connection import get_db_pool
from src.db.queries import log_translation
from src.api.middleware import ai_rate_limiter
from src.services.cache import (
    get_cached_translation,
    set_cached_translation,
)
from src.services.code_extractor import extract_code, restore_code


router = APIRouter(prefix="/api", tags=["Translation"])


# ============================================================================
# Request/Response Models
# ============================================================================


class TranslateRequest(BaseModel):
    """Request body for content translation."""

    chapter_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=500000)  # ~500KB max
    target_language: str = Field(default="ur", pattern="^ur$")  # Only Urdu supported


class TranslationMetadata(BaseModel):
    """Metadata about the translation operation."""

    source_language: str
    target_language: str
    preserved_blocks: int
    cached: bool
    response_time_ms: int


class TranslateResponse(BaseModel):
    """Response from translation endpoint."""

    success: bool
    translated_content: str
    metadata: TranslationMetadata


class ErrorDetail(BaseModel):
    """Structured error response."""

    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: ErrorDetail


# ============================================================================
# Helper Functions
# ============================================================================


async def get_current_user_for_translate(
    session_token: Optional[str] = Cookie(None, alias="session")
) -> UUID:
    """Validate session and return user_id.

    Args:
        session_token: Session cookie value.

    Returns:
        The authenticated user's UUID.

    Raises:
        HTTPException: If not authenticated.
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
            SELECT s.user_id
            FROM auth_sessions s
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

        return session["user_id"]


# ============================================================================
# Translation Endpoint
# ============================================================================


@router.post("/translate", response_model=TranslateResponse)
async def translate_content(
    request: TranslateRequest,
    response: Response,
    session_token: Optional[str] = Cookie(None, alias="session"),
):
    """
    Translate chapter content to Urdu.

    Requires:
    - Valid authentication session
    - Rate limit: 10 requests per minute per user

    Process:
    1. Validate authentication
    2. Check cache for existing translation
    3. Extract code blocks from content
    4. Run Translation Agent
    5. Restore code blocks
    6. Cache result and log to database

    Returns translated content with RTL styling support and preserved code blocks.
    """
    start_time = time.time()

    # Step 1: Validate authentication
    user_id = await get_current_user_for_translate(session_token)
    user_id_str = str(user_id)

    # Step 2: Validate target language (currently only Urdu supported)
    if request.target_language != "ur":
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "TRANSLATE_001",
                    "message": "Only Urdu (ur) translation is currently supported",
                    "field": "target_language",
                }
            },
        )

    # Step 3: Check rate limit
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

    # Step 4: Check cache
    cached_content = get_cached_translation(user_id_str, request.chapter_id, request.target_language)
    if cached_content:
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log cache hit
        await log_translation(
            user_id=user_id,
            chapter_id=request.chapter_id,
            target_language=request.target_language,
            status="success",
            response_time_ms=response_time_ms,
            error_message=None,
        )

        return TranslateResponse(
            success=True,
            translated_content=cached_content,
            metadata=TranslationMetadata(
                source_language="en",
                target_language=request.target_language,
                preserved_blocks=0,  # Not tracked for cached results
                cached=True,
                response_time_ms=response_time_ms,
            ),
        )

    # Step 5: Extract code blocks before AI processing
    extracted = extract_code(request.content)

    # Step 6: Run Translation Agent with timeout
    try:
        # Import here to avoid circular imports
        from src.agents.translation import translate_content as run_translation

        result = await asyncio.wait_for(
            run_translation(
                content=extracted.text,
                target_language=request.target_language,
            ),
            timeout=AI_TIMEOUT_SECONDS,
        )

        translated_text = result.content
        preserved_blocks = result.preserved_blocks

    except asyncio.TimeoutError:
        # Log timeout error
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_translation(
            user_id=user_id,
            chapter_id=request.chapter_id,
            target_language=request.target_language,
            status="timeout",
            response_time_ms=response_time_ms,
            error_message=f"AI request timed out after {AI_TIMEOUT_SECONDS} seconds",
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": {
                    "code": "AI_TIMEOUT",
                    "message": f"Translation request timed out after {AI_TIMEOUT_SECONDS} seconds. Please try again.",
                    "field": None,
                }
            },
        )
    except Exception as e:
        # Log error and return failure
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_translation(
            user_id=user_id,
            chapter_id=request.chapter_id,
            target_language=request.target_language,
            status="error",
            response_time_ms=response_time_ms,
            error_message=str(e)[:500],
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "AI_002",
                    "message": "Failed to translate content. Please try again.",
                    "field": None,
                }
            },
        )

    # Step 7: Restore code blocks
    final_content = restore_code(translated_text, extracted)

    # Step 8: Cache the result
    set_cached_translation(user_id_str, request.chapter_id, final_content, request.target_language)

    # Step 9: Log success
    response_time_ms = int((time.time() - start_time) * 1000)
    await log_translation(
        user_id=user_id,
        chapter_id=request.chapter_id,
        target_language=request.target_language,
        status="success",
        response_time_ms=response_time_ms,
        error_message=None,
    )

    return TranslateResponse(
        success=True,
        translated_content=final_content,
        metadata=TranslationMetadata(
            source_language="en",
            target_language=request.target_language,
            preserved_blocks=len(extracted.code_blocks) + len(extracted.inline_codes),
            cached=False,
            response_time_ms=response_time_ms,
        ),
    )
