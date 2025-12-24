"""SQL query functions for CRUD operations on all entities."""

import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID

import asyncpg

from .connection import get_db_pool
from .models import (
    User,
    UserCreate,
    Session,
    SessionCreate,
    Message,
    MessageCreate,
    RetrievalLog,
    RetrievalLogCreate,
    PerformanceMetric,
    PerformanceMetricCreate,
    Citation,
    UserProfile,
    SoftwareBackground,
    HardwareBackground,
    ProfileUpdateRequest,
    PersonalizationLogEntry,
    TranslationLogEntry,
)


# ============================================================================
# User Operations
# ============================================================================


async def create_user(data: UserCreate) -> User:
    """Create a new user."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (email, display_name)
            VALUES ($1, $2)
            RETURNING id, email, display_name, created_at, last_login
            """,
            data.email,
            data.display_name,
        )
        return User(**dict(row))


async def get_user_by_id(user_id: UUID) -> Optional[User]:
    """Get a user by ID."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, email, display_name, created_at, last_login
            FROM users WHERE id = $1
            """,
            user_id,
        )
        return User(**dict(row)) if row else None


async def get_user_by_email(email: str) -> Optional[User]:
    """Get a user by email."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, email, display_name, created_at, last_login
            FROM users WHERE email = $1
            """,
            email,
        )
        return User(**dict(row)) if row else None


async def update_user_last_login(user_id: UUID) -> None:
    """Update user's last login timestamp."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_login = NOW() WHERE id = $1",
            user_id,
        )


async def get_or_create_user(email: str, display_name: Optional[str] = None) -> User:
    """Get existing user or create new one."""
    user = await get_user_by_email(email)
    if user:
        await update_user_last_login(user.id)
        return user
    return await create_user(UserCreate(email=email, display_name=display_name))


# ============================================================================
# Session Operations
# ============================================================================


async def create_session(data: SessionCreate) -> Session:
    """Create a new chat session."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id)
            VALUES ($1)
            RETURNING id, user_id, created_at, last_activity, is_active
            """,
            data.user_id,
        )
        return Session(**dict(row))


async def get_session_by_id(session_id: UUID) -> Optional[Session]:
    """Get a session by ID."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, created_at, last_activity, is_active
            FROM sessions WHERE id = $1
            """,
            session_id,
        )
        return Session(**dict(row)) if row else None


async def get_user_sessions(
    user_id: UUID, limit: int = 50, include_inactive: bool = False
) -> List[Session]:
    """Get all sessions for a user."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        query = """
            SELECT id, user_id, created_at, last_activity, is_active
            FROM sessions
            WHERE user_id = $1
        """
        if not include_inactive:
            query += " AND is_active = TRUE"
        query += " ORDER BY last_activity DESC LIMIT $2"

        rows = await conn.fetch(query, user_id, limit)
        return [Session(**dict(row)) for row in rows]


async def update_session_activity(session_id: UUID) -> None:
    """Update session's last activity timestamp."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET last_activity = NOW() WHERE id = $1",
            session_id,
        )


async def deactivate_old_sessions(days: int = 30) -> int:
    """Deactivate sessions older than specified days.

    Returns:
        Number of sessions deactivated.
    """
    pool = await get_db_pool()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET is_active = FALSE
            WHERE last_activity < $1 AND is_active = TRUE
            """,
            cutoff,
        )
        # Extract affected row count from result string
        return int(result.split()[-1]) if result else 0


async def delete_old_sessions(days: int = 30) -> int:
    """Delete sessions older than specified days.

    Returns:
        Number of sessions deleted.
    """
    pool = await get_db_pool()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM sessions WHERE last_activity < $1",
            cutoff,
        )
        return int(result.split()[-1]) if result else 0


# ============================================================================
# Message Operations
# ============================================================================


async def create_message(data: MessageCreate) -> Message:
    """Create a new message."""
    pool = await get_db_pool()

    # Convert citations to JSON
    citations_json = json.dumps([c.model_dump() for c in data.citations])

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, citations)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id, session_id, role, content, citations, created_at
            """,
            data.session_id,
            data.role,
            data.content,
            citations_json,
        )
        result = dict(row)
        # Parse citations back from JSON (asyncpg returns JSONB as Python objects)
        if result["citations"]:
            # Ensure citations is a list, handle both string and list cases
            citations_data = result["citations"]
            if isinstance(citations_data, str):
                citations_data = json.loads(citations_data)
            result["citations"] = [Citation(**c) for c in citations_data] if citations_data else []
        else:
            result["citations"] = []
        return Message(**result)


async def get_session_messages(
    session_id: UUID, limit: int = 100, offset: int = 0
) -> List[Message]:
    """Get all messages for a session."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, role, content, citations, created_at
            FROM messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            session_id,
            limit,
            offset,
        )
        messages = []
        for row in rows:
            result = dict(row)
            # Parse citations (asyncpg returns JSONB as Python objects)
            if result["citations"]:
                citations_data = result["citations"]
                if isinstance(citations_data, str):
                    citations_data = json.loads(citations_data)
                result["citations"] = [Citation(**c) for c in citations_data] if citations_data else []
            else:
                result["citations"] = []
            messages.append(Message(**result))
        return messages


# ============================================================================
# Retrieval Log Operations
# ============================================================================


async def create_retrieval_log(data: RetrievalLogCreate) -> RetrievalLog:
    """Create a new retrieval log entry."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO retrieval_logs (session_id, message_id, query_text, vector_ids, similarity_scores)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, session_id, message_id, query_text, vector_ids, similarity_scores, created_at
            """,
            data.session_id,
            data.message_id,
            data.query_text,
            data.vector_ids,
            data.similarity_scores,
        )
        return RetrievalLog(**dict(row))


# ============================================================================
# Performance Metric Operations
# ============================================================================


async def create_performance_metric(data: PerformanceMetricCreate) -> PerformanceMetric:
    """Create a new performance metric entry."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO performance_metrics
            (session_id, message_id, latency_ms, input_tokens, output_tokens, model_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, session_id, message_id, latency_ms, input_tokens, output_tokens, model_id, created_at
            """,
            data.session_id,
            data.message_id,
            data.latency_ms,
            data.input_tokens,
            data.output_tokens,
            data.model_id,
        )
        return PerformanceMetric(**dict(row))


# ============================================================================
# User Profile Operations (005-user-personalization)
# ============================================================================


async def get_user_profile(user_id: UUID) -> Optional[UserProfile]:
    """
    Get a user's full profile including technical background.

    Args:
        user_id: UUID of the user

    Returns:
        UserProfile or None if not found
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, email, display_name, auth_provider,
                   software_background, hardware_background, profile_completed
            FROM users WHERE id = $1
            """,
            user_id,
        )
        if not row:
            return None

        result = dict(row)

        # Parse JSONB fields (asyncpg may return JSONB as strings or dicts depending on driver)
        if result["software_background"]:
            if isinstance(result["software_background"], str):
                result["software_background"] = SoftwareBackground(**json.loads(result["software_background"]))
            else:
                result["software_background"] = SoftwareBackground(**result["software_background"])
        if result["hardware_background"]:
            if isinstance(result["hardware_background"], str):
                result["hardware_background"] = HardwareBackground(**json.loads(result["hardware_background"]))
            else:
                result["hardware_background"] = HardwareBackground(**result["hardware_background"])

        return UserProfile(**result)


async def update_user_profile(user_id: UUID, data: ProfileUpdateRequest) -> Optional[UserProfile]:
    """
    Update a user's profile with display name and/or technical background.

    Automatically sets profile_completed=true when both software and hardware
    background levels are provided.

    Args:
        user_id: UUID of the user
        data: ProfileUpdateRequest with fields to update

    Returns:
        Updated UserProfile or None if user not found
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Build dynamic update query
        updates = []
        params = [user_id]
        param_count = 1

        if data.display_name is not None:
            param_count += 1
            updates.append(f"display_name = ${param_count}")
            params.append(data.display_name.strip())

        if data.software_background is not None:
            param_count += 1
            updates.append(f"software_background = ${param_count}::jsonb")
            params.append(json.dumps(data.software_background.model_dump()))

        if data.hardware_background is not None:
            param_count += 1
            updates.append(f"hardware_background = ${param_count}::jsonb")
            params.append(json.dumps(data.hardware_background.model_dump()))

        # Auto-set profile_completed if both backgrounds have levels
        if data.software_background is not None and data.hardware_background is not None:
            if data.software_background.level and data.hardware_background.level:
                updates.append("profile_completed = TRUE")

        if not updates:
            # Nothing to update, just return current profile
            return await get_user_profile(user_id)

        query = f"""
            UPDATE users
            SET {", ".join(updates)}
            WHERE id = $1
            RETURNING id, email, display_name, auth_provider,
                      software_background, hardware_background, profile_completed
        """

        row = await conn.fetchrow(query, *params)
        if not row:
            return None

        result = dict(row)

        # Parse JSONB fields (asyncpg may return JSONB as strings or dicts depending on driver)
        if result["software_background"]:
            if isinstance(result["software_background"], str):
                result["software_background"] = SoftwareBackground(**json.loads(result["software_background"]))
            else:
                result["software_background"] = SoftwareBackground(**result["software_background"])
        if result["hardware_background"]:
            if isinstance(result["hardware_background"], str):
                result["hardware_background"] = HardwareBackground(**json.loads(result["hardware_background"]))
            else:
                result["hardware_background"] = HardwareBackground(**result["hardware_background"])

        return UserProfile(**result)


# ============================================================================
# Audit Log Operations (005-user-personalization)
# ============================================================================


async def log_personalization(
    user_id: UUID,
    chapter_id: str,
    status: str,
    response_time_ms: Optional[int] = None,
    error_message: Optional[str] = None,
) -> UUID:
    """
    Log a personalization request for auditing.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
        status: 'success', 'failure', or 'timeout'
        response_time_ms: Optional response time in milliseconds
        error_message: Optional error message if failed

    Returns:
        UUID of the created log entry
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO personalization_logs
            (user_id, chapter_id, status, response_time_ms, error_message)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            user_id,
            chapter_id,
            status,
            response_time_ms,
            error_message,
        )
        return row["id"]


async def log_translation(
    user_id: UUID,
    chapter_id: str,
    target_language: str,
    status: str,
    response_time_ms: Optional[int] = None,
    error_message: Optional[str] = None,
) -> UUID:
    """
    Log a translation request for auditing.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
        target_language: Target language code (e.g., 'ur')
        status: 'success', 'failure', or 'timeout'
        response_time_ms: Optional response time in milliseconds
        error_message: Optional error message if failed

    Returns:
        UUID of the created log entry
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO translation_logs
            (user_id, chapter_id, target_language, status, response_time_ms, error_message)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            user_id,
            chapter_id,
            target_language,
            status,
            response_time_ms,
            error_message,
        )
        return row["id"]


async def get_user_personalization_history(
    user_id: UUID, limit: int = 20
) -> List[PersonalizationLogEntry]:
    """
    Get a user's personalization request history.

    Args:
        user_id: UUID of the user
        limit: Maximum number of entries to return

    Returns:
        List of PersonalizationLogEntry
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, chapter_id, request_timestamp,
                   response_time_ms, status, error_message
            FROM personalization_logs
            WHERE user_id = $1
            ORDER BY request_timestamp DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
        return [PersonalizationLogEntry(**dict(row)) for row in rows]


async def get_user_translation_history(
    user_id: UUID, limit: int = 20
) -> List[TranslationLogEntry]:
    """
    Get a user's translation request history.

    Args:
        user_id: UUID of the user
        limit: Maximum number of entries to return

    Returns:
        List of TranslationLogEntry
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, user_id, chapter_id, target_language, request_timestamp,
                   response_time_ms, status, error_message
            FROM translation_logs
            WHERE user_id = $1
            ORDER BY request_timestamp DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
        return [TranslationLogEntry(**dict(row)) for row in rows]
