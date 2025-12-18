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
