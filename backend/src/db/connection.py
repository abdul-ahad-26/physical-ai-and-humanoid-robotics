"""Database connection module with asyncpg pool initialization."""

import asyncpg
from typing import Optional

from src.config import get_settings

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    """Get or create the database connection pool.

    Returns:
        asyncpg.Pool: The database connection pool.

    Raises:
        ValueError: If DATABASE_URL is not configured.
    """
    global _pool

    if _pool is None:
        settings = get_settings()

        if not settings.database_url:
            raise ValueError("DATABASE_URL environment variable is not set")

        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=1,
            max_size=10,
            ssl="require",
            command_timeout=30,
        )

    return _pool


async def close_db_pool() -> None:
    """Close the database connection pool."""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_db_schema() -> None:
    """Initialize the database schema.

    Creates all required tables if they don't exist.
    """
    pool = await get_db_pool()

    schema_sql = """
    -- Users table (managed by Better Auth, extended for app)
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255),
        display_name VARCHAR(100),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        last_login TIMESTAMP WITH TIME ZONE
    );

    -- Auth sessions table (Better Auth session management)
    CREATE TABLE IF NOT EXISTS auth_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        session_token VARCHAR(255) UNIQUE NOT NULL,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Chat sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        is_active BOOLEAN DEFAULT TRUE
    );

    -- Messages table
    CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        citations JSONB DEFAULT '[]',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Retrieval logs table
    CREATE TABLE IF NOT EXISTS retrieval_logs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
        query_text TEXT NOT NULL,
        vector_ids UUID[] NOT NULL,
        similarity_scores FLOAT[] NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Performance metrics table
    CREATE TABLE IF NOT EXISTS performance_metrics (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
        latency_ms INTEGER NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        model_id VARCHAR(100) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_token ON auth_sessions(session_token);
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);
    CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
    CREATE INDEX IF NOT EXISTS idx_retrieval_logs_session_id ON retrieval_logs(session_id);
    CREATE INDEX IF NOT EXISTS idx_performance_metrics_session_id ON performance_metrics(session_id);
    CREATE INDEX IF NOT EXISTS idx_performance_metrics_created_at ON performance_metrics(created_at);
    """

    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
