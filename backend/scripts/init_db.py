#!/usr/bin/env python3
"""
Database Initialization Script

This script creates all required database tables for the RAG chatbot system.
It is idempotent - safe to run multiple times.

Usage:
    python scripts/init_db.py

Environment Variables:
    DATABASE_URL - Neon Postgres connection string
"""

import asyncio
import os
import sys
from typing import List, Tuple

import asyncpg
from dotenv import load_dotenv


async def check_table_exists(conn: asyncpg.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = $1
        );
        """,
        table_name,
    )
    return result


async def check_column_exists(
    conn: asyncpg.Connection, table_name: str, column_name: str
) -> bool:
    """Check if a column exists in a table."""
    result = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = $1
            AND column_name = $2
        );
        """,
        table_name,
        column_name,
    )
    return result


async def init_database() -> None:
    """Initialize all database tables with idempotent operations."""

    # Load environment variables
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print("❌ ERROR: DATABASE_URL not found in environment")
        print("Please set DATABASE_URL in your .env file")
        sys.exit(1)

    print("Connecting to database...")

    try:
        conn = await asyncpg.connect(database_url)
        print("✓ Connected successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    try:
        async with conn.transaction():
            created_tables: List[str] = []
            existing_tables: List[str] = []
            created_indexes: List[str] = []

            # Table 1: Users
            table_name = "users"
            if not await check_table_exists(conn, table_name):
                print(f"Creating table: {table_name}")
                await conn.execute(
                    """
                    CREATE TABLE users (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        email VARCHAR(255) UNIQUE NOT NULL,
                        display_name VARCHAR(100),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        last_login TIMESTAMP WITH TIME ZONE
                    );
                    """
                )
                created_tables.append(table_name)
                print(f"✓ Created table: {table_name}")
            else:
                existing_tables.append(table_name)
                print(f"  Table already exists: {table_name}")

            # Add password_hash column to users if it doesn't exist
            if not await check_column_exists(conn, "users", "password_hash"):
                print("Adding password_hash column to users table")
                await conn.execute(
                    """
                    ALTER TABLE users ADD COLUMN password_hash VARCHAR(255);
                    """
                )
                print("✓ Added column: password_hash")
            else:
                print("  Column already exists: users.password_hash")

            # Table 2: Auth Sessions (Better Auth)
            table_name = "auth_sessions"
            if not await check_table_exists(conn, table_name):
                print(f"Creating table: {table_name}")
                await conn.execute(
                    """
                    CREATE TABLE auth_sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        session_token VARCHAR(255) UNIQUE NOT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    """
                )
                created_tables.append(table_name)
                print(f"✓ Created table: {table_name}")
            else:
                existing_tables.append(table_name)
                print(f"  Table already exists: {table_name}")

            # Table 3: Chat Sessions
            table_name = "sessions"
            if not await check_table_exists(conn, table_name):
                print(f"Creating table: {table_name}")
                await conn.execute(
                    """
                    CREATE TABLE sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT TRUE
                    );
                    """
                )
                created_tables.append(table_name)
                print(f"✓ Created table: {table_name}")
            else:
                existing_tables.append(table_name)
                print(f"  Table already exists: {table_name}")

            # Table 4: Messages
            table_name = "messages"
            if not await check_table_exists(conn, table_name):
                print(f"Creating table: {table_name}")
                await conn.execute(
                    """
                    CREATE TABLE messages (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        citations JSONB DEFAULT '[]',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    """
                )
                created_tables.append(table_name)
                print(f"✓ Created table: {table_name}")
            else:
                existing_tables.append(table_name)
                print(f"  Table already exists: {table_name}")

            # Table 5: Retrieval Logs
            table_name = "retrieval_logs"
            if not await check_table_exists(conn, table_name):
                print(f"Creating table: {table_name}")
                await conn.execute(
                    """
                    CREATE TABLE retrieval_logs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
                        query_text TEXT NOT NULL,
                        vector_ids UUID[] NOT NULL,
                        similarity_scores FLOAT[] NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    """
                )
                created_tables.append(table_name)
                print(f"✓ Created table: {table_name}")
            else:
                existing_tables.append(table_name)
                print(f"  Table already exists: {table_name}")

            # Table 6: Performance Metrics
            table_name = "performance_metrics"
            if not await check_table_exists(conn, table_name):
                print(f"Creating table: {table_name}")
                await conn.execute(
                    """
                    CREATE TABLE performance_metrics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
                        latency_ms INTEGER NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        model_id VARCHAR(100) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    """
                )
                created_tables.append(table_name)
                print(f"✓ Created table: {table_name}")
            else:
                existing_tables.append(table_name)
                print(f"  Table already exists: {table_name}")

            # Create indexes
            print("\nCreating indexes...")

            indexes = [
                ("idx_auth_sessions_token", "auth_sessions", "session_token"),
                ("idx_auth_sessions_user_id", "auth_sessions", "user_id"),
                ("idx_sessions_user_id", "sessions", "user_id"),
                ("idx_sessions_last_activity", "sessions", "last_activity"),
                ("idx_messages_session_id", "messages", "session_id"),
                ("idx_messages_created_at", "messages", "created_at"),
                ("idx_retrieval_logs_session_id", "retrieval_logs", "session_id"),
                ("idx_performance_metrics_session_id", "performance_metrics", "session_id"),
            ]

            for index_name, table_name, column_name in indexes:
                # Check if index exists
                index_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM pg_indexes
                        WHERE schemaname = 'public'
                        AND indexname = $1
                    );
                    """,
                    index_name,
                )

                if not index_exists:
                    await conn.execute(
                        f"CREATE INDEX {index_name} ON {table_name}({column_name});"
                    )
                    created_indexes.append(index_name)
                    print(f"✓ Created index: {index_name}")

            print("\n" + "=" * 60)
            print("DATABASE INITIALIZATION COMPLETE")
            print("=" * 60)
            print(f"\nTables created: {len(created_tables)}")
            if created_tables:
                for table in created_tables:
                    print(f"  - {table}")

            print(f"\nTables already existed: {len(existing_tables)}")
            if existing_tables:
                for table in existing_tables:
                    print(f"  - {table}")

            print(f"\nIndexes created: {len(created_indexes)}")
            if created_indexes:
                for index in created_indexes:
                    print(f"  - {index}")

            print("\n✓ Schema initialization complete")

    except Exception as e:
        print(f"\n❌ ERROR during database initialization: {e}")
        raise

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_database())
