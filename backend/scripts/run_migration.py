#!/usr/bin/env python3
"""
Database Migration Runner Script

This script runs SQL migration files against the Neon Postgres database.
It is idempotent - safe to run multiple times.

Usage:
    python scripts/run_migration.py scripts/migrations/005_add_profile_fields.sql

Environment Variables:
    DATABASE_URL - Neon Postgres connection string
"""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


async def run_migration(migration_file: str) -> None:
    """Run a SQL migration file against the database."""

    # Load environment variables
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print("❌ ERROR: DATABASE_URL not found in environment")
        print("Please set DATABASE_URL in your .env file")
        sys.exit(1)

    # Check if migration file exists
    migration_path = Path(migration_file)
    if not migration_path.exists():
        print(f"❌ ERROR: Migration file not found: {migration_file}")
        sys.exit(1)

    print(f"Reading migration file: {migration_file}")
    migration_sql = migration_path.read_text()

    print("Connecting to database...")

    try:
        conn = await asyncpg.connect(database_url)
        print("✓ Connected successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    try:
        print(f"\nRunning migration: {migration_path.name}")
        print("=" * 60)

        # Execute the migration SQL
        async with conn.transaction():
            await conn.execute(migration_sql)

        print("=" * 60)
        print("✓ Migration executed successfully")

        # Run verification query if present in the SQL
        if "verification query" in migration_sql.lower():
            print("\nRunning verification checks...")
            print("-" * 60)

            # Extract verification query (after the last comment block)
            lines = migration_sql.split('\n')
            verification_lines = []
            in_verification = False

            for line in lines:
                if 'verification query' in line.lower():
                    in_verification = True
                    continue
                if in_verification and line.strip() and not line.strip().startswith('--'):
                    verification_lines.append(line)

            if verification_lines:
                verification_sql = '\n'.join(verification_lines)
                try:
                    results = await conn.fetch(verification_sql)
                    print("\nVerification Results:")
                    for row in results:
                        column_name = row.get("column_name", "N/A")
                        exists = row.get("exists", False)
                        dropped = row.get("dropped", None)

                        if dropped is not None:
                            status = "✓ DROPPED" if dropped else "✗ STILL EXISTS"
                        else:
                            status = "✓ EXISTS" if exists else "✗ MISSING"

                        print(f"  {status}: {column_name}")
                except Exception as e:
                    print(f"  (Verification query skipped: {e})")

    except Exception as e:
        print(f"\n❌ ERROR during migration: {e}")
        raise

    finally:
        await conn.close()
        print("\nDatabase connection closed")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_migration.py <migration_file>")
        print("\nExample:")
        print("  python scripts/run_migration.py scripts/migrations/005_add_profile_fields.sql")
        sys.exit(1)

    migration_file = sys.argv[1]
    asyncio.run(run_migration(migration_file))


if __name__ == "__main__":
    main()
