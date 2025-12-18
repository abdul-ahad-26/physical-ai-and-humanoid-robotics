# Research: Better Auth Authentication & Infrastructure Setup

**Feature**: 003-auth-infrastructure
**Date**: 2025-12-17
**Status**: ✅ COMPLETED & VALIDATED (2025-12-18)
**Purpose**: Resolve all technical decisions and best practices for authentication infrastructure implementation

**Implementation Outcome**: All decisions below were validated during implementation. Additional decisions (custom chat UI, direct ingestion, Qdrant upgrade) are documented in [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

---

## Decision 1: Better Auth Integration Strategy

### Decision
Use Better Auth with a **hybrid integration approach**:
- **Frontend**: Better Auth client via npm package for React/Docusaurus
- **Backend**: Direct database interaction with Better Auth-compatible session schema
- **Session Management**: Cookie-based sessions stored in Neon Postgres

### Rationale
1. **Better Auth Python Package Limitation**: The `better-auth-python` package mentioned in the spec does not exist. Better Auth is primarily a TypeScript/JavaScript library.
2. **FastAPI Compatibility**: FastAPI can validate Better Auth sessions by directly querying the session table created by Better Auth client, avoiding unnecessary language barriers.
3. **Session Schema Compatibility**: Better Auth creates standardized session tables that can be queried from any backend.
4. **Simplified Stack**: Reduces dependency on non-existent Python packages while maintaining full functionality.

### Alternatives Considered
- **Option A**: Use Better Auth TypeScript backend → Rejected: Would require Node.js backend alongside FastAPI, adding complexity.
- **Option B**: Replace Better Auth with pure FastAPI solution (e.g., FastAPI-Users) → Rejected: Spec requires Better Auth specifically; rewriting authentication from scratch increases scope.
- **Option C**: Use PyJWT for custom session validation → Accepted as implementation detail: FastAPI will validate Better Auth session tokens using direct database queries.

### Implementation Approach
```typescript
// Frontend (Docusaurus): Better Auth client
import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient({
  baseURL: process.env.REACT_APP_API_URL,
});
```

```python
# Backend (FastAPI): Direct session validation
async def get_current_user(session_token: str):
    # Query Better Auth session table directly
    session = await db.query("SELECT * FROM sessions WHERE token = $1", session_token)
    if not session or session.expires_at < now():
        raise HTTPException(401, "Invalid session")
    return session.user_id
```

---

## Decision 2: Frontend Authentication Pages

### Decision
Create **standalone React pages** for login/signup using Docusaurus `@docusaurus/plugin-client-redirects` and custom page components.

### Rationale
1. **Docusaurus Pages**: Docusaurus supports custom React pages in `src/pages/` directory.
2. **Better Auth Integration**: Better Auth client can be imported directly into custom pages.
3. **Theme Consistency**: Pages can use Docusaurus theme variables and custom CSS to match the green (#10B981) chatbot theme.
4. **Routing**: Docusaurus handles routing automatically for files in `src/pages/`.

### Alternatives Considered
- **Modal-based auth**: Login/signup in modals → Rejected: Spec requires dedicated `/login` and `/signup` routes.
- **Third-party UI library**: Use Material-UI or Chakra → Rejected: Adds unnecessary dependencies; custom forms are simple and theme-aligned.

### Implementation Structure
```
frontend/src/pages/
├── login.tsx          # Login page at /login
├── signup.tsx         # Signup page at /signup
└── login.module.css   # Shared styles for auth pages
```

---

## Decision 3: Database Initialization Script

### Decision
Use **asyncpg** (PostgreSQL async driver) with **idempotent SQL statements** using `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`.

### Rationale
1. **Async Compatibility**: FastAPI backend is async; asyncpg is the standard async Postgres driver for Python.
2. **Idempotency**: `IF NOT EXISTS` clauses make the script safe to run multiple times without errors.
3. **Schema Migration**: For existing tables, use `ALTER TABLE IF EXISTS` to add columns like `password_hash`.
4. **Transaction Safety**: Wrap all operations in a transaction to ensure atomicity.

### Alternatives Considered
- **Alembic (migration tool)**: → Rejected: Overkill for one-time initialization; adds complexity.
- **psycopg3**: → Rejected: asyncpg is more mature for async operations.
- **Manual SQL files**: → Rejected: Python script allows for error handling and conditional logic.

### Implementation Pattern
```python
async def init_database(conn):
    async with conn.transaction():
        # Create tables with IF NOT EXISTS
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255),
                display_name VARCHAR(100),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_login TIMESTAMPTZ
            );
        """)

        # Add password_hash column if it doesn't exist
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='users' AND column_name='password_hash'
                ) THEN
                    ALTER TABLE users ADD COLUMN password_hash VARCHAR(255);
                END IF;
            END $$;
        """)

        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id
            ON sessions(user_id);
        """)
```

---

## Decision 4: Content Ingestion Script

### Decision
Use **httpx** (async HTTP client) with **frontmatter** (Python library) for markdown parsing and **pathlib** for file traversal.

### Rationale
1. **Async HTTP**: httpx is the async-compatible equivalent of requests.
2. **Frontmatter Parsing**: `python-frontmatter` library extracts YAML metadata from markdown files.
3. **Path Extraction**: Extract chapter_id and section_id from file paths using regex or path parsing.
4. **Progress Tracking**: Use `tqdm` for progress bars or simple print statements.

### Alternatives Considered
- **Synchronous requests library**: → Rejected: Async allows concurrent ingestion for faster processing.
- **Custom frontmatter parser**: → Rejected: Reinventing the wheel; `python-frontmatter` is battle-tested.
- **Recursive glob**: → Accepted: `Path.rglob('*.md')` is idiomatic for recursive file discovery.

### Implementation Pattern
```python
import frontmatter
import httpx
from pathlib import Path

async def ingest_directory(docs_path: Path, api_url: str):
    async with httpx.AsyncClient() as client:
        for md_file in docs_path.rglob('*.md'):
            # Parse frontmatter and content
            post = frontmatter.load(md_file)
            content = post.content
            metadata = post.metadata

            # Extract chapter_id from path (e.g., "docs/chapter-1/intro.md" -> "chapter-1")
            parts = md_file.relative_to(docs_path).parts
            chapter_id = parts[0] if parts else "unknown"
            section_id = metadata.get('id', md_file.stem)

            # Generate anchor URL
            anchor_url = f"/docs/{chapter_id}/{section_id}"

            # Call ingestion endpoint
            try:
                response = await client.post(
                    f"{api_url}/api/ingest",
                    json={
                        "content": content,
                        "chapter_id": chapter_id,
                        "section_id": section_id,
                        "anchor_url": anchor_url
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                print(f"✓ {md_file.name}: {result['chunks_created']} chunks")
            except Exception as e:
                print(f"✗ {md_file.name}: {e}")
```

---

## Decision 5: Session Validation on Backend

### Decision
Implement a **FastAPI dependency** that validates session cookies by querying the Better Auth session table directly.

### Rationale
1. **Dependency Injection**: FastAPI's dependency system allows reusable authentication logic.
2. **Cookie Extraction**: FastAPI's `Cookie` parameter automatically extracts cookies from requests.
3. **Database Query**: Direct SQL query to validate session token and expiration.
4. **User Context**: Dependency returns user object that can be injected into route handlers.

### Alternatives Considered
- **Middleware-based auth**: → Rejected: Dependencies are more explicit and testable.
- **JWT tokens**: → Rejected: Better Auth uses session-based auth with database-stored tokens.

### Implementation Pattern
```python
from fastapi import Depends, HTTPException, Cookie
import asyncpg

async def get_current_user(
    session_token: str = Cookie(None, alias="session"),
    db: asyncpg.Connection = Depends(get_db_connection)
):
    if not session_token:
        raise HTTPException(401, "Not authenticated")

    # Query Better Auth session table
    session = await db.fetchrow("""
        SELECT s.*, u.id as user_id, u.email, u.display_name
        FROM auth_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_token = $1 AND s.expires_at > NOW()
    """, session_token)

    if not session:
        raise HTTPException(401, "Session expired or invalid")

    return {
        "id": session["user_id"],
        "email": session["email"],
        "display_name": session["display_name"]
    }

# Usage in routes
@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    user = Depends(get_current_user)
):
    # user is guaranteed to be authenticated
    pass
```

---

## Decision 6: Password Hashing

### Decision
Use **Argon2** via `argon2-cffi` library (Python implementation of Argon2).

### Rationale
1. **Security Best Practice**: Argon2 is the winner of the Password Hashing Competition (2015) and recommended by OWASP.
2. **Better Than bcrypt**: Argon2 provides better resistance against GPU/ASIC attacks.
3. **Python Library**: `argon2-cffi` is the standard Python implementation.
4. **Better Auth Compatibility**: Better Auth supports custom password hashing strategies.

### Alternatives Considered
- **bcrypt**: → Acceptable but Argon2 is superior.
- **scrypt**: → Less commonly used than Argon2.
- **PBKDF2**: → Older, less secure against modern attacks.

### Implementation Pattern
```python
from argon2 import PasswordHasher

ph = PasswordHasher()

# Hash password
password_hash = ph.hash(plain_password)

# Verify password
try:
    ph.verify(password_hash, plain_password)
    # Password correct
except:
    # Password incorrect
    raise HTTPException(401, "Invalid credentials")
```

---

## Decision 7: Environment Configuration

### Decision
Use **python-dotenv** for backend and **Docusaurus environment variables** for frontend.

### Rationale
1. **Backend (.env)**: `python-dotenv` loads `.env` files into environment variables.
2. **Frontend (.env)**: Docusaurus supports `.env` files with `REACT_APP_` prefix.
3. **Security**: `.env` files are gitignored by default.
4. **Development**: Allows local development without hardcoding credentials.

### Required Environment Variables

**Backend (.env)**:
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
QDRANT_URL=https://xyz.qdrant.io
QDRANT_API_KEY=xxx
OPENAI_API_KEY=sk-xxx
AUTH_SECRET=random-secret-key-for-session-signing
```

**Frontend (.env)**:
```bash
REACT_APP_API_URL=http://localhost:8000
```

---

## Decision 8: Docusaurus Navbar Logout Button

### Decision
Create a **custom navbar component** using Docusaurus swizzling to add conditional logout button.

### Rationale
1. **Swizzling**: Docusaurus allows "swizzling" (overriding) theme components.
2. **Conditional Rendering**: Show logout button only when user is authenticated.
3. **Better Auth Integration**: Use `useSession` hook to check authentication status.

### Implementation Approach
```bash
# Swizzle navbar component
npm run swizzle @docusaurus/theme-classic Navbar -- --eject
```

```tsx
// src/theme/Navbar/Content/index.tsx
import { useSession, signOut } from '@/lib/auth';

export default function NavbarContent() {
  const { data: session } = useSession();

  return (
    <div className="navbar__items navbar__items--right">
      {/* Existing navbar items */}

      {session && (
        <button
          onClick={() => signOut.email()}
          className="button button--secondary button--sm"
        >
          Logout
        </button>
      )}
    </div>
  );
}
```

---

## Decision 9: Error Handling Strategy

### Decision
Use **structured error responses** with consistent format across all endpoints.

### Rationale
1. **Frontend Clarity**: Structured errors allow frontend to display specific messages.
2. **Debugging**: Include error codes for tracking and logging.
3. **Security**: Generic messages for sensitive operations (e.g., "Invalid credentials" instead of "User not found").

### Error Response Format
```json
{
  "error": {
    "code": "AUTH_001",
    "message": "Invalid email or password",
    "field": "password"
  }
}
```

### Error Codes
- `AUTH_001`: Invalid credentials
- `AUTH_002`: Email already registered
- `AUTH_003`: Session expired
- `AUTH_004`: Invalid email format
- `AUTH_005`: Password too short
- `DB_001`: Database connection error
- `DB_002`: Query failed
- `INGEST_001`: File read error
- `INGEST_002`: API call failed

---

## Decision 10: Testing Strategy

### Decision
Focus on **integration testing** for authentication flows and **idempotency testing** for scripts.

### Rationale
1. **Authentication**: Integration tests verify entire signup/login/logout flows.
2. **Scripts**: Run scripts twice to verify idempotency.
3. **API Endpoints**: Test authentication requirements using pytest.

### Test Structure
```
backend/tests/
├── test_auth.py           # Auth endpoint tests
├── test_init_db.py        # Database initialization tests
├── test_ingest_docs.py    # Content ingestion tests
└── test_session.py        # Session validation tests

frontend/tests/
├── auth.test.tsx          # Login/signup page tests
└── session.test.tsx       # useSession hook tests
```

---

## Summary

All technical decisions resolved:

1. **Better Auth Integration**: Hybrid approach (frontend client + backend direct DB queries)
2. **Frontend Pages**: Standalone React pages in Docusaurus
3. **Database Init**: asyncpg with idempotent SQL statements
4. **Content Ingestion**: httpx + frontmatter + pathlib
5. **Session Validation**: FastAPI dependency with direct DB queries
6. **Password Hashing**: Argon2 via argon2-cffi
7. **Environment Config**: python-dotenv (backend) + Docusaurus env vars (frontend)
8. **Navbar Logout**: Custom swizzled navbar component
9. **Error Handling**: Structured error responses with codes
10. **Testing**: Integration tests for flows, idempotency tests for scripts

No remaining NEEDS CLARIFICATION items. Ready for Phase 1 (data model and contracts).
