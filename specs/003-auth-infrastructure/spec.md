# Feature Specification: Better Auth Authentication & Infrastructure Setup

**Feature Branch**: `003-auth-infrastructure`
**Created**: 2025-12-17
**Status**: COMPLETED ✅
**Completed**: 2025-12-18
**Input**: User description: "Set up Better Auth authentication and complete infrastructure configuration for the RAG chatbot feature."

**Implementation Notes**: See `IMPLEMENTATION_NOTES.md` for key differences between original spec and actual implementation (custom chat UI, direct ingestion, etc.)

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Sign Up for New Account (Priority: P1)

As a new visitor to the textbook, I want to create an account with my email and password so that I can use the AI chatbot to ask questions about the book content.

**Why this priority**: Core requirement for system operation - users cannot access the chatbot without authentication. This is the first step in the user journey.

**Independent Test**: Can be fully tested by visiting the signup page, entering email and password, and verifying account creation and automatic login.

**Acceptance Scenarios**:

1. **Given** an unauthenticated user visits /signup, **When** they enter a valid email and password (min 8 characters), **Then** a new account is created, they are logged in automatically, and redirected to the homepage with an active session.

2. **Given** a user attempts to sign up, **When** they enter an email that already exists, **Then** the system displays an error message "Email already registered" and suggests logging in instead.

3. **Given** a user attempts to sign up, **When** they enter an invalid email format or password shorter than 8 characters, **Then** the system displays validation errors and prevents submission.

---

### User Story 2 - Log In to Existing Account (Priority: P1)

As a returning user, I want to log in with my email and password so that I can access the chatbot with my conversation history.

**Why this priority**: Essential for returning users - without login, users cannot access the chatbot they've already used.

**Independent Test**: Can be fully tested by visiting the login page, entering valid credentials, and verifying successful authentication and redirect.

**Acceptance Scenarios**:

1. **Given** an unauthenticated user visits /login, **When** they enter valid email and password, **Then** they are logged in and redirected to the homepage with an active session.

2. **Given** a user attempts to log in, **When** they enter incorrect credentials, **Then** the system displays "Invalid email or password" and allows retry.

3. **Given** a logged-in user, **When** they navigate to /login or /signup, **Then** they are automatically redirected to the homepage.

---

### User Story 3 - Access Chat as Authenticated User (Priority: P1)

As a logged-in user, I want to click the chat icon and immediately see the chat interface (not a login prompt) so that I can start asking questions without interruption.

**Why this priority**: Core value delivery - this is what differentiates authenticated from unauthenticated experience. Without this, authentication has no visible benefit.

**Independent Test**: Can be fully tested by logging in, clicking the chat icon, and verifying the chat interface appears (not login prompt).

**Acceptance Scenarios**:

1. **Given** a logged-in user on any textbook page, **When** they click the chat icon, **Then** the full chat interface opens (not a login prompt).

2. **Given** a logged-in user with the chat open, **When** they send a message, **Then** the backend validates their session and processes the request.

3. **Given** a logged-in user, **When** their session expires (e.g., after 7 days), **Then** they are prompted to log in again when attempting to use the chat.

---

### User Story 4 - Log Out from Account (Priority: P2)

As a logged-in user, I want to log out from the navbar so that I can end my session on a shared device or switch accounts.

**Why this priority**: Important for security and multi-user scenarios, but not blocking core functionality.

**Independent Test**: Can be fully tested by logging in, clicking logout, and verifying session termination and redirect to login.

**Acceptance Scenarios**:

1. **Given** a logged-in user, **When** they click the logout button in the navbar, **Then** their session is terminated, and they are redirected to /login.

2. **Given** a user has logged out, **When** they click the chat icon, **Then** they see the login prompt (not the chat interface).

3. **Given** a user logs out, **When** they use the browser back button, **Then** they cannot access authenticated pages without logging in again.

---

### User Story 5 - Initialize Database Schema (Priority: P3)

As a system administrator, I want to run a database initialization script so that all required tables, indexes, and constraints are created in Neon Postgres.

**Why this priority**: Required once during setup, but not a user-facing feature. Can be done manually initially.

**Independent Test**: Can be fully tested by running the init_db.py script and verifying all tables exist with correct schemas.

**Acceptance Scenarios**:

1. **Given** a fresh Neon Postgres database, **When** the init_db.py script is run, **Then** all tables (users, sessions, messages, retrieval_logs, performance_metrics) are created with proper indexes.

2. **Given** the database already has tables, **When** the init_db.py script is run again, **Then** it detects existing tables and exits without errors (idempotent).

3. **Given** the script runs successfully, **When** an administrator verifies the schema, **Then** all foreign key constraints and indexes match the specification in specs/002-rag-chatbot/spec.md.

---

### User Story 6 - Ingest Textbook Content (Priority: P3)

As a content administrator, I want to run a script that ingests all markdown files from frontend/docs/ so that the chatbot can answer questions about the textbook content.

**Why this priority**: Required for chatbot functionality but not user-facing. Can be done manually during setup.

**Independent Test**: Can be fully tested by running the ingest_docs.py script and verifying chunks are created in Qdrant.

**Acceptance Scenarios**:

1. **Given** markdown files exist in frontend/docs/, **When** the ingest_docs.py script is run, **Then** all files are processed, chunked, embedded, and stored in Qdrant with progress displayed.

2. **Given** content is ingested, **When** a user asks a question about that content, **Then** the chatbot retrieves and uses the ingested chunks to generate an answer.

3. **Given** the script is run, **When** it processes files, **Then** chapter_id and section_id are correctly extracted from file paths and frontmatter.

---

### Edge Cases

- **Session Expiration**: User session expires during chat; system returns 401, frontend prompts re-login.
- **Invalid Credentials**: User enters credentials for non-existent account; system returns generic "Invalid email or password" (no account enumeration).
- **Concurrent Logins**: User logs in from multiple devices; all sessions remain valid (Better Auth supports multiple sessions).
- **CSRF Protection**: Better Auth includes CSRF protection by default; all auth requests include CSRF tokens.
- **Database Connection Failure**: init_db.py script cannot connect to Neon; script exits with clear error message.
- **Missing .env Credentials**: Backend cannot find DATABASE_URL; startup fails with clear error message.
- **Empty Docs Directory**: ingest_docs.py runs but finds no markdown files; script logs warning and exits gracefully.
- **Ingestion API Failure**: ingest_docs.py calls /api/ingest but receives error; script retries once, then logs error and continues to next file.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Authentication Requirements (Better Auth)

- **FR-001**: System MUST install Better Auth client library on the Docusaurus frontend for session management.
- **FR-002**: System MUST install Better Auth server library on the FastAPI backend for session validation.
- **FR-003**: System MUST create a signup page at /login route with email and password fields.
- **FR-004**: System MUST create a login page at /signup route with email and password fields.
- **FR-005**: System MUST validate email format and password strength (minimum 8 characters) on both client and server.
- **FR-006**: System MUST hash passwords using a secure algorithm (bcrypt or argon2) before storage.
- **FR-007**: System MUST create a session cookie on successful login/signup with HttpOnly, Secure, and SameSite=Lax flags.
- **FR-008**: System MUST store session data in the Neon Postgres database using Better Auth's session table schema.
- **FR-009**: System MUST provide a logout button in the navbar that terminates the session and clears cookies.
- **FR-010**: System MUST validate user sessions on all protected API endpoints using Better Auth's `auth.api.getSession()` method.
- **FR-011**: System MUST return 401 Unauthorized for API requests without valid sessions.
- **FR-012**: System MUST use the Better Auth `useSession` hook in frontend/src/theme/Root.tsx to check authentication status.
- **FR-013**: System MUST match the green (#10B981) theme on login/signup pages consistent with the chatbot UI.

#### Database Initialization Requirements

- **FR-014**: System MUST provide an initialization script at backend/scripts/init_db.py that creates all database tables.
- **FR-015**: Script MUST implement the exact schema from specs/002-rag-chatbot/spec.md (users, sessions, messages, retrieval_logs, performance_metrics).
- **FR-016**: Script MUST check if tables already exist before creating them (idempotent).
- **FR-017**: Script MUST create all indexes as specified in the schema (idx_sessions_user_id, idx_messages_session_id, etc.).
- **FR-018**: Script MUST create all foreign key constraints with proper CASCADE behavior.
- **FR-019**: Script MUST connect to Neon Postgres using DATABASE_URL from environment variables.
- **FR-020**: Script MUST provide clear success/error messages indicating which tables were created or already existed.
- **FR-021**: System MUST document how to run the initialization script in a README or setup guide.

#### Content Ingestion Requirements

- **FR-022**: System MUST provide a content ingestion script at backend/scripts/ingest_docs.py.
- **FR-023**: Script MUST read all markdown files recursively from frontend/docs/ directory.
- **FR-024**: Script MUST extract chapter_id from file path (e.g., "docs/chapter-1/intro.md" → chapter_id="chapter-1").
- **FR-025**: Script MUST extract section_id from file frontmatter or filename.
- **FR-026**: Script MUST call the existing /api/ingest endpoint for each markdown file with content, chapter_id, section_id, and anchor_url.
- **FR-027**: Script MUST display progress information (files processed, chunks created, errors encountered).
- **FR-028**: Script MUST handle errors gracefully (log errors, continue processing remaining files).
- **FR-029**: System MUST document the ingestion process in a setup guide or README.

#### Integration Requirements

- **FR-030**: System MUST update frontend/src/theme/Root.tsx to use the Better Auth `useSession` hook instead of hardcoded `isAuthenticated=false`.
- **FR-031**: System MUST pass session data to the ChatWidget component when user is authenticated.
- **FR-032**: System MUST configure Better Auth to use cookie-based sessions with 7-day expiration.
- **FR-033**: System MUST store all configuration (database URLs, API keys) in environment variables (.env files).
- **FR-034**: System MUST not commit .env files to version control (.gitignore must exclude them).

---

### Key Entities

- **User**: A registered user account. Attributes: ID (UUID), email (unique), password_hash, display_name, created_at, last_login. Stored in Neon Postgres `users` table.
- **Session**: An authenticated user session. Attributes: ID, user_id (foreign key), session_token, expires_at, created_at. Stored in Neon Postgres by Better Auth.
- **DatabaseSchema**: The complete database structure. Entities: users, sessions, messages, retrieval_logs, performance_metrics with indexes and foreign keys.
- **MarkdownFile**: A textbook content file. Attributes: file_path, chapter_id, section_id, content, frontmatter metadata.
- **IngestedChunk**: A processed chunk stored in Qdrant. Attributes: vector_id, content, chapter_id, section_id, anchor_url, embedding vector.

---

## API Contracts

### POST /api/auth/signup

Create a new user account and return a session.

**Request**:
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (Success - 201 Created)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "display_name": null,
    "created_at": "2025-12-17T10:00:00Z"
  },
  "session": {
    "token": "session-token-string",
    "expires_at": "2025-12-24T10:00:00Z"
  }
}
```

**Error Responses**:
- `400 Bad Request`: Invalid email format or password too short
- `409 Conflict`: Email already registered
- `500 Internal Server Error`: Database error

---

### POST /api/auth/login

Authenticate an existing user and return a session.

**Request**:
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (Success - 200 OK)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "display_name": null,
    "last_login": "2025-12-17T10:00:00Z"
  },
  "session": {
    "token": "session-token-string",
    "expires_at": "2025-12-24T10:00:00Z"
  }
}
```

**Error Responses**:
- `401 Unauthorized`: Invalid email or password
- `500 Internal Server Error`: Database error

---

### POST /api/auth/logout

Terminate the current session.

**Request**:
```
No body required (session identified by cookie)
```

**Response (Success - 200 OK)**:
```json
{
  "message": "Logged out successfully"
}
```

**Error Responses**:
- `401 Unauthorized`: No active session

---

### GET /api/auth/session

Validate the current session and return user data.

**Response (Success - 200 OK)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "display_name": null
  },
  "session": {
    "expires_at": "2025-12-24T10:00:00Z"
  }
}
```

**Error Responses**:
- `401 Unauthorized`: No active session or session expired

---

### POST /api/ingest (Existing - No Changes)

Ingest textbook content into the vector store. This endpoint already exists in the backend.

**Request**:
```json
{
  "content": "markdown content string",
  "chapter_id": "chapter-1",
  "section_id": "introduction",
  "anchor_url": "/docs/chapter-1#introduction"
}
```

**Response (Success - 200 OK)**:
```json
{
  "status": "success",
  "chunks_created": 12,
  "vector_ids": ["uuid-1", "uuid-2", "..."]
}
```

---

## Database Schemas

### Better Auth Session Management Tables

Better Auth will automatically create and manage session tables. These are in addition to the tables defined in specs/002-rag-chatbot/spec.md:

```sql
-- Better Auth creates these tables automatically
CREATE TABLE auth_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_auth_sessions_token ON auth_sessions(session_token);
CREATE INDEX idx_auth_sessions_user_id ON auth_sessions(user_id);
```

### Existing Tables (From specs/002-rag-chatbot/spec.md)

The init_db.py script must create these exact tables:

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Chat sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Retrieval logs table
CREATE TABLE retrieval_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    vector_ids UUID[] NOT NULL,
    similarity_scores FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics table
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

-- Indexes for common queries
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_last_activity ON sessions(last_activity);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_retrieval_logs_session_id ON retrieval_logs(session_id);
CREATE INDEX idx_performance_metrics_session_id ON performance_metrics(session_id);
```

**Note**: The init_db.py script must add a `password_hash` column to the users table for Better Auth integration:

```sql
ALTER TABLE users ADD COLUMN password_hash VARCHAR(255) NOT NULL;
```

---

## Frontend Integration Details

### Better Auth Client Setup (Docusaurus)

**Installation**:
```bash
npm install better-auth
```

**Configuration** (frontend/src/lib/auth.ts):
```typescript
import { createAuthClient } from "better-auth/client";

export const authClient = createAuthClient({
  baseURL: process.env.REACT_APP_API_URL || "http://localhost:8000",
});

export const { useSession, signIn, signUp, signOut } = authClient;
```

**Usage in Root.tsx**:
```typescript
import { useSession } from "@/lib/auth";

export default function Root({ children }: RootProps): JSX.Element {
  const { data: session, isLoading } = useSession();

  return (
    <>
      {children}
      <ChatWidget isAuthenticated={!!session && !isLoading} session={session} />
    </>
  );
}
```

### Login Page Component (frontend/src/pages/login.tsx)

```tsx
import { signIn } from "@/lib/auth";
import { useState } from "react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await signIn.email({ email, password });
      window.location.href = "/";
    } catch (err) {
      setError("Invalid email or password");
    }
  };

  return (
    <div className="login-container">
      <h1>Log In</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit" style={{ backgroundColor: "#10B981" }}>
          Log In
        </button>
        {error && <p className="error">{error}</p>}
      </form>
    </div>
  );
}
```

### Signup Page Component (frontend/src/pages/signup.tsx)

Similar structure to login page, but calls `signUp.email({ email, password })`.

### Logout Button in Navbar

Add logout button to Docusaurus navbar configuration:

```javascript
// docusaurus.config.js
navbar: {
  items: [
    // ... other items
    {
      type: 'custom',
      position: 'right',
      component: 'LogoutButton', // Custom component
    },
  ],
}
```

---

## Backend Integration Details

### Better Auth Server Setup (FastAPI)

**Installation**:
```bash
pip install better-auth-python
```

**Configuration** (backend/auth.py):
```python
from better_auth import Auth
import os

auth = Auth(
    secret=os.getenv("AUTH_SECRET"),
    database_url=os.getenv("DATABASE_URL"),
    session_expiration_days=7,
)

async def get_current_user(request: Request):
    """Dependency to validate session and get current user."""
    session = await auth.api.get_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session.user
```

**Protecting Routes**:
```python
from fastapi import Depends
from backend.auth import get_current_user

@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    user = Depends(get_current_user)
):
    # User is authenticated
    # Process chat request
    pass
```

---

## Script Specifications

### Database Initialization Script (backend/scripts/init_db.py)

**Purpose**: Create all database tables idempotently.

**Usage**:
```bash
cd backend
python scripts/init_db.py
```

**Behavior**:
1. Load DATABASE_URL from environment
2. Connect to Neon Postgres
3. Check if each table exists (using `information_schema.tables`)
4. If table doesn't exist, execute CREATE TABLE statement
5. Create indexes if they don't exist
6. Print summary: "Created tables: users, sessions, messages | Already existed: retrieval_logs, performance_metrics"
7. Exit with code 0 on success, 1 on error

**Error Handling**:
- Cannot connect to database: Print error, exit code 1
- SQL error during creation: Print error with table name, continue to next table
- Missing DATABASE_URL: Print "DATABASE_URL not found in environment", exit code 1

---

### Content Ingestion Script (backend/scripts/direct_ingest.py)

**Purpose**: Directly ingest all markdown files from frontend/docs/ into Qdrant (bypasses API for admin operations).

**Usage**:
```bash
cd backend
python scripts/direct_ingest.py
```

**Behavior**:
1. Load environment variables (QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY)
2. Initialize Qdrant collection if it doesn't exist
3. Recursively find all .md files in frontend/docs/
4. For each file:
   - Read file content and frontmatter
   - Extract chapter_id from path (e.g., "docs/chapter-1/section.md" → "chapter-1")
   - Extract section_id from frontmatter or filename
   - Generate anchor_url (e.g., "/docs/chapter-1") - removes "/index" for Docusaurus compatibility
   - Chunk markdown content using tiktoken
   - Generate embeddings via OpenAI API
   - Upsert chunks directly to Qdrant
   - Print progress: "Processing: chapter-1/intro.md ... ✅ Created 12 chunks"
5. Print summary: "Ingested 41 files, created 527 chunks, 3 errors"

**Error Handling**:
- File read error: Log error, continue to next file
- Embedding API fails: Retry with exponential backoff
- No files found: Print warning "No markdown files found in frontend/docs/", exit code 0

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can create an account and log in within 30 seconds on first attempt.
- **SC-002**: Authenticated users see the chat interface (not login prompt) 100% of the time when clicking the chat icon.
- **SC-003**: Session cookies persist for 7 days allowing users to stay logged in across browser sessions.
- **SC-004**: Database initialization script completes in under 10 seconds and is idempotent (can be run multiple times safely).
- **SC-005**: Content ingestion script processes 50+ markdown files and creates 500+ chunks within 5 minutes.
- **SC-006**: 100% of protected API endpoints return 401 for unauthenticated requests.
- **SC-007**: Zero API keys or secrets exposed in frontend code or network requests (verified by browser DevTools inspection).
- **SC-008**: Users can send a chat message and receive an AI-generated answer based on ingested content within 5 seconds.
- **SC-009**: Logout functionality terminates sessions and prevents access to protected resources 100% of the time.

---

## Assumptions

1. **Better Auth Compatibility**: Better Auth React client works with Docusaurus. Backend implements Better Auth-compatible API patterns.
2. **Database Access**: User has a Neon Serverless Postgres instance with valid connection string in DATABASE_URL.
3. **Existing Backend**: The FastAPI backend and /api/ingest endpoint are already implemented per specs/002-rag-chatbot/spec.md.
4. **Frontend Framework**: Docusaurus site uses React and can integrate React components for login/signup pages.
5. **Content Structure**: Markdown files in frontend/docs/ follow a consistent structure (chapters/sections) with frontmatter.
6. **Environment Configuration**: User has .env files configured in both backend/ and frontend/ directories.
7. **Custom Chat UI**: The frontend uses a custom React chat UI (not ChatKit) that integrates with the existing REST API backend.

---

## Out of Scope

- Social login (Google, GitHub OAuth)
- Password reset via email
- Email verification for new accounts
- Two-factor authentication (2FA)
- Admin dashboard for user management
- Rate limiting for login attempts
- Automatic content re-ingestion on file changes
- Real-time sync between multiple ingestion processes
- Content versioning or rollback
- Custom embedding models (using OpenAI text-embedding-3-small)

---

## Dependencies

- **External Services**: Neon Serverless Postgres (for user accounts and sessions), Qdrant Cloud v1.16.2 (for embeddings), OpenAI API (for embeddings)
- **Frontend Libraries**: Better Auth React client (`better-auth/react`), react-markdown, remark-gfm (for markdown rendering)
- **Backend Libraries**: FastAPI, asyncpg (for Postgres), qdrant-client v1.16.2, OpenAI Agents SDK, argon2-cffi (password hashing)
- **Scripts**: Python 3.11+, frontmatter parsing library, tiktoken (for token counting)
- **Environment**: DATABASE_URL (Neon connection string), OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, AUTH_SECRET (for session signing)

---

## Constraints

- **No Schema Modifications**: Must use existing database schema from specs/002-rag-chatbot/spec.md without changes (except adding password_hash to users table).
- **No Ingestion Endpoint Changes**: Must use existing /api/ingest endpoint without modifications.
- **Session Storage**: Better Auth must store sessions in Neon Postgres (not in-memory or Redis).
- **Theme Consistency**: Login/signup pages must use the same green (#10B981) accent color as the chatbot.
- **Environment Variables Only**: All credentials must be stored in .env files, never hardcoded.
- **Idempotent Scripts**: Both init_db.py and ingest_docs.py must be safe to run multiple times.
