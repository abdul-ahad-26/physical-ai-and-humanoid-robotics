# Data Model: Better Auth Authentication & Infrastructure

**Feature**: 003-auth-infrastructure
**Date**: 2025-12-17
**Status**: ✅ IMPLEMENTED (2025-12-18)
**Purpose**: Define all entities, relationships, and validation rules

**Note**: This data model was implemented as specified. No changes to database schema were required during implementation.

---

## Entity Relationship Diagram

```
┌─────────────────┐
│     User        │
│  (users table)  │
└────────┬────────┘
         │ 1
         │
         │ n
┌────────┴────────────────┐
│   AuthSession           │
│ (auth_sessions table)   │
└─────────────────────────┘

┌─────────────────┐
│     User        │
│  (users table)  │
└────────┬────────┘
         │ 1
         │
         │ n
┌────────┴────────────────┐
│   ChatSession           │
│  (sessions table)       │
└────────┬────────────────┘
         │ 1
         │
         │ n
┌────────┴────────────────┐
│     Message             │
│  (messages table)       │
└─────────────────────────┘

┌─────────────────────────┐
│     ChatSession         │
│  (sessions table)       │
└────────┬────────────────┘
         │ 1
         │
         │ n
┌────────┴────────────────┐
│   RetrievalLog          │
│ (retrieval_logs table)  │
└─────────────────────────┘

┌─────────────────────────┐
│     ChatSession         │
│  (sessions table)       │
└────────┬────────────────┘
         │ 1
         │
         │ n
┌────────┴────────────────┐
│  PerformanceMetric      │
│(performance_metrics)    │
└─────────────────────────┘
```

---

## Entity Definitions

### 1. User

**Purpose**: Represents a registered user account in the system.

**Table**: `users`

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique user identifier |
| email | VARCHAR(255) | UNIQUE NOT NULL | User email address |
| password_hash | VARCHAR(255) | NOT NULL | Argon2 hashed password |
| display_name | VARCHAR(100) | NULLABLE | Optional display name |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Account creation timestamp |
| last_login | TIMESTAMPTZ | NULLABLE | Last successful login |

**Validation Rules**:
- `email`: Must match email regex pattern: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
- `password_hash`: Generated from password meeting minimum 8 characters
- `display_name`: If provided, must be 1-100 characters

**Relationships**:
- One-to-many with `AuthSession` (user can have multiple active sessions)
- One-to-many with `ChatSession` (user can have multiple chat sessions)

**State Transitions**:
```
[New User] --signup--> [Active User]
[Active User] --login--> [Authenticated User]
[Authenticated User] --logout--> [Active User]
```

**Indexes**:
- `UNIQUE INDEX` on `email` (implicit from UNIQUE constraint)

---

### 2. AuthSession

**Purpose**: Represents an authenticated session for Better Auth.

**Table**: `auth_sessions` (created by Better Auth)

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique session identifier |
| user_id | UUID | FOREIGN KEY → users(id) ON DELETE CASCADE | User owning this session |
| session_token | VARCHAR(255) | UNIQUE NOT NULL | Secure random session token |
| expires_at | TIMESTAMPTZ | NOT NULL | Session expiration time |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Session creation time |

**Validation Rules**:
- `session_token`: 64+ character secure random string
- `expires_at`: Must be in the future at creation time (7 days from creation)
- `user_id`: Must reference existing user

**Relationships**:
- Many-to-one with `User` (session belongs to one user)

**State Transitions**:
```
[No Session] --login/signup--> [Active Session]
[Active Session] --logout--> [Terminated Session]
[Active Session] --timeout--> [Expired Session]
```

**Indexes**:
- `UNIQUE INDEX idx_auth_sessions_token` on `session_token`
- `INDEX idx_auth_sessions_user_id` on `user_id`

---

### 3. ChatSession

**Purpose**: Represents a conversation session between user and AI chatbot.

**Table**: `sessions`

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique chat session identifier |
| user_id | UUID | FOREIGN KEY → users(id) ON DELETE CASCADE | User owning this session |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Session creation time |
| last_activity | TIMESTAMPTZ | DEFAULT NOW() | Last message timestamp |
| is_active | BOOLEAN | DEFAULT TRUE | Whether session is active |

**Validation Rules**:
- `user_id`: Must reference existing user
- `last_activity`: Must be >= `created_at`
- `is_active`: Automatically set to FALSE if `last_activity` > 30 days ago

**Relationships**:
- Many-to-one with `User` (session belongs to one user)
- One-to-many with `Message` (session contains multiple messages)
- One-to-many with `RetrievalLog` (session generates retrieval events)
- One-to-many with `PerformanceMetric` (session generates performance metrics)

**State Transitions**:
```
[No Session] --first-message--> [Active Session]
[Active Session] --message--> [Active Session] (updates last_activity)
[Active Session] --30-days--> [Inactive Session] (is_active = FALSE)
```

**Indexes**:
- `INDEX idx_sessions_user_id` on `user_id`
- `INDEX idx_sessions_last_activity` on `last_activity`

---

### 4. Message

**Purpose**: Represents a single message in a chat conversation (user or assistant).

**Table**: `messages`

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique message identifier |
| session_id | UUID | FOREIGN KEY → sessions(id) ON DELETE CASCADE | Parent chat session |
| role | VARCHAR(20) | CHECK (role IN ('user', 'assistant')) | Message sender role |
| content | TEXT | NOT NULL | Message text content |
| citations | JSONB | DEFAULT '[]' | Citations for assistant messages |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Message creation time |

**Validation Rules**:
- `session_id`: Must reference existing chat session
- `role`: Must be either 'user' or 'assistant'
- `content`: Must be 1-10000 characters
- `citations`: Must be valid JSON array of citation objects

**Citation Schema** (JSONB):
```json
[
  {
    "chapter_id": "string",
    "section_id": "string",
    "anchor_url": "string",
    "display_text": "string"
  }
]
```

**Relationships**:
- Many-to-one with `ChatSession` (message belongs to one session)
- One-to-one with `RetrievalLog` (optional, for assistant messages)
- One-to-one with `PerformanceMetric` (optional, for assistant messages)

**State Transitions**:
```
[Draft] --send--> [Sent] (for user messages)
[Sent] --process--> [Processing] (for assistant responses)
[Processing] --complete--> [Delivered] (assistant message saved)
```

**Indexes**:
- `INDEX idx_messages_session_id` on `session_id`
- `INDEX idx_messages_created_at` on `created_at`

---

### 5. RetrievalLog

**Purpose**: Records vector search operations for debugging and analytics.

**Table**: `retrieval_logs`

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique log identifier |
| session_id | UUID | FOREIGN KEY → sessions(id) ON DELETE CASCADE | Parent chat session |
| message_id | UUID | FOREIGN KEY → messages(id) ON DELETE SET NULL | Related message (nullable) |
| query_text | TEXT | NOT NULL | User query that triggered retrieval |
| vector_ids | UUID[] | NOT NULL | Array of Qdrant vector IDs returned |
| similarity_scores | FLOAT[] | NOT NULL | Corresponding similarity scores |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Log creation time |

**Validation Rules**:
- `session_id`: Must reference existing chat session
- `message_id`: Must reference existing message (if provided)
- `query_text`: Must be 1-2000 characters
- `vector_ids`: Array length must match `similarity_scores` length
- `similarity_scores`: Each score must be between 0.0 and 1.0

**Relationships**:
- Many-to-one with `ChatSession` (log belongs to one session)
- One-to-one with `Message` (optional link to assistant message)

**Indexes**:
- `INDEX idx_retrieval_logs_session_id` on `session_id`

---

### 6. PerformanceMetric

**Purpose**: Records performance metrics for AI response generation.

**Table**: `performance_metrics`

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() | Unique metric identifier |
| session_id | UUID | FOREIGN KEY → sessions(id) ON DELETE CASCADE | Parent chat session |
| message_id | UUID | FOREIGN KEY → messages(id) ON DELETE SET NULL | Related message (nullable) |
| latency_ms | INTEGER | NOT NULL | Total response latency in ms |
| input_tokens | INTEGER | NOT NULL | Token count for input |
| output_tokens | INTEGER | NOT NULL | Token count for output |
| model_id | VARCHAR(100) | NOT NULL | OpenAI model identifier |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Metric creation time |

**Validation Rules**:
- `session_id`: Must reference existing chat session
- `message_id`: Must reference existing message (if provided)
- `latency_ms`: Must be > 0
- `input_tokens`: Must be >= 0
- `output_tokens`: Must be >= 0
- `model_id`: Must match pattern `gpt-*` or `o1-*`

**Relationships**:
- Many-to-one with `ChatSession` (metric belongs to one session)
- One-to-one with `Message` (optional link to assistant message)

**Indexes**:
- `INDEX idx_performance_metrics_session_id` on `session_id`

---

## Non-Relational Data

### Qdrant Vector Store

**Collection**: `textbook_chunks`

**Vector Configuration**:
- **Dimension**: 1536 (OpenAI text-embedding-3-small)
- **Distance**: Cosine similarity
- **Indexed**: Yes (HNSW algorithm)

**Payload Schema**:
```json
{
  "id": "uuid-string",
  "content": "string - the chunk text",
  "chapter_id": "string - chapter identifier",
  "section_id": "string - section identifier",
  "anchor_url": "string - URL path with anchor",
  "source_file": "string - original markdown file path",
  "token_count": "integer - tokens in this chunk"
}
```

**Validation Rules**:
- `content`: Must be 1-4096 characters (embedding model limit)
- `chapter_id`: Must match pattern `[a-z0-9-]+`
- `section_id`: Must match pattern `[a-z0-9-]+`
- `anchor_url`: Must start with `/docs/`
- `token_count`: Must be > 0

**Search Parameters**:
- Top-k: 5 results
- Minimum similarity threshold: 0.7
- Include payload: true

---

## Data Integrity Rules

### Foreign Key Cascades

1. **User deletion**:
   - Cascade deletes all `AuthSession` records
   - Cascade deletes all `ChatSession` records
   - Cascade deletes all `Message` records (via sessions)
   - Cascade deletes all `RetrievalLog` records (via sessions)
   - Cascade deletes all `PerformanceMetric` records (via sessions)

2. **ChatSession deletion**:
   - Cascade deletes all `Message` records
   - Cascade deletes all `RetrievalLog` records
   - Cascade deletes all `PerformanceMetric` records

3. **Message deletion**:
   - Sets `message_id` to NULL in `RetrievalLog` records
   - Sets `message_id` to NULL in `PerformanceMetric` records

### Data Retention

1. **ChatSessions**: Automatically marked inactive after 30 days of inactivity
2. **Messages**: Retained indefinitely (linked to user account)
3. **AuthSessions**: Automatically expired after 7 days
4. **Logs/Metrics**: Retained for 90 days (optional cleanup job)

### Constraints

1. **Email uniqueness**: Enforced at database level with UNIQUE constraint
2. **Session token uniqueness**: Enforced at database level with UNIQUE constraint
3. **Message role validation**: Enforced with CHECK constraint
4. **Timestamp ordering**: `last_activity` >= `created_at` for sessions

---

## Data Access Patterns

### Authentication Flow

1. **Signup**:
   - INSERT into `users` (email, password_hash)
   - INSERT into `auth_sessions` (user_id, session_token, expires_at)
   - RETURN user + session

2. **Login**:
   - SELECT from `users` WHERE email = ?
   - VERIFY password_hash
   - INSERT into `auth_sessions` (user_id, session_token, expires_at)
   - UPDATE `users` SET last_login = NOW()
   - RETURN user + session

3. **Session Validation**:
   - SELECT from `auth_sessions` JOIN `users` WHERE session_token = ? AND expires_at > NOW()
   - RETURN user

4. **Logout**:
   - DELETE from `auth_sessions` WHERE session_token = ?

### Chat Flow

1. **Send Message**:
   - VALIDATE session (see authentication flow)
   - SELECT or INSERT `sessions` (get or create chat session)
   - INSERT into `messages` (session_id, role='user', content)
   - UPDATE `sessions` SET last_activity = NOW()

2. **Generate Response**:
   - QUERY Qdrant for relevant chunks
   - INSERT into `retrieval_logs` (session_id, message_id, query_text, vector_ids, scores)
   - CALL OpenAI API
   - INSERT into `messages` (session_id, role='assistant', content, citations)
   - INSERT into `performance_metrics` (session_id, message_id, latency_ms, tokens, model_id)
   - RETURN message

3. **Load History**:
   - SELECT from `sessions` WHERE user_id = ? ORDER BY last_activity DESC
   - SELECT from `messages` WHERE session_id = ? ORDER BY created_at ASC

---

## Migration Strategy

### Phase 1: Initial Setup (init_db.py script)

1. Create `users` table (if not exists)
2. Add `password_hash` column to `users` (if not exists)
3. Create `sessions` table (if not exists)
4. Create `messages` table (if not exists)
5. Create `retrieval_logs` table (if not exists)
6. Create `performance_metrics` table (if not exists)
7. Create all indexes (if not exist)

### Phase 2: Better Auth Setup (frontend initialization)

1. Better Auth client creates `auth_sessions` table automatically
2. Better Auth creates indexes automatically

### Phase 3: Content Ingestion (ingest_docs.py script)

1. Create Qdrant collection `textbook_chunks` (if not exists)
2. Upload vectors with payloads

---

## Summary

**Total Entities**: 6 relational + 1 non-relational
- `User`: User accounts
- `AuthSession`: Authentication sessions
- `ChatSession`: Conversation sessions
- `Message`: Chat messages
- `RetrievalLog`: Vector search logs
- `PerformanceMetric`: Performance tracking
- `Qdrant Vector Store`: Embedded textbook content

**Key Relationships**:
- User → AuthSession (1:n)
- User → ChatSession (1:n)
- ChatSession → Message (1:n)
- ChatSession → RetrievalLog (1:n)
- ChatSession → PerformanceMetric (1:n)

**Data Integrity**: Foreign key cascades, unique constraints, check constraints
**Performance**: Indexes on all foreign keys and frequently queried fields
