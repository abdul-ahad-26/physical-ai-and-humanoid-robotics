# Data Model: RAG Chatbot for Docusaurus Textbook

**Feature Branch**: `002-rag-chatbot`
**Date**: 2025-12-17
**Status**: Complete

---

## Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│      User       │       │    Session      │       │    Message      │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │──1:N──│ id (PK)         │──1:N──│ id (PK)         │
│ email           │       │ user_id (FK)    │       │ session_id (FK) │
│ display_name    │       │ created_at      │       │ role            │
│ created_at      │       │ last_activity   │       │ content         │
│ last_login      │       │ is_active       │       │ citations       │
└─────────────────┘       └─────────────────┘       │ created_at      │
                                    │               └─────────────────┘
                                    │                        │
                          ┌────────┴────────┐               │
                          │                 │               │
               ┌──────────▼──────┐  ┌───────▼───────┐      │
               │  RetrievalLog   │  │ PerfMetric    │      │
               ├─────────────────┤  ├───────────────┤      │
               │ id (PK)         │  │ id (PK)       │      │
               │ session_id (FK) │  │ session_id(FK)│      │
               │ message_id (FK) │──│ message_id(FK)│──────┘
               │ query_text      │  │ latency_ms    │
               │ vector_ids      │  │ input_tokens  │
               │ scores          │  │ output_tokens │
               │ created_at      │  │ model_id      │
               └─────────────────┘  │ created_at    │
                                    └───────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Qdrant Vector Storage                            │
├─────────────────────────────────────────────────────────────────────┤
│  Collection: textbook_chunks                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ id (UUID)                                                    │   │
│  │ vector (1536 dimensions - text-embedding-3-small)           │   │
│  │ payload:                                                     │   │
│  │   - content (string)                                         │   │
│  │   - chapter_id (string)                                      │   │
│  │   - section_id (string)                                      │   │
│  │   - anchor_url (string)                                      │   │
│  │   - source_file (string)                                     │   │
│  │   - token_count (integer)                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Neon Postgres Entities

### 1. User

Represents a registered reader of the textbook. Managed by Better Auth.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT gen_random_uuid() | Unique identifier |
| `email` | VARCHAR(255) | UNIQUE, NOT NULL | User's email address |
| `display_name` | VARCHAR(100) | NULL | Display name for UI |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Account creation time |
| `last_login` | TIMESTAMPTZ | NULL | Last login timestamp |

**Validation Rules**:
- Email must be valid format (validated by Better Auth)
- Email uniqueness enforced at database level
- display_name max 100 characters

**Indexes**:
- Primary key on `id`
- Unique index on `email`

---

### 2. Session

Represents a chat conversation instance.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT gen_random_uuid() | Unique identifier |
| `user_id` | UUID | FK → users(id), NOT NULL | Owner of session |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Session start time |
| `last_activity` | TIMESTAMPTZ | DEFAULT NOW() | Last message time |
| `is_active` | BOOLEAN | DEFAULT TRUE | Session active flag |

**Validation Rules**:
- user_id must reference valid user
- Sessions older than 30 days auto-purged (via scheduled job)

**Indexes**:
- Primary key on `id`
- Index on `user_id` for user session lookups
- Index on `last_activity` for cleanup queries

**State Transitions**:
```
Created (is_active=true)
    → Active (messages being added)
    → Inactive (is_active=false, no activity for 30 days)
    → Deleted (purged from database)
```

---

### 3. Message

Represents a single message in a conversation.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT gen_random_uuid() | Unique identifier |
| `session_id` | UUID | FK → sessions(id), NOT NULL | Parent session |
| `role` | VARCHAR(20) | CHECK (role IN ('user', 'assistant')), NOT NULL | Message sender |
| `content` | TEXT | NOT NULL | Message content |
| `citations` | JSONB | DEFAULT '[]' | Array of citation objects |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Message timestamp |

**Validation Rules**:
- role must be 'user' or 'assistant'
- content cannot be empty
- citations must be valid JSON array

**Citations JSONB Schema**:
```json
[
  {
    "chapter_id": "string",
    "section_id": "string",
    "anchor_url": "/docs/chapter-1#section-name",
    "display_text": "Chapter 1: Introduction"
  }
]
```

**Indexes**:
- Primary key on `id`
- Index on `session_id` for message retrieval
- Index on `created_at` for ordering

---

### 4. RetrievalLog

Logs vector search operations for debugging and analytics.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT gen_random_uuid() | Unique identifier |
| `session_id` | UUID | FK → sessions(id), NOT NULL | Associated session |
| `message_id` | UUID | FK → messages(id), NULL | Associated message |
| `query_text` | TEXT | NOT NULL | Search query |
| `vector_ids` | UUID[] | NOT NULL | Returned vector IDs |
| `similarity_scores` | FLOAT[] | NOT NULL | Similarity scores |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Log timestamp |

**Validation Rules**:
- vector_ids and similarity_scores arrays should have same length
- Scores should be between 0.0 and 1.0

**Indexes**:
- Primary key on `id`
- Index on `session_id` for session analytics

---

### 5. PerformanceMetric

Captures system performance data for monitoring.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT gen_random_uuid() | Unique identifier |
| `session_id` | UUID | FK → sessions(id), NOT NULL | Associated session |
| `message_id` | UUID | FK → messages(id), NULL | Associated message |
| `latency_ms` | INTEGER | NOT NULL | Response latency in ms |
| `input_tokens` | INTEGER | NOT NULL | Tokens in prompt |
| `output_tokens` | INTEGER | NOT NULL | Tokens in response |
| `model_id` | VARCHAR(100) | NOT NULL | Model identifier |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Metric timestamp |

**Validation Rules**:
- latency_ms must be positive
- Token counts must be non-negative
- model_id follows OpenAI naming convention

**Indexes**:
- Primary key on `id`
- Index on `session_id` for session analytics
- Index on `created_at` for time-series queries

---

## Qdrant Vector Entity

### BookChunk

Represents a segment of textbook content stored as a vector.

| Field | Type | Location | Description |
|-------|------|----------|-------------|
| `id` | UUID | Point ID | Unique chunk identifier |
| `vector` | float[1536] | Vector | Embedding from text-embedding-3-small |
| `content` | string | Payload | Original chunk text |
| `chapter_id` | string | Payload | Chapter identifier (e.g., "chapter-1") |
| `section_id` | string | Payload | Section identifier (e.g., "intro") |
| `anchor_url` | string | Payload | URL path with anchor (e.g., "/docs/chapter-1#intro") |
| `source_file` | string | Payload | Original markdown file path |
| `token_count` | integer | Payload | Number of tokens in chunk |

**Collection Configuration**:
```python
{
    "collection_name": "textbook_chunks",
    "vectors_config": {
        "size": 1536,
        "distance": "Cosine"
    }
}
```

**Search Parameters**:
- Top-k: 5 results
- Similarity threshold: 0.7
- With payload: true

---

## SQL Schema

```sql
-- Users table (managed by Better Auth, extended for app)
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

-- Indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_last_activity ON sessions(last_activity);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_retrieval_logs_session_id ON retrieval_logs(session_id);
CREATE INDEX idx_performance_metrics_session_id ON performance_metrics(session_id);
CREATE INDEX idx_performance_metrics_created_at ON performance_metrics(created_at);
```

---

## Data Retention Policy

| Entity | Retention | Cleanup Method |
|--------|-----------|----------------|
| Users | Indefinite | Manual deletion only |
| Sessions | 30 days | Scheduled job |
| Messages | Cascades with session | FK constraint |
| RetrievalLog | Cascades with session | FK constraint |
| PerformanceMetric | Cascades with session | FK constraint |
| BookChunk (Qdrant) | Indefinite | Manual via ingestion API |

**Cleanup Query**:
```sql
-- Delete sessions older than 30 days
DELETE FROM sessions
WHERE last_activity < NOW() - INTERVAL '30 days';
```
