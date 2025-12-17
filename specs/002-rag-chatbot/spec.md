# Feature Specification: RAG Chatbot for Docusaurus Textbook

**Feature Branch**: `002-rag-chatbot`
**Created**: 2025-12-17
**Status**: Draft
**Input**: User description: "Generate a complete specification for an embedded Retrieval-Augmented Generation (RAG) chatbot integrated into a Docusaurus textbook"

---

## Clarifications

### Session 2025-12-17

- Q: Frontend UI framework - React or ChatKit? → A: **OpenAI ChatKit** (`@openai/chatkit-react`) - batteries-included chat UI framework with built-in streaming, theming, and message components
- Q: Authentication provider? → A: **Better Auth** - TypeScript framework-agnostic authentication and authorization library with session management

---

## System Architecture Overview

The RAG Chatbot system consists of three primary components:

1. **Frontend**: A floating chat widget using **OpenAI ChatKit** (`@openai/chatkit-react`) embedded in the Docusaurus site
2. **Backend**: A FastAPI service orchestrating multi-agent workflows via OpenAI Agents SDK
3. **Storage Layer**: Qdrant Cloud for vector embeddings + Neon Serverless Postgres for relational data

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DOCUSAURUS FRONTEND                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              OpenAI ChatKit Widget (@openai/chatkit-react)       │    │
│  │  ┌─────────────┐  ┌──────────────────────────────────────────┐  │    │
│  │  │ Text Select │  │          Chat Interface                   │  │    │
│  │  │  Context    │──│  - Message bubbles (user/assistant)       │  │    │
│  │  └─────────────┘  │  - Input bar with send action             │  │    │
│  │                   │  - Citation links to book sections        │  │    │
│  │                   └──────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ HTTPS
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI BACKEND                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   OpenAI Agents SDK Orchestration                │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │  Retrieval   │  │   Answer     │  │  Citation    │           │    │
│  │  │    Agent     │──│  Generation  │──│    Agent     │           │    │
│  │  │              │  │    Agent     │  │              │           │    │
│  │  └──────┬───────┘  └──────────────┘  └──────────────┘           │    │
│  │         │                                                        │    │
│  │  ┌──────▼───────────────────────────────────────────────────┐   │    │
│  │  │              Session & Logging Agent                      │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────┬───────────────────┘
                                │                     │
                    ┌───────────▼───────────┐  ┌──────▼──────────────┐
                    │    QDRANT CLOUD       │  │  NEON POSTGRES      │
                    │  (Vector Embeddings)  │  │  (Relational Data)  │
                    │  - Book chunks        │  │  - Users            │
                    │  - Token-based splits │  │  - Sessions         │
                    │  - Similarity search  │  │  - Messages         │
                    └───────────────────────┘  │  - Retrieval logs   │
                                               │  - Metrics          │
                                               └─────────────────────┘
```

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask a Question About Book Content (Priority: P1)

As a logged-in reader studying the textbook, I want to ask questions about concepts covered in the book so that I can get explanations grounded in the textbook material with citations to relevant sections.

**Why this priority**: Core value proposition - users need to ask questions and receive book-grounded answers. Without this, the chatbot has no purpose.

**Independent Test**: Can be fully tested by a logged-in user typing a question related to book content and receiving an answer with chapter/section citations.

**Acceptance Scenarios**:

1. **Given** a logged-in user on any textbook page, **When** they type "What is reinforcement learning?" in the chat widget, **Then** the system returns an answer derived exclusively from book content with citations linking to relevant chapter/section anchors.

2. **Given** a logged-in user asks a question, **When** no relevant content exists in the book, **Then** the system responds with "I don't know based on this book."

3. **Given** a logged-in user asks a question, **When** the answer is generated, **Then** each citation is a clickable link that navigates to the exact book section.

---

### User Story 2 - Contextual Question from Selected Text (Priority: P2)

As a reader who has highlighted a confusing passage, I want to select text and ask a question about it so that my question is enriched with the selected context for more relevant answers.

**Why this priority**: Enhances the core Q&A experience by allowing users to ground questions in specific passages they're reading.

**Independent Test**: Can be fully tested by selecting text on a page, opening the chat, and asking a question that references the selection.

**Acceptance Scenarios**:

1. **Given** a user has selected text on the page, **When** they open the chat widget and type a question, **Then** the selected text is appended as additional context to the retrieval query.

2. **Given** selected text is provided, **When** the system generates an answer, **Then** the answer considers the selected text as context but is not limited to only that text.

3. **Given** no text is selected, **When** the user asks a question, **Then** the system processes the question without additional context.

---

### User Story 3 - View Conversation History (Priority: P3)

As a returning user, I want to see my previous conversations so that I can continue learning from where I left off.

**Why this priority**: Supports learning continuity and user engagement over multiple sessions.

**Independent Test**: Can be fully tested by logging in, having a conversation, logging out, logging back in, and viewing the previous conversation.

**Acceptance Scenarios**:

1. **Given** a logged-in user with previous chat sessions, **When** they open the chat widget, **Then** they see their most recent conversation history.

2. **Given** a user has multiple sessions, **When** they access history, **Then** sessions from the last 30 days are available.

3. **Given** a session is older than 30 days, **When** the user tries to access it, **Then** it is no longer available (automatically purged).

---

### User Story 4 - Content Ingestion by Administrator (Priority: P4)

As a content administrator, I want to ingest textbook content into the vector store so that the chatbot can answer questions about new or updated chapters.

**Why this priority**: Required for system operation but not user-facing; can be done manually initially.

**Independent Test**: Can be fully tested by calling the ingestion endpoint with markdown content and verifying it appears in search results.

**Acceptance Scenarios**:

1. **Given** markdown content for a chapter, **When** the ingestion endpoint is called, **Then** the content is chunked, embedded, and stored in Qdrant with chapter/section metadata.

2. **Given** content is ingested, **When** a user asks a relevant question, **Then** the newly ingested content is retrievable.

---

### Edge Cases

- **Empty Query**: User submits an empty or whitespace-only question; system prompts for a valid question.
- **Very Long Query**: Query exceeds token limits; system truncates or rejects with a helpful message.
- **Network Failure**: Backend is unreachable; frontend displays a retry option with a user-friendly error.
- **No Authentication**: User not logged in; chat widget is hidden or displays a login prompt.
- **Rate Limiting**: User sends too many requests; system returns a rate limit message.
- **Malformed Citations**: Retrieved content lacks proper metadata; system omits citation rather than showing broken links.
- **Concurrent Sessions**: User opens chat in multiple tabs; sessions are synchronized or handled gracefully.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Frontend Requirements (OpenAI ChatKit)

- **FR-001**: System MUST display an OpenAI ChatKit widget (`@openai/chatkit-react`) anchored to the bottom-right corner on all textbook pages.
- **FR-002**: System MUST render user messages and assistant messages with clear visual distinction (different colors/alignment).
- **FR-003**: System MUST provide a scrollable conversation area that auto-scrolls to the latest message.
- **FR-004**: System MUST include a persistent input bar with a send button/action.
- **FR-005**: System MUST use a color system with green as the primary accent, white background, and black text.
- **FR-006**: System MUST be responsive and work on mobile, tablet, and desktop viewports.
- **FR-007**: System MUST support keyboard navigation (Tab to focus, Enter to send).
- **FR-008**: System MUST meet WCAG 2.1 AA contrast requirements for accessibility.
- **FR-009**: System MUST capture user-selected text on the page and include it as additional context when the user submits a question.
- **FR-010**: System MUST show the chat widget icon to all users, but display a login prompt when unauthenticated users attempt to use it.

#### Backend Requirements

- **FR-011**: System MUST expose a chat query endpoint that accepts user questions and returns AI-generated answers.
- **FR-012**: System MUST use the OpenAI Agents SDK to orchestrate a multi-agent workflow.
- **FR-013**: System MUST implement a Retrieval Agent that queries Qdrant Cloud to find relevant book chunks.
- **FR-014**: System MUST implement an Answer Generation Agent that produces answers constrained strictly to retrieved book content.
- **FR-015**: System MUST implement a Citation Agent that maps answer segments to chapter/section anchors.
- **FR-016**: System MUST implement a Session & Logging Agent that persists conversation state and logs retrieval metrics.
- **FR-017**: System MUST respond with "I don't know based on this book." when no relevant content is retrieved (similarity threshold not met).
- **FR-018**: System MUST use a cost-effective OpenAI model suitable for RAG (gpt-4o-mini recommended based on current documentation).
- **FR-019**: System MUST expose an embeddings ingestion endpoint for administrators to add/update book content.
- **FR-020**: System MUST expose a session/history retrieval endpoint to fetch past conversations.

#### Storage Requirements

- **FR-021**: System MUST store vector embeddings in Qdrant Cloud (free tier compatible).
- **FR-022**: System MUST chunk book content using token-based splitting (target: 512 tokens with 50-token overlap).
- **FR-023**: System MUST store chunk metadata including chapter ID, section ID, anchor URL, and source file path.
- **FR-024**: System MUST store user accounts in Neon Serverless Postgres.
- **FR-025**: System MUST store chat sessions with a 30-day retention policy.
- **FR-026**: System MUST store individual messages linked to sessions.
- **FR-027**: System MUST log retrieval events including vector IDs returned, similarity scores, and query text.
- **FR-028**: System MUST log performance metrics including response latency, token usage, and model identifier.

#### Security & Access Requirements (Better Auth)

- **FR-029**: System MUST require user authentication via Better Auth to access chat functionality.
- **FR-030**: System MUST validate user sessions on each API request using Better Auth's `auth.api.getSession()`.
- **FR-031**: System MUST not expose API keys or secrets to the frontend.
- **FR-032**: System MUST integrate Better Auth client on the frontend for session management using the `useSession` hook.

---

### Key Entities

- **User**: A registered reader of the textbook. Attributes: ID, email, display name, created timestamp, last login.
- **Session**: A chat conversation instance. Attributes: ID, user ID, created timestamp, last activity timestamp, is active.
- **Message**: A single message in a conversation. Attributes: ID, session ID, role (user/assistant), content, created timestamp, citations (array).
- **Citation**: A reference to book content. Attributes: chapter ID, section ID, anchor URL, display text.
- **BookChunk**: A segment of textbook content stored as a vector. Attributes: vector ID, content text, chapter ID, section ID, anchor URL, token count.
- **RetrievalLog**: A record of a vector search operation. Attributes: ID, session ID, message ID, query text, vector IDs returned, similarity scores, timestamp.
- **PerformanceMetric**: A record of system performance. Attributes: ID, session ID, message ID, latency (ms), input tokens, output tokens, model ID, timestamp.

---

## API Contracts

### POST /api/chat

Submit a user query and receive an AI-generated response.

**Request**:
```json
{
  "session_id": "uuid-string (optional, creates new if omitted)",
  "query": "string (required, max 2000 characters)",
  "selected_text": "string (optional, user-selected page text)"
}
```

**Response (Success)**:
```json
{
  "session_id": "uuid-string",
  "message_id": "uuid-string",
  "answer": "string",
  "citations": [
    {
      "chapter_id": "string",
      "section_id": "string",
      "anchor_url": "/docs/chapter-1#section-name",
      "display_text": "Chapter 1: Introduction"
    }
  ],
  "created_at": "ISO-8601 timestamp"
}
```

**Response (No Content Found)**:
```json
{
  "session_id": "uuid-string",
  "message_id": "uuid-string",
  "answer": "I don't know based on this book.",
  "citations": [],
  "created_at": "ISO-8601 timestamp"
}
```

**Error Responses**:
- `401 Unauthorized`: User not authenticated
- `400 Bad Request`: Invalid query (empty, too long)
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Backend failure

---

### POST /api/ingest

Ingest textbook content into the vector store (admin only).

**Request**:
```json
{
  "content": "string (markdown content)",
  "chapter_id": "string",
  "section_id": "string",
  "anchor_url": "string"
}
```

**Response (Success)**:
```json
{
  "status": "success",
  "chunks_created": 15,
  "vector_ids": ["uuid-1", "uuid-2", "..."]
}
```

**Error Responses**:
- `401 Unauthorized`: Not authenticated
- `403 Forbidden`: Not an administrator
- `400 Bad Request`: Invalid content or metadata

---

### GET /api/sessions

Retrieve user's chat sessions.

**Response (Success)**:
```json
{
  "sessions": [
    {
      "session_id": "uuid-string",
      "created_at": "ISO-8601 timestamp",
      "last_activity": "ISO-8601 timestamp",
      "message_count": 12,
      "preview": "First message preview..."
    }
  ]
}
```

---

### GET /api/sessions/{session_id}/messages

Retrieve messages for a specific session.

**Response (Success)**:
```json
{
  "session_id": "uuid-string",
  "messages": [
    {
      "message_id": "uuid-string",
      "role": "user | assistant",
      "content": "string",
      "citations": [],
      "created_at": "ISO-8601 timestamp"
    }
  ]
}
```

---

## Agent Definitions and Workflows

### Agent Architecture (OpenAI Agents SDK)

The system uses the OpenAI Agents SDK with a multi-agent handoff pattern:

```python
# Conceptual agent structure (not implementation)
TriggerAgent
    ├── validates user input
    └── hands off to → RetrievalAgent
                           ├── queries Qdrant
                           └── hands off to → AnswerGenerationAgent
                                                  ├── generates answer from context
                                                  └── hands off to → CitationAgent
                                                                         └── maps to anchors
                                                                         └── hands off to → SessionLoggingAgent
                                                                                               └── persists & logs
```

### Agent Definitions

1. **Retrieval Agent**
   - **Purpose**: Query Qdrant Cloud for relevant book chunks
   - **Input**: User query, optional selected text context
   - **Output**: List of relevant chunks with similarity scores
   - **Behavior**: Combines query + selected text for embedding, searches Qdrant, filters by similarity threshold (0.7 minimum), returns top 5 chunks

2. **Answer Generation Agent**
   - **Purpose**: Generate answers strictly from retrieved content
   - **Input**: User query, retrieved chunks
   - **Output**: Natural language answer
   - **Behavior**: Uses system prompt enforcing "only answer from provided context", generates response, returns "I don't know based on this book." if context is insufficient
   - **Model**: gpt-4o-mini (cost-effective for RAG)

3. **Citation Agent**
   - **Purpose**: Map answer segments to source locations
   - **Input**: Generated answer, source chunks with metadata
   - **Output**: Answer with inline citations and citation objects
   - **Behavior**: Identifies which chunks contributed to answer, creates citation links with chapter/section anchors

4. **Session & Logging Agent**
   - **Purpose**: Persist state and capture metrics
   - **Input**: Complete response data, session context
   - **Output**: Confirmation of persistence
   - **Behavior**: Saves message to Neon, logs retrieval details (vector IDs, scores), logs performance metrics (latency, tokens)

### Guardrails

- **Input Guardrail**: Validates query is non-empty, within length limits, and from authenticated user
- **Output Guardrail**: Ensures response doesn't contain content not present in retrieved chunks (hallucination prevention)

---

## Database Schemas

### Neon Serverless Postgres

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

-- Session cleanup (30-day retention)
-- Implemented via scheduled job or application logic
```

---

## Vector Storage Design

### Qdrant Cloud Configuration

**Collection**: `textbook_chunks`

**Vector Configuration**:
- Dimension: 1536 (OpenAI text-embedding-3-small)
- Distance: Cosine
- On-disk: false (free tier)

**Payload Schema**:
```json
{
  "content": "string - the chunk text",
  "chapter_id": "string - chapter identifier",
  "section_id": "string - section identifier",
  "anchor_url": "string - URL path with anchor",
  "source_file": "string - original markdown file path",
  "token_count": "integer - tokens in this chunk"
}
```

**Chunking Strategy**:
- Method: Token-based splitting
- Target chunk size: 512 tokens
- Overlap: 50 tokens (for context continuity)
- Preserve: Sentence boundaries where possible

**Search Configuration**:
- Top-k: 5 results
- Similarity threshold: 0.7 minimum score
- With payload: true

---

## Frontend Integration Details

### OpenAI ChatKit Integration

The frontend uses **OpenAI ChatKit** (`@openai/chatkit-react`), a batteries-included framework for building AI-powered chat experiences with deep UI customization and built-in response streaming.

**Installation**:
```bash
npm install @openai/chatkit-react
```

**Component Structure**:
```
ChatKitWrapper/
├── useChatKit hook      # Initializes ChatKit control with API config
├── ChatKit component    # Main chat UI from @openai/chatkit-react
│   ├── Built-in header  # Configurable via options
│   ├── Message threads  # Streaming message display
│   ├── Composer         # Input bar with customizable placeholder
│   └── History panel    # Built-in session history (optional)
└── SelectedTextContext  # Custom wrapper capturing page text selection
```

**ChatKit Configuration**:
```javascript
const { control } = useChatKit({
  api: {
    url: '/api/chatkit',      // FastAPI backend endpoint
    domainKey: 'textbook-rag' // Application identifier
  },
  theme: {
    colorScheme: 'light',
    color: {
      accent: { primary: '#10B981' } // Green accent
    }
  },
  startScreen: {
    greeting: 'Ask me about this textbook!',
    prompts: [
      { label: 'Explain a concept', prompt: 'Explain...', icon: 'lightbulb' }
    ]
  },
  composer: {
    placeholder: 'Ask a question about the book...'
  },
  threadItemActions: {
    feedback: true,
    retry: true
  }
});
```

### Styling Guidelines

| Element | Color | Notes |
|---------|-------|-------|
| Primary accent | Green (#10B981) | Send button, links, active states |
| Background | White (#FFFFFF) | Chat window background |
| Text | Black (#000000) | Primary text color |
| User message bg | Light green (#D1FAE5) | User message bubbles |
| Assistant message bg | Light gray (#F3F4F6) | Assistant message bubbles |
| Border radius | 12px | Rounded message containers |

### Accessibility Requirements

- Focus indicators visible on all interactive elements
- ARIA labels for chat widget toggle
- ARIA live region for new messages
- Keyboard: Tab navigation, Enter to send, Escape to close
- Screen reader: Message role announcements

---

## Guardrails and Failure Behaviors

### Input Validation

| Condition | Behavior |
|-----------|----------|
| Empty query | Return error: "Please enter a question." |
| Query > 2000 chars | Return error: "Question is too long. Please shorten to under 2000 characters." |
| Unauthenticated | Return 401, hide chat widget |
| Rate limit (>10 req/min) | Return 429: "Too many requests. Please wait a moment." |

### Retrieval Failures

| Condition | Behavior |
|-----------|----------|
| No chunks above threshold | Return: "I don't know based on this book." |
| Qdrant unavailable | Return 503: "Search service temporarily unavailable. Please try again." |
| Timeout (>30s) | Return 504: "Request timed out. Please try again." |

### Generation Failures

| Condition | Behavior |
|-----------|----------|
| OpenAI API error | Return 502: "AI service temporarily unavailable." |
| Content filter triggered | Return: "I cannot answer this question." |
| Token limit exceeded | Truncate context, retry with fewer chunks |

### Data Persistence Failures

| Condition | Behavior |
|-----------|----------|
| Neon unavailable | Log error, return response without persistence (graceful degradation) |
| Session not found | Create new session transparently |

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users receive an answer to their question within 5 seconds on average.
- **SC-002**: 95% of questions about content present in the book receive a relevant answer (not "I don't know").
- **SC-003**: 100% of answers include at least one citation when book content is found.
- **SC-004**: System handles 100 concurrent users without degradation (response time <10s at p95).
- **SC-005**: Chat widget loads in under 1 second on standard broadband connections.
- **SC-006**: 90% of users can successfully complete a Q&A interaction on first attempt (measured by session completion).
- **SC-007**: Session history is accessible for 30 days with 99.9% reliability.
- **SC-008**: Zero exposure of API keys or user credentials in frontend code or network requests.

---

## Assumptions

1. **Authentication Provider**: **Better Auth** is the designated authentication framework. Better Auth is a TypeScript framework-agnostic authentication library providing session management, user accounts, and API route protection. The backend will use `auth.api.getSession()` to validate requests.
2. **Content Format**: Textbook content is in Markdown format with consistent heading structure (H1 for chapters, H2 for sections).
3. **Embedding Model**: OpenAI text-embedding-3-small is suitable; if costs are prohibitive, can switch to open-source alternatives.
4. **Free Tier Limits**: Qdrant Cloud free tier (1GB storage, 1M vectors) is sufficient for initial textbook content.
5. **Neon Free Tier**: Neon free tier (0.5 GB storage, 3 GB transfer) is sufficient for initial user base.

---

## Out of Scope
- Admin dashboard for content management
- Multi-language support
- Voice input/output
- Export conversation history
- Real-time collaborative features
- Custom model fine-tuning
- Analytics dashboard

---

## Dependencies

- **External Services**: OpenAI API (Agents SDK + Embeddings + ChatKit), Qdrant Cloud, Neon Serverless Postgres
- **Frontend**: Docusaurus site (existing), OpenAI ChatKit (`@openai/chatkit-react`), Better Auth client
- **Backend**: Python 3.11+, FastAPI, uv package manager, Better Auth (for session validation)
- **Authentication**: Better Auth (TypeScript framework-agnostic authentication library)
