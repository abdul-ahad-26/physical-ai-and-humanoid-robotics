# Implementation Plan: RAG Chatbot for Docusaurus Textbook

**Branch**: `002-rag-chatbot` | **Date**: 2025-12-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-rag-chatbot/spec.md`

---

## Summary

Build an embedded RAG chatbot for a Docusaurus textbook that answers user questions grounded exclusively in textbook content. The system uses:
- **Frontend**: OpenAI ChatKit (`@openai/chatkit-react`) floating widget with Better Auth session management
- **Backend**: FastAPI with OpenAI Agents SDK orchestrating multi-agent RAG workflow
- **Storage**: Qdrant Cloud for vector embeddings + Neon Serverless Postgres for relational data
- **Deployment**: Render (backend) + existing Docusaurus hosting (frontend)

---

## Technical Context

**Language/Version**: Python 3.11 (backend), TypeScript/JavaScript (frontend)
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, Qdrant Client, asyncpg, ChatKit, Better Auth
**Storage**: Neon Serverless Postgres (relational), Qdrant Cloud (vectors)
**Testing**: pytest, pytest-asyncio (backend); Jest (frontend)
**Target Platform**: Web (Docusaurus site)
**Project Type**: Web application (monorepo with frontend + backend)
**Performance Goals**: <5s average response time, 100 concurrent users
**Constraints**: Free tiers for Qdrant/Neon, gpt-4o-mini for cost efficiency
**Scale/Scope**: Single textbook, ~10k users initial

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Spec-Driven Development | PASS | Full spec at `spec.md`, plan follows SDD workflow |
| II. Technical Accuracy | PASS | All code patterns verified against official docs via Context7 |
| III. Modularity & Reusability | PASS | Agents are modular, API contracts defined |
| IV. Docusaurus-First Architecture | PASS | ChatKit integrates via Docusaurus theme |
| V. AI-Native Content Design | PASS | RAG system supports AI consumption of textbook |
| VI. Practical Application Focus | PASS | Working code examples in quickstart.md |

**Gate Result**: PASS - No violations requiring justification

---

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-chatbot/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research decisions
├── data-model.md        # Entity definitions and schemas
├── quickstart.md        # Setup guide
├── contracts/
│   └── openapi.yaml     # API contract
└── tasks.md             # (Created by /sp.tasks)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Environment configuration
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── retrieval.py     # Retrieval Agent
│   │   ├── answer.py        # Answer Generation Agent
│   │   ├── citation.py      # Citation Agent
│   │   ├── session.py       # Session & Logging Agent
│   │   └── orchestrator.py  # Multi-agent workflow
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py          # POST /api/chat
│   │   ├── sessions.py      # GET /api/sessions
│   │   ├── ingest.py        # POST /api/ingest
│   │   └── middleware.py    # Auth middleware
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py    # asyncpg pool
│   │   ├── models.py        # Pydantic models
│   │   └── queries.py       # SQL queries
│   └── services/
│       ├── __init__.py
│       ├── qdrant.py        # Qdrant client
│       ├── embeddings.py    # OpenAI embeddings
│       └── chunking.py      # Token-based chunking
├── tests/
│   ├── conftest.py
│   ├── test_chat.py
│   ├── test_agents.py
│   └── test_ingest.py
├── pyproject.toml
├── Dockerfile
└── .env.example

frontend/
├── src/
│   ├── components/
│   │   └── ChatWidget/
│   │       ├── index.tsx        # Main ChatKit wrapper
│   │       ├── SelectedText.tsx # Text selection capture
│   │       └── AuthGate.tsx     # Better Auth session check
│   └── theme/
│       └── Root.tsx             # Docusaurus root with ChatWidget
├── package.json
└── .env.example
```

**Structure Decision**: Web application pattern with separate `frontend/` and `backend/` directories. Frontend extends existing Docusaurus site; backend is standalone FastAPI service.

---

## Implementation Phases

### Phase 1: Infrastructure Setup

**Goal**: Establish monorepo structure and external service connections

| Task | Description | Dependencies |
|------|-------------|--------------|
| 1.1 | Create monorepo structure (`frontend/`, `backend/`) | None |
| 1.2 | Initialize FastAPI project with uv | 1.1 |
| 1.3 | Provision Neon Postgres and create schema | 1.2 |
| 1.4 | Provision Qdrant Cloud and create collection | 1.2 |
| 1.5 | Configure environment variables | 1.3, 1.4 |
| 1.6 | Set up asyncpg connection pool | 1.3 |
| 1.7 | Set up Qdrant async client | 1.4 |

**Acceptance Criteria**:
- [ ] `uv run uvicorn src.main:app` starts without errors
- [ ] Database connection pool initializes successfully
- [ ] Qdrant collection `textbook_chunks` exists
- [ ] Health endpoint returns 200

---

### Phase 2: Content Ingestion Pipeline

**Goal**: Parse, chunk, embed, and store textbook content

| Task | Description | Dependencies |
|------|-------------|--------------|
| 2.1 | Implement token-based chunking with tiktoken | 1.2 |
| 2.2 | Implement OpenAI embeddings service | 1.2 |
| 2.3 | Implement Qdrant upsert service | 1.7 |
| 2.4 | Create POST /api/ingest endpoint | 2.1, 2.2, 2.3 |
| 2.5 | Ingest sample textbook content | 2.4 |

**Acceptance Criteria**:
- [ ] Markdown content is chunked into 512-token segments with 50-token overlap
- [ ] Each chunk has metadata: chapter_id, section_id, anchor_url
- [ ] POST /api/ingest returns chunk count and vector IDs
- [ ] Qdrant collection contains ingested vectors

---

### Phase 3: OpenAI Agents SDK Implementation

**Goal**: Build multi-agent RAG workflow

| Task | Description | Dependencies |
|------|-------------|--------------|
| 3.1 | Implement Retrieval Agent with Qdrant search tool | 1.7 |
| 3.2 | Implement Answer Generation Agent with book-only constraint | 3.1 |
| 3.3 | Implement Citation Agent | 3.2 |
| 3.4 | Implement Session & Logging Agent | 1.6 |
| 3.5 | Create orchestrator with handoffs | 3.1, 3.2, 3.3, 3.4 |
| 3.6 | Add input guardrail (query validation) | 3.5 |
| 3.7 | Add output guardrail (hallucination check) | 3.5 |

**Agent Workflow**:
```
User Query → Input Guardrail → Retrieval Agent → Answer Agent → Citation Agent → Session Agent → Response
```

**Acceptance Criteria**:
- [ ] Retrieval Agent returns top 5 chunks above 0.7 similarity
- [ ] Answer Agent generates responses only from provided context
- [ ] "I don't know based on this book." returned when context insufficient
- [ ] Citation Agent maps answers to chapter/section anchors
- [ ] All metrics logged to performance_metrics table

---

### Phase 4: API Endpoints

**Goal**: Expose REST API for frontend consumption

| Task | Description | Dependencies |
|------|-------------|--------------|
| 4.1 | Implement Better Auth session validation middleware | 1.5 |
| 4.2 | Create POST /api/chat endpoint | 3.5, 4.1 |
| 4.3 | Create GET /api/sessions endpoint | 1.6, 4.1 |
| 4.4 | Create GET /api/sessions/{id}/messages endpoint | 1.6, 4.1 |
| 4.5 | Add rate limiting (10 req/min) | 4.2 |
| 4.6 | Configure CORS for frontend origins | 4.2 |

**Acceptance Criteria**:
- [ ] All endpoints require valid Better Auth session
- [ ] POST /api/chat returns answer with citations
- [ ] GET /api/sessions returns user's sessions (last 30 days)
- [ ] Rate limiting returns 429 when exceeded
- [ ] CORS allows configured frontend origins

---

### Phase 5: Frontend Integration

**Goal**: Embed ChatKit widget in Docusaurus

| Task | Description | Dependencies |
|------|-------------|--------------|
| 5.1 | Install @openai/chatkit-react and @better-auth/react | None |
| 5.2 | Create ChatWidget component with useChatKit | 5.1 |
| 5.3 | Configure ChatKit theme (green accent) | 5.2 |
| 5.4 | Implement Better Auth session check (show login prompt for unauthenticated) | 5.1 |
| 5.5 | Implement selected text capture | 5.2 |
| 5.6 | Add floating widget positioning (bottom-right) | 5.2 |
| 5.7 | Ensure keyboard accessibility | 5.6 |
| 5.8 | Add to Docusaurus Root theme | 5.6 |

**Acceptance Criteria**:
- [ ] ChatWidget icon visible to all users, login prompt shown for unauthenticated
- [ ] Green (#10B981) accent color applied
- [ ] Selected text captured and sent with queries
- [ ] Widget anchored to bottom-right
- [ ] Tab/Enter/Escape keyboard navigation works

---

### Phase 6: Testing & Validation

**Goal**: Verify all acceptance criteria from spec

| Task | Description | Dependencies |
|------|-------------|--------------|
| 6.1 | Unit tests for chunking service | 2.1 |
| 6.2 | Unit tests for each agent | 3.1-3.4 |
| 6.3 | Integration tests for /api/chat | 4.2 |
| 6.4 | Integration tests for session endpoints | 4.3, 4.4 |
| 6.5 | E2E test: book-grounded answer | 5.8 |
| 6.6 | E2E test: selected-text context | 5.5 |
| 6.7 | E2E test: no-answer fallback | 3.2 |
| 6.8 | E2E test: session persistence | 4.3 |
| 6.9 | E2E test: auth flow | 5.4 |

**Acceptance Criteria**:
- [ ] 95% of book-content questions get relevant answers
- [ ] 100% of answers include citations when content found
- [ ] "I don't know" returned for out-of-scope questions
- [ ] Session history persists for 30 days
- [ ] Chat widget shows login prompt for unauthenticated users

---

### Phase 7: Deployment

**Goal**: Deploy to production

| Task | Description | Dependencies |
|------|-------------|--------------|
| 7.1 | Create Dockerfile for FastAPI | 4.6 |
| 7.2 | Create render.yaml for Render deployment | 7.1 |
| 7.3 | Configure production environment variables | 7.2 |
| 7.4 | Deploy backend to Render | 7.3 |
| 7.5 | Configure frontend NEXT_PUBLIC_API_URL | 7.4 |
| 7.6 | Deploy frontend with Docusaurus | 7.5 |
| 7.7 | Verify production health checks | 7.6 |

**Acceptance Criteria**:
- [ ] Backend accessible at Render URL
- [ ] Frontend ChatWidget connects to production API
- [ ] <5s response time in production
- [ ] No API keys exposed in frontend

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Qdrant free tier limits | Monitor usage; upgrade if needed |
| Neon cold starts | Use pooled connection; warm with keep-alive |
| OpenAI rate limits | Implement retry with exponential backoff |
| Better Auth session validation latency | Cache session for short TTL |
| Hallucination in answers | Output guardrail + strict prompt engineering |

---

## Related Artifacts

- [spec.md](./spec.md) - Feature specification
- [research.md](./research.md) - Technology decisions
- [data-model.md](./data-model.md) - Entity definitions
- [quickstart.md](./quickstart.md) - Setup guide
- [contracts/openapi.yaml](./contracts/openapi.yaml) - API contract

---

## Next Steps

Run `/sp.tasks` to generate detailed task breakdown with test cases.
