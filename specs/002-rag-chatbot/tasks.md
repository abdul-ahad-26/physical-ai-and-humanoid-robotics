# Tasks: RAG Chatbot for Docusaurus Textbook

**Input**: Design documents from `/specs/002-rag-chatbot/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml, quickstart.md
**Branch**: `002-rag-chatbot`
**Date**: 2025-12-17

**Tests**: Test tasks are included for validation of key components.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Exact file paths included in descriptions

## Path Conventions

This is a **web app** with separate frontend and backend:
- **Backend**: `backend/src/`, `backend/tests/`
- **Frontend**: `frontend/src/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create monorepo structure with `backend/` and `frontend/` directories at repository root
- [X] T002 Initialize FastAPI project with uv in `backend/` per quickstart.md
- [X] T003 [P] Create `backend/pyproject.toml` with dependencies: fastapi, uvicorn[standard], asyncpg, qdrant-client, openai, openai-agents, tiktoken, httpx, python-dotenv, pydantic
- [X] T004 [P] Create `backend/.env.example` with environment variables: DATABASE_URL, QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY, BETTER_AUTH_URL, CORS_ORIGINS
- [X] T005 [P] Add dev dependencies to `backend/pyproject.toml`: pytest, pytest-asyncio, httpx
- [X] T006 Create `backend/src/__init__.py` and module structure: agents/, api/, db/, services/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

### Database Foundation

- [X] T007 Create database connection module in `backend/src/db/connection.py` with asyncpg pool initialization per research.md
- [X] T008 Create Pydantic models in `backend/src/db/models.py` for User, Session, Message, RetrievalLog, PerformanceMetric per data-model.md
- [X] T009 Create SQL query functions in `backend/src/db/queries.py` for CRUD operations on all entities

### Vector Storage Foundation

- [X] T010 Create Qdrant client module in `backend/src/services/qdrant.py` with AsyncQdrantClient initialization per research.md
- [X] T011 Create embeddings service in `backend/src/services/embeddings.py` using OpenAI text-embedding-3-small

### Authentication Foundation

- [X] T012 Create Better Auth middleware in `backend/src/api/middleware.py` with session validation via auth.api.getSession() per research.md
- [X] T013 Create config module in `backend/src/config.py` with environment variable loading using python-dotenv

### API Foundation

- [X] T014 Create FastAPI app entry point in `backend/src/main.py` with CORS, middleware, and router registration
- [X] T015 Create health endpoint in `backend/src/api/health.py` returning status and timestamp per contracts/openapi.yaml

### Initialize External Services

- [X] T016 Create database schema initialization script (can be run via quickstart.md SQL) to create all tables per data-model.md
- [X] T017 Create Qdrant collection initialization script for `textbook_chunks` collection with 1536-dim cosine vectors

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Ask a Question About Book Content (Priority: P1) MVP

**Goal**: Enable logged-in readers to ask questions about textbook content and receive answers grounded in book material with citations

**Independent Test**: A logged-in user can type a question, receive an AI-generated answer from book content, and see clickable citations to chapter/section anchors

### Tests for User Story 1

- [X] T018 [P] [US1] Unit test for Retrieval Agent in `backend/tests/test_agents.py::test_retrieval_agent`
- [X] T019 [P] [US1] Unit test for Answer Agent in `backend/tests/test_agents.py::test_answer_agent`
- [X] T020 [P] [US1] Unit test for Citation Agent in `backend/tests/test_agents.py::test_citation_agent`
- [X] T021 [P] [US1] Integration test for POST /api/chat in `backend/tests/test_chat.py::test_chat_endpoint`
- [X] T022 [P] [US1] Integration test for "I don't know" response in `backend/tests/test_chat.py::test_no_content_found`

### Implementation for User Story 1

#### Backend Agents (OpenAI Agents SDK)

- [X] T023 [P] [US1] Implement Retrieval Agent in `backend/src/agents/retrieval.py` with Qdrant search tool, top-5 results, 0.7 similarity threshold per plan.md
- [X] T024 [P] [US1] Implement Answer Generation Agent in `backend/src/agents/answer.py` with book-only constraint system prompt, gpt-4o-mini model per plan.md
- [X] T025 [P] [US1] Implement Citation Agent in `backend/src/agents/citation.py` mapping answer segments to chapter/section anchors per plan.md
- [X] T026 [P] [US1] Implement Session & Logging Agent in `backend/src/agents/session.py` for message persistence and metrics logging per plan.md
- [X] T027 [US1] Create orchestrator with multi-agent handoffs in `backend/src/agents/orchestrator.py` per plan.md workflow diagram
- [X] T028 [US1] Add input guardrail (query validation) in `backend/src/agents/orchestrator.py` per spec.md guardrails
- [X] T029 [US1] Add output guardrail (hallucination check) in `backend/src/agents/orchestrator.py` per spec.md guardrails

#### Backend API

- [X] T030 [US1] Implement POST /api/chat endpoint in `backend/src/api/chat.py` per contracts/openapi.yaml ChatRequest/ChatResponse schemas
- [X] T031 [US1] Add rate limiting (10 req/min) to chat endpoint in `backend/src/api/chat.py` per spec.md FR-029

#### Frontend ChatWidget

- [X] T032 [P] [US1] Install @openai/chatkit-react and @better-auth/react in `frontend/package.json`
- [X] T033 [US1] Create ChatWidget component in `frontend/src/components/ChatWidget/index.tsx` with useChatKit hook per quickstart.md
- [X] T034 [US1] Configure ChatKit theme (green #10B981 accent, light scheme) in `frontend/src/components/ChatWidget/index.tsx` per spec.md styling
- [X] T035 [US1] Implement Better Auth session check (AuthGate) in `frontend/src/components/ChatWidget/AuthGate.tsx` to show login prompt for unauthenticated users per FR-010
- [X] T073 [US1] Update ChatWidget to show icon to all users but display login prompt when unauthenticated user clicks it per FR-010
- [X] T036 [US1] Add floating widget positioning (bottom-right) in `frontend/src/components/ChatWidget/index.tsx` per plan.md
- [X] T037 [US1] Ensure keyboard accessibility (Tab/Enter/Escape) in `frontend/src/components/ChatWidget/index.tsx` per spec.md FR-007
- [X] T038 [US1] Add ChatWidget to Docusaurus Root theme in `frontend/src/theme/Root.tsx` per quickstart.md

**Checkpoint**: User Story 1 (MVP) complete - users can ask questions and receive book-grounded answers with citations

---

## Phase 4: User Story 2 - Contextual Question from Selected Text (Priority: P2)

**Goal**: Enable readers to select text on the page and ask questions enriched with that context

**Independent Test**: User selects text on a page, opens chat, asks a question, and the answer references the selected text context

### Tests for User Story 2

- [X] T039 [P] [US2] Integration test for selected text context in `backend/tests/test_chat.py::test_selected_text_context`
- [X] T040 [P] [US2] Unit test for selected text capture in `frontend/` (Jest)

### Implementation for User Story 2

#### Backend Enhancement

- [X] T041 [US2] Update Retrieval Agent in `backend/src/agents/retrieval.py` to combine selected_text with query for enhanced retrieval

#### Frontend Selected Text Capture

- [X] T042 [US2] Create SelectedText component in `frontend/src/components/ChatWidget/SelectedText.tsx` to capture user text selection on page
- [X] T043 [US2] Integrate SelectedText capture with ChatWidget in `frontend/src/components/ChatWidget/index.tsx` to include selected_text in chat requests
- [X] T044 [US2] Add visual indicator when text is selected in `frontend/src/components/ChatWidget/index.tsx`

**Checkpoint**: User Story 2 complete - users can ask questions with selected text context

---

## Phase 5: User Story 3 - View Conversation History (Priority: P3)

**Goal**: Enable returning users to see their previous conversations and continue learning

**Independent Test**: User logs in, has a conversation, logs out, logs back in, and can view/continue previous conversation

### Tests for User Story 3

- [X] T045 [P] [US3] Integration test for GET /api/sessions in `backend/tests/test_chat.py::test_list_sessions`
- [X] T046 [P] [US3] Integration test for GET /api/sessions/{id}/messages in `backend/tests/test_chat.py::test_get_session_messages`

### Implementation for User Story 3

#### Backend Session Endpoints

- [X] T047 [US3] Implement GET /api/sessions endpoint in `backend/src/api/sessions.py` per contracts/openapi.yaml SessionListResponse schema
- [X] T048 [US3] Implement GET /api/sessions/{id}/messages endpoint in `backend/src/api/sessions.py` per contracts/openapi.yaml MessagesResponse schema
- [X] T049 [US3] Add session ownership validation in `backend/src/api/sessions.py` to ensure users only see their own sessions
- [X] T050 [US3] Implement 30-day session retention logic in `backend/src/db/queries.py` per data-model.md retention policy

#### Frontend History Display

- [X] T051 [US3] Add session history panel to ChatWidget in `frontend/src/components/ChatWidget/index.tsx` using ChatKit history feature
- [X] T052 [US3] Implement session switching UI in `frontend/src/components/ChatWidget/index.tsx`

**Checkpoint**: User Story 3 complete - users can view and continue past conversations

---

## Phase 6: User Story 4 - Content Ingestion by Administrator (Priority: P4)

**Goal**: Enable administrators to ingest textbook content into the vector store

**Independent Test**: Admin calls ingestion endpoint with markdown content, then a user question retrieves that content

### Tests for User Story 4

- [X] T053 [P] [US4] Unit test for chunking service in `backend/tests/test_ingest.py::test_chunking_service`
- [X] T054 [P] [US4] Integration test for POST /api/ingest in `backend/tests/test_ingest.py::test_ingest_endpoint`

### Implementation for User Story 4

#### Backend Chunking Service

- [X] T055 [US4] Implement token-based chunking in `backend/src/services/chunking.py` with tiktoken, 512 tokens, 50 token overlap per spec.md FR-022

#### Backend Ingestion Endpoint

- [X] T056 [US4] Implement POST /api/ingest endpoint in `backend/src/api/ingest.py` per contracts/openapi.yaml IngestRequest/IngestResponse schemas
- [X] T057 [US4] Add admin-only authorization check in `backend/src/api/ingest.py` per contracts/openapi.yaml 403 response
- [X] T058 [US4] Implement chunk embedding and Qdrant upsert in `backend/src/api/ingest.py` using services/embeddings.py and services/qdrant.py

**Checkpoint**: User Story 4 complete - administrators can ingest new content

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Deployment

- [X] T059 [P] Create Dockerfile in `backend/Dockerfile` per quickstart.md deployment section
- [X] T060 [P] Create render.yaml in `backend/render.yaml` for Render deployment per quickstart.md
- [ ] T061 Configure production environment variables on Render per quickstart.md
- [ ] T062 Configure frontend NEXT_PUBLIC_API_URL for production API per quickstart.md
- [ ] T063 Deploy backend to Render and verify health endpoint

### Error Handling & Logging

- [X] T064 [P] Add comprehensive error handling in `backend/src/api/chat.py` for all error responses per contracts/openapi.yaml ErrorResponse
- [X] T065 [P] Add structured logging throughout backend using Python logging module
- [X] T066 Implement graceful degradation when Neon is unavailable per spec.md failure behaviors

### Performance & Security

- [X] T067 [P] Add OpenAI API retry with exponential backoff in `backend/src/agents/` per plan.md risk mitigation
- [X] T068 Verify no API keys exposed in frontend code per spec.md SC-008
- [X] T069 Verify CORS configuration allows only specified origins per plan.md

### Validation

- [ ] T070 Run quickstart.md manual testing checklist
- [ ] T071 Verify <5s average response time per spec.md SC-001
- [ ] T072 Verify 100 concurrent users without degradation per spec.md SC-004

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Phase 2 completion
- **User Story 2 (Phase 4)**: Depends on Phase 2 completion (can run parallel to US1)
- **User Story 3 (Phase 5)**: Depends on Phase 2 completion (can run parallel to US1, US2)
- **User Story 4 (Phase 6)**: Depends on Phase 2 completion (can run parallel to others)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on other stories - Core Q&A functionality
- **User Story 2 (P2)**: Integrates with US1 chat flow, but independently testable
- **User Story 3 (P3)**: Uses US1 session infrastructure, but independently testable
- **User Story 4 (P4)**: Independent admin feature, provides content for US1-3

### Within Each User Story

1. Tests written and verified to FAIL before implementation
2. Agents (backend) before API endpoints
3. API endpoints before frontend components
4. Core implementation before integration
5. Story complete before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005 can run in parallel (different files)

**Phase 2 (Foundation)**:
- T010, T011 (vector services) can run parallel to T007, T008, T009 (database)
- T012, T013 (auth/config) can run parallel to database/vector work

**Phase 3 (User Story 1)**:
- All tests T018-T022 can run in parallel
- Agents T023-T026 can run in parallel (different agent files)
- T032 (npm install) can run parallel to all backend work
- T033-T037 (frontend components) can run in parallel after T032

**Phase 4-6 (User Stories 2-4)**:
- Can be implemented in parallel by different developers
- Each story's tests can run in parallel within that story

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test for Retrieval Agent in backend/tests/test_agents.py::test_retrieval_agent"
Task: "Unit test for Answer Agent in backend/tests/test_agents.py::test_answer_agent"
Task: "Unit test for Citation Agent in backend/tests/test_agents.py::test_citation_agent"
Task: "Integration test for POST /api/chat in backend/tests/test_chat.py::test_chat_endpoint"
Task: "Integration test for 'I don't know' response in backend/tests/test_chat.py::test_no_content_found"

# Launch all agents for User Story 1 together:
Task: "Implement Retrieval Agent in backend/src/agents/retrieval.py"
Task: "Implement Answer Generation Agent in backend/src/agents/answer.py"
Task: "Implement Citation Agent in backend/src/agents/citation.py"
Task: "Implement Session & Logging Agent in backend/src/agents/session.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T006)
2. Complete Phase 2: Foundational (T007-T017)
3. Complete Phase 3: User Story 1 (T018-T038)
4. **STOP and VALIDATE**: Test US1 independently using quickstart.md checklist
5. Deploy MVP to Render

### Incremental Delivery

1. Setup + Foundation → Foundation ready
2. Add User Story 1 → Deploy MVP (users can ask questions)
3. Add User Story 2 → Deploy (selected text context)
4. Add User Story 3 → Deploy (conversation history)
5. Add User Story 4 → Deploy (admin ingestion)
6. Polish → Final deployment

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundation together
2. Once Foundation is done:
   - Developer A: User Story 1 (Backend agents)
   - Developer B: User Story 1 (Frontend ChatWidget)
3. After US1:
   - Developer A: User Story 2 + User Story 4 (Backend)
   - Developer B: User Story 2 + User Story 3 (Frontend)
4. All: Polish phase

---

## Task Summary

| Phase | Task Count | User Story |
|-------|------------|------------|
| Setup | 6 | - |
| Foundation | 11 | - |
| User Story 1 | 21 | US1 (MVP) |
| User Story 2 | 6 | US2 |
| User Story 3 | 8 | US3 |
| User Story 4 | 6 | US4 |
| Polish | 14 | - |
| **Total** | **72** | |

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
