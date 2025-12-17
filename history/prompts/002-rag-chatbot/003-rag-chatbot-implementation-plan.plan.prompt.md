---
id: 003
title: RAG Chatbot Implementation Plan Generation
stage: plan
date: 2025-12-17
surface: agent
model: claude-opus-4-5-20251101
feature: 002-rag-chatbot
branch: 002-rag-chatbot
user: abdul
command: /sp.plan
labels: ["planning", "implementation", "architecture", "fastapi", "openai-agents-sdk", "chatkit", "better-auth"]
links:
  spec: specs/002-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/002-rag-chatbot/plan.md
  - specs/002-rag-chatbot/research.md
  - specs/002-rag-chatbot/data-model.md
  - specs/002-rag-chatbot/quickstart.md
  - specs/002-rag-chatbot/contracts/openapi.yaml
  - CLAUDE.md
tests:
  - N/A (planning phase)
---

## Prompt

/sp.plan

Create a structured implementation plan following a strict Spec-Kit Plus workflow to build the RAG chatbot defined in the specification.

Plan must include:

- Monorepo setup with /frontend (Docusaurus + ChatKit SDK) and /backend (FastAPI).
- FastAPI backend initialization using uv.
- Neon Serverless Postgres provisioning and schema creation for:
  - Logged-in users
  - Chat sessions (30-day persistence)
  - Message history
  - Retrieval logs (vector IDs, similarity scores)
  - Performance metrics (latency, token usage, model)
- Integration of Better Auth for signup/signin and authentication of users.
- Qdrant Cloud setup and vector collection configuration.
- Parsing Docusaurus markdown content and token-based chunking.
- Embedding generation and ingestion pipeline.
- OpenAI Agents SDK implementation:
  - Retrieval agent
  - Answer agent with book-only constraint
  - Citation agent
  - Session and logging agent
- API endpoint implementation for chat queries, embeddings ingestion, and session retrieval.
- Frontend integration of ChatKit inside Docusaurus:
  - Floating chat widget with modern UI/UX (green/white/black theme)
  - Selected-text capture and augmentation of requests
  - Keyboard and mobile accessibility
- Logging, metrics, and error handling.
- Dockerization and deployment of FastAPI backend to Render.
- Environment variable management, CORS configuration, and frontend-backend connectivity.
- Testing and validation of:
  - Book-grounded answers
  - Selected-text queries
  - No-answer fallback behavior
  - Session persistence and logging
  - Authentication flow with Better Auth

Use the latest official documentation for OpenAI Agents SDK (via context7 MCP), OpenAI ChatKit, Better Auth, FastAPI, Qdrant, and Neon.

## Response snapshot

Generated comprehensive implementation plan with 7 phases:

**Phase 0 - Research (research.md)**:
- Documented technology decisions for all major components
- OpenAI Agents SDK: Multi-agent handoff pattern with function_tool, guardrails
- Better Auth: HTTP session validation from FastAPI via auth.api.getSession()
- Qdrant Cloud: AsyncQdrantClient, 1536-dim vectors, cosine distance
- Neon Postgres: asyncpg with pooled connections
- ChatKit: useChatKit hook with green (#10B981) theme
- Deployment: Render (free tier) for backend

**Phase 1 - Data Model (data-model.md)**:
- 5 Postgres entities: User, Session, Message, RetrievalLog, PerformanceMetric
- 1 Qdrant entity: BookChunk with payload schema
- Complete SQL schema with indexes and retention policy

**Phase 1 - Contracts (contracts/openapi.yaml)**:
- OpenAPI 3.1 specification
- 5 endpoints: /api/chat, /api/sessions, /api/sessions/{id}/messages, /api/ingest, /api/health
- Better Auth cookie-based security scheme

**Implementation Plan (plan.md)**:
- 7 phases: Infrastructure, Ingestion, Agents, API, Frontend, Testing, Deployment
- 40+ tasks with dependencies and acceptance criteria
- Constitution check: All 6 principles PASS
- Risk mitigation strategies documented

**Quickstart Guide (quickstart.md)**:
- Step-by-step setup for local development
- External service provisioning (Neon, Qdrant, OpenAI)
- Code snippets for initialization

**Agent Context Updated**:
- CLAUDE.md updated with 002-rag-chatbot technologies

## Outcome

- Impact: Complete implementation roadmap with all design artifacts
- Tests: N/A (planning phase)
- Files: 6 files created/updated
- Next prompts: `/sp.tasks` to generate detailed task breakdown with test cases
- Reflection: Documentation via Context7 MCP was essential for accurate code patterns; multi-agent workflow design aligns with OpenAI Agents SDK best practices

## Evaluation notes (flywheel)

- Failure modes observed: None - all technologies well-documented via Context7
- Graders run and results (PASS/FAIL): Constitution check PASS (6/6 principles)
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Consider adding sequence diagrams for agent handoffs in future plans
