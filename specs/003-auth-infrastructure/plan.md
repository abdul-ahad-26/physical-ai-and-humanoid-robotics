# Implementation Plan: Better Auth Authentication & Infrastructure Setup

**Branch**: `003-auth-infrastructure` | **Date**: 2025-12-17 | **Spec**: [spec.md](spec.md)
**Status**: ✅ COMPLETED (2025-12-18)

**⚠️ ACTUAL IMPLEMENTATION DIFFERS**: See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for detailed differences. Key changes:
- Custom React chat UI (not ChatKit)
- Direct ingestion script (not API-based)
- Qdrant v1.16.2 with `.query_points()` API
- Conversational responses support
- Navbar auth buttons added

---

## Summary

Set up Better Auth authentication system and complete infrastructure configuration for the RAG chatbot feature. This includes:

1. **Frontend Authentication**: Login/signup pages, Better Auth React client integration, navbar logout button
2. **Backend Session Validation**: FastAPI authentication dependency using direct database queries for Better Auth sessions
3. **Database Initialization**: Idempotent script to create all required tables (users, sessions, messages, retrieval_logs, performance_metrics)
4. **Content Ingestion**: Script to process markdown files from frontend/docs/ and populate Qdrant vector store via /api/ingest endpoint
5. **Integration**: Update Root.tsx to use useSession hook for conditional chat widget rendering

**Technical Approach**: Hybrid Better Auth integration - frontend uses Better Auth React client for authentication UI, backend validates sessions via direct Postgres queries to avoid non-existent Python package dependency.

---

## Technical Context

**Language/Version**: Python 3.11 (backend), TypeScript/JavaScript (frontend with Docusaurus/React)

**Primary Dependencies**:
- **Backend**: FastAPI, asyncpg, argon2-cffi, httpx, python-frontmatter, python-dotenv, OpenAI SDK, qdrant-client v1.16.2, tiktoken
- **Frontend**: better-auth/react (React client), react-markdown, remark-gfm (custom chat UI, NOT ChatKit)

**Storage**: Neon Serverless Postgres (users, sessions, messages, logs, metrics), Qdrant Cloud (vector embeddings)

**Testing**: pytest (backend integration tests), Jest + React Testing Library (frontend component tests)

**Target Platform**: Web application (Linux server for backend, browser for frontend)

**Project Type**: web

**Performance Goals**:
- Auth endpoints: <200ms p95 latency
- Database init script: <10 seconds
- Content ingestion: 50+ files in <5 minutes
- Chat response: <5 seconds including retrieval

**Constraints**:
- Must use existing database schema from 002-rag-chatbot (no modifications except password_hash column)
- Must use existing /api/ingest endpoint (no changes)
- Session storage must be in Neon Postgres (not in-memory or Redis)
- All credentials via environment variables (no hardcoding)
- Scripts must be idempotent (safe to run multiple times)

**Scale/Scope**:
- Expected users: 100-1000 concurrent
- Database tables: 6 relational tables
- Vector storage: 500-5000 document chunks
- Session duration: 7 days

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Spec-Driven Development ✅ PASS
- Complete specification created: `specs/003-auth-infrastructure/spec.md`
- 6 prioritized user stories with acceptance criteria
- 34 functional requirements across 4 categories
- Planning follows Spec-Kit Plus methodology

### Principle II: Technical Accuracy & Educational Clarity ✅ PASS
- All technical decisions documented in research.md
- Working code examples provided in quickstart.md
- Step-by-step setup instructions with verification checklist
- Error handling and troubleshooting guidance included

### Principle III: Modularity & Reusability ✅ PASS
- Authentication system designed as independent module
- Database initialization script reusable for any Postgres schema
- Content ingestion script adaptable for different document structures
- Clear interfaces between frontend/backend components

### Principle IV: Docusaurus-First Architecture ✅ PASS
- Authentication pages integrated as custom Docusaurus pages
- Login/signup routes follow Docusaurus conventions (src/pages/)
- Styling uses Docusaurus theme variables
- Preserves existing Docusaurus navigation and structure

### Principle V: AI-Native Content Design ✅ PASS
- Content ingestion script prepares data for RAG system
- Vector embeddings optimized for semantic search
- Metadata (chapter_id, section_id) enables precise retrieval
- Citation generation links answers to source content

### Principle VI: Practical Application Focus ✅ PASS
- Complete quickstart guide with hands-on steps
- Verification checklist for testing setup
- Troubleshooting section for common issues
- End-to-end flow testing (signup → login → chat → logout)

**Post-Design Re-evaluation**: All principles maintained through Phase 1. No additional complexity introduced beyond requirements.

---

## Project Structure

### Documentation (this feature)

```text
specs/003-auth-infrastructure/
├── spec.md              # Feature specification
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0: Technical decisions
├── data-model.md        # Phase 1: Entity definitions
├── quickstart.md        # Phase 1: Setup guide
├── contracts/           # Phase 1: API specifications
│   └── auth-api.yaml    # OpenAPI spec for auth endpoints
├── checklists/          # Quality validation
│   └── requirements.md  # Specification quality checklist
└── tasks.md             # Phase 2: NOT YET CREATED (use /sp.tasks)
```

### Source Code (repository root)

```text
backend/
├── routers/
│   ├── auth.py          # NEW: Authentication endpoints
│   ├── chat.py          # EXISTING: Chat endpoint (002-rag-chatbot)
│   └── ingest.py        # EXISTING: Content ingestion (002-rag-chatbot)
├── scripts/
│   ├── init_db.py       # NEW: Database initialization script
│   └── ingest_docs.py   # NEW: Content ingestion script
├── tests/
│   ├── test_auth.py     # NEW: Auth endpoint tests
│   ├── test_init_db.py  # NEW: Database init tests
│   └── test_ingest_docs.py  # NEW: Ingestion script tests
├── main.py              # MODIFIED: Add auth router
├── .env                 # NEW: Environment variables
└── requirements.txt     # MODIFIED: Add new dependencies

frontend/
├── src/
│   ├── pages/
│   │   ├── login.tsx    # NEW: Login page
│   │   ├── signup.tsx   # NEW: Signup page
│   │   └── login.module.css  # NEW: Auth page styles
│   ├── lib/
│   │   └── auth.ts      # NEW: Better Auth client config
│   ├── theme/
│   │   └── Root.tsx     # MODIFIED: Use useSession hook
│   └── components/
│       └── ChatWidget.tsx  # EXISTING: Chat widget (002-rag-chatbot)
├── docs/                # EXISTING: Textbook markdown content
├── .env                 # NEW: Frontend environment variables
└── package.json         # MODIFIED: Add better-auth dependency
```

**Structure Decision**: Web application with separate backend/ and frontend/ directories. Backend contains FastAPI application with authentication router, database scripts, and tests. Frontend contains Docusaurus site with custom authentication pages, Better Auth client configuration, and updated Root component. This structure separates concerns while maintaining clear integration points via API contracts.

---

## Complexity Tracking

No violations of constitution principles. No additional complexity beyond requirements.
