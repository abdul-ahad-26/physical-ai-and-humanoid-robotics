---
id: 0002
title: auth-infrastructure-planning
stage: plan
date: 2025-12-17
surface: agent
model: claude-opus-4-5-20251101
feature: 003-auth-infrastructure
branch: 003-auth-infrastructure
user: system
command: /sp.plan
labels: ["authentication", "infrastructure", "planning", "architecture", "better-auth"]
links:
  spec: specs/003-auth-infrastructure/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/003-auth-infrastructure/plan.md
  - specs/003-auth-infrastructure/research.md
  - specs/003-auth-infrastructure/data-model.md
  - specs/003-auth-infrastructure/quickstart.md
  - specs/003-auth-infrastructure/contracts/auth-api.yaml
tests:
  - Constitution compliance validation (all 6 principles PASS)
---

## Prompt

/sp.plan

## Response snapshot

Created comprehensive implementation plan for feature 003-auth-infrastructure (Better Auth Authentication & Infrastructure Setup).

**Artifacts Generated**:

### Phase 0: Research (research.md)
Resolved all technical decisions:
1. Better Auth Integration: Hybrid approach (frontend React client + backend direct DB queries)
2. Frontend Pages: Standalone React pages in Docusaurus
3. Database Init: asyncpg with idempotent SQL
4. Content Ingestion: httpx + frontmatter + pathlib
5. Session Validation: FastAPI dependency with direct DB queries
6. Password Hashing: Argon2 via argon2-cffi
7. Environment Config: python-dotenv + Docusaurus env vars
8. Navbar Logout: Swizzled navbar component
9. Error Handling: Structured error responses with codes
10. Testing: Integration tests + idempotency tests

**Key Finding**: Better Auth Python package does not exist → Mitigated with hybrid architecture.

### Phase 1: Design (data-model.md, contracts/, quickstart.md)

**Data Model**:
- 6 relational entities (User, AuthSession, ChatSession, Message, RetrievalLog, PerformanceMetric)
- 1 non-relational entity (Qdrant vector store)
- Foreign key cascades, unique constraints, check constraints
- Indexes on all foreign keys and frequently queried fields

**API Contracts (auth-api.yaml)**:
- OpenAPI 3.0.3 specification
- 4 endpoints: POST /api/auth/signup, POST /api/auth/login, POST /api/auth/logout, GET /api/auth/session
- Structured error responses with codes (AUTH_001-005, DB_001-002)
- Session cookies: HttpOnly, Secure, SameSite=Lax, 7-day expiration

**Quickstart Guide**:
- 9-step setup process from environment configuration to end-to-end testing
- Verification checklist (10 items)
- Troubleshooting section
- Code examples for auth.ts, login.tsx, auth router

**Agent Context Update**: Added new technologies to CLAUDE.md (asyncpg, argon2-cffi, better-auth)

**Constitution Check (Re-evaluated after Phase 1)**: All 6 principles PASS
- Spec-Driven Development ✅
- Technical Accuracy & Educational Clarity ✅
- Modularity & Reusability ✅
- Docusaurus-First Architecture ✅
- AI-Native Content Design ✅
- Practical Application Focus ✅

**Plan Completion (plan.md)**:
- Summary with technical approach
- Technical context (Python 3.11 backend, TypeScript/React frontend)
- Constitution check results
- Project structure (backend/ + frontend/)
- Phase 0 research summary
- Phase 1 design artifacts summary
- Implementation sequence (6 stages with dependencies)
- Risk analysis (4 risks with mitigations)
- Success metrics (9 measurable outcomes)

## Outcome

- ✅ Impact: Complete implementation plan created for authentication infrastructure. All technical decisions resolved, design artifacts generated, ready for /sp.tasks.
- 🧪 Tests: Constitution compliance validation PASS (all 6 principles), specification quality checklist PASS (from previous session)
- 📁 Files: plan.md, research.md, data-model.md, quickstart.md, contracts/auth-api.yaml (5 planning artifacts)
- 🔁 Next prompts: `/sp.tasks` to generate tasks.md from this plan, or begin implementation directly using quickstart.md
- 🧠 Reflection: Successfully resolved Better Auth Python package issue with hybrid architecture. All constitution principles maintained. Design artifacts provide clear path to implementation.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Constitution check PASS (6/6 principles)
- Prompt variant (if applicable): Standard /sp.plan workflow
- Next experiment (smallest change to try): None - planning complete and validated
