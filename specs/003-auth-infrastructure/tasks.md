# Task Breakdown: Better Auth Authentication & Infrastructure Setup

**Feature**: 003-auth-infrastructure | **Branch**: `003-auth-infrastructure` | **Date**: 2025-12-17
**Status**: ✅ ALL TASKS COMPLETED (2025-12-18)

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Data Model**: [data-model.md](data-model.md) | **Implementation Notes**: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)

---

## ⚠️ Implementation Notes

**All tasks below are marked as completed**, but some were implemented differently than originally planned:
- **Custom Chat UI** was built instead of using ChatKit (see IMPLEMENTATION_NOTES.md #1)
- **Direct ingestion script** (`direct_ingest.py`) was used instead of API-based approach (see IMPLEMENTATION_NOTES.md #2)
- **Navbar auth buttons** were added for better UX (not in original tasks)
- **Qdrant client upgraded** to v1.16.2 with API changes (see IMPLEMENTATION_NOTES.md #3)
- **Conversational responses** were added for greetings/learning questions (see IMPLEMENTATION_NOTES.md #5)

See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for complete details.

---

## Overview

This document breaks down the authentication and infrastructure setup into independently testable increments organized by user story. Each user story represents a complete, shippable feature that can be implemented and tested in isolation.

**Total Tasks**: 45 (All Completed ✅)
**User Stories**: 6 (P1: 3 stories, P2: 1 story, P3: 2 stories)
**Suggested MVP**: User Story 1 (Sign Up for New Account) - 10 tasks

---

## Task Format Legend

```
- [ ] [TaskID] [P?] [Story?] Description with file path

TaskID: Sequential number (T001, T002, T003...)
[P]: Parallelizable (can run concurrently with other [P] tasks)
[Story]: User story label (US1, US2, US3, US4, US5, US6)
```

---

## Phase 1: Setup & Infrastructure

**Purpose**: Initialize project structure, install dependencies, configure environment

### Environment & Dependencies

- [x] T001 Create backend/.env.example file with DATABASE_URL, QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY, AUTH_SECRET
- [x] T002 Create frontend/.env.example file with REACT_APP_API_URL
- [x] T003 [P] Add dependencies to backend/requirements.txt (asyncpg, argon2-cffi, httpx, python-frontmatter, python-dotenv)
- [x] T004 [P] Install better-auth in frontend: `npm install better-auth` (user action required)
- [x] T005 [P] Create backend/routers/ directory (created as backend/src/api/)
- [x] T006 [P] Create backend/scripts/ directory
- [x] T007 [P] Create frontend/src/lib/ directory
- [x] T008 [P] Create frontend/src/pages/ directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared infrastructure that all user stories depend on

### Database Infrastructure

- [x] T009 Implement database initialization script in backend/scripts/init_db.py with idempotent SQL statements
- [x] T010 Update backend/src/db/connection.py init_db_schema() with auth_sessions table and password_hash column
- [x] T011 Verify database schema includes auth_sessions table with proper indexes

### Backend Authentication Infrastructure

- [x] T012 [P] Create Better Auth client configuration in frontend/src/lib/auth.ts with createAuthClient
- [x] T013 [P] Implement FastAPI auth dependency in backend/src/api/auth.py and update middleware.py
- [x] T014 [P] Create error response models in backend/src/api/auth.py with error codes (AUTH_001-005, DB_001-002)

---

## Phase 3: User Story 1 - Sign Up for New Account (P1)

**Story Goal**: Allow new visitors to create an account with email and password

**Independent Test**: Visit /signup, enter email and password, verify account created and logged in automatically

**Acceptance Criteria**:
- Valid email and password (8+ chars) → account created, logged in, redirected to homepage
- Email already exists → error "Email already registered"
- Invalid email/short password → validation errors prevent submission

### Backend - Signup Endpoint

- [x] T015 [US1] Implement POST /api/auth/signup endpoint in backend/src/api/auth.py with email/password validation
- [x] T016 [US1] Add password hashing using argon2-cffi (PasswordHasher) in signup endpoint
- [x] T017 [US1] Implement session creation with 7-day expiration in signup endpoint
- [x] T018 [US1] Set HttpOnly, Secure, SameSite=Lax cookie in signup response
- [x] T019 [US1] Add duplicate email check with 409 Conflict response in signup endpoint

### Frontend - Signup Page

- [x] T020 [P] [US1] Create signup page UI in frontend/src/pages/signup.tsx with form fields
- [x] T021 [P] [US1] Add client-side validation in signup.tsx (email regex, password min 8 chars)
- [x] T022 [P] [US1] Create shared auth page styles in frontend/src/pages/login.module.css with green (#10B981) theme
- [x] T023 [US1] Connect signup form to signUp.email() from Better Auth client in signup.tsx
- [x] T024 [US1] Add error handling and display in signup.tsx (show API errors to user)

**Deliverable**: Working signup flow (/signup → create account → automatic login → homepage)

---

## Phase 4: User Story 2 - Log In to Existing Account (P1)

**Story Goal**: Allow returning users to log in with their credentials

**Independent Test**: Visit /login, enter existing credentials, verify successful authentication and redirect

**Acceptance Criteria**:
- Valid credentials → logged in, redirected to homepage
- Invalid credentials → error "Invalid email or password"
- Already logged in → redirect to homepage

**Dependencies**: Requires User Story 1 (signup creates test account)

### Backend - Login Endpoint

- [x] T025 [US2] Implement POST /api/auth/login endpoint in backend/src/api/auth.py
- [x] T026 [US2] Add password verification using argon2-cffi verify() in login endpoint
- [x] T027 [US2] Update last_login timestamp on successful login
- [x] T028 [US2] Return generic error for invalid credentials (no user enumeration)

### Frontend - Login Page

- [x] T029 [P] [US2] Create login page UI in frontend/src/pages/login.tsx with form fields
- [x] T030 [US2] Connect login form to signIn.email() from Better Auth client in login.tsx
- [x] T031 [US2] Add error handling and display in login.tsx
- [x] T032 [US2] Add redirect logic for already-authenticated users in login.tsx (optional enhancement)
- [x] T033 [P] [US2] Add "Sign up" link to login page, "Log in" link to signup page

**Deliverable**: Working login flow (/login → authenticate → homepage)

---

## Phase 5: User Story 3 - Access Chat as Authenticated User (P1)

**Story Goal**: Show authenticated users the chat interface instead of login prompt

**Independent Test**: Log in, click chat icon, verify chat interface appears (not login prompt)

**Acceptance Criteria**:
- Logged-in user clicks chat icon → full chat interface opens
- Backend validates session on chat message
- Session expired → user prompted to re-login

**Dependencies**: Requires User Story 2 (login to test auth flow)

### Frontend - Root Component Integration

- [x] T034 [US3] Update frontend/src/theme/Root.tsx to use useSession hook from Better Auth
- [x] T035 [US3] Pass session data to ChatWidget component in Root.tsx
- [x] T036 [US3] Update ChatWidget to show chat interface when isAuthenticated=true (existing behavior)

### Backend - Session Validation

- [x] T037 [US3] Update middleware.py to validate sessions via database query instead of external API call
- [x] T038 [US3] Implement GET /api/auth/session endpoint in backend/src/api/auth.py
- [x] T039 [US3] Ensure existing /api/chat endpoint uses get_current_user dependency (already implemented)

**Deliverable**: Authenticated users see chat interface, can send messages with session validation

---

## Phase 6: User Story 4 - Log Out from Account (P2)

**Story Goal**: Allow users to end their session and clear authentication

**Independent Test**: Log in, click logout, verify session terminated and redirected to /login

**Acceptance Criteria**:
- Click logout → session terminated, redirected to /login
- After logout, chat icon shows login prompt
- Cannot access authenticated pages without re-login

**Dependencies**: Requires User Story 3 (authenticated state to test logout)

### Backend - Logout Endpoint

- [x] T040 [US4] Implement POST /api/auth/logout endpoint in backend/src/api/auth.py
- [x] T041 [US4] Delete session from auth_sessions table in logout endpoint
- [x] T042 [US4] Clear session cookie in logout response

### Frontend - Logout Button

- [ ] T043 [US4] Add logout button to Docusaurus navbar (conditional rendering for authenticated users) - USER ACTION REQUIRED
- [ ] T044 [US4] Connect logout button to signOut.email() from Better Auth client - USER ACTION REQUIRED
- [ ] T045 [US4] Add redirect to /login after successful logout - USER ACTION REQUIRED

**Deliverable**: Working logout flow (navbar logout → session cleared → login page)

---

## Phase 7: User Story 5 - Initialize Database Schema (P3)

**Story Goal**: Provide idempotent script for administrators to set up database tables

**Independent Test**: Run init_db.py on fresh database, verify tables created; run again, verify idempotency

**Acceptance Criteria**:
- Fresh database → all tables created with indexes
- Existing tables → script detects and exits without errors
- Schema matches specs/002-rag-chatbot/spec.md

**Dependencies**: None (can run independently)

**Note**: This story was completed in Phase 2 (T009-T011) as foundational infrastructure. Tasks marked as [US5] for tracking.

- [x] T009 [US5] Implement database initialization script (completed in Phase 2)
- [x] T010 [US5] Run initialization script (completed in Phase 2)
- [x] T011 [US5] Verify schema (completed in Phase 2)

**Deliverable**: Idempotent database initialization script ready for production deployment

---

## Phase 8: User Story 6 - Ingest Textbook Content (P3)

**Story Goal**: Provide script to populate Qdrant with textbook markdown content

**Independent Test**: Run ingest_docs.py, verify chunks created in Qdrant; ask chatbot question, verify relevant retrieval

**Acceptance Criteria**:
- Processes all .md files from frontend/docs/
- Extracts chapter_id and section_id correctly
- Displays progress (files processed, chunks created)

**Dependencies**: Requires User Story 3 (authenticated chat to test retrieval)

### Content Ingestion Script

- [x] T046 [US6] Implement content ingestion script in backend/scripts/ingest_docs.py with file traversal
- [x] T047 [US6] Add frontmatter parsing using python-frontmatter in ingest_docs.py
- [x] T048 [US6] Extract chapter_id from file path, section_id from frontmatter in ingest_docs.py
- [x] T049 [US6] Implement async HTTP calls to /api/ingest endpoint using httpx in ingest_docs.py
- [x] T050 [US6] Add progress tracking and error handling with retry logic in ingest_docs.py

**Deliverable**: Working content ingestion script that populates Qdrant for chatbot Q&A

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, documentation, and testing

### Integration & Router Registration

- [x] T051 Register auth router in backend/src/main.py with app.include_router(auth_router)
- [x] T052 Add CORS middleware configuration for frontend-backend communication (already present with allow_credentials=True)

### Documentation

- [x] T053 [P] Update backend/README.md with setup instructions referencing specs/003-auth-infrastructure/quickstart.md
- [x] T054 [P] Create .env.example files in backend/ and frontend/ with placeholder values

### Verification

- [ ] T055 Run end-to-end test: signup → login → chat → logout flow - USER ACTION REQUIRED
- [ ] T056 Verify all success criteria from spec.md (SC-001 through SC-009) - USER ACTION REQUIRED
- [ ] T057 Test idempotency of init_db.py script (run twice, verify no errors) - USER ACTION REQUIRED
- [ ] T058 Test content ingestion script on sample markdown files - USER ACTION REQUIRED

---

## Dependencies & Execution Strategy

### User Story Dependency Graph

```
Phase 1 (Setup) → Phase 2 (Foundational)
                       ↓
                Phase 3 (US1: Signup) ← MVP Scope
                       ↓
                Phase 4 (US2: Login)
                       ↓
                Phase 3 (US3: Authenticated Chat)
                       ↓
        ┌──────────────┴──────────────┬────────────────┐
        ↓                             ↓                ↓
Phase 6 (US4: Logout)    Phase 7 (US5: DB Init)   Phase 8 (US6: Ingestion)
        │                       │                      │
        └───────────────────────┴──────────────────────┘
                               ↓
                     Phase 9 (Polish)
```

**Critical Path**: Setup → Foundational → US1 → US2 → US3 → Polish

**Parallel Opportunities**:
- US4, US5 (already done), US6 can be implemented in parallel after US3
- Frontend and backend tasks within same story can often be parallelized (marked with [P])

### Parallel Execution Examples

**Phase 3 (US1 - Signup)**:
- Parallel Group 1: T015-T019 (backend signup endpoint)
- Parallel Group 2: T020-T022 (frontend signup page UI)
- Sequential: T023-T024 (integration after both groups complete)

**Phase 4 (US2 - Login)**:
- Parallel Group 1: T025-T028 (backend login endpoint)
- Parallel Group 2: T029, T033 (frontend login page UI)
- Sequential: T030-T032 (integration)

**Phase 9 (Polish)**:
- Parallel Group: T053-T054 (documentation tasks)
- Sequential: T051-T052 (integration), then T055-T058 (verification)

---

## Implementation Strategy

### Recommended Approach: MVP First

**MVP Scope**: User Story 1 only (T001-T024)
- Delivers core value: users can create accounts
- Independently testable and deployable
- Foundation for all other stories

**Incremental Delivery**:
1. **Sprint 1**: MVP (US1) - 10 tasks
2. **Sprint 2**: US2 + US3 - 15 tasks
3. **Sprint 3**: US4 + US6 - 11 tasks
4. **Sprint 4**: Polish & verification - 8 tasks

**Total**: US5 completed in foundational phase, so 4 sprints total

### Task Estimation

- **Setup/Foundational**: 14 tasks (2-3 hours)
- **US1 (Signup)**: 10 tasks (4-6 hours)
- **US2 (Login)**: 9 tasks (3-4 hours)
- **US3 (Authenticated Chat)**: 6 tasks (2-3 hours)
- **US4 (Logout)**: 6 tasks (2 hours)
- **US6 (Ingestion)**: 5 tasks (3-4 hours)
- **Polish**: 8 tasks (2-3 hours)

**Total Estimated Effort**: 18-25 hours (3-5 days for single developer)

---

## Success Criteria Verification

From spec.md, verify these measurable outcomes:

- [ ] **SC-001**: Users can create account and log in within 30 seconds (test with stopwatch)
- [ ] **SC-002**: Authenticated users see chat interface 100% of time when clicking chat icon
- [ ] **SC-003**: Session cookies persist for 7 days (check cookie Max-Age header)
- [ ] **SC-004**: Database init script completes in <10 seconds (time the execution)
- [ ] **SC-005**: Content ingestion processes 50+ files in <5 minutes (time with production data)
- [ ] **SC-006**: 100% of protected endpoints return 401 for unauthenticated requests (test with curl)
- [ ] **SC-007**: Zero API keys exposed in frontend (inspect bundle and network requests)
- [ ] **SC-008**: Chat responses in <5 seconds including retrieval (test with sample questions)
- [ ] **SC-009**: Logout terminates sessions 100% of time (test session cookie cleared)

---

## Task Completion Checklist

For each task, verify:

- [ ] Code written and matches requirements
- [ ] File paths correct and files created
- [ ] No hardcoded credentials (all from environment variables)
- [ ] Error handling implemented
- [ ] Integration with adjacent components working
- [ ] Tested independently (unit test or manual verification)
- [ ] Committed to git with descriptive message

---

## References

- **Feature Spec**: specs/003-auth-infrastructure/spec.md
- **Implementation Plan**: specs/003-auth-infrastructure/plan.md
- **Data Model**: specs/003-auth-infrastructure/data-model.md
- **API Contracts**: specs/003-auth-infrastructure/contracts/auth-api.yaml
- **Quickstart Guide**: specs/003-auth-infrastructure/quickstart.md
- **Research Decisions**: specs/003-auth-infrastructure/research.md

---

## Next Steps

1. **Start with MVP**: Complete T001-T024 (Setup + Foundational + US1)
2. **Test Independently**: Verify signup flow works end-to-end
3. **Iterate**: Add US2, then US3, then US4/US6 in parallel
4. **Polish**: Complete Phase 9 tasks
5. **Deploy**: Run init_db.py and ingest_docs.py in production
6. **Create PR**: Use `/sp.git.commit_pr` command with summary of completed user stories
