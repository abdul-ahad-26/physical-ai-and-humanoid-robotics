# Tasks: User Profiling, Content Personalization & Translation

**Input**: Design documents from `/specs/005-user-personalization/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Manual E2E testing per plan.md - unit tests included for core logic only.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, etc.)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/src/`, `backend/tests/`, `backend/scripts/migrations/`
- **Frontend**: `frontend/src/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and configuration

- [x] T001 Install backend dependencies: `openai-agents` and `cachetools` in backend/requirements.txt
- [x] T002 [P] Add new environment variables to backend/.env.example: OPENAI_MODEL, PERSONALIZATION_CACHE_TTL, TRANSLATION_CACHE_TTL
- [x] T003 [P] Add Noto Nastaliq Urdu font import to frontend/docusaurus.config.js stylesheets

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Database Migration

- [x] T004 Create migration file backend/scripts/migrations/005_add_profile_fields.sql with schema from data-model.md
- [x] T005 Run database migration against Neon Postgres and verify tables/columns exist

### Backend Models & Shared Services

- [x] T006 [P] Create Pydantic models in backend/src/db/models.py: SoftwareBackground, HardwareBackground, UserProfile, ProfileUpdateRequest, ProfileUpdateResponse
- [x] T007 [P] Create cache service in backend/src/services/cache.py with TTLCache for personalization and translation
- [x] T008 [P] Create code block extractor in backend/src/services/code_extractor.py with extract_code() and restore_code() functions
- [x] T009 [P] Create AI rate limiter class in backend/src/api/middleware.py: AIRateLimiter with 10 req/min per user
- [x] T010 Add profile database queries in backend/src/db/queries.py: get_user_profile(), update_user_profile(), log_personalization(), log_translation()

### Backend Config Update

- [x] T011 Update backend/src/config.py to load new environment variables: OPENAI_MODEL, PERSONALIZATION_CACHE_TTL, TRANSLATION_CACHE_TTL

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Complete Technical Background Profile (Priority: P1) 🎯 MVP

**Goal**: New users can complete a 3-step profile wizard to capture display name and technical background

**Independent Test**: Create a new account, complete profile wizard, verify data is stored and profile_completed=true

### Backend Implementation for US1

- [x] T012 [P] [US1] Create profile API endpoint file backend/src/api/profile.py with GET /api/profile endpoint
- [x] T013 [US1] Add POST /api/profile endpoint in backend/src/api/profile.py for profile creation/update
- [x] T014 [US1] Add profile_completed logic: set true when both software_background.level and hardware_background.level are provided
- [x] T015 [US1] Register profile routes in backend/src/main.py

### Frontend Implementation for US1

- [x] T016 [P] [US1] Create ProfileWizard types in frontend/src/components/ProfileWizard/types.ts
- [x] T017 [P] [US1] Create step component frontend/src/components/ProfileWizard/StepName.tsx for display name input
- [x] T018 [P] [US1] Create step component frontend/src/components/ProfileWizard/StepSoftware.tsx for software background (level dropdown, language/framework multi-select)
- [x] T019 [P] [US1] Create step component frontend/src/components/ProfileWizard/StepHardware.tsx for hardware background (level dropdown, domains multi-select)
- [x] T020 [US1] Create main ProfileWizard component in frontend/src/components/ProfileWizard/index.tsx with step navigation, progress indicator, skip warning
- [x] T021 [US1] Add profile API client functions in frontend/src/lib/personalization.ts: getProfile(), updateProfile()
- [x] T022 [US1] Integrate ProfileWizard into auth flow: show modal when profile_completed=false after login

### Unit Tests for US1

- [x] T023 [P] [US1] Create unit test backend/tests/unit/test_profile.py for profile CRUD operations and validation

**Checkpoint**: User Story 1 complete - users can create and complete profiles

---

## Phase 4: User Story 2 - Display User Name in Navbar (Priority: P1)

**Goal**: Logged-in users see their display name (not email) in the navbar

**Independent Test**: Log in with completed profile, verify navbar shows "Welcome, {display_name}" instead of email

### Backend Implementation for US2

- [x] T024 [US2] Update GET /api/auth/session response in backend/src/api/auth.py to include display_name and profile_completed fields

### Frontend Implementation for US2

- [x] T025 [US2] Update AuthButton component in frontend/src/components/AuthButton/ to display display_name when available, fallback to email
- [x] T026 [US2] Add name truncation with ellipsis in AuthButton if display_name exceeds 20 characters

**Checkpoint**: User Stories 1 AND 2 complete - profile wizard works and navbar shows personalized greeting

---

## Phase 5: User Story 3 - Personalize Chapter Content (Priority: P2)

**Goal**: Users can click "Personalize Content" to get AI-adapted chapter content based on their technical background

**Independent Test**: Log in with completed profile, navigate to chapter, click "Personalize Content", verify content adapts to background level

### Backend Implementation for US3

- [x] T027 [P] [US3] Create Personalization Agent in backend/src/agents/personalization.py with UserProfile model, PersonalizedOutput model, and personalize_content() async function
- [x] T028 [US3] Create personalize API endpoint in backend/src/api/personalize.py with POST /api/personalize
- [x] T029 [US3] Add request validation: require authentication, profile_completed=true, rate limit check
- [x] T030 [US3] Integrate cache check in personalize endpoint: return cached content if exists, otherwise call agent
- [x] T031 [US3] Add code block extraction before AI call and restoration after in personalize endpoint
- [x] T032 [US3] Add audit logging to personalization_logs table in personalize endpoint
- [x] T033 [US3] Register personalize routes in backend/src/main.py

### Frontend Implementation for US3

- [x] T034 [P] [US3] Create ChapterActions component styles in frontend/src/components/ChapterActions/styles.module.css
- [x] T035 [US3] Create ChapterActions component in frontend/src/components/ChapterActions/index.tsx with "Personalize Content" / "Show Original" toggle button
- [x] T036 [US3] Add personalize API client function in frontend/src/lib/personalization.ts: personalizeContent(chapterId, content, title)
- [x] T037 [US3] Add loading state and error handling in ChapterActions for personalization requests
- [x] T038 [US3] Integrate ChapterActions into DocItem wrapper in frontend/src/theme/DocItem/ to show on all chapter pages

### Unit Tests for US3

- [x] T039 [P] [US3] Create unit test backend/tests/unit/test_personalization.py for code extraction/restoration and agent output validation

**Checkpoint**: User Story 3 complete - personalization feature fully functional

---

## Phase 6: User Story 4 - Translate Chapter to Urdu (Priority: P2)

**Goal**: Users can click "Translate to Urdu" to see chapter content in Urdu with RTL support and preserved code blocks

**Independent Test**: Log in, navigate to chapter, click "Translate to Urdu", verify prose is translated, code blocks unchanged, RTL styling applied

### Backend Implementation for US4

- [x] T040 [P] [US4] Create Translation Agent in backend/src/agents/translation.py with TranslatedOutput model and translate_content() async function
- [x] T041 [US4] Create translate API endpoint in backend/src/api/translate.py with POST /api/translate
- [x] T042 [US4] Add request validation: require authentication, rate limit check, validate target_language="ur"
- [x] T043 [US4] Integrate cache check in translate endpoint: return cached content if exists, otherwise call agent
- [x] T044 [US4] Add code block extraction before AI call and restoration after in translate endpoint
- [x] T045 [US4] Add audit logging to translation_logs table in translate endpoint
- [x] T046 [US4] Register translate routes in backend/src/main.py

### Frontend Implementation for US4

- [x] T047 [P] [US4] Create RTL CSS styles in frontend/src/css/rtl.css for .translated-content-urdu class with proper code block LTR override
- [x] T048 [US4] Add "Translate to Urdu" / "Show English" toggle button to ChapterActions component in frontend/src/components/ChapterActions/index.tsx
- [x] T049 [US4] Add translate API client function in frontend/src/lib/personalization.ts: translateContent(chapterId, content, targetLanguage)
- [x] T050 [US4] Add RTL wrapper with dir="rtl" and translated-content-urdu class when displaying translated content
- [x] T051 [US4] Import rtl.css in frontend/src/css/custom.css or appropriate global stylesheet

### Unit Tests for US4

- [x] T052 [P] [US4] Create unit test backend/tests/unit/test_translation.py for code preservation validation in translation

**Checkpoint**: User Story 4 complete - translation feature fully functional

---

## Phase 7: User Story 5 - Update Profile Settings (Priority: P3)

**Goal**: Returning users can update their display name and technical background from a settings page

**Independent Test**: Log in, navigate to /profile, modify fields, save, verify changes persist and affect future personalization

### Frontend Implementation for US5

- [x] T053 [US5] Create profile settings page in frontend/src/pages/profile.tsx with form for display_name, software_background, hardware_background
- [x] T054 [US5] Add form validation in profile settings: display_name max 100 chars, required level fields
- [x] T055 [US5] Add save/cancel functionality with success/error feedback messages
- [x] T056 [US5] Add link to /profile page in navbar user dropdown menu (already exists from auth feature)

**Checkpoint**: User Story 5 complete - users can update their profiles

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T057 [P] Add 30-second timeout handling for AI requests in both personalize and translate endpoints
- [x] T058 [P] Add 500ms debounce for rapid toggle clicks in ChapterActions component
- [x] T059 Validate code block preservation with integration test: before/after content comparison
- [x] T060 [P] Add error boundary for AI service failures: display user-friendly error with retry option
- [x] T061 Run quickstart.md validation: test all setup steps and API examples

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 and US2 are both P1 priority - complete before P2 stories
  - US3 and US4 are P2 priority - can proceed after US1/US2
  - US5 is P3 priority - can proceed after US1/US2
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: After Foundational → No dependencies on other stories
- **US2 (P1)**: After Foundational → Requires profile data from US1 backend (T024 depends on T012-T015)
- **US3 (P2)**: After Foundational → Requires profile_completed check (integrates with US1)
- **US4 (P2)**: After Foundational → Independent of US1 (translation doesn't require profile)
- **US5 (P3)**: After US1 → Uses same profile API

### Within Each User Story

- Backend models and services first
- API endpoints before frontend
- Core functionality before integration
- Tests can run in parallel within story

### Parallel Opportunities

**Setup (Phase 1)**:
```text
Parallel: T002 + T003
```

**Foundational (Phase 2)**:
```text
Parallel: T006 + T007 + T008 + T009
```

**User Story 1 (Phase 3)**:
```text
Parallel backend: T012 (GET endpoint file created)
Parallel frontend: T016 + T017 + T018 + T019 (all step components)
```

**User Story 3 (Phase 5)**:
```text
Parallel: T027 + T034 (agent + styles)
```

**User Story 4 (Phase 6)**:
```text
Parallel: T040 + T047 (agent + RTL CSS)
```

---

## Parallel Example: Foundational Phase

```bash
# Launch shared services in parallel (all different files):
Task T006: "Create Pydantic models in backend/src/db/models.py"
Task T007: "Create cache service in backend/src/services/cache.py"
Task T008: "Create code block extractor in backend/src/services/code_extractor.py"
Task T009: "Create AI rate limiter in backend/src/api/middleware.py"

# Then sequential:
Task T010: "Add profile database queries" (depends on T006)
Task T011: "Update config" (independent but sequential for clarity)
```

## Parallel Example: User Story 1 Frontend

```bash
# Launch all ProfileWizard step components in parallel:
Task T016: "Create ProfileWizard types in frontend/src/components/ProfileWizard/types.ts"
Task T017: "Create StepName.tsx"
Task T018: "Create StepSoftware.tsx"
Task T019: "Create StepHardware.tsx"

# Then main component that integrates them:
Task T020: "Create main ProfileWizard component" (depends on T016-T019)
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Profile Wizard)
4. Complete Phase 4: User Story 2 (Navbar Display Name)
5. **STOP and VALIDATE**: Test profile creation and navbar display
6. Deploy/demo - users can create profiles and see their name

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 + US2 → Test → Deploy (Profile MVP!)
3. Add US3 (Personalization) → Test → Deploy (Core AI feature!)
4. Add US4 (Translation) → Test → Deploy (Urdu support!)
5. Add US5 (Settings) → Test → Deploy (Profile management!)
6. Polish → Final release

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: US1 (Profile Wizard) + US2 (Navbar)
   - Developer B: US3 (Personalization) - can start backend while A does US1
   - Developer C: US4 (Translation) - fully independent
3. After P1/P2 complete:
   - Developer A: US5 (Profile Settings)
   - Developer B + C: Polish phase

---

## Summary

| Phase | Tasks | Parallel Opportunities |
|-------|-------|----------------------|
| Setup | T001-T003 | 2 tasks parallel |
| Foundational | T004-T011 | 4 tasks parallel |
| US1: Profile Wizard | T012-T023 | 5 tasks parallel (frontend steps) |
| US2: Navbar Display | T024-T026 | None (sequential) |
| US3: Personalization | T027-T039 | 2 tasks parallel |
| US4: Translation | T040-T052 | 2 tasks parallel |
| US5: Profile Settings | T053-T056 | None (sequential) |
| Polish | T057-T061 | 3 tasks parallel |

**Total Tasks**: 61
**Tasks per User Story**: US1=12, US2=3, US3=13, US4=13, US5=4, Setup=3, Foundational=8, Polish=5
**MVP Scope**: Phases 1-4 (Setup + Foundational + US1 + US2 = 26 tasks)
**Independent Test Criteria**: Each user story has clear independent test in its header

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- US3 requires profile_completed=true; US4 does not
- Both US3 and US4 share cache service and code extractor from Foundational phase
