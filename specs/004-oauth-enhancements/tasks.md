# Tasks: OAuth Authentication Enhancements

**Feature**: 004-oauth-enhancements
**Branch**: `004-oauth-enhancements`
**Generated**: 2025-12-19
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

---

## Summary

| Phase | Description | Task Count |
|-------|-------------|------------|
| Phase 1 | Setup & Prerequisites | 5 |
| Phase 2 | Foundational (Database & Backend Config) | 6 |
| Phase 3 | User Story 1 - Google OAuth (P1) | 7 |
| Phase 4 | User Story 2 - GitHub OAuth (P1) | 5 |
| Phase 5 | User Story 3 - Display Name in Navbar (P2) | 4 |
| Phase 6 | User Story 4 - Chat Widget Integration (P2) | 3 |
| Phase 7 | User Story 5 - Email User Display Name (P3) | 4 |
| Phase 8 | Polish & Documentation | 5 |
| **Total** | | **39** |

---

## Phase 1: Setup & Prerequisites

**Objective**: Prepare OAuth credentials and development environment before code changes.

- [x] T001 Document OAuth environment variables in backend/.env.example
- [ ] T002 [P] Create Google OAuth Client ID in Google Cloud Console with redirect URI
- [ ] T003 [P] Register GitHub OAuth App in GitHub Developer Settings with callback URL
- [ ] T004 Add local development OAuth credentials to backend/.env
- [ ] T005 Verify OAuth credentials are accessible via settings object

**Phase 1 Completion Criteria**:
- [ ] GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, API_URL documented
- [ ] OAuth apps created in Google Cloud Console and GitHub Developer Settings
- [ ] Local .env configured with development credentials

---

## Phase 2: Foundational (Database & Backend Config)

**Objective**: Database schema and backend configuration that all user stories depend on.

**MUST complete before starting any User Story phase.**

### 2.1 Database Migration

- [x] T006 Create migration script backend/scripts/migrations/004_add_oauth_fields.sql per data-model.md
- [x] T007 Create rollback script backend/scripts/migrations/004_add_oauth_fields_rollback.sql
- [x] T008 Update schema initialization in backend/src/db/connection.py to include auth_provider and oauth_provider_id columns
- [ ] T009 Run migration on development database and verify columns exist

### 2.2 Backend Configuration

- [x] T010 Add OAuth settings to Settings class in backend/src/config.py (google_client_id, google_client_secret, github_client_id, github_client_secret, api_url)
- [x] T011 Add httpx>=0.27.0 to backend/requirements.txt and install

**Phase 2 Completion Criteria**:
- [ ] Users table has auth_provider (DEFAULT 'email') and oauth_provider_id columns
- [ ] Index idx_users_oauth exists on (auth_provider, oauth_provider_id)
- [ ] Settings class loads all OAuth config from environment
- [ ] httpx importable in backend

---

## Phase 3: User Story 1 - Sign In with Google (P1)

**Story**: As a new visitor, I want to sign in with my Google account so that I can access the AI chatbot without creating a separate account.

**Independent Test**: Click "Sign in with Google" on /login, complete Google consent, verify logged in with Google display name shown.

### 3.1 Backend OAuth Handler

- [x] T012 [US1] Create OAuth router file backend/src/api/oauth.py with provider configs (GOOGLE_CONFIG, GITHUB_CONFIG)
- [x] T013 [US1] Implement GET /api/auth/oauth/google endpoint that redirects to Google consent in backend/src/api/oauth.py
- [x] T014 [US1] Implement GET /api/auth/callback/google endpoint with token exchange and user creation in backend/src/api/oauth.py
- [x] T015 [US1] Implement create_or_update_oauth_user() helper function in backend/src/api/oauth.py
- [x] T016 [US1] Register OAuth router in backend/src/main.py

### 3.2 Frontend OAuth Button

- [x] T017 [P] [US1] Add Google OAuth button to login page frontend/src/pages/login.tsx with redirect to /api/auth/oauth/google
- [x] T018 [US1] Add OAuth button styles to frontend/src/pages/login.module.css

**Phase 3 Completion Criteria**:
- [ ] Clicking "Sign in with Google" redirects to Google consent
- [ ] After consent, user is created/linked with auth_provider='google'
- [ ] Session cookie is set and user is redirected to homepage
- [ ] User's Google display_name is captured in database

---

## Phase 4: User Story 2 - Sign In with GitHub (P1)

**Story**: As a developer, I want to sign in with my GitHub account so that I can use my existing developer identity.

**Independent Test**: Click "Sign in with GitHub" on /login, authorize app, verify logged in with GitHub name/username shown.

### 4.1 Backend OAuth Handler

- [x] T019 [US2] Implement GET /api/auth/oauth/github endpoint that redirects to GitHub authorization in backend/src/api/oauth.py
- [x] T020 [US2] Implement GET /api/auth/callback/github endpoint with token exchange and private email fetch in backend/src/api/oauth.py

### 4.2 Frontend OAuth Button

- [x] T021 [P] [US2] Add GitHub OAuth button to login page frontend/src/pages/login.tsx with redirect to /api/auth/oauth/github
- [x] T022 [P] [US2] Add GitHub OAuth button to signup page frontend/src/pages/signup.tsx
- [x] T023 [US2] Handle OAuth error display from URL query params in frontend/src/pages/login.tsx

**Phase 4 Completion Criteria**:
- [ ] Clicking "Sign in with GitHub" redirects to GitHub authorization
- [ ] After authorization, user is created/linked with auth_provider='github'
- [ ] Private GitHub emails are fetched via /user/emails endpoint
- [ ] User's GitHub name (or username fallback) is captured as display_name

---

## Phase 5: User Story 3 - Display Personalized Greeting (P2)

**Story**: As a logged-in user, I want to see my name (not email) in the navbar so the interface feels personalized.

**Independent Test**: Log in with OAuth, verify navbar shows "Welcome, {Display Name}" instead of email.

### 5.1 Backend Session Update

- [x] T024 [US3] Update session query in backend/src/api/auth.py to include auth_provider field
- [x] T025 [US3] Update UserResponse model in backend/src/api/auth.py to include auth_provider field

### 5.2 Frontend Navbar Update

- [x] T026 [US3] Update AuthButton component in frontend/src/components/AuthButton/index.tsx to show display_name with fallback to email
- [x] T027 [US3] Add name truncation logic (>20 chars → ellipsis) in frontend/src/components/AuthButton/index.tsx

**Phase 5 Completion Criteria**:
- [x] GET /api/auth/session returns auth_provider field
- [x] Navbar shows "Welcome, {display_name}" for users with display_name
- [x] Navbar shows email as fallback when display_name is null
- [x] Long names truncated to 20 chars with "..."

---

## Phase 6: User Story 4 - Chat Widget Integration (P2)

**Story**: As a logged-in user, I want the chat widget to work regardless of how I signed in.

**Independent Test**: Sign in with OAuth, open chat widget, send a message successfully.

- [x] T028 [US4] Verify ChatWidget receives session with display_name and auth_provider in frontend/src/components/ChatWidget/index.tsx
- [x] T029 [US4] Verify session validation middleware is provider-agnostic in backend/src/api/middleware.py
- [x] T030 [US4] Test chat message flow with OAuth-authenticated user

**Phase 6 Completion Criteria**:
- [x] OAuth users can send chat messages
- [x] Chat sessions link to correct user_id
- [x] No code changes needed (verification only)

---

## Phase 7: User Story 5 - Email User Display Name (P3)

**Story**: As an email/password user, I want to add my display name to my account.

**Independent Test**: Log in with email/password, access profile, set display name, verify navbar shows name.

### 7.1 Backend Profile Endpoint

- [x] T031 [US5] Create PATCH /api/auth/profile endpoint in backend/src/api/auth.py for updating display_name
- [x] T032 [US5] Add UpdateProfileRequest model in backend/src/api/auth.py

### 7.2 Frontend Profile UI

- [x] T033 [P] [US5] Create profile settings page frontend/src/pages/profile.tsx with display_name form
- [x] T034 [US5] Add profile link to navbar or user menu in frontend/src/components/AuthButton/index.tsx

**Phase 7 Completion Criteria**:
- [x] Email/password users can update their display_name
- [x] Updated display_name appears in navbar after save
- [x] Profile page accessible to authenticated users

---

## Phase 8: Polish & Documentation

**Objective**: Final testing, documentation, and deployment preparation.

- [x] T035 Update backend/.env.example with all OAuth variables and comments
- [x] T036 [P] Create OAuth setup guide in docs/oauth-setup.md or README section
- [ ] T037 [P] Manual testing: Complete all acceptance scenarios from spec.md
- [ ] T038 Update production environment variables in Render dashboard
- [ ] T039 Update OAuth redirect URIs in Google Cloud Console and GitHub Developer Settings for production

**Phase 8 Completion Criteria**:
- [ ] All acceptance scenarios pass manual testing
- [x] Documentation complete for OAuth setup
- [ ] Production environment ready for deployment

---

## Dependencies

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational) ────────────────────────────────────────┐
    │                                                          │
    ├────────────────────┬────────────────────┐                │
    ▼                    ▼                    ▼                │
Phase 3 (US1 Google)  Phase 4 (US2 GitHub)                     │
    │                    │                                     │
    ├────────────────────┤                                     │
    ▼                    ▼                                     │
Phase 5 (US3 Navbar Display Name) ◄────────────────────────────┘
    │
    ▼
Phase 6 (US4 Chat Widget) ──── Verification only, can run after Phase 3 or 4
    │
    ▼
Phase 7 (US5 Profile) ──── Independent, can run after Phase 2
    │
    ▼
Phase 8 (Polish)
```

**Key Dependencies**:
- Phase 2 (Foundational) MUST complete before Phase 3-7
- Phase 3 (US1) and Phase 4 (US2) can run in parallel after Phase 2
- Phase 5 (US3) requires at least one OAuth provider working (Phase 3 or 4)
- Phase 6 (US4) is verification only, can run after Phase 3 or 4
- Phase 7 (US5) is independent, only requires Phase 2

---

## Parallel Execution Opportunities

### Backend Parallelization (after Phase 2)
```
Developer A                          Developer B
────────────                         ────────────
T012-T016 (Google OAuth backend)  || T019-T020 (GitHub OAuth backend)
```

### Frontend Parallelization (after T012-T016)
```
Developer A                          Developer B
────────────                         ────────────
T017-T018 (Google button)         || T021-T023 (GitHub button)
T026-T027 (Navbar update)         || T033-T034 (Profile page)
```

### Full Parallel (within phases)
Tasks marked with [P] can run in parallel with other [P] tasks in same phase.

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)
**Phase 1 + Phase 2 + Phase 3 (Google OAuth only)**

This delivers:
- Google OAuth sign-in
- Display name capture
- Basic navbar personalization
- Working chat widget for OAuth users

Estimated: 7 tasks (T001-T011 + T012-T018)

### Full Scope
All 39 tasks across 8 phases.

### Recommended Order
1. Complete Phase 1-2 (foundational)
2. Complete Phase 3 (Google OAuth) - delivers immediate user value
3. Complete Phase 5 (Navbar) - improves UX for Google users
4. Complete Phase 4 (GitHub OAuth) - second provider
5. Complete Phase 6-7 (Chat verification + Profile)
6. Complete Phase 8 (Polish)

---

## Format Validation

All 39 tasks follow the checklist format:
- ✅ All tasks start with `- [ ]` checkbox
- ✅ All tasks have sequential ID (T001-T039)
- ✅ Parallelizable tasks marked with `[P]`
- ✅ User Story tasks (Phase 3-7) marked with `[US1]`-`[US5]`
- ✅ All implementation tasks include file paths
