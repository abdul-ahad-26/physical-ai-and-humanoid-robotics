---
id: 0003
title: OAuth Implementation Planning Session
stage: plan
date: 2025-12-19
surface: agent
model: claude-opus-4-5-20251101
feature: 004-oauth-enhancements
branch: 003-auth-infrastructure
user: architect
command: /sp.plan
labels: ["planning", "oauth", "implementation", "architecture", "better-auth"]
links:
  spec: specs/004-oauth-enhancements/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/004-oauth-enhancements/plan.md
tests:
  - null
---

## Prompt

/sp.plan

Create a comprehensive implementation plan for adding OAuth authentication enhancements to the existing Docusaurus + FastAPI RAG system.

Requirements:
1. Review current authentication setup (Better Auth React client + custom FastAPI backend)
2. Enable Google and GitHub OAuth following Better Auth patterns
3. Update frontend: OAuth buttons on login page, display name in navbar
4. Update backend: OAuth callback handlers, profile fetching, session creation
5. Database migration: Add auth_provider and oauth_provider_id columns
6. Ensure session consistency across frontend and chat widget
7. Security: CSRF protection (Better Auth state), account linking, timeout handling
8. Testing checklist: unit tests, integration tests, manual OAuth flow testing
9. Documentation: Update README, environment variables, deployment notes
10. Extensibility: Design for future OAuth providers

## Response snapshot

Created comprehensive 11-phase implementation plan at `specs/004-oauth-enhancements/plan.md`.

**Architecture Decision**: Backend implements OAuth manually (not Better Auth server library), so frontend will redirect directly to backend OAuth initiation endpoints (`/api/auth/oauth/{provider}`) rather than using `signIn.social()`.

**Phases:**
- Phase 0: Prerequisites (OAuth credentials, environment setup)
- Phase 1: Database Migration (auth_provider, oauth_provider_id columns, indexes)
- Phase 2: Backend Configuration (Settings class, OAuth endpoints)
- Phase 3: Backend OAuth Handlers (initiate, callback, token exchange, profile fetch)
- Phase 4: Session Endpoint Update (include auth_provider in response)
- Phase 5: Frontend Google OAuth Button
- Phase 6: Frontend GitHub OAuth Button
- Phase 7: Navbar Display Name (show user name instead of email)
- Phase 8: Chat Widget Verification
- Phase 9: Testing (unit, integration, E2E)
- Phase 10: Documentation & Deployment

**Data Flow:**
```
Frontend OAuth button click
    ↓
window.location = /api/auth/oauth/{provider}
    ↓
Backend generates OAuth URL, redirects to provider
    ↓
User consents, provider redirects to /api/auth/callback/{provider}
    ↓
Backend exchanges code for token, fetches profile
    ↓
create_or_update_oauth_user() - creates/links user, creates session
    ↓
Redirect to frontend with session cookie
```

## Outcome

- ✅ Impact: Created detailed 11-phase implementation plan with architecture diagram, risk assessment, and extensibility notes
- 🧪 Tests: N/A (planning phase - testing included in Phase 9)
- 📁 Files: 1 created (specs/004-oauth-enhancements/plan.md)
- 🔁 Next prompts: `/sp.tasks` to generate actionable implementation tasks
- 🧠 Reflection: Key insight was recognizing backend uses custom OAuth handlers (not Better Auth server), so frontend must redirect to backend initiation endpoint rather than using signIn.social()

## Evaluation notes (flywheel)

- Failure modes observed: None - plan generated successfully with all phases detailed
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Validate httpx OAuth token exchange with Google/GitHub in Phase 3
