---
id: 0001
title: OAuth Authentication Enhancements Specification
stage: spec
date: 2025-12-19
surface: agent
model: claude-opus-4-5-20251101
feature: 004-oauth-enhancements
branch: 003-auth-infrastructure
user: architect
command: /sp.specify
labels: ["oauth", "authentication", "better-auth", "google", "github", "display-name"]
links:
  spec: specs/004-oauth-enhancements/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/004-oauth-enhancements/spec.md
tests:
  - null
---

## Prompt

/sp.specify

Generate an extended specification to add authentication enhancements to an existing Docusaurus + FastAPI RAG system using Better Auth.

Scope:
- Extend the current authentication system to support OAuth sign-in with Google and GitHub using Better Auth.
- Authentication applies to both the Docusaurus frontend and FastAPI backend.
- The system must capture and persist the user's display name at signup/signin time and use it as the primary identity shown in the UI (not email).

Frontend requirements:
- Integrate Google and GitHub sign-in buttons using Better Auth in the Docusaurus frontend.
- On successful authentication, store and access:
  - user_id
  - display_name (full name or provider name)
  - email (secondary, not shown in UI)
  - auth provider (google | github)
- Update the global navbar to display:
  - "Welcome, {User Name}" or avatar/name instead of email.
- Ensure authenticated state is shared across pages and the embedded chatbot.
- Handle loading, unauthenticated, and signed-out states cleanly.

Backend requirements:
- FastAPI backend must validate Better Auth sessions or tokens.
- Extend Neon Postgres schema to store:
  - user_id
  - display_name
  - email
  - auth_provider
  - created_at
- Ensure chat sessions and logs link to authenticated user_id.
- Authentication logic must be provider-agnostic (supports future providers).

Constraints:
- Google and GitHub OAuth only (email/password is optional or out of scope).
- User name must be captured at the earliest point (first successful login).
- Display name is the primary identifier across UI and logs.

Documentation:
- Use latest official Better Auth documentation.
- Reference FastAPI auth integration patterns and OAuth best practices.
- Align with existing RAG chatbot architecture.

Produce:
- Auth flow diagrams (Google & GitHub)
- Updated frontend auth integration points
- Backend auth validation flow
- Updated database schema
- Navbar personalization behavior
- Security and session considerations

## Response snapshot

Created comprehensive specification for OAuth authentication enhancements at `specs/004-oauth-enhancements/spec.md`.

Key deliverables:
1. **5 User Stories** with acceptance scenarios covering Google OAuth, GitHub OAuth, navbar personalization, chat widget integration, and email/password user display name support.

2. **34 Functional Requirements** covering:
   - OAuth provider configuration (FR-001 to FR-005)
   - Frontend OAuth integration (FR-006 to FR-011)
   - Display name management (FR-012 to FR-015)
   - Navbar personalization (FR-016 to FR-019)
   - Backend OAuth validation (FR-020 to FR-024)
   - Database schema extensions (FR-025 to FR-028)
   - Chat widget integration (FR-029 to FR-031)

3. **Auth Flow Diagrams** - ASCII sequence diagrams for both Google and GitHub OAuth flows showing the complete authentication lifecycle.

4. **Database Schema Updates**:
   - Added `auth_provider VARCHAR(20)` column
   - Added `oauth_provider_id VARCHAR(255)` column
   - Created index for OAuth lookups
   - Migration and rollback scripts included

5. **API Contracts** for OAuth callbacks and updated session endpoint.

6. **Implementation Code Examples**:
   - Updated login page with OAuth buttons
   - Updated AuthButton showing "Welcome, {Name}"
   - Complete OAuth callback handler for FastAPI

7. **Security Considerations** including CSRF protection, secret storage, and account linking strategy.

## Outcome

- Impact: Created foundational specification for OAuth enhancement feature
- Tests: N/A (specification phase)
- Files: 1 created (specs/004-oauth-enhancements/spec.md)
- Next prompts: `/sp.plan` to create implementation plan, `/sp.tasks` to generate actionable tasks
- Reflection: Spec builds on existing 003-auth-infrastructure, maintaining backwards compatibility with email/password auth while adding OAuth providers

## Evaluation notes (flywheel)

- Failure modes observed: None - specification successfully generated
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Validate Better Auth `signIn.social()` works with manual backend OAuth handlers (not using Better Auth server)
