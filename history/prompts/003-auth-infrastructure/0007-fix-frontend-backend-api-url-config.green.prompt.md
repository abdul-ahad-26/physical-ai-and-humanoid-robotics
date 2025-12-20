---
id: 0007
title: Fix Frontend Backend API URL Config
stage: green
date: 2025-12-19
surface: agent
model: claude-opus-4-5-20251101
feature: 003-auth-infrastructure
branch: 003-auth-infrastructure
user: abdul-ahad-26
command: /sp.implement
labels: ["deployment", "cors", "api-url", "vercel", "render", "cookies", "cross-domain"]
links:
  spec: specs/003-auth-infrastructure/spec.md
  ticket: null
  adr: null
  pr: https://github.com/abdul-ahad-26/physical-ai-and-humanoid-robotics/pull/6
files:
  - frontend/src/lib/auth.ts
  - frontend/src/lib/config.ts
  - frontend/src/components/ChatWidget/index.tsx
  - frontend/src/components/ChatWidget/AuthGate.tsx
  - frontend/src/pages/login.tsx
  - frontend/src/pages/signup.tsx
  - frontend/src/components/AuthButton/index.tsx
  - backend/src/main.py
  - backend/src/api/auth.py
  - backend/render.yaml
tests:
  - Manual browser testing of sign-in/sign-up/chat on production
---

## Prompt

Multiple deployment issues with frontend-backend communication:
1. Frontend hardcoded localhost:8000/localhost:3000 URLs not respecting Vercel API_URL env var
2. Auth client initialized at module load time before Docusaurus context available
3. CORS errors when frontend (Vercel) calls backend (Render)
4. Cross-domain cookies not working with samesite="lax"
5. Python 3.11.0 typing incompatibility with openai-agents package
6. Double slash in API URL (trailing slash in env var)

## Response snapshot

Fixed all frontend-backend communication issues through multiple commits:
1. Created useDocusaurusContext-based hooks for auth (useSignIn, useSignUp, useSignOut, useSession)
2. Updated ChatWidget to use useDocusaurusContext directly for API URL
3. Fixed backend CORS to allow Vercel origin and expose Set-Cookie header
4. Changed cookie samesite from "lax" to "none" for cross-domain auth
5. Upgraded Python version to 3.12 in render.yaml
6. Instructed user to remove trailing slash from API_URL env var

## Outcome

- ✅ Impact: Sign-in and sign-up now work in production; chat widget pending trailing slash fix
- 🧪 Tests: Manual browser testing on production Vercel deployment
- 📁 Files: 10 files modified across frontend and backend
- 🔁 Next prompts: Verify chat works after removing trailing slash from API_URL
- 🧠 Reflection: Module-level initialization in SSR/SSG frameworks must be avoided; use hooks for runtime config access

## Evaluation notes (flywheel)

- Failure modes observed: getApiUrl() called at module load time returned fallback; Docusaurus customFields only available after hydration via hooks
- Graders run and results (PASS/FAIL): Manual testing - auth PASS, chat PENDING
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Add URL sanitization to strip trailing slashes automatically
