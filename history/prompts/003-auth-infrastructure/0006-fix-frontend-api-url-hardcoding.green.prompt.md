---
id: 0006
title: Fix Frontend API URL Hardcoding
stage: green
date: 2025-12-19
surface: agent
model: claude-opus-4-5-20251101
feature: 003-auth-infrastructure
branch: 003-auth-infrastructure
user: abdul-ahad-26
command: /sp.implement
labels: ["deployment", "vercel", "api-url", "configuration", "fix"]
links:
  spec: specs/003-auth-infrastructure/spec.md
  ticket: null
  adr: null
  pr: https://github.com/abdul-ahad-26/physical-ai-and-humanoid-robotics/pull/6
files:
  - frontend/src/lib/config.ts
  - frontend/src/components/ChatWidget/index.tsx
  - frontend/src/components/ChatWidget/AuthGate.tsx
  - frontend/src/lib/auth.ts
tests:
  - Manual verification of no remaining hardcoded localhost URLs in src/
---

## Prompt

in frontend i think you have hardcoded local host 3000. so even if i put the the production backend url in the deployment env of frontend on vercel it take localhoast 3000.

## Response snapshot

Fixed hardcoded localhost URLs that prevented Vercel deployment from using the `API_URL` environment variable:

1. Created `src/lib/config.ts` with `getApiUrl()` utility that reads from Docusaurus `siteConfig.customFields.apiUrl`
2. Updated `ChatWidget/index.tsx` to use `getApiUrl()` instead of `window.__API_URL__ || 'http://localhost:8000'`
3. Fixed `AuthGate.tsx` which had wrong hardcoded URL (`localhost:3000` instead of backend `localhost:8000`)
4. Updated `auth.ts` to use the shared config utility

The `API_URL` set in Vercel environment variables will now be properly used at build time via Docusaurus customFields.

## Outcome

- ✅ Impact: Frontend now correctly uses API_URL from Vercel environment variables at build time
- 🧪 Tests: Verified no hardcoded localhost URLs remain in src/ (only fallback in config.ts)
- 📁 Files: 4 files changed (1 new config.ts, 3 updated to use it)
- 🔁 Next prompts: Redeploy Vercel and verify backend connection works
- 🧠 Reflection: The issue was multiple files had their own hardcoded fallbacks; centralizing in config.ts ensures consistency

## Evaluation notes (flywheel)

- Failure modes observed: AuthGate.tsx had localhost:3000 (frontend URL) instead of backend URL - a copy-paste error
- Graders run and results (PASS/FAIL): N/A - deployment fix
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Consider using Docusaurus useDocusaurusContext hook for reactive config access
