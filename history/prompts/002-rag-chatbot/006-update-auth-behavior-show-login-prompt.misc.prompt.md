---
id: 006
title: Update Auth Behavior Show Login Prompt
stage: misc
date: 2025-12-17
surface: agent
model: claude-opus-4-5-20251101
feature: 002-rag-chatbot
branch: 002-rag-chatbot
user: abdul-ahad-26
command: manual
labels: ["ux-improvement", "authentication", "spec-update"]
links:
  spec: specs/002-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/002-rag-chatbot/spec.md
  - specs/002-rag-chatbot/plan.md
  - specs/002-rag-chatbot/tasks.md
  - frontend/src/components/ChatWidget/index.tsx
  - frontend/src/theme/Root.tsx
tests:
  - Manual verification: icon visible to all users
  - Manual verification: login prompt shown when unauthenticated user clicks
---

## Prompt

after all configuration unauthenticated user even canot see there is a chatbot icon??

yes change.start change from the spec till implement

## Response snapshot

Updated FR-010 behavior from "hide widget for unauthenticated users" to "show icon to all users, display login prompt when clicked if unauthenticated".

**Changes made following full SDD cycle:**

1. **Spec Update (spec.md)**:
   - Changed FR-010: "System MUST show the chat widget icon to all users, but display a login prompt when unauthenticated users attempt to use it."

2. **Plan Update (plan.md)**:
   - Task 5.4: "Implement Better Auth session check (show login prompt for unauthenticated)"
   - Acceptance: "ChatWidget icon visible to all users, login prompt shown for unauthenticated"

3. **Tasks Update (tasks.md)**:
   - Added T073: "Update ChatWidget to show icon to all users but display login prompt when unauthenticated user clicks it per FR-010"

4. **Implementation (ChatWidget/index.tsx)**:
   - Removed `if (!isAuthenticated && !session) return null;` check
   - Added `isLoggedIn` state check
   - Added login prompt panel with:
     - User icon
     - "Login Required" heading
     - "Log In" button linking to /login
     - "Sign up" link
   - Chat panel only renders when `isOpen && isLoggedIn`

5. **Integration (Root.tsx)**:
   - Enabled ChatWidget with `isAuthenticated={false}` to trigger login prompt for testing
   - Added TODO comment for Better Auth integration

**Behavior:**
- All users see green chat icon in bottom-right
- Unauthenticated users clicking icon see login prompt with "Log In" button
- Authenticated users (after Better Auth setup) see full chat interface

## Outcome

- ✅ Impact: Improved UX - users can discover chatbot exists before logging in
- 🧪 Tests: Manual testing required after Better Auth setup
- 📁 Files: 5 files updated following spec→plan→tasks→implementation cycle
- 🔁 Next prompts: Set up Better Auth (feature 003-auth-infrastructure)
- 🧠 Reflection: Complete SDD cycle executed for UX improvement; login prompt provides clear call-to-action for unauthenticated users

## Evaluation notes (flywheel)

- Failure modes observed: None - straightforward UX change
- Graders run and results (PASS/FAIL): Manual verification pending
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): Test with Better Auth configured to verify login flow works
