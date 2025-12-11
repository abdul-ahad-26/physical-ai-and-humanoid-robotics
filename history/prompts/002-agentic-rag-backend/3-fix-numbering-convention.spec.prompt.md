---
id: 3
title: "fix-numbering-convention"
stage: "spec"
date: "2025-12-10"
surface: "agent"
model: "Claude Sonnet 4.5"
feature: "agentic-rag-backend"
branch: "002-agentic-rag-backend"
user: "user"
command: "/sp.clarify"
labels: ["correction", "naming", "numbering", "structure"]
links:
  spec: "specs/002-agentic-rag-backend/spec.md"
  ticket: null
  adr: null
  pr: null
files:
  - "specs/002-agentic-rag-backend/spec.md"
  - "history/prompts/002-agentic-rag-backend/1-specify-rag-agentic-backend.spec.prompt.md"
  - "history/prompts/002-agentic-rag-backend/2-clarify-agentic-backend-spec.spec.prompt.md"
tests: []
---

# Correction: Naming and Numbering Convention Fix

## User Input (PROMPT_TEXT)
— In the specs and prompt-history directories, you created multiple folders with different names but the same meaning (for example: 001-agentic-rag-backend, ai-textbook, and 001-ai-textbook).

In the specs directory specifically, you created a folder named 001-agentic-rag-backend. I need clarification on the numbering system you're using because there are two different folders labeled with 001.

Please explain the naming and numbering convention you're following, and fix any incorrect or duplicated folder names if something was created inconsistently.

## Response Summary (RESPONSE_TEXT)
Identified and fixed a numbering convention issue where the system incorrectly assigned the same number "001" to different features. The problem was:

1. `001-ai-textbook` existed with actual content in specs/ but was empty in history/prompts/001-ai-textbook/
2. The actual content for the ai-textbook feature was in the unnumbered directory history/prompts/ai-textbook/
3. `001-agentic-rag-backend` was incorrectly created with the same number instead of "002"

Fixed by:
1. Moving content from history/prompts/ai-textbook/ to history/prompts/001-ai-textbook/
2. Renaming the agentic backend directories from 001 to 002: specs/001-agentic-rag-backend → specs/002-agentic-rag-backend
3. Renaming history/prompts/001-agentic-rag-backend → history/prompts/002-agentic-rag-backend
4. Updating the git branch from 001-agentic-rag-backend to 002-agentic-rag-backend
5. Updating the spec file to reflect the correct branch name

## Outcome
The naming and numbering convention is now consistent:
- 001-ai-textbook: Original textbook project (specs/001-ai-textbook, history/prompts/001-ai-textbook)
- 002-agentic-rag-backend: New agentic backend project (specs/002-agentic-rag-backend, history/prompts/002-agentic-rag-backend)
- Git branch correctly named 002-agentic-rag-backend