# Implementation Plan: User Profiling, Content Personalization & Translation

**Branch**: `005-user-personalization` | **Date**: 2025-12-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-user-personalization/spec.md`

---

## Summary

Extend the existing Docusaurus + FastAPI RAG chatbot system with:
1. **User Profiling**: Capture software/hardware background during signup via a 3-step wizard
2. **Content Personalization**: AI-powered chapter adaptation using OpenAI Agents SDK
3. **Urdu Translation**: Preserve code blocks while translating prose to Urdu with RTL support
4. **Profile Management**: Settings page for profile updates, navbar display name

Technical approach uses OpenAI Agents SDK with dedicated Personalization and Translation agents, session-scoped caching, and audit logging to PostgreSQL.

---

## Technical Context

**Language/Version**: Python 3.11 (backend), TypeScript/JavaScript (frontend)
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, asyncpg, React, Docusaurus v3
**Storage**: Neon Serverless Postgres (existing), in-memory cache (cachetools)
**Testing**: pytest (backend), manual E2E testing (frontend)
**Target Platform**: Linux server (backend), Static site + browser (frontend)
**Project Type**: Web application (frontend + backend monorepo)
**Performance Goals**: <15s personalization, <30s translation, 50 concurrent users
**Constraints**: 10 req/min rate limit per user, 30s AI timeout, session-scoped cache only
**Scale/Scope**: ~100 concurrent users, 18 chapters, 2 AI agents

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Spec-Driven Development | PASS | Full spec created with 42 functional requirements, 10 success criteria |
| II. Technical Accuracy | PASS | AI guardrails preserve code blocks, factual accuracy mandated |
| III. Modularity & Reusability | PASS | Personalization/translation are independent, toggle-able features |
| IV. Docusaurus-First | PASS | Frontend components integrate with Docusaurus theme system |
| V. AI-Native Content Design | PASS | Content structured for AI consumption, RAG integration maintained |
| VI. Practical Application Focus | PASS | Real API contracts, runnable code examples in spec |

**Gate Status**: PASSED - Proceeding to Phase 0

---

## Project Structure

### Documentation (this feature)

```text
specs/005-user-personalization/
├── plan.md              # This file
├── research.md          # Phase 0 output - technology research
├── data-model.md        # Phase 1 output - entity schemas
├── quickstart.md        # Phase 1 output - developer setup guide
├── contracts/           # Phase 1 output - OpenAPI specs
│   ├── profile-api.yaml
│   ├── personalize-api.yaml
│   └── translate-api.yaml
├── checklists/
│   └── requirements.md  # Spec quality checklist (already created)
└── tasks.md             # Phase 2 output (/sp.tasks command)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/
│   │   ├── auth.py              # Existing - extend with profile_completed
│   │   ├── profile.py           # NEW - GET/POST /api/profile
│   │   ├── personalize.py       # NEW - POST /api/personalize
│   │   └── translate.py         # NEW - POST /api/translate
│   ├── agents/
│   │   ├── personalization.py   # NEW - Personalization Agent
│   │   └── translation.py       # NEW - Translation Agent
│   ├── db/
│   │   ├── connection.py        # Existing - add new tables
│   │   ├── models.py            # Existing - extend with profile models
│   │   └── queries.py           # Existing - add profile/log queries
│   ├── services/
│   │   └── cache.py             # NEW - Session-scoped caching
│   └── config.py                # Existing - add new env vars
├── scripts/
│   └── migrations/
│       └── 005_add_profile_fields.sql  # NEW - Schema migration
└── tests/
    ├── unit/
    │   ├── test_profile.py
    │   ├── test_personalization.py
    │   └── test_translation.py
    └── integration/
        └── test_profile_flow.py

frontend/
├── src/
│   ├── components/
│   │   ├── ProfileWizard/       # NEW - 3-step profile wizard
│   │   │   ├── index.tsx
│   │   │   ├── StepName.tsx
│   │   │   ├── StepSoftware.tsx
│   │   │   ├── StepHardware.tsx
│   │   │   └── types.ts
│   │   ├── ChapterActions/      # NEW - Personalize/Translate buttons
│   │   │   ├── index.tsx
│   │   │   └── styles.module.css
│   │   └── AuthButton/          # Existing - update for display_name
│   ├── pages/
│   │   └── profile.tsx          # NEW - Profile settings page
│   ├── lib/
│   │   ├── auth.ts              # Existing - extend with profile hooks
│   │   └── personalization.ts   # NEW - Personalization API client
│   ├── theme/
│   │   └── DocItem/             # NEW - Wrap doc pages with ChapterActions
│   └── css/
│       └── rtl.css              # NEW - RTL styling for Urdu
└── tests/
    └── e2e/
        └── profile.spec.ts
```

**Structure Decision**: Web application with existing backend/frontend separation. New code follows established patterns from 003-auth-infrastructure and 004-oauth-enhancements.

---

## Complexity Tracking

> No violations requiring justification. Design follows existing patterns.

---

## Phase 0: Research Findings

### R1: OpenAI Agents SDK for Personalization & Translation

**Decision**: Use OpenAI Agents SDK with dedicated agents for personalization and translation

**Rationale**:
- SDK provides structured agent composition with `Agent`, `Runner`, and `function_tool`
- Built-in guardrails via `@input_guardrail` and `@output_guardrail` decorators
- Supports async execution matching FastAPI async patterns
- Native Pydantic integration for structured outputs

**Alternatives Considered**:
- Raw OpenAI API: More control but requires manual prompt management
- LangChain: Heavier dependency, overkill for two focused agents
- Custom implementation: More maintenance burden

**Implementation Pattern**:
```python
from agents import Agent, Runner, function_tool
from pydantic import BaseModel

class PersonalizedContent(BaseModel):
    content: str
    adaptations_made: list[str]

personalization_agent = Agent(
    name="Personalization Agent",
    instructions="""Adapt the chapter content based on user's background.
    For beginners: Add foundational context, simpler analogies.
    For advanced: Skip basics, add optimization tips.
    NEVER change code blocks or factual information.""",
    output_type=PersonalizedContent,
)
```

### R2: Session-Scoped Caching Strategy

**Decision**: Use `cachetools.TTLCache` with user_id + chapter_id keys

**Rationale**:
- Lightweight, in-memory solution sufficient for <100 concurrent users
- TTL automatically expires stale entries
- No external service dependency (Redis not needed at this scale)
- Cache keys: `{user_id}:{chapter_id}:personalized` or `{user_id}:{chapter_id}:translated`

**Alternatives Considered**:
- Redis: Overkill for session-scoped, non-persistent cache
- Database: Too slow for frequent reads, defeats caching purpose
- Browser localStorage: Can't control TTL, security concerns

**Implementation Pattern**:
```python
from cachetools import TTLCache

# Max 1000 entries, 1 hour TTL
personalization_cache = TTLCache(maxsize=1000, ttl=3600)
translation_cache = TTLCache(maxsize=1000, ttl=3600)

def get_cache_key(user_id: str, chapter_id: str, content_type: str) -> str:
    return f"{user_id}:{chapter_id}:{content_type}"
```

### R3: RTL (Right-to-Left) Support for Urdu

**Decision**: CSS-based RTL with `dir="rtl"` attribute and Docusaurus CSS modules

**Rationale**:
- Docusaurus supports RTL through CSS
- Only translated content needs RTL, not entire page
- Inline `dir="rtl"` on translated content container

**Implementation Pattern**:
```css
/* frontend/src/css/rtl.css */
.translated-content-urdu {
  direction: rtl;
  text-align: right;
  font-family: 'Noto Nastaliq Urdu', serif;
}

.translated-content-urdu pre,
.translated-content-urdu code {
  direction: ltr;
  text-align: left;
}
```

### R4: Markdown Code Block Preservation

**Decision**: Use regex-based code block extraction before AI processing

**Rationale**:
- Extract code blocks before sending to AI
- Replace with placeholders
- Restore after translation/personalization
- Guarantees 100% code preservation

**Implementation Pattern**:
```python
import re

CODE_BLOCK_PATTERN = r'```[\s\S]*?```'
INLINE_CODE_PATTERN = r'`[^`]+`'

def extract_code_blocks(content: str) -> tuple[str, list[str]]:
    """Extract code blocks and replace with placeholders."""
    blocks = re.findall(CODE_BLOCK_PATTERN, content)
    for i, block in enumerate(blocks):
        content = content.replace(block, f"[CODE_BLOCK_{i}]", 1)
    return content, blocks

def restore_code_blocks(content: str, blocks: list[str]) -> str:
    """Restore code blocks from placeholders."""
    for i, block in enumerate(blocks):
        content = content.replace(f"[CODE_BLOCK_{i}]", block)
    return content
```

### R5: Rate Limiting Strategy

**Decision**: Per-user token bucket using existing middleware pattern

**Rationale**:
- 10 requests/minute per user matches AI cost constraints
- Reuse existing rate limiting from /api/chat endpoint
- Return 429 Too Many Requests with Retry-After header

**Implementation**: Extend `backend/src/api/middleware.py` with dedicated rate limiter for AI endpoints.

### R6: Profile Wizard UX Pattern

**Decision**: Modal-based 3-step wizard with skip option

**Rationale**:
- Modal doesn't interrupt navigation flow
- Progress indicator shows completion status
- Skip button with warning preserves user agency
- Persists partial progress on close

**Implementation Pattern**:
```typescript
// ProfileWizard state
interface WizardState {
  step: 1 | 2 | 3;
  displayName: string;
  softwareBackground: SoftwareBackground;
  hardwareBackground: HardwareBackground;
  isSkipped: boolean;
}
```

---

## Phase 1: Architecture Design

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Docusaurus)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  ProfileWizard│  │ChapterActions│  │  ProfilePage │  │  AuthButton  │   │
│  │   (Modal)     │  │  (DocItem)   │  │   (/profile) │  │   (Navbar)   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │                 │            │
│         └─────────────────┼─────────────────┼─────────────────┘            │
│                           │                 │                              │
│                    ┌──────┴─────────────────┴──────┐                       │
│                    │       usePersonalization()     │                       │
│                    │       useProfile()             │                       │
│                    └────────────────┬───────────────┘                       │
│                                     │ HTTP (fetch + credentials)            │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │
┌─────────────────────────────────────┼───────────────────────────────────────┐
│                              BACKEND (FastAPI)                              │
├─────────────────────────────────────┼───────────────────────────────────────┤
│                                     │                                       │
│  ┌──────────────────────────────────┴────────────────────────────────────┐ │
│  │                         API ROUTES                                     │ │
│  ├────────────────┬────────────────┬────────────────┬────────────────────┤ │
│  │ /api/profile   │ /api/personalize│ /api/translate │ /api/auth/session  │ │
│  │   GET/POST     │     POST        │     POST       │   GET (extended)   │ │
│  └───────┬────────┴───────┬─────────┴───────┬────────┴─────────┬─────────┘ │
│          │                │                 │                  │           │
│  ┌───────┴────────────────┴─────────────────┴──────────────────┴────────┐  │
│  │                        MIDDLEWARE                                     │  │
│  │  - Authentication (get_current_user)                                  │  │
│  │  - Rate Limiting (10 req/min for AI endpoints)                        │  │
│  │  - Request Logging                                                    │  │
│  └───────┬────────────────┬─────────────────┬───────────────────────────┘  │
│          │                │                 │                              │
│  ┌───────┴────────┐ ┌─────┴─────────┐ ┌─────┴─────────┐                    │
│  │   Profile      │ │Personalization│ │  Translation  │                    │
│  │   Service      │ │    Agent      │ │    Agent      │                    │
│  └───────┬────────┘ └───────┬───────┘ └───────┬───────┘                    │
│          │                  │                 │                            │
│  ┌───────┴──────────────────┴─────────────────┴───────────────────────┐    │
│  │                         SERVICES                                    │    │
│  │  - Cache (TTLCache)                                                 │    │
│  │  - Code Block Extractor                                             │    │
│  │  - Audit Logger                                                     │    │
│  └───────┬──────────────────┬─────────────────────────────────────────┘    │
│          │                  │                                              │
└──────────┼──────────────────┼──────────────────────────────────────────────┘
           │                  │
┌──────────┼──────────────────┼──────────────────────────────────────────────┐
│          │    EXTERNAL      │                                              │
│  ┌───────┴────────┐  ┌──────┴───────┐                                      │
│  │  Neon Postgres │  │  OpenAI API  │                                      │
│  │  - users       │  │  - GPT-4o    │                                      │
│  │  - *_logs      │  │              │                                      │
│  └────────────────┘  └──────────────┘                                      │
└────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Personalization Request

```
1. User clicks "Personalize Content" button
   │
2. ChapterActions component extracts chapter markdown content
   │
3. POST /api/personalize { chapter_id, content, title }
   │
4. Middleware: Validate session, check rate limit
   │
5. Check cache for existing personalization
   │  ├── HIT: Return cached content
   │  └── MISS: Continue
   │
6. Load user profile (software_background, hardware_background)
   │
7. Extract code blocks from content (replace with placeholders)
   │
8. Run Personalization Agent with:
   │  - Original content (code-free)
   │  - User profile
   │  - Chapter metadata
   │
9. Restore code blocks in personalized content
   │
10. Cache result (user_id:chapter_id:personalized)
    │
11. Log to personalization_logs table
    │
12. Return { personalized_content, metadata }
```

### Agent Implementation Design

#### Personalization Agent

```python
# backend/src/agents/personalization.py

from agents import Agent, Runner
from pydantic import BaseModel

class UserProfile(BaseModel):
    software_level: str  # beginner | intermediate | advanced
    software_languages: list[str]
    software_frameworks: list[str]
    hardware_level: str  # none | basic | intermediate | advanced
    hardware_domains: list[str]

class PersonalizedOutput(BaseModel):
    content: str
    adaptations_made: list[str]

PERSONALIZATION_INSTRUCTIONS = """
You are a content personalization agent for a Physical AI & Robotics textbook.

Your task is to adapt chapter content based on the user's technical background.

USER PROFILE:
- Software Experience: {software_level}
- Languages: {languages}
- Frameworks: {frameworks}
- Hardware Experience: {hardware_level}
- Hardware Domains: {domains}

ADAPTATION RULES:
1. For BEGINNERS:
   - Add foundational context before complex concepts
   - Use everyday analogies
   - Expand acronyms on first use
   - Add "Why this matters" explanations

2. For INTERMEDIATE users:
   - Balance theory with practical examples
   - Reference tools/frameworks they know
   - Add "Pro tips" for efficiency

3. For ADVANCED users:
   - Skip basic explanations
   - Add edge cases and optimization notes
   - Reference papers/specs where relevant
   - Include performance considerations

CRITICAL GUARDRAILS:
- NEVER change code examples (marked as [CODE_BLOCK_N])
- NEVER alter factual information
- NEVER remove safety warnings
- PRESERVE all markdown formatting
- MAINTAIN section structure and headers

OUTPUT: Return the adapted content with the same markdown structure.
"""

personalization_agent = Agent(
    name="Personalization Agent",
    instructions=PERSONALIZATION_INSTRUCTIONS,
    output_type=PersonalizedOutput,
    model="gpt-4o",
)

async def personalize_content(
    content: str,
    profile: UserProfile,
    chapter_title: str,
) -> PersonalizedOutput:
    """Personalize chapter content based on user profile."""

    # Format instructions with profile data
    formatted_instructions = PERSONALIZATION_INSTRUCTIONS.format(
        software_level=profile.software_level,
        languages=", ".join(profile.software_languages) or "None specified",
        frameworks=", ".join(profile.software_frameworks) or "None specified",
        hardware_level=profile.hardware_level,
        domains=", ".join(profile.hardware_domains) or "None specified",
    )

    agent = Agent(
        name="Personalization Agent",
        instructions=formatted_instructions,
        output_type=PersonalizedOutput,
        model="gpt-4o",
    )

    prompt = f"""
    Chapter: {chapter_title}

    Content to personalize:
    {content}
    """

    result = await Runner.run(agent, prompt)
    return result.final_output_as(PersonalizedOutput)
```

#### Translation Agent

```python
# backend/src/agents/translation.py

from agents import Agent, Runner
from pydantic import BaseModel

class TranslatedOutput(BaseModel):
    content: str
    preserved_blocks: int

TRANSLATION_INSTRUCTIONS = """
You are a technical translation agent specializing in Urdu translation.

Your task is to translate educational content about Physical AI & Robotics to Urdu.

TRANSLATION RULES:
1. Translate all prose text to Urdu
2. PRESERVE code blocks exactly as-is (marked as [CODE_BLOCK_N])
3. PRESERVE inline code exactly as-is
4. Keep technical terms in English with Urdu transliteration in parentheses
   Example: "API (اے پی آئی)"
5. PRESERVE all URLs, file paths, and command examples
6. MAINTAIN markdown formatting (headers, lists, emphasis)

TECHNICAL TERM HANDLING:
- GPU → GPU (جی پی یو)
- Neural Network → Neural Network (نیورل نیٹ ورک)
- Machine Learning → Machine Learning (مشین لرننگ)
- API → API (اے پی آئی)
- Framework → Framework (فریم ورک)

OUTPUT FORMAT:
- Use proper Urdu script
- Maintain paragraph structure
- Keep code placeholders intact

Translate the following content to Urdu:
"""

translation_agent = Agent(
    name="Translation Agent",
    instructions=TRANSLATION_INSTRUCTIONS,
    output_type=TranslatedOutput,
    model="gpt-4o",
)

async def translate_content(
    content: str,
    target_language: str = "ur",
) -> TranslatedOutput:
    """Translate chapter content to Urdu."""

    result = await Runner.run(translation_agent, content)
    return result.final_output_as(TranslatedOutput)
```

---

## API Design Summary

| Endpoint | Method | Auth | Rate Limit | Purpose |
|----------|--------|------|------------|---------|
| `/api/profile` | GET | Required | - | Get current user's profile |
| `/api/profile` | POST | Required | - | Create/update profile |
| `/api/personalize` | POST | Required | 10/min | Generate personalized content |
| `/api/translate` | POST | Required | 10/min | Translate content to Urdu |
| `/api/auth/session` | GET | Required | - | Extended with profile_completed |

See `contracts/` directory for full OpenAPI specifications.

---

## Database Changes Summary

### New Columns on `users` Table

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| software_background | JSONB | NULL | User's software experience |
| hardware_background | JSONB | NULL | User's hardware experience |
| profile_completed | BOOLEAN | FALSE | Profile wizard completion flag |

### New Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| personalization_logs | Audit personalization requests | user_id, chapter_id, status, response_time_ms |
| translation_logs | Audit translation requests | user_id, chapter_id, target_language, status |

See `data-model.md` for complete schema definitions.

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AI timeout (>30s) | Medium | High | Timeout handling, fallback to original content |
| AI cost overrun | Medium | Medium | Rate limiting, caching, monitoring |
| Code block corruption | Low | High | Extraction/restoration with validation |
| Profile wizard abandonment | Medium | Low | Persist partial progress, re-prompt on login |
| RTL rendering issues | Low | Medium | CSS isolation, browser testing |

---

## Security Considerations

1. **Input Sanitization**: Profile inputs sanitized before DB storage
2. **Prompt Injection Prevention**: User data escaped in AI prompts
3. **Rate Limiting**: 10 req/min per user for AI endpoints
4. **Audit Trail**: All AI requests logged with user_id
5. **Authentication**: All endpoints require valid session

---

## Testing Strategy

### Unit Tests
- Profile CRUD operations
- Code block extraction/restoration
- Cache key generation
- Rate limiter logic

### Integration Tests
- Full personalization flow (mock AI)
- Full translation flow (mock AI)
- Profile wizard completion

### E2E Tests (Manual)
- New user profile flow
- Personalization toggle
- Translation toggle with RTL
- Profile settings update

---

## Deployment Order

1. **Database Migration**: Add columns and tables
2. **Backend Deployment**:
   - Profile API endpoints
   - Personalization/Translation agents
   - Caching service
3. **Frontend Deployment**:
   - ProfileWizard component
   - ChapterActions component
   - Profile settings page
   - RTL CSS
4. **Monitoring Setup**: AI cost tracking, error rate alerts

---

## Next Steps

1. Run `/sp.tasks` to generate implementation tasks from this plan
2. Execute database migration
3. Implement backend endpoints in order: profile → personalize → translate
4. Implement frontend components in order: ProfileWizard → ChapterActions → ProfilePage
5. Integration testing
6. Deployment and monitoring

---

## Constitution Re-Check (Post-Design)

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Spec-Driven Development | PASS | Plan derived from spec, tasks will follow |
| II. Technical Accuracy | PASS | Code preservation guaranteed by extraction pattern |
| III. Modularity & Reusability | PASS | Agents are independent, reusable components |
| IV. Docusaurus-First | PASS | Components integrate with theme system |
| V. AI-Native Content Design | PASS | Structured prompts, Pydantic outputs |
| VI. Practical Application Focus | PASS | Concrete implementation patterns throughout |

**Final Gate Status**: PASSED - Ready for /sp.tasks
