# Feature Specification: User Profiling, Content Personalization & Translation

**Feature Branch**: `005-user-personalization`
**Created**: 2025-12-23
**Status**: Draft
**Input**: User description: "Add user profiling, content personalization, and chapter-level translation features for logged-in users. Extend user model with display_name, capture software/hardware background during signup, personalize chapter content using AI, and provide Urdu translation."

---

## Overview

This specification extends the existing authentication system (003-auth-infrastructure, 004-oauth-enhancements) to add:

1. **User Profiling**: Capture display name and technical background during signup/first login
2. **Content Personalization**: AI-powered chapter content adaptation based on user background
3. **Chapter Translation**: Urdu translation with preserved technical accuracy
4. **Navbar Personalization**: Display user's name (not email) in the global navbar

### Current State

The existing system provides:
- Email/password and OAuth (Google, GitHub) authentication via Better Auth
- Cookie-based session management with 7-day expiration
- `users` table with `display_name`, `auth_provider`, `oauth_provider_id` columns
- AuthButton component showing user name/email in navbar
- RAG chatbot with vector search over textbook content

### Target State

After this feature:
- New users complete a technical background profile during signup or first login
- Users can personalize any chapter's content based on their stored profile
- Users can translate any chapter to Urdu with technical terms preserved
- Both personalization and translation are reversible (toggle on/off)
- Original content remains source of truth; personalized/translated content is generated dynamically

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Complete Technical Background Profile (Priority: P1)

As a new user signing up or logging in for the first time, I want to answer questions about my technical background so that the textbook can adapt its content to my knowledge level.

**Why this priority**: Foundation for all personalization features. Without user profile data, personalization cannot work. This unlocks all other user stories.

**Independent Test**: Can be fully tested by creating a new account, completing the profile wizard, and verifying data is stored in the database.

**Acceptance Scenarios**:

1. **Given** a new user completes signup (email or OAuth), **When** they are redirected to the homepage for the first time, **Then** a profile completion modal/page appears prompting them to complete their technical background.

2. **Given** a user is presented with the profile wizard, **When** they select their software background level (beginner/intermediate/advanced), languages, and frameworks, **Then** these selections are stored in the database linked to their user account.

3. **Given** a user is presented with the profile wizard, **When** they select their hardware background (no hardware experience, basic electronics, robotics kits, GPUs/accelerators, Jetson/edge devices), **Then** these selections are stored in the database linked to their user account.

4. **Given** a user has already completed their profile, **When** they log in again, **Then** the profile wizard does NOT appear again (profile_completed flag is true).

5. **Given** a user wants to update their profile, **When** they access their profile settings page, **Then** they can modify their technical background and save changes.

---

### User Story 2 - Display User Name in Navbar (Priority: P1)

As a logged-in user, I want to see my display name (not email) in the navbar so that the interface feels personalized.

**Why this priority**: Essential UX improvement that builds user trust and confirms identity. Required before personalization features make sense.

**Independent Test**: Can be tested by logging in and verifying the navbar shows the user's display name.

**Acceptance Scenarios**:

1. **Given** a logged-in user with a display name set, **When** they view any page, **Then** the navbar displays "Welcome, {Display Name}" (e.g., "Welcome, Sarah").

2. **Given** a logged-in user without a display name, **When** they view any page, **Then** the navbar falls back to showing their email address.

3. **Given** a user updates their display name in profile settings, **When** they refresh or navigate, **Then** the navbar immediately reflects the updated name.

---

### User Story 3 - Personalize Chapter Content (Priority: P2)

As a logged-in user reading a chapter, I want to press a "Personalize Content" button to adapt the explanations, examples, and emphasis to my technical background so that I can understand concepts more easily.

**Why this priority**: Core value proposition of this feature set. Requires profile data (P1) to function. High impact on learning outcomes.

**Independent Test**: Can be tested by logging in with a completed profile, navigating to a chapter, clicking "Personalize Content", and verifying the content changes to match user background.

**Acceptance Scenarios**:

1. **Given** a logged-in user with a completed profile views a chapter, **When** they click "Personalize Content", **Then** the chapter content is replaced with a personalized version tailored to their software and hardware background.

2. **Given** a beginner user clicks "Personalize Content" on a chapter with advanced concepts, **When** the personalized content loads, **Then** it includes more foundational explanations, analogies, and simplified examples.

3. **Given** an advanced user clicks "Personalize Content", **When** the personalized content loads, **Then** it skips basic explanations and dives deeper into technical details.

4. **Given** a user has personalized content displayed, **When** they click "Show Original", **Then** the original chapter content is restored immediately.

5. **Given** a user navigates away from a personalized chapter, **When** they return to the same chapter, **Then** they see the original content (personalization is not persisted).

6. **Given** a user's personalization request is processing, **When** waiting for the AI response, **Then** they see a loading indicator and cannot trigger another request until complete.

---

### User Story 4 - Translate Chapter to Urdu (Priority: P2)

As a logged-in user who prefers to read in Urdu, I want to press a "Translate to Urdu" button to see the chapter content translated while preserving technical accuracy and code examples.

**Why this priority**: Expands accessibility to Urdu-speaking audience. Same technical infrastructure as personalization. Important for inclusivity.

**Independent Test**: Can be tested by logging in, navigating to a chapter, clicking "Translate to Urdu", and verifying content is translated with code blocks unchanged.

**Acceptance Scenarios**:

1. **Given** a logged-in user views a chapter, **When** they click "Translate to Urdu", **Then** the prose content is translated to Urdu while code blocks remain in English.

2. **Given** a chapter contains technical terms (e.g., "API", "GPU", "tensor"), **When** translated to Urdu, **Then** these terms are either kept in English or include transliteration with English in parentheses.

3. **Given** a chapter contains shell commands or code snippets, **When** translated, **Then** the commands/code remain exactly as-is (unchanged).

4. **Given** a user has translated content displayed, **When** they click "Show English", **Then** the original English content is restored.

5. **Given** a user's translation request is processing, **When** waiting for the AI response, **Then** they see a loading indicator and cannot trigger another translation until complete.

---

### User Story 5 - Update Profile Settings (Priority: P3)

As a returning user, I want to update my display name and technical background from a profile settings page so that my personalization stays relevant as I learn.

**Why this priority**: Important for long-term engagement but not blocking core personalization. Users may outgrow their initial profile.

**Independent Test**: Can be tested by logging in, accessing profile settings, modifying fields, saving, and verifying changes persist.

**Acceptance Scenarios**:

1. **Given** a logged-in user, **When** they navigate to /profile, **Then** they see their current display name, software background, and hardware background.

2. **Given** a user modifies their profile settings, **When** they click "Save Changes", **Then** the updates are persisted to the database and a success message is displayed.

3. **Given** a user changes their background from "beginner" to "advanced", **When** they personalize a chapter, **Then** the personalization uses the updated background data.

---

### Edge Cases

- **Incomplete Profile**: User closes profile wizard without completing. System tracks partial state; next login re-prompts for completion.
- **Long Display Name**: Display name exceeds 100 characters. System truncates to 100 characters on save.
- **Personalization Failure**: AI service unavailable or times out (>30 seconds). System displays error message and keeps original content visible.
- **Translation Failure**: AI service unavailable. System displays error message "Translation unavailable. Please try again later."
- **Empty Chapter Content**: Chapter has minimal content (<100 characters). Personalization/translation still attempts but may produce minimal changes.
- **Rapid Toggle**: User rapidly toggles personalization on/off. System debounces requests (500ms minimum between requests).
- **Session Expiration**: User's session expires while personalized content is displayed. Next API call returns 401; user is prompted to log in again.
- **Profile Not Complete**: User tries to personalize without completing profile. System prompts them to complete their profile first.
- **RTL Layout**: Urdu translation requires right-to-left text direction. System applies RTL styling to translated content container.
- **Mixed Content**: Chapter has both code and prose. System correctly identifies and preserves code blocks during translation.

---

## Requirements *(mandatory)*

### Functional Requirements

#### User Profile Extension

- **FR-001**: System MUST extend the `users` table with `software_background` (JSONB) and `hardware_background` (JSONB) columns.
- **FR-002**: System MUST add `profile_completed` (BOOLEAN, default FALSE) column to track whether user has completed initial profiling.
- **FR-003**: System MUST capture display name during signup if not provided by OAuth (prompt in profile wizard).
- **FR-004**: Software background MUST include: experience level (beginner/intermediate/advanced), known languages (list), known frameworks (list).
- **FR-005**: Hardware background MUST include: experience level (none/basic/intermediate/advanced), domains (list: electronics, robotics, GPUs, Jetson, embedded systems).

#### Profile Wizard

- **FR-006**: System MUST display a profile completion wizard for users with `profile_completed=FALSE` after authentication.
- **FR-007**: Profile wizard MUST be skippable but system MUST remind users to complete it on subsequent logins until completed.
- **FR-008**: Profile wizard MUST support step-by-step completion: Step 1 - Display Name, Step 2 - Software Background, Step 3 - Hardware Background.
- **FR-009**: Profile wizard MUST validate that at least experience level is selected for both software and hardware sections before marking complete.
- **FR-010**: System MUST persist partial profile progress if user leaves wizard mid-completion.

#### Navbar Display Name

- **FR-011**: System MUST display `display_name` in navbar greeting when available.
- **FR-012**: System MUST fall back to email when `display_name` is null or empty.
- **FR-013**: Navbar greeting MUST be truncated with ellipsis if name exceeds 20 characters.

#### Content Personalization

- **FR-014**: System MUST provide a "Personalize Content" button at the top of each chapter for logged-in users with completed profiles.
- **FR-015**: Personalization MUST use OpenAI Agents SDK to generate adapted content based on user's stored profile.
- **FR-016**: Personalized content MUST preserve all factual information, code examples, and technical accuracy.
- **FR-017**: Personalized content MUST adapt: explanation depth, analogy complexity, example relevance, and assumed prerequisites.
- **FR-018**: System MUST provide a "Show Original" button to instantly revert to original content.
- **FR-019**: Personalization MUST NOT permanently modify the source markdown files.
- **FR-020**: System MUST display a loading state during personalization generation.
- **FR-021**: System MUST cache personalized content per user per chapter for the session duration.
- **FR-022**: Personalization requests MUST be auditable (logged with user_id, chapter_id, timestamp).

#### Translation (Urdu)

- **FR-023**: System MUST provide a "Translate to Urdu" button at the top of each chapter for logged-in users.
- **FR-024**: Translation MUST use OpenAI Agents SDK to translate prose content to Urdu.
- **FR-025**: Translation MUST preserve: code blocks, shell commands, technical terms, URLs, and file paths unchanged.
- **FR-026**: Translation MUST apply RTL (right-to-left) text direction to translated content.
- **FR-027**: System MUST provide a "Show English" button to revert to original language.
- **FR-028**: Translated content MUST be cached per user per chapter for the session duration.
- **FR-029**: Translation requests MUST be auditable (logged with user_id, chapter_id, language, timestamp).

#### Profile Settings Page

- **FR-030**: System MUST provide a /profile page accessible to logged-in users.
- **FR-031**: Profile page MUST allow editing: display_name, software_background, hardware_background.
- **FR-032**: Profile page MUST show current values and provide form inputs for modification.
- **FR-033**: System MUST validate and sanitize all profile inputs before persistence.

#### API & Backend

- **FR-034**: Backend MUST provide `POST /api/profile` endpoint to create/update user profile.
- **FR-035**: Backend MUST provide `GET /api/profile` endpoint to retrieve current user's profile.
- **FR-036**: Backend MUST provide `POST /api/personalize` endpoint accepting chapter content and returning personalized version.
- **FR-037**: Backend MUST provide `POST /api/translate` endpoint accepting content and target language, returning translated version.
- **FR-038**: All personalization/translation endpoints MUST require authentication (401 for unauthenticated requests).
- **FR-039**: All personalization/translation requests MUST be rate-limited (10 requests per minute per user).

#### Audit & Logging

- **FR-040**: System MUST log all personalization requests to `personalization_logs` table.
- **FR-041**: System MUST log all translation requests to `translation_logs` table.
- **FR-042**: Logs MUST include: user_id, chapter_id, request_timestamp, response_time_ms, success/failure status.

---

### Key Entities

- **User Profile**: Extended user data including technical background. Attributes: software_background (JSONB with level, languages, frameworks), hardware_background (JSONB with level, domains), profile_completed (boolean). Stored in `users` table.

- **Personalization Request**: A user's request to personalize chapter content. Attributes: user_id, chapter_id, request_timestamp, response_time_ms, status. Stored in `personalization_logs` table.

- **Translation Request**: A user's request to translate chapter content. Attributes: user_id, chapter_id, target_language, request_timestamp, response_time_ms, status. Stored in `translation_logs` table.

- **Chapter Content**: Original textbook content served from markdown files. Not modified by personalization or translation.

- **Personalized Content**: Dynamically generated adaptation of chapter content. Cached per-session, not persisted permanently.

- **Translated Content**: Dynamically generated Urdu translation of chapter content. Cached per-session, not persisted permanently.

---

## API Contracts

### POST /api/profile

Create or update the current user's profile.

**Request**:
```json
{
  "display_name": "Sarah Khan",
  "software_background": {
    "level": "intermediate",
    "languages": ["Python", "JavaScript"],
    "frameworks": ["TensorFlow", "React"]
  },
  "hardware_background": {
    "level": "basic",
    "domains": ["basic_electronics", "robotics_kits"]
  }
}
```

**Response (Success - 200 OK)**:
```json
{
  "success": true,
  "user": {
    "id": "uuid-string",
    "email": "sarah@example.com",
    "display_name": "Sarah Khan",
    "software_background": {
      "level": "intermediate",
      "languages": ["Python", "JavaScript"],
      "frameworks": ["TensorFlow", "React"]
    },
    "hardware_background": {
      "level": "basic",
      "domains": ["basic_electronics", "robotics_kits"]
    },
    "profile_completed": true
  }
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input data or validation failure
- `401 Unauthorized`: No active session
- `500 Internal Server Error`: Database error

---

### GET /api/profile

Retrieve the current user's profile.

**Response (Success - 200 OK)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "sarah@example.com",
    "display_name": "Sarah Khan",
    "software_background": {
      "level": "intermediate",
      "languages": ["Python", "JavaScript"],
      "frameworks": ["TensorFlow", "React"]
    },
    "hardware_background": {
      "level": "basic",
      "domains": ["basic_electronics", "robotics_kits"]
    },
    "profile_completed": true
  }
}
```

**Error Responses**:
- `401 Unauthorized`: No active session

---

### POST /api/personalize

Generate personalized chapter content based on user profile.

**Request**:
```json
{
  "chapter_id": "chapter-3",
  "content": "# Neural Networks\n\nNeural networks are computational models inspired by biological neurons...",
  "title": "Introduction to Neural Networks"
}
```

**Response (Success - 200 OK)**:
```json
{
  "success": true,
  "personalized_content": "# Neural Networks\n\nSince you have experience with Python and TensorFlow, let's explore neural networks from a practical standpoint...",
  "metadata": {
    "user_level": "intermediate",
    "adaptations_made": ["adjusted_depth", "added_code_examples", "framework_specific_references"]
  }
}
```

**Error Responses**:
- `400 Bad Request`: Missing required fields
- `401 Unauthorized`: No active session
- `403 Forbidden`: Profile not completed
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: AI service error
- `504 Gateway Timeout`: AI service timeout (>30s)

---

### POST /api/translate

Translate chapter content to specified language.

**Request**:
```json
{
  "chapter_id": "chapter-3",
  "content": "# Neural Networks\n\nNeural networks are computational models...\n\n```python\nimport tensorflow as tf\n```",
  "target_language": "ur"
}
```

**Response (Success - 200 OK)**:
```json
{
  "success": true,
  "translated_content": "# نیورل نیٹ ورکس\n\nنیورل نیٹ ورکس کمپیوٹیشنل ماڈلز ہیں...\n\n```python\nimport tensorflow as tf\n```",
  "metadata": {
    "source_language": "en",
    "target_language": "ur",
    "preserved_blocks": 1
  }
}
```

**Error Responses**:
- `400 Bad Request`: Missing required fields or unsupported language
- `401 Unauthorized`: No active session
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: AI service error
- `504 Gateway Timeout`: AI service timeout (>30s)

---

### GET /api/auth/session (Updated)

Returns session with profile data.

**Response (Success - 200 OK)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "sarah@example.com",
    "display_name": "Sarah Khan",
    "auth_provider": "google",
    "profile_completed": true
  },
  "session": {
    "expires_at": "2025-12-30T10:00:00Z"
  }
}
```

---

## Database Schema

### Users Table (Extended)

```sql
-- Migration: 005_add_profile_fields.sql

-- Add profile columns to existing users table
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS software_background JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS hardware_background JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS profile_completed BOOLEAN DEFAULT FALSE NOT NULL;

-- Add index for profile queries
CREATE INDEX IF NOT EXISTS idx_users_profile_completed ON users(profile_completed);

-- Create personalization logs table
CREATE TABLE IF NOT EXISTS personalization_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chapter_id VARCHAR(100) NOT NULL,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms INTEGER,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failure', 'timeout')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_personalization_logs_user ON personalization_logs(user_id);
CREATE INDEX idx_personalization_logs_chapter ON personalization_logs(chapter_id);
CREATE INDEX idx_personalization_logs_timestamp ON personalization_logs(request_timestamp);

-- Create translation logs table
CREATE TABLE IF NOT EXISTS translation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chapter_id VARCHAR(100) NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms INTEGER,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failure', 'timeout')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_translation_logs_user ON translation_logs(user_id);
CREATE INDEX idx_translation_logs_chapter ON translation_logs(chapter_id);
CREATE INDEX idx_translation_logs_timestamp ON translation_logs(request_timestamp);
```

### Final Users Table Schema

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),  -- NULL for OAuth-only users
    display_name VARCHAR(100),
    auth_provider VARCHAR(20) DEFAULT 'email' NOT NULL,
    oauth_provider_id VARCHAR(255),
    software_background JSONB DEFAULT NULL,
    hardware_background JSONB DEFAULT NULL,
    profile_completed BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_auth_provider CHECK (auth_provider IN ('email', 'google', 'github'))
);
```

### JSONB Schema Examples

**software_background**:
```json
{
  "level": "beginner|intermediate|advanced",
  "languages": ["Python", "JavaScript", "C++", "Java", "Go", "Rust"],
  "frameworks": ["TensorFlow", "PyTorch", "React", "FastAPI", "ROS"]
}
```

**hardware_background**:
```json
{
  "level": "none|basic|intermediate|advanced",
  "domains": [
    "basic_electronics",
    "robotics_kits",
    "gpus_accelerators",
    "jetson_edge",
    "embedded_systems",
    "sensors_actuators"
  ]
}
```

---

## Agent Architecture

### Personalization Agent

**Purpose**: Adapt chapter content based on user's technical background.

**Input**:
- Original chapter content (markdown)
- User profile (software_background, hardware_background)
- Chapter metadata (title, chapter_id)

**Output**:
- Personalized chapter content (markdown)
- Metadata about adaptations made

**Behavior**:
- For beginners: Add foundational context, use simpler analogies, expand acronyms, add more examples
- For intermediate: Balance theory with practical examples, reference common tools
- For advanced: Skip basics, dive into edge cases, include optimization tips, reference papers/specs

**Guardrails**:
- MUST preserve all factual accuracy
- MUST NOT change code examples (except adding comments)
- MUST NOT remove safety warnings or critical information
- MUST maintain consistent terminology throughout
- Response timeout: 30 seconds maximum

### Translation Agent

**Purpose**: Translate chapter content to Urdu while preserving technical elements.

**Input**:
- Original chapter content (markdown)
- Target language code ("ur" for Urdu)

**Output**:
- Translated chapter content (markdown with RTL markers)
- Metadata about preserved blocks

**Behavior**:
- Translate all prose text to Urdu
- Preserve code blocks exactly as-is (between ``` markers)
- Preserve inline code exactly as-is (between ` markers)
- Keep technical terms in English with optional Urdu transliteration
- Preserve all URLs, file paths, and command examples
- Add dir="rtl" markers for proper text direction

**Guardrails**:
- MUST NOT translate code or commands
- MUST NOT alter URLs or paths
- MUST preserve markdown formatting (headers, lists, emphasis)
- MUST maintain technical term consistency
- Response timeout: 45 seconds maximum (translation is slower)

---

## Frontend Integration Details

### Profile Wizard Component

Location: `/frontend/src/components/ProfileWizard/index.tsx`

**Features**:
- Multi-step wizard (3 steps)
- Progress indicator
- Skip button (with warning about limited functionality)
- Form validation per step
- Persist on completion

### Chapter Action Buttons

Location: `/frontend/src/components/ChapterActions/index.tsx`

**Features**:
- Two buttons: "Personalize Content" and "Translate to Urdu"
- Only visible to logged-in users
- "Personalize Content" only enabled if profile_completed
- Loading states during API calls
- Toggle behavior (shows "Show Original" / "Show English" after activation)

### Profile Settings Page

Location: `/frontend/src/pages/profile.tsx`

**Features**:
- Display current profile data
- Edit form for all profile fields
- Save/Cancel buttons
- Success/error feedback
- Link to this page from navbar user menu

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: New users can complete the profile wizard within 2 minutes on first attempt.
- **SC-002**: 90% of users who start the profile wizard complete it successfully.
- **SC-003**: Personalized content is generated and displayed within 15 seconds of button click.
- **SC-004**: Translated content is generated and displayed within 30 seconds of button click.
- **SC-005**: Toggle between original/personalized/translated content happens within 500ms.
- **SC-006**: 100% of code blocks remain unchanged after personalization and translation.
- **SC-007**: System handles 50 concurrent personalization/translation requests without degradation.
- **SC-008**: All personalization and translation requests are logged with user_id and timestamp.
- **SC-009**: Profile updates are persisted and reflected in navbar within 2 seconds.
- **SC-010**: Users without completed profiles see appropriate prompts when attempting to personalize content.

---

## Assumptions

1. **OpenAI Agents SDK**: OpenAI Agents SDK supports custom agents that can be configured for personalization and translation tasks.
2. **AI Service Availability**: OpenAI API is available with reasonable latency (<10s for most requests).
3. **Chapter Content Size**: Individual chapters are typically <20,000 tokens, within AI model context limits.
4. **Urdu RTL Support**: Docusaurus can render RTL text with appropriate CSS styling.
5. **Session Caching**: In-memory or Redis caching is available for session-scoped personalization cache.
6. **User Profile Stability**: Most users won't change their profile frequently after initial setup.
7. **Markdown Parsing**: Backend can parse and reassemble markdown content while preserving structure.

---

## Out of Scope

- Multiple language translations (only Urdu in this phase)
- Persistent storage of personalized/translated content
- User feedback on personalization quality
- A/B testing of personalization approaches
- Offline/downloaded personalized content
- Per-section personalization (only whole chapter)
- Custom terminology glossaries per user
- Voice/audio translations
- Automatic profile inference from chat history
- Social features (sharing personalized content)

---

## Dependencies

- **External Services**:
  - OpenAI API (GPT-4 or GPT-4o for personalization and translation agents)
  - Neon Serverless Postgres (existing)

- **Frontend Libraries**:
  - React state management (existing context or add Zustand for complex state)
  - React Hook Form (for profile wizard)
  - CSS RTL support

- **Backend Libraries**:
  - OpenAI Agents SDK (for agent implementation)
  - `asyncpg` (existing)
  - `pydantic` (for request/response validation)
  - `cachetools` or Redis (for session-scoped caching)

- **Environment Variables** (new):
  - `OPENAI_API_KEY` (may already exist)
  - `OPENAI_MODEL` (e.g., "gpt-4o" for agents)
  - `PERSONALIZATION_CACHE_TTL` (seconds, default 3600)
  - `TRANSLATION_CACHE_TTL` (seconds, default 3600)

---

## Constraints

- **Original Content Immutability**: Original markdown files are never modified by this feature.
- **Session-Scoped Cache**: Personalized/translated content is cached only for the user's session, not permanently.
- **Rate Limiting**: Maximum 10 personalization/translation requests per user per minute to control AI costs.
- **Profile Completion Required**: Personalization requires a completed profile; translation does not.
- **Single Active View**: User can only view one variant at a time (original, personalized, OR translated - not combined).
- **No Combined Features**: Users cannot request "personalized AND translated" in one action (pick one).
- **Audit Trail**: All AI requests must be logged for cost tracking and abuse detection.

---

## Security Considerations

1. **Input Sanitization**: All user profile inputs must be sanitized before storage and before sending to AI.
2. **Prompt Injection**: AI prompts must be constructed to prevent prompt injection attacks from user profile data.
3. **Rate Limiting**: Strict rate limits prevent abuse of expensive AI endpoints.
4. **Authentication Required**: All personalization/translation endpoints require valid session.
5. **Audit Logging**: All requests logged for security review and cost monitoring.
6. **Content Validation**: AI responses should be validated to ensure they don't contain malicious content.

---

## Migration Plan

### Database Migration

```sql
-- Migration: 005_add_profile_fields.sql
-- See Database Schema section above for full SQL

-- Rollback migration
-- DROP TABLE IF EXISTS translation_logs;
-- DROP TABLE IF EXISTS personalization_logs;
-- ALTER TABLE users DROP COLUMN IF EXISTS profile_completed;
-- ALTER TABLE users DROP COLUMN IF EXISTS hardware_background;
-- ALTER TABLE users DROP COLUMN IF EXISTS software_background;
```

### Deployment Order

1. Deploy database migration (add columns and tables)
2. Deploy backend with new endpoints (profile, personalize, translate)
3. Deploy frontend with profile wizard and chapter actions
4. Monitor logs and error rates
5. Adjust rate limits if needed based on usage patterns
