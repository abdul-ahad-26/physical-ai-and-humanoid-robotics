# Data Model: User Personalization Feature

**Feature**: 005-user-personalization
**Date**: 2025-12-23
**Status**: Draft

---

## Overview

This document defines the data model for user profiling, content personalization, and translation features. It extends the existing `users` table and adds two new audit tables.

---

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   users                                      │
│─────────────────────────────────────────────────────────────────────────────│
│ id: UUID (PK)                                                               │
│ email: VARCHAR(255) UNIQUE                                                  │
│ password_hash: VARCHAR(255)                                                 │
│ display_name: VARCHAR(100)                                                  │
│ auth_provider: VARCHAR(20)                                                  │
│ oauth_provider_id: VARCHAR(255)                                             │
│ software_background: JSONB          ← NEW                                   │
│ hardware_background: JSONB          ← NEW                                   │
│ profile_completed: BOOLEAN          ← NEW                                   │
│ created_at: TIMESTAMP WITH TIME ZONE                                        │
│ last_login: TIMESTAMP WITH TIME ZONE                                        │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         │ 1:N
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          personalization_logs                                │
│─────────────────────────────────────────────────────────────────────────────│
│ id: UUID (PK)                                                               │
│ user_id: UUID (FK → users.id)                                               │
│ chapter_id: VARCHAR(100)                                                    │
│ request_timestamp: TIMESTAMP WITH TIME ZONE                                 │
│ response_time_ms: INTEGER                                                   │
│ status: VARCHAR(20)                                                         │
│ error_message: TEXT                                                         │
│ created_at: TIMESTAMP WITH TIME ZONE                                        │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         │ 1:N
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            translation_logs                                  │
│─────────────────────────────────────────────────────────────────────────────│
│ id: UUID (PK)                                                               │
│ user_id: UUID (FK → users.id)                                               │
│ chapter_id: VARCHAR(100)                                                    │
│ target_language: VARCHAR(10)                                                │
│ request_timestamp: TIMESTAMP WITH TIME ZONE                                 │
│ response_time_ms: INTEGER                                                   │
│ status: VARCHAR(20)                                                         │
│ error_message: TEXT                                                         │
│ created_at: TIMESTAMP WITH TIME ZONE                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table Definitions

### users (Extended)

Extends the existing `users` table with profile fields.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | UUID | NOT NULL | gen_random_uuid() | Primary key |
| email | VARCHAR(255) | NOT NULL | - | Unique email address |
| password_hash | VARCHAR(255) | NULL | - | Argon2 hash (NULL for OAuth) |
| display_name | VARCHAR(100) | NULL | - | User's preferred name |
| auth_provider | VARCHAR(20) | NOT NULL | 'email' | 'email', 'google', 'github' |
| oauth_provider_id | VARCHAR(255) | NULL | - | OAuth provider's user ID |
| **software_background** | JSONB | NULL | NULL | **NEW**: Software experience |
| **hardware_background** | JSONB | NULL | NULL | **NEW**: Hardware experience |
| **profile_completed** | BOOLEAN | NOT NULL | FALSE | **NEW**: Wizard completion flag |
| created_at | TIMESTAMPTZ | NOT NULL | NOW() | Account creation time |
| last_login | TIMESTAMPTZ | NULL | - | Last login timestamp |

**Constraints:**
- PRIMARY KEY (id)
- UNIQUE (email)
- CHECK (auth_provider IN ('email', 'google', 'github'))

**Indexes:**
- idx_users_email (existing)
- idx_users_oauth (existing)
- **idx_users_profile_completed** (NEW)

---

### personalization_logs (New)

Audit table for tracking personalization requests.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | UUID | NOT NULL | gen_random_uuid() | Primary key |
| user_id | UUID | NOT NULL | - | Foreign key to users |
| chapter_id | VARCHAR(100) | NOT NULL | - | Chapter identifier |
| request_timestamp | TIMESTAMPTZ | NOT NULL | NOW() | When request was made |
| response_time_ms | INTEGER | NULL | - | AI response time in ms |
| status | VARCHAR(20) | NOT NULL | - | 'success', 'failure', 'timeout' |
| error_message | TEXT | NULL | - | Error details if failed |
| created_at | TIMESTAMPTZ | NOT NULL | NOW() | Row creation time |

**Constraints:**
- PRIMARY KEY (id)
- FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
- CHECK (status IN ('success', 'failure', 'timeout'))

**Indexes:**
- idx_personalization_logs_user (user_id)
- idx_personalization_logs_chapter (chapter_id)
- idx_personalization_logs_timestamp (request_timestamp)

---

### translation_logs (New)

Audit table for tracking translation requests.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | UUID | NOT NULL | gen_random_uuid() | Primary key |
| user_id | UUID | NOT NULL | - | Foreign key to users |
| chapter_id | VARCHAR(100) | NOT NULL | - | Chapter identifier |
| target_language | VARCHAR(10) | NOT NULL | - | Target language code (e.g., 'ur') |
| request_timestamp | TIMESTAMPTZ | NOT NULL | NOW() | When request was made |
| response_time_ms | INTEGER | NULL | - | AI response time in ms |
| status | VARCHAR(20) | NOT NULL | - | 'success', 'failure', 'timeout' |
| error_message | TEXT | NULL | - | Error details if failed |
| created_at | TIMESTAMPTZ | NOT NULL | NOW() | Row creation time |

**Constraints:**
- PRIMARY KEY (id)
- FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
- CHECK (status IN ('success', 'failure', 'timeout'))

**Indexes:**
- idx_translation_logs_user (user_id)
- idx_translation_logs_chapter (chapter_id)
- idx_translation_logs_timestamp (request_timestamp)

---

## JSONB Schemas

### software_background

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["level"],
  "properties": {
    "level": {
      "type": "string",
      "enum": ["beginner", "intermediate", "advanced"],
      "description": "Overall software development experience level"
    },
    "languages": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "default": [],
      "description": "Programming languages the user knows"
    },
    "frameworks": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "default": [],
      "description": "Frameworks and libraries the user has experience with"
    }
  }
}
```

**Example:**
```json
{
  "level": "intermediate",
  "languages": ["Python", "JavaScript", "C++"],
  "frameworks": ["TensorFlow", "React", "FastAPI"]
}
```

### hardware_background

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["level"],
  "properties": {
    "level": {
      "type": "string",
      "enum": ["none", "basic", "intermediate", "advanced"],
      "description": "Overall hardware/robotics experience level"
    },
    "domains": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "basic_electronics",
          "robotics_kits",
          "gpus_accelerators",
          "jetson_edge",
          "embedded_systems",
          "sensors_actuators",
          "3d_printing",
          "pcb_design"
        ]
      },
      "default": [],
      "description": "Hardware domains the user has experience with"
    }
  }
}
```

**Example:**
```json
{
  "level": "basic",
  "domains": ["basic_electronics", "robotics_kits"]
}
```

---

## Migration SQL

```sql
-- Migration: 005_add_profile_fields.sql
-- Feature: 005-user-personalization
-- Date: 2025-12-23

-- =============================================================================
-- Step 1: Add new columns to users table
-- =============================================================================

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS software_background JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS hardware_background JSONB DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS profile_completed BOOLEAN DEFAULT FALSE NOT NULL;

-- Add index for profile completion queries
CREATE INDEX IF NOT EXISTS idx_users_profile_completed ON users(profile_completed);

-- =============================================================================
-- Step 2: Create personalization_logs table
-- =============================================================================

CREATE TABLE IF NOT EXISTS personalization_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chapter_id VARCHAR(100) NOT NULL,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    response_time_ms INTEGER,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    CONSTRAINT personalization_logs_status_check
        CHECK (status IN ('success', 'failure', 'timeout'))
);

-- Indexes for personalization_logs
CREATE INDEX IF NOT EXISTS idx_personalization_logs_user
    ON personalization_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_personalization_logs_chapter
    ON personalization_logs(chapter_id);
CREATE INDEX IF NOT EXISTS idx_personalization_logs_timestamp
    ON personalization_logs(request_timestamp);

-- =============================================================================
-- Step 3: Create translation_logs table
-- =============================================================================

CREATE TABLE IF NOT EXISTS translation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chapter_id VARCHAR(100) NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    response_time_ms INTEGER,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    CONSTRAINT translation_logs_status_check
        CHECK (status IN ('success', 'failure', 'timeout'))
);

-- Indexes for translation_logs
CREATE INDEX IF NOT EXISTS idx_translation_logs_user
    ON translation_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_translation_logs_chapter
    ON translation_logs(chapter_id);
CREATE INDEX IF NOT EXISTS idx_translation_logs_timestamp
    ON translation_logs(request_timestamp);

-- =============================================================================
-- Verification query
-- =============================================================================

SELECT
    'users.software_background' as column_name,
    EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'software_background'
    ) as exists
UNION ALL
SELECT
    'users.hardware_background',
    EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'hardware_background'
    )
UNION ALL
SELECT
    'users.profile_completed',
    EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'profile_completed'
    )
UNION ALL
SELECT
    'personalization_logs table',
    EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'personalization_logs'
    )
UNION ALL
SELECT
    'translation_logs table',
    EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'translation_logs'
    );
```

---

## Rollback SQL

```sql
-- Rollback: 005_add_profile_fields_rollback.sql

-- Drop tables first (depends on users)
DROP TABLE IF EXISTS translation_logs;
DROP TABLE IF EXISTS personalization_logs;

-- Drop index
DROP INDEX IF EXISTS idx_users_profile_completed;

-- Remove columns from users
ALTER TABLE users DROP COLUMN IF EXISTS profile_completed;
ALTER TABLE users DROP COLUMN IF EXISTS hardware_background;
ALTER TABLE users DROP COLUMN IF EXISTS software_background;
```

---

## Pydantic Models

### Backend Models (Python)

```python
# backend/src/db/models.py

from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime

class SoftwareBackground(BaseModel):
    level: str = Field(..., pattern="^(beginner|intermediate|advanced)$")
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)

class HardwareBackground(BaseModel):
    level: str = Field(..., pattern="^(none|basic|intermediate|advanced)$")
    domains: list[str] = Field(default_factory=list)

class UserProfile(BaseModel):
    id: UUID
    email: str
    display_name: Optional[str] = None
    auth_provider: str
    software_background: Optional[SoftwareBackground] = None
    hardware_background: Optional[HardwareBackground] = None
    profile_completed: bool = False

class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(None, max_length=100)
    software_background: Optional[SoftwareBackground] = None
    hardware_background: Optional[HardwareBackground] = None

class ProfileUpdateResponse(BaseModel):
    success: bool
    user: UserProfile

class PersonalizationLogEntry(BaseModel):
    id: UUID
    user_id: UUID
    chapter_id: str
    request_timestamp: datetime
    response_time_ms: Optional[int] = None
    status: str
    error_message: Optional[str] = None

class TranslationLogEntry(BaseModel):
    id: UUID
    user_id: UUID
    chapter_id: str
    target_language: str
    request_timestamp: datetime
    response_time_ms: Optional[int] = None
    status: str
    error_message: Optional[str] = None
```

### Frontend Types (TypeScript)

```typescript
// frontend/src/lib/types.ts

export interface SoftwareBackground {
  level: 'beginner' | 'intermediate' | 'advanced';
  languages: string[];
  frameworks: string[];
}

export interface HardwareBackground {
  level: 'none' | 'basic' | 'intermediate' | 'advanced';
  domains: string[];
}

export interface UserProfile {
  id: string;
  email: string;
  display_name: string | null;
  auth_provider: 'email' | 'google' | 'github';
  software_background: SoftwareBackground | null;
  hardware_background: HardwareBackground | null;
  profile_completed: boolean;
}

export interface ProfileUpdateRequest {
  display_name?: string;
  software_background?: SoftwareBackground;
  hardware_background?: HardwareBackground;
}

export interface ProfileUpdateResponse {
  success: boolean;
  user: UserProfile;
}
```

---

## Query Examples

### Get User Profile

```sql
SELECT
    id,
    email,
    display_name,
    auth_provider,
    software_background,
    hardware_background,
    profile_completed
FROM users
WHERE id = $1;
```

### Update User Profile

```sql
UPDATE users
SET
    display_name = COALESCE($2, display_name),
    software_background = COALESCE($3, software_background),
    hardware_background = COALESCE($4, hardware_background),
    profile_completed = CASE
        WHEN $3 IS NOT NULL AND $4 IS NOT NULL
             AND ($3->>'level') IS NOT NULL
             AND ($4->>'level') IS NOT NULL
        THEN TRUE
        ELSE profile_completed
    END
WHERE id = $1
RETURNING *;
```

### Log Personalization Request

```sql
INSERT INTO personalization_logs (
    user_id,
    chapter_id,
    response_time_ms,
    status,
    error_message
) VALUES ($1, $2, $3, $4, $5)
RETURNING id;
```

### Get User's Personalization History

```sql
SELECT
    chapter_id,
    request_timestamp,
    response_time_ms,
    status
FROM personalization_logs
WHERE user_id = $1
ORDER BY request_timestamp DESC
LIMIT 20;
```

---

## Validation Rules

| Field | Rule | Error Message |
|-------|------|---------------|
| display_name | max 100 chars | "Display name must be 100 characters or less" |
| software_background.level | required enum | "Software level is required" |
| hardware_background.level | required enum | "Hardware level is required" |
| software_background.languages | optional array | - |
| hardware_background.domains | enum values only | "Invalid hardware domain" |

---

## Data Retention

| Table | Retention Policy | Reason |
|-------|------------------|--------|
| users | Indefinite | Core user data |
| personalization_logs | 90 days | Cost tracking, debugging |
| translation_logs | 90 days | Cost tracking, debugging |

---

## Future Considerations

1. **Personalization Feedback**: Add `feedback_rating` column to logs for quality tracking
2. **Multi-language Support**: Extend translation_logs for additional languages
3. **Profile Versioning**: Track profile changes over time for personalization optimization
4. **Cache Analytics**: Log cache hit/miss rates for performance optimization
