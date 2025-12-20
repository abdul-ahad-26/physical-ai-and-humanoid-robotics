# Data Model: OAuth Authentication Enhancements

**Feature**: 004-oauth-enhancements
**Date**: 2025-12-19
**Status**: Complete

---

## Entity Overview

This feature extends the existing `users` table to support OAuth authentication while maintaining backwards compatibility with email/password users.

```
┌──────────────────────────────────────────────────────────────────────┐
│                              users                                    │
├──────────────────────────────────────────────────────────────────────┤
│ id (PK)                                                              │
│ email (UNIQUE)                                                       │
│ password_hash (NULL for OAuth-only)                                  │
│ display_name (from OAuth profile)                     [NEW]          │
│ auth_provider (email|google|github)                   [NEW]          │
│ oauth_provider_id (provider's user ID)                [NEW]          │
│ created_at                                                           │
│ last_login                                                           │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ 1:N
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          auth_sessions                                │
├──────────────────────────────────────────────────────────────────────┤
│ id (PK)                                                              │
│ user_id (FK → users.id)                                              │
│ session_token                                                        │
│ expires_at                                                           │
│ created_at                                                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Entity: User (Extended)

### Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, DEFAULT gen_random_uuid() | Unique user identifier |
| `email` | VARCHAR(255) | UNIQUE, NOT NULL | User's email address |
| `password_hash` | VARCHAR(255) | NULLABLE | Argon2 hash (NULL for OAuth-only users) |
| `display_name` | VARCHAR(100) | NULLABLE | User's display name from OAuth or self-set |
| `auth_provider` | VARCHAR(20) | NOT NULL, DEFAULT 'email' | Auth method: 'email', 'google', 'github' |
| `oauth_provider_id` | VARCHAR(255) | NULLABLE | Provider's unique user ID |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Account creation timestamp |
| `last_login` | TIMESTAMPTZ | NULLABLE | Last successful login timestamp |

### Constraints

```sql
-- Primary key
PRIMARY KEY (id)

-- Unique email
UNIQUE (email)

-- Valid auth provider values
CHECK (auth_provider IN ('email', 'google', 'github'))

-- OAuth provider ID index for fast lookups
INDEX idx_users_oauth ON users(auth_provider, oauth_provider_id)
```

### Validation Rules

| Field | Rule | Error Message |
|-------|------|---------------|
| email | Valid email format | "Invalid email address" |
| email | Unique across all users | "Email already registered" |
| password_hash | Required when auth_provider='email' | "Password required for email signup" |
| display_name | Max 100 characters | "Display name too long" |
| auth_provider | One of: email, google, github | "Invalid auth provider" |
| oauth_provider_id | Required when auth_provider != 'email' | "OAuth provider ID required" |

### State Transitions

```
┌─────────────────┐
│   No Account    │
└────────┬────────┘
         │
         ├──── Email Signup ──────────────────┐
         │                                    │
         │    ┌───────────────────────────────▼───────────────────┐
         │    │              ACTIVE USER                          │
         │    │  auth_provider: 'email'                          │
         │    │  password_hash: SET                              │
         │    │  oauth_provider_id: NULL                         │
         │    └───────────────────────────────┬───────────────────┘
         │                                    │
         │                                    │ OAuth Link (same email)
         │                                    ▼
         │    ┌───────────────────────────────────────────────────┐
         │    │              ACTIVE USER (Linked)                 │
         │    │  auth_provider: 'google'|'github' (updated)      │
         │    │  password_hash: SET (preserved)                  │
         │    │  oauth_provider_id: SET                          │
         │    └───────────────────────────────────────────────────┘
         │
         └──── OAuth Signup ──────────────────┐
                                              │
              ┌───────────────────────────────▼───────────────────┐
              │              ACTIVE USER (OAuth-only)             │
              │  auth_provider: 'google'|'github'                │
              │  password_hash: NULL                             │
              │  oauth_provider_id: SET                          │
              │  display_name: FROM PROVIDER                     │
              └───────────────────────────────────────────────────┘
```

---

## Entity: AuthSession (Unchanged)

The `auth_sessions` table requires no changes. Sessions are provider-agnostic.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK | Session identifier |
| `user_id` | UUID | FK → users.id | Reference to user |
| `session_token` | VARCHAR(255) | UNIQUE, NOT NULL | Secure random token |
| `expires_at` | TIMESTAMPTZ | NOT NULL | Session expiration time |
| `created_at` | TIMESTAMPTZ | DEFAULT NOW() | Session creation time |

---

## Migration Script

### Forward Migration (004_add_oauth_fields.sql)

```sql
-- Migration: 004_add_oauth_fields
-- Feature: 004-oauth-enhancements
-- Date: 2025-12-19
-- Description: Add OAuth provider fields to users table

BEGIN;

-- Step 1: Add auth_provider column with default for existing users
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) DEFAULT 'email' NOT NULL;

-- Step 2: Add oauth_provider_id column (nullable)
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS oauth_provider_id VARCHAR(255);

-- Step 3: Make password_hash nullable (OAuth users don't have passwords)
-- Check if column is NOT NULL first
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'users'
      AND column_name = 'password_hash'
      AND is_nullable = 'NO'
  ) THEN
    ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;
  END IF;
END $$;

-- Step 4: Create index for OAuth lookups
CREATE INDEX IF NOT EXISTS idx_users_oauth
  ON users(auth_provider, oauth_provider_id);

-- Step 5: Add check constraint for valid providers
ALTER TABLE users
  DROP CONSTRAINT IF EXISTS valid_auth_provider;

ALTER TABLE users
  ADD CONSTRAINT valid_auth_provider
  CHECK (auth_provider IN ('email', 'google', 'github'));

-- Step 6: Backfill existing users (all are email auth)
UPDATE users
SET auth_provider = 'email'
WHERE auth_provider IS NULL;

COMMIT;
```

### Rollback Migration (004_add_oauth_fields_rollback.sql)

```sql
-- Rollback: 004_add_oauth_fields
-- WARNING: Will fail if OAuth users exist (password_hash is NULL)

BEGIN;

-- Step 1: Remove constraint
ALTER TABLE users DROP CONSTRAINT IF EXISTS valid_auth_provider;

-- Step 2: Remove index
DROP INDEX IF EXISTS idx_users_oauth;

-- Step 3: Remove columns
ALTER TABLE users DROP COLUMN IF EXISTS oauth_provider_id;
ALTER TABLE users DROP COLUMN IF EXISTS auth_provider;

-- Step 4: Restore password_hash NOT NULL (will fail if OAuth users exist)
-- ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL;

COMMIT;
```

---

## Query Patterns

### Find User by OAuth Provider

```sql
-- Used during OAuth callback to check if user already exists
SELECT id, email, display_name
FROM users
WHERE auth_provider = $1 AND oauth_provider_id = $2;
```

### Find User by Email (Account Linking)

```sql
-- Used to detect existing account for potential linking
SELECT id, auth_provider, oauth_provider_id
FROM users
WHERE email = $1;
```

### Create OAuth User

```sql
INSERT INTO users (email, display_name, auth_provider, oauth_provider_id, created_at, last_login)
VALUES ($1, $2, $3, $4, NOW(), NOW())
RETURNING id, email, display_name, auth_provider;
```

### Link OAuth to Existing Account

```sql
UPDATE users
SET auth_provider = $1,
    oauth_provider_id = $2,
    display_name = COALESCE(display_name, $3),
    last_login = NOW()
WHERE id = $4;
```

### Get User with Display Name for Session

```sql
-- Used by GET /api/auth/session
SELECT s.expires_at, u.id, u.email, u.display_name, u.auth_provider
FROM auth_sessions s
JOIN users u ON s.user_id = u.id
WHERE s.session_token = $1 AND s.expires_at > NOW();
```

---

## Data Integrity Rules

1. **Email Uniqueness**: Each email can only exist once in the system
2. **OAuth Provider Uniqueness**: Same `(auth_provider, oauth_provider_id)` pair cannot exist twice
3. **Account Linking**: When OAuth email matches existing email, update existing account (don't create duplicate)
4. **Display Name Preservation**: Once set, display_name is not overwritten on subsequent OAuth logins
5. **Password Compatibility**: Email users can still login with password after OAuth is linked

---

## Performance Considerations

### Indexes

| Index | Columns | Purpose |
|-------|---------|---------|
| PRIMARY KEY | (id) | User lookup by ID |
| UNIQUE | (email) | User lookup by email, prevent duplicates |
| idx_users_oauth | (auth_provider, oauth_provider_id) | Fast OAuth user lookup |

### Expected Query Patterns

| Query | Frequency | Index Used |
|-------|-----------|------------|
| Find by email | High (login, signup) | UNIQUE(email) |
| Find by OAuth | Medium (OAuth callback) | idx_users_oauth |
| Find by ID | High (session validation) | PRIMARY KEY |
