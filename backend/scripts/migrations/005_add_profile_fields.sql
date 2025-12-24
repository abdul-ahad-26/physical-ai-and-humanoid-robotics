-- Migration: 005_add_profile_fields.sql
-- Feature: 005-user-personalization
-- Date: 2025-12-23
-- Description: Add user profile fields and audit tables for personalization/translation

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
        CHECK (status IN ('success', 'failure', 'timeout', 'error'))
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
        CHECK (status IN ('success', 'failure', 'timeout', 'error'))
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
