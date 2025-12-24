-- Rollback Migration: 005_add_profile_fields_rollback.sql
-- Feature: 005-user-personalization
-- Date: 2025-12-23
-- Description: Rollback user profile fields and audit tables

-- WARNING: This will delete all personalization and translation audit data!

-- =============================================================================
-- Step 1: Drop audit tables
-- =============================================================================

DROP TABLE IF EXISTS translation_logs;
DROP TABLE IF EXISTS personalization_logs;

-- =============================================================================
-- Step 2: Remove columns from users table
-- =============================================================================

ALTER TABLE users
  DROP COLUMN IF EXISTS software_background,
  DROP COLUMN IF EXISTS hardware_background,
  DROP COLUMN IF EXISTS profile_completed;

-- Drop index (will be dropped automatically with column, but explicit for clarity)
DROP INDEX IF EXISTS idx_users_profile_completed;

-- =============================================================================
-- Verification query
-- =============================================================================

SELECT
    'users.software_background' as column_name,
    NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'software_background'
    ) as dropped
UNION ALL
SELECT
    'users.hardware_background',
    NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'hardware_background'
    )
UNION ALL
SELECT
    'users.profile_completed',
    NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'users' AND column_name = 'profile_completed'
    )
UNION ALL
SELECT
    'personalization_logs table',
    NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'personalization_logs'
    )
UNION ALL
SELECT
    'translation_logs table',
    NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'translation_logs'
    );
