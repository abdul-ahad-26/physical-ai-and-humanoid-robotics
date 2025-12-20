-- Migration: 004_add_oauth_fields
-- Feature: 004-oauth-enhancements
-- Date: 2025-12-19
-- Description: Add OAuth provider fields to users table for Google/GitHub authentication

BEGIN;

-- Step 1: Add auth_provider column with default for existing users
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) DEFAULT 'email' NOT NULL;

-- Step 2: Add oauth_provider_id column (nullable - only set for OAuth users)
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS oauth_provider_id VARCHAR(255);

-- Step 3: Make password_hash nullable (OAuth users don't have passwords)
-- Check if column has NOT NULL constraint and drop it if so
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

-- Step 4: Create index for OAuth lookups (provider + provider_id)
CREATE INDEX IF NOT EXISTS idx_users_oauth
  ON users(auth_provider, oauth_provider_id);

-- Step 5: Add check constraint for valid providers
ALTER TABLE users
  DROP CONSTRAINT IF EXISTS valid_auth_provider;

ALTER TABLE users
  ADD CONSTRAINT valid_auth_provider
  CHECK (auth_provider IN ('email', 'google', 'github'));

-- Step 6: Backfill existing users (should already have 'email' from default, but ensure)
UPDATE users
  SET auth_provider = 'email'
  WHERE auth_provider IS NULL;

COMMIT;
