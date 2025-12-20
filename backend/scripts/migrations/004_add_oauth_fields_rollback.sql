-- Rollback: 004_add_oauth_fields
-- Feature: 004-oauth-enhancements
-- WARNING: This will fail if OAuth users exist with NULL password_hash

BEGIN;

-- Step 1: Remove constraint
ALTER TABLE users DROP CONSTRAINT IF EXISTS valid_auth_provider;

-- Step 2: Remove index
DROP INDEX IF EXISTS idx_users_oauth;

-- Step 3: Remove columns
ALTER TABLE users DROP COLUMN IF EXISTS oauth_provider_id;
ALTER TABLE users DROP COLUMN IF EXISTS auth_provider;

-- Step 4: Restore password_hash NOT NULL (will fail if OAuth users exist)
-- Uncomment only if you've deleted all OAuth-only users
-- ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL;

COMMIT;
