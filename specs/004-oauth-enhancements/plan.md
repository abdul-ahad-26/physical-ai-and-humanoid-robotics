# Implementation Plan: OAuth Authentication Enhancements

**Feature Branch**: `004-oauth-enhancements`
**Spec Reference**: `specs/004-oauth-enhancements/spec.md`
**Created**: 2025-12-19
**Status**: Ready for Implementation

---

## Executive Summary

This plan extends the existing Better Auth email/password authentication system (003-auth-infrastructure) to support Google and GitHub OAuth sign-in. The implementation follows a layered approach: database schema first, then backend OAuth handlers, frontend OAuth buttons, and finally UI personalization with display names.

**Key Architecture Decision**: The existing system uses a custom FastAPI backend that implements Better Auth-compatible endpoints (not the actual Better Auth server library). This pattern continues for OAuth - we implement OAuth callback handlers manually in FastAPI while leveraging Better Auth React client's `signIn.social()` for frontend OAuth initiation.

---

## Current Architecture Analysis

### Existing Components

| Layer | Component | Location | Status |
|-------|-----------|----------|--------|
| Frontend | Auth Client | `frontend/src/lib/auth.ts` | Uses `better-auth/react` with `createAuthClient()` |
| Frontend | Login Page | `frontend/src/pages/login.tsx` | Email/password form only |
| Frontend | AuthButton | `frontend/src/components/AuthButton/index.tsx` | Shows email, not display name |
| Backend | Auth Router | `backend/src/api/auth.py` | Email/password + session endpoints |
| Backend | Config | `backend/src/config.py` | No OAuth settings |
| Database | Users Table | Neon Postgres | Missing `auth_provider`, `oauth_provider_id` |
| Database | Sessions | `auth_sessions` table | Provider-agnostic (no changes needed) |

### Integration Points

1. **Frontend → Backend**: `signIn.social()` triggers redirect to backend OAuth initiation endpoint
2. **Backend → OAuth Provider**: Token exchange via HTTPS (Google/GitHub APIs)
3. **Backend → Frontend**: Redirect with session cookie after successful OAuth
4. **Session Validation**: Existing `/api/auth/session` endpoint (no changes needed)

---

## Implementation Phases

### Phase 0: Prerequisites & Setup
**Objective**: Prepare OAuth credentials and environment before any code changes.

#### 0.1 Google Cloud Console Setup
- Create OAuth 2.0 Client ID in Google Cloud Console
- Configure authorized redirect URI: `{API_URL}/api/auth/callback/google`
- Obtain `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
- Scopes required: `email`, `profile`, `openid`

#### 0.2 GitHub Developer Settings Setup
- Register new OAuth App in GitHub Developer Settings
- Configure callback URL: `{API_URL}/api/auth/callback/github`
- Obtain `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`
- Scopes required: `user:email`, `read:user`

#### 0.3 Environment Variables
Update `.env` files for both local and production:

```bash
# Backend .env additions
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
API_URL=http://localhost:8000  # Or production URL
```

**Acceptance Criteria**:
- [ ] Google OAuth Client ID created with correct redirect URI
- [ ] GitHub OAuth App registered with correct callback URL
- [ ] All 5 new environment variables documented in `.env.example`

---

### Phase 1: Database Schema Migration
**Objective**: Extend users table to support OAuth without breaking existing users.

#### 1.1 Create Migration Script
**File**: `backend/scripts/migrations/004_add_oauth_fields.sql`

```sql
-- Migration: Add OAuth provider fields to users table
-- Version: 004
-- Date: 2025-12-19
-- Description: Adds auth_provider and oauth_provider_id columns for OAuth support

-- Step 1: Add new columns with defaults
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) DEFAULT 'email' NOT NULL;

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS oauth_provider_id VARCHAR(255);

-- Step 2: Create index for OAuth lookups (provider + provider_id)
CREATE INDEX IF NOT EXISTS idx_users_oauth
  ON users(auth_provider, oauth_provider_id);

-- Step 3: Add check constraint for valid providers
ALTER TABLE users
  DROP CONSTRAINT IF EXISTS valid_auth_provider;

ALTER TABLE users
  ADD CONSTRAINT valid_auth_provider
  CHECK (auth_provider IN ('email', 'google', 'github'));

-- Step 4: Make password_hash nullable (OAuth users don't have passwords)
ALTER TABLE users
  ALTER COLUMN password_hash DROP NOT NULL;

-- Step 5: Backfill existing users (should already have 'email' from default)
UPDATE users
  SET auth_provider = 'email'
  WHERE auth_provider IS NULL;
```

#### 1.2 Create Rollback Script
**File**: `backend/scripts/migrations/004_add_oauth_fields_rollback.sql`

```sql
-- Rollback: Remove OAuth provider fields
-- WARNING: This will fail if OAuth users exist with NULL password_hash

ALTER TABLE users DROP CONSTRAINT IF EXISTS valid_auth_provider;
DROP INDEX IF EXISTS idx_users_oauth;
ALTER TABLE users DROP COLUMN IF EXISTS oauth_provider_id;
ALTER TABLE users DROP COLUMN IF EXISTS auth_provider;
-- ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL; -- Manual only
```

#### 1.3 Update Schema Initialization
**File**: `backend/src/db/connection.py` (modify `init_db_schema`)

Add to users table definition:
```sql
auth_provider VARCHAR(20) DEFAULT 'email' NOT NULL,
oauth_provider_id VARCHAR(255),
CONSTRAINT valid_auth_provider CHECK (auth_provider IN ('email', 'google', 'github'))
```

Add to indexes section:
```sql
CREATE INDEX IF NOT EXISTS idx_users_oauth ON users(auth_provider, oauth_provider_id);
```

**Acceptance Criteria**:
- [ ] Migration runs successfully on existing database
- [ ] Existing users have `auth_provider = 'email'`
- [ ] New columns visible in Neon console
- [ ] Index `idx_users_oauth` created
- [ ] Rollback script tested (on test database)

---

### Phase 2: Backend Configuration
**Objective**: Add OAuth configuration and httpx dependency.

#### 2.1 Update Settings
**File**: `backend/src/config.py`

```python
class Settings(BaseModel):
    # ... existing settings ...

    # OAuth Provider Settings
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    github_client_id: str = os.getenv("GITHUB_CLIENT_ID", "")
    github_client_secret: str = os.getenv("GITHUB_CLIENT_SECRET", "")

    # API URL for OAuth callbacks
    api_url: str = os.getenv("API_URL", "http://localhost:8000")
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
```

#### 2.2 Add Dependencies
**File**: `backend/requirements.txt` (or `pyproject.toml`)

```
httpx>=0.25.0  # Async HTTP client for OAuth token exchange
```

**Acceptance Criteria**:
- [ ] Settings class includes all OAuth config fields
- [ ] `httpx` installed and importable
- [ ] Settings load correctly from environment

---

### Phase 3: Backend OAuth Handlers
**Objective**: Implement OAuth callback endpoints for Google and GitHub.

#### 3.1 Create OAuth Router
**File**: `backend/src/api/oauth.py` (new file)

**Structure**:
```python
# OAuth Provider Configuration
GOOGLE_CONFIG = {
    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_url": "https://oauth2.googleapis.com/token",
    "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
    "scopes": ["email", "profile", "openid"],
}

GITHUB_CONFIG = {
    "auth_url": "https://github.com/login/oauth/authorize",
    "token_url": "https://github.com/login/oauth/access_token",
    "userinfo_url": "https://api.github.com/user",
    "emails_url": "https://api.github.com/user/emails",
    "scopes": ["user:email", "read:user"],
}

# Endpoints
GET /api/auth/oauth/google          # Initiate Google OAuth (redirect)
GET /api/auth/callback/google       # Handle Google callback
GET /api/auth/oauth/github          # Initiate GitHub OAuth (redirect)
GET /api/auth/callback/github       # Handle GitHub callback
```

#### 3.2 OAuth Initiation Endpoints
These endpoints generate the OAuth authorization URL and redirect the user:

```python
@router.get("/oauth/google")
async def initiate_google_oauth(request: Request):
    """Redirect user to Google OAuth consent screen."""
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": f"{settings.api_url}/api/auth/callback/google",
        "response_type": "code",
        "scope": " ".join(GOOGLE_CONFIG["scopes"]),
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{GOOGLE_CONFIG['auth_url']}?{urlencode(params)}"
    return RedirectResponse(url=auth_url, status_code=302)
```

#### 3.3 OAuth Callback Handlers
Handle the OAuth provider's redirect with authorization code:

**Key Implementation Details**:
1. Exchange authorization code for access token (10s timeout)
2. Fetch user profile from provider
3. Extract email and display_name
4. Call `create_or_update_oauth_user()` helper
5. Set session cookie
6. Redirect to frontend homepage

**Error Handling**:
- Missing/invalid code → Redirect to `/login?error=oauth_failed`
- Token exchange timeout → Redirect with timeout message
- No email from provider → Redirect with email required message

#### 3.4 User Creation/Linking Logic
**Function**: `create_or_update_oauth_user()`

**Logic Flow**:
```
1. Check for existing user by (auth_provider, oauth_provider_id)
   → Found: Update last_login, return existing user

2. Check for existing user by email
   → Found: Auto-link (update auth_provider, oauth_provider_id,
            set display_name if null), return linked user

3. No existing user
   → Create new user with email, display_name, auth_provider,
     oauth_provider_id, null password_hash

4. Create session token (same as email/password flow)
5. Return session token
```

#### 3.5 Register OAuth Router
**File**: `backend/src/main.py`

```python
from src.api.oauth import router as oauth_router
app.include_router(oauth_router)
```

**Acceptance Criteria**:
- [ ] `/api/auth/oauth/google` redirects to Google consent
- [ ] `/api/auth/callback/google` handles callback and creates session
- [ ] `/api/auth/oauth/github` redirects to GitHub authorization
- [ ] `/api/auth/callback/github` handles callback and creates session
- [ ] Account linking works (same email → merge)
- [ ] New OAuth user gets `display_name` from provider
- [ ] Error cases redirect with error message

---

### Phase 4: Backend Session Endpoint Update
**Objective**: Include `auth_provider` in session response.

#### 4.1 Update Session Query
**File**: `backend/src/api/auth.py`

Modify the session query to include `auth_provider`:

```python
session = await conn.fetchrow(
    """
    SELECT s.expires_at, u.id, u.email, u.display_name, u.auth_provider
    FROM auth_sessions s
    JOIN users u ON s.user_id = u.id
    WHERE s.session_token = $1 AND s.expires_at > NOW()
    """,
    session_token,
)
```

#### 4.2 Update Response Model
```python
class UserResponse(BaseModel):
    id: str
    email: str
    display_name: Optional[str]
    auth_provider: Optional[str] = "email"  # New field
    created_at: Optional[str] = None
    last_login: Optional[str] = None
```

#### 4.3 Update Response Construction
```python
return SessionResponse(
    user=UserResponse(
        id=str(session["id"]),
        email=session["email"],
        display_name=session["display_name"],
        auth_provider=session["auth_provider"],  # New field
    ),
    session={"expires_at": session["expires_at"].isoformat()},
)
```

**Acceptance Criteria**:
- [ ] `/api/auth/session` returns `auth_provider` field
- [ ] OAuth users show `auth_provider: "google"` or `"github"`
- [ ] Email/password users show `auth_provider: "email"`

---

### Phase 5: Frontend OAuth Buttons
**Objective**: Add Google and GitHub sign-in buttons to login page.

#### 5.1 Create OAuth Button Icons
**File**: `frontend/src/components/OAuthIcons/index.tsx` (new)

Create or import SVG icons for Google and GitHub buttons.

#### 5.2 Update Login Page
**File**: `frontend/src/pages/login.tsx`

**Changes**:
1. Add OAuth button section above email form
2. Add "or" divider between OAuth and email sections
3. Implement `handleGoogleSignIn` and `handleGitHubSignIn` functions
4. Add OAuth error handling from URL params
5. Add loading state for OAuth buttons

**OAuth Sign-In Handler**:
```typescript
const handleGoogleSignIn = async () => {
  setError("");
  setLoading(true);
  try {
    // Redirect to backend OAuth initiation
    window.location.href = `${apiUrl}/api/auth/oauth/google`;
  } catch (err) {
    setError("Failed to initiate Google sign-in");
    setLoading(false);
  }
};
```

**Note**: We redirect to backend OAuth initiation endpoint rather than using `signIn.social()` directly, because our backend implements OAuth manually (not using Better Auth server).

#### 5.3 Add OAuth Button Styles
**File**: `frontend/src/pages/login.module.css`

```css
.oauthButtons {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.oauthButton {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.375rem;
  background: white;
  cursor: pointer;
  font-size: 0.875rem;
  font-weight: 500;
  transition: background-color 0.2s;
}

.oauthButton:hover {
  background-color: #f3f4f6;
}

.oauthButton.google { /* Google colors */ }
.oauthButton.github { /* GitHub colors */ }

.divider {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1.5rem 0;
  color: #6b7280;
}

.divider::before,
.divider::after {
  content: "";
  flex: 1;
  height: 1px;
  background: #e5e7eb;
}
```

#### 5.4 Handle OAuth Errors
Parse error from URL params on page load:

```typescript
useEffect(() => {
  const params = new URLSearchParams(window.location.search);
  const oauthError = params.get("error");
  const oauthMessage = params.get("message");
  if (oauthError) {
    setError(oauthMessage || "OAuth sign-in failed. Please try again.");
    // Clean URL
    window.history.replaceState({}, "", "/login");
  }
}, []);
```

**Acceptance Criteria**:
- [ ] Google sign-in button visible on login page
- [ ] GitHub sign-in button visible on login page
- [ ] Buttons styled consistently with existing UI
- [ ] Click redirects to backend OAuth endpoint
- [ ] Error messages display from OAuth failures
- [ ] Loading state shown during redirect

---

### Phase 6: Signup Page OAuth Buttons
**Objective**: Add same OAuth buttons to signup page for consistency.

#### 6.1 Update Signup Page
**File**: `frontend/src/pages/signup.tsx`

Add the same OAuth buttons section as login page. Users signing up via OAuth are automatically registered (no separate signup flow).

**Acceptance Criteria**:
- [ ] OAuth buttons on signup page match login page
- [ ] OAuth signup creates new account if not exists

---

### Phase 7: Navbar Display Name
**Objective**: Show "Welcome, {Name}" instead of email in navbar.

#### 7.1 Update AuthButton Component
**File**: `frontend/src/components/AuthButton/index.tsx`

**Changes**:
1. Get `display_name` from session data
2. Fall back to email if `display_name` is null/empty
3. Truncate long names (>20 chars) with ellipsis
4. Display "Welcome, {name}" format

```typescript
if (session) {
  // Prefer display_name, fall back to email
  const displayName = session.user?.display_name || session.user?.email;

  // Truncate long names
  const truncatedName = displayName && displayName.length > 20
    ? displayName.substring(0, 17) + '...'
    : displayName;

  return (
    <div>
      <span>Welcome, {truncatedName}</span>
      <button onClick={handleSignOut}>Sign Out</button>
    </div>
  );
}
```

**Acceptance Criteria**:
- [ ] OAuth users see "Welcome, {Full Name}"
- [ ] Email/password users with display_name see their name
- [ ] Email/password users without display_name see email
- [ ] Long names truncated to 20 chars + "..."

---

### Phase 8: Chat Widget Integration
**Objective**: Ensure chat widget works identically for OAuth users.

#### 8.1 Verify Session Data Flow
**File**: `frontend/src/components/ChatWidget/index.tsx`

The chat widget already receives `session` prop and uses it for authentication. Verify:
1. Session data includes `display_name` and `auth_provider`
2. Chat API calls use session cookie (already implemented)
3. No special handling needed for OAuth vs email users

#### 8.2 Verify Backend Validation
**File**: `backend/src/api/middleware.py`

The `validate_session()` middleware is already provider-agnostic - it only checks the session token, not the auth method. No changes needed.

**Acceptance Criteria**:
- [ ] OAuth users can send chat messages
- [ ] Chat sessions link to correct user_id
- [ ] Rate limiting works for OAuth users
- [ ] Session expiration works for OAuth users

---

### Phase 9: Testing
**Objective**: Comprehensive testing of all OAuth flows.

#### 9.1 Manual Testing Checklist

**Google OAuth Flow**:
- [ ] Click "Sign in with Google" → Redirects to Google consent
- [ ] Complete consent → Redirects back with session cookie
- [ ] New user created with correct email, display_name, auth_provider
- [ ] Navbar shows "Welcome, {Google Name}"
- [ ] Subsequent Google sign-in → Existing user recognized
- [ ] Cancel consent → Redirect to login with error message
- [ ] Chat widget works after Google sign-in

**GitHub OAuth Flow**:
- [ ] Click "Sign in with GitHub" → Redirects to GitHub authorization
- [ ] Complete authorization → Redirects back with session cookie
- [ ] New user created with email from GitHub (may need emails API)
- [ ] Display name uses GitHub name or username as fallback
- [ ] Navbar shows "Welcome, {GitHub Name/Username}"
- [ ] Subsequent GitHub sign-in → Existing user recognized
- [ ] Deny authorization → Redirect to login with error message
- [ ] Chat widget works after GitHub sign-in

**Account Linking**:
- [ ] Email/password user signs in with OAuth (same email) → Accounts linked
- [ ] Linked user can still sign in with email/password
- [ ] Linked user can sign in with OAuth
- [ ] Display name preserved if already set

**Edge Cases**:
- [ ] OAuth provider timeout (simulate with slow network) → Error message
- [ ] Missing email from GitHub → Error message
- [ ] Display name > 100 chars → Truncated in database
- [ ] Navbar name > 20 chars → Truncated with ellipsis
- [ ] Session expiration → Prompted to re-login

#### 9.2 Automated Tests (Optional)

**Backend Unit Tests**:
- `test_google_callback_success`
- `test_google_callback_no_code`
- `test_github_callback_success`
- `test_github_callback_private_email`
- `test_account_linking`
- `test_oauth_timeout`

**Frontend Component Tests**:
- `test_oauth_buttons_render`
- `test_oauth_error_display`
- `test_display_name_truncation`

**Acceptance Criteria**:
- [ ] All manual test cases pass
- [ ] (Optional) Automated tests written and passing

---

### Phase 10: Documentation & Deployment
**Objective**: Document setup and deploy to production.

#### 10.1 Update Environment Documentation
**File**: `backend/.env.example`

Add new variables with comments:
```bash
# OAuth: Google
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# OAuth: GitHub
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# API URL (used for OAuth callbacks)
API_URL=http://localhost:8000
```

#### 10.2 Create Setup Guide
**File**: `docs/oauth-setup.md` (or README section)

Document:
1. Google Cloud Console OAuth setup steps
2. GitHub Developer Settings OAuth app setup
3. Environment variable configuration
4. Testing the OAuth flow

#### 10.3 Production Deployment
1. Add environment variables to Render (backend)
2. Update Google OAuth authorized redirect URIs for production URL
3. Update GitHub OAuth callback URL for production URL
4. Run database migration on production Neon
5. Deploy backend changes
6. Deploy frontend changes
7. Test OAuth flows in production

**Acceptance Criteria**:
- [ ] `.env.example` updated with OAuth variables
- [ ] Setup documentation complete
- [ ] Production environment variables configured
- [ ] OAuth redirect URIs updated for production
- [ ] Database migrated in production
- [ ] End-to-end OAuth flows working in production

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                     │
│  ┌──────────────────────┐     ┌──────────────────────┐                   │
│  │   Login Page          │     │   AuthButton         │                   │
│  │   ┌──────────────┐   │     │   "Welcome, {Name}"  │                   │
│  │   │ Google Btn   │───┼────>│                      │                   │
│  │   │ GitHub Btn   │   │     └──────────────────────┘                   │
│  │   │ Email Form   │   │                                                 │
│  │   └──────────────┘   │                                                 │
│  └──────────────────────┘                                                 │
│             │                                                              │
│             │ window.location = /api/auth/oauth/{provider}                │
└─────────────┼──────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (FastAPI)                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ OAuth Router (/api/auth)                                            │ │
│  │                                                                      │ │
│  │  GET /oauth/google ──────> Redirect to Google OAuth                 │ │
│  │  GET /callback/google <─── Google redirects with code               │ │
│  │       │                                                              │ │
│  │       ├── Exchange code for token (httpx, 10s timeout)              │ │
│  │       ├── Fetch user profile                                         │ │
│  │       ├── create_or_update_oauth_user()                              │ │
│  │       │      ├── Check by (provider, provider_id)                   │ │
│  │       │      ├── Check by email (auto-link)                         │ │
│  │       │      └── Create new user                                     │ │
│  │       ├── Create session token                                       │ │
│  │       └── Redirect to / with Set-Cookie                              │ │
│  │                                                                      │ │
│  │  GET /oauth/github ──────> Redirect to GitHub OAuth                 │ │
│  │  GET /callback/github <─── GitHub redirects with code               │ │
│  │       └── (same flow as Google)                                      │ │
│  │                                                                      │ │
│  │  GET /session ──────────> Returns user + auth_provider              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
              │                           │
              │                           │
              ▼                           ▼
┌──────────────────────┐    ┌──────────────────────────────────────────┐
│   OAuth Providers     │    │           Neon Postgres                   │
│                       │    │                                          │
│  ┌─────────────────┐ │    │  users                                   │
│  │ Google OAuth    │ │    │  ├── id, email, password_hash            │
│  │ token exchange  │ │    │  ├── display_name                        │
│  │ /userinfo       │ │    │  ├── auth_provider  ← NEW                │
│  └─────────────────┘ │    │  └── oauth_provider_id  ← NEW            │
│                       │    │                                          │
│  ┌─────────────────┐ │    │  auth_sessions                           │
│  │ GitHub OAuth    │ │    │  └── (unchanged)                         │
│  │ token exchange  │ │    │                                          │
│  │ /user, /emails  │ │    └──────────────────────────────────────────┘
│  └─────────────────┘ │
└──────────────────────┘
```

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Better Auth `signIn.social()` incompatible with manual backend | High | Medium | Use direct redirect to backend OAuth endpoint instead |
| OAuth provider timeout | Medium | Low | 10s timeout + graceful error redirect |
| Account linking security | Medium | Low | Trust provider email verification |
| Cross-domain cookie issues | High | Low | Already working with SameSite=None |
| Production OAuth URL mismatch | High | Medium | Document exact URLs needed in setup |
| GitHub private email | Medium | Medium | Fetch from /user/emails endpoint |

---

## Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| httpx | Async HTTP client for OAuth | >=0.25.0 |
| better-auth/react | Frontend auth client | (existing) |
| asyncpg | Database access | (existing) |

---

## Extensibility for Future Providers

The implementation is designed for easy addition of new OAuth providers:

1. **Add provider config**: New entry in `PROVIDER_CONFIGS` dict
2. **Add endpoints**: Copy/modify Google/GitHub handlers
3. **Update constraint**: Add provider to `valid_auth_provider` check
4. **Add button**: New button on login page with provider branding

No changes needed to:
- Session management
- User model (auth_provider is generic string)
- Chat widget integration
- Rate limiting

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| OAuth sign-in completion rate | >95% | Google Analytics events |
| OAuth sign-in latency | <30s | Server-side timing logs |
| Display name capture rate | 100% | Database query (display_name NOT NULL for OAuth users) |
| Navbar personalization | 100% OAuth users see name | Manual verification |
| Existing auth regression | 0 failures | Manual + automated tests |

---

## Implementation Order Summary

```
Phase 0: Prerequisites (OAuth credentials, env vars)
    ↓
Phase 1: Database Migration (auth_provider, oauth_provider_id)
    ↓
Phase 2: Backend Config (Settings + httpx)
    ↓
Phase 3: Backend OAuth Handlers (oauth.py)
    ↓
Phase 4: Backend Session Update (include auth_provider)
    ↓
Phase 5: Frontend Login OAuth Buttons
    ↓
Phase 6: Frontend Signup OAuth Buttons
    ↓
Phase 7: Navbar Display Name
    ↓
Phase 8: Chat Widget Verification
    ↓
Phase 9: Testing
    ↓
Phase 10: Documentation & Deployment
```

**Estimated Implementation**: 10 phases, can be executed sequentially with each phase independently testable.
