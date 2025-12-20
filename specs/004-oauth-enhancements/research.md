# Research: OAuth Authentication Enhancements

**Feature**: 004-oauth-enhancements
**Date**: 2025-12-19
**Status**: Complete

---

## Research Questions

### RQ-1: Better Auth OAuth Client Integration

**Question**: How does Better Auth React client handle OAuth social sign-in?

**Finding**: Better Auth React client provides `signIn.social()` method that:
1. Generates a CSRF state parameter automatically
2. Constructs the OAuth authorization URL
3. Redirects the user to the OAuth provider
4. Expects a callback endpoint at `/api/auth/callback/{provider}`

**Decision**: Use `signIn.social({ provider: "google" | "github" })` for OAuth initiation
**Rationale**: Handles CSRF state automatically, consistent with existing email auth patterns
**Alternatives Rejected**:
- Manual OAuth URL construction (requires custom CSRF handling)
- `signIn.oauth()` method (not available in Better Auth client)

---

### RQ-2: Backend OAuth Implementation Pattern

**Question**: Should we use Better Auth server library or implement OAuth callbacks manually?

**Finding**: The existing `003-auth-infrastructure` uses a custom FastAPI backend that implements Better Auth-compatible endpoints (not actual Better Auth server library). This pattern provides:
- Full control over user creation and session management
- Compatibility with existing asyncpg/Neon Postgres setup
- No additional dependencies on Better Auth server library

**Decision**: Implement OAuth callback handlers manually in FastAPI
**Rationale**: Maintains consistency with existing architecture, avoids adding new dependencies
**Alternatives Rejected**:
- Better Auth server library (would require rewriting existing auth)
- OAuth library like Authlib (unnecessary complexity for two providers)

---

### RQ-3: Google OAuth Token Exchange

**Question**: What are the Google OAuth endpoints and required parameters?

**Finding**: Google OAuth 2.0 flow:
1. Authorization URL: `https://accounts.google.com/o/oauth2/v2/auth`
2. Token URL: `https://oauth2.googleapis.com/token`
3. Userinfo URL: `https://www.googleapis.com/oauth2/v2/userinfo`
4. Required scopes: `email`, `profile`, `openid`

**Decision**: Use standard Google OAuth 2.0 endpoints with authorization code flow
**Rationale**: Well-documented, stable API
**Alternatives Rejected**: None (standard approach)

---

### RQ-4: GitHub OAuth Token Exchange

**Question**: What are the GitHub OAuth endpoints and required parameters?

**Finding**: GitHub OAuth flow:
1. Authorization URL: `https://github.com/login/oauth/authorize`
2. Token URL: `https://github.com/login/oauth/access_token`
3. User API: `https://api.github.com/user`
4. Emails API: `https://api.github.com/user/emails` (for private emails)
5. Required scopes: `user:email`, `read:user`

**Decision**: Use GitHub OAuth with additional emails endpoint for private email retrieval
**Rationale**: Some users have private emails; need to fetch from emails API
**Alternatives Rejected**: None (standard approach)

---

### RQ-5: Display Name Extraction

**Question**: How to reliably extract display names from OAuth providers?

**Finding**:
- **Google**: Returns `name` (full name), `given_name`, `family_name` in userinfo response
- **GitHub**: Returns `name` (can be null), `login` (username, always present)

**Decision**:
- Google: Use `name` field, fallback to `{given_name} {family_name}`
- GitHub: Use `name` field, fallback to `login` (username)
**Rationale**: Provides best available name while ensuring fallback exists
**Alternatives Rejected**: Email prefix as fallback (less personal)

---

### RQ-6: Account Linking Strategy

**Question**: What happens when OAuth email matches existing email/password account?

**Finding**: Three common strategies:
1. **Silent merge**: Auto-link OAuth to existing account (most convenient)
2. **User prompt**: Ask user to confirm linking (more secure but friction)
3. **Separate account**: Create new account (creates confusion)

**Decision**: Silent merge (auto-link OAuth to existing account)
**Rationale**:
- OAuth providers verify email ownership
- User expectation is that same email = same account
- Reduces confusion for users who signed up with email then try OAuth
**Alternatives Rejected**:
- User prompt (unnecessary friction for verified emails)
- Separate accounts (confusing, splits user identity)

---

### RQ-7: HTTP Client for Token Exchange

**Question**: Which async HTTP client should be used for OAuth token exchange?

**Finding**: Options considered:
- `httpx`: Modern async HTTP client, widely adopted
- `aiohttp`: Older but battle-tested
- `requests`: Synchronous only (not suitable)

**Decision**: Use `httpx` with 10-second timeout
**Rationale**:
- Already async-first design
- Good API ergonomics
- Lightweight dependency
**Alternatives Rejected**:
- `aiohttp` (heavier, more complex API)

---

### RQ-8: Frontend OAuth Button Approach

**Question**: Should OAuth buttons redirect or use popup flow?

**Finding**: Two approaches:
1. **Redirect flow**: Full page redirect to OAuth provider, then back
2. **Popup flow**: Opens new window for OAuth, parent listens for completion

**Decision**: Use redirect flow via `signIn.social()`
**Rationale**:
- Simpler to implement
- Better mobile support (popups problematic on mobile)
- Consistent with Better Auth client design
**Alternatives Rejected**:
- Popup flow (complexity, mobile issues)

---

### RQ-9: OAuth State/CSRF Handling

**Question**: How to implement CSRF protection for OAuth flow?

**Finding**: Better Auth's `signIn.social()` handles state parameter automatically:
1. Generates cryptographic state value
2. Stores in session/cookie
3. Validates on callback
4. No backend state management needed

**Decision**: Rely on Better Auth client's built-in CSRF state handling
**Rationale**: Already implemented, well-tested
**Alternatives Rejected**:
- Manual state generation (reinventing the wheel)

---

### RQ-10: Session Cookie Cross-Domain

**Question**: Will session cookies work across domains (Vercel → Render)?

**Finding**: Existing implementation already handles this:
- `SameSite=None` allows cross-site cookies
- `Secure=True` required for SameSite=None
- `credentials: "include"` in fetch options

**Decision**: Use existing cookie configuration (no changes needed)
**Rationale**: Already working for email/password auth
**Alternatives Rejected**: None

---

## Technology Decisions Summary

| Area | Decision | Library/API |
|------|----------|-------------|
| Frontend OAuth | Better Auth client | `signIn.social()` |
| Backend OAuth | Custom FastAPI handlers | httpx + asyncpg |
| Google OAuth | Authorization code flow | Google OAuth 2.0 APIs |
| GitHub OAuth | Authorization code flow | GitHub OAuth APIs |
| HTTP Client | Async with timeout | httpx (10s timeout) |
| CSRF Protection | Built-in | Better Auth state param |
| Account Linking | Silent merge | Auto-link on email match |

---

## Dependencies Added

| Dependency | Version | Purpose |
|------------|---------|---------|
| httpx | ^0.27.0 | Async HTTP for OAuth token exchange |

---

## Environment Variables Required

| Variable | Example | Description |
|----------|---------|-------------|
| GOOGLE_CLIENT_ID | 123456789.apps.googleusercontent.com | Google OAuth Client ID |
| GOOGLE_CLIENT_SECRET | GOCSPX-xxxxxx | Google OAuth Client Secret |
| GITHUB_CLIENT_ID | Ov23lixxxxxxxxx | GitHub OAuth App Client ID |
| GITHUB_CLIENT_SECRET | xxxxxxxxxxxxxxxx | GitHub OAuth App Client Secret |
| API_URL | https://api.example.com | Backend URL for OAuth callbacks |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OAuth provider downtime | Low | Medium | Graceful error handling, timeout |
| Email collision edge cases | Low | Medium | Auto-link strategy handles this |
| CORS issues with OAuth | Low | Low | Already handled in existing setup |
| Display name missing | Medium | Low | Fallback to username/email |

---

## Next Steps

1. Phase 1: Create data model documentation
2. Phase 1: Define API contracts (OpenAPI spec)
3. Phase 1: Create quickstart guide
