# Feature Specification: OAuth Authentication Enhancements with Better Auth

**Feature Branch**: `004-oauth-enhancements`
**Created**: 2025-12-19
**Status**: Draft
**Input**: User description: "Extend authentication system to support OAuth sign-in with Google and GitHub using Better Auth, capture display names, and update UI to show user names instead of emails."

---

## Overview

This specification extends the existing Better Auth authentication infrastructure (003-auth-infrastructure) to add:

1. **OAuth Social Providers**: Google and GitHub sign-in buttons
2. **Display Name Capture**: Capture and persist user's full name from OAuth providers
3. **UI Personalization**: Show "Welcome, {User Name}" instead of email in navbar
4. **Provider-Agnostic Architecture**: Support future OAuth providers without code changes

### Current State

The existing system (`003-auth-infrastructure`) provides:
- Email/password authentication via Better Auth React client
- Cookie-based session management (7-day expiration)
- FastAPI backend with custom Better Auth-compatible endpoints
- Neon Postgres database with `users` and `auth_sessions` tables
- AuthButton component showing email in navbar

### Target State

After this feature:
- Users can sign in with Google or GitHub in addition to email/password
- Display names are captured from OAuth providers at first login
- Navbar shows "Welcome, {Display Name}" or avatar/name
- All existing email/password functionality continues to work
- Backend validates sessions regardless of authentication method

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Sign In with Google (Priority: P1)

As a new visitor, I want to sign in with my Google account so that I can access the AI chatbot without creating a separate account or remembering another password.

**Why this priority**: Google is the most widely used OAuth provider. This provides the lowest friction path to authentication for most users.

**Independent Test**: Can be fully tested by clicking "Sign in with Google" on the login page, authenticating with Google, and verifying the user is logged in with their Google profile name displayed.

**Acceptance Scenarios**:

1. **Given** an unauthenticated user on /login, **When** they click "Sign in with Google", **Then** they are redirected to Google's OAuth consent screen.

2. **Given** a user completes Google OAuth consent, **When** redirected back to the app, **Then** a new account is created (if first time) with their Google display name, email, and auth provider recorded, and they are logged in automatically.

3. **Given** an existing user who previously signed in with Google, **When** they click "Sign in with Google" again, **Then** their existing account is recognized, last_login is updated, and they are logged in without creating a duplicate account.

4. **Given** a user completes Google sign-in, **When** they view the navbar, **Then** they see "Welcome, {Google Display Name}" (not their email address).

---

### User Story 2 - Sign In with GitHub (Priority: P1)

As a developer visiting the Physical AI textbook, I want to sign in with my GitHub account so that I can use my existing developer identity and not manage another password.

**Why this priority**: GitHub is highly relevant for a technical textbook audience. Many readers will already have GitHub accounts.

**Independent Test**: Can be fully tested by clicking "Sign in with GitHub" on the login page, authenticating with GitHub, and verifying the user is logged in with their GitHub username displayed.

**Acceptance Scenarios**:

1. **Given** an unauthenticated user on /login, **When** they click "Sign in with GitHub", **Then** they are redirected to GitHub's OAuth authorization screen.

2. **Given** a user authorizes the app on GitHub, **When** redirected back, **Then** a new account is created with their GitHub display name (or username if name is not set), email, and auth_provider="github", and they are logged in automatically.

3. **Given** an existing user who previously signed in with GitHub, **When** they sign in again, **Then** their account is recognized and they are logged in without creating a duplicate.

4. **Given** a user who has both a GitHub account and an email account with the same email, **When** they sign in with GitHub, **Then** the system links to the existing account or prompts for account linking (implementation choice).

---

### User Story 3 - Display Personalized Greeting in Navbar (Priority: P2)

As a logged-in user, I want to see my name (not email) in the navbar so that the interface feels personalized and I can confirm I'm logged into the correct account.

**Why this priority**: Enhances user experience and builds trust by showing the user's actual name. Important for multi-account scenarios.

**Independent Test**: Can be tested by logging in and verifying the navbar shows the user's display name.

**Acceptance Scenarios**:

1. **Given** a user logged in via Google OAuth, **When** viewing any page, **Then** the navbar displays "Welcome, {Google Full Name}" (e.g., "Welcome, John Smith").

2. **Given** a user logged in via GitHub OAuth, **When** viewing any page, **Then** the navbar displays "Welcome, {GitHub Name}" or "Welcome, {GitHub Username}" if name is not set.

3. **Given** a user logged in via email/password without a display name, **When** viewing any page, **Then** the navbar falls back to showing their email address.

4. **Given** a user whose session data changes (e.g., name update), **When** they refresh the page, **Then** the navbar reflects the updated name.

---

### User Story 4 - Unified Authentication State Across Chat Widget (Priority: P2)

As a logged-in user, I want the chat widget to recognize my authentication regardless of how I signed in (Google, GitHub, or email) so that I can use the chatbot immediately without re-authenticating.

**Why this priority**: Core functionality - the chat widget is the primary value proposition. Auth method should be transparent to chat functionality.

**Independent Test**: Can be tested by signing in with any OAuth provider and immediately using the chat widget.

**Acceptance Scenarios**:

1. **Given** a user authenticated via Google OAuth, **When** they open the chat widget, **Then** they see the chat interface (not login prompt) and can send messages.

2. **Given** a user authenticated via GitHub OAuth, **When** they send a chat message, **Then** the backend validates their session and processes the request normally.

3. **Given** a user's OAuth session expires, **When** they try to use the chat widget, **Then** they are prompted to re-authenticate via the same or any other method.

---

### User Story 5 - Email/Password Users Can Add Display Name (Priority: P3)

As an existing email/password user, I want to add my display name to my account so that I see my name in the navbar like OAuth users do.

**Why this priority**: Nice-to-have for existing users but not blocking core OAuth functionality.

**Independent Test**: Can be tested by logging in with email/password and updating profile.

**Acceptance Scenarios**:

1. **Given** a logged-in email/password user, **When** they access a profile settings page, **Then** they can enter and save a display name.

2. **Given** an email/password user has set a display name, **When** they view the navbar, **Then** it shows "Welcome, {Display Name}" instead of their email.

---

### Edge Cases

- **Email Collision**: User tries to sign in with Google using an email that already exists from GitHub sign-in. System should either link accounts or show "Email already registered with different provider" message.
- **Missing Display Name from Provider**: OAuth provider returns null/empty name field. System falls back to username (GitHub) or email prefix (Google).
- **OAuth Consent Denied**: User cancels OAuth flow at Google/GitHub. System redirects back to login with error message.
- **OAuth Provider Downtime**: Google or GitHub OAuth is unavailable or times out (>10 seconds). System shows "Unable to connect to [provider]. Try again or use email/password."
- **Token Refresh Failure**: OAuth access token expires and cannot be refreshed. Session remains valid (we store our own sessions, not relying on provider tokens).
- **Account Linking Conflict**: User has email/password account, then tries to sign in with OAuth using same email. System auto-links silently, merging OAuth credentials into existing account.
- **Session Cookie Missing**: Frontend sends request without session cookie. Backend returns 401, frontend shows login prompt.
- **Display Name Too Long**: OAuth provider returns name > 100 characters. System truncates to 100 characters.

---

## Requirements *(mandatory)*

### Functional Requirements

#### OAuth Provider Configuration (Better Auth Server)

- **FR-001**: System MUST configure Google OAuth provider in Better Auth with `clientId` and `clientSecret` from environment variables (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`).
- **FR-002**: System MUST configure GitHub OAuth provider in Better Auth with `clientId` and `clientSecret` from environment variables (`GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`).
- **FR-003**: System MUST request OAuth scopes: `email` and `profile` from Google; `user:email` and `read:user` from GitHub.
- **FR-004**: System MUST configure OAuth callback URLs as `{API_URL}/api/auth/callback/google` and `{API_URL}/api/auth/callback/github`.
- **FR-005**: System MUST use `mapProfileToUser` to extract and store display name from OAuth profile data at first sign-in.

#### Frontend OAuth Integration (Better Auth Client)

- **FR-006**: System MUST add a "Sign in with Google" button on the /login page using Better Auth's `signIn.social({ provider: "google" })`.
- **FR-007**: System MUST add a "Sign in with GitHub" button on the /login page using Better Auth's `signIn.social({ provider: "github" })`.
- **FR-008**: OAuth sign-in buttons MUST be styled consistently with existing UI (green accent #10B981, matching button styles).
- **FR-009**: System MUST display loading state while OAuth redirect is in progress.
- **FR-010**: System MUST handle OAuth errors (consent denied, provider error) and display user-friendly messages.
- **FR-011**: System MUST redirect to homepage (/) after successful OAuth sign-in.

#### Display Name Management

- **FR-012**: System MUST capture `display_name` from OAuth provider profile at first sign-in:
  - Google: `profile.name` or `profile.given_name + profile.family_name`
  - GitHub: `profile.name` or `profile.login` (username) as fallback
- **FR-013**: System MUST store `display_name` in the `users` table alongside email.
- **FR-014**: System MUST NOT overwrite existing `display_name` on subsequent sign-ins (user may have customized it).
- **FR-015**: System MUST expose `display_name` in session data returned by `/api/auth/session`.

#### Navbar Personalization

- **FR-016**: AuthButton component MUST display "Welcome, {display_name}" when user has a display name set.
- **FR-017**: AuthButton component MUST fall back to displaying email when display_name is null/empty.
- **FR-018**: Navbar greeting MUST be truncated with ellipsis if name exceeds 20 characters (e.g., "Welcome, Alexandros Pa...").
- **FR-019**: System MUST update navbar immediately after successful sign-in without requiring page refresh.

#### Backend OAuth Validation (FastAPI)

- **FR-020**: Backend MUST implement OAuth callback endpoints that handle provider redirects and create/update user accounts. HTTP requests to OAuth providers (token exchange, userinfo) MUST use a 10-second timeout.
- **FR-021**: Backend MUST store `auth_provider` field in users table (values: "email", "google", "github").
- **FR-022**: Backend MUST auto-link accounts: when OAuth email matches existing email/password account, silently merge OAuth credentials into the existing account (no user prompt required, since OAuth providers verify email ownership).
- **FR-023**: Backend MUST generate session tokens for OAuth users using the same mechanism as email/password users.
- **FR-024**: Backend MUST validate sessions without checking auth_provider (session validation is provider-agnostic).

#### Database Schema Extensions

- **FR-025**: System MUST add `auth_provider VARCHAR(20) DEFAULT 'email'` column to users table.
- **FR-026**: System MUST add `oauth_provider_id VARCHAR(255)` column to store provider's unique user ID (for account linking).
- **FR-027**: System MUST add index on `(auth_provider, oauth_provider_id)` for fast lookups.
- **FR-028**: Migration MUST be backwards-compatible (existing users get auth_provider='email').

#### Chat Widget Integration

- **FR-029**: ChatWidget MUST receive session data including `display_name` and `auth_provider`.
- **FR-030**: Chat sessions and messages MUST be linked to user_id regardless of auth_provider.
- **FR-031**: ChatWidget MUST NOT change behavior based on auth_provider (provider-agnostic).

---

### Key Entities

- **User**: Extended to include `display_name`, `auth_provider`, and `oauth_provider_id`. Primary identity shown in UI is display_name. Stored in Neon Postgres `users` table.

- **OAuthAccount**: Implicit entity representing the link between a user and their OAuth provider account. Composed of `auth_provider` and `oauth_provider_id` fields on User.

- **Session**: Unchanged from 003-auth-infrastructure. Contains `user_id`, `session_token`, `expires_at`. Provider-agnostic.

---

## API Contracts

### POST /api/auth/callback/google

Handle Google OAuth callback after user consents.

**Request** (Query params from Google redirect):
```
?code=AUTHORIZATION_CODE&state=CSRF_STATE
```

**Response (Success - 302 Redirect)**:
```
Location: /
Set-Cookie: session=SESSION_TOKEN; HttpOnly; Secure; SameSite=None; Max-Age=604800
```

**Error Response (302 Redirect)**:
```
Location: /login?error=oauth_failed&message=Unable+to+complete+Google+sign-in
```

---

### POST /api/auth/callback/github

Handle GitHub OAuth callback after user authorizes.

**Request** (Query params from GitHub redirect):
```
?code=AUTHORIZATION_CODE&state=CSRF_STATE
```

**Response (Success - 302 Redirect)**:
```
Location: /
Set-Cookie: session=SESSION_TOKEN; HttpOnly; Secure; SameSite=None; Max-Age=604800
```

**Error Response (302 Redirect)**:
```
Location: /login?error=oauth_failed&message=Unable+to+complete+GitHub+sign-in
```

---

### GET /api/auth/session (Updated)

Returns session with display_name and auth_provider fields.

**Response (Success - 200 OK)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "display_name": "John Smith",
    "auth_provider": "google"
  },
  "session": {
    "expires_at": "2025-12-26T10:00:00Z"
  }
}
```

---

### Client-Side OAuth Sign-In

Using Better Auth React client:

```typescript
// Google sign-in
await authClient.signIn.social({
  provider: "google",
  callbackURL: "/"
});

// GitHub sign-in
await authClient.signIn.social({
  provider: "github",
  callbackURL: "/"
});
```

---

## Database Schema

### Users Table (Updated)

```sql
-- Add new columns to existing users table
ALTER TABLE users
  ADD COLUMN auth_provider VARCHAR(20) DEFAULT 'email' NOT NULL,
  ADD COLUMN oauth_provider_id VARCHAR(255);

-- Add index for OAuth lookups
CREATE INDEX idx_users_oauth ON users(auth_provider, oauth_provider_id);

-- Add check constraint for valid providers
ALTER TABLE users
  ADD CONSTRAINT valid_auth_provider
  CHECK (auth_provider IN ('email', 'google', 'github'));

-- Backfill existing users
UPDATE users SET auth_provider = 'email' WHERE auth_provider IS NULL;
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_auth_provider CHECK (auth_provider IN ('email', 'google', 'github'))
);

CREATE INDEX idx_users_oauth ON users(auth_provider, oauth_provider_id);
```

---

## Authentication Flow Diagrams

### Google OAuth Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User/Browser  │     │  Docusaurus FE  │     │  FastAPI Backend │     │   Google OAuth  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │                       │
         │ 1. Click "Sign in     │                       │                       │
         │    with Google"       │                       │                       │
         │──────────────────────>│                       │                       │
         │                       │                       │                       │
         │                       │ 2. signIn.social({    │                       │
         │                       │    provider: "google" │                       │
         │                       │    })                 │                       │
         │                       │──────────────────────>│                       │
         │                       │                       │                       │
         │                       │                       │ 3. Generate state,    │
         │                       │                       │    build auth URL     │
         │                       │                       │                       │
         │ 4. Redirect to Google OAuth                   │                       │
         │<──────────────────────────────────────────────│                       │
         │                       │                       │                       │
         │ 5. User enters credentials, consents          │                       │
         │──────────────────────────────────────────────────────────────────────>│
         │                       │                       │                       │
         │ 6. Redirect with authorization code           │                       │
         │<──────────────────────────────────────────────────────────────────────│
         │                       │                       │                       │
         │ 7. GET /api/auth/callback/google?code=XXX     │                       │
         │──────────────────────────────────────────────>│                       │
         │                       │                       │                       │
         │                       │                       │ 8. Exchange code for  │
         │                       │                       │    tokens             │
         │                       │                       │──────────────────────>│
         │                       │                       │                       │
         │                       │                       │ 9. Return tokens +    │
         │                       │                       │    user profile       │
         │                       │                       │<──────────────────────│
         │                       │                       │                       │
         │                       │                       │ 10. Create/update user│
         │                       │                       │     Create session    │
         │                       │                       │     Set cookie        │
         │                       │                       │                       │
         │ 11. Redirect to / with session cookie         │                       │
         │<──────────────────────────────────────────────│                       │
         │                       │                       │                       │
         │ 12. Load page with    │                       │                       │
         │     session cookie    │                       │                       │
         │──────────────────────>│                       │                       │
         │                       │                       │                       │
         │                       │ 13. useSession() -    │                       │
         │                       │     GET /session      │                       │
         │                       │──────────────────────>│                       │
         │                       │                       │                       │
         │                       │ 14. Return user data  │                       │
         │                       │     with display_name │                       │
         │                       │<──────────────────────│                       │
         │                       │                       │                       │
         │ 15. Show "Welcome,    │                       │                       │
         │     {Display Name}"   │                       │                       │
         │<──────────────────────│                       │                       │
         │                       │                       │                       │
```

### GitHub OAuth Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User/Browser  │     │  Docusaurus FE  │     │  FastAPI Backend │     │   GitHub OAuth  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │                       │
         │ 1. Click "Sign in     │                       │                       │
         │    with GitHub"       │                       │                       │
         │──────────────────────>│                       │                       │
         │                       │                       │                       │
         │                       │ 2. signIn.social({    │                       │
         │                       │    provider: "github" │                       │
         │                       │    })                 │                       │
         │                       │──────────────────────>│                       │
         │                       │                       │                       │
         │ 4. Redirect to GitHub authorization           │                       │
         │<──────────────────────────────────────────────│                       │
         │                       │                       │                       │
         │ 5. User authorizes app                        │                       │
         │──────────────────────────────────────────────────────────────────────>│
         │                       │                       │                       │
         │ 6. Redirect with code │                       │                       │
         │<──────────────────────────────────────────────────────────────────────│
         │                       │                       │                       │
         │ 7. GET /api/auth/callback/github?code=XXX     │                       │
         │──────────────────────────────────────────────>│                       │
         │                       │                       │                       │
         │                       │                       │ 8. Exchange code,     │
         │                       │                       │    get user profile   │
         │                       │                       │<─────────────────────>│
         │                       │                       │                       │
         │                       │                       │ 9. Extract name:      │
         │                       │                       │    profile.name ??    │
         │                       │                       │    profile.login      │
         │                       │                       │                       │
         │                       │                       │ 10. Create/update user│
         │                       │                       │     with display_name │
         │                       │                       │                       │
         │ 11. Redirect with session cookie              │                       │
         │<──────────────────────────────────────────────│                       │
         │                       │                       │                       │
         │ 12. Show "Welcome, {GitHub Name/Username}"    │                       │
         │<──────────────────────│                       │                       │
```

---

## Frontend Integration Details

### Updated Login Page (`/frontend/src/pages/login.tsx`)

```tsx
import { useSignIn, useSession } from "@/lib/auth";
import { useHistory } from "@docusaurus/router";
import { useState, useEffect } from "react";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const signIn = useSignIn();
  const { data: session } = useSession();
  const history = useHistory();

  // Redirect if already logged in
  useEffect(() => {
    if (session) {
      history.push("/");
    }
  }, [session, history]);

  // Handle OAuth error from redirect
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const oauthError = params.get("error");
    if (oauthError) {
      setError(params.get("message") || "OAuth sign-in failed");
    }
  }, []);

  const handleEmailSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);
    try {
      await signIn.email({ email, password });
      history.push("/");
    } catch (err) {
      setError("Invalid email or password");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setError("");
    setIsLoading(true);
    try {
      await signIn.social({ provider: "google", callbackURL: "/" });
    } catch (err) {
      setError("Failed to initiate Google sign-in");
      setIsLoading(false);
    }
  };

  const handleGitHubSignIn = async () => {
    setError("");
    setIsLoading(true);
    try {
      await signIn.social({ provider: "github", callbackURL: "/" });
    } catch (err) {
      setError("Failed to initiate GitHub sign-in");
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <h1>Log In</h1>

      {error && <p className="error">{error}</p>}

      {/* OAuth Buttons */}
      <div className="oauth-buttons">
        <button
          onClick={handleGoogleSignIn}
          disabled={isLoading}
          className="oauth-button google"
        >
          <GoogleIcon />
          Sign in with Google
        </button>

        <button
          onClick={handleGitHubSignIn}
          disabled={isLoading}
          className="oauth-button github"
        >
          <GitHubIcon />
          Sign in with GitHub
        </button>
      </div>

      <div className="divider">
        <span>or</span>
      </div>

      {/* Email/Password Form */}
      <form onSubmit={handleEmailSignIn}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          disabled={isLoading}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Signing in..." : "Sign in with Email"}
        </button>
      </form>

      <p className="signup-link">
        Don't have an account? <a href="/signup">Sign up</a>
      </p>
    </div>
  );
}
```

### Updated AuthButton Component (`/frontend/src/components/AuthButton/index.tsx`)

```tsx
import React from 'react';
import { useSession, useSignOut } from '../../lib/auth';
import { useHistory } from '@docusaurus/router';

export default function AuthButton() {
  const { data: session, isPending } = useSession();
  const history = useHistory();
  const signOut = useSignOut();

  const handleSignOut = async () => {
    try {
      await signOut();
      window.location.href = '/';
    } catch (error) {
      console.error('Sign out error:', error);
      window.location.href = '/';
    }
  };

  const handleSignIn = () => {
    history.push('/login');
  };

  if (isPending) {
    return null;
  }

  if (session) {
    // Get display name, fallback to email
    const displayName = session.user?.display_name || session.user?.email;

    // Truncate long names
    const truncatedName = displayName && displayName.length > 20
      ? displayName.substring(0, 17) + '...'
      : displayName;

    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span
          style={{
            fontSize: '0.875rem',
            color: 'var(--ifm-navbar-link-color)',
            marginRight: '0.5rem',
          }}
        >
          Welcome, {truncatedName}
        </span>
        <button
          onClick={handleSignOut}
          style={{
            padding: '0.375rem 0.75rem',
            backgroundColor: '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '0.375rem',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: 500,
          }}
        >
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={handleSignIn}
      style={{
        padding: '0.375rem 0.75rem',
        backgroundColor: '#10B981',
        color: 'white',
        border: 'none',
        borderRadius: '0.375rem',
        cursor: 'pointer',
        fontSize: '0.875rem',
        fontWeight: 500,
      }}
    >
      Sign In
    </button>
  );
}
```

---

## Backend Integration Details

### OAuth Callback Handler (`/backend/src/api/oauth.py`)

```python
"""
OAuth Callback Handler

Handles OAuth callbacks from Google and GitHub, creates/updates user accounts,
and establishes sessions.
"""

import httpx
import secrets
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Response, Request
from fastapi.responses import RedirectResponse

from src.config import settings
from src.db.connection import get_db_pool

router = APIRouter(prefix="/api/auth", tags=["OAuth"])


# OAuth Provider Configuration
GOOGLE_CONFIG = {
    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_url": "https://oauth2.googleapis.com/token",
    "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
    "scopes": ["email", "profile"],
}

GITHUB_CONFIG = {
    "auth_url": "https://github.com/login/oauth/authorize",
    "token_url": "https://github.com/login/oauth/access_token",
    "userinfo_url": "https://api.github.com/user",
    "emails_url": "https://api.github.com/user/emails",
    "scopes": ["user:email", "read:user"],
}


@router.get("/callback/google")
async def google_callback(
    request: Request,
    response: Response,
    code: Optional[str] = None,
    error: Optional[str] = None,
):
    """Handle Google OAuth callback."""

    if error:
        return RedirectResponse(
            url=f"/login?error=oauth_failed&message={error}",
            status_code=302,
        )

    if not code:
        return RedirectResponse(
            url="/login?error=oauth_failed&message=No+authorization+code",
            status_code=302,
        )

    try:
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                GOOGLE_CONFIG["token_url"],
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": f"{settings.api_url}/api/auth/callback/google",
                    "grant_type": "authorization_code",
                },
            )
            tokens = token_response.json()

            if "error" in tokens:
                raise HTTPException(400, tokens.get("error_description", "Token exchange failed"))

            # Get user info
            userinfo_response = await client.get(
                GOOGLE_CONFIG["userinfo_url"],
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            profile = userinfo_response.json()

        # Extract user data
        email = profile.get("email")
        display_name = profile.get("name") or f"{profile.get('given_name', '')} {profile.get('family_name', '')}".strip()
        oauth_provider_id = profile.get("id")

        if not email:
            return RedirectResponse(
                url="/login?error=oauth_failed&message=No+email+from+Google",
                status_code=302,
            )

        # Create or update user and session
        session_token = await create_or_update_oauth_user(
            email=email,
            display_name=display_name,
            auth_provider="google",
            oauth_provider_id=oauth_provider_id,
        )

        # Set session cookie and redirect
        redirect = RedirectResponse(url="/", status_code=302)
        redirect.set_cookie(
            key="session",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="none",
            max_age=604800,
        )
        return redirect

    except Exception as e:
        return RedirectResponse(
            url=f"/login?error=oauth_failed&message={str(e)}",
            status_code=302,
        )


@router.get("/callback/github")
async def github_callback(
    request: Request,
    response: Response,
    code: Optional[str] = None,
    error: Optional[str] = None,
):
    """Handle GitHub OAuth callback."""

    if error:
        return RedirectResponse(
            url=f"/login?error=oauth_failed&message={error}",
            status_code=302,
        )

    if not code:
        return RedirectResponse(
            url="/login?error=oauth_failed&message=No+authorization+code",
            status_code=302,
        )

    try:
        async with httpx.AsyncClient() as client:
            # Exchange code for token
            token_response = await client.post(
                GITHUB_CONFIG["token_url"],
                data={
                    "code": code,
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret,
                },
                headers={"Accept": "application/json"},
            )
            tokens = token_response.json()

            if "error" in tokens:
                raise HTTPException(400, tokens.get("error_description", "Token exchange failed"))

            access_token = tokens["access_token"]

            # Get user profile
            userinfo_response = await client.get(
                GITHUB_CONFIG["userinfo_url"],
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            profile = userinfo_response.json()

            # Get primary email (may be private)
            email = profile.get("email")
            if not email:
                emails_response = await client.get(
                    GITHUB_CONFIG["emails_url"],
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github.v3+json",
                    },
                )
                emails = emails_response.json()
                primary_email = next(
                    (e for e in emails if e.get("primary") and e.get("verified")),
                    None
                )
                email = primary_email["email"] if primary_email else None

        if not email:
            return RedirectResponse(
                url="/login?error=oauth_failed&message=No+verified+email+from+GitHub",
                status_code=302,
            )

        # Extract user data - prefer name, fallback to login (username)
        display_name = profile.get("name") or profile.get("login")
        oauth_provider_id = str(profile.get("id"))

        # Create or update user and session
        session_token = await create_or_update_oauth_user(
            email=email,
            display_name=display_name,
            auth_provider="github",
            oauth_provider_id=oauth_provider_id,
        )

        # Set session cookie and redirect
        redirect = RedirectResponse(url="/", status_code=302)
        redirect.set_cookie(
            key="session",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="none",
            max_age=604800,
        )
        return redirect

    except Exception as e:
        return RedirectResponse(
            url=f"/login?error=oauth_failed&message={str(e)}",
            status_code=302,
        )


async def create_or_update_oauth_user(
    email: str,
    display_name: str,
    auth_provider: str,
    oauth_provider_id: str,
) -> str:
    """
    Create or update user from OAuth and return session token.

    Strategy for existing users:
    1. If user exists with same oauth_provider_id -> update last_login, create session
    2. If user exists with same email but different provider -> link account (set oauth fields)
    3. If no user exists -> create new user
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Check for existing user by OAuth provider ID
        existing_oauth = await conn.fetchrow(
            """
            SELECT id, email, display_name FROM users
            WHERE auth_provider = $1 AND oauth_provider_id = $2
            """,
            auth_provider,
            oauth_provider_id,
        )

        if existing_oauth:
            # User exists with this OAuth account - update and create session
            user_id = existing_oauth["id"]
            await conn.execute(
                "UPDATE users SET last_login = NOW() WHERE id = $1",
                user_id,
            )
        else:
            # Check for existing user by email (potential account linking)
            existing_email = await conn.fetchrow(
                "SELECT id, auth_provider FROM users WHERE email = $1",
                email,
            )

            if existing_email:
                # Link OAuth to existing email account
                user_id = existing_email["id"]
                # Only update if display_name was not set
                await conn.execute(
                    """
                    UPDATE users SET
                        auth_provider = $1,
                        oauth_provider_id = $2,
                        display_name = COALESCE(display_name, $3),
                        last_login = NOW()
                    WHERE id = $4
                    """,
                    auth_provider,
                    oauth_provider_id,
                    display_name,
                    user_id,
                )
            else:
                # Create new user
                user = await conn.fetchrow(
                    """
                    INSERT INTO users (email, display_name, auth_provider, oauth_provider_id, created_at, last_login)
                    VALUES ($1, $2, $3, $4, NOW(), NOW())
                    RETURNING id
                    """,
                    email,
                    display_name[:100] if display_name else None,  # Truncate to 100 chars
                    auth_provider,
                    oauth_provider_id,
                )
                user_id = user["id"]

        # Create session
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.utcnow() + timedelta(days=7)

        await conn.execute(
            """
            INSERT INTO auth_sessions (user_id, session_token, expires_at)
            VALUES ($1, $2, $3)
            """,
            user_id,
            session_token,
            expires_at,
        )

        return session_token
```

### Updated Config (`/backend/src/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Existing settings...
    database_url: str
    cors_origins: list[str] = ["http://localhost:3000"]
    rate_limit_requests: int = 10

    # OAuth settings
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""
    api_url: str = "http://localhost:8000"  # For OAuth callbacks

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete Google OAuth sign-in within 30 seconds (from click to logged-in state).
- **SC-002**: Users can complete GitHub OAuth sign-in within 30 seconds (from click to logged-in state).
- **SC-003**: 100% of OAuth sign-ins capture and store the user's display name from the provider.
- **SC-004**: Navbar displays "Welcome, {Name}" for all users with a display name, not email.
- **SC-005**: Chat widget functions identically for OAuth users and email/password users (provider-agnostic).
- **SC-006**: Existing email/password authentication continues to work without changes.
- **SC-007**: OAuth sign-in buttons are visible and clickable on mobile viewports (responsive design).
- **SC-008**: Account linking correctly handles users who have both email and OAuth accounts with same email.
- **SC-009**: Zero secrets (client IDs, secrets) exposed in frontend code or network requests.

---

## Assumptions

1. **Better Auth Social Providers**: Better Auth React client `signIn.social()` works with Docusaurus and triggers OAuth redirects.
2. **Backend OAuth Handling**: FastAPI can handle OAuth callbacks and token exchange without Better Auth server library (manual implementation).
3. **Provider Availability**: Google and GitHub OAuth services are available and responsive.
4. **Email Uniqueness**: Each OAuth provider returns a verified email that can be used as unique identifier.
5. **Display Name Availability**: Both Google and GitHub provide some form of name or username in their profile data.
6. **Cross-Domain Cookies**: Session cookies work across domains (Vercel frontend to Render backend) with SameSite=None.
7. **HTTPS Everywhere**: Both frontend and backend are served over HTTPS in production (required for secure cookies).

---

## Out of Scope

- Additional OAuth providers (Apple, Microsoft, Twitter, etc.)
- Account unlinking (removing OAuth connection)
- Profile picture/avatar sync from OAuth providers
- Two-factor authentication (2FA)
- Email verification for OAuth accounts
- Password reset for OAuth users (they don't have passwords)
- Custom OAuth provider branding on consent screens
- Session revocation from provider side (revoking Google/GitHub access)
- Multiple OAuth accounts per user (one Google + one GitHub)
- User-initiated account merging UI

---

## Dependencies

- **External Services**:
  - Google Cloud Console (for OAuth client credentials)
  - GitHub Developer Settings (for OAuth app credentials)
  - Neon Serverless Postgres (existing)

- **Frontend Libraries**:
  - `better-auth/react` (existing)
  - React Icons or custom SVG icons for Google/GitHub buttons

- **Backend Libraries**:
  - `httpx` (async HTTP client for OAuth token exchange)
  - `asyncpg` (existing)

- **Environment Variables** (new):
  - `GOOGLE_CLIENT_ID`
  - `GOOGLE_CLIENT_SECRET`
  - `GITHUB_CLIENT_ID`
  - `GITHUB_CLIENT_SECRET`
  - `API_URL` (for OAuth callback URLs)

---

## Constraints

- **Backwards Compatibility**: All existing email/password users must continue to work.
- **Single Auth Provider Per User**: Users authenticate with one provider (cannot link multiple OAuth providers to same account currently).
- **Display Name Not Editable**: This spec does not include UI for users to edit their display name after OAuth signup.
- **No Provider Token Storage**: We do not store OAuth access/refresh tokens (only our own session tokens).
- **Callback URL Restrictions**: OAuth callback URLs must match exactly what's configured in Google/GitHub consoles.

---

## Clarifications

### Session 2025-12-19

- Q: How should the OAuth state parameter be generated and validated? → A: Better Auth handles state (use signIn.social built-in CSRF)
- Q: When a user signs in with OAuth using an email that already exists (email/password account), what should happen? → A: Auto-link silently (merge OAuth into existing account)
- Q: What should be the HTTP timeout for OAuth token exchange requests to Google/GitHub? → A: 10 seconds (balanced timeout for provider latency)

---

## Security Considerations

1. **CSRF Protection**: OAuth state parameter is handled automatically by Better Auth's `signIn.social()` method, which generates and validates CSRF state internally. No custom state management required on backend callbacks.
2. **Secret Storage**: OAuth client secrets must be stored in environment variables, never in code.
3. **Token Exposure**: Access tokens from providers should not be logged or exposed to frontend.
4. **Email Verification**: Both Google and GitHub provide verified emails; we trust their verification.
5. **Account Takeover Prevention**: If user has email/password account, OAuth with same email links (doesn't create new account that would shadow the old one).
6. **Session Security**: OAuth sessions use the same secure cookie configuration as email/password sessions.

---

## Migration Plan

### Database Migration

```sql
-- Migration: 004_add_oauth_fields.sql

-- Add columns
ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) DEFAULT 'email' NOT NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_provider_id VARCHAR(255);

-- Add index
CREATE INDEX IF NOT EXISTS idx_users_oauth ON users(auth_provider, oauth_provider_id);

-- Add constraint
ALTER TABLE users DROP CONSTRAINT IF EXISTS valid_auth_provider;
ALTER TABLE users ADD CONSTRAINT valid_auth_provider CHECK (auth_provider IN ('email', 'google', 'github'));

-- Make password_hash nullable for OAuth users
ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;

-- Backfill existing users
UPDATE users SET auth_provider = 'email' WHERE auth_provider IS NULL;
```

### Rollback Migration

```sql
-- Rollback: 004_add_oauth_fields_rollback.sql

-- Remove constraint
ALTER TABLE users DROP CONSTRAINT IF EXISTS valid_auth_provider;

-- Remove index
DROP INDEX IF EXISTS idx_users_oauth;

-- Remove columns
ALTER TABLE users DROP COLUMN IF EXISTS oauth_provider_id;
ALTER TABLE users DROP COLUMN IF EXISTS auth_provider;

-- Restore password_hash requirement (may fail if OAuth users exist)
-- ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL;
```
