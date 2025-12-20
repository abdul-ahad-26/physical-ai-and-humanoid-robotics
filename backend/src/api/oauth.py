"""
OAuth Authentication Router

Handles OAuth authentication flows for Google and GitHub providers.
Implements manual OAuth callback handling with token exchange and user creation.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from src.config import get_settings
from src.db.connection import get_db_pool

router = APIRouter(prefix="/api/auth", tags=["OAuth"])

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

# HTTP timeout for OAuth requests (10 seconds as per spec)
OAUTH_TIMEOUT = 10.0


# ============================================================================
# Google OAuth Endpoints
# ============================================================================


@router.get("/oauth/google")
async def initiate_google_oauth(request: Request):
    """
    Initiate Google OAuth flow by redirecting to Google's consent screen.
    """
    settings = get_settings()

    if not settings.google_client_id:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=Google+OAuth+not+configured",
            status_code=302,
        )

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


@router.get("/callback/google")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    """
    Handle Google OAuth callback after user consents.
    Exchanges authorization code for tokens, fetches user profile,
    creates/updates user account, and establishes session.
    """
    settings = get_settings()

    # Handle errors from Google
    if error:
        message = error_description or error
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message={message}",
            status_code=302,
        )

    if not code:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=No+authorization+code",
            status_code=302,
        )

    try:
        async with httpx.AsyncClient(timeout=OAUTH_TIMEOUT) as client:
            # Exchange code for tokens
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
                error_msg = tokens.get("error_description", tokens.get("error", "Token exchange failed"))
                return RedirectResponse(
                    url=f"{settings.frontend_url}/login?error=oauth_failed&message={error_msg}",
                    status_code=302,
                )

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
                url=f"{settings.frontend_url}/login?error=oauth_failed&message=No+email+from+Google",
                status_code=302,
            )

        # Create or update user and session
        session_token = await create_or_update_oauth_user(
            email=email,
            display_name=display_name,
            auth_provider="google",
            oauth_provider_id=oauth_provider_id,
        )

        # Redirect with session cookie (environment-aware security)
        redirect = RedirectResponse(url=settings.frontend_url, status_code=302)
        redirect.set_cookie(
            key="session",
            value=session_token,
            httponly=True,
            secure=settings.cookie_secure,
            samesite=settings.cookie_samesite,
            max_age=604800,  # 7 days
        )
        return redirect

    except httpx.TimeoutException:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=Google+OAuth+timed+out",
            status_code=302,
        )
    except Exception as e:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=OAuth+error",
            status_code=302,
        )


# ============================================================================
# GitHub OAuth Endpoints
# ============================================================================


@router.get("/oauth/github")
async def initiate_github_oauth(request: Request):
    """
    Initiate GitHub OAuth flow by redirecting to GitHub's authorization screen.
    """
    settings = get_settings()

    if not settings.github_client_id:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=GitHub+OAuth+not+configured",
            status_code=302,
        )

    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": f"{settings.api_url}/api/auth/callback/github",
        "scope": " ".join(GITHUB_CONFIG["scopes"]),
    }

    auth_url = f"{GITHUB_CONFIG['auth_url']}?{urlencode(params)}"
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/callback/github")
async def github_callback(
    request: Request,
    code: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    """
    Handle GitHub OAuth callback after user authorizes.
    Exchanges authorization code for tokens, fetches user profile,
    creates/updates user account, and establishes session.
    """
    settings = get_settings()

    # Handle errors from GitHub
    if error:
        message = error_description or error
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message={message}",
            status_code=302,
        )

    if not code:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=No+authorization+code",
            status_code=302,
        )

    try:
        async with httpx.AsyncClient(timeout=OAUTH_TIMEOUT) as client:
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
                error_msg = tokens.get("error_description", tokens.get("error", "Token exchange failed"))
                return RedirectResponse(
                    url=f"{settings.frontend_url}/login?error=oauth_failed&message={error_msg}",
                    status_code=302,
                )

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
                url=f"{settings.frontend_url}/login?error=oauth_failed&message=No+verified+email+from+GitHub",
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

        # Redirect with session cookie (environment-aware security)
        redirect = RedirectResponse(url=settings.frontend_url, status_code=302)
        redirect.set_cookie(
            key="session",
            value=session_token,
            httponly=True,
            secure=settings.cookie_secure,
            samesite=settings.cookie_samesite,
            max_age=604800,  # 7 days
        )
        return redirect

    except httpx.TimeoutException:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=GitHub+OAuth+timed+out",
            status_code=302,
        )
    except Exception as e:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=oauth_failed&message=OAuth+error",
            status_code=302,
        )


# ============================================================================
# Helper Functions
# ============================================================================


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
    2. If user exists with same email but different provider -> link account (auto-link)
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
                # Link OAuth to existing email account (auto-link per spec)
                user_id = existing_email["id"]
                # Only update display_name if not already set
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
                    display_name[:100] if display_name else None,
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
