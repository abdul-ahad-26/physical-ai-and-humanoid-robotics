# Quickstart: OAuth Authentication Enhancements

**Feature**: 004-oauth-enhancements
**Time to Complete**: ~2 hours (implementation) + ~30 min (OAuth app setup)

---

## Prerequisites

Before implementing OAuth, ensure:

1. **003-auth-infrastructure is deployed** - Email/password auth working
2. **Google Cloud Console access** - For creating OAuth credentials
3. **GitHub Developer account** - For creating OAuth app
4. **Environment variable access** - Ability to update `.env` files

---

## Step 1: Create OAuth Applications (30 minutes)

### 1.1 Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create or select a project
3. Navigate to **APIs & Services → Credentials**
4. Click **Create Credentials → OAuth client ID**
5. Configure consent screen if prompted:
   - User Type: External
   - App name: Physical AI Textbook
   - Scopes: `email`, `profile`, `openid`
6. Create OAuth Client ID:
   - Application type: **Web application**
   - Authorized redirect URIs:
     - Development: `http://localhost:8000/api/auth/callback/google`
     - Production: `https://your-api-domain.com/api/auth/callback/google`
7. Copy **Client ID** and **Client Secret**

### 1.2 GitHub OAuth Setup

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **OAuth Apps → New OAuth App**
3. Fill in:
   - Application name: Physical AI Textbook
   - Homepage URL: `https://your-frontend-domain.com`
   - Authorization callback URL:
     - Development: `http://localhost:8000/api/auth/callback/github`
     - Production: `https://your-api-domain.com/api/auth/callback/github`
4. Click **Register application**
5. Generate a **Client Secret**
6. Copy **Client ID** and **Client Secret**

---

## Step 2: Configure Environment Variables (5 minutes)

### Backend `.env`

```bash
# Add to backend/.env
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your-google-secret
GITHUB_CLIENT_ID=Ov23li-your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
API_URL=http://localhost:8000  # Or production URL
```

### Update `.env.example`

```bash
# Add to backend/.env.example (without values)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
API_URL=http://localhost:8000
```

---

## Step 3: Run Database Migration (5 minutes)

```bash
# Connect to your Neon database and run migration
cd backend

# Option A: Using psql
psql $DATABASE_URL < scripts/migrations/004_add_oauth_fields.sql

# Option B: Using Python script (if available)
python scripts/migrate.py 004_add_oauth_fields

# Option C: Run SQL directly in Neon Console
# Copy contents of scripts/migrations/004_add_oauth_fields.sql
```

**Verify migration:**

```sql
-- Check columns exist
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'users'
  AND column_name IN ('auth_provider', 'oauth_provider_id');

-- Expected output:
-- auth_provider    | character varying | NO
-- oauth_provider_id| character varying | YES
```

---

## Step 4: Add httpx Dependency (2 minutes)

```bash
cd backend

# Add to requirements.txt
echo "httpx>=0.27.0" >> requirements.txt

# Install
pip install httpx>=0.27.0
```

---

## Step 5: Implement Backend OAuth (30 minutes)

### 5.1 Update Config

**File**: `backend/src/config.py`

```python
# Add to Settings class
google_client_id: str = ""
google_client_secret: str = ""
github_client_id: str = ""
github_client_secret: str = ""
api_url: str = "http://localhost:8000"
```

### 5.2 Create OAuth Router

**File**: `backend/src/api/oauth.py`

Create the OAuth callback handlers as specified in `spec.md` (see Backend Integration Details section).

### 5.3 Register OAuth Router

**File**: `backend/src/main.py`

```python
from src.api.oauth import router as oauth_router

# Add to app
app.include_router(oauth_router)
```

### 5.4 Update Session Endpoint

**File**: `backend/src/api/auth.py`

Add `auth_provider` to the session response in `get_session()`.

---

## Step 6: Implement Frontend OAuth (30 minutes)

### 6.1 Update Login Page

**File**: `frontend/src/pages/login.tsx`

Add Google and GitHub OAuth buttons as specified in `spec.md`.

### 6.2 Update AuthButton

**File**: `frontend/src/components/AuthButton/index.tsx`

Update to show `display_name` instead of email.

---

## Step 7: Test OAuth Flow (15 minutes)

### 7.1 Start Development Servers

```bash
# Terminal 1: Backend
cd backend
uvicorn src.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm start
```

### 7.2 Test Google OAuth

1. Navigate to `http://localhost:3000/login`
2. Click "Sign in with Google"
3. Complete Google consent
4. Verify redirect to homepage
5. Check navbar shows "Welcome, {Your Name}"

### 7.3 Test GitHub OAuth

1. Navigate to `http://localhost:3000/login`
2. Click "Sign in with GitHub"
3. Authorize the app
4. Verify redirect to homepage
5. Check navbar shows "Welcome, {Your Name}"

### 7.4 Test Account Linking

1. Create email/password account with email X
2. Sign out
3. Sign in with OAuth using same email X
4. Verify accounts are linked (same user ID)

---

## Step 8: Deploy (15 minutes)

### 8.1 Update Production Environment

```bash
# Render Dashboard (or your hosting provider)
# Add environment variables:
GOOGLE_CLIENT_ID=production-google-client-id
GOOGLE_CLIENT_SECRET=production-google-secret
GITHUB_CLIENT_ID=production-github-client-id
GITHUB_CLIENT_SECRET=production-github-secret
API_URL=https://your-api.onrender.com
```

### 8.2 Update OAuth Redirect URIs

1. **Google Cloud Console**: Add production callback URL
2. **GitHub Developer Settings**: Update callback URL to production

### 8.3 Deploy

```bash
# Push to trigger deployment
git add .
git commit -m "feat(auth): add Google and GitHub OAuth sign-in"
git push origin 004-oauth-enhancements
```

---

## Verification Checklist

- [ ] Google OAuth: Users can sign in with Google
- [ ] GitHub OAuth: Users can sign in with GitHub
- [ ] Display Name: Navbar shows user's name from OAuth
- [ ] Account Linking: OAuth with existing email links to existing account
- [ ] Session: OAuth sessions work with chat widget
- [ ] Fallback: Error handling shows friendly messages
- [ ] Mobile: OAuth buttons work on mobile viewports

---

## Troubleshooting

### OAuth redirect fails

1. Check callback URL matches exactly (including trailing slash)
2. Verify environment variables are set
3. Check browser console for CORS errors

### "No email from provider"

1. GitHub: Ensure `user:email` scope is requested
2. Check if user has verified email on provider

### Session cookie not set

1. Verify `SameSite=None` and `Secure=True`
2. Check if running on HTTPS (required for cross-domain)

### Account linking not working

1. Check if email matches exactly (case-sensitive)
2. Verify `auth_provider` and `oauth_provider_id` columns exist

---

## Next Steps

After completing this quickstart:

1. Run full test suite: `pytest backend/tests/`
2. Add E2E tests for OAuth flow
3. Update user documentation
4. Consider adding more OAuth providers (Apple, Microsoft)
