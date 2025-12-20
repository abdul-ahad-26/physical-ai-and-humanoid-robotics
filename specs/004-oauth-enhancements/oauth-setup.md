# OAuth Setup Guide

This guide explains how to configure Google and GitHub OAuth for the Physical AI and Humanoid Robotics platform.

## Prerequisites

- Backend deployed and running
- Frontend deployed and running
- Access to Google Cloud Console
- Access to GitHub Developer Settings

## Environment Variables

Add the following to your `backend/.env` file:

```env
# OAuth: Google
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your-google-secret

# OAuth: GitHub
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# URLs (adjust for your deployment)
API_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

## Google OAuth Setup

### 1. Create OAuth Client

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services** > **Credentials**
4. Click **Create Credentials** > **OAuth client ID**
5. Select **Web application**
6. Enter a name (e.g., "Physical AI Textbook")

### 2. Configure Authorized Redirect URIs

Add the following redirect URIs:

**Development:**
```
http://localhost:8000/api/auth/callback/google
```

**Production:**
```
https://your-backend-domain.com/api/auth/callback/google
```

### 3. Configure OAuth Consent Screen

1. Go to **APIs & Services** > **OAuth consent screen**
2. Choose **External** user type
3. Fill in required fields:
   - App name: "Physical AI Textbook"
   - User support email: your email
   - Developer contact: your email
4. Add scopes:
   - `email`
   - `profile`
   - `openid`
5. Add test users if in testing mode

### 4. Get Credentials

1. After creating the OAuth client, copy:
   - **Client ID** (ends with `.apps.googleusercontent.com`)
   - **Client Secret** (starts with `GOCSPX-`)
2. Add these to your `.env` file

## GitHub OAuth Setup

### 1. Register OAuth App

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **OAuth Apps** > **New OAuth App**
3. Fill in the form:
   - **Application name**: "Physical AI Textbook"
   - **Homepage URL**: `http://localhost:3000` (or production URL)
   - **Authorization callback URL**: See below

### 2. Configure Callback URL

**Development:**
```
http://localhost:8000/api/auth/callback/github
```

**Production:**
```
https://your-backend-domain.com/api/auth/callback/github
```

### 3. Get Credentials

1. After registration, copy the **Client ID**
2. Click **Generate a new client secret** and copy it
3. Add these to your `.env` file

## Production Deployment

### Render (Backend)

1. Go to your Render dashboard
2. Select your backend service
3. Go to **Environment** tab
4. Add the OAuth environment variables:
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
   - `GITHUB_CLIENT_ID`
   - `GITHUB_CLIENT_SECRET`
   - `API_URL` (your Render backend URL, e.g., `https://your-app.onrender.com`)
   - `FRONTEND_URL` (your Vercel frontend URL, e.g., `https://your-app.vercel.app`)

### Update OAuth Redirect URIs

After deployment, update your OAuth apps with production callback URLs:

**Google Cloud Console:**
```
https://your-backend.onrender.com/api/auth/callback/google
```

**GitHub Developer Settings:**
```
https://your-backend.onrender.com/api/auth/callback/github
```

## Testing Checklist

After setup, verify the following:

- [ ] Google OAuth button on login page redirects to Google consent screen
- [ ] After Google consent, user is redirected to homepage and logged in
- [ ] User's Google display name appears in navbar
- [ ] GitHub OAuth button redirects to GitHub authorization
- [ ] After GitHub authorization, user is logged in with GitHub name
- [ ] OAuth users can use the chat widget
- [ ] Session persists across page refreshes
- [ ] Sign out works correctly for OAuth users

## Troubleshooting

### "redirect_uri_mismatch" Error

The callback URL in your OAuth app doesn't match what the backend is sending. Ensure:
1. The `API_URL` environment variable matches your backend URL exactly
2. The redirect URI in Google/GitHub settings matches `{API_URL}/api/auth/callback/{provider}`

### "access_denied" Error

1. For Google: Check that the OAuth consent screen is properly configured
2. For GitHub: Ensure the app has access to the user's email

### Cookies Not Being Set

Cross-domain cookie issues require:
1. `SameSite=None` and `Secure=true` on cookies
2. HTTPS in production
3. Proper CORS configuration with `credentials: true`

### User Not Created

Check the database migration has been applied:
```sql
-- Verify columns exist
SELECT column_name FROM information_schema.columns
WHERE table_name = 'users' AND column_name IN ('auth_provider', 'oauth_provider_id');
```

## Security Notes

1. **Never commit secrets** - Always use environment variables
2. **Use HTTPS in production** - Required for secure cookies
3. **Keep client secrets secure** - Rotate if exposed
4. **Review OAuth scopes** - Only request what you need
5. **Monitor OAuth usage** - Check Google/GitHub console for unusual activity
