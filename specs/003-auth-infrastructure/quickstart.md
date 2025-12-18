# Quickstart: Better Auth Authentication & Infrastructure Setup

**Feature**: 003-auth-infrastructure
**Date**: 2025-12-17
**Status**: ✅ COMPLETED (2025-12-18)
**Purpose**: Step-by-step guide to set up authentication and infrastructure for the RAG chatbot

**⚠️ Note**: This guide reflects the actual implementation. See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for differences from original plan (custom chat UI, direct ingestion, etc.)

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Neon Serverless Postgres** account with database created
- [ ] **Qdrant Cloud** account with cluster created
- [ ] **OpenAI API** key with access to Embeddings and Chat Completion APIs
- [ ] **Node.js 18+** installed
- [ ] **Python 3.11+** installed
- [ ] **Git** repository cloned locally

---

## Step 1: Configure Environment Variables

### Backend Configuration

Create `backend/.env`:

```bash
# Database
DATABASE_URL=postgresql://user:password@host.neon.tech:5432/database?sslmode=require

# Qdrant Vector Store
QDRANT_URL=https://xyz-abc.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key

# Better Auth
AUTH_SECRET=generate-random-secret-key-here

# Optional: API URL for scripts
API_URL=http://localhost:8000
```

**Generate AUTH_SECRET**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Frontend Configuration

Create `frontend/.env`:

```bash
# Backend API URL
REACT_APP_API_URL=http://localhost:8000
```

---

## Step 2: Install Dependencies

### Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Required packages** (add to `requirements.txt` if missing):
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
asyncpg>=0.29.0
argon2-cffi>=23.1.0
python-dotenv>=1.0.0
httpx>=0.25.0
python-frontmatter>=1.0.0
openai>=1.0.0
qdrant-client>=1.7.0
```

### Frontend Dependencies

```bash
cd frontend
npm install better-auth
```

---

## Step 3: Initialize Database Schema

Run the database initialization script to create all required tables:

```bash
cd backend
python scripts/init_db.py
```

**Expected Output**:
```
Connecting to database...
✓ Connected successfully
Creating table: users
✓ Created table: users
Adding password_hash column to users
✓ Added column: password_hash
Creating table: sessions
✓ Created table: sessions
Creating table: messages
✓ Created table: messages
Creating table: retrieval_logs
✓ Created table: retrieval_logs
Creating table: performance_metrics
✓ Created table: performance_metrics
Creating indexes...
✓ Created all indexes

Summary:
- Tables created: 5
- Indexes created: 6
- Schema initialization complete
```

**Idempotency Check** (run again to verify):
```bash
python scripts/init_db.py
```

Expected: "Table already exists" messages with no errors.

---

## Step 4: Ingest Textbook Content

Run the content ingestion script to populate Qdrant with textbook content:

```bash
cd backend
python scripts/ingest_docs.py
```

**Expected Output**:
```
Scanning directory: /path/to/frontend/docs
Found 52 markdown files

Processing files:
✓ chapter-1/introduction.md: 12 chunks created
✓ chapter-1/overview.md: 8 chunks created
✓ chapter-2/basics.md: 15 chunks created
...

Summary:
- Files processed: 52
- Chunks created: 687
- Errors: 0
- Duration: 3m 24s

Ingestion complete. Chatbot ready to answer questions.
```

**Troubleshooting**:
- **"No markdown files found"**: Check that `frontend/docs/` exists and contains `.md` files
- **"API call failed"**: Verify backend is running and `API_URL` in `.env` is correct
- **"Qdrant connection error"**: Check `QDRANT_URL` and `QDRANT_API_KEY` in `.env`

---

## Step 5: Set Up Better Auth on Frontend

### Create Auth Client Configuration

Create `frontend/src/lib/auth.ts`:

```typescript
import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient({
  baseURL: process.env.REACT_APP_API_URL || "http://localhost:8000",
});

export const { useSession, signIn, signUp, signOut } = authClient;
```

### Create Login Page

Create `frontend/src/pages/login.tsx`:

```tsx
import React, { useState } from "react";
import { signIn } from "@site/src/lib/auth";
import styles from "./login.module.css";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await signIn.email({ email, password });
      window.location.href = "/";
    } catch (err: any) {
      setError(err.message || "Invalid email or password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1>Log In</h1>
        <form onSubmit={handleSubmit}>
          <div className={styles.field}>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className={styles.field}>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              placeholder="Minimum 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button type="submit" disabled={loading} className={styles.button}>
            {loading ? "Logging in..." : "Log In"}
          </button>

          <p className={styles.link}>
            Don't have an account? <a href="/signup">Sign up</a>
          </p>
        </form>
      </div>
    </div>
  );
}
```

### Create Signup Page

Create `frontend/src/pages/signup.tsx` (similar structure to login, using `signUp.email()`).

### Create Styles

Create `frontend/src/pages/login.module.css`:

```css
.container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 200px);
  padding: 2rem;
}

.card {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  max-width: 400px;
  width: 100%;
}

.card h1 {
  margin-bottom: 1.5rem;
  color: #1a1a1a;
  font-size: 1.75rem;
}

.field {
  margin-bottom: 1.25rem;
}

.field label {
  display: block;
  margin-bottom: 0.5rem;
  color: #333;
  font-weight: 500;
}

.field input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 1rem;
}

.field input:focus {
  outline: none;
  border-color: #10B981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.button {
  width: 100%;
  padding: 0.875rem;
  background: #10B981;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.button:hover:not(:disabled) {
  background: #059669;
}

.button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error {
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: #fee;
  border: 1px solid #fcc;
  border-radius: 8px;
  color: #c33;
  font-size: 0.875rem;
}

.link {
  margin-top: 1.5rem;
  text-align: center;
  color: #666;
  font-size: 0.875rem;
}

.link a {
  color: #10B981;
  text-decoration: none;
  font-weight: 600;
}

.link a:hover {
  text-decoration: underline;
}
```

---

## Step 6: Update Root Component

Update `frontend/src/theme/Root.tsx` to use Better Auth session:

```tsx
import React, { ReactNode } from 'react';
import { ChatWidget } from '../components/ChatWidget';
import { useSession } from '@site/src/lib/auth';

interface RootProps {
  children: ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  const { data: session, isLoading } = useSession();

  return (
    <>
      {children}
      <ChatWidget
        isAuthenticated={!!session && !isLoading}
        session={session}
      />
    </>
  );
}
```

---

## Step 7: Add Backend Authentication Routes

Create `backend/routers/auth.py`:

```python
from fastapi import APIRouter, HTTPException, Response, Cookie
from pydantic import BaseModel, EmailStr
from argon2 import PasswordHasher
import asyncpg
import uuid
from datetime import datetime, timedelta
import secrets

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
ph = PasswordHasher()

class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/signup")
async def signup(request: SignupRequest, response: Response, db: asyncpg.Connection):
    if len(request.password) < 8:
        raise HTTPException(400, detail={"code": "AUTH_005", "message": "Password must be at least 8 characters"})

    # Check if user exists
    existing = await db.fetchrow("SELECT id FROM users WHERE email = $1", request.email)
    if existing:
        raise HTTPException(409, detail={"code": "AUTH_002", "message": "Email already registered"})

    # Create user
    password_hash = ph.hash(request.password)
    user = await db.fetchrow("""
        INSERT INTO users (email, password_hash, created_at)
        VALUES ($1, $2, NOW())
        RETURNING id, email, display_name, created_at
    """, request.email, password_hash)

    # Create session
    session_token = secrets.token_urlsafe(64)
    expires_at = datetime.utcnow() + timedelta(days=7)

    await db.execute("""
        INSERT INTO auth_sessions (user_id, session_token, expires_at)
        VALUES ($1, $2, $3)
    """, user['id'], session_token, expires_at)

    # Set cookie
    response.set_cookie(
        key="session",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=604800  # 7 days
    )

    return {
        "user": dict(user),
        "session": {"token": session_token, "expires_at": expires_at.isoformat()}
    }

@router.post("/login")
async def login(request: LoginRequest, response: Response, db: asyncpg.Connection):
    # Get user
    user = await db.fetchrow("SELECT * FROM users WHERE email = $1", request.email)
    if not user:
        raise HTTPException(401, detail={"code": "AUTH_001", "message": "Invalid email or password"})

    # Verify password
    try:
        ph.verify(user['password_hash'], request.password)
    except:
        raise HTTPException(401, detail={"code": "AUTH_001", "message": "Invalid email or password"})

    # Create session
    session_token = secrets.token_urlsafe(64)
    expires_at = datetime.utcnow() + timedelta(days=7)

    await db.execute("""
        INSERT INTO auth_sessions (user_id, session_token, expires_at)
        VALUES ($1, $2, $3)
    """, user['id'], session_token, expires_at)

    # Update last_login
    await db.execute("UPDATE users SET last_login = NOW() WHERE id = $1", user['id'])

    # Set cookie
    response.set_cookie(
        key="session",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=604800
    )

    return {
        "user": {"id": str(user['id']), "email": user['email'], "display_name": user['display_name']},
        "session": {"token": session_token, "expires_at": expires_at.isoformat()}
    }

@router.post("/logout")
async def logout(response: Response, session_token: str = Cookie(None, alias="session"), db: asyncpg.Connection):
    if not session_token:
        raise HTTPException(401, detail={"code": "AUTH_003", "message": "No active session"})

    await db.execute("DELETE FROM auth_sessions WHERE session_token = $1", session_token)

    response.delete_cookie("session")
    return {"message": "Logged out successfully"}

@router.get("/session")
async def get_session(session_token: str = Cookie(None, alias="session"), db: asyncpg.Connection):
    if not session_token:
        raise HTTPException(401, detail={"code": "AUTH_003", "message": "No active session"})

    session = await db.fetchrow("""
        SELECT s.expires_at, u.id, u.email, u.display_name
        FROM auth_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_token = $1 AND s.expires_at > NOW()
    """, session_token)

    if not session:
        raise HTTPException(401, detail={"code": "AUTH_003", "message": "Session expired or invalid"})

    return {
        "user": {"id": str(session['id']), "email": session['email'], "display_name": session['display_name']},
        "session": {"expires_at": session['expires_at'].isoformat()}
    }
```

Add to `backend/main.py`:
```python
from routers.auth import router as auth_router

app.include_router(auth_router)
```

---

## Step 8: Start the Application

### Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Start Frontend

```bash
cd frontend
npm start
```

---

## Step 9: Test the Complete Flow

### 1. Test Signup
1. Navigate to `http://localhost:3000/signup`
2. Enter email and password (min 8 characters)
3. Click "Sign Up"
4. Verify redirect to homepage
5. Check that chat icon shows chat interface (not login prompt)

### 2. Test Login
1. Log out (if logged in)
2. Navigate to `http://localhost:3000/login`
3. Enter credentials from signup
4. Click "Log In"
5. Verify redirect to homepage

### 3. Test Authenticated Chat
1. Log in
2. Click chat icon
3. Send message: "What is reinforcement learning?"
4. Verify AI response with citations

### 4. Test Logout
1. Click logout button in navbar
2. Verify redirect to login page
3. Click chat icon
4. Verify login prompt appears

---

## Verification Checklist

- [ ] Database initialized with all tables
- [ ] Textbook content ingested (500+ chunks)
- [ ] User can sign up for new account
- [ ] User can log in with credentials
- [ ] Session persists across page reloads
- [ ] Authenticated user sees chat interface
- [ ] Chatbot returns AI-generated answers with citations
- [ ] User can log out
- [ ] Session expires after 7 days
- [ ] Protected API endpoints return 401 for unauthenticated requests

---

## Troubleshooting

### "Database connection error"
- Check `DATABASE_URL` in `backend/.env`
- Verify Neon Postgres database is active
- Test connection: `psql $DATABASE_URL`

### "Qdrant connection error"
- Check `QDRANT_URL` and `QDRANT_API_KEY` in `backend/.env`
- Verify Qdrant cluster is running
- Test: `curl -H "api-key: $QDRANT_API_KEY" $QDRANT_URL/collections`

### "Better Auth session not working"
- Check browser cookies (DevTools → Application → Cookies)
- Verify `session` cookie is set with `HttpOnly` and `Secure` flags
- Check backend logs for authentication errors

### "Chat widget shows login prompt after login"
- Check `useSession` hook is properly configured
- Verify `REACT_APP_API_URL` in `frontend/.env`
- Check browser console for errors

---

## Next Steps

After completing quickstart:

1. **Production Deployment**: Update environment variables for production URLs
2. **HTTPS Setup**: Configure SSL certificates for secure cookies
3. **Monitoring**: Add logging and metrics collection
4. **Testing**: Run integration tests for auth flows
5. **Performance**: Monitor response times and optimize queries

For detailed implementation, see:
- **Research**: `specs/003-auth-infrastructure/research.md`
- **Data Model**: `specs/003-auth-infrastructure/data-model.md`
- **API Contracts**: `specs/003-auth-infrastructure/contracts/auth-api.yaml`
