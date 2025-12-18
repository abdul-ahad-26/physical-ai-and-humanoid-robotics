# RAG Chatbot Backend

FastAPI backend for the RAG-powered chatbot that answers questions about the Physical AI and Humanoid Robotics textbook.

## Features

- Multi-agent RAG workflow using OpenAI Agents SDK
- Vector search with Qdrant for relevant content retrieval
- Session management with PostgreSQL (Neon)
- Better Auth integration for authentication
- Rate limiting and input/output guardrails
- **Production-ready**: Configured for Render deployment

## Quick Start

### 1. Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `DATABASE_URL` - Neon Postgres connection string
- `QDRANT_URL` - Qdrant Cloud cluster URL
- `QDRANT_API_KEY` - Qdrant API key
- `OPENAI_API_KEY` - OpenAI API key
- `AUTH_SECRET` - Random secret for session signing (generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)

### 2. Install Dependencies

```bash
uv sync
# or
pip install -r requirements.txt
```

### 3. Initialize Database

Run the database initialization script to create all required tables:

```bash
python scripts/init_db.py
```

This script is idempotent - safe to run multiple times.

### 4. Ingest Textbook Content

Run the content ingestion script to populate Qdrant with textbook content:

```bash
python scripts/ingest_docs.py
```

This processes all markdown files in `frontend/docs/` and creates vector embeddings.

### 5. Start the Server

```bash
uv run uvicorn src.main:app --reload --port 8000
# or
uvicorn src.main:app --reload --port 8000
```

Server will be available at: `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Endpoints

### Authentication

- `POST /api/auth/signup` - Create new user account
- `POST /api/auth/login` - Log in existing user
- `POST /api/auth/logout` - Log out current user
- `GET /api/auth/session` - Check current session

### Chat

- `POST /api/chat` - Send a chat message (requires authentication)
- `GET /api/sessions` - List user's chat sessions (requires authentication)
- `GET /api/sessions/{id}/messages` - Get session messages (requires authentication)

### Content Management

- `POST /api/ingest` - Ingest markdown content (admin only)

### Health

- `GET /health` - Health check

## Testing

```bash
uv run pytest -v
# or
pytest -v
```

## Recent Fixes (2025-12-18)

### Better Auth Integration
- ✅ Fixed Better Auth React client configuration with `credentials: "include"`
- ✅ Backend auth endpoints properly configured with cookie-based sessions
- ✅ Session validation middleware working correctly with direct database queries
- ✅ Auth router includes Better Auth-compatible aliases: `/sign-up/email` and `/sign-in/email`

### ChatKit Integration
- ✅ Fixed ChatKit component to use `useChatKit` hook properly
- ✅ Removed incorrect `options` prop, now using `control` object from hook
- ✅ Authentication state properly passed to ChatWidget component
- ✅ ChatKit properly integrated with RAG backend endpoint

### Frontend Fixes
- ✅ Updated Root.tsx to use `isPending` instead of `isLoading` (matches Better Auth API)
- ✅ Auth client now properly exports methods from `authClient`
- ✅ Package.json has correct dependencies: `better-auth@^1.4.7` and `@openai/chatkit-react@^1.4.0`

## Troubleshooting

See detailed troubleshooting guide in `specs/003-auth-infrastructure/quickstart.md`

Common issues:
- **Database connection error**: Check `DATABASE_URL` in `.env`
- **Qdrant connection error**: Verify `QDRANT_URL` and `QDRANT_API_KEY`
- **Session cookie not set**: Ensure `CORS_ORIGINS` includes frontend URL and `allow_credentials=True`
- **Import errors**: Run `pip install -r requirements.txt`
- **ChatKit not loading**: Verify `/api/chat` endpoint accessible and session valid

## Production Deployment

### Deploy to Render

This backend is configured for easy deployment to Render:

```bash
# 1. Push your code to GitHub
git push origin main

# 2. Follow the step-by-step guide
See DEPLOYMENT.md for complete instructions
```

**Quick Deploy:**
1. Create account at https://render.com
2. Connect your GitHub repository
3. Set Root Directory to `backend`
4. Add environment variables (see DEPLOYMENT.md)
5. Deploy!

**Required Environment Variables for Production:**
- `DATABASE_URL` - Your Neon Postgres connection string
- `QDRANT_URL` - Your Qdrant Cloud cluster URL
- `QDRANT_API_KEY` - Your Qdrant API key
- `OPENAI_API_KEY` - Your OpenAI API key
- `AUTH_SECRET` - Generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- `BETTER_AUTH_URL` - Your frontend URL (e.g., https://your-app.vercel.app)
- `CORS_ORIGINS` - Your frontend URL

📖 **See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment guide**

---

## Documentation

For detailed implementation documentation, see:
- **Deployment Guide**: `DEPLOYMENT.md` ⭐ **Start here for production**
- **Feature Spec**: `specs/003-auth-infrastructure/spec.md`
- **Implementation Plan**: `specs/003-auth-infrastructure/plan.md`
- **Data Model**: `specs/003-auth-infrastructure/data-model.md`
- **API Contracts**: `specs/003-auth-infrastructure/contracts/auth-api.yaml`
- **Quickstart Guide**: `specs/003-auth-infrastructure/quickstart.md`
