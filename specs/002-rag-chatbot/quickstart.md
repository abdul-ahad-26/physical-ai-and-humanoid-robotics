# Quickstart: RAG Chatbot for Docusaurus Textbook

**Feature Branch**: `002-rag-chatbot`
**Date**: 2025-12-17

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- uv (Python package manager)
- npm or yarn
- Git

## External Services Setup

### 1. Neon Serverless Postgres

1. Create account at [neon.tech](https://neon.tech)
2. Create a new project
3. Copy the **pooled** connection string (contains `-pooler` in hostname)
4. Save as `DATABASE_URL` in `.env`

### 2. Qdrant Cloud

1. Create account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster
3. Copy the cluster URL and API key
4. Save as `QDRANT_URL` and `QDRANT_API_KEY` in `.env`

### 3. OpenAI API

1. Create API key at [platform.openai.com](https://platform.openai.com)
2. Save as `OPENAI_API_KEY` in `.env`

---

## Repository Structure

```
physical-ai-and-humanoid-robotics/
├── frontend/                    # Docusaurus site with ChatKit
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatWidget/      # ChatKit integration
│   │   └── theme/               # Docusaurus theme customizations
│   ├── docs/                    # Textbook content
│   └── package.json
│
├── backend/                     # FastAPI backend
│   ├── src/
│   │   ├── agents/              # OpenAI Agents SDK implementations
│   │   ├── api/                 # FastAPI routes
│   │   ├── db/                  # Database models and queries
│   │   └── services/            # Business logic
│   ├── tests/
│   ├── pyproject.toml           # uv project file
│   └── Dockerfile
│
└── specs/                       # Feature specifications
    └── 002-rag-chatbot/
```

---

## Backend Setup

### 1. Initialize FastAPI Project with uv

```bash
cd backend

# Initialize uv project
uv init

# Add dependencies
uv add fastapi uvicorn[standard] asyncpg qdrant-client openai tiktoken httpx python-dotenv pydantic

# Add OpenAI Agents SDK
uv add openai-agents

# Add dev dependencies
uv add --dev pytest pytest-asyncio httpx
```

### 2. Create Environment File

```bash
# backend/.env
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.neon.tech/dbname?sslmode=require
QDRANT_URL=https://xxx.qdrant.cloud:6333
QDRANT_API_KEY=your-qdrant-api-key
OPENAI_API_KEY=sk-your-openai-key
BETTER_AUTH_URL=http://localhost:3000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 3. Initialize Database Schema

```bash
# Run the schema migration
uv run python -c "
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

SCHEMA = '''
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    vector_ids UUID[] NOT NULL,
    similarity_scores FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    latency_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
'''

async def init_db():
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    await conn.execute(SCHEMA)
    await conn.close()
    print('Database initialized')

asyncio.run(init_db())
"
```

### 4. Initialize Qdrant Collection

```bash
uv run python -c "
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance
import os
from dotenv import load_dotenv

load_dotenv()

async def init_qdrant():
    client = AsyncQdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY')
    )

    # Check if collection exists
    collections = await client.get_collections()
    if 'textbook_chunks' not in [c.name for c in collections.collections]:
        await client.create_collection(
            collection_name='textbook_chunks',
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print('Qdrant collection created')
    else:
        print('Qdrant collection already exists')

asyncio.run(init_qdrant())
"
```

### 5. Run Backend

```bash
uv run uvicorn src.main:app --reload --port 8000
```

---

## Frontend Setup

### 1. Install ChatKit and Better Auth

```bash
cd frontend

# Install ChatKit
npm install @openai/chatkit-react

# Install Better Auth client
npm install @better-auth/react
```

### 2. Create ChatWidget Component

Create `frontend/src/components/ChatWidget/index.tsx`:

```tsx
import React from 'react';
import { ChatKit, useChatKit } from '@openai/chatkit-react';
import { useSession } from '@better-auth/react';

export function ChatWidget() {
  const { data: session, isPending } = useSession();

  const { control } = useChatKit({
    api: {
      url: process.env.NEXT_PUBLIC_API_URL + '/api/chatkit',
      domainKey: 'textbook-rag'
    },
    theme: {
      colorScheme: 'light',
      color: {
        accent: { primary: '#10B981' }
      }
    },
    startScreen: {
      greeting: 'Ask me about this textbook!',
      prompts: [
        { label: 'Explain a concept', prompt: 'Explain...', icon: 'lightbulb' }
      ]
    },
    composer: {
      placeholder: 'Ask a question about the book...'
    }
  });

  // Hide for unauthenticated users
  if (isPending || !session) {
    return null;
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <ChatKit control={control} className="h-[600px] w-[360px]" />
    </div>
  );
}
```

### 3. Add to Docusaurus Layout

Edit `frontend/src/theme/Root.tsx`:

```tsx
import React from 'react';
import { ChatWidget } from '../components/ChatWidget';

export default function Root({ children }) {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
}
```

### 4. Run Frontend

```bash
npm run start
```

---

## Testing

### Backend Tests

```bash
cd backend
uv run pytest tests/ -v
```

### Manual Testing Checklist

- [ ] Create user account via Better Auth
- [ ] Log in and see chat widget appear
- [ ] Ask a question about book content
- [ ] Verify answer includes citations
- [ ] Click citation link and navigate to section
- [ ] Ask question with no relevant content, get "I don't know based on this book."
- [ ] Select text on page, ask question, verify context is used
- [ ] Log out and verify chat widget disappears
- [ ] Log back in and verify conversation history

---

## Deployment

### Backend to Render

1. Create `backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Create `render.yaml`:

```yaml
services:
  - type: web
    name: rag-chatbot-api
    env: docker
    plan: free
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: QDRANT_URL
        sync: false
      - key: QDRANT_API_KEY
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      - key: BETTER_AUTH_URL
        sync: false
      - key: CORS_ORIGINS
        sync: false
```

3. Connect GitHub repo to Render and deploy

### Frontend

Deploy with existing Docusaurus hosting (Vercel/Netlify) - just add environment variable:

```
NEXT_PUBLIC_API_URL=https://your-render-api.onrender.com
```

---

## Environment Variables Summary

| Variable | Location | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Backend | Neon pooled connection string |
| `QDRANT_URL` | Backend | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Backend | Qdrant Cloud API key |
| `OPENAI_API_KEY` | Backend | OpenAI API key |
| `BETTER_AUTH_URL` | Backend | URL where Better Auth is running |
| `CORS_ORIGINS` | Backend | Allowed CORS origins (comma-separated) |
| `NEXT_PUBLIC_API_URL` | Frontend | Backend API URL |
