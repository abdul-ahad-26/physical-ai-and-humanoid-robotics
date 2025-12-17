# Research: RAG Chatbot for Docusaurus Textbook

**Feature Branch**: `002-rag-chatbot`
**Date**: 2025-12-17
**Status**: Complete

---

## 1. OpenAI Agents SDK Integration

### Decision
Use OpenAI Agents SDK (Python) with multi-agent handoff pattern for RAG workflow orchestration.

### Rationale
- Official OpenAI library with first-class support for agent workflows
- Built-in support for `function_tool`, `handoffs`, `guardrails`, and `Runner`
- Pydantic integration for structured output validation
- Async/await pattern aligns with FastAPI's async design
- Model recommendation: `gpt-4o-mini` for cost-effective RAG (per documentation)

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| LangChain | More complex abstraction layer; OpenAI SDK is more direct |
| Custom orchestration | Reinventing the wheel; SDK provides battle-tested patterns |
| LlamaIndex | Good for RAG but less flexible for multi-agent workflows |

### Implementation Pattern
```python
from agents import Agent, Runner, function_tool, handoff, InputGuardrail

# Retrieval Agent with function_tool for Qdrant queries
@function_tool
async def search_book_content(query: str, selected_text: str = "") -> list:
    """Search textbook content in Qdrant"""
    # Qdrant search implementation
    pass

retrieval_agent = Agent(
    name="Retrieval Agent",
    instructions="Find relevant textbook content for user queries",
    tools=[search_book_content]
)

# Answer Generation Agent with book-only constraint
answer_agent = Agent(
    name="Answer Agent",
    model="gpt-4o-mini",
    instructions="Generate answers ONLY from provided context. If insufficient, say 'I don't know based on this book.'"
)

# Multi-agent workflow with handoffs
triage_agent = Agent(
    name="RAG Triage",
    handoffs=[retrieval_agent, answer_agent, citation_agent]
)
```

---

## 2. Better Auth Integration with FastAPI

### Decision
Use Better Auth's session validation API (`auth.api.getSession()`) called from FastAPI middleware.

### Rationale
- Better Auth is TypeScript-first but provides HTTP API for session validation
- FastAPI can validate sessions by calling Better Auth's session endpoint
- Frontend uses `@better-auth/react` hooks (`useSession`) for client-side state
- Session cookies are passed through to FastAPI for validation

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| JWT-only validation | Loses Better Auth's session management features |
| Separate Python auth | Duplicates logic; Better Auth already handles this |
| Firebase Auth | External dependency; Better Auth is self-hosted |

### Implementation Pattern
```python
# FastAPI middleware calling Better Auth session endpoint
from fastapi import Request, HTTPException
import httpx

async def validate_session(request: Request):
    session_cookie = request.cookies.get("better-auth.session_token")
    if not session_cookie:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Call Better Auth API to validate session
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BETTER_AUTH_URL}/api/auth/get-session",
            cookies={"better-auth.session_token": session_cookie}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session")
        return response.json()
```

### Frontend Pattern
```typescript
// Docusaurus frontend with Better Auth
import { useSession } from "@better-auth/react";

function ChatWidget() {
  const { data: session, isPending } = useSession();

  if (!session) return null; // Hide for unauthenticated users

  return <ChatKit control={control} />;
}
```

---

## 3. Qdrant Cloud Vector Storage

### Decision
Use Qdrant Cloud (free tier) with `qdrant-client` Python SDK for vector storage and similarity search.

### Rationale
- Free tier supports 1GB storage, 1M vectors - sufficient for textbook
- Native Python async client (`AsyncQdrantClient`)
- Built-in payload filtering for metadata queries
- Cosine distance for semantic similarity

### Configuration
```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Collection configuration
COLLECTION_NAME = "textbook_chunks"
VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small
DISTANCE = Distance.COSINE

# Search configuration
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7
```

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Pinecone | No free tier with similar features |
| Weaviate | More complex setup for this use case |
| ChromaDB | Less production-ready than Qdrant Cloud |
| pgvector | Adds complexity to Neon setup |

---

## 4. Neon Serverless Postgres

### Decision
Use Neon Serverless Postgres with `asyncpg` for async database operations.

### Rationale
- Free tier: 0.5GB storage, 3GB transfer - sufficient for initial user base
- Native async support with `asyncpg`
- Connection pooling via Neon's `-pooler` endpoint
- Auto-scaling compute for serverless workloads

### Connection Pattern
```python
import asyncpg
import os

DATABASE_URL = os.getenv("DATABASE_URL")  # Neon pooler URL

async def get_db_pool():
    return await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=10,
        ssl="require"
    )
```

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Supabase | More opinionated; Neon is pure Postgres |
| PlanetScale | MySQL-based, not Postgres |
| CockroachDB | Overkill for this scale |

---

## 5. OpenAI ChatKit Frontend

### Decision
Use `@openai/chatkit-react` for the chat UI with custom theming.

### Rationale
- Official OpenAI component library for chat interfaces
- Built-in streaming support
- Deep customization via `useChatKit` hook
- Handles message rendering, input, and history automatically

### Configuration
```javascript
import { ChatKit, useChatKit } from '@openai/chatkit-react';

const { control } = useChatKit({
  api: {
    url: '/api/chatkit',
    domainKey: 'textbook-rag'
  },
  theme: {
    colorScheme: 'light',
    color: { accent: { primary: '#10B981' } }  // Green
  },
  startScreen: {
    greeting: 'Ask me about this textbook!'
  }
});
```

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Custom React components | Reinventing the wheel |
| Vercel AI SDK UI | Less integrated with OpenAI ecosystem |
| Streamlit | Not suitable for Docusaurus integration |

---

## 6. Deployment Architecture

### Decision
- **Frontend**: Deploy with Docusaurus to existing hosting (Vercel/Netlify)
- **Backend**: Deploy FastAPI to Render (free tier with Docker)

### Rationale
- Render provides free tier for web services with Docker support
- Separation allows independent scaling
- CORS configuration for cross-origin requests

### Render Configuration
```yaml
# render.yaml
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
```

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Railway | No free tier |
| Fly.io | More complex setup for this use case |
| AWS Lambda | Cold starts impact chat latency |

---

## 7. Token-Based Chunking Strategy

### Decision
Use `tiktoken` with 512-token chunks and 50-token overlap.

### Rationale
- 512 tokens balances context richness with retrieval precision
- 50-token overlap ensures sentence continuity across chunks
- `tiktoken` is OpenAI's official tokenizer for accurate counting

### Implementation
```python
import tiktoken

def chunk_markdown(content: str, chunk_size: int = 512, overlap: int = 50) -> list:
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoder.encode(content)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoder.decode(chunk_tokens))
        start += chunk_size - overlap

    return chunks
```

---

## 8. Environment Variables

### Required Variables
```bash
# Backend (.env)
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.neon.tech/dbname?sslmode=require
QDRANT_URL=https://xxx.qdrant.cloud:6333
QDRANT_API_KEY=xxx
OPENAI_API_KEY=sk-xxx
BETTER_AUTH_URL=http://localhost:3000  # or production URL
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# Frontend (.env)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Summary

All technical unknowns have been resolved:

| Area | Decision |
|------|----------|
| Agent Orchestration | OpenAI Agents SDK (Python) |
| Authentication | Better Auth with HTTP session validation |
| Vector Storage | Qdrant Cloud (free tier) |
| Relational Storage | Neon Serverless Postgres |
| Frontend UI | OpenAI ChatKit (`@openai/chatkit-react`) |
| Deployment | Render (FastAPI) + existing Docusaurus host |
| Chunking | tiktoken, 512 tokens, 50 overlap |
| Model | gpt-4o-mini (cost-effective for RAG) |
