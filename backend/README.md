# RAG Chatbot Backend

FastAPI backend for the RAG-powered chatbot that answers questions about the Physical AI and Humanoid Robotics textbook.

## Features

- Multi-agent RAG workflow using OpenAI Agents SDK
- Vector search with Qdrant for relevant content retrieval
- Session management with PostgreSQL (Neon)
- Better Auth integration for authentication
- Rate limiting and input/output guardrails

## Setup

1. Copy `.env.example` to `.env` and fill in the values
2. Install dependencies: `uv sync`
3. Run the server: `uv run uvicorn src.main:app --reload`

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Send a chat message
- `GET /api/sessions` - List user sessions
- `GET /api/sessions/{id}/messages` - Get session messages
- `POST /api/ingest` - Ingest content (admin only)

## Testing

```bash
uv run pytest -v
```
