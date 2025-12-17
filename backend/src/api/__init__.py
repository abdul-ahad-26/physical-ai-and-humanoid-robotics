"""FastAPI routes for the RAG chatbot API."""

from .chat import router as chat_router
from .sessions import router as sessions_router
from .ingest import router as ingest_router
from .health import router as health_router

__all__ = [
    "chat_router",
    "sessions_router",
    "ingest_router",
    "health_router",
]
