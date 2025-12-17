"""FastAPI app entry point with CORS, middleware, and router registration."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.db.connection import close_db_pool, init_db_schema
from src.services.qdrant import close_qdrant_client, init_qdrant_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    try:
        await init_db_schema()
        await init_qdrant_collection()
    except Exception as e:
        print(f"Warning: Failed to initialize services: {e}")

    yield

    # Shutdown
    await close_db_pool()
    await close_qdrant_client()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="RAG Chatbot API",
        description="RAG Chatbot backend for Docusaurus textbook",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register routers
    from src.api.health import router as health_router
    from src.api.chat import router as chat_router
    from src.api.sessions import router as sessions_router
    from src.api.ingest import router as ingest_router

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api")
    app.include_router(sessions_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")

    return app


# Create app instance
app = create_app()
