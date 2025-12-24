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

    # Debug: Print CORS origins at startup
    print(f"[CORS] Configured origins: {settings.cors_origins}")

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Set-Cookie"],
    )

    # Register routers
    from src.api.health import router as health_router
    from src.api.chat import router as chat_router
    from src.api.sessions import router as sessions_router
    from src.api.ingest import router as ingest_router
    from src.api.auth import router as auth_router
    from src.api.oauth import router as oauth_router
    from src.api.profile import router as profile_router
    from src.api.personalize import router as personalize_router
    from src.api.translate import router as translate_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(oauth_router)
    app.include_router(profile_router)  # 005-user-personalization
    app.include_router(personalize_router)  # 005-user-personalization
    app.include_router(translate_router)  # 005-user-personalization
    app.include_router(chat_router, prefix="/api")
    app.include_router(sessions_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")

    return app


# Create app instance
app = create_app()
