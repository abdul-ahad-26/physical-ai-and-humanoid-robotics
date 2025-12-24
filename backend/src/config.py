"""Configuration module with environment variable loading."""

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = "textbook_chunks"

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Personalization Cache (005-user-personalization)
    personalization_cache_ttl: int = int(os.getenv("PERSONALIZATION_CACHE_TTL", "3600"))
    translation_cache_ttl: int = int(os.getenv("TRANSLATION_CACHE_TTL", "3600"))

    # Better Auth
    better_auth_url: str = os.getenv("BETTER_AUTH_URL", "http://localhost:3000")

    # OAuth Provider Settings
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    github_client_id: str = os.getenv("GITHUB_CLIENT_ID", "")
    github_client_secret: str = os.getenv("GITHUB_CLIENT_SECRET", "")

    # API URL for OAuth callbacks
    api_url: str = os.getenv("API_URL", "http://localhost:8000")
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Cookie security - disable secure cookies for localhost development
    @property
    def is_development(self) -> bool:
        """Check if running in development mode (localhost)."""
        return "localhost" in self.api_url or "127.0.0.1" in self.api_url

    @property
    def cookie_secure(self) -> bool:
        """Cookie secure flag - False for localhost, True for production."""
        return not self.is_development

    @property
    def cookie_samesite(self) -> str:
        """Cookie SameSite policy - Lax for localhost, None for cross-domain production."""
        return "lax" if self.is_development else "none"

    # CORS
    cors_origins: List[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    ]

    # RAG Settings
    similarity_threshold: float = 0.5  # Lowered from 0.7 to capture more relevant results
    top_k_results: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Rate Limiting
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    model_config = {"extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
