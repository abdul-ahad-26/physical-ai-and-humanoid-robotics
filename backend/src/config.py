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

    # Better Auth
    better_auth_url: str = os.getenv("BETTER_AUTH_URL", "http://localhost:3000")

    # CORS
    cors_origins: List[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    ]

    # RAG Settings
    similarity_threshold: float = 0.7
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
