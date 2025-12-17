"""Pytest configuration and fixtures for backend tests."""

import asyncio
import os
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Set test environment variables before importing app
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test-key"
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["BETTER_AUTH_URL"] = "http://localhost:3000"
os.environ["CORS_ORIGINS"] = "http://localhost:3000"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    from src.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def mock_user_id() -> str:
    """Generate a mock user ID."""
    return str(uuid4())


@pytest.fixture
def mock_session_id() -> str:
    """Generate a mock session ID."""
    return str(uuid4())


@pytest.fixture
def mock_auth_headers(mock_user_id: str) -> dict:
    """Create mock authentication headers."""
    return {
        "Cookie": "better-auth.session_token=mock-token",
    }


@pytest.fixture
def sample_chat_request() -> dict:
    """Create a sample chat request."""
    return {
        "message": "What is physical AI?",
        "session_id": None,
        "selected_text": None,
    }


@pytest.fixture
def sample_ingest_request() -> dict:
    """Create a sample ingest request."""
    return {
        "content": "# Introduction to Physical AI\n\nPhysical AI refers to artificial intelligence systems that interact with the physical world through sensors and actuators.",
        "chapter_id": "chapter-1",
        "section_id": "introduction",
        "anchor_url": "/docs/chapter-1#introduction",
        "source_file": "docs/chapter-1.md",
    }


@pytest.fixture
def sample_chunks() -> list:
    """Create sample retrieved chunks for testing."""
    return [
        {
            "content": "Physical AI refers to artificial intelligence systems that interact with the physical world.",
            "chapter_id": "chapter-1",
            "section_id": "introduction",
            "anchor_url": "/docs/chapter-1#introduction",
            "score": 0.92,
        },
        {
            "content": "These systems use sensors to perceive their environment and actuators to manipulate it.",
            "chapter_id": "chapter-1",
            "section_id": "components",
            "anchor_url": "/docs/chapter-1#components",
            "score": 0.85,
        },
    ]
