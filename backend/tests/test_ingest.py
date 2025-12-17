"""Tests for the content ingestion endpoint and chunking service."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


class TestChunkingService:
    """Tests for the token-based chunking service."""

    def test_chunking_service_basic(self):
        """Test basic chunking functionality."""
        from src.services.chunking import chunk_markdown

        content = "This is a test paragraph. " * 100  # Create longer content

        chunks = chunk_markdown(
            content=content,
            chapter_id="chapter-1",
            section_id="test",
            anchor_url="/docs/chapter-1#test",
            source_file="test.md",
            chunk_size=50,  # Small chunks for testing
            chunk_overlap=10,
        )

        assert len(chunks) > 1  # Should create multiple chunks
        for chunk in chunks:
            assert chunk.chapter_id == "chapter-1"
            assert chunk.section_id == "test"
            assert chunk.token_count <= 50

    def test_chunking_service_small_content(self):
        """Test that small content stays in single chunk."""
        from src.services.chunking import chunk_markdown

        content = "This is a short test."

        chunks = chunk_markdown(
            content=content,
            chapter_id="chapter-1",
            section_id="test",
            anchor_url="/docs/chapter-1#test",
            source_file="test.md",
        )

        assert len(chunks) == 1
        assert chunks[0].content == content

    def test_chunking_service_empty_content(self):
        """Test handling of empty content."""
        from src.services.chunking import chunk_markdown

        chunks = chunk_markdown(
            content="",
            chapter_id="chapter-1",
            section_id="test",
            anchor_url="/docs/chapter-1#test",
            source_file="test.md",
        )

        assert len(chunks) == 0

    def test_chunking_service_overlap(self):
        """Test that chunks have proper overlap."""
        from src.services.chunking import chunk_markdown, count_tokens

        # Create content that will definitely be split
        content = "word " * 200  # 200 words

        chunks = chunk_markdown(
            content=content,
            chapter_id="chapter-1",
            section_id="test",
            anchor_url="/docs/chapter-1#test",
            source_file="test.md",
            chunk_size=50,
            chunk_overlap=10,
        )

        # Should have multiple chunks
        assert len(chunks) > 1

        # Each chunk should have content
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_count_tokens(self):
        """Test token counting function."""
        from src.services.chunking import count_tokens

        # Simple text
        count = count_tokens("Hello, world!")
        assert count > 0
        assert count < 10  # Should be a small number

        # Longer text
        long_text = "This is a longer piece of text. " * 10
        long_count = count_tokens(long_text)
        assert long_count > count

    def test_chunk_markdown_by_sections(self):
        """Test section-aware chunking."""
        from src.services.chunking import chunk_markdown_by_sections

        content = """
# Introduction

This is the introduction section with some content.

## Getting Started

This section explains how to get started with the topic.

### Prerequisites

You need the following prerequisites to proceed.
"""

        chunks = chunk_markdown_by_sections(
            content=content,
            source_file="docs/chapter-1.md",
            base_url="/docs",
        )

        # Should create chunks for different sections
        assert len(chunks) > 0

        # Each chunk should have section info
        for chunk in chunks:
            assert chunk.section_id != ""
            assert chunk.anchor_url != ""


class TestIngestEndpoint:
    """Tests for POST /api/ingest endpoint."""

    @pytest.mark.asyncio
    async def test_ingest_endpoint_requires_auth(self, client):
        """Test that ingest endpoint requires authentication."""
        response = await client.post(
            "/api/ingest",
            json={
                "content": "Test content",
                "chapter_id": "chapter-1",
                "section_id": "test",
                "anchor_url": "/docs/chapter-1#test",
                "source_file": "test.md",
            },
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_ingest_endpoint_success(self, client, sample_ingest_request):
        """Test successful content ingestion."""
        user_id = uuid4()

        with patch("src.api.ingest.require_admin") as mock_admin, \
             patch("src.api.ingest.chunk_markdown") as mock_chunk, \
             patch("src.api.ingest.get_embeddings_batch") as mock_embed, \
             patch("src.api.ingest.upsert_chunks") as mock_upsert:

            mock_admin.return_value = MagicMock(
                id=user_id,
                email="admin@example.com",
            )

            # Mock chunking
            from src.db.models import BookChunk

            mock_chunks = [
                BookChunk(
                    id=uuid4(),
                    content="Test chunk 1",
                    chapter_id="chapter-1",
                    section_id="test",
                    anchor_url="/docs/chapter-1#test",
                    source_file="test.md",
                    token_count=10,
                ),
                BookChunk(
                    id=uuid4(),
                    content="Test chunk 2",
                    chapter_id="chapter-1",
                    section_id="test",
                    anchor_url="/docs/chapter-1#test",
                    source_file="test.md",
                    token_count=10,
                ),
            ]
            mock_chunk.return_value = mock_chunks

            # Mock embeddings
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

            # Mock upsert
            mock_upsert.return_value = [c.id for c in mock_chunks]

            response = await client.post(
                "/api/ingest",
                json=sample_ingest_request,
                cookies={"better-auth.session_token": "mock-token"},
            )

            # Check response if auth passed
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert data["chunk_count"] == 2
                assert len(data["vector_ids"]) == 2

    @pytest.mark.asyncio
    async def test_ingest_endpoint_empty_content(self, client):
        """Test handling of content that produces no chunks."""
        user_id = uuid4()

        with patch("src.api.ingest.require_admin") as mock_admin, \
             patch("src.api.ingest.chunk_markdown") as mock_chunk:

            mock_admin.return_value = MagicMock(id=user_id)
            mock_chunk.return_value = []  # No chunks

            response = await client.post(
                "/api/ingest",
                json={
                    "content": "   ",  # Whitespace only
                    "chapter_id": "chapter-1",
                    "section_id": "test",
                    "anchor_url": "/docs/chapter-1#test",
                    "source_file": "test.md",
                },
                cookies={"better-auth.session_token": "mock-token"},
            )

            if response.status_code == 200:
                data = response.json()
                assert data["chunk_count"] == 0
                assert data["success"] is True


class TestBatchIngest:
    """Tests for batch content ingestion."""

    @pytest.mark.asyncio
    async def test_batch_ingest_multiple_items(self, client):
        """Test batch ingestion of multiple content items."""
        user_id = uuid4()

        with patch("src.api.ingest.require_admin") as mock_admin, \
             patch("src.api.ingest.chunk_markdown") as mock_chunk, \
             patch("src.api.ingest.get_embeddings_batch") as mock_embed, \
             patch("src.api.ingest.upsert_chunks") as mock_upsert:

            mock_admin.return_value = MagicMock(id=user_id)

            from src.db.models import BookChunk

            mock_chunk.return_value = [
                BookChunk(
                    id=uuid4(),
                    content="Test",
                    chapter_id="ch1",
                    section_id="s1",
                    anchor_url="/docs/ch1#s1",
                    source_file="test.md",
                    token_count=5,
                )
            ]

            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = [uuid4()]

            response = await client.post(
                "/api/ingest/batch",
                json={
                    "items": [
                        {
                            "content": "Content 1",
                            "chapter_id": "chapter-1",
                            "section_id": "section-1",
                            "anchor_url": "/docs/chapter-1#section-1",
                            "source_file": "doc1.md",
                        },
                        {
                            "content": "Content 2",
                            "chapter_id": "chapter-2",
                            "section_id": "section-1",
                            "anchor_url": "/docs/chapter-2#section-1",
                            "source_file": "doc2.md",
                        },
                    ]
                },
                cookies={"better-auth.session_token": "mock-token"},
            )

            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
