"""Token-based chunking service using tiktoken."""

import re
from typing import List, Optional
from uuid import uuid4

import tiktoken

from src.config import get_settings
from src.db.models import BookChunk


def _get_encoder() -> tiktoken.Encoding:
    """Get the tiktoken encoder for gpt-4o-mini."""
    return tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text.

    Args:
        text: The text to count tokens for.

    Returns:
        Number of tokens.
    """
    encoder = _get_encoder()
    return len(encoder.encode(text))


def chunk_markdown(
    content: str,
    chapter_id: str,
    section_id: str,
    anchor_url: str,
    source_file: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[BookChunk]:
    """Chunk markdown content into token-based segments.

    Args:
        content: The markdown content to chunk.
        chapter_id: Chapter identifier (e.g., "chapter-1").
        section_id: Section identifier (e.g., "intro").
        anchor_url: URL path with anchor (e.g., "/docs/chapter-1#intro").
        source_file: Original markdown file path.
        chunk_size: Maximum tokens per chunk (default from settings).
        chunk_overlap: Overlap tokens between chunks (default from settings).

    Returns:
        List of BookChunk objects.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    encoder = _get_encoder()

    # Clean content
    content = content.strip()
    if not content:
        return []

    # Encode entire content to tokens
    tokens = encoder.encode(content)

    if len(tokens) <= chunk_size:
        # Content fits in single chunk
        return [
            BookChunk(
                id=uuid4(),
                content=content,
                chapter_id=chapter_id,
                section_id=section_id,
                anchor_url=anchor_url,
                source_file=source_file,
                token_count=len(tokens),
            )
        ]

    # Split into overlapping chunks
    chunks = []
    start = 0

    while start < len(tokens):
        # Calculate end position
        end = min(start + chunk_size, len(tokens))

        # Get chunk tokens and decode
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)

        chunks.append(
            BookChunk(
                id=uuid4(),
                content=chunk_text,
                chapter_id=chapter_id,
                section_id=section_id,
                anchor_url=anchor_url,
                source_file=source_file,
                token_count=len(chunk_tokens),
            )
        )

        # Move start position, accounting for overlap
        start += chunk_size - chunk_overlap

        # Break if we've reached the end
        if start >= len(tokens):
            break

    return chunks


def chunk_markdown_by_sections(
    content: str,
    source_file: str,
    base_url: str = "/docs",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[BookChunk]:
    """Chunk markdown content, respecting section boundaries where possible.

    This function attempts to split content at section headers (## and ###)
    before falling back to token-based chunking.

    Args:
        content: The markdown content to chunk.
        source_file: Original markdown file path.
        base_url: Base URL for anchor links.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Overlap tokens between chunks.

    Returns:
        List of BookChunk objects.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size

    # Extract chapter ID from source file
    chapter_match = re.search(r"(chapter-\d+|ch\d+)", source_file, re.IGNORECASE)
    chapter_id = chapter_match.group(1).lower() if chapter_match else "unknown"

    # Split content by section headers
    section_pattern = r"(^#{1,3}\s+.+$)"
    parts = re.split(section_pattern, content, flags=re.MULTILINE)

    all_chunks = []
    current_section_id = "intro"
    current_anchor = f"{base_url}/{chapter_id}#intro"

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Check if this is a header
        header_match = re.match(r"^(#{1,3})\s+(.+)$", part)
        if header_match:
            # Update section info
            header_text = header_match.group(2)
            current_section_id = _slugify(header_text)
            current_anchor = f"{base_url}/{chapter_id}#{current_section_id}"
            continue

        # Chunk this section's content
        section_chunks = chunk_markdown(
            content=part,
            chapter_id=chapter_id,
            section_id=current_section_id,
            anchor_url=current_anchor,
            source_file=source_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(section_chunks)

    return all_chunks


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug.

    Args:
        text: The text to slugify.

    Returns:
        URL-safe slug.
    """
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces and special chars with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    return slug or "section"
