#!/usr/bin/env python3
"""
Direct Content Ingestion Script

Ingests markdown files directly into Qdrant without using the API.
This bypasses authentication requirements.

Usage:
    python scripts/direct_ingest.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import frontmatter
from dotenv import load_dotenv

from src.services.chunking import chunk_markdown
from src.services.embeddings import get_embeddings_batch
from src.services.qdrant import upsert_chunks, init_qdrant_collection


async def ingest_file(md_file: Path, docs_root: Path) -> int:
    """Ingest a single markdown file.

    Args:
        md_file: Path to markdown file
        docs_root: Root docs directory

    Returns:
        Number of chunks created
    """
    # Get relative path from docs root
    relative_path = md_file.relative_to(docs_root)
    parts = relative_path.parts

    # Extract chapter_id from first directory level
    chapter_id = parts[0] if len(parts) > 1 else "unknown"

    # Parse frontmatter for section_id
    try:
        post = frontmatter.load(md_file)
        section_id = post.metadata.get("id", md_file.stem)
        content = post.content
    except Exception:
        section_id = md_file.stem
        content = md_file.read_text(encoding="utf-8")

    # Generate anchor URL (remove /index for Docusaurus routing)
    path_str = str(relative_path.with_suffix("")).replace("\\", "/")
    # If file is index.md, use parent directory path
    if path_str.endswith("/index"):
        path_str = path_str[:-6]  # Remove '/index'
    anchor_url = f"/docs/{path_str}"
    source_file_str = str(relative_path)

    # Chunk the content
    chunks = chunk_markdown(
        content=content,
        chapter_id=chapter_id,
        section_id=section_id,
        anchor_url=anchor_url,
        source_file=source_file_str
    )

    if not chunks:
        return 0

    # Get embeddings
    embeddings = await get_embeddings_batch([chunk.content for chunk in chunks])

    # Upload to Qdrant
    vector_ids = await upsert_chunks(chunks, embeddings)

    return len(vector_ids)


async def main():
    """Main ingestion function."""
    # Load environment variables
    load_dotenv()

    # Find docs directory
    backend_dir = Path(__file__).parent.parent
    frontend_dir = backend_dir.parent / "frontend"
    docs_dir = frontend_dir / "docs"

    if not docs_dir.exists():
        print(f"❌ Error: Docs directory not found at {docs_dir}")
        return 1

    # Initialize Qdrant collection
    print("🔄 Initializing Qdrant collection...")
    await init_qdrant_collection()

    # Find all markdown files
    md_files = list(docs_dir.rglob("*.md"))

    if not md_files:
        print(f"❌ No markdown files found in {docs_dir}")
        return 1

    print(f"📚 Found {len(md_files)} markdown files")
    print(f"🔄 Starting ingestion...\n")

    total_chunks = 0
    errors = 0

    for i, md_file in enumerate(md_files, 1):
        try:
            relative = md_file.relative_to(docs_dir)
            print(f"[{i}/{len(md_files)}] Processing: {relative} ... ", end="", flush=True)

            chunks = await ingest_file(md_file, docs_dir)
            total_chunks += chunks
            print(f"✅ Created {chunks} chunks")

        except Exception as e:
            errors += 1
            print(f"❌ Error: {e}")

    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Files processed: {len(md_files)}")
    print(f"   Chunks created: {total_chunks}")
    print(f"   Errors: {errors}")
    print(f"{'='*60}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
