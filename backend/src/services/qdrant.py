"""Qdrant client module for vector storage and similarity search."""

from typing import List, Optional
from uuid import UUID, uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

from src.config import get_settings
from src.db.models import BookChunk

# Global client instance
_client: Optional[AsyncQdrantClient] = None


async def get_qdrant_client() -> AsyncQdrantClient:
    """Get or create the Qdrant async client.

    Returns:
        AsyncQdrantClient: The Qdrant client instance.

    Raises:
        ValueError: If Qdrant configuration is missing.
    """
    global _client

    if _client is None:
        settings = get_settings()

        if not settings.qdrant_url or not settings.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be configured")

        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    return _client


async def close_qdrant_client() -> None:
    """Close the Qdrant client connection."""
    global _client

    if _client is not None:
        await _client.close()
        _client = None


async def init_qdrant_collection() -> None:
    """Initialize the Qdrant collection for textbook chunks.

    Creates the collection if it doesn't exist.
    """
    client = await get_qdrant_client()
    settings = get_settings()

    # Check if collection exists
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]

    if settings.qdrant_collection not in collection_names:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small dimension
                distance=Distance.COSINE,
            ),
        )


class SearchResult:
    """Result from a vector similarity search."""

    def __init__(
        self,
        id: UUID,
        score: float,
        content: str,
        chapter_id: str,
        section_id: str,
        anchor_url: str,
        source_file: str,
        token_count: int,
    ):
        self.id = id
        self.score = score
        self.content = content
        self.chapter_id = chapter_id
        self.section_id = section_id
        self.anchor_url = anchor_url
        self.source_file = source_file
        self.token_count = token_count

    def to_book_chunk(self) -> BookChunk:
        """Convert to BookChunk model."""
        return BookChunk(
            id=self.id,
            content=self.content,
            chapter_id=self.chapter_id,
            section_id=self.section_id,
            anchor_url=self.anchor_url,
            source_file=self.source_file,
            token_count=self.token_count,
        )


async def search_similar_chunks(
    query_vector: List[float],
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    chapter_filter: Optional[str] = None,
) -> List[SearchResult]:
    """Search for similar chunks in the vector store.

    Args:
        query_vector: The embedding vector to search with.
        top_k: Maximum number of results to return.
        score_threshold: Minimum similarity score threshold.
        chapter_filter: Optional filter by chapter_id.

    Returns:
        List of SearchResult objects sorted by similarity score.
    """
    client = await get_qdrant_client()
    settings = get_settings()

    top_k = top_k or settings.top_k_results
    score_threshold = score_threshold or settings.similarity_threshold

    # Build filter if chapter specified
    search_filter = None
    if chapter_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="chapter_id",
                    match=MatchValue(value=chapter_filter),
                )
            ]
        )

    # Perform search using query_points (new API in qdrant-client >= 1.9.0)
    response = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=search_filter,
        with_payload=True,
    )

    # Convert to SearchResult objects
    # Note: query_points returns QueryResponse with .points attribute
    search_results = []
    for point in response.points:
        payload = point.payload or {}
        search_results.append(
            SearchResult(
                id=UUID(str(point.id)),
                score=point.score,
                content=payload.get("content", ""),
                chapter_id=payload.get("chapter_id", ""),
                section_id=payload.get("section_id", ""),
                anchor_url=payload.get("anchor_url", ""),
                source_file=payload.get("source_file", ""),
                token_count=payload.get("token_count", 0),
            )
        )

    return search_results


async def upsert_chunks(
    chunks: List[BookChunk],
    vectors: List[List[float]],
) -> List[UUID]:
    """Upsert chunks with their embedding vectors to Qdrant.

    Args:
        chunks: List of BookChunk objects to store.
        vectors: List of embedding vectors (must match chunks length).

    Returns:
        List of point IDs that were upserted.

    Raises:
        ValueError: If chunks and vectors lengths don't match.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunks ({len(chunks)}) and vectors ({len(vectors)}) must have same length"
        )

    client = await get_qdrant_client()
    settings = get_settings()

    # Create points with payloads
    points = []
    point_ids = []

    for chunk, vector in zip(chunks, vectors):
        point_id = chunk.id or uuid4()
        point_ids.append(point_id)

        points.append(
            PointStruct(
                id=str(point_id),
                vector=vector,
                payload={
                    "content": chunk.content,
                    "chapter_id": chunk.chapter_id,
                    "section_id": chunk.section_id,
                    "anchor_url": chunk.anchor_url,
                    "source_file": chunk.source_file,
                    "token_count": chunk.token_count,
                },
            )
        )

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch,
        )

    return point_ids


async def delete_chunks(chunk_ids: List[UUID]) -> None:
    """Delete chunks from the vector store.

    Args:
        chunk_ids: List of chunk IDs to delete.
    """
    client = await get_qdrant_client()
    settings = get_settings()

    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=[str(id) for id in chunk_ids],
    )


async def get_collection_info() -> dict:
    """Get information about the textbook chunks collection.

    Returns:
        Dictionary with collection statistics.
    """
    client = await get_qdrant_client()
    settings = get_settings()

    info = await client.get_collection(settings.qdrant_collection)

    return {
        "name": settings.qdrant_collection,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status.value,
    }
