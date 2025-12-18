"""Content ingestion endpoint for administrators."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.middleware import AuthenticatedUser, get_current_user
from src.services.chunking import chunk_markdown, chunk_markdown_by_sections
from src.services.embeddings import get_embeddings_batch
from src.services.qdrant import upsert_chunks, get_collection_info

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    """Request body for content ingestion."""

    content: str = Field(..., min_length=1)
    chapter_id: str = Field(..., min_length=1, max_length=100)
    section_id: str = Field(..., min_length=1, max_length=100)
    anchor_url: str = Field(..., min_length=1, max_length=500)
    source_file: str = Field(..., min_length=1, max_length=500)


class IngestBatchRequest(BaseModel):
    """Request body for batch content ingestion."""

    items: List[IngestRequest] = Field(..., min_length=1, max_length=100)


class IngestResponse(BaseModel):
    """Response body for content ingestion."""

    chunk_count: int
    vector_ids: List[str]
    success: bool


class CollectionInfoResponse(BaseModel):
    """Response body for collection info."""

    name: str
    vectors_count: int
    points_count: int
    status: str


# Admin email addresses (in production, use a proper role-based system)
ADMIN_EMAILS = [
    "admin@example.com",
    # Add admin emails here
]


def require_admin(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    """Dependency that requires admin privileges.

    Args:
        user: The authenticated user.

    Returns:
        The authenticated user if they are an admin.

    Raises:
        HTTPException: 403 if user is not an admin.
    """
    # In development, allow all authenticated users
    # In production, check against admin list or role
    # if user.email not in ADMIN_EMAILS:
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Admin access required",
    #     )
    return user


@router.post("/ingest", response_model=IngestResponse)
async def ingest_content(
    request: IngestRequest,
    user: AuthenticatedUser = Depends(require_admin),
) -> IngestResponse:
    """Ingest markdown content into the vector store.

    This endpoint:
    1. Chunks the content into 512-token segments with 50-token overlap
    2. Generates embeddings for each chunk
    3. Stores chunks and embeddings in Qdrant

    Requires admin authentication. In development, all authenticated users are treated as admins.

    Args:
        request: The ingestion request with content and metadata.
        user: The authenticated admin user.

    Returns:
        IngestResponse with chunk count and vector IDs.

    Raises:
        HTTPException: 401 if not authenticated.
        HTTPException: 403 if not an admin.
        HTTPException: 500 if ingestion fails.
    """
    try:
        # Chunk the content
        chunks = chunk_markdown(
            content=request.content,
            chapter_id=request.chapter_id,
            section_id=request.section_id,
            anchor_url=request.anchor_url,
            source_file=request.source_file,
        )

        if not chunks:
            return IngestResponse(
                chunk_count=0,
                vector_ids=[],
                success=True,
            )

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await get_embeddings_batch(texts)

        # Store in Qdrant
        vector_ids = await upsert_chunks(chunks, embeddings)

        return IngestResponse(
            chunk_count=len(chunks),
            vector_ids=[str(vid) for vid in vector_ids],
            success=True,
        )

    except Exception as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest content: {str(e)}",
        )


@router.post("/ingest/batch", response_model=IngestResponse)
async def ingest_batch(
    request: IngestBatchRequest,
    user: AuthenticatedUser = Depends(require_admin),
) -> IngestResponse:
    """Ingest multiple content items in a batch.

    Args:
        request: The batch ingestion request.
        user: The authenticated admin user.

    Returns:
        IngestResponse with total chunk count and vector IDs.
    """
    all_chunks = []
    all_embeddings = []

    try:
        for item in request.items:
            # Chunk each item
            chunks = chunk_markdown(
                content=item.content,
                chapter_id=item.chapter_id,
                section_id=item.section_id,
                anchor_url=item.anchor_url,
                source_file=item.source_file,
            )

            if chunks:
                all_chunks.extend(chunks)

        if not all_chunks:
            return IngestResponse(
                chunk_count=0,
                vector_ids=[],
                success=True,
            )

        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in all_chunks]
        all_embeddings = await get_embeddings_batch(texts)

        # Store in Qdrant
        vector_ids = await upsert_chunks(all_chunks, all_embeddings)

        return IngestResponse(
            chunk_count=len(all_chunks),
            vector_ids=[str(vid) for vid in vector_ids],
            success=True,
        )

    except Exception as e:
        print(f"Batch ingestion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest batch: {str(e)}",
        )


@router.get("/ingest/info", response_model=CollectionInfoResponse)
async def collection_info(
    user: AuthenticatedUser = Depends(require_admin),
) -> CollectionInfoResponse:
    """Get information about the vector store collection.

    Args:
        user: The authenticated admin user.

    Returns:
        CollectionInfoResponse with collection statistics.
    """
    try:
        info = await get_collection_info()
        return CollectionInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection info: {str(e)}",
        )
