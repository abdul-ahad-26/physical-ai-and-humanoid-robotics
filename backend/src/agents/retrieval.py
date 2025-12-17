"""Retrieval Agent - Searches textbook content in Qdrant."""

from typing import List, Optional

from agents import Agent, function_tool
from pydantic import BaseModel

from src.config import get_settings
from src.services.embeddings import get_embedding
from src.services.qdrant import search_similar_chunks, SearchResult


class RetrievalContext(BaseModel):
    """Context passed to retrieval operations."""

    session_id: str
    selected_text: Optional[str] = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk from the vector store."""

    content: str
    chapter_id: str
    section_id: str
    anchor_url: str
    score: float


class RetrievalResult(BaseModel):
    """Result from the retrieval agent."""

    chunks: List[RetrievedChunk]
    query: str
    found_relevant_content: bool


@function_tool
async def search_book_content(
    query: str,
    selected_text: str = "",
) -> str:
    """Search textbook content in Qdrant vector store.

    Args:
        query: The user's question or search query.
        selected_text: Optional text selected by the user for context.

    Returns:
        JSON string containing retrieved chunks with content and citations.
    """
    import json

    settings = get_settings()

    # Combine query with selected text for better retrieval
    search_query = query
    if selected_text:
        search_query = f"{selected_text}\n\nQuestion: {query}"

    # Get embedding for the query
    query_vector = await get_embedding(search_query)

    # Search for similar chunks
    results = await search_similar_chunks(
        query_vector=query_vector,
        top_k=settings.top_k_results,
        score_threshold=settings.similarity_threshold,
    )

    if not results:
        return json.dumps({
            "found_relevant_content": False,
            "chunks": [],
            "message": "No relevant content found in the textbook."
        })

    # Convert to serializable format
    chunks = [
        {
            "content": r.content,
            "chapter_id": r.chapter_id,
            "section_id": r.section_id,
            "anchor_url": r.anchor_url,
            "score": round(r.score, 4),
        }
        for r in results
    ]

    return json.dumps({
        "found_relevant_content": True,
        "chunks": chunks,
        "total_chunks": len(chunks),
    })


# Create the Retrieval Agent
retrieval_agent = Agent(
    name="Retrieval Agent",
    instructions="""You are a retrieval specialist for a textbook assistant.

Your job is to search the textbook content to find relevant passages that can answer the user's question.

When given a query:
1. Use the search_book_content tool to find relevant textbook passages
2. If selected_text context is provided, include it in your search to find more relevant content
3. Return the search results so they can be used to generate an answer

Always use the search tool - do not try to answer questions from your own knowledge.
The textbook content is the ONLY source of truth for answers.""",
    tools=[search_book_content],
    model="gpt-4o-mini",
)
