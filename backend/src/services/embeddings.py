"""OpenAI embeddings service using text-embedding-3-small."""

from typing import List

from openai import AsyncOpenAI

from src.config import get_settings

# Global client instance
_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """Get or create the OpenAI async client."""
    global _client

    if _client is None:
        settings = get_settings()

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        _client = AsyncOpenAI(api_key=settings.openai_api_key)

    return _client


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a single text.

    Args:
        text: The text to embed.

    Returns:
        List of floats representing the embedding vector (1536 dimensions).
    """
    client = _get_openai_client()
    settings = get_settings()

    # Clean and truncate text if needed
    text = text.replace("\n", " ").strip()
    if not text:
        raise ValueError("Cannot embed empty text")

    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )

    return response.data[0].embedding


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embedding vectors for multiple texts in a batch.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors (each 1536 dimensions).

    Raises:
        ValueError: If texts list is empty or contains empty strings.
    """
    if not texts:
        raise ValueError("Cannot embed empty list of texts")

    client = _get_openai_client()
    settings = get_settings()

    # Clean texts
    cleaned_texts = [t.replace("\n", " ").strip() for t in texts]

    # Validate no empty strings
    if any(not t for t in cleaned_texts):
        raise ValueError("Cannot embed empty text strings")

    # OpenAI allows up to 2048 inputs per batch, but we'll chunk at 100 for safety
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i : i + batch_size]

        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])

    return all_embeddings
