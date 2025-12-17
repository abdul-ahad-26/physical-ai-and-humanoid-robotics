"""Session & Logging Agent - Handles message persistence and metrics logging."""

from typing import List, Optional
from uuid import UUID

from agents import Agent, function_tool
from pydantic import BaseModel

from src.db.models import (
    Citation,
    MessageCreate,
    RetrievalLogCreate,
    PerformanceMetricCreate,
)
from src.db.queries import (
    create_message,
    create_retrieval_log,
    create_performance_metric,
    update_session_activity,
)


class SessionContext(BaseModel):
    """Context for session operations."""

    session_id: UUID
    user_id: UUID


@function_tool
async def save_user_message(
    session_id: str,
    content: str,
) -> str:
    """Save a user message to the database.

    Args:
        session_id: The session UUID.
        content: The message content.

    Returns:
        JSON string with the created message ID.
    """
    import json

    message = await create_message(
        MessageCreate(
            session_id=UUID(session_id),
            role="user",
            content=content,
            citations=[],
        )
    )

    await update_session_activity(UUID(session_id))

    return json.dumps({
        "message_id": str(message.id),
        "role": "user",
        "saved": True,
    })


@function_tool
async def save_assistant_message(
    session_id: str,
    content: str,
    citations_json: str = "[]",
) -> str:
    """Save an assistant message with citations to the database.

    Args:
        session_id: The session UUID.
        content: The message content.
        citations_json: JSON string of citation objects.

    Returns:
        JSON string with the created message ID.
    """
    import json

    # Parse citations
    citations_data = json.loads(citations_json)
    citations = [Citation(**c) for c in citations_data]

    message = await create_message(
        MessageCreate(
            session_id=UUID(session_id),
            role="assistant",
            content=content,
            citations=citations,
        )
    )

    await update_session_activity(UUID(session_id))

    return json.dumps({
        "message_id": str(message.id),
        "role": "assistant",
        "saved": True,
        "citation_count": len(citations),
    })


@function_tool
async def log_retrieval(
    session_id: str,
    message_id: str,
    query_text: str,
    vector_ids_json: str,
    scores_json: str,
) -> str:
    """Log a vector retrieval operation.

    Args:
        session_id: The session UUID.
        message_id: The associated message UUID.
        query_text: The search query.
        vector_ids_json: JSON array of vector UUIDs.
        scores_json: JSON array of similarity scores.

    Returns:
        JSON string confirming the log was created.
    """
    import json

    vector_ids = [UUID(vid) for vid in json.loads(vector_ids_json)]
    scores = json.loads(scores_json)

    log = await create_retrieval_log(
        RetrievalLogCreate(
            session_id=UUID(session_id),
            message_id=UUID(message_id) if message_id else None,
            query_text=query_text,
            vector_ids=vector_ids,
            similarity_scores=scores,
        )
    )

    return json.dumps({
        "log_id": str(log.id),
        "logged": True,
    })


@function_tool
async def log_performance(
    session_id: str,
    message_id: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    model_id: str,
) -> str:
    """Log performance metrics for a request.

    Args:
        session_id: The session UUID.
        message_id: The associated message UUID.
        latency_ms: Response latency in milliseconds.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model_id: The model identifier used.

    Returns:
        JSON string confirming the metric was logged.
    """
    import json

    metric = await create_performance_metric(
        PerformanceMetricCreate(
            session_id=UUID(session_id),
            message_id=UUID(message_id) if message_id else None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
        )
    )

    return json.dumps({
        "metric_id": str(metric.id),
        "logged": True,
    })


# Create the Session & Logging Agent
session_agent = Agent(
    name="Session Agent",
    instructions="""You are a session management specialist for a textbook assistant.

Your responsibilities:
1. Save user messages to the database
2. Save assistant responses with their citations
3. Log retrieval operations for analytics
4. Track performance metrics

When handling a conversation turn:
1. First save the user's message using save_user_message
2. After the response is generated, save it with save_assistant_message
3. Log any retrieval operations with log_retrieval
4. Record performance metrics with log_performance

Always ensure data is persisted correctly for conversation history and analytics.""",
    tools=[
        save_user_message,
        save_assistant_message,
        log_retrieval,
        log_performance,
    ],
    model="gpt-4o-mini",
)


# Utility functions for direct use without agent
async def persist_user_message(
    session_id: UUID,
    content: str,
) -> UUID:
    """Persist a user message directly."""
    message = await create_message(
        MessageCreate(
            session_id=session_id,
            role="user",
            content=content,
            citations=[],
        )
    )
    await update_session_activity(session_id)
    return message.id


async def persist_assistant_message(
    session_id: UUID,
    content: str,
    citations: List[Citation],
) -> UUID:
    """Persist an assistant message directly."""
    message = await create_message(
        MessageCreate(
            session_id=session_id,
            role="assistant",
            content=content,
            citations=citations,
        )
    )
    await update_session_activity(session_id)
    return message.id


async def log_retrieval_direct(
    session_id: UUID,
    message_id: Optional[UUID],
    query_text: str,
    vector_ids: List[UUID],
    similarity_scores: List[float],
) -> UUID:
    """Log retrieval operation directly."""
    log = await create_retrieval_log(
        RetrievalLogCreate(
            session_id=session_id,
            message_id=message_id,
            query_text=query_text,
            vector_ids=vector_ids,
            similarity_scores=similarity_scores,
        )
    )
    return log.id


async def log_performance_direct(
    session_id: UUID,
    message_id: Optional[UUID],
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    model_id: str,
) -> UUID:
    """Log performance metrics directly."""
    metric = await create_performance_metric(
        PerformanceMetricCreate(
            session_id=session_id,
            message_id=message_id,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
        )
    )
    return metric.id
