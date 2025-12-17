"""POST /api/chat endpoint for the RAG chatbot."""

import time
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.middleware import AuthenticatedUser, check_rate_limit
from src.db.models import Citation, SessionCreate
from src.db.queries import create_session, get_session_by_id, get_or_create_user
from src.agents.orchestrator import run_rag_workflow
from src.logging_config import get_logger, log_request, log_response, log_error

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    selected_text: Optional[str] = Field(None, max_length=2000)


class CitationResponse(BaseModel):
    """Citation in chat response."""

    chapter_id: str
    section_id: str
    anchor_url: str
    display_text: str


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    answer: str
    citations: List[CitationResponse]
    session_id: str
    found_relevant_content: bool
    latency_ms: int


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: AuthenticatedUser = Depends(check_rate_limit),
) -> ChatResponse:
    """Process a chat message and return an AI-generated response.

    This endpoint:
    1. Validates the user session
    2. Creates or retrieves a chat session
    3. Runs the RAG workflow to generate a response
    4. Returns the answer with citations

    Args:
        request: The chat request with message and optional context.
        user: The authenticated user (from rate-limited dependency).

    Returns:
        ChatResponse with answer, citations, and metadata.

    Raises:
        HTTPException: 400 if session doesn't belong to user.
        HTTPException: 429 if rate limit exceeded.
    """
    # Get or create user in our database
    db_user = await get_or_create_user(user.email, user.display_name)

    # Handle session
    if request.session_id:
        # Validate existing session
        session = await get_session_by_id(UUID(request.session_id))
        if not session:
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID",
            )
        if session.user_id != db_user.id:
            raise HTTPException(
                status_code=403,
                detail="Session does not belong to this user",
            )
        session_id = session.id
    else:
        # Create new session
        session = await create_session(SessionCreate(user_id=db_user.id))
        session_id = session.id

    # Log request
    log_request(
        logger,
        endpoint="/api/chat",
        user_id=str(db_user.id),
        session_id=str(session_id),
        has_selected_text=bool(request.selected_text),
    )

    start_time = time.time()

    # Run RAG workflow
    try:
        result = await run_rag_workflow(
            query=request.message,
            session_id=session_id,
            user_id=db_user.id,
            selected_text=request.selected_text,
        )

        response = ChatResponse(
            answer=result.answer,
            citations=[
                CitationResponse(
                    chapter_id=c.chapter_id,
                    section_id=c.section_id,
                    anchor_url=c.anchor_url,
                    display_text=c.display_text,
                )
                for c in result.citations
            ],
            session_id=str(session_id),
            found_relevant_content=result.found_content,
            latency_ms=result.latency_ms,
        )

        # Log successful response
        log_response(
            logger,
            endpoint="/api/chat",
            status_code=200,
            latency_ms=result.latency_ms,
            found_content=result.found_content,
            citation_count=len(result.citations),
        )

        return response

    except ValueError as e:
        # Input validation errors
        log_error(logger, "Validation error in chat", e, session_id=str(session_id))
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except ConnectionError as e:
        # Database or external service connection errors
        log_error(logger, "Connection error in chat", e, session_id=str(session_id))
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again later.",
        )

    except Exception as e:
        # Log error and return graceful failure
        log_error(logger, "Unexpected error in chat", e, session_id=str(session_id))
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again.",
        )
