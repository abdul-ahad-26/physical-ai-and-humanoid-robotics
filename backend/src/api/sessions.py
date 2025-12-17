"""Session management endpoints."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.middleware import AuthenticatedUser, get_current_user
from src.db.models import Citation
from src.db.queries import (
    get_or_create_user,
    get_user_sessions,
    get_session_by_id,
    get_session_messages,
)

router = APIRouter(tags=["sessions"])


class SessionSummary(BaseModel):
    """Summary of a chat session."""

    id: str
    created_at: str
    last_activity: str
    is_active: bool


class SessionListResponse(BaseModel):
    """Response for listing user sessions."""

    sessions: List[SessionSummary]
    total: int


class CitationResponse(BaseModel):
    """Citation in message response."""

    chapter_id: str
    section_id: str
    anchor_url: str
    display_text: str


class MessageResponse(BaseModel):
    """A message in a session."""

    id: str
    role: str
    content: str
    citations: List[CitationResponse]
    created_at: str


class MessagesResponse(BaseModel):
    """Response for listing session messages."""

    messages: List[MessageResponse]
    session_id: str
    total: int


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    include_inactive: bool = False,
    user: AuthenticatedUser = Depends(get_current_user),
) -> SessionListResponse:
    """List all chat sessions for the authenticated user.

    Args:
        limit: Maximum number of sessions to return (default 50).
        include_inactive: Include inactive sessions (default False).
        user: The authenticated user.

    Returns:
        SessionListResponse with list of sessions.
    """
    # Get user from database
    db_user = await get_or_create_user(user.email, user.display_name)

    # Get sessions
    sessions = await get_user_sessions(
        user_id=db_user.id,
        limit=limit,
        include_inactive=include_inactive,
    )

    return SessionListResponse(
        sessions=[
            SessionSummary(
                id=str(s.id),
                created_at=s.created_at.isoformat(),
                last_activity=s.last_activity.isoformat(),
                is_active=s.is_active,
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
async def get_messages(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    user: AuthenticatedUser = Depends(get_current_user),
) -> MessagesResponse:
    """Get all messages in a session.

    Args:
        session_id: The session UUID.
        limit: Maximum number of messages to return (default 100).
        offset: Number of messages to skip (default 0).
        user: The authenticated user.

    Returns:
        MessagesResponse with list of messages.

    Raises:
        HTTPException: 404 if session not found.
        HTTPException: 403 if session doesn't belong to user.
    """
    # Get user from database
    db_user = await get_or_create_user(user.email, user.display_name)

    # Validate session exists and belongs to user
    try:
        session = await get_session_by_id(UUID(session_id))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID format",
        )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    if session.user_id != db_user.id:
        raise HTTPException(
            status_code=403,
            detail="Session does not belong to this user",
        )

    # Get messages
    messages = await get_session_messages(
        session_id=UUID(session_id),
        limit=limit,
        offset=offset,
    )

    return MessagesResponse(
        messages=[
            MessageResponse(
                id=str(m.id),
                role=m.role,
                content=m.content,
                citations=[
                    CitationResponse(
                        chapter_id=c.chapter_id,
                        section_id=c.section_id,
                        anchor_url=c.anchor_url,
                        display_text=c.display_text,
                    )
                    for c in m.citations
                ],
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ],
        session_id=session_id,
        total=len(messages),
    )
