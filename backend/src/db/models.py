"""Pydantic models for database entities."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# =============================================================================
# User Profile Models (005-user-personalization)
# =============================================================================


class SoftwareBackground(BaseModel):
    """User's software development background."""

    level: str = Field(..., pattern="^(beginner|intermediate|advanced)$")
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)


class HardwareBackground(BaseModel):
    """User's hardware/robotics background."""

    level: str = Field(..., pattern="^(none|basic|intermediate|advanced)$")
    domains: List[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    """Extended user profile with technical background."""

    id: UUID
    email: EmailStr
    display_name: Optional[str] = None
    auth_provider: str = "email"
    software_background: Optional[SoftwareBackground] = None
    hardware_background: Optional[HardwareBackground] = None
    profile_completed: bool = False


class ProfileUpdateRequest(BaseModel):
    """Request to update user profile."""

    display_name: Optional[str] = Field(None, max_length=100)
    software_background: Optional[SoftwareBackground] = None
    hardware_background: Optional[HardwareBackground] = None


class ProfileUpdateResponse(BaseModel):
    """Response from profile update."""

    success: bool
    user: UserProfile


class PersonalizationLogEntry(BaseModel):
    """Log entry for personalization requests."""

    id: UUID
    user_id: UUID
    chapter_id: str
    request_timestamp: datetime
    response_time_ms: Optional[int] = None
    status: str
    error_message: Optional[str] = None


class TranslationLogEntry(BaseModel):
    """Log entry for translation requests."""

    id: UUID
    user_id: UUID
    chapter_id: str
    target_language: str
    request_timestamp: datetime
    response_time_ms: Optional[int] = None
    status: str
    error_message: Optional[str] = None


class Citation(BaseModel):
    """Citation reference to a textbook section."""

    chapter_id: str
    section_id: str
    anchor_url: str
    display_text: str


class User(BaseModel):
    """Registered reader of the textbook."""

    id: UUID
    email: EmailStr
    display_name: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None


class UserCreate(BaseModel):
    """Data for creating a new user."""

    email: EmailStr
    display_name: Optional[str] = None


class Session(BaseModel):
    """Chat conversation instance."""

    id: UUID
    user_id: UUID
    created_at: datetime
    last_activity: datetime
    is_active: bool = True


class SessionCreate(BaseModel):
    """Data for creating a new session."""

    user_id: UUID


class Message(BaseModel):
    """Single message in a conversation."""

    id: UUID
    session_id: UUID
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    citations: List[Citation] = Field(default_factory=list)
    created_at: datetime


class MessageCreate(BaseModel):
    """Data for creating a new message."""

    session_id: UUID
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    citations: List[Citation] = Field(default_factory=list)


class RetrievalLog(BaseModel):
    """Log entry for vector search operations."""

    id: UUID
    session_id: UUID
    message_id: Optional[UUID] = None
    query_text: str
    vector_ids: List[UUID]
    similarity_scores: List[float]
    created_at: datetime


class RetrievalLogCreate(BaseModel):
    """Data for creating a retrieval log entry."""

    session_id: UUID
    message_id: Optional[UUID] = None
    query_text: str
    vector_ids: List[UUID]
    similarity_scores: List[float]


class PerformanceMetric(BaseModel):
    """System performance data for monitoring."""

    id: UUID
    session_id: UUID
    message_id: Optional[UUID] = None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model_id: str
    created_at: datetime


class PerformanceMetricCreate(BaseModel):
    """Data for creating a performance metric entry."""

    session_id: UUID
    message_id: Optional[UUID] = None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model_id: str


class BookChunk(BaseModel):
    """Segment of textbook content stored as a vector."""

    id: UUID
    content: str
    chapter_id: str
    section_id: str
    anchor_url: str
    source_file: str
    token_count: int
