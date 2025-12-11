"""
Database models for the RAG + Agentic AI-Textbook Chatbot
Based on the data-model.md specification
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from uuid import UUID


@dataclass
class UserSession:
    """
    Represents a user interaction session with the AI chatbot
    """
    id: Optional[UUID] = None
    user_id: Optional[str] = None
    query: str = ""
    response: Optional[str] = None
    retrieved_context: Optional[List[Dict]] = None
    timestamp: Optional[datetime] = None
    session_metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TextbookContent:
    """
    Represents indexed textbook material with embeddings
    """
    id: Optional[UUID] = None
    content: str = ""
    embeddings: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "active"  # active, deleted, archived

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryContext:
    """
    Represents the context used for a specific query, including original question and retrieved content
    """
    id: Optional[UUID] = None
    original_question: str = ""
    highlight_override: Optional[str] = None
    retrieved_chunks: Optional[List[Dict]] = None
    processed_context: Optional[str] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[UUID] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AgentTool:
    """
    Represents a function/tool available to the AI agent
    """
    id: Optional[UUID] = None
    name: str = ""
    description: str = ""
    parameters: Optional[Dict] = None
    implementation_path: str = ""
    active: bool = True

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class AgentExecutionLog:
    """
    Log of agent executions for debugging and monitoring
    """
    id: Optional[UUID] = None
    agent_id: str = ""
    tool_calls: Optional[List[Dict]] = None
    input_params: Optional[Dict] = None
    output_result: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[UUID] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.tool_calls is None:
            self.tool_calls = []
        if self.input_params is None:
            self.input_params = {}