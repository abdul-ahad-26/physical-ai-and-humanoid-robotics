"""Database models and queries for Neon Postgres."""

from .connection import get_db_pool, close_db_pool
from .models import User, Session, Message, RetrievalLog, PerformanceMetric, Citation
from .queries import (
    create_user,
    get_user_by_id,
    get_user_by_email,
    create_session,
    get_session_by_id,
    get_user_sessions,
    update_session_activity,
    create_message,
    get_session_messages,
    create_retrieval_log,
    create_performance_metric,
)

__all__ = [
    "get_db_pool",
    "close_db_pool",
    "User",
    "Session",
    "Message",
    "RetrievalLog",
    "PerformanceMetric",
    "Citation",
    "create_user",
    "get_user_by_id",
    "get_user_by_email",
    "create_session",
    "get_session_by_id",
    "get_user_sessions",
    "update_session_activity",
    "create_message",
    "get_session_messages",
    "create_retrieval_log",
    "create_performance_metric",
]
