"""Structured logging configuration for the RAG chatbot backend."""

import logging
import sys
from typing import Any, Dict

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: The name for the logger (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def log_request(
    logger: logging.Logger,
    endpoint: str,
    user_id: str,
    **extra: Any,
) -> None:
    """Log an API request.

    Args:
        logger: The logger instance.
        endpoint: The API endpoint being called.
        user_id: The user making the request.
        **extra: Additional context to log.
    """
    logger.info(
        f"Request: {endpoint}",
        extra={
            "endpoint": endpoint,
            "user_id": user_id,
            **extra,
        },
    )


def log_response(
    logger: logging.Logger,
    endpoint: str,
    status_code: int,
    latency_ms: int,
    **extra: Any,
) -> None:
    """Log an API response.

    Args:
        logger: The logger instance.
        endpoint: The API endpoint.
        status_code: The HTTP status code.
        latency_ms: Response latency in milliseconds.
        **extra: Additional context to log.
    """
    logger.info(
        f"Response: {endpoint} - {status_code} ({latency_ms}ms)",
        extra={
            "endpoint": endpoint,
            "status_code": status_code,
            "latency_ms": latency_ms,
            **extra,
        },
    )


def log_error(
    logger: logging.Logger,
    message: str,
    error: Exception,
    **extra: Any,
) -> None:
    """Log an error with exception details.

    Args:
        logger: The logger instance.
        message: Error message.
        error: The exception that occurred.
        **extra: Additional context to log.
    """
    logger.error(
        f"{message}: {str(error)}",
        exc_info=True,
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            **extra,
        },
    )


def log_rag_workflow(
    logger: logging.Logger,
    session_id: str,
    query: str,
    chunks_found: int,
    latency_ms: int,
    **extra: Any,
) -> None:
    """Log RAG workflow execution.

    Args:
        logger: The logger instance.
        session_id: The chat session ID.
        query: The user's query.
        chunks_found: Number of relevant chunks found.
        latency_ms: Workflow latency in milliseconds.
        **extra: Additional context to log.
    """
    logger.info(
        f"RAG workflow: {chunks_found} chunks, {latency_ms}ms",
        extra={
            "session_id": session_id,
            "query_length": len(query),
            "chunks_found": chunks_found,
            "latency_ms": latency_ms,
            **extra,
        },
    )
