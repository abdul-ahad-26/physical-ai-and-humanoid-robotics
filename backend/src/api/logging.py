import logging
import uuid
import json
import time
from typing import Dict, Any, Optional
from contextvars import ContextVar
from datetime import datetime
from enum import Enum

# Context variable to store correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

# Custom JSON formatter
class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": correlation_id_var.get() or getattr(record, 'correlation_id', None)
        }

        # Add any extra fields
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Structured logger with correlation ID support for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self, name: str = "rag_backend", level: str = "INFO"):
        """
        Initialize the structured logger.

        Args:
            name: Name of the logger
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        """
        Set the correlation ID for the current context.

        Args:
            correlation_id: Optional correlation ID. If not provided, one will be generated.

        Returns:
            The correlation ID that was set
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        correlation_id_var.set(correlation_id)
        return correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """
        Get the current correlation ID.

        Returns:
            The current correlation ID or None if not set
        """
        return correlation_id_var.get()

    def _log(self, level: LogLevel, message: str, extra_data: Optional[Dict[str, Any]] = None,
             exception: Optional[Exception] = None):
        """
        Internal method to log a message with structured data.

        Args:
            level: Log level
            message: Log message
            extra_data: Additional data to include in the log
            exception: Exception to include in the log
        """
        if extra_data is None:
            extra_data = {}

        # Add correlation ID to extra data
        extra_data['correlation_id'] = self.get_correlation_id()

        # Create a LogRecord with extra data
        record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.value),
            __file__,
            0,  # Line number - not available here
            message,
            (),
            exception,
            extra={'extra_data': extra_data}
        )

        self.logger.handle(record)

    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message, extra_data)

    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self._log(LogLevel.INFO, message, extra_data)

    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self._log(LogLevel.WARNING, message, extra_data)

    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
              exception: Optional[Exception] = None):
        """Log an error message."""
        self._log(LogLevel.ERROR, message, extra_data, exception)

    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
                 exception: Optional[Exception] = None):
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, message, extra_data, exception)

    def log_api_request(self, method: str, endpoint: str, response_time: float,
                       status_code: int, user_id: Optional[str] = None):
        """
        Log an API request with relevant details.

        Args:
            method: HTTP method
            endpoint: API endpoint
            response_time: Response time in seconds
            status_code: HTTP status code
            user_id: User ID if available
        """
        extra_data = {
            "event_type": "api_request",
            "http_method": method,
            "endpoint": endpoint,
            "response_time_ms": round(response_time * 1000, 2),
            "status_code": status_code
        }

        if user_id:
            extra_data["user_id"] = user_id

        level = LogLevel.INFO if status_code < 400 else LogLevel.WARNING if status_code < 500 else LogLevel.ERROR
        self._log(level, f"API request completed: {method} {endpoint}", extra_data)

    def log_agent_execution(self, agent_name: str, operation: str, execution_time: float,
                           success: bool = True, error_message: Optional[str] = None):
        """
        Log an agent execution with relevant details.

        Args:
            agent_name: Name of the agent
            operation: Operation performed
            execution_time: Execution time in seconds
            success: Whether the operation was successful
            error_message: Error message if operation failed
        """
        extra_data = {
            "event_type": "agent_execution",
            "agent_name": agent_name,
            "operation": operation,
            "execution_time_ms": round(execution_time * 1000, 2),
            "success": success
        }

        if error_message:
            extra_data["error_message"] = error_message

        level = LogLevel.INFO if success else LogLevel.ERROR
        self._log(level, f"Agent execution completed: {agent_name}.{operation}", extra_data)

    def log_retrieval(self, query: str, results_count: int, retrieval_time: float,
                     success: bool = True, error_message: Optional[str] = None):
        """
        Log a retrieval operation with relevant details.

        Args:
            query: Query string (truncated)
            results_count: Number of results returned
            retrieval_time: Retrieval time in seconds
            success: Whether the operation was successful
            error_message: Error message if operation failed
        """
        extra_data = {
            "event_type": "retrieval",
            "query_length": len(query),
            "results_count": results_count,
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "success": success
        }

        if error_message:
            extra_data["error_message"] = error_message

        level = LogLevel.INFO if success else LogLevel.ERROR
        self._log(level, f"Retrieval completed: {results_count} results in {retrieval_time:.3f}s", extra_data)

    def log_database_operation(self, operation: str, table: str, execution_time: float,
                              success: bool = True, error_message: Optional[str] = None):
        """
        Log a database operation with relevant details.

        Args:
            operation: Type of database operation
            table: Table name
            execution_time: Execution time in seconds
            success: Whether the operation was successful
            error_message: Error message if operation failed
        """
        extra_data = {
            "event_type": "database_operation",
            "operation": operation,
            "table": table,
            "execution_time_ms": round(execution_time * 1000, 2),
            "success": success
        }

        if error_message:
            extra_data["error_message"] = error_message

        level = LogLevel.INFO if success else LogLevel.ERROR
        self._log(level, f"Database operation completed: {operation} on {table}", extra_data)


# Global logger instance
logger = StructuredLogger()


def get_logger() -> StructuredLogger:
    """
    Get the global structured logger instance.

    Returns:
        StructuredLogger instance
    """
    return logger


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current context.

    Args:
        correlation_id: Optional correlation ID. If not provided, one will be generated.

    Returns:
        The correlation ID that was set
    """
    return logger.set_correlation_id(correlation_id)


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID.

    Returns:
        The current correlation ID or None if not set
    """
    return logger.get_correlation_id()


# Context manager for correlation ID
class CorrelationIdContext:
    """
    Context manager for setting a correlation ID for a block of code.
    """

    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize the context manager.

        Args:
            correlation_id: Optional correlation ID. If not provided, one will be generated.
        """
        self.correlation_id = correlation_id
        self.token = None

    def __enter__(self):
        self.token = correlation_id_var.set(self.correlation_id or str(uuid.uuid4()))
        return correlation_id_var.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        correlation_id_var.reset(self.token)


# Example usage
if __name__ == "__main__":
    # Example of using the structured logger
    log = get_logger()

    # Set a correlation ID
    correlation_id = set_correlation_id()
    print(f"Set correlation ID: {correlation_id}")

    # Log different types of messages
    log.info("Application started", {"version": "1.0.0", "environment": "development"})
    log.debug("Debug information", {"debug_info": "some debug data"})

    # Log an API request
    log.log_api_request("POST", "/api/v1/query", 0.245, 200, "user123")

    # Log an agent execution
    log.log_agent_execution("RAGAgent", "process_query", 0.187, True)

    # Log a retrieval operation
    log.log_retrieval("What is machine learning?", 5, 0.156, True)

    # Use the context manager
    with CorrelationIdContext() as cid:
        log.info(f"Processing in context {cid}")
        log.warning("This is a warning in the context", {"context_data": "value"})

    print("Structured logging example completed")