"""Retry utilities with exponential backoff for API calls."""

import asyncio
import random
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec

from src.logging_config import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts.
            base_delay: Initial delay in seconds.
            max_delay: Maximum delay in seconds.
            exponential_base: Base for exponential backoff.
            jitter: Whether to add random jitter to delays.
            retryable_exceptions: Tuple of exception types to retry on.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


# Default configuration for OpenAI API calls
OPENAI_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        # Add OpenAI-specific exceptions here when available
    ),
)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for a retry attempt with exponential backoff.

    Args:
        attempt: The current attempt number (0-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter (0-25% of the delay)
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


def with_retry(config: RetryConfig = OPENAI_RETRY_CONFIG):
    """Decorator that adds retry logic with exponential backoff.

    Args:
        config: Retry configuration.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{config.max_retries} "
                            f"for {func.__name__} after {delay:.2f}s delay. "
                            f"Error: {str(e)}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries} retries exhausted "
                            f"for {func.__name__}. Final error: {str(e)}"
                        )

            raise last_exception  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import time

            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{config.max_retries} "
                            f"for {func.__name__} after {delay:.2f}s delay. "
                            f"Error: {str(e)}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries} retries exhausted "
                            f"for {func.__name__}. Final error: {str(e)}"
                        )

            raise last_exception  # type: ignore

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def retry_async(
    func: Callable[P, T],
    *args: P.args,
    config: RetryConfig = OPENAI_RETRY_CONFIG,
    **kwargs: P.kwargs,
) -> T:
    """Execute an async function with retry logic.

    Args:
        func: The async function to execute.
        *args: Positional arguments for the function.
        config: Retry configuration.
        **kwargs: Keyword arguments for the function.

    Returns:
        The function's return value.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"Retry attempt {attempt + 1}/{config.max_retries} "
                    f"for {func.__name__} after {delay:.2f}s delay. "
                    f"Error: {str(e)}"
                )
                await asyncio.sleep(delay)

    raise last_exception  # type: ignore
