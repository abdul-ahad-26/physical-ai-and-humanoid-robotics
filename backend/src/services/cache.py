"""
Session-scoped caching service for personalization and translation.

Uses TTLCache from cachetools for in-memory caching with automatic expiration.
Cache keys are formatted as: {user_id}:{chapter_id}:{content_type}
"""

from typing import Optional

from cachetools import TTLCache

from src.config import get_settings


def _get_cache_ttl() -> int:
    """Get cache TTL from settings, default 3600 seconds (1 hour)."""
    settings = get_settings()
    return getattr(settings, "personalization_cache_ttl", 3600)


# Global caches with configurable TTL
# Max 1000 entries per cache to prevent memory bloat
personalization_cache: TTLCache = TTLCache(maxsize=1000, ttl=_get_cache_ttl())
translation_cache: TTLCache = TTLCache(maxsize=1000, ttl=_get_cache_ttl())


def get_cache_key(user_id: str, chapter_id: str, content_type: str) -> str:
    """
    Generate a cache key for personalization or translation.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
        content_type: Type of content ('personalized' or 'translated')

    Returns:
        Cache key string
    """
    return f"{user_id}:{chapter_id}:{content_type}"


# =============================================================================
# Personalization Cache
# =============================================================================


def get_cached_personalization(user_id: str, chapter_id: str) -> Optional[str]:
    """
    Retrieve cached personalized content.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter

    Returns:
        Cached content string or None if not found/expired
    """
    key = get_cache_key(user_id, chapter_id, "personalized")
    return personalization_cache.get(key)


def set_cached_personalization(user_id: str, chapter_id: str, content: str) -> None:
    """
    Cache personalized content.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
        content: Personalized content to cache
    """
    key = get_cache_key(user_id, chapter_id, "personalized")
    personalization_cache[key] = content


def invalidate_personalization_cache(user_id: str, chapter_id: str) -> None:
    """
    Invalidate cached personalized content for a user+chapter.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
    """
    key = get_cache_key(user_id, chapter_id, "personalized")
    personalization_cache.pop(key, None)


# =============================================================================
# Translation Cache
# =============================================================================


def get_cached_translation(
    user_id: str, chapter_id: str, target_language: str = "ur"
) -> Optional[str]:
    """
    Retrieve cached translated content.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
        target_language: Target language code (default: 'ur' for Urdu)

    Returns:
        Cached content string or None if not found/expired
    """
    key = get_cache_key(user_id, chapter_id, f"translated_{target_language}")
    return translation_cache.get(key)


def set_cached_translation(
    user_id: str, chapter_id: str, content: str, target_language: str = "ur"
) -> None:
    """
    Cache translated content.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
        content: Translated content to cache
        target_language: Target language code (default: 'ur' for Urdu)
    """
    key = get_cache_key(user_id, chapter_id, f"translated_{target_language}")
    translation_cache[key] = content


def invalidate_translation_cache(
    user_id: str, chapter_id: str, target_language: str = "ur"
) -> None:
    """
    Invalidate cached translated content for a user+chapter+language.

    Args:
        user_id: UUID of the user
        chapter_id: Identifier of the chapter
        target_language: Target language code (default: 'ur' for Urdu)
    """
    key = get_cache_key(user_id, chapter_id, f"translated_{target_language}")
    translation_cache.pop(key, None)


# =============================================================================
# Cache Management
# =============================================================================


def clear_user_cache(user_id: str) -> None:
    """
    Clear all cached content for a specific user.

    Useful when user updates their profile (invalidates personalized content).

    Args:
        user_id: UUID of the user
    """
    # Clear personalization cache entries for this user
    keys_to_remove = [k for k in personalization_cache.keys() if k.startswith(f"{user_id}:")]
    for key in keys_to_remove:
        personalization_cache.pop(key, None)

    # Clear translation cache entries for this user
    keys_to_remove = [k for k in translation_cache.keys() if k.startswith(f"{user_id}:")]
    for key in keys_to_remove:
        translation_cache.pop(key, None)


def get_cache_stats() -> dict:
    """
    Get cache statistics for monitoring.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "personalization": {
            "current_size": len(personalization_cache),
            "max_size": personalization_cache.maxsize,
            "ttl": personalization_cache.ttl,
        },
        "translation": {
            "current_size": len(translation_cache),
            "max_size": translation_cache.maxsize,
            "ttl": translation_cache.ttl,
        },
    }
