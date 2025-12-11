import re
from typing import Union, List, Dict, Any
from html import escape, unescape
import bleach


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent XSS and other injection attacks.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return text

    # Remove potentially dangerous HTML tags and attributes
    cleaned = bleach.clean(
        text,
        tags=['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
        attributes={},
        strip=True
    )

    # Escape any remaining HTML entities
    cleaned = escape(cleaned)

    # Remove any potential script tags that might have slipped through
    cleaned = re.sub(r'<script[^>]*>.*?</script>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<iframe[^>]*>.*?</iframe>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<object[^>]*>.*?</object>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<embed[^>]*>.*?</embed>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove javascript:, vbscript:, and other dangerous protocols
    cleaned = re.sub(r'javascript:', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'vbscript:', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'on\w+\s*=', '', cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def sanitize_markdown_content(content: str) -> str:
    """
    Sanitize Markdown content to prevent injection attacks while preserving formatting.

    Args:
        content: Markdown content to sanitize

    Returns:
        Sanitized Markdown content
    """
    if not content or not isinstance(content, str):
        return content

    # Sanitize the content by escaping HTML
    sanitized = escape(content)

    # Allow basic markdown formatting characters but prevent HTML injection
    # This is a simplified approach - a more robust solution would use a proper Markdown parser
    return sanitized


def sanitize_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize query parameters.

    Args:
        params: Dictionary of query parameters

    Returns:
        Sanitized query parameters
    """
    sanitized_params = {}
    for key, value in params.items():
        if isinstance(value, str):
            sanitized_params[key] = sanitize_input(value)
        elif isinstance(value, (list, tuple)):
            sanitized_params[key] = [sanitize_input(str(item)) if isinstance(item, str) else item for item in value]
        else:
            sanitized_params[key] = value
    return sanitized_params


def validate_content_type(content: str, expected_type: str = "markdown") -> bool:
    """
    Validate that content matches the expected type.

    Args:
        content: Content to validate
        expected_type: Expected content type ("markdown" or "html")

    Returns:
        True if content is valid, False otherwise
    """
    if not content:
        return True  # Empty content is considered valid

    if expected_type.lower() == "html":
        # Check for potentially dangerous HTML patterns
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'<iframe',
            r'<object',
            r'<embed',
            r'on\w+\s*='
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False

    elif expected_type.lower() == "markdown":
        # For markdown, we'll allow more flexibility but still check for obvious HTML injection
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'<iframe',
            r'<object',
            r'<embed'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False

    return True


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata dictionary.

    Args:
        metadata: Metadata dictionary to sanitize

    Returns:
        Sanitized metadata
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_input(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_metadata(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_input(item) if isinstance(item, str) else item for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "<script>alert('xss')</script>Hello World",
        "Normal text with <b>formatting</b>",
        "javascript:alert('xss')",
        "<iframe src='http://malicious.com'></iframe>",
        "Valid markdown content with #headers and **formatting**"
    ]

    print("Input sanitization test results:")
    for i, test_case in enumerate(test_cases, 1):
        sanitized = sanitize_input(test_case)
        print(f"Test {i}:")
        print(f"  Original: {test_case}")
        print(f"  Sanitized: {sanitized}")
        print()