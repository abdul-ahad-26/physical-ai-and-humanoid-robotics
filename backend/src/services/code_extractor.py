"""
Code block extraction and restoration for AI content processing.

Extracts code blocks from markdown content before sending to AI,
replaces with placeholders, then restores after processing.
This ensures 100% code preservation during personalization/translation.
"""

import re
from dataclasses import dataclass
from typing import List


# Regex patterns for code detection
CODE_BLOCK_PATTERN = r"```[\s\S]*?```"
INLINE_CODE_PATTERN = r"`[^`\n]+`"


@dataclass
class ExtractedContent:
    """Container for extracted content with code blocks removed."""

    text: str
    code_blocks: List[str]
    inline_codes: List[str]


def extract_code(content: str) -> ExtractedContent:
    """
    Extract code blocks and inline code from markdown content.

    Replaces code with placeholders that the AI will preserve.
    Fenced code blocks (```) are extracted first, then inline code (`).

    Args:
        content: Original markdown content

    Returns:
        ExtractedContent with text (placeholders) and lists of extracted code
    """
    # Extract fenced code blocks first (they may contain backticks)
    code_blocks = re.findall(CODE_BLOCK_PATTERN, content)
    for i, block in enumerate(code_blocks):
        content = content.replace(block, f"[CODE_BLOCK_{i}]", 1)

    # Extract inline code
    inline_codes = re.findall(INLINE_CODE_PATTERN, content)
    for i, code in enumerate(inline_codes):
        content = content.replace(code, f"[INLINE_CODE_{i}]", 1)

    return ExtractedContent(
        text=content,
        code_blocks=code_blocks,
        inline_codes=inline_codes,
    )


def restore_code(content: str, extracted: ExtractedContent) -> str:
    """
    Restore code blocks and inline code from placeholders.

    Args:
        content: Processed content with placeholders
        extracted: ExtractedContent containing original code

    Returns:
        Content with code blocks and inline code restored
    """
    # Restore code blocks
    for i, block in enumerate(extracted.code_blocks):
        content = content.replace(f"[CODE_BLOCK_{i}]", block)

    # Restore inline code
    for i, code in enumerate(extracted.inline_codes):
        content = content.replace(f"[INLINE_CODE_{i}]", code)

    return content


def validate_code_preservation(original: str, processed: str) -> bool:
    """
    Verify all code blocks are preserved exactly.

    Compares code blocks between original and processed content
    to ensure no modifications occurred.

    Args:
        original: Original content before processing
        processed: Content after processing

    Returns:
        True if all code blocks match exactly
    """
    original_blocks = re.findall(CODE_BLOCK_PATTERN, original)
    processed_blocks = re.findall(CODE_BLOCK_PATTERN, processed)

    if len(original_blocks) != len(processed_blocks):
        return False

    for orig, proc in zip(original_blocks, processed_blocks):
        if orig != proc:
            return False

    return True


def count_code_blocks(content: str) -> dict:
    """
    Count code blocks and inline code in content.

    Useful for metadata in API responses.

    Args:
        content: Markdown content

    Returns:
        Dictionary with counts
    """
    code_blocks = re.findall(CODE_BLOCK_PATTERN, content)
    inline_codes = re.findall(INLINE_CODE_PATTERN, content)

    return {
        "code_blocks": len(code_blocks),
        "inline_codes": len(inline_codes),
        "total": len(code_blocks) + len(inline_codes),
    }
