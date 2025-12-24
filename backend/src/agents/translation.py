"""Translation Agent - Translates chapter content to Urdu with code block preservation."""

from typing import List

from agents import Agent, Runner
from pydantic import BaseModel

from src.config import get_settings


class TranslatedOutput(BaseModel):
    """Structured output from the translation agent."""

    content: str
    preserved_blocks: int


TRANSLATION_INSTRUCTIONS = """
You are a technical translation agent specializing in Urdu translation for a Physical AI & Robotics textbook.

Your task is to translate educational content about Physical AI & Robotics to Urdu (اردو).

TRANSLATION RULES:

1. Translate ALL prose text to proper Urdu script
   - Use natural Urdu sentence structure
   - Maintain the educational tone

2. PRESERVE code blocks EXACTLY as-is
   - Code blocks are marked as [CODE_BLOCK_N] placeholders
   - NEVER translate or modify these placeholders
   - They will be restored to original code after translation

3. PRESERVE inline code EXACTLY as-is
   - Inline code is marked as [INLINE_CODE_N] placeholders
   - Keep these placeholders unchanged

4. Keep technical terms in English WITH Urdu transliteration in parentheses:
   - API → API (اے پی آئی)
   - GPU → GPU (جی پی یو)
   - CPU → CPU (سی پی یو)
   - Neural Network → Neural Network (نیورل نیٹ ورک)
   - Machine Learning → Machine Learning (مشین لرننگ)
   - Deep Learning → Deep Learning (ڈیپ لرننگ)
   - Robot → Robot (روبوٹ)
   - Sensor → Sensor (سینسر)
   - Actuator → Actuator (ایکچویٹر)
   - Framework → Framework (فریم ورک)
   - Algorithm → Algorithm (الگورتھم)
   - ROS → ROS (آر او ایس)
   - Simulation → Simulation (سمولیشن)
   - Python → Python (پائتھون)
   - Linux → Linux (لینکس)

5. PRESERVE all URLs, file paths, and command examples
   - URLs should remain unchanged
   - File paths like `/home/user/` stay as-is
   - Shell commands stay in English

6. MAINTAIN markdown formatting:
   - Headers (# ## ###) - translate the header text, keep the markdown syntax
   - Lists (- or 1.) - translate list items, keep the syntax
   - Bold (**text**) and italic (*text*) - translate the text, keep the syntax
   - Links [text](url) - translate the display text, keep the URL

7. Numbers and measurements:
   - Keep numerical values as-is
   - Translate units: meters → میٹر, seconds → سیکنڈ, etc.

OUTPUT FORMAT:
- Return the translated content with the same markdown structure
- Report the number of preserved code blocks

Translate the following content to Urdu:
"""


async def translate_content(
    content: str,
    target_language: str = "ur",
) -> TranslatedOutput:
    """Translate chapter content to Urdu.

    Args:
        content: The chapter content with code blocks already extracted
                (code blocks replaced with [CODE_BLOCK_N] placeholders).
        target_language: Target language code (default: "ur" for Urdu).

    Returns:
        TranslatedOutput with translated content and count of preserved blocks.
    """
    settings = get_settings()

    # Count code block placeholders in content
    import re
    code_block_count = len(re.findall(r'\[CODE_BLOCK_\d+\]', content))
    inline_code_count = len(re.findall(r'\[INLINE_CODE_\d+\]', content))
    total_preserved = code_block_count + inline_code_count

    # Create the translation agent
    agent = Agent(
        name="Translation Agent",
        instructions=TRANSLATION_INSTRUCTIONS,
        output_type=TranslatedOutput,
        model=settings.openai_model,
    )

    # Run the agent
    result = await Runner.run(agent, content)

    output = result.final_output_as(TranslatedOutput)

    # Ensure preserved_blocks count is accurate
    output.preserved_blocks = total_preserved

    return output


# Pre-configured agent for direct use
translation_agent = Agent(
    name="Translation Agent",
    instructions=TRANSLATION_INSTRUCTIONS,
    output_type=TranslatedOutput,
    model="gpt-4o",
)
