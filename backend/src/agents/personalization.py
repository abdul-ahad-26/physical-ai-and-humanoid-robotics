"""Personalization Agent - Adapts chapter content based on user's technical background."""

from typing import List, Optional

from agents import Agent, Runner
from pydantic import BaseModel

from src.config import get_settings


class UserProfileData(BaseModel):
    """User profile data for personalization context."""

    software_level: str  # beginner | intermediate | advanced
    software_languages: List[str]
    software_frameworks: List[str]
    hardware_level: str  # none | basic | intermediate | advanced
    hardware_domains: List[str]


class PersonalizedOutput(BaseModel):
    """Structured output from the personalization agent."""

    content: str
    adaptations_made: List[str]


PERSONALIZATION_INSTRUCTIONS = """
You are a content personalization agent for a Physical AI & Robotics textbook.

Your task is to adapt chapter content based on the user's technical background.

USER PROFILE:
- Software Experience: {software_level}
- Languages: {languages}
- Frameworks: {frameworks}
- Hardware Experience: {hardware_level}
- Hardware Domains: {domains}

ADAPTATION RULES:
1. For BEGINNER software users:
   - Add foundational context before complex concepts
   - Use everyday analogies to explain technical terms
   - Expand acronyms on first use (e.g., "API (Application Programming Interface)")
   - Add "Why this matters" explanations
   - Provide step-by-step breakdowns of processes

2. For INTERMEDIATE software users:
   - Balance theory with practical examples
   - Reference tools/frameworks they already know
   - Add "Pro tips" for efficiency
   - Include common pitfalls to avoid

3. For ADVANCED software users:
   - Skip basic explanations they already know
   - Add edge cases and optimization notes
   - Reference papers/specs where relevant
   - Include performance considerations
   - Discuss architectural tradeoffs

4. For users with NO/BASIC hardware experience:
   - Add context about physical components
   - Explain hardware concepts in software terms when possible
   - Provide visual analogies for physical processes

5. For users with INTERMEDIATE/ADVANCED hardware experience:
   - Reference their domain expertise ({domains})
   - Draw connections to their hardware background
   - Add technical depth to hardware discussions

CRITICAL GUARDRAILS:
- NEVER change code examples (marked as [CODE_BLOCK_N])
- NEVER alter factual information, numbers, or specifications
- NEVER remove safety warnings or important caveats
- PRESERVE all markdown formatting exactly
- MAINTAIN section structure, headers, and lists
- PRESERVE all URLs and links exactly as they appear
- Keep the same overall length (within 20% of original)

OUTPUT FORMAT:
- Return the adapted content with the same markdown structure
- List the specific adaptations you made in adaptations_made

Adapt the following chapter content:
"""


async def personalize_content(
    content: str,
    profile: UserProfileData,
    chapter_title: str,
) -> PersonalizedOutput:
    """Personalize chapter content based on user profile.

    Args:
        content: The chapter content with code blocks already extracted
                (code blocks replaced with [CODE_BLOCK_N] placeholders).
        profile: The user's technical background profile.
        chapter_title: The title of the chapter being personalized.

    Returns:
        PersonalizedOutput with adapted content and list of adaptations made.
    """
    settings = get_settings()

    # Format instructions with profile data
    formatted_instructions = PERSONALIZATION_INSTRUCTIONS.format(
        software_level=profile.software_level,
        languages=", ".join(profile.software_languages) if profile.software_languages else "None specified",
        frameworks=", ".join(profile.software_frameworks) if profile.software_frameworks else "None specified",
        hardware_level=profile.hardware_level,
        domains=", ".join(profile.hardware_domains) if profile.hardware_domains else "None specified",
    )

    # Create agent with formatted instructions
    agent = Agent(
        name="Personalization Agent",
        instructions=formatted_instructions,
        output_type=PersonalizedOutput,
        model=settings.openai_model,
    )

    # Create the prompt with chapter context
    prompt = f"""
Chapter Title: {chapter_title}

Content to personalize:
{content}
"""

    # Run the agent
    result = await Runner.run(agent, prompt)

    return result.final_output_as(PersonalizedOutput)


# Pre-configured agent for simpler use cases (without profile interpolation)
personalization_agent = Agent(
    name="Personalization Agent",
    instructions="""You are a content personalization agent for a Physical AI & Robotics textbook.

Adapt the chapter content based on the user profile provided in the prompt.

CRITICAL GUARDRAILS:
- NEVER change code examples (marked as [CODE_BLOCK_N])
- NEVER alter factual information
- NEVER remove safety warnings
- PRESERVE all markdown formatting
- MAINTAIN section structure and headers""",
    output_type=PersonalizedOutput,
    model="gpt-4o",
)
