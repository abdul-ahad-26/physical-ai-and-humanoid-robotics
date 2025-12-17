"""Citation Agent - Maps answer segments to chapter/section anchors."""

from typing import List

from agents import Agent
from pydantic import BaseModel


class CitationItem(BaseModel):
    """A single citation reference."""

    chapter_id: str
    section_id: str
    anchor_url: str
    display_text: str


class CitationOutput(BaseModel):
    """Output from the citation agent."""

    citations: List[CitationItem]
    answer_with_citations: str


# Create the Citation Agent
citation_agent = Agent(
    name="Citation Agent",
    instructions="""You are a citation specialist for a textbook assistant.

Your job is to:
1. Review the generated answer and the source chunks it was based on
2. Create proper citations that link answer content to specific textbook sections
3. Format citations with clickable anchor URLs

For each key claim or piece of information in the answer:
- Identify which source chunk it came from
- Create a citation with:
  - chapter_id: The chapter identifier (e.g., "chapter-1")
  - section_id: The section identifier (e.g., "introduction")
  - anchor_url: The URL path with anchor (e.g., "/docs/chapter-1#introduction")
  - display_text: Human-readable text (e.g., "Chapter 1: Introduction")

Return:
- citations: Array of citation objects
- answer_with_citations: The answer with citation markers [1], [2], etc. inserted

Example citation format in answer:
"Physical AI systems use sensors to perceive their environment [1] and actuators to interact with it [2]."

Citations should be numbered sequentially and each unique source should have its own citation.""",
    model="gpt-4o-mini",
    output_type=CitationOutput,
)


def format_citations_for_display(citations: List[CitationItem]) -> str:
    """Format citations list for display to user.

    Args:
        citations: List of citation items.

    Returns:
        Formatted string with numbered citations.
    """
    if not citations:
        return ""

    lines = ["\n\n**Sources:**"]
    for i, citation in enumerate(citations, 1):
        lines.append(
            f"[{i}] [{citation.display_text}]({citation.anchor_url})"
        )

    return "\n".join(lines)
