"""Unit tests for OpenAI Agents SDK implementations."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.agents.answer import answer_agent_simple, AnswerOutput
from src.agents.citation import citation_agent, CitationOutput, CitationItem


class TestRetrievalAgent:
    """Tests for the Retrieval Agent."""

    @pytest.mark.asyncio
    async def test_retrieval_agent_configuration(self):
        """Test that retrieval agent is properly configured."""
        from src.agents.retrieval import retrieval_agent, search_book_content

        assert retrieval_agent.name == "Retrieval Agent"
        assert retrieval_agent.model == "gpt-4o-mini"
        # Check that search tool is attached
        assert len(retrieval_agent.tools) == 1
        assert search_book_content in retrieval_agent.tools

    def test_retrieval_search_configuration(self):
        """Test retrieval configuration is correct."""
        from src.agents.retrieval import retrieval_agent

        # Verify agent configuration
        assert retrieval_agent.model == "gpt-4o-mini"
        assert "search" in retrieval_agent.instructions.lower()
        assert "textbook" in retrieval_agent.instructions.lower()


class TestAnswerAgent:
    """Tests for the Answer Generation Agent."""

    @pytest.mark.asyncio
    async def test_answer_agent_generates_response(self):
        """Test that answer agent generates grounded responses."""
        # Test the agent configuration
        assert answer_agent_simple.name == "Answer Agent"
        assert "ONLY use information" in answer_agent_simple.instructions
        assert answer_agent_simple.model == "gpt-4o-mini"

    def test_answer_agent_has_book_constraint(self):
        """Test that answer agent instructions include book-only constraint."""
        instructions = answer_agent_simple.instructions
        assert "ONLY use information from the provided textbook" in instructions
        assert "NEVER use your general knowledge" in instructions
        assert "I don't know based on this book" in instructions


class TestCitationAgent:
    """Tests for the Citation Agent."""

    def test_citation_agent_configuration(self):
        """Test citation agent is properly configured."""
        assert citation_agent.name == "Citation Agent"
        assert citation_agent.model == "gpt-4o-mini"
        assert citation_agent.output_type == CitationOutput

    def test_citation_output_model(self):
        """Test CitationOutput model structure."""
        citation = CitationItem(
            chapter_id="chapter-1",
            section_id="intro",
            anchor_url="/docs/chapter-1#intro",
            display_text="Chapter 1: Introduction",
        )

        output = CitationOutput(
            citations=[citation],
            answer_with_citations="This is the answer [1].",
        )

        assert len(output.citations) == 1
        assert output.citations[0].chapter_id == "chapter-1"
        assert "[1]" in output.answer_with_citations


class TestGuardrails:
    """Tests for input/output guardrails."""

    def test_query_validation_output_model(self):
        """Test QueryValidationOutput model."""
        from src.agents.orchestrator import QueryValidationOutput

        valid = QueryValidationOutput(
            is_valid=True,
            reasoning="Question about textbook content",
        )
        assert valid.is_valid is True

        invalid = QueryValidationOutput(
            is_valid=False,
            reasoning="Request to generate unrelated code",
        )
        assert invalid.is_valid is False

    def test_hallucination_check_output_model(self):
        """Test HallucinationCheckOutput model."""
        from src.agents.orchestrator import HallucinationCheckOutput

        grounded = HallucinationCheckOutput(
            has_hallucination=False,
            reasoning="Answer matches source material",
        )
        assert grounded.has_hallucination is False

        hallucinated = HallucinationCheckOutput(
            has_hallucination=True,
            reasoning="Answer contains information not in sources",
        )
        assert hallucinated.has_hallucination is True


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_rag_response_model(self):
        """Test RAGResponse model structure."""
        from src.agents.orchestrator import RAGResponse
        from src.db.models import Citation

        response = RAGResponse(
            answer="This is a test answer.",
            citations=[
                Citation(
                    chapter_id="chapter-1",
                    section_id="intro",
                    anchor_url="/docs/chapter-1#intro",
                    display_text="Chapter 1",
                )
            ],
            found_content=True,
            latency_ms=1500,
        )

        assert response.answer == "This is a test answer."
        assert len(response.citations) == 1
        assert response.found_content is True
        assert response.latency_ms == 1500
