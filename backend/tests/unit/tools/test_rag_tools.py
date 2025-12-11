import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools.rag_tools import RAGTools
from src.agent.rag_agent import RAGAgent


class TestRAGTools:
    """Unit tests for RAGTools class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.rag_tools = RAGTools()

    def test_retrieve_context_success(self):
        """Test successful context retrieval."""
        # Mock the RAG agent's process_query method
        mock_result = {
            "retrieved_contexts": [
                {
                    "content": "Test content",
                    "metadata": {"source_file": "test.md"},
                    "score": 0.9
                }
            ],
            "assembled_context": "Test assembled context",
            "query_id": "test_query_id"
        }

        with patch.object(self.rag_tools.rag_agent, 'process_query', return_value=mock_result):
            result = self.rag_tools.retrieve_context("test query", k=3)

        assert result == mock_result

    def test_retrieve_by_source_success(self):
        """Test successful retrieval by source."""
        mock_results = [
            {
                "content": "Test content",
                "metadata": {"source_file": "test.md"},
                "score": 0.9
            }
        ]

        with patch.object(self.rag_tools.rag_agent, 'retrieve_by_source', return_value=mock_results):
            result = self.rag_tools.retrieve_by_source("test.md", k=5)

        assert result == mock_results

    def test_add_content_to_knowledge_base_success(self):
        """Test successful addition of content to knowledge base."""
        with patch.object(self.rag_tools.rag_agent, 'add_content', return_value=True):
            result = self.rag_tools.add_content_to_knowledge_base("test content", {"source_file": "test.md"})

        assert result is True

    def test_add_content_to_knowledge_base_without_metadata(self):
        """Test addition of content without metadata."""
        with patch.object(self.rag_tools.rag_agent, 'add_content', return_value=True):
            result = self.rag_tools.add_content_to_knowledge_base("test content")

        assert result is True

    def test_batch_add_content_to_knowledge_base(self):
        """Test batch addition of content to knowledge base."""
        mock_contents = [
            {"content": "content 1", "metadata": {"source_file": "file1.md"}},
            {"content": "content 2", "metadata": {"source_file": "file2.md"}}
        ]

        with patch.object(self.rag_tools.rag_agent, 'batch_add_content', return_value=2):
            result = self.rag_tools.batch_add_content_to_knowledge_base(mock_contents)

        assert result == 2

    def test_delete_content_from_knowledge_base(self):
        """Test deletion of content from knowledge base."""
        with patch.object(self.rag_tools.rag_agent, 'delete_content_by_source', return_value=True):
            result = self.rag_tools.delete_content_from_knowledge_base("test.md")

        assert result is True

    def test_rank_results(self):
        """Test ranking of results."""
        mock_results = [
            {"content": "content1", "metadata": {}, "score": 0.5},
            {"content": "content2", "metadata": {}, "score": 0.9},
            {"content": "content3", "metadata": {}, "score": 0.7}
        ]

        ranked_results = self.rag_tools.rank_results("test query", mock_results)

        # Check that results are sorted by score in descending order
        assert ranked_results[0]["score"] == 0.9  # Highest score first
        assert ranked_results[1]["score"] == 0.7
        assert ranked_results[2]["score"] == 0.5  # Lowest score last

    def test_get_retrieval_statistics(self):
        """Test retrieval statistics."""
        mock_stats = {"total_chunks": 100, "collections": ["test_collection"], "embedding_model": "test_model"}

        with patch.object(self.rag_tools.rag_agent, 'get_retrieval_stats', return_value=mock_stats):
            result = self.rag_tools.get_retrieval_statistics()

        assert result == mock_stats


if __name__ == "__main__":
    pytest.main([__file__])