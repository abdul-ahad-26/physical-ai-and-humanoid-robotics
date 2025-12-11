import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools.indexing_tools import IndexingTools
from src.agent.indexing_agent import IndexingAgent


class TestIndexingTools:
    """Unit tests for IndexingTools class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.indexing_tools = IndexingTools()

    def test_index_content_success(self):
        """Test successful content indexing."""
        mock_result = {
            "status": "success",
            "indexed_chunks": 3,
            "content_id": "content_abc123",
            "processing_time": 0.5
        }

        with patch.object(self.indexing_tools.indexing_agent, 'index_content', return_value=mock_result):
            result = self.indexing_tools.index_content("test content", "test.md", "markdown", "Chapter 1")

        assert result == mock_result

    def test_update_content_success(self):
        """Test successful content update."""
        mock_result = {
            "status": "success",
            "indexed_chunks": 3,
            "content_id": "content_def456",
            "processing_time": 0.7
        }

        with patch.object(self.indexing_tools.indexing_agent, 'update_content', return_value=mock_result):
            result = self.indexing_tools.update_content("updated content", "test.md", "markdown", "Chapter 1")

        assert result == mock_result

    def test_delete_content_success(self):
        """Test successful content deletion."""
        mock_result = {
            "status": "success",
            "deleted_source": "test.md",
            "processing_time": 0.2
        }

        with patch.object(self.indexing_tools.indexing_agent, 'delete_content', return_value=mock_result):
            result = self.indexing_tools.delete_content("test.md")

        assert result == mock_result

    def test_validate_content_format_success(self):
        """Test successful content format validation."""
        mock_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        with patch.object(self.indexing_tools.indexing_agent, 'validate_content_format', return_value=mock_result):
            result = self.indexing_tools.validate_content_format("valid markdown content", "markdown")

        assert result == mock_result

    def test_get_indexing_statistics(self):
        """Test retrieval of indexing statistics."""
        mock_stats = {
            "retriever_stats": {"total_chunks": 100},
            "chunker_info": {"max_chunk_size": 1000}
        }

        with patch.object(self.indexing_tools.indexing_agent, 'get_indexing_stats', return_value=mock_stats):
            result = self.indexing_tools.get_indexing_statistics()

        assert result == mock_stats

    def test_batch_index_content(self):
        """Test batch content indexing."""
        mock_contents = [
            {"content": "content 1", "source_file": "file1.md", "document_type": "markdown"},
            {"content": "content 2", "source_file": "file2.md", "document_type": "html"}
        ]

        mock_results = [
            {
                "status": "success",
                "indexed_chunks": 2,
                "content_id": "content_abc123",
                "processing_time": 0.3
            },
            {
                "status": "success",
                "indexed_chunks": 3,
                "content_id": "content_def456",
                "processing_time": 0.4
            }
        ]

        with patch.object(self.indexing_tools, 'index_content') as mock_index_content:
            mock_index_content.side_effect = mock_results
            results = self.indexing_tools.batch_index_content(mock_contents)

        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__])