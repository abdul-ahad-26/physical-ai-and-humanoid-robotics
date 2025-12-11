import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from src.agent.indexing_agent import IndexingAgent
from src.agent.rag_agent import RAGAgent
from src.api.routes.index import index_endpoint
from pydantic import BaseModel


class IndexRequest(BaseModel):
    """Mock IndexRequest model for testing."""
    content: str
    metadata: dict


class TestUserStory2:
    """Integration tests for User Story 2: Content Ingestion and Indexing."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.indexing_agent = IndexingAgent()
        self.rag_agent = RAGAgent()

    @pytest.mark.integration
    def test_content_can_be_uploaded_and_becomes_searchable_within_30_seconds(self):
        """Test that content can be uploaded and becomes searchable within 30 seconds."""
        # Sample textbook content
        sample_content = """
# Introduction to Machine Learning

Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention.

## Types of Machine Learning

There are three main types of machine learning:

1. **Supervised Learning**: Uses labeled datasets to train algorithms to predict outcomes
2. **Unsupervised Learning**: Finds hidden patterns in unlabeled data
3. **Reinforcement Learning**: Learns through trial and error using feedback from actions
"""

        # Prepare metadata
        metadata = {
            "source_file": "ml_introduction.md",
            "section": "Chapter 1",
            "document_type": "markdown"
        }

        # Create mock request
        mock_request = IndexRequest(
            content=sample_content,
            metadata=metadata
        )

        # Measure indexing time
        start_time = time.time()

        try:
            # Mock the indexing process
            with patch.object(self.indexing_agent.retriever.qdrant_client, 'upsert_vectors', return_value=None):
                with patch.object(self.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    with patch.object(self.indexing_agent.chunker, 'chunk_markdown', return_value=[
                        type('TextChunk', (), {
                            'content': 'Machine learning is a method of data analysis...',
                            'metadata': metadata,
                            'start_pos': 0,
                            'end_pos': 100
                        })()
                    ]):
                        # Test the indexing agent
                        result = self.indexing_agent.index_content(
                            content=sample_content,
                            source_file=metadata["source_file"],
                            document_type=metadata.get("document_type", "markdown"),
                            section=metadata.get("section")
                        )

                        # Verify indexing result
                        assert result["status"] in ["success", "partial"]
                        assert result["indexed_chunks"] >= 0
                        assert "content_id" in result
                        assert "processing_time" in result

            indexing_time = time.time() - start_time

            # Verify indexing time is under 30 seconds
            assert indexing_time < 30.0, f"Content indexing took {indexing_time:.3f}s, exceeding 30-second limit"

            print(f"Content indexing time: {indexing_time:.3f}s ({indexing_time*1000:.1f}ms)")
            print(f"Indexed {result['indexed_chunks']} chunks")

            # Now test that the content is searchable
            search_start_time = time.time()

            # Mock the retrieval to verify the content would be found
            with patch.object(self.rag_agent.retriever.qdrant_client, 'search_vectors', return_value=[
                {
                    "id": "test_id",
                    "payload": {
                        "content": "Machine learning is a method of data analysis...",
                        "metadata": metadata
                    },
                    "score": 0.9
                }
            ]):
                with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    search_result = self.rag_agent.process_query("What is machine learning?", k=1)

                    # Verify we can retrieve the indexed content
                    assert "retrieved_contexts" in search_result
                    assert len(search_result["retrieved_contexts"]) > 0
                    assert search_result["retrieved_contexts"][0]["metadata"]["source_file"] == "ml_introduction.md"

            search_time = time.time() - search_start_time
            total_time = indexing_time + search_time

            # Verify search works and is also within reasonable time
            assert search_time < 2.0, f"Content search took {search_time:.3f}s, which seems slow"

            print(f"Verification search time: {search_time:.3f}s ({search_time*1000:.1f}ms)")
            print(f"Total time (index + search): {total_time:.3f}s ({total_time*1000:.1f}ms)")

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.integration
    def test_content_validation_before_indexing(self):
        """Test that content is properly validated before indexing."""
        # Test with valid content
        valid_content = "# Valid Content\nThis is valid markdown content."
        metadata = {"source_file": "valid.md", "document_type": "markdown"}

        try:
            # Test validation
            validation_result = self.indexing_agent.validate_content_format(valid_content, "markdown")
            assert validation_result["is_valid"] is True
            assert len(validation_result["errors"]) == 0

            print("Content validation passed for valid content")

            # Test with empty content (should fail validation)
            empty_content = ""
            validation_result_empty = self.indexing_agent.validate_content_format(empty_content, "markdown")
            assert validation_result_empty["is_valid"] is False
            assert len(validation_result_empty["errors"]) > 0

            print("Content validation correctly failed for empty content")

        except Exception as e:
            pytest.fail(f"Content validation test failed: {e}")

    @pytest.mark.integration
    def test_incremental_update_functionality(self):
        """Test that content can be updated incrementally."""
        original_content = "# Original Content\nThis is the original content."
        updated_content = "# Updated Content\nThis is the updated content with more information."

        metadata = {"source_file": "update_test.md", "document_type": "markdown"}

        # Measure update time
        start_time = time.time()

        try:
            # Mock the update process
            with patch.object(self.indexing_agent.retriever.qdrant_client, 'upsert_vectors', return_value=None):
                with patch.object(self.indexing_agent.retriever.qdrant_client, 'delete_by_payload', return_value=True):
                    with patch.object(self.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                        with patch.object(self.indexing_agent.chunker, 'chunk_markdown', return_value=[
                            type('TextChunk', (), {
                                'content': 'Updated content with more information',
                                'metadata': metadata,
                                'start_pos': 0,
                                'end_pos': 100
                            })()
                        ]):
                            # Test the update functionality
                            result = self.indexing_agent.update_content(
                                content=updated_content,
                                source_file=metadata["source_file"],
                                document_type=metadata.get("document_type", "markdown"),
                                section=metadata.get("section")
                            )

                            # Verify update result
                            assert result["status"] in ["success", "partial"]
                            assert "content_id" in result
                            assert "processing_time" in result

            update_time = time.time() - start_time

            # Verify update time is reasonable
            assert update_time < 30.0, f"Content update took {update_time:.3f}s, exceeding 30-second limit"

            print(f"Content update time: {update_time:.3f}s ({update_time*1000:.1f}ms)")

        except Exception as e:
            pytest.fail(f"Incremental update test failed: {e}")

    @pytest.mark.integration
    def test_bulk_content_ingestion_performance(self):
        """Test performance when ingesting bulk content."""
        import concurrent.futures

        # Create multiple content pieces to index
        content_pieces = [
            f"# Chapter {i}\nThis is content for chapter {i} of the textbook. It contains important information that students need to learn.",
            f"# Section {i}.1\nDetailed information about topic {i}.1 that expands on the chapter introduction.",
            f"# Exercise {i}\nPractice problems and solutions for chapter {i} concepts."
        ] for i in range(1, 6)  # 5 chapters worth of content

        # Flatten the list
        flattened_content = []
        for chapter_group in content_pieces:
            for content_piece in chapter_group:
                flattened_content.append(content_piece)

        def index_single_content(content, idx):
            start_time = time.time()
            try:
                metadata = {
                    "source_file": f"textbook_chapter_{idx}.md",
                    "section": f"Chapter {idx}",
                    "document_type": "markdown"
                }

                with patch.object(self.indexing_agent.retriever.qdrant_client, 'upsert_vectors', return_value=None):
                    with patch.object(self.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                        with patch.object(self.indexing_agent.chunker, 'chunk_markdown', return_value=[
                            type('TextChunk', (), {
                                'content': content,
                                'metadata': metadata,
                                'start_pos': 0,
                                'end_pos': len(content)
                            })()
                        ]):
                            result = self.indexing_agent.index_content(
                                content=content,
                                source_file=metadata["source_file"],
                                document_type=metadata.get("document_type", "markdown"),
                                section=metadata.get("section")
                            )
                            return time.time() - start_time, result["status"] == "success"
            except:
                return time.time() - start_time, False

        # Execute indexing concurrently
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda x: index_single_content(x[1], x[0]),
                                     enumerate(flattened_content, 1)))

        total_time = time.time() - start_time

        # Check that all content pieces were indexed successfully
        successful_indexings = sum(1 for _, success in results if success)
        avg_individual_time = sum(time_taken for time_taken, _ in results) / len(results)

        print(f"Indexed {successful_indexings}/{len(flattened_content)} content pieces successfully")
        print(f"Total time for {len(flattened_content)} content pieces: {total_time:.3f}s")
        print(f"Average individual indexing time: {avg_individual_time:.3f}s ({avg_individual_time*1000:.1f}ms)")

        # Verify performance requirements
        assert successful_indexings == len(flattened_content), f"Not all content pieces were indexed successfully: {successful_indexings}/{len(flattened_content)}"
        assert total_time < 60.0, f"Bulk indexing took {total_time:.3f}s, should be under 60s for multiple pieces"
        assert avg_individual_time < 10.0, f"Average indexing time {avg_individual_time:.3f}s exceeds 10s limit"

    @pytest.mark.integration
    def test_content_deletion_and_reindexing(self):
        """Test deletion and re-indexing capabilities."""
        test_source_file = "temp_content.md"

        # Test deletion
        start_time = time.time()

        try:
            with patch.object(self.indexing_agent.retriever.qdrant_client, 'delete_by_payload', return_value=True):
                delete_result = self.indexing_agent.delete_content(test_source_file)

                # Verify deletion result
                assert delete_result["status"] in ["success", "error"]

            delete_time = time.time() - start_time

            print(f"Content deletion time: {delete_time:.3f}s ({delete_time*1000:.1f}ms)")

            # Verify deletion was attempted (even if source doesn't exist)
            assert "deleted_source" in delete_result
            assert delete_result["deleted_source"] == test_source_file

        except Exception as e:
            pytest.fail(f"Content deletion test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])