import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from src.agent.rag_agent import RAGAgent
from src.agent.orchestrator import MainOrchestratorAgent
from src.api.routes.query import query_endpoint
from src.api.routes.answer import answer_endpoint
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Mock QueryRequest model for testing."""
    question: str
    highlight_override: str = None


class AnswerRequest(BaseModel):
    """Mock AnswerRequest model for testing."""
    question: str
    k: int = 3
    highlight_override: str = None


class TestUserStory1:
    """Integration tests for User Story 1: Textbook Q&A Interaction."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.rag_agent = RAGAgent()
        self.orchestrator_agent = MainOrchestratorAgent()

    @pytest.mark.integration
    def test_student_can_ask_questions_and_receive_answers_within_2_seconds(self):
        """Test that students can ask questions and receive contextual answers within 2 seconds."""
        # Mock a realistic query
        test_query = "What is machine learning?"

        # Create mock request
        mock_request = QueryRequest(
            question=test_query,
            highlight_override=None
        )

        # Measure the time it takes to process the query
        start_time = time.time()

        try:
            # Mock the retrieval process to avoid needing actual Qdrant
            with patch.object(self.rag_agent.retriever.qdrant_client, 'search_vectors', return_value=[
                {
                    "id": "test_id",
                    "payload": {
                        "content": "Machine learning is a method of data analysis that automates analytical model building.",
                        "metadata": {"source_file": "ml_basics.md", "section": "Introduction"}
                    },
                    "score": 0.9
                }
            ]):
                with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    # Test the RAG agent directly
                    result = self.rag_agent.process_query(test_query, k=3)

                    # Verify we got results
                    assert "retrieved_contexts" in result
                    assert len(result["retrieved_contexts"]) > 0
                    assert "assembled_context" in result
                    assert "query_id" in result

            query_processing_time = time.time() - start_time

            # Verify query processing time is under 2 seconds
            assert query_processing_time < 2.0, f"Query processing took {query_processing_time:.3f}s, exceeding 2-second limit"

            print(f"Query processing time: {query_processing_time:.3f}s ({query_processing_time*1000:.1f}ms)")

            # Now test the answer generation (which should also be under 2 seconds)
            answer_start_time = time.time()

            # Test answer endpoint functionality
            answer_request = AnswerRequest(
                question=test_query,
                k=3
            )

            # Mock the answer generation process
            with patch.object(self.orchestrator_agent.rag_agent, 'process_query') as mock_process_query:
                # Mock a realistic response
                mock_result = {
                    "answer": "Machine learning is a method of data analysis that automates analytical model building...",
                    "retrieved_contexts": [
                        {
                            "content": "Machine learning is a method of data analysis that automates analytical model building.",
                            "metadata": {"source_file": "ml_basics.md", "section": "Introduction"},
                            "score": 0.9
                        }
                    ],
                    "confidence_score": 0.85,
                    "assembled_context": "Based on textbook content...",
                    "query_id": "test_query_123",
                    "answer_id": "test_answer_456"
                }
                mock_process_query.return_value = mock_result

                answer_result = self.orchestrator_agent.process_query(
                    query=answer_request.question,
                    k=answer_request.k
                )

                # Verify answer result structure
                assert "answer" in answer_result
                assert "retrieved_contexts" in answer_result
                assert "confidence_score" in answer_result

            answer_processing_time = time.time() - answer_start_time
            total_time = query_processing_time + answer_processing_time

            # Verify answer processing time is under 2 seconds
            assert answer_processing_time < 2.0, f"Answer processing took {answer_processing_time:.3f}s, exceeding 2-second limit"
            assert total_time < 2.0, f"Total processing took {total_time:.3f}s, exceeding 2-second limit"

            print(f"Answer processing time: {answer_processing_time:.3f}s ({answer_processing_time*1000:.1f}ms)")
            print(f"Total processing time: {total_time:.3f}s ({total_time*1000:.1f}ms)")

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.integration
    def test_query_with_highlight_override_within_time_limit(self):
        """Test that queries with highlight override work within time limits."""
        test_query = "Explain neural networks"
        highlight_text = "Neural networks are computing systems inspired by the human brain"

        start_time = time.time()

        try:
            # Mock the retrieval process
            with patch.object(self.rag_agent.retriever.qdrant_client, 'search_vectors', return_value=[
                {
                    "id": "nn_id",
                    "payload": {
                        "content": "Neural networks are computing systems inspired by the human brain...",
                        "metadata": {"source_file": "neural_networks.md", "section": "Basics"}
                    },
                    "score": 0.85
                }
            ]):
                with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    result = self.rag_agent.process_query(
                        test_query,
                        k=3,
                        highlight_override=highlight_text
                    )

                    # Verify results
                    assert "retrieved_contexts" in result
                    assert len(result["retrieved_contexts"]) > 0
                    assert result["retrieved_contexts"][0]["content"] is not None

            processing_time = time.time() - start_time

            # Verify time constraint
            assert processing_time < 2.0, f"Processing with highlight override took {processing_time:.3f}s, exceeding 2-second limit"

            print(f"Query with highlight override processing time: {processing_time:.3f}s ({processing_time*1000:.1f}ms)")

        except Exception as e:
            pytest.fail(f"Test with highlight override failed: {e}")

    @pytest.mark.integration
    def test_multiple_simultaneous_queries_performance(self):
        """Test performance with multiple simultaneous queries."""
        import concurrent.futures

        test_queries = [
            "What is artificial intelligence?",
            "Explain deep learning concepts",
            "What are neural networks?",
            "How does machine learning work?",
            "What is natural language processing?"
        ]

        def process_single_query(query):
            start_time = time.time()
            try:
                with patch.object(self.rag_agent.retriever.qdrant_client, 'search_vectors', return_value=[
                    {
                        "id": f"test_id_{hash(query)}",
                        "payload": {
                            "content": f"Content related to {query}",
                            "metadata": {"source_file": "test.md", "section": "Test"}
                        },
                        "score": 0.8
                    }
                ]):
                    with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                        result = self.rag_agent.process_query(query, k=2)
                        return time.time() - start_time, result is not None
            except:
                return time.time() - start_time, False

        # Execute queries concurrently
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_single_query, test_queries))

        total_time = time.time() - start_time

        # Check that all queries completed within reasonable time
        successful_queries = sum(1 for _, success in results if success)
        avg_individual_time = sum(time_taken for time_taken, _ in results) / len(results)

        print(f"Processed {successful_queries}/{len(test_queries)} queries successfully")
        print(f"Total time for {len(test_queries)} concurrent queries: {total_time:.3f}s")
        print(f"Average individual query time: {avg_individual_time:.3f}s ({avg_individual_time*1000:.1f}ms)")

        # Verify performance requirements
        assert successful_queries == len(test_queries), f"Not all queries were successful: {successful_queries}/{len(test_queries)}"
        assert total_time < 10.0, f"All queries took {total_time:.3f}s, should be under 10s for 5 concurrent queries"
        assert avg_individual_time < 2.0, f"Average query time {avg_individual_time:.3f}s exceeds 2s limit"

    @pytest.mark.integration
    def test_confidence_scoring_works_correctly(self):
        """Test that confidence scoring is properly calculated."""
        test_query = "What is supervised learning?"

        try:
            with patch.object(self.rag_agent.retriever.qdrant_client, 'search_vectors', return_value=[
                {
                    "id": "sl_id",
                    "payload": {
                        "content": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs",
                        "metadata": {"source_file": "supervised_learning.md", "section": "Definition"}
                    },
                    "score": 0.92
                },
                {
                    "id": "sl2_id",
                    "payload": {
                        "content": "Common supervised learning algorithms include linear regression and classification",
                        "metadata": {"source_file": "supervised_learning.md", "section": "Algorithms"}
                    },
                    "score": 0.85
                }
            ]):
                with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    result = self.rag_agent.process_query(test_query, k=2)

                    # Verify confidence score is calculated
                    assert "confidence_score" in result
                    assert 0.0 <= result["confidence_score"] <= 1.0

                    print(f"Calculated confidence score: {result['confidence_score']:.3f}")

                    # Confidence should be relatively high with good matching content
                    assert result["confidence_score"] > 0.5, "Confidence score should be reasonably high for relevant content"

        except Exception as e:
            pytest.fail(f"Confidence scoring test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])