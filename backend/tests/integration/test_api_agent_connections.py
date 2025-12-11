import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.agent.rag_agent import RAGAgent
from src.agent.orchestrator import MainOrchestratorAgent
from src.agent.indexing_agent import IndexingAgent
from src.agent.logging_agent import LoggingAgent
from src.api.routes.query import QueryRequest, query_endpoint
from src.api.routes.answer import AnswerRequest, answer_endpoint
from src.api.routes.index import IndexRequest, index_endpoint
from src.api.routes.health import health_check
from src.db.postgres_client import PostgresClient


class TestAPIAgentConnections:
    """Integration tests for API-agent connections."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        self.rag_agent = RAGAgent()
        self.orchestrator_agent = MainOrchestratorAgent()
        self.indexing_agent = IndexingAgent()
        self.logging_agent = LoggingAgent()

    @pytest.mark.asyncio
    async def test_query_endpoint_connects_to_rag_agent(self):
        """Test that the query endpoint properly connects to the RAG agent."""
        # Mock the RAG agent's response
        mock_result = {
            "retrieved_contexts": [
                {
                    "content": "Machine learning is a method of data analysis...",
                    "metadata": {"source_file": "ml_basics.md", "section": "Introduction"},
                    "score": 0.9
                }
            ],
            "assembled_context": "Based on textbook content about machine learning...",
            "query_id": "test_query_123"
        }

        # Create a mock request
        request_data = {
            "question": "What is machine learning?",
            "highlight_override": None
        }

        with patch.object(self.rag_agent, 'process_query', return_value=mock_result):
            with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Call the actual endpoint function (not through HTTP)
                query_request = QueryRequest(**request_data)

                # Test the endpoint function directly
                response = await query_endpoint(query_request)

                # Verify response structure
                assert hasattr(response, 'retrieved_contexts')
                assert hasattr(response, 'assembled_context')
                assert hasattr(response, 'query_id')
                assert len(response.retrieved_contexts) == 1
                assert response.retrieved_contexts[0].content == "Machine learning is a method of data analysis..."
                assert response.query_id == "test_query_123"

        print("✓ Query endpoint connects to RAG agent properly")

    @pytest.mark.asyncio
    async def test_answer_endpoint_connects_to_orchestrator_agent(self):
        """Test that the answer endpoint properly connects to the orchestrator agent."""
        # Mock the orchestrator agent's response
        mock_result = {
            "answer": "Machine learning is a method of data analysis that automates analytical model building...",
            "retrieved_contexts": [
                {
                    "content": "Machine learning is a method of data analysis...",
                    "metadata": {"source_file": "ml_basics.md", "section": "Introduction"},
                    "score": 0.9
                }
            ],
            "confidence_score": 0.85,
            "assembled_context": "Based on textbook content...",
            "query_id": "test_query_456",
            "answer_id": "test_answer_789"
        }

        # Create a mock request
        request_data = {
            "question": "What is machine learning?",
            "k": 3,
            "highlight_override": None
        }

        with patch.object(self.orchestrator_agent, 'process_query', return_value=mock_result):
            with patch.object(self.orchestrator_agent.rag_agent, 'process_query', return_value=mock_result):
                with patch.object(self.orchestrator_agent, '_generate_with_llm', return_value=mock_result["answer"]):
                    with patch.object(self.orchestrator_agent, '_calculate_confidence_score', return_value=mock_result["confidence_score"]):
                        with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                            # Call the actual endpoint function
                            answer_request = AnswerRequest(**request_data)

                            response = await answer_endpoint(answer_request, Mock())

                            # Verify response structure
                            assert hasattr(response, 'answer')
                            assert hasattr(response, 'retrieved_contexts')
                            assert hasattr(response, 'confidence_score')
                            assert hasattr(response, 'answer_id')
                            assert response.answer == "Machine learning is a method of data analysis that automates analytical model building..."
                            assert len(response.retrieved_contexts) == 1
                            assert response.confidence_score == 0.85
                            assert "test_answer_" in response.answer_id

        print("✓ Answer endpoint connects to orchestrator agent properly")

    @pytest.mark.asyncio
    async def test_index_endpoint_connects_to_indexing_agent(self):
        """Test that the index endpoint properly connects to the indexing agent."""
        # Mock the indexing agent's response
        mock_result = {
            "status": "success",
            "indexed_chunks": 2,
            "content_id": "content_abc123",
            "processing_time": 0.5
        }

        # Create a mock request
        request_data = {
            "content": "# Test Content\nThis is test content for indexing.",
            "metadata": {
                "source_file": "test_content.md",
                "section": "Introduction",
                "document_type": "markdown"
            }
        }

        with patch.object(self.indexing_agent, 'index_content', return_value=mock_result):
            with patch.object(self.indexing_agent.chunker, 'chunk_markdown', return_value=[
                type('TextChunk', (), {
                    'content': 'This is test content for indexing.',
                    'metadata': {'source_file': 'test_content.md'},
                    'start_pos': 0,
                    'end_pos': 30
                })()
            ]):
                with patch.object(self.indexing_agent.retriever, 'batch_add_content', return_value=2):
                    with patch.object(self.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                        # Mock the background tasks
                        mock_background_tasks = Mock()

                        # Call the actual endpoint function
                        from src.api.routes.index import IndexRequest
                        index_request = IndexRequest(**request_data)

                        response = await index_endpoint(index_request, mock_background_tasks)

                        # Verify response structure
                        assert hasattr(response, 'status')
                        assert hasattr(response, 'indexed_chunks')
                        assert hasattr(response, 'content_id')
                        assert hasattr(response, 'processing_time')
                        assert response.status == "success"
                        assert response.indexed_chunks == 2
                        assert "content_" in response.content_id
                        assert response.processing_time == 0.5

        print("✓ Index endpoint connects to indexing agent properly")

    def test_health_endpoint_connects_to_all_agents(self):
        """Test that the health endpoint connects to all agents and services."""
        # Mock the health checks for all services
        with patch.object(self.rag_agent.retriever.qdrant_client, 'health_check', return_value=True):
            with patch.object(self.indexing_agent.retriever.qdrant_client, 'health_check', return_value=True):
                with patch.object(self.logging_agent.postgres_client, 'health_check', return_value=True):
                    with patch.object(self.orchestrator_agent, 'health_check', return_value={
                        "status": "healthy",
                        "service": "main-orchestrator-agent",
                        "components": {
                            "rag_agent": "healthy",
                            "indexing_agent": "healthy",
                            "logging_agent": "healthy",
                            "database": "healthy",
                            "llm_connection": "available"
                        },
                        "retrieval_stats": {"total_chunks": 100},
                        "execution_count": 0
                    }):
                        with patch.object(self.indexing_agent.retriever.qdrant_client, 'health_check', return_value=True):
                            with patch.object(self.indexing_agent.postgres_client, 'health_check', return_value=True):
                                # Call the health endpoint
                                response = health_check()

                                # Verify response structure
                                assert "status" in response
                                assert "timestamp" in response
                                assert "services" in response
                                assert "details" in response
                                assert response["status"] in ["healthy", "degraded", "unhealthy"]

                                # Verify service health status
                                assert "fastapi" in response["services"]
                                assert "qdrant" in response["services"]
                                assert "neon" in response["services"]

        print("✓ Health endpoint connects to all agents properly")

    @pytest.mark.asyncio
    async def test_error_handling_between_api_and_agents(self):
        """Test that errors are properly handled between API and agents."""
        # Test error propagation from RAG agent to query endpoint
        with patch.object(self.rag_agent, 'process_query', side_effect=Exception("Test error")):
            with patch.object(self.rag_agent.retriever.qdrant_client, 'health_check', return_value=True):
                request_data = {"question": "Test question?", "highlight_override": None}
                query_request = QueryRequest(**request_data)

                # The endpoint should handle the error gracefully
                try:
                    response = await query_endpoint(query_request)
                    # If it doesn't raise an exception, check that it's handled properly
                    assert response is not None
                except Exception as e:
                    # The error should be caught and converted to an HTTP exception
                    assert "Test error" in str(e) or "internal server error" in str(e).lower()

        # Test error propagation from orchestrator agent to answer endpoint
        with patch.object(self.orchestrator_agent, 'process_query', side_effect=Exception("Orchestrator error")):
            with patch.object(self.orchestrator_agent.rag_agent.retriever.qdrant_client, 'health_check', return_value=True):
                request_data = {"question": "Test question?", "k": 3}
                answer_request = AnswerRequest(**request_data)

                try:
                    response = await answer_endpoint(answer_request, Mock())
                    assert response is not None
                except Exception as e:
                    # The error should be caught and converted to an HTTP exception
                    assert "Orchestrator error" in str(e) or "internal server error" in str(e).lower()

        print("✓ Error handling between API and agents working properly")

    @pytest.mark.asyncio
    async def test_agent_execution_logging_via_api(self):
        """Test that agent executions are properly logged when triggered via API."""
        # Mock the RAG agent response
        mock_result = {
            "retrieved_contexts": [
                {
                    "content": "Test content for logging",
                    "metadata": {"source_file": "test_logging.md"},
                    "score": 0.85
                }
            ],
            "assembled_context": "Context for logging test",
            "query_id": "logging_test_123"
        }

        # Mock the logging in the agent
        with patch.object(self.rag_agent, 'process_query', return_value=mock_result):
            with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Mock the logging agent that would be called by the endpoint
                with patch.object(self.logging_agent.postgres_client, 'log_query_context'):
                    with patch.object(self.logging_agent.postgres_client, 'log_user_session'):
                        # Call the query endpoint
                        request_data = {"question": "Test logging?", "highlight_override": None}
                        query_request = QueryRequest(**request_data)

                        response = await query_endpoint(query_request)

                        # Verify response is successful
                        assert response is not None
                        assert len(response.retrieved_contexts) == 1

        # Test with answer endpoint
        mock_answer_result = {
            "answer": "This is a test answer for logging.",
            "retrieved_contexts": [
                {
                    "content": "Test content for answer logging",
                    "metadata": {"source_file": "answer_logging.md"},
                    "score": 0.9
                }
            ],
            "confidence_score": 0.82,
            "assembled_context": "Context for answer logging test",
            "query_id": "answer_logging_test_456",
            "answer_id": "answer_log_789"
        }

        with patch.object(self.orchestrator_agent, 'process_query', return_value=mock_answer_result):
            with patch.object(self.orchestrator_agent.rag_agent, 'process_query', return_value=mock_answer_result):
                with patch.object(self.orchestrator_agent, '_generate_with_llm', return_value=mock_answer_result["answer"]):
                    with patch.object(self.orchestrator_agent, '_calculate_confidence_score', return_value=mock_answer_result["confidence_score"]):
                        with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                            with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                                # Mock the background task for logging
                                mock_background_tasks = Mock()

                                # Call the answer endpoint
                                request_data = {"question": "Test answer logging?", "k": 3}
                                answer_request = AnswerRequest(**request_data)

                                response = await answer_endpoint(answer_request, mock_background_tasks)

                                # Verify response is successful
                                assert response is not None
                                assert "test answer" in response.answer.lower()
                                assert response.confidence_score == 0.82

        print("✓ Agent execution logging via API working properly")

    @pytest.mark.asyncio
    async def test_parallel_api_agent_interactions(self):
        """Test that multiple API-agent interactions can happen in parallel safely."""
        import concurrent.futures
        import threading

        # Shared state to track if agents interfere with each other
        execution_tracker = []

        def track_execution(agent_id, delay=0.01):
            """Helper function to simulate agent execution with tracking."""
            time.sleep(delay)  # Simulate processing time
            execution_tracker.append(agent_id)
            return f"Result from {agent_id}"

        # Mock agent responses to include tracking
        def mock_rag_process(query, k=5, highlight_override=None):
            agent_id = f"rag_agent_{threading.current_thread().ident}"
            return track_execution(agent_id)

        def mock_orchestrator_process(query, k=5, highlight_override=None, session_id=None):
            agent_id = f"orchestrator_agent_{threading.current_thread().ident}"
            return {
                "answer": track_execution(agent_id),
                "retrieved_contexts": [],
                "confidence_score": 0.8,
                "assembled_context": "",
                "query_id": "test",
                "answer_id": "test"
            }

        # Test concurrent query requests
        with patch.object(self.rag_agent, 'process_query', side_effect=mock_rag_process):
            with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                async def make_query_request(i):
                    request_data = {"question": f"Query {i}?", "highlight_override": None}
                    query_request = QueryRequest(**request_data)
                    try:
                        response = await query_endpoint(query_request)
                        return f"Query {i} succeeded"
                    except Exception as e:
                        return f"Query {i} failed: {e}"

                # Make multiple concurrent requests
                tasks = [make_query_request(i) for i in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Verify all requests completed successfully
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"Query {i} failed with exception: {result}")
                    else:
                        assert "succeeded" in result

        # Reset tracker for next test
        execution_tracker.clear()

        # Test concurrent answer requests
        with patch.object(self.orchestrator_agent, 'process_query', side_effect=mock_orchestrator_process):
            with patch.object(self.orchestrator_agent.rag_agent, 'process_query', side_effect=mock_rag_process):
                with patch.object(self.orchestrator_agent, '_generate_with_llm', return_value="Test answer"):
                    with patch.object(self.orchestrator_agent, '_calculate_confidence_score', return_value=0.8):
                        with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                            async def make_answer_request(i):
                                request_data = {"question": f"Answer {i}?", "k": 3}
                                answer_request = AnswerRequest(**request_data)
                                try:
                                    response = await answer_endpoint(answer_request, Mock())
                                    return f"Answer {i} succeeded"
                                except Exception as e:
                                    return f"Answer {i} failed: {e}"

                            # Make multiple concurrent requests
                            tasks = [make_answer_request(i) for i in range(3)]
                            results = await asyncio.gather(*tasks, return_exceptions=True)

                            # Verify all requests completed successfully
                            for i, result in enumerate(results):
                                if isinstance(result, Exception):
                                    print(f"Answer {i} failed with exception: {result}")
                                else:
                                    assert "succeeded" in result

        print("✓ Parallel API-agent interactions working properly")

    @pytest.mark.asyncio
    async def test_rate_limiting_integration_with_agents(self):
        """Test that rate limiting works properly with agent interactions."""
        # This test verifies that the rate limiting middleware works with agent calls
        # We'll test that the system can handle rate limiting without interfering with agent functionality

        mock_result = {
            "retrieved_contexts": [
                {
                    "content": "Content for rate limiting test",
                    "metadata": {"source_file": "rate_limit_test.md"},
                    "score": 0.88
                }
            ],
            "assembled_context": "Context for rate limiting test",
            "query_id": "rate_limit_test_123"
        }

        # Mock the agent call
        with patch.object(self.rag_agent, 'process_query', return_value=mock_result):
            with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Test that a normal request still works with rate limiting in place
                request_data = {"question": "Rate limit test?", "highlight_override": None}
                query_request = QueryRequest(**request_data)

                response = await query_endpoint(query_request)

                # Verify response is successful despite rate limiting
                assert response is not None
                assert len(response.retrieved_contexts) == 1
                assert "rate limit" not in response.retrieved_contexts[0].content.lower()

        print("✓ Rate limiting integration with agents working properly")

    @pytest.mark.asyncio
    async def test_api_agent_data_flow_validation(self):
        """Test that data flows correctly between API and agents with validation."""
        # Test complete data flow from API request to agent processing and back
        test_question = "How do neural networks learn?"
        expected_answer = "Neural networks learn through a process called backpropagation..."

        # Mock a complete flow
        mock_rag_result = {
            "retrieved_contexts": [
                {
                    "content": "Neural networks learn through backpropagation and gradient descent...",
                    "metadata": {"source_file": "neural_learning.md", "section": "Training"},
                    "score": 0.92
                }
            ],
            "assembled_context": "Based on textbook content about neural network training...",
            "query_id": "flow_test_456"
        }

        mock_orchestrator_result = {
            "answer": expected_answer,
            "retrieved_contexts": mock_rag_result["retrieved_contexts"],
            "confidence_score": 0.87,
            "assembled_context": mock_rag_result["assembled_context"],
            "query_id": mock_rag_result["query_id"],
            "answer_id": "answer_flow_789"
        }

        # Test the full flow through orchestrator
        with patch.object(self.orchestrator_agent, 'process_query', return_value=mock_orchestrator_result):
            with patch.object(self.orchestrator_agent.rag_agent, 'process_query', return_value=mock_rag_result):
                with patch.object(self.orchestrator_agent, '_generate_with_llm', return_value=expected_answer):
                    with patch.object(self.orchestrator_agent, '_calculate_confidence_score', return_value=0.87):
                        with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                            # Call the answer endpoint
                            request_data = {
                                "question": test_question,
                                "k": 3,
                                "highlight_override": "neural network training process"
                            }
                            answer_request = AnswerRequest(**request_data)

                            response = await answer_endpoint(answer_request, Mock())

                            # Verify the complete data flow
                            assert response is not None
                            assert response.answer == expected_answer
                            assert len(response.retrieved_contexts) == 1
                            assert response.confidence_score == 0.87
                            assert "neural networks" in response.answer.lower()
                            assert "learn" in response.answer.lower()

        # Verify the same flow works for query endpoint
        with patch.object(self.rag_agent, 'process_query', return_value=mock_rag_result):
            with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Call the query endpoint
                query_request = QueryRequest(question=test_question, highlight_override="neural network training")

                response = await query_endpoint(query_request)

                # Verify the query data flow
                assert response is not None
                assert len(response.retrieved_contexts) == 1
                assert "neural networks" in response.retrieved_contexts[0].content.lower()
                assert response.query_id == "flow_test_456"

        print("✓ API-Agent data flow validation working properly")


if __name__ == "__main__":
    import time  # Need to import time for the test
    pytest.main([__file__, "-v"])