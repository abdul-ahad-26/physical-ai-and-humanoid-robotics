import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.agent.rag_agent import RAGAgent
from src.agent.indexing_agent import IndexingAgent
from src.agent.logging_agent import LoggingAgent
from src.agent.orchestrator import MainOrchestratorAgent
from src.rag.retriever import Retriever
from src.db.postgres_client import PostgresClient


class TestRAGAgent:
    """Unit tests for RAGAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.rag_agent = RAGAgent()

    def test_retrieve_content_success(self):
        """Test successful content retrieval."""
        mock_results = [
            {
                "content": "Test content about machine learning",
                "metadata": {"source_file": "ml_basics.md", "section": "Introduction"},
                "score": 0.92
            }
        ]

        with patch.object(self.rag_agent.retriever, 'retrieve_with_fallback', return_value=mock_results):
            results = self.rag_agent.retrieve_content("What is ML?", k=1)

        assert len(results) == 1
        assert results[0]["content"] == "Test content about machine learning"
        assert results[0]["score"] == 0.92

    def test_retrieve_by_source_success(self):
        """Test successful retrieval by source."""
        mock_results = [
            {
                "content": "Content from specific source",
                "metadata": {"source_file": "specific_source.md"},
                "score": 0.85
            }
        ]

        with patch.object(self.rag_agent.retriever, 'retrieve_by_source', return_value=mock_results):
            results = self.rag_agent.retrieve_by_source("specific_source.md", k=1)

        assert len(results) == 1
        assert results[0]["metadata"]["source_file"] == "specific_source.md"

    def test_add_content_success(self):
        """Test successful content addition."""
        with patch.object(self.rag_agent.retriever, 'add_content', return_value=True):
            result = self.rag_agent.add_content("Test content", {"source_file": "test.md"})

        assert result is True

    def test_process_query_success(self):
        """Test successful query processing."""
        mock_retrieval_result = {
            "retrieved_contexts": [
                {
                    "content": "Machine learning content",
                    "metadata": {"source_file": "ml_basics.md"},
                    "score": 0.88
                }
            ],
            "assembled_context": "Assembled context for ML query",
            "query_id": "test_query_123"
        }

        with patch.object(self.rag_agent.retriever, 'retrieve_with_fallback', return_value=mock_retrieval_result["retrieved_contexts"]):
            with patch.object(self.rag_agent, '_assemble_context', return_value="Assembled context for ML query"):
                with patch.object(self.rag_agent, '_generate_query_id', return_value="test_query_123"):
                    result = self.rag_agent.process_query("What is ML?")

        assert result["query_id"] == "test_query_123"
        assert len(result["retrieved_contexts"]) == 1
        assert "assembled_context" in result

    def test_get_retrieval_stats(self):
        """Test retrieval statistics."""
        mock_stats = {"total_chunks": 100, "collections": ["test_collection"], "embedding_model": "test_model"}

        with patch.object(self.rag_agent.retriever, 'get_retrieval_stats', return_value=mock_stats):
            stats = self.rag_agent.get_retrieval_stats()

        assert stats == mock_stats


class TestIndexingAgent:
    """Unit tests for IndexingAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.indexing_agent = IndexingAgent()

    def test_index_content_success(self):
        """Test successful content indexing."""
        mock_result = {
            "status": "success",
            "indexed_chunks": 2,
            "content_id": "content_abc123",
            "processing_time": 0.5
        }

        with patch.object(self.indexing_agent.retriever, 'batch_add_content', return_value=2):
            with patch.object(self.indexing_agent.chunker, 'chunk_markdown', return_value=[
                type('TextChunk', (), {
                    'content': 'Content chunk 1',
                    'metadata': {'source_file': 'test.md'},
                    'start_pos': 0,
                    'end_pos': 50
                })(),
                type('TextChunk', (), {
                    'content': 'Content chunk 2',
                    'metadata': {'source_file': 'test.md'},
                    'start_pos': 50,
                    'end_pos': 100
                })()
            ]):
                result = self.indexing_agent.index_content(
                    content="# Test Content\nThis is test content.",
                    source_file="test.md",
                    document_type="markdown",
                    section="Introduction"
                )

        assert result["status"] == "success"
        assert result["indexed_chunks"] == 2
        assert "content_id" in result
        assert "processing_time" in result

    def test_update_content_success(self):
        """Test successful content update."""
        mock_delete_result = {
            "status": "success",
            "deleted_source": "test.md",
            "processing_time": 0.2
        }
        mock_index_result = {
            "status": "success",
            "indexed_chunks": 2,
            "content_id": "content_def456",
            "processing_time": 0.5
        }

        with patch.object(self.indexing_agent, 'delete_content', return_value=mock_delete_result):
            with patch.object(self.indexing_agent, 'index_content', return_value=mock_index_result):
                result = self.indexing_agent.update_content(
                    content="Updated content",
                    source_file="test.md",
                    document_type="markdown",
                    section="Introduction"
                )

        assert result["status"] == "success"
        assert result["indexed_chunks"] == 2

    def test_delete_content_success(self):
        """Test successful content deletion."""
        with patch.object(self.indexing_agent.retriever, 'delete_content_by_source', return_value=True):
            result = self.indexing_agent.delete_content("test.md")

        assert result["status"] == "success"
        assert result["deleted_source"] == "test.md"

    def test_validate_content_format_success(self):
        """Test content format validation."""
        result = self.indexing_agent.validate_content_format(
            "Valid content",
            "markdown"
        )

        assert "is_valid" in result
        assert "errors" in result
        assert "warnings" in result

    def test_get_indexing_stats(self):
        """Test indexing statistics."""
        mock_stats = {
            "retriever_stats": {"total_chunks": 100},
            "chunker_info": {"max_chunk_size": 1000}
        }

        with patch.object(self.indexing_agent.retriever, 'get_retrieval_stats', return_value={"total_chunks": 100}):
            with patch.object(self.indexing_agent.chunker, 'max_chunk_size', 1000):
                stats = self.indexing_agent.get_indexing_stats()

        assert "retriever_stats" in stats
        assert "chunker_info" in stats


class TestLoggingAgent:
    """Unit tests for LoggingAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.logging_agent = LoggingAgent()

    def test_log_interaction_success(self):
        """Test successful interaction logging."""
        mock_session_id = "session_123"

        with patch.object(self.logging_agent.postgres_client, 'log_user_session', return_value=mock_session_id):
            result = self.logging_agent.log_interaction(
                session_id=None,
                query="Test question",
                response="Test answer",
                retrieved_context=[{"content": "test context", "score": 0.8}],
                user_id="user_456"
            )

        assert result == mock_session_id

    def test_log_query_context_success(self):
        """Test successful query context logging."""
        mock_context_id = "context_789"

        with patch.object(self.logging_agent.postgres_client, 'log_query_context', return_value=mock_context_id):
            result = self.logging_agent.log_query_context(
                session_id="session_123",
                original_question="Test question",
                retrieved_chunks=[{"content": "test context", "score": 0.8}],
                processed_context="Processed context"
            )

        assert result == mock_context_id

    def test_log_agent_execution_success(self):
        """Test successful agent execution logging."""
        mock_log_id = "log_101"

        with patch.object(self.logging_agent.postgres_client, 'log_agent_execution', return_value=mock_log_id):
            result = self.logging_agent.log_agent_execution(
                agent_id="TestAgent",
                session_id="session_123",
                tool_calls=["test_tool"],
                input_params={"param": "value"},
                output_result="result",
                execution_time=0.25
            )

        assert result == mock_log_id

    def test_cleanup_expired_sessions(self):
        """Test expired session cleanup."""
        with patch.object(self.logging_agent.postgres_client, 'cleanup_expired_sessions', return_value=5):
            result = self.logging_agent.cleanup_expired_sessions(days=30)

        assert result == 5

    def test_get_session_analytics(self):
        """Test session analytics retrieval."""
        result = self.logging_agent.get_session_analytics(days=7)

        assert "period_days" in result
        assert "start_date" in result
        assert "end_date" in result
        assert "total_sessions" in result

    def test_get_agent_execution_analytics(self):
        """Test agent execution analytics retrieval."""
        result = self.logging_agent.get_agent_execution_analytics(days=7)

        assert "period_days" in result
        assert "start_date" in result
        assert "end_date" in result
        assert "total_executions" in result

    def test_apply_retention_policy(self):
        """Test retention policy application."""
        retention_policy = {
            "user_sessions": 30,
            "query_contexts": 90,
            "agent_executions": 365
        }

        # Mock the cleanup methods
        with patch.object(self.logging_agent, 'cleanup_expired_sessions', return_value=3):
            self.logging_agent.apply_retention_policy(retention_policy)

        # Just verify the method runs without errors for now
        assert True

    def test_logging_health_check(self):
        """Test logging system health check."""
        with patch.object(self.logging_agent.postgres_client, 'health_check', return_value=True):
            health = self.logging_agent.health_check()

        assert "status" in health
        assert "service" in health
        assert health["service"] == "logging-agent"


class TestMainOrchestratorAgent:
    """Unit tests for MainOrchestratorAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.orchestrator_agent = MainOrchestratorAgent()

    def test_process_query_success(self):
        """Test successful query processing by orchestrator."""
        mock_rag_result = {
            "retrieved_contexts": [
                {
                    "content": "Test content from RAG",
                    "metadata": {"source_file": "test.md", "section": "Intro"},
                    "score": 0.85
                }
            ],
            "assembled_context": "Assembled context for query",
            "query_id": "query_123"
        }

        mock_answer = "This is a generated answer based on the context."
        mock_confidence = 0.82

        with patch.object(self.orchestrator_agent.rag_agent, 'process_query', return_value=mock_rag_result):
            with patch.object(self.orchestrator_agent, '_generate_answer_from_contexts', return_value=mock_answer):
                with patch.object(self.orchestrator_agent, '_calculate_confidence_score', return_value=mock_confidence):
                    with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                        result = self.orchestrator_agent.process_query("Test question", k=3)

        assert "answer" in result
        assert "retrieved_contexts" in result
        assert "confidence_score" in result
        assert result["answer"] == mock_answer
        assert result["confidence_score"] == mock_confidence

    def test_generate_with_llm_success(self):
        """Test LLM generation."""
        context = "This is the context for the LLM."

        # Mock the gemini adapter
        with patch.object(self.orchestrator_agent, 'gemini_adapter') as mock_adapter:
            mock_adapter.chat_generate_response.return_value = {
                "response": "Generated response from LLM",
                "model": "gemini-pro",
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "response_time": 0.5,
                "timestamp": time.time()
            }

            result = self.orchestrator_agent._generate_with_llm(context)

        assert result == "Generated response from LLM"

    def test_generate_with_llm_fallback(self):
        """Test LLM generation fallback when adapter is not available."""
        self.orchestrator_agent.gemini_adapter = None

        result = self.orchestrator_agent._generate_with_llm("Test context")

        assert "fallback" in result.lower() or "without LLM" in result

    def test_prepare_context_for_llm(self):
        """Test context preparation for LLM."""
        question = "What is machine learning?"
        retrieved_contexts = [
            {
                "content": "Machine learning is a method of data analysis...",
                "metadata": {"source_file": "ml_basics.md", "section": "Intro"},
                "score": 0.9
            }
        ]
        highlight_override = "Important highlight"

        context = self.orchestrator_agent._prepare_context_for_llm(question, retrieved_contexts, highlight_override)

        assert question in context
        assert retrieved_contexts[0]["content"] in context
        assert highlight_override in context
        assert "textbook" in context.lower()

    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        retrieved_contexts = [
            {"score": 0.9},
            {"score": 0.8},
            {"score": 0.7}
        ]

        confidence = self.orchestrator_agent._calculate_confidence_score(retrieved_contexts)

        assert 0.0 <= confidence <= 1.0
        assert abs(confidence - 0.8) < 0.01  # Average of 0.9, 0.8, 0.7 is 0.8

    def test_calculate_confidence_score_empty(self):
        """Test confidence score calculation with empty contexts."""
        confidence = self.orchestrator_agent._calculate_confidence_score([])

        assert confidence == 0.1  # Should return low confidence for empty contexts

    def test_generate_fallback_answer(self):
        """Test fallback answer generation."""
        question = "What is AI?"
        retrieved_contexts = [
            {
                "content": "Artificial Intelligence is...",
                "metadata": {"source_file": "ai_basics.md"},
                "score": 0.85
            }
        ]

        answer = self.orchestrator_agent._generate_fallback_answer(question, retrieved_contexts)

        assert question[:50] in answer or "Artificial Intelligence" in answer
        assert "provide relevant information" in answer

    def test_health_check(self):
        """Test orchestrator health check."""
        with patch.object(self.orchestrator_agent.postgres_client, 'health_check', return_value=True):
            health = self.orchestrator_agent.health_check()

        assert "status" in health
        assert "service" in health
        assert "components" in health
        assert health["service"] == "main-orchestrator-agent"
        assert "rag_agent" in health["components"]
        assert "database" in health["components"]

    def test_get_execution_efficiency(self):
        """Test execution efficiency calculation."""
        efficiency = self.orchestrator_agent.get_execution_efficiency()

        assert "efficiency" in efficiency
        assert "total_executions" in efficiency
        assert "average_execution_time" in efficiency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])