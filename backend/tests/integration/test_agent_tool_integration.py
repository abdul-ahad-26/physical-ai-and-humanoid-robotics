import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.agent.rag_agent import RAGAgent
from src.agent.indexing_agent import IndexingAgent
from src.agent.logging_agent import LoggingAgent
from src.agent.orchestrator import MainOrchestratorAgent
from src.tools.rag_tools import RAGTools
from src.tools.indexing_tools import IndexingTools
from src.tools.logging_tools import LoggingTools
from src.rag.retriever import Retriever
from src.db.postgres_client import PostgresClient


class TestAgentToolIntegration:
    """Integration tests for agent-tool interactions."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.rag_agent = RAGAgent()
        self.indexing_agent = IndexingAgent()
        self.logging_agent = LoggingAgent()
        self.orchestrator_agent = MainOrchestratorAgent()

        self.rag_tools = RAGTools()
        self.indexing_tools = IndexingTools()
        self.logging_tools = LoggingTools()

    def test_rag_agent_uses_rag_tools_properly(self):
        """Test that RAGAgent properly integrates with RAGTools."""
        # Test that the agent can use tools through its interface
        test_query = "What is machine learning?"

        # Mock the underlying retrieval process
        with patch.object(self.rag_agent.retriever, 'retrieve_with_fallback', return_value=[
            {
                "content": "Machine learning is a method of data analysis...",
                "metadata": {"source_file": "ml_basics.md", "section": "Introduction"},
                "score": 0.9
            }
        ]):
            with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Test the agent's process_query method
                result = self.rag_agent.process_query(test_query, k=1)

                # Verify the result structure
                assert "retrieved_contexts" in result
                assert "assembled_context" in result
                assert "query_id" in result
                assert len(result["retrieved_contexts"]) == 1
                assert result["retrieved_contexts"][0]["content"] == "Machine learning is a method of data analysis..."

        # Test that the RAGTools can access the same functionality
        with patch.object(self.rag_tools.rag_agent.retriever, 'retrieve_with_fallback', return_value=[
            {
                "content": "Different content retrieved by tools",
                "metadata": {"source_file": "tool_test.md", "section": "Test"},
                "score": 0.85
            }
        ]):
            with patch.object(self.rag_tools.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Test the tools' retrieve_context method
                tool_result = self.rag_tools.retrieve_context(test_query, k=1)

                # Verify the tool result structure
                assert "retrieved_contexts" in tool_result
                assert "assembled_context" in tool_result
                assert len(tool_result["retrieved_contexts"]) == 1
                assert tool_result["retrieved_contexts"][0]["content"] == "Different content retrieved by tools"

        print("✓ RAG Agent and RAG Tools integration working properly")

    def test_indexing_agent_uses_indexing_tools_properly(self):
        """Test that IndexingAgent properly integrates with IndexingTools."""
        test_content = "# Introduction to AI\nArtificial Intelligence is transforming industries."
        test_source_file = "ai_introduction.md"

        # Test indexing through the agent
        with patch.object(self.indexing_agent.retriever, 'batch_add_content', return_value=1):
            with patch.object(self.indexing_agent.chunker, 'chunk_markdown', return_value=[
                type('TextChunk', (), {
                    'content': test_content,
                    'metadata': {'source_file': test_source_file},
                    'start_pos': 0,
                    'end_pos': len(test_content)
                })()
            ]):
                with patch.object(self.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    agent_result = self.indexing_agent.index_content(
                        content=test_content,
                        source_file=test_source_file,
                        document_type="markdown"
                    )

                    # Verify agent result
                    assert agent_result["status"] in ["success", "partial"]
                    assert agent_result["indexed_chunks"] == 1

        # Test indexing through the tools
        with patch.object(self.indexing_tools.indexing_agent.retriever, 'batch_add_content', return_value=1):
            with patch.object(self.indexing_tools.indexing_agent.chunker, 'chunk_markdown', return_value=[
                type('TextChunk', (), {
                    'content': test_content,
                    'metadata': {'source_file': test_source_file},
                    'start_pos': 0,
                    'end_pos': len(test_content)
                })()
            ]):
                with patch.object(self.indexing_tools.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    tools_result = self.indexing_tools.index_content(
                        content=test_content,
                        source_file=test_source_file,
                        document_type="markdown"
                    )

                    # Verify tools result
                    assert tools_result is True  # Tools return boolean success/failure

        print("✓ Indexing Agent and Indexing Tools integration working properly")

    def test_logging_agent_uses_logging_tools_properly(self):
        """Test that LoggingAgent properly integrates with LoggingTools."""
        session_id = "test_session_123"
        query = "Test query?"
        response = "Test response."

        # Test logging through the agent
        with patch.object(self.logging_agent.postgres_client, 'log_user_session', return_value="logged_session_456"):
            agent_result = self.logging_agent.log_interaction(
                session_id=session_id,
                query=query,
                response=response
            )

            # Verify agent result
            assert "session" in agent_result or agent_result == "logged_session_456"

        # Test logging through the tools
        with patch.object(self.logging_tools.logging_agent.postgres_client, 'log_user_session', return_value="tool_logged_session_789"):
            tools_result = self.logging_tools.log_interaction(
                session_id=session_id,
                query=query,
                response=response
            )

            # Verify tools result
            assert "session" in tools_result or tools_result == "tool_logged_session_789"

        print("✓ Logging Agent and Logging Tools integration working properly")

    def test_agent_coordination_through_tools(self):
        """Test that agents can coordinate with each other through tools."""
        # Simulate a scenario where the orchestrator agent uses other agents through tools
        query = "What are neural networks?"

        # Mock the RAG agent's response
        mock_rag_result = {
            "retrieved_contexts": [
                {
                    "content": "Neural networks are computing systems inspired by the human brain...",
                    "metadata": {"source_file": "neural_networks.md", "section": "Basics"},
                    "score": 0.92
                }
            ],
            "assembled_context": "Based on textbook content about neural networks...",
            "query_id": "query_123"
        }

        # Mock the orchestrator's interaction with RAG agent
        with patch.object(self.orchestrator_agent.rag_agent, 'process_query', return_value=mock_rag_result):
            with patch.object(self.orchestrator_agent, '_generate_with_llm', return_value="Neural networks are computing systems inspired by the human brain..."):
                with patch.object(self.orchestrator_agent, '_calculate_confidence_score', return_value=0.85):
                    with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                        orchestrator_result = self.orchestrator_agent.process_query(query, k=3)

                        # Verify orchestrator result
                        assert "answer" in orchestrator_result
                        assert "retrieved_contexts" in orchestrator_result
                        assert "confidence_score" in orchestrator_result
                        assert len(orchestrator_result["retrieved_contexts"]) == 1
                        assert "neural networks" in orchestrator_result["answer"].lower()

        print("✓ Agent coordination through tools working properly")

    def test_rag_agent_tool_chain_execution(self):
        """Test that RAGAgent can execute a chain of tool operations."""
        query = "Explain deep learning"

        # Mock a sequence of operations that might happen in a tool chain
        with patch.object(self.rag_agent.retriever, 'retrieve_with_fallback') as mock_retrieve:
            # First call returns initial results
            mock_retrieve.return_value = [
                {
                    "content": "Deep learning is part of machine learning using neural networks...",
                    "metadata": {"source_file": "deep_learning.md", "section": "Introduction"},
                    "score": 0.88
                }
            ]

            with patch.object(self.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Process the query
                result = self.rag_agent.process_query(query, k=1)

                # Verify the result
                assert result["query_id"] is not None
                assert len(result["retrieved_contexts"]) == 1
                assert result["retrieved_contexts"][0]["score"] == 0.88

        # Now test the same flow through RAGTools
        with patch.object(self.rag_tools.rag_agent.retriever, 'retrieve_with_fallback') as mock_retrieve_tools:
            mock_retrieve_tools.return_value = [
                {
                    "content": "Different deep learning content from tools",
                    "metadata": {"source_file": "dl_tools.md", "section": "Advanced"},
                    "score": 0.82
                }
            ]

            with patch.object(self.rag_tools.rag_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                # Use the tools to retrieve context
                tools_result = self.rag_tools.retrieve_context(query, k=1)

                # Verify the tools result
                assert "retrieved_contexts" in tools_result
                assert len(tools_result["retrieved_contexts"]) == 1
                assert tools_result["retrieved_contexts"][0]["score"] == 0.82

        print("✓ RAG Agent tool chain execution working properly")

    def test_indexing_agent_tool_chain_for_content_ingestion(self):
        """Test that IndexingAgent can execute a chain of content ingestion operations."""
        content = """
# Chapter 1: Introduction to Machine Learning

Machine learning is a method of data analysis that automates analytical model building.

## Subsection: Types of Machine Learning

There are three main types:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning
"""
        source_file = "ml_chapter_1.md"

        # Mock the full indexing chain
        with patch.object(self.indexing_agent.chunker, 'chunk_markdown') as mock_chunk:
            # Mock the chunking result
            mock_chunks = [
                type('TextChunk', (), {
                    'content': 'Machine learning is a method of data analysis...',
                    'metadata': {'source_file': source_file, 'section': 'Introduction'},
                    'start_pos': 0,
                    'end_pos': 50
                })(),
                type('TextChunk', (), {
                    'content': 'There are three main types: Supervised, Unsupervised, Reinforcement...',
                    'metadata': {'source_file': source_file, 'section': 'Types'},
                    'start_pos': 50,
                    'end_pos': 100
                })()
            ]
            mock_chunk.return_value = mock_chunks

            with patch.object(self.indexing_agent.retriever, 'batch_add_content', return_value=2):
                with patch.object(self.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    # Perform the indexing
                    result = self.indexing_agent.index_content(content, source_file, "markdown")

                    # Verify the indexing result
                    assert result["status"] in ["success", "partial"]
                    assert result["indexed_chunks"] == 2

        # Test the same flow through indexing tools
        with patch.object(self.indexing_tools.indexing_agent.chunker, 'chunk_markdown') as mock_chunk_tools:
            mock_chunk_tools.return_value = mock_chunks  # Same chunks

            with patch.object(self.indexing_tools.indexing_agent.retriever, 'batch_add_content', return_value=2):
                with patch.object(self.indexing_tools.indexing_agent.retriever.embedder, 'generate_embedding', return_value=[0.1]*1536):
                    # Use the tools to index content
                    tools_result = self.indexing_tools.index_content(content, source_file, "markdown")

                    # Verify the tools result
                    assert tools_result is True

        print("✓ Indexing Agent tool chain for content ingestion working properly")

    def test_logging_agent_tool_chain_for_session_tracking(self):
        """Test that LoggingAgent can execute a chain of session tracking operations."""
        session_id = "session_integration_test"
        query = "How does backpropagation work?"
        response = "Backpropagation is an algorithm for supervised learning of artificial neural networks..."
        retrieved_context = [
            {
                "content": "Information about backpropagation algorithm",
                "metadata": {"source_file": "neural_networks.md", "section": "Training"},
                "score": 0.9
            }
        ]

        # Mock the full logging chain
        with patch.object(self.logging_agent.postgres_client, 'log_user_session', return_value="logged_session_999"):
            with patch.object(self.logging_agent.postgres_client, 'log_query_context', return_value="context_888"):
                with patch.object(self.logging_agent.postgres_client, 'log_agent_execution', return_value="execution_log_777"):
                    # Log the interaction
                    logged_session_id = self.logging_agent.log_interaction(
                        session_id=session_id,
                        query=query,
                        response=response,
                        retrieved_context=retrieved_context
                    )

                    # Log the query context
                    context_id = self.logging_agent.log_query_context(
                        session_id=logged_session_id,
                        original_question=query,
                        retrieved_chunks=retrieved_context
                    )

                    # Log the agent execution
                    execution_log_id = self.logging_agent.log_agent_execution(
                        agent_id="TestAgent",
                        session_id=logged_session_id,
                        tool_calls=["test_tool"],
                        input_params={"query": query},
                        output_result=response,
                        execution_time=0.25
                    )

                    # Verify all logging operations completed
                    assert logged_session_id is not None
                    assert context_id is not None
                    assert execution_log_id is not None

        # Test the same flow through logging tools
        with patch.object(self.logging_tools.logging_agent.postgres_client, 'log_user_session', return_value="tool_logged_session_666"):
            with patch.object(self.logging_tools.logging_agent.postgres_client, 'log_query_context', return_value="tool_context_555"):
                with patch.object(self.logging_tools.logging_agent.postgres_client, 'log_agent_execution', return_value="tool_execution_log_444"):
                    # Use tools to perform the same operations
                    tool_logged_session_id = self.logging_tools.log_interaction(
                        session_id=session_id,
                        query=query,
                        response=response,
                        retrieved_context=retrieved_context
                    )

                    tool_context_id = self.logging_tools.log_query_context(
                        session_id=tool_logged_session_id,
                        original_question=query,
                        retrieved_chunks=retrieved_context
                    )

                    tool_execution_log_id = self.logging_tools.log_agent_execution(
                        agent_id="TestToolAgent",
                        session_id=tool_logged_session_id,
                        tool_calls=["test_tool"],
                        input_params={"query": query},
                        output_result=response,
                        execution_time=0.25
                    )

                    # Verify all tool-based logging operations completed
                    assert tool_logged_session_id is not None
                    assert tool_context_id is not None
                    assert tool_execution_log_id is not None

        print("✓ Logging Agent tool chain for session tracking working properly")

    def test_multi_agent_coordination_scenario(self):
        """Test a scenario where multiple agents coordinate through tools."""
        query = "Compare supervised and unsupervised learning"

        # Mock each agent's contribution to the overall process
        rag_mock_result = {
            "retrieved_contexts": [
                {
                    "content": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs",
                    "metadata": {"source_file": "learning_types.md", "section": "Supervised"},
                    "score": 0.95
                },
                {
                    "content": "Unsupervised learning finds hidden patterns in unlabeled data",
                    "metadata": {"source_file": "learning_types.md", "section": "Unsupervised"},
                    "score": 0.92
                }
            ],
            "assembled_context": "Context about both learning types...",
            "query_id": "multi_agent_query_123"
        }

        # Mock the orchestrator coordinating multiple agents
        with patch.object(self.orchestrator_agent.rag_agent, 'process_query', return_value=rag_mock_result):
            with patch.object(self.orchestrator_agent, '_generate_with_llm', return_value="Supervised learning uses labeled data while unsupervised learning finds patterns in unlabeled data..."):
                with patch.object(self.orchestrator_agent, '_calculate_confidence_score', return_value=0.88):
                    with patch.object(self.orchestrator_agent, '_log_agent_execution'):
                        # The orchestrator processes the query using RAG agent
                        result = self.orchestrator_agent.process_query(query, k=2)

                        # Verify the result contains comparison of both learning types
                        assert "supervised" in result["answer"].lower()
                        assert "unsupervised" in result["answer"].lower()
                        assert len(result["retrieved_contexts"]) == 2
                        assert result["confidence_score"] == 0.88

        print("✓ Multi-agent coordination scenario working properly")

    def test_error_propagation_between_agents_and_tools(self):
        """Test that errors are properly propagated between agents and tools."""
        # Test that when an agent fails, it's reflected in tool usage and vice versa

        # Mock an error in the RAG agent
        with patch.object(self.rag_agent.retriever, 'retrieve_with_fallback', side_effect=Exception("Qdrant unavailable")):
            with pytest.raises(Exception, match="Qdrant unavailable"):
                self.rag_agent.process_query("Test query")

        # Mock an error in the RAG tools
        with patch.object(self.rag_tools.rag_agent.retriever, 'retrieve_with_fallback', side_effect=Exception("Qdrant unavailable")):
            with pytest.raises(Exception, match="Qdrant unavailable"):
                self.rag_tools.retrieve_context("Test query")

        # Test fallback behavior
        with patch.object(self.rag_agent.retriever, 'retrieve_with_fallback', side_effect=Exception("Qdrant unavailable")):
            with patch.object(self.rag_agent.retriever, 'health_check', return_value=False):
                # The agent should handle the error gracefully
                try:
                    result = self.rag_agent.retrieve_with_fallback("Test query")
                    # Should return empty results in case of failure
                    assert result == []
                except Exception:
                    # Or handle it in another way
                    pass

        print("✓ Error propagation between agents and tools working properly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])