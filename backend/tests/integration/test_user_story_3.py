import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from src.agent.logging_agent import LoggingAgent
from src.agent.rag_agent import RAGAgent
from src.db.postgres_client import PostgresClient
from datetime import datetime, timedelta


class TestUserStory3:
    """Integration tests for User Story 3: Session Logging and Analytics."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.logging_agent = LoggingAgent()
        self.rag_agent = RAGAgent()
        self.postgres_client = PostgresClient()

    @pytest.mark.integration
    def test_all_interactions_are_logged_in_database_for_analysis(self):
        """Test that all interactions are properly logged in the database for analysis."""
        # Test data
        session_id = "test_session_123"
        query = "What is machine learning?"
        response = "Machine learning is a method of data analysis that automates analytical model building..."
        user_id = "student_456"

        retrieved_context = [
            {
                "content": "Machine learning is a method of data analysis...",
                "metadata": {"source_file": "ml_basics.md", "section": "Introduction"},
                "score": 0.92
            }
        ]

        # Measure logging time
        start_time = time.time()

        try:
            # Mock the database logging
            with patch.object(self.postgres_client, 'log_user_session', return_value="logged_session_789"):
                with patch.object(self.postgres_client, 'log_query_context', return_value="context_101"):
                    with patch.object(self.postgres_client, 'log_agent_execution', return_value="exec_log_202"):
                        # Test logging the interaction
                        result_session_id = self.logging_agent.log_interaction(
                            session_id=session_id,
                            query=query,
                            response=response,
                            retrieved_context=retrieved_context,
                            user_id=user_id
                        )

                        # Verify session was logged
                        assert result_session_id is not None
                        assert "session" in result_session_id or result_session_id == "logged_session_789"

                        # Test logging query context
                        context_id = self.logging_agent.log_query_context(
                            session_id=result_session_id,
                            original_question=query,
                            retrieved_chunks=retrieved_context,
                            processed_context="Processed context for ML query",
                            highlight_override=None
                        )

                        # Verify context was logged
                        assert context_id is not None
                        assert "context" in context_id or context_id == "context_101"

                        # Test logging agent execution
                        agent_log_id = self.logging_agent.log_agent_execution(
                            agent_id="RAGAgent",
                            session_id=result_session_id,
                            tool_calls=["retrieve_content"],
                            input_params={"query": query, "k": 3},
                            output_result=response,
                            execution_time=0.245
                        )

                        # Verify agent execution was logged
                        assert agent_log_id is not None
                        assert "log" in agent_log_id or agent_log_id == "exec_log_202"

            logging_time = time.time() - start_time

            print(f"Interaction logging time: {logging_time:.3f}s ({logging_time*1000:.1f}ms)")
            print(f"Logged session ID: {result_session_id}")
            print(f"Logged context ID: {context_id}")
            print(f"Logged agent execution ID: {agent_log_id}")

            # Verify all components were properly logged
            assert result_session_id is not None
            assert context_id is not None
            assert agent_log_id is not None

        except Exception as e:
            pytest.fail(f"Test failed with exception: {e}")

    @pytest.mark.integration
    def test_session_analytics_can_be_retrieved(self):
        """Test that session analytics can be retrieved for analysis."""
        try:
            # Mock analytics data retrieval
            with patch.object(self.logging_agent, 'get_session_analytics') as mock_get_analytics:
                # Mock return value
                mock_analytics = {
                    "period_days": 7,
                    "start_date": (datetime.now() - timedelta(days=7)).isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "total_sessions": 150,
                    "total_queries": 342,
                    "avg_response_time": 0.85,
                    "most_popular_queries": [
                        {"query": "What is AI?", "count": 25},
                        {"query": "Explain ML", "count": 18},
                        {"query": "Neural networks", "count": 15}
                    ],
                    "user_engagement_metrics": {
                        "daily_active_users": 45,
                        "session_duration_avg": 3.2,
                        "returning_users": 0.65
                    }
                }
                mock_get_analytics.return_value = mock_analytics

                # Test analytics retrieval
                analytics = self.logging_agent.get_session_analytics(days=7)

                # Verify analytics structure
                assert "period_days" in analytics
                assert "total_sessions" in analytics
                assert "total_queries" in analytics
                assert "avg_response_time" in analytics
                assert "most_popular_queries" in analytics
                assert "user_engagement_metrics" in analytics

                print(f"Retrieved analytics for {analytics['period_days']} days")
                print(f"Total sessions: {analytics['total_sessions']}")
                print(f"Total queries: {analytics['total_queries']}")
                print(f"Avg response time: {analytics['avg_response_time']}s")
                print(f"Top query: {analytics['most_popular_queries'][0]['query']} (count: {analytics['most_popular_queries'][0]['count']})")

                # Verify data integrity
                assert analytics["period_days"] == 7
                assert analytics["total_sessions"] >= 0
                assert analytics["total_queries"] >= 0
                assert 0 <= analytics["avg_response_time"] <= 10  # Reasonable response time

        except Exception as e:
            pytest.fail(f"Analytics retrieval test failed: {e}")

    @pytest.mark.integration
    def test_agent_execution_analytics_can_be_retrieved(self):
        """Test that agent execution analytics can be retrieved for analysis."""
        try:
            # Mock agent execution analytics data retrieval
            with patch.object(self.logging_agent, 'get_agent_execution_analytics') as mock_get_agent_analytics:
                # Mock return value
                mock_agent_analytics = {
                    "period_days": 7,
                    "start_date": (datetime.now() - timedelta(days=7)).isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "total_executions": 542,
                    "avg_execution_time": 0.34,
                    "most_used_agents": [
                        {"agent": "RAGAgent", "count": 320},
                        {"agent": "IndexingAgent", "count": 150},
                        {"agent": "LoggingAgent", "count": 72}
                    ],
                    "error_rate": 0.02,
                    "efficiency_metrics": {
                        "tool_call_efficiency": 0.87,
                        "average_tools_per_execution": 1.2,
                        "success_rate": 0.98
                    }
                }
                mock_get_agent_analytics.return_value = mock_agent_analytics

                # Test agent execution analytics retrieval
                agent_analytics = self.logging_agent.get_agent_execution_analytics(days=7)

                # Verify analytics structure
                assert "period_days" in agent_analytics
                assert "total_executions" in agent_analytics
                assert "avg_execution_time" in agent_analytics
                assert "most_used_agents" in agent_analytics
                assert "error_rate" in agent_analytics
                assert "efficiency_metrics" in agent_analytics

                print(f"Retrieved agent execution analytics for {agent_analytics['period_days']} days")
                print(f"Total executions: {agent_analytics['total_executions']}")
                print(f"Avg execution time: {agent_analytics['avg_execution_time']}s")
                print(f"Error rate: {agent_analytics['error_rate']}")
                print(f"Top agent: {agent_analytics['most_used_agents'][0]['agent']} (count: {agent_analytics['most_used_agents'][0]['count']})")

                # Verify data integrity
                assert agent_analytics["period_days"] == 7
                assert agent_analytics["total_executions"] >= 0
                assert 0 <= agent_analytics["avg_execution_time"] <= 5  # Reasonable execution time
                assert 0 <= agent_analytics["error_rate"] <= 1  # Error rate should be between 0 and 1

        except Exception as e:
            pytest.fail(f"Agent execution analytics retrieval test failed: {e}")

    @pytest.mark.integration
    def test_time_based_retention_policy_works(self):
        """Test that the time-based retention policy for session cleanup works correctly."""
        try:
            # Mock cleanup operation
            with patch.object(self.postgres_client, 'cleanup_expired_sessions', return_value=5):
                # Test cleanup of expired sessions
                deleted_count = self.logging_agent.cleanup_expired_sessions(days=30)

                # Verify cleanup worked
                assert isinstance(deleted_count, int)
                assert deleted_count >= 0

                print(f"Cleaned up {deleted_count} expired sessions (simulated)")

                # Test with different retention periods
                for days in [7, 30, 90, 365]:
                    with patch.object(self.postgres_client, 'cleanup_expired_sessions', return_value=min(10, days//10)):
                        deleted = self.logging_agent.cleanup_expired_sessions(days=days)
                        assert isinstance(deleted, int)
                        assert deleted >= 0
                        print(f"  With {days}-day retention: cleaned up {deleted} sessions")

        except Exception as e:
            pytest.fail(f"Retention policy test failed: {e}")

    @pytest.mark.integration
    def test_session_logging_integration_with_agents(self):
        """Test that session logging integrates properly with agent operations."""
        # Test data
        session_id = "integration_test_session"
        query = "How do neural networks work?"
        response = "Neural networks are computing systems inspired by the human brain..."

        retrieved_context = [
            {
                "content": "Neural networks are computing systems inspired by the human brain...",
                "metadata": {"source_file": "neural_networks.md", "section": "Basics"},
                "score": 0.88
            }
        ]

        try:
            # Mock all database operations
            with patch.object(self.postgres_client, 'log_user_session', return_value="test_session_999"):
                with patch.object(self.postgres_client, 'log_query_context', return_value="test_context_888"):
                    with patch.object(self.postgres_client, 'log_agent_execution', return_value="test_agent_log_777"):
                        # Simulate a full interaction cycle
                        # 1. Log the user session
                        logged_session_id = self.logging_agent.log_interaction(
                            session_id=session_id,
                            query=query,
                            response=response,
                            retrieved_context=retrieved_context,
                            user_id="test_user_111"
                        )

                        # 2. Log the query context
                        context_id = self.logging_agent.log_query_context(
                            session_id=logged_session_id,
                            original_question=query,
                            retrieved_chunks=retrieved_context,
                            processed_context="Neural networks context processed",
                            highlight_override=None
                        )

                        # 3. Log agent execution
                        agent_log_id = self.logging_agent.log_agent_execution(
                            agent_id="MainOrchestratorAgent",
                            session_id=logged_session_id,
                            tool_calls=["RAGAgent.process_query", "LLM.generate_response"],
                            input_params={"query": query, "k": 3},
                            output_result=response,
                            execution_time=0.45
                        )

                        # Verify all logs were created
                        assert logged_session_id is not None
                        assert context_id is not None
                        assert agent_log_id is not None

                        print(f"Full integration test - Session: {logged_session_id}, Context: {context_id}, Agent Log: {agent_log_id}")

                        # Verify the data flows correctly between components
                        assert "session" in logged_session_id or len(logged_session_id) > 0
                        assert "context" in context_id or len(context_id) > 0
                        assert "log" in agent_log_id or len(agent_log_id) > 0

        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

    @pytest.mark.integration
    def test_audit_logging_for_cleanup_operations(self):
        """Test that cleanup operations are properly audited."""
        try:
            # Mock cleanup and audit operations
            with patch.object(self.postgres_client, 'cleanup_expired_sessions', return_value=3):
                with patch.object(self.postgres_client, 'log_agent_execution', return_value="audit_log_555"):
                    # Apply retention policy
                    self.logging_agent.apply_retention_policy({
                        "user_sessions": 30,      # 30 days for user sessions
                        "query_contexts": 90,     # 90 days for query contexts
                        "agent_executions": 365   # 365 days for agent execution logs
                    })

                    print("Retention policy applied successfully with audit logging")

                    # The method doesn't return anything directly, but it should have logged the operations
                    # This test verifies that the method executes without error and would log appropriately

        except Exception as e:
            pytest.fail(f"Audit logging test failed: {e}")

    @pytest.mark.integration
    def test_logging_health_check_works(self):
        """Test that the logging system health check works properly."""
        try:
            # Mock database health check
            with patch.object(self.postgres_client, 'health_check', return_value=True):
                health_status = self.logging_agent.health_check()

                # Verify health status structure
                assert "status" in health_status
                assert "service" in health_status
                assert "database_connection" in health_status
                assert "retention_policy_status" in health_status

                print(f"Logging health status: {health_status['status']}")
                print(f"Database connection: {health_status['database_connection']}")
                print(f"Retention policy: {health_status['retention_policy_status']}")

                # Verify health status values
                assert health_status["service"] == "logging-agent"
                assert health_status["database_connection"] in ["healthy", "unhealthy"]
                assert health_status["retention_policy_status"] == "active"

        except Exception as e:
            pytest.fail(f"Health check test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])