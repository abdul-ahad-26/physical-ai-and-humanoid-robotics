from typing import Dict, List, Any
from ..db.postgres_client import PostgresClient
from datetime import datetime, timedelta
import uuid


class LoggingAgent:
    """
    Specialized agent responsible for session logging with time-based retention policy
    and analytics data collection for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        """
        Initialize the Logging Agent with database client.
        """
        self.postgres_client = PostgresClient()

    def log_interaction(self, session_id: str, query: str, response: str,
                       retrieved_context: List[Dict] = None, user_id: str = None,
                       session_metadata: Dict = None) -> str:
        """
        Log a user interaction to the database.

        Args:
            session_id: The session ID (if existing) or None to create new
            query: The user's query
            response: The AI's response
            retrieved_context: Context used to generate the response
            user_id: Optional user identifier
            session_metadata: Additional metadata about the session

        Returns:
            The session ID (either provided or newly created)
        """
        try:
            # If no session_id provided, create a new session
            if not session_id:
                session_id = self.postgres_client.log_user_session(
                    query=query,
                    response=response,
                    user_id=user_id,
                    retrieved_context=retrieved_context,
                    session_metadata=session_metadata
                )
            else:
                # Update existing session with response
                # For now, we'll create a new session entry, but in a real system
                # you might want to update the existing session
                session_id = self.postgres_client.log_user_session(
                    query=query,
                    response=response,
                    user_id=user_id,
                    retrieved_context=retrieved_context,
                    session_metadata=session_metadata
                )

            return session_id

        except Exception as e:
            print(f"Error logging interaction: {e}")
            # Return a generated session ID if logging fails
            return f"session_{uuid.uuid4().hex[:8]}"

    def log_query_context(self, session_id: str, original_question: str,
                         retrieved_chunks: List[Dict] = None,
                         processed_context: str = None,
                         highlight_override: str = None) -> str:
        """
        Log query context to the database.

        Args:
            session_id: The associated session ID
            original_question: The original question from the user
            retrieved_chunks: List of content chunks retrieved from knowledge base
            processed_context: The final context sent to the AI model
            highlight_override: Optional highlighted text that replaced search context

        Returns:
            The ID of the created query context
        """
        try:
            context_id = self.postgres_client.log_query_context(
                original_question=original_question,
                session_id=session_id,
                highlight_override=highlight_override,
                retrieved_chunks=retrieved_chunks,
                processed_context=processed_context
            )
            return context_id

        except Exception as e:
            print(f"Error logging query context: {e}")
            return f"context_{uuid.uuid4().hex[:8]}"

    def log_agent_execution(self, agent_id: str, session_id: str,
                           tool_calls: List[Dict] = None,
                           input_params: Dict = None,
                           output_result: str = None,
                           execution_time: float = None) -> str:
        """
        Log agent execution to the database.

        Args:
            agent_id: The ID of the agent that executed
            session_id: The associated session ID
            tool_calls: List of tools called during execution
            input_params: Parameters passed to the agent
            output_result: Result returned by the agent
            execution_time: Time taken for execution in seconds

        Returns:
            The ID of the created execution log
        """
        try:
            log_id = self.postgres_client.log_agent_execution(
                agent_id=agent_id,
                session_id=session_id,
                tool_calls=tool_calls,
                input_params=input_params,
                output_result=output_result,
                execution_time=execution_time
            )
            return log_id

        except Exception as e:
            print(f"Error logging agent execution: {e}")
            return f"log_{uuid.uuid4().hex[:8]}"

    def cleanup_old_sessions(self, days: int = 30) -> Dict[str, Any]:
        """
        Clean up old sessions based on retention policy.

        Args:
            days: Number of days to retain sessions (default 30)

        Returns:
            Dictionary with cleanup results
        """
        try:
            deleted_count = self.postgres_client.cleanup_expired_sessions(days)
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "retention_days": days,
                "message": f"Successfully deleted {deleted_count} sessions older than {days} days"
            }
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return {
                "status": "error",
                "deleted_count": 0,
                "error": str(e),
                "message": f"Error cleaning up old sessions: {e}"
            }

    def cleanup_expired_sessions(self, days: int = 30) -> int:
        """
        Clean up expired sessions based on retention policy.

        Args:
            days: Number of days to retain sessions (default 30)

        Returns:
            Number of sessions deleted
        """
        try:
            deleted_count = self.postgres_client.cleanup_expired_sessions(days)
            return deleted_count
        except Exception as e:
            print(f"Error cleaning up expired sessions: {e}")
            return 0

    def get_session_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics for user sessions over the specified number of days.

        Args:
            days: Number of days to look back for analytics

        Returns:
            Dictionary with session analytics
        """
        # This would query the database for analytics
        # For now, we'll return placeholder values
        cutoff_date = datetime.now() - timedelta(days=days)

        return {
            "period_days": days,
            "start_date": cutoff_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "total_sessions": 0,  # Would be retrieved from database
            "total_queries": 0,   # Would be retrieved from database
            "avg_response_time": 0.0,  # Would be calculated from database
            "most_popular_queries": [],  # Would be retrieved from database
            "user_engagement_metrics": {}  # Would be calculated from database
        }

    def get_agent_execution_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics for agent executions over the specified number of days.

        Args:
            days: Number of days to look back for analytics

        Returns:
            Dictionary with agent execution analytics
        """
        # This would query the database for agent execution analytics
        # For now, we'll return placeholder values
        cutoff_date = datetime.now() - timedelta(days=days)

        return {
            "period_days": days,
            "start_date": cutoff_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "total_executions": 0,  # Would be retrieved from database
            "avg_execution_time": 0.0,  # Would be calculated from database
            "most_used_agents": [],  # Would be retrieved from database
            "error_rate": 0.0,  # Would be calculated from database
            "efficiency_metrics": {}  # Would be calculated from database
        }

    def apply_retention_policy(self, retention_days: Dict[str, int] = None):
        """
        Apply data retention policy to clean up old data.

        Args:
            retention_days: Dictionary mapping data types to retention periods in days
                           Example: {"user_sessions": 30, "agent_logs": 90}
        """
        if retention_days is None:
            # Default retention policy
            retention_days = {
                "user_sessions": 30,      # 30 days for user sessions
                "query_contexts": 90,     # 90 days for query contexts
                "agent_executions": 365   # 365 days for agent execution logs
            }

        # Clean up user sessions
        if "user_sessions" in retention_days:
            deleted_sessions = self.cleanup_expired_sessions(retention_days["user_sessions"])
            print(f"Deleted {deleted_sessions} expired user sessions")

        # Additional cleanup for other data types would go here
        # In a real implementation, you'd have specific cleanup methods for each data type

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the logging system.

        Returns:
            Health status of the logging system
        """
        db_health = self.postgres_client.health_check()

        return {
            "status": "healthy" if db_health else "unhealthy",
            "service": "logging-agent",
            "database_connection": "healthy" if db_health else "unhealthy",
            "retention_policy_status": "active"
        }


# Example usage
if __name__ == "__main__":
    agent = LoggingAgent()

    print("LoggingAgent initialized successfully")
    print("Ready to handle session logging and analytics")

    # Example health check
    health = agent.health_check()
    print(f"Health status: {health['status']}")
    print(f"Database connection: {health['database_connection']}")