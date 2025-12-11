from typing import Dict, List, Any
from ..agent.logging_agent import LoggingAgent


class LoggingTools:
    """
    Tools for session logging and analytics collection for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        """
        Initialize logging tools with the logging agent.
        """
        self.logging_agent = LoggingAgent()

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
        return self.logging_agent.log_interaction(
            session_id, query, response, retrieved_context, user_id, session_metadata
        )

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
        return self.logging_agent.log_query_context(
            session_id, original_question, retrieved_chunks, processed_context, highlight_override
        )

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
        return self.logging_agent.log_agent_execution(
            agent_id, session_id, tool_calls, input_params, output_result, execution_time
        )

    def cleanup_expired_sessions(self, days: int = 30) -> int:
        """
        Clean up expired sessions based on retention policy.

        Args:
            days: Number of days to retain sessions (default 30)

        Returns:
            Number of sessions deleted
        """
        return self.logging_agent.cleanup_expired_sessions(days)

    def get_session_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics for user sessions over the specified number of days.

        Args:
            days: Number of days to look back for analytics

        Returns:
            Dictionary with session analytics
        """
        return self.logging_agent.get_session_analytics(days)

    def get_agent_execution_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics for agent executions over the specified number of days.

        Args:
            days: Number of days to look back for analytics

        Returns:
            Dictionary with agent execution analytics
        """
        return self.logging_agent.get_agent_execution_analytics(days)

    def apply_retention_policy(self, retention_days: Dict[str, int] = None):
        """
        Apply data retention policy to clean up old data.

        Args:
            retention_days: Dictionary mapping data types to retention periods in days
        """
        self.logging_agent.apply_retention_policy(retention_days)


# Example usage and tool definitions for agent integration
def create_logging_tool_definitions() -> List[Dict]:
    """
    Create tool definitions for integration with OpenAI Agents API.

    Returns:
        List of tool definitions in OpenAI-compatible format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "log_interaction",
                "description": "Log a user interaction to the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The session ID (if existing) or None to create new"
                        },
                        "query": {
                            "type": "string",
                            "description": "The user's query"
                        },
                        "response": {
                            "type": "string",
                            "description": "The AI's response"
                        },
                        "retrieved_context": {
                            "type": "array",
                            "items": {
                                "type": "object"
                            },
                            "description": "Context used to generate the response"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Optional user identifier"
                        },
                        "session_metadata": {
                            "type": "object",
                            "description": "Additional metadata about the session"
                        }
                    },
                    "required": ["query", "response"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "log_query_context",
                "description": "Log query context to the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The associated session ID"
                        },
                        "original_question": {
                            "type": "string",
                            "description": "The original question from the user"
                        },
                        "retrieved_chunks": {
                            "type": "array",
                            "items": {
                                "type": "object"
                            },
                            "description": "List of content chunks retrieved from knowledge base"
                        },
                        "processed_context": {
                            "type": "string",
                            "description": "The final context sent to the AI model"
                        },
                        "highlight_override": {
                            "type": "string",
                            "description": "Optional highlighted text that replaced search context"
                        }
                    },
                    "required": ["session_id", "original_question"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cleanup_expired_sessions",
                "description": "Clean up expired sessions based on retention policy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to retain sessions (default 30)",
                            "default": 30
                        }
                    }
                }
            }
        }
    ]


if __name__ == "__main__":
    tools = LoggingTools()

    # Example of using the tools
    print("Logging Tools initialized successfully")
    print(f"Available functions: log_interaction, log_query_context, cleanup_expired_sessions, etc.")

    # Example tool definitions for agent integration
    tool_defs = create_logging_tool_definitions()
    print(f"Generated {len(tool_defs)} tool definitions for agent integration")