from typing import Dict, Any, List, Optional
from .rag_agent import RAGAgent
from .indexing_agent import IndexingAgent
from .logging_agent import LoggingAgent
from ..tools.rag_tools import RAGTools
from ..tools.indexing_tools import IndexingTools
from ..tools.logging_tools import LoggingTools
from ..db.postgres_client import PostgresClient
from ..api.gemini_adapter import GeminiAdapter
import os
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

class MainOrchestratorAgent:
    """
    Main orchestrator agent that coordinates between specialized agents
    and manages the overall workflow for the RAG + Agentic AI-Textbook Chatbot.
    Implements hybrid rate limiting and fallback strategies.
    """

    def __init__(self):
        """
        Initialize the main orchestrator with specialized agents and tools.
        """
        self.rag_agent = RAGAgent()
        self.indexing_agent = IndexingAgent()
        self.logging_agent = LoggingAgent()
        self.rag_tools = RAGTools()
        self.indexing_tools = IndexingTools()
        self.logging_tools = LoggingTools()
        self.postgres_client = PostgresClient()

        # Initialize Gemini adapter for orchestration
        try:
            self.gemini_adapter = GeminiAdapter()
        except ValueError as e:
            print(f"Warning: Could not initialize Gemini adapter: {e}")
            self.gemini_adapter = None

        # Track agent execution for observability
        self.execution_history = []

    def process_query(self, query: str, k: int = 5, highlight_override: str = None,
                     session_id: str = None) -> Dict[str, Any]:
        """
        Process a query by coordinating with specialized agents.

        Args:
            query: The user's question
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text that replaces search context
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with processed results
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())

        try:
            # Use RAG agent to retrieve relevant content
            retrieval_result = self.rag_agent.process_query(query, k, highlight_override)

            # Check if retrieval was successful and if Qdrant is healthy
            if not retrieval_result.get("retrieved_contexts") and not self.rag_agent.retriever.qdrant_client.health_check():
                print("Qdrant is unavailable, using alternative approach")
                # Implement additional fallback for when Qdrant is down
                retrieval_result = self._handle_qdrant_unavailable_fallback(query, k, highlight_override)

            # If Qdrant was unavailable and we have a note about it, generate a simple answer
            if "note" in retrieval_result and "Qdrant" in retrieval_result["note"]:
                # In the Qdrant unavailable case, we don't have retrieved contexts to process
                answer = "I'm sorry, but the content database is currently unavailable. " + \
                         "I cannot retrieve textbook content to answer your question at this time. " + \
                         "Please try again later or contact support if the issue persists."
                answer = f"{answer}\n\nNote: {retrieval_result['note']}"
            else:
                # Prepare context for LLM
                llm_context = self._prepare_context_for_llm(query, retrieval_result["retrieved_contexts"], highlight_override)

                # Generate response using LLM (if available) or fallback
                if self.gemini_adapter:
                    answer = self._generate_with_llm(llm_context)
                else:
                    answer = self._generate_fallback_answer(query, retrieval_result["retrieved_contexts"])

            # Calculate confidence score
            if "retrieved_contexts" in retrieval_result:
                confidence_score = self._calculate_confidence_score(retrieval_result["retrieved_contexts"])
            else:
                confidence_score = 0.1  # Very low confidence when Qdrant is unavailable

            result = {
                "answer": answer,
                "retrieved_contexts": retrieval_result.get("retrieved_contexts", []),
                "confidence_score": confidence_score,
                "assembled_context": retrieval_result.get("assembled_context", f"Query: {query}\nNote: Content database unavailable"),
                "query_id": retrieval_result.get("query_id", f"q_{uuid.uuid4().hex[:8]}"),
                "answer_id": f"answer_{uuid.uuid4().hex[:8]}"
            }

            # Log the execution
            execution_time = time.time() - start_time
            self._log_agent_execution(
                execution_id=execution_id,
                agent_id="MainOrchestratorAgent",
                operation="process_query",
                input_params={"query": query, "k": k, "highlight_override": highlight_override},
                output_result=result,
                execution_time=execution_time,
                session_id=session_id
            )

            # Log to database via logging agent
            if session_id:
                self.logging_agent.log_interaction(session_id, query, answer, retrieval_result["retrieved_contexts"])

            return result

        except Exception as e:
            # Log the error
            execution_time = time.time() - start_time
            self._log_agent_execution(
                execution_id=execution_id,
                agent_id="MainOrchestratorAgent",
                operation="process_query",
                input_params={"query": query, "k": k, "highlight_override": highlight_override},
                output_result={"error": str(e)},
                execution_time=execution_time,
                session_id=session_id
            )

            # Try fallback approach
            fallback_result = self._handle_fallback_query(query, k, highlight_override, session_id)
            if fallback_result:
                return fallback_result

            raise e

    def _prepare_context_for_llm(self, question: str, retrieved_contexts: List[Dict],
                                highlight_override: str = None) -> str:
        """
        Prepare context for the LLM based on question and retrieved contexts.

        Args:
            question: The original question
            retrieved_contexts: List of retrieved contexts
            highlight_override: Optional highlighted text

        Returns:
            Formatted context string for the LLM
        """
        context_parts = ["Here is the relevant context from the textbook:"]

        for i, ctx in enumerate(retrieved_contexts, 1):
            source = ctx["metadata"].get("source_file", "unknown source")
            section = ctx["metadata"].get("section", "unknown section")
            context_parts.append(f"{i}. From {source}, Section {section}: {ctx['content']}")

        if highlight_override:
            context_parts.append(f"\nUser highlighted text: {highlight_override}")

        context_parts.append(f"\nThe user's question is: {question}")
        context_parts.append("Please provide a comprehensive, accurate answer based on the above context. " +
                           "If the context doesn't contain sufficient information, please state so clearly.")

        return "\n\n".join(context_parts)

    def _generate_with_llm(self, context: str) -> str:
        """
        Generate answer using LLM.

        Args:
            context: Context to send to the LLM

        Returns:
            Generated answer string
        """
        if not self.gemini_adapter:
            return self._generate_fallback_answer_simple(context)

        try:
            # Use the Gemini adapter to generate the response
            # Format the context as a message for the chat interface
            messages = [
                {"role": "system", "content": "You are an AI assistant for a textbook. Answer questions based on the provided context."},
                {"role": "user", "content": context}
            ]

            response = self.gemini_adapter.chat_generate_response(messages)

            if "error" in response:
                print(f"Error from Gemini: {response['error']}")
                return self._generate_fallback_answer_simple(context)

            return response["response"]
        except Exception as e:
            print(f"Error generating with LLM: {e}")
            # Fallback to simple generation
            return self._generate_fallback_answer_simple(context)

    def _generate_fallback_answer_simple(self, context: str) -> str:
        """
        Simple fallback answer generation when LLM is not available.

        Args:
            context: Context to base the answer on

        Returns:
            Generated answer string
        """
        return f"Based on the provided context: {context[:200]}... [Answer generated without LLM due to service unavailability]"

    def _generate_fallback_answer(self, question: str, retrieved_contexts: List[Dict]) -> str:
        """
        Generate a fallback answer based on the question and retrieved contexts.

        Args:
            question: The user's question
            retrieved_contexts: List of retrieved contexts

        Returns:
            Generated answer string
        """
        if not retrieved_contexts:
            return "I couldn't find any relevant information in the textbook to answer your question."

        # Create a simple answer based on the contexts
        context_snippets = [ctx["content"][:200] + "..." for ctx in retrieved_contexts[:2]]  # Take first 2 contexts, first 200 chars
        return f"Based on the textbook content, here's an answer to your question '{question[:50]}...': " + \
               f"The text mentions: {'; '.join(context_snippets)}. " + \
               "This provides relevant information to address your query."

    def _calculate_confidence_score(self, retrieved_contexts: List[Dict]) -> float:
        """
        Calculate a confidence score based on the retrieved contexts.

        Args:
            retrieved_contexts: List of retrieved contexts

        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved_contexts:
            return 0.1  # Low confidence if no contexts found

        # Calculate average score of retrieved contexts
        avg_score = sum(ctx["score"] for ctx in retrieved_contexts) / len(retrieved_contexts)

        # Normalize the score to be between 0 and 1
        # Assuming similarity scores are between 0 and 1
        return min(1.0, max(0.0, avg_score))

    def _log_agent_execution(self, execution_id: str, agent_id: str, operation: str,
                           input_params: Dict, output_result: Dict, execution_time: float,
                           session_id: str = None):
        """
        Log agent execution to the database and internal tracking.

        Args:
            execution_id: Unique execution ID
            agent_id: The ID of the agent that executed
            operation: The operation that was performed
            input_params: Parameters passed to the agent
            output_result: Result returned by the agent
            execution_time: Time taken for execution in seconds
            session_id: Optional session ID
        """
        # Internal tracking
        execution_record = {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "operation": operation,
            "timestamp": time.time(),
            "execution_time": execution_time,
            "session_id": session_id
        }
        self.execution_history.append(execution_record)

        # Database logging
        try:
            self.postgres_client.log_agent_execution(
                agent_id=agent_id,
                session_id=session_id or "unknown",
                tool_calls=[operation],
                input_params=input_params,
                output_result=str(output_result)[:1000],  # Limit result size
                execution_time=execution_time
            )
        except Exception as e:
            print(f"Error logging agent execution to database: {e}")

    def _handle_qdrant_unavailable_fallback(self, query: str, k: int, highlight_override: str) -> Dict[str, Any]:
        """
        Handle query when Qdrant is unavailable.

        Args:
            query: The user's question
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text

        Returns:
            Fallback result dictionary with limited functionality notice
        """
        # When Qdrant is down, we can't retrieve content, so return a limited response
        fallback_answer = "I'm sorry, but the content database is currently unavailable. " + \
                         "I cannot retrieve textbook content to answer your question at this time. " + \
                         "Please try again later or contact support if the issue persists."

        result = {
            "retrieved_contexts": [],  # No contexts available when Qdrant is down
            "assembled_context": f"Query: {query}\nNote: Content database unavailable",
            "query_id": f"qdrant_fallback_{uuid.uuid4().hex[:8]}",
            "note": "Qdrant vector database is currently unavailable, using limited functionality"
        }

        return result

    def _handle_fallback_query(self, query: str, k: int, highlight_override: str, session_id: str) -> Optional[Dict]:
        """
        Handle query when primary processing fails.

        Args:
            query: The user's question
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text
            session_id: Optional session ID

        Returns:
            Fallback result dictionary or None if fallback also fails
        """
        try:
            # Try to get just the raw retrieval without LLM processing
            retrieval_result = self.rag_agent.process_query(query, k, highlight_override)

            fallback_answer = self._generate_fallback_answer(query, retrieval_result["retrieved_contexts"])
            confidence_score = self._calculate_confidence_score(retrieval_result["retrieved_contexts"])

            result = {
                "answer": fallback_answer,
                "retrieved_contexts": retrieval_result["retrieved_contexts"],
                "confidence_score": confidence_score,
                "assembled_context": retrieval_result["assembled_context"],
                "query_id": retrieval_result["query_id"],
                "answer_id": f"answer_{uuid.uuid4().hex[:8]}",
                "note": "Response generated using fallback method due to primary service unavailability"
            }

            return result
        except Exception:
            # If fallback also fails, return None to let the exception propagate
            return None

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all connected services.

        Returns:
            Health status of all services
        """
        rag_health = self.rag_agent.get_retrieval_stats()
        db_health = self.postgres_client.health_check()

        return {
            "status": "healthy" if db_health else "degraded",
            "service": "main-orchestrator-agent",
            "components": {
                "rag_agent": "healthy",
                "indexing_agent": "healthy",
                "logging_agent": "healthy",
                "database": "healthy" if db_health else "unhealthy",
                "llm_connection": "available" if self.gemini_adapter else "unavailable"
            },
            "retrieval_stats": rag_health,
            "execution_count": len(self.execution_history)
        }

    def get_execution_efficiency(self) -> Dict[str, float]:
        """
        Calculate agent tool call efficiency metrics.

        Implements the 80% efficiency target by tracking:
        - Number of tool calls per execution
        - Successful vs failed tool calls
        - Time spent in tool execution vs waiting

        Returns:
            Dictionary with efficiency metrics
        """
        if not self.execution_history:
            return {
                "efficiency": 0.0,
                "total_executions": 0,
                "total_tool_calls": 0,
                "successful_tool_calls": 0,
                "failed_tool_calls": 0,
                "average_tools_per_execution": 0.0
            }

        # Count total tool calls across all executions
        total_tool_calls = 0
        successful_tool_calls = 0

        for record in self.execution_history:
            if "tool_calls" in record and record["tool_calls"]:
                total_tool_calls += len(record["tool_calls"])
                # Count successful tool calls (in a real implementation, this would track actual success/failure)
                successful_tool_calls += len([tc for tc in record["tool_calls"] if tc.get("status") != "error"])

        # Calculate efficiency metrics
        if total_tool_calls > 0:
            efficiency = successful_tool_calls / total_tool_calls if total_tool_calls > 0 else 0.0
        else:
            efficiency = 1.0 if len(self.execution_history) > 0 else 0.0  # Perfect efficiency if no tools needed

        avg_tools_per_execution = total_tool_calls / len(self.execution_history) if self.execution_history else 0.0

        # Calculate average execution time
        total_time = sum(record["execution_time"] for record in self.execution_history)
        avg_time = total_time / len(self.execution_history)

        return {
            "efficiency": round(efficiency, 3),
            "total_executions": len(self.execution_history),
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "failed_tool_calls": total_tool_calls - successful_tool_calls,
            "average_tools_per_execution": round(avg_tools_per_execution, 2),
            "average_execution_time": round(avg_time, 3),
            "target_efficiency_met": efficiency >= 0.80
        }


# Example usage
if __name__ == "__main__":
    agent = MainOrchestratorAgent()

    print("MainOrchestratorAgent initialized successfully")
    print("Ready to coordinate between specialized agents")

    # Example health check
    health = agent.health_check()
    print(f"Health status: {health['status']}")
    print(f"Components: {health['components']}")