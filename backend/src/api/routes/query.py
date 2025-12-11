from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from ...agent.rag_agent import RAGAgent
from ...db.postgres_client import PostgresClient
from ...agent.orchestrator import MainOrchestratorAgent
from ...api.logging import get_logger
import uuid
import time
import re


router = APIRouter()

# Initialize logger
logger = get_logger()

# Request models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000, description="The user's question about textbook content")
    highlight_override: Optional[str] = Field(None, min_length=1, max_length=5000, description="Optional highlighted text that replaces search context")

    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')

        # Check for potentially malicious patterns
        if re.search(r'<script|javascript:|on\w+\s*=', v, re.IGNORECASE):
            raise ValueError('Question contains invalid characters or patterns')

        # Remove any HTML tags that might be malicious
        sanitized = re.sub(r'<[^>]*>', '', v)
        return sanitized.strip()

    @validator('highlight_override')
    def validate_highlight_override(cls, v):
        if v is not None:
            # Check for potentially malicious patterns
            if re.search(r'<script|javascript:|on\w+\s*=', v, re.IGNORECASE):
                raise ValueError('Highlight override contains invalid characters or patterns')

            # Remove any HTML tags that might be malicious
            sanitized = re.sub(r'<[^>]*>', '', v)
            return sanitized.strip()

        return v


# Response models
class RetrievedContext(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    retrieved_contexts: List[RetrievedContext]
    assembled_context: str
    query_id: str


@router.post("/query",
             response_model=QueryResponse,
             summary="Query textbook content",
             description="Accepts a user question and optional highlight override, returns retrieved contexts and assembled prompt context without generating a final answer.")
async def query_endpoint(request: QueryRequest):
    """
    Query endpoint that accepts a user question and optional highlight override,
    returns retrieved contexts and assembled prompt context without generating a final answer.

    Args:
        request (QueryRequest): The query request containing the question and optional highlight override

    Returns:
        QueryResponse: Contains retrieved contexts, assembled context, and query ID

    Raises:
        HTTPException: If there are validation errors or service unavailability
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Log incoming request
        logger.info(f"Query request received", {
            "request_id": request_id,
            "question_length": len(request.question),
            "has_highlight_override": request.highlight_override is not None
        })

        # Initialize orchestrator agent
        orchestrator_agent = MainOrchestratorAgent()

        # Process the query
        start_time = time.time()
        result = orchestrator_agent.process_query(
            query=request.question,
            k=5,  # Default to 5 chunks
            highlight_override=request.highlight_override
        )
        processing_time = time.time() - start_time

        # Log agent execution
        logger.log_agent_execution(
            agent_name="MainOrchestratorAgent",
            operation="process_query",
            execution_time=processing_time,
            success=True
        )

        # Ensure the result has the expected format
        if not isinstance(result, dict):
            logger.error(f"Unexpected result format from orchestrator_agent.process_query", {
                "request_id": request_id,
                "result_type": type(result).__name__
            })
            raise HTTPException(status_code=500, detail={
                "error": "Invalid response format",
                "details": "Orchestrator returned unexpected result format",
                "request_id": request_id
            })

        # Validate the response before creating the response object
        if not isinstance(result.get("retrieved_contexts"), list):
            logger.error(f"Invalid context format", {
                "request_id": request_id,
                "contexts_type": type(result.get("retrieved_contexts")).__name__
            })
            raise HTTPException(status_code=500, detail={
                "error": "Invalid response format",
                "details": "Retrieved contexts format is invalid",
                "request_id": request_id
            })

        if not isinstance(result.get("assembled_context"), str):
            logger.error(f"Invalid assembled context format", {
                "request_id": request_id,
                "assembled_context_type": type(result.get("assembled_context")).__name__
            })
            raise HTTPException(status_code=500, detail={
                "error": "Invalid response format",
                "details": "Assembled context format is invalid",
                "request_id": request_id
            })

        # Return the result in the expected format
        response = QueryResponse(
            retrieved_contexts=[
                RetrievedContext(
                    content=ctx["content"],
                    metadata=ctx["metadata"],
                    score=ctx["score"]
                ) for ctx in result["retrieved_contexts"]
            ],
            assembled_context=result["assembled_context"],
            query_id=result["query_id"]
        )

        # Log the query context to the database in the background
        def log_query_context():
            try:
                postgres_client = PostgresClient()
                postgres_client.log_query_context(
                    original_question=request.question,
                    session_id=None,  # We'll create a session ID if needed
                    highlight_override=request.highlight_override,
                    retrieved_chunks=result["retrieved_contexts"],
                    processed_context=result["assembled_context"]
                )
            except Exception as e:
                # Log the error but don't fail the query if logging fails
                print(f"ERROR: Error logging query context for request {request_id}: {e}")

        # Run logging in background to not block the response
        import threading
        log_thread = threading.Thread(target=log_query_context)
        log_thread.start()

        # Performance monitoring
        total_time = time.time() - start_time
        if total_time > 0.3:  # More lenient for query endpoint (only retrieval)
            logger.warning(f"Query endpoint performance threshold exceeded", {
                "request_id": request_id,
                "response_time": round(total_time, 3),
                "threshold": 0.3
            })
        else:
            logger.info(f"Query endpoint completed successfully", {
                "request_id": request_id,
                "response_time": round(total_time, 3)
            })

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error in query endpoint", {
            "request_id": request_id,
            "error": str(e)
        })
        raise HTTPException(status_code=422, detail={
            "error": "Validation error",
            "details": str(e),
            "request_id": request_id
        })
    except Exception as e:
        # Log the error with timing information
        elapsed_time = time.time() - start_time
        logger.error(f"Unexpected error in query endpoint", {
            "request_id": request_id,
            "elapsed_time": round(elapsed_time, 3),
            "error": str(e),
            "error_type": type(e).__name__
        })

        # Check if the error is related to Qdrant or other services being unavailable
        try:
            rag_agent = RAGAgent()
            if not rag_agent.retriever.qdrant_client.health_check():
                logger.warning(f"Qdrant service unavailable", {
                    "request_id": request_id
                })
                raise HTTPException(status_code=503, detail={
                    "error": "Service temporarily unavailable",
                    "details": "Vector database is currently down, using limited functionality",
                    "request_id": request_id
                })
        except Exception as health_check_error:
            logger.warning(f"Could not check service health", {
                "request_id": request_id,
                "error": str(health_check_error)
            })

        # Return a user-friendly error message
        raise HTTPException(status_code=500, detail={
            "error": "Internal server error",
            "details": "An unexpected error occurred while processing your request",
            "request_id": request_id
        })


@router.get("/query/health",
            summary="Query endpoint health check",
            description="Health check for the query functionality.")
async def query_health():
    """
    Health check for the query functionality.

    Returns:
        dict: Health status of the query service

    Raises:
        HTTPException: If the query service is unavailable
    """
    try:
        rag_agent = RAGAgent()
        stats = rag_agent.get_retrieval_stats()

        return {
            "status": "healthy",
            "service": "query-endpoint",
            "retrieval_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "error": "Query service unavailable",
            "details": str(e)
        })


def validate_query_request(request: QueryRequest) -> bool:
    """
    Validate the query request parameters.

    Args:
        request: QueryRequest object to validate

    Returns:
        True if valid, raises exception if invalid
    """
    import re

    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"question": "Question is required and cannot be empty"}
        })

    if len(request.question) > 10000:
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"question": "Question exceeds maximum length of 10000 characters"}
        })

    # Additional validation for potentially malicious content
    if re.search(r'<script|javascript:|on\w+\s*=', request.question, re.IGNORECASE):
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"question": "Question contains potentially malicious content"}
        })

    if request.highlight_override and len(request.highlight_override) > 5000:
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"highlight_override": "Highlight override exceeds maximum length of 5000 characters"}
        })

    # Additional validation for highlight override
    if request.highlight_override and re.search(r'<script|javascript:|on\w+\s*=', request.highlight_override, re.IGNORECASE):
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"highlight_override": "Highlight override contains potentially malicious content"}
        })

    return True