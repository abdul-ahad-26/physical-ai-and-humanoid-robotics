from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from ...agent.indexing_agent import IndexingAgent
from ...db.postgres_client import PostgresClient
from ..middleware.auth import APIKeyValidator
from ...api.logging import get_logger
import uuid
import time


router = APIRouter()

# Initialize logger
logger = get_logger()

# Request models
class IndexRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100000, description="Textbook content in Markdown or HTML format")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the content")


class IndexMetadata(BaseModel):
    source_file: str = Field(..., description="Original filename of the content")
    section: Optional[str] = Field(None, description="Section identifier (e.g., chapter, heading)")
    document_type: Optional[str] = Field("markdown", description="Type of document format", pattern="^(markdown|html)$")


# Response models
class IndexResponse(BaseModel):
    status: str  # success, partial, queued
    indexed_chunks: int
    content_id: str
    processing_time: float


async def verify_indexing_api_key(request: Request):
    """
    Dependency to verify the indexing API key.
    """
    APIKeyValidator.validate_api_key(request)


@router.post("/index",
             response_model=IndexResponse,
             dependencies=[Depends(verify_indexing_api_key)],
             summary="Index textbook content",
             description="Accepts textbook content in Markdown/HTML format and indexes it for retrieval by the AI agent. Requires API key authentication.")
async def index_endpoint(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index endpoint that accepts textbook content in Markdown/HTML format
    and indexes it for retrieval by the AI agent.

    Args:
        request (IndexRequest): The index request containing content and metadata
        background_tasks (BackgroundTasks): Background tasks for logging

    Returns:
        IndexResponse: Contains indexing status, number of indexed chunks, content ID, and processing time

    Raises:
        HTTPException: If there are validation errors, authentication failures, or service unavailability
    """
    start_time = time.time()

    try:
        # Extract metadata
        source_file = request.metadata.get("source_file")
        section = request.metadata.get("section")
        document_type = request.metadata.get("document_type", "markdown")

        # Log incoming request
        logger.info(f"Index request received", {
            "source_file": source_file,
            "document_type": document_type,
            "content_length": len(request.content),
            "section": section
        })

        # Validate required metadata
        if not source_file:
            logger.error(f"Missing source file in metadata", {
                "request_id": request_id if 'request_id' in locals() else None
            })
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "details": {"source_file": "Source file is required"}
            })

        # Validate content format
        indexing_agent = IndexingAgent()
        start_time = time.time()
        validation_result = indexing_agent.validate_content_format(request.content, document_type)
        validation_time = time.time() - start_time

        if not validation_result["is_valid"]:
            logger.error(f"Content validation failed", {
                "source_file": source_file,
                "validation_time": round(validation_time, 3),
                "errors": validation_result["errors"]
            })
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "details": validation_result["errors"]
            })

        # Log successful validation
        logger.info(f"Content validation successful", {
            "source_file": source_file,
            "validation_time": round(validation_time, 3)
        })

        # Index the content
        start_time = time.time()
        result = indexing_agent.index_content(
            content=request.content,
            source_file=source_file,
            document_type=document_type,
            section=section
        )
        indexing_time = time.time() - start_time

        # Log agent execution
        logger.log_agent_execution(
            agent_name="IndexingAgent",
            operation="index_content",
            execution_time=indexing_time,
            success=True
        )

        # Ensure the result has the expected format
        if not isinstance(result, dict):
            logger.error(f"Unexpected result format from indexing_agent.index_content", {
                "source_file": source_file,
                "result_type": type(result).__name__
            })
            raise HTTPException(status_code=500, detail={
                "error": "Invalid response format",
                "details": "Indexing agent returned unexpected result format",
                "source_file": source_file
            })

        # Prepare response
        response = IndexResponse(
            status=result["status"],
            indexed_chunks=result["indexed_chunks"],
            content_id=result["content_id"],
            processing_time=result["processing_time"]
        )

        # Log the indexing operation in the background
        def log_indexing():
            try:
                postgres_client = PostgresClient()
                postgres_client.log_agent_execution(
                    agent_id="IndexingAgent",
                    session_id=None,
                    tool_calls=["index_content"],
                    input_params={
                        "source_file": source_file,
                        "document_type": document_type,
                        "section": section
                    },
                    output_result=str(result),
                    execution_time=result["processing_time"]
                )
            except Exception as e:
                print(f"Error logging indexing operation: {e}")

        background_tasks.add_task(log_indexing)

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        logger.error(f"Unexpected error in index endpoint", {
            "source_file": source_file if 'source_file' in locals() else None,
            "error": str(e),
            "error_type": type(e).__name__
        })

        # Check if the error is related to Qdrant being unavailable
        try:
            indexing_agent = IndexingAgent()
            if not indexing_agent.retriever.qdrant_client.health_check():
                logger.warning(f"Qdrant service unavailable for indexing", {
                    "source_file": source_file if 'source_file' in locals() else None
                })
                raise HTTPException(status_code=503, detail={
                    "error": "Service temporarily unavailable",
                    "details": "Vector database is currently down, indexing failed"
                })
        except Exception as health_check_error:
            logger.warning(f"Could not check Qdrant service health during error", {
                "source_file": source_file if 'source_file' in locals() else None,
                "error": str(health_check_error)
            })

        raise HTTPException(status_code=500, detail={
            "error": "Internal server error",
            "details": str(e)
        })


@router.put("/index",
            response_model=IndexResponse,
            dependencies=[Depends(verify_indexing_api_key)],
            summary="Update indexed textbook content",
            description="Updates existing textbook content in the index. Requires API key authentication.")
async def update_index_endpoint(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Update index endpoint that updates existing textbook content.

    Args:
        request (IndexRequest): The update request containing content and metadata
        background_tasks (BackgroundTasks): Background tasks for logging

    Returns:
        IndexResponse: Contains update status, number of indexed chunks, content ID, and processing time

    Raises:
        HTTPException: If there are validation errors, authentication failures, or service unavailability
    """
    start_time = time.time()

    try:
        # Extract metadata
        source_file = request.metadata.get("source_file")
        section = request.metadata.get("section")
        document_type = request.metadata.get("document_type", "markdown")

        # Validate required metadata
        if not source_file:
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "details": {"source_file": "Source file is required"}
            })

        # Validate content format
        indexing_agent = IndexingAgent()
        validation_result = indexing_agent.validate_content_format(request.content, document_type)

        if not validation_result["is_valid"]:
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "details": validation_result["errors"]
            })

        # Update the content
        result = indexing_agent.update_content(
            content=request.content,
            source_file=source_file,
            document_type=document_type,
            section=section
        )

        # Ensure the result has the expected format
        if not isinstance(result, dict):
            print(f"ERROR: Unexpected result format from indexing_agent.update_content for source_file {source_file}")
            raise HTTPException(status_code=500, detail={
                "error": "Invalid response format",
                "details": "Indexing agent returned unexpected result format",
                "source_file": source_file
            })

        # Prepare response
        response = IndexResponse(
            status=result["status"],
            indexed_chunks=result["indexed_chunks"],
            content_id=result["content_id"],
            processing_time=result["processing_time"]
        )

        # Log the update operation in the background
        def log_update():
            try:
                postgres_client = PostgresClient()
                postgres_client.log_agent_execution(
                    agent_id="IndexingAgent",
                    session_id=None,
                    tool_calls=["update_content"],
                    input_params={
                        "source_file": source_file,
                        "document_type": document_type,
                        "section": section
                    },
                    output_result=str(result),
                    execution_time=result["processing_time"]
                )
            except Exception as e:
                print(f"Error logging update operation: {e}")

        background_tasks.add_task(log_update)

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        print(f"Error in update index endpoint: {e}")

        raise HTTPException(status_code=500, detail={
            "error": "Internal server error",
            "details": str(e)
        })


@router.delete("/index/{source_file}",
               dependencies=[Depends(verify_indexing_api_key)],
               summary="Delete indexed textbook content",
               description="Removes content by source file from the index. Requires API key authentication.")
async def delete_index_endpoint(source_file: str):
    """
    Delete index endpoint that removes content by source file.

    Args:
        source_file (str): The source file name to delete from the index

    Returns:
        dict: Status message confirming deletion

    Raises:
        HTTPException: If there are authentication failures or service unavailability
    """
    try:
        indexing_agent = IndexingAgent()
        result = indexing_agent.delete_content(source_file)

        # Ensure the result has the expected format
        if not isinstance(result, dict):
            print(f"ERROR: Unexpected result format from indexing_agent.delete_content for source_file {source_file}")
            raise HTTPException(status_code=500, detail={
                "error": "Invalid response format",
                "details": "Indexing agent returned unexpected result format",
                "source_file": source_file
            })

        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"Successfully deleted content from {source_file}",
                "deleted_source": source_file
            }
        else:
            raise HTTPException(status_code=500, detail={
                "error": "Failed to delete content",
                "details": result.get("error", "Unknown error")
            })

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        print(f"Error in delete index endpoint: {e}")

        raise HTTPException(status_code=500, detail={
            "error": "Internal server error",
            "details": str(e)
        })


@router.get("/index/health",
            summary="Index endpoint health check",
            description="Health check for the index functionality.")
async def index_health():
    """
    Health check for the index functionality.

    Returns:
        dict: Health status of the index service

    Raises:
        HTTPException: If the index service is unavailable
    """
    try:
        indexing_agent = IndexingAgent()
        stats = indexing_agent.get_indexing_stats()

        return {
            "status": "healthy",
            "service": "index-endpoint",
            "indexing_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "error": "Index service unavailable",
            "details": str(e)
        })


def validate_index_request(request: IndexRequest) -> bool:
    """
    Validate the index request parameters.

    Args:
        request: IndexRequest object to validate

    Returns:
        True if valid, raises exception if invalid
    """
    import re

    if not request.content or len(request.content.strip()) == 0:
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"content": "Content is required and cannot be empty"}
        })

    if len(request.content) > 100000:
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"content": "Content exceeds maximum length of 100000 characters"}
        })

    # Additional validation for potentially malicious content
    if re.search(r'<script|javascript:|on\w+\s*=', request.content, re.IGNORECASE):
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"content": "Content contains potentially malicious scripts"}
        })

    if not request.metadata or "source_file" not in request.metadata:
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"metadata": "Metadata with source_file is required"}
        })

    # Validate source file format
    source_file = request.metadata.get("source_file", "")
    if source_file:
        # Check for potentially dangerous file paths
        if ".." in source_file or source_file.startswith("/") or ":/" in source_file:
            raise HTTPException(status_code=422, detail={
                "error": "Validation failed",
                "details": {"source_file": "Invalid file path format"}
            })

    document_type = request.metadata.get("document_type", "markdown")
    if document_type not in ["markdown", "html"]:
        raise HTTPException(status_code=422, detail={
            "error": "Validation failed",
            "details": {"document_type": "Document type must be 'markdown' or 'html'"}
        })

    return True