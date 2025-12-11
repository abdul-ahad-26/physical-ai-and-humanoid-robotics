from fastapi import FastAPI
from dotenv import load_dotenv
import os
import time
from .middleware.rate_limiter import rate_limit_middleware
from .middleware.security import (
    SecurityHeadersMiddleware,
    RequestSanitizationMiddleware,
    AbuseDetectionMiddleware
)
from .logging import get_logger, set_correlation_id

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger()

app = FastAPI(
    title="RAG + Agentic Backend for AI-Textbook Chatbot",
    description="""
This API provides a textbook-embedded AI assistant using agentic architecture.

## Features

- **Query Endpoint**: Retrieve relevant textbook content without generating answers
- **Answer Endpoint**: Generate natural-language answers to textbook questions
- **Index Endpoint**: Add, update, or delete textbook content for retrieval
- **Health Endpoint**: Monitor service status and readiness

## Authentication

- Query and Answer endpoints: No authentication required (public)
- Index endpoint: Requires API key in Authorization header

## Rate Limiting

All endpoints are subject to rate limiting to prevent abuse.

## Error Handling

All errors follow the format:
```json
{
  "error": "Error type",
  "details": "Error details",
  "request_id": "Unique request identifier"
}
```
    """,
    version="0.1.0",
    contact={
        "name": "AI-Textbook Chatbot Support",
        "url": "https://github.com/abdul-ahad-26/physical-ai-and-humanoid-robotics",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)


# Correlation ID middleware
@app.middleware("http")
async def add_correlation_id(request, call_next):
    correlation_id = set_correlation_id()
    start_time = time.time()

    # Add correlation ID to response headers
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id

    # Log API request
    logger.log_api_request(
        method=request.method,
        endpoint=request.url.path,
        response_time=time.time() - start_time,
        status_code=response.status_code
    )

    return response

# Add security middlewares in order of operation
app.add_middleware(AbuseDetectionMiddleware)
app.add_middleware(RequestSanitizationMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.middleware("http")(rate_limit_middleware)

@app.get("/",
         summary="API Root",
         description="Welcome endpoint for the RAG + Agentic Backend for AI-Textbook Chatbot.")
async def root():
    """
    Root endpoint that provides a welcome message and API information.

    Returns:
        dict: Welcome message and basic API information
    """
    return {
        "message": "Welcome to the RAG + Agentic Backend for AI-Textbook Chatbot",
        "version": "0.1.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "api_status": "operational"
    }

@app.get("/health",
         summary="Basic health check",
         description="Basic health check for the entire API service.")
async def health_check():
    """
    Basic health check endpoint that verifies the API is running.

    Returns:
        dict: Basic health status information
    """
    return {"status": "healthy", "service": "backend-api"}

# Include routes
from .routes import query, answer, index, health
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(answer.router, prefix="/api/v1", tags=["answer"])
app.include_router(index.router, prefix="/api/v1", tags=["index"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)