from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ...db.qdrant_client import QdrantClientWrapper
from ...db.postgres_client import PostgresClient
from ...agent.orchestrator import MainOrchestratorAgent
import time
from datetime import datetime


router = APIRouter()

# Response models
class HealthServiceStatus(BaseModel):
    fastapi: str
    qdrant: str
    neon: str


class HealthResponse(BaseModel):
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    services: HealthServiceStatus
    details: Optional[Dict[str, Any]] = None


@router.get("/health",
            response_model=HealthResponse,
            summary="Health check",
            description="Returns the status of FastAPI, Qdrant, and Neon database.")
async def health_check():
    """
    Health check endpoint that returns the status of FastAPI, Qdrant, and Neon database.

    Returns:
        HealthResponse: Contains overall status and individual service statuses

    Raises:
        HTTPException: If there are service unavailability issues
    """
    start_time = time.time()

    # Initialize components
    qdrant_client = QdrantClientWrapper()
    postgres_client = PostgresClient()
    orchestrator_agent = MainOrchestratorAgent()

    # Check each service
    fastapi_status = "ok"
    qdrant_status = "ok" if qdrant_client.health_check() else "error"
    neon_status = "ok" if postgres_client.health_check() else "error"

    # Determine overall status
    if qdrant_status == "error" and neon_status == "error":
        overall_status = "unhealthy"
    elif qdrant_status == "error" or neon_status == "error":
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Prepare service status object
    services_status = HealthServiceStatus(
        fastapi=fastapi_status,
        qdrant=qdrant_status if qdrant_status == "ok" else ("error" if qdrant_status == "error" else "degraded"),
        neon=neon_status if neon_status == "ok" else ("error" if neon_status == "error" else "degraded")
    )

    # Prepare detailed information
    details = {
        "fastapi": {
            "version": "0.1.0",
            "uptime": f"{time.time() - start_time:.2f}s"
        }
    }

    # Add Qdrant details if available
    try:
        if qdrant_status == "ok":
            # Get basic Qdrant info (this is a simplified version)
            details["qdrant"] = {
                "status": "connected",
                "response_time": f"{time.time() - start_time:.2f}s"
            }
        else:
            details["qdrant"] = {
                "error": "Connection failed"
            }
    except Exception as e:
        details["qdrant"] = {"error": str(e)}

    # Add Neon details if available
    try:
        if neon_status == "ok":
            details["neon"] = {
                "status": "connected",
                "response_time": f"{time.time() - start_time:.2f}s"
            }
        else:
            details["neon"] = {
                "error": "Connection failed"
            }
    except Exception as e:
        details["neon"] = {"error": str(e)}

    # Add orchestrator details
    try:
        orchestrator_health = orchestrator_agent.health_check()
        details["orchestrator"] = orchestrator_health
    except Exception as e:
        details["orchestrator"] = {"error": str(e)}

    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status,
        details=details
    )

    return response


@router.get("/health/degraded",
            response_model=HealthResponse,
            summary="Degraded health check (test)",
            description="Simulates a degraded state for testing purposes.")
async def degraded_health_check():
    """
    Health check endpoint that simulates a degraded state (for testing purposes).

    Returns:
        HealthResponse: Contains degraded status and service information

    Raises:
        HTTPException: If there are service unavailability issues
    """
    # Simulate a scenario where Qdrant is down but other services are up
    qdrant_client = QdrantClientWrapper()
    postgres_client = PostgresClient()

    fastapi_status = "ok"
    # For this test endpoint, pretend Qdrant is down
    qdrant_status = "error"  # Simulated failure
    neon_status = "ok" if postgres_client.health_check() else "error"

    overall_status = "degraded" if neon_status == "ok" else "unhealthy"

    services_status = HealthServiceStatus(
        fastapi=fastapi_status,
        qdrant="error",  # Simulated failure
        neon=neon_status
    )

    details = {
        "fastapi": {
            "version": "0.1.0",
            "status": "operational"
        },
        "qdrant": {
            "error": "Service temporarily down, using limited functionality"
        },
        "neon": {
            "status": "connected" if neon_status == "ok" else "error"
        }
    }

    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status,
        details=details
    )

    return response


# Readiness endpoint
@router.get("/health/ready",
            summary="Readiness check",
            description="Indicates whether the service is ready to accept traffic. Returns 200 if ready, 503 if not ready.")
async def readiness_check():
    """
    Readiness check endpoint that indicates whether the service is ready to accept traffic.
    Returns 200 if ready, 503 if not ready.

    Returns:
        dict: Contains readiness status and service information

    Raises:
        HTTPException: If services are not ready (returns 503)
    """
    qdrant_client = QdrantClientWrapper()
    postgres_client = PostgresClient()

    # Check if critical services are available
    qdrant_ready = qdrant_client.health_check()
    neon_ready = postgres_client.health_check()

    # For readiness, we require both Qdrant and Neon to be available
    if qdrant_ready and neon_ready:
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    else:
        # Return 503 if not ready
        from fastapi import status
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "qdrant": "ready" if qdrant_ready else "not_ready",
                    "neon": "ready" if neon_ready else "not_ready"
                }
            }
        )


# Additional utility endpoints
@router.get("/health/stats",
            summary="Health statistics",
            description="Get detailed statistics about the health of all services.")
async def health_stats():
    """
    Get detailed statistics about the health of all services.

    Returns:
        dict: Contains detailed health statistics for all services

    Raises:
        HTTPException: If there are service unavailability issues
    """
    qdrant_client = QdrantClientWrapper()
    postgres_client = PostgresClient()
    orchestrator_agent = MainOrchestratorAgent()

    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "qdrant": {
            "health": qdrant_client.health_check(),
            "collection_name": qdrant_client.collection_name,
        },
        "neon": {
            "health": postgres_client.health_check(),
        },
        "orchestrator": orchestrator_agent.health_check(),
        "uptime_seconds": time.time()  # In a real app, track actual uptime
    }

    return stats


# Health check utilities
def check_all_services() -> Dict[str, Any]:
    """
    Utility function to check all services at once.

    Returns:
        Dictionary with health status of all services
    """
    qdrant_client = QdrantClientWrapper()
    postgres_client = PostgresClient()

    return {
        "qdrant": {
            "status": "ok" if qdrant_client.health_check() else "error",
            "health_check_time": time.time()
        },
        "neon": {
            "status": "ok" if postgres_client.health_check() else "error",
            "health_check_time": time.time()
        },
        "fastapi": {
            "status": "ok",  # FastAPI is running if this endpoint is accessible
            "health_check_time": time.time()
        }
    }