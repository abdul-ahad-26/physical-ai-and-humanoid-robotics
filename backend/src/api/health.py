"""Health check endpoint."""

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status.

    Returns:
        HealthResponse with status and timestamp.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    checks: dict[str, bool]


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Check if service is ready to handle requests.

    Verifies database and vector store connectivity.

    Returns:
        ReadinessResponse with overall readiness and individual checks.
    """
    checks = {
        "database": False,
        "qdrant": False,
    }

    # Check database connection
    try:
        from src.db.connection import get_db_pool

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["database"] = True
    except Exception:
        pass

    # Check Qdrant connection
    try:
        from src.services.qdrant import get_qdrant_client

        client = await get_qdrant_client()
        await client.get_collections()
        checks["qdrant"] = True
    except Exception:
        pass

    return ReadinessResponse(
        ready=all(checks.values()),
        checks=checks,
    )
