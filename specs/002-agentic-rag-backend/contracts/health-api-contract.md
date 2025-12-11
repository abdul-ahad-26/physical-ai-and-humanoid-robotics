# API Contract: Health Check Endpoint

## Endpoint
`GET /health`

## Purpose
Returns the health status of the FastAPI application and its dependent services (Qdrant vector database and Neon Postgres database). This endpoint is used for monitoring and orchestration tools to determine service availability.

## Request

### Query Parameters
- None

### Headers
- None required

### Example Request
```
GET /health
```

## Response

### Success Response (200 OK)
```json
{
  "status": {
    "type": "string",
    "enum": ["healthy", "degraded", "unhealthy"],
    "description": "Overall health status of the system"
  },
  "timestamp": {
    "type": "string",
    "format": "date-time",
    "description": "UTC timestamp when the health check was performed"
  },
  "services": {
    "type": "object",
    "properties": {
      "fastapi": {
        "type": "string",
        "enum": ["ok", "error"],
        "description": "Health status of the FastAPI application"
      },
      "qdrant": {
        "type": "string",
        "enum": ["ok", "error", "degraded"],
        "description": "Health status of the Qdrant vector database"
      },
      "neon": {
        "type": "string",
        "enum": ["ok", "error", "degraded"],
        "description": "Health status of the Neon Postgres database"
      }
    }
  },
  "details": {
    "type": "object",
    "description": "Additional details about each service's health"
  }
}
```

### Example Success Response
```json
{
  "status": "healthy",
  "timestamp": "2025-12-10T22:30:00Z",
  "services": {
    "fastapi": "ok",
    "qdrant": "ok",
    "neon": "ok"
  },
  "details": {
    "fastapi": {
      "version": "0.1.0",
      "uptime": "2h 15m"
    },
    "qdrant": {
      "version": "1.5.0",
      "collections_count": 3,
      "vectors_count": 15420
    },
    "neon": {
      "version": "PostgreSQL 15.4",
      "active_connections": 4,
      "tables_count": 5
    }
  }
}
```

### Degraded Response (200 OK)
When Qdrant is unavailable but the system can still function with limited capabilities:
```json
{
  "status": "degraded",
  "timestamp": "2025-12-10T22:30:00Z",
  "services": {
    "fastapi": "ok",
    "qdrant": "error",
    "neon": "ok"
  },
  "details": {
    "fastapi": {
      "version": "0.1.0",
      "uptime": "2h 15m"
    },
    "qdrant": {
      "error": "Connection timeout"
    },
    "neon": {
      "version": "PostgreSQL 15.4",
      "active_connections": 4,
      "tables_count": 5
    }
  }
}
```

### Error Responses

#### 503 Service Unavailable
- **Condition**: Core service (FastAPI) is down
- **Response Body**:
```json
{
  "error": "Service unavailable",
  "details": "Core application is not responding"
}
```

## Security
- No authentication required (public endpoint for monitoring)
- No sensitive information exposed in health responses
- Basic status information only

## Performance Requirements
- Response time: < 100ms for 95th percentile
- Should not depend on external services for basic health check
- Should implement timeout for dependent service checks (max 2 seconds each)

## Monitoring
- Monitor health check endpoint for system availability
- Alert on status changes from "healthy" to "degraded" or "unhealthy"
- Track response times for performance monitoring
- Log health check failures for troubleshooting