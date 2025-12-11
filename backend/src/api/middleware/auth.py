from fastapi import Request, HTTPException
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Get the expected API key from environment variables
EXPECTED_API_KEY = os.getenv("INDEXING_API_KEY", "your_indexing_api_key_here")


class APIKeyValidator:
    """
    Middleware class for API key validation in the RAG + Agentic AI-Textbook Chatbot.
    """

    @staticmethod
    def validate_api_key(request: Request, required_scopes: Optional[list] = None) -> bool:
        """
        Validate the API key in the request headers.

        Args:
            request: FastAPI request object
            required_scopes: Optional list of required scopes for the API key

        Returns:
            True if API key is valid, raises HTTPException if invalid
        """
        # Extract API key from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Unauthorized",
                    "details": "Authorization header is required"
                }
            )

        # Handle different authorization header formats
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
        elif auth_header.startswith("ApiKey "):
            api_key = auth_header[7:]  # Remove "ApiKey " prefix
        else:
            # Assume the whole header is the key (for simple API key auth)
            api_key = auth_header

        # Validate the API key
        if not APIKeyValidator._is_valid_key(api_key):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Unauthorized",
                    "details": "Invalid or expired API key"
                }
            )

        # If scopes are required, validate them (simplified implementation)
        if required_scopes:
            if not APIKeyValidator._has_required_scopes(api_key, required_scopes):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Forbidden",
                        "details": "API key does not have required scopes"
                    }
                )

        return True

    @staticmethod
    def _is_valid_key(api_key: str) -> bool:
        """
        Check if the provided API key is valid.

        Args:
            api_key: The API key to validate

        Returns:
            True if valid, False otherwise
        """
        # In a real implementation, this would check against a database or cache of valid keys
        # For now, we just compare against the expected key
        return api_key == EXPECTED_API_KEY

    @staticmethod
    def _has_required_scopes(api_key: str, required_scopes: list) -> bool:
        """
        Check if the API key has the required scopes.

        Args:
            api_key: The API key to check
            required_scopes: List of required scopes

        Returns:
            True if key has required scopes, False otherwise
        """
        # In a real implementation, this would check the scopes associated with the API key
        # For now, we'll return True for all keys (no scope enforcement)
        return True


def get_api_key_from_request(request: Request) -> Optional[str]:
    """
    Extract API key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        API key if found, None otherwise
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    elif auth_header.startswith("ApiKey "):
        return auth_header[7:]  # Remove "ApiKey " prefix
    else:
        return auth_header


def is_authorized_request(request: Request, endpoint: str = None) -> tuple[bool, str]:
    """
    Check if a request is authorized for the given endpoint.

    Args:
        request: FastAPI request object
        endpoint: The endpoint being accessed (for permission checking)

    Returns:
        Tuple of (is_authorized, reason_for_rejection)
    """
    # For public endpoints, no API key is required
    public_endpoints = ["/health", "/query", "/answer"]
    if endpoint and any(endpoint.startswith(pub_ep) for pub_ep in public_endpoints):
        return True, "Public endpoint - no authorization required"

    # For protected endpoints, validate API key
    try:
        APIKeyValidator.validate_api_key(request)
        return True, "Valid API key provided"
    except HTTPException as e:
        return False, e.detail.get("details", "Authorization failed") if isinstance(e.detail, dict) else "Authorization failed"


# Alternative: Decorator approach for route protection
def require_api_key(scopes: Optional[list] = None):
    """
    Decorator to require API key for specific routes.

    Args:
        scopes: Optional list of required scopes
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from function arguments (assumes request is passed)
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                raise HTTPException(
                    status_code=500,
                    detail="Request object not found in function arguments"
                )

            # Validate API key
            APIKeyValidator.validate_api_key(request, scopes)

            # If validation passes, call the original function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Example usage in FastAPI app:
"""
from fastapi import FastAPI, Depends
from .middleware.auth import APIKeyValidator

app = FastAPI()

# Method 1: Manual validation in route
@app.post("/protected-endpoint")
async def protected_route(request: Request):
    APIKeyValidator.validate_api_key(request)
    return {"message": "Access granted"}

# Method 2: Using dependency
async def verify_api_key(request: Request):
    APIKeyValidator.validate_api_key(request)

@app.post("/another-protected-endpoint")
async def another_protected_route(request: Request, _: None = Depends(verify_api_key)):
    return {"message": "Access granted"}
"""