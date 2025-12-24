"""
Profile API Router

Provides endpoints for user profile management:
- GET /api/profile - Get current user's profile with technical background
- POST /api/profile - Create/update user profile with software/hardware background

These endpoints support the 005-user-personalization feature.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.middleware import AuthenticatedUser, get_current_user
from src.db.models import (
    SoftwareBackground,
    HardwareBackground,
    UserProfile,
    ProfileUpdateRequest,
)
from src.db.queries import get_user_profile, update_user_profile
from src.services.cache import clear_user_cache

router = APIRouter(prefix="/api", tags=["Profile"])


# =============================================================================
# Response Models
# =============================================================================


class ProfileResponse(BaseModel):
    """Response containing user profile data."""

    user: UserProfile


class ProfileUpdateResponse(BaseModel):
    """Response from profile update."""

    success: bool
    user: UserProfile


class ErrorDetail(BaseModel):
    """Structured error response."""

    code: str
    message: str
    field: Optional[str] = None


# =============================================================================
# Profile Endpoints
# =============================================================================


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Get the current user's profile including technical background.

    Returns:
        ProfileResponse with user profile data including:
        - Basic info (id, email, display_name)
        - Software background (level, languages, frameworks)
        - Hardware background (level, domains)
        - profile_completed flag
    """
    profile = await get_user_profile(user.id)

    if not profile:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "PROFILE_001",
                    "message": "User profile not found",
                    "field": None,
                }
            },
        )

    return ProfileResponse(user=profile)


@router.post("/profile", response_model=ProfileUpdateResponse)
async def update_profile(
    request: ProfileUpdateRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Create or update the current user's profile.

    Updates display_name, software_background, and/or hardware_background.
    Automatically sets profile_completed=true when both software and hardware
    background levels are provided.

    When profile is updated, any cached personalized content is invalidated
    since the personalization depends on the user's background.

    Args:
        request: ProfileUpdateRequest with fields to update

    Returns:
        ProfileUpdateResponse with success status and updated profile
    """
    # Validate display_name length
    if request.display_name is not None:
        if len(request.display_name.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Display name cannot be empty",
                        "field": "display_name",
                    }
                },
            )
        if len(request.display_name.strip()) > 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Display name must be 100 characters or less",
                        "field": "display_name",
                    }
                },
            )

    # Validate software background level
    if request.software_background is not None:
        valid_software_levels = ["beginner", "intermediate", "advanced"]
        if request.software_background.level not in valid_software_levels:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": f"Software level must be one of: {', '.join(valid_software_levels)}",
                        "field": "software_background.level",
                    }
                },
            )

    # Validate hardware background level
    if request.hardware_background is not None:
        valid_hardware_levels = ["none", "basic", "intermediate", "advanced"]
        if request.hardware_background.level not in valid_hardware_levels:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": f"Hardware level must be one of: {', '.join(valid_hardware_levels)}",
                        "field": "hardware_background.level",
                    }
                },
            )

        # Validate hardware domains
        valid_domains = [
            "basic_electronics",
            "robotics_kits",
            "gpus_accelerators",
            "jetson_edge",
            "embedded_systems",
            "sensors_actuators",
            "3d_printing",
            "pcb_design",
        ]
        for domain in request.hardware_background.domains:
            if domain not in valid_domains:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": f"Invalid hardware domain: {domain}",
                            "field": "hardware_background.domains",
                        }
                    },
                )

    # Update profile
    updated_profile = await update_user_profile(user.id, request)

    if not updated_profile:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "PROFILE_001",
                    "message": "User profile not found",
                    "field": None,
                }
            },
        )

    # Clear cached personalized content when profile changes
    # (personalization depends on user's background)
    if request.software_background is not None or request.hardware_background is not None:
        clear_user_cache(str(user.id))

    return ProfileUpdateResponse(
        success=True,
        user=updated_profile,
    )
