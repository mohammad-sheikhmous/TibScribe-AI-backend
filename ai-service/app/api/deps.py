"""Accessors for shared, startup-loaded objects on app.state."""
from __future__ import annotations

from fastapi import Header, HTTPException, Request, status
from typing import TYPE_CHECKING, Any

from ..config import Settings, get_settings
from ..jobs.store import JobStore

if TYPE_CHECKING:
    from ..core.pipeline import MedicalScribePipeline


def get_pipeline(request: Request) -> "MedicalScribePipeline":
    return request.app.state.pipeline


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def get_app_settings() -> Settings:
    return get_settings()


def require_service_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Authenticate the trusted Laravel backend.

    User/doctor Sanctum tokens are deliberately *not* accepted here. Laravel validates
    the doctor and then calls this private service with its own service credential.
    """
    import hmac

    settings = get_settings()
    if not settings.service_auth_required:
        return
    expected = settings.service_token.strip()
    if not expected:
        # Misconfiguration must never silently expose clinical endpoints.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service authentication is not configured.",
        )
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid AI service credential.",
            headers={"WWW-Authenticate": "Bearer"},
        )
