from __future__ import annotations

from fastapi import APIRouter, Request

from music_api.config import Settings

router = APIRouter()


@router.get(
    "/",
    name="api_root",
    summary="API information",
    description="Returns safe service identification and API version information.",
)
async def api_root(request: Request) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    return {
        "application": settings.app_name,
        "api_version": settings.api_version,
        "environment": settings.app_env.value,
    }
