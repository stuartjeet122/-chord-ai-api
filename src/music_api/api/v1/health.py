from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from music_api.api.dependencies import get_resources
from music_api.audio.toolchain import AudioToolchain
from music_api.config import Settings
from music_api.resources import RuntimeResources

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "",
    name="health",
    summary="Basic process health",
    description="Reports that the API process can serve requests.",
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/live",
    name="health_live",
    summary="Liveness probe",
    description="Reports whether the API process is alive; does not depend on external services.",
)
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get(
    "/ready",
    name="health_ready",
    summary="Readiness probe",
    description="Checks connectivity to required external dependencies.",
    responses={503: {"description": "One or more required dependencies are unavailable."}},
)
async def readiness(
    response: Response,
    resources: Annotated[RuntimeResources, Depends(get_resources)],
) -> dict[str, object]:
    checks: dict[str, str] = {}
    try:
        await resources.database.ping()
        checks["database"] = "ok"
    except SQLAlchemyError:
        logger.exception("Database readiness check failed")
        checks["database"] = "unavailable"

    settings: Settings = resources.settings
    checks.update(await AudioToolchain(settings).availability())
    if settings.redis_enabled:
        if resources.redis is None:
            checks["redis"] = "unavailable"
        else:
            try:
                await resources.redis.ping()
                checks["redis"] = "ok"
            except RedisError:
                logger.exception("Redis readiness check failed")
                checks["redis"] = "unavailable"

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}
