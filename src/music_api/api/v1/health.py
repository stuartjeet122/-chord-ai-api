from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("", name="health", summary="Basic process health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/live", name="health_live", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready", name="health_ready", summary="Readiness probe")
async def readiness() -> dict[str, str]:
    return {"status": "ready"}
