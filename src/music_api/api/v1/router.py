from __future__ import annotations

from fastapi import APIRouter

from music_api.api.v1.health import router as health_router
from music_api.api.v1.root import router as root_router
from music_api.music.rhythm_api import router as rhythm_router
from music_api.music.theory_api import theory_router

router = APIRouter()
router.include_router(root_router, tags=["General"])
router.include_router(health_router, prefix="/health", tags=["Health"])
router.include_router(theory_router)
router.include_router(rhythm_router)
