from __future__ import annotations

from fastapi import APIRouter

from music_api.ai.api import router as ai_router
from music_api.api.v1.health import router as health_router
from music_api.api.v1.root import router as root_router
from music_api.audio.api import router as audio_router
from music_api.jobs.api import router as jobs_router
from music_api.music.api import router as music_router
from music_api.music.theory_api import analysis_theory_router, theory_router
from music_api.storage.api import router as assets_router
from music_api.storage.usage_api import router as storage_router

router = APIRouter()
router.include_router(ai_router)
router.include_router(root_router, tags=["General"])
router.include_router(health_router, prefix="/health", tags=["Health"])
router.include_router(assets_router)
router.include_router(storage_router)
router.include_router(audio_router)
router.include_router(jobs_router)
router.include_router(music_router)
router.include_router(theory_router)
router.include_router(analysis_theory_router)
