from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from music_api.api.dependencies import get_db_session, get_owner_id
from music_api.audio.schemas import JobResponse
from music_api.errors import ApplicationError
from music_api.jobs.service import JobNotFoundError, cancel_job, get_owned_job
from music_api.music.models import MusicAnalysisRecord

router = APIRouter(prefix="/jobs", tags=["Jobs"])
OwnerId = Annotated[uuid.UUID, Depends(get_owner_id)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("/{job_id}", response_model=JobResponse, summary="Get owned job status")
async def get_job(
    job_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> JobResponse:
    try:
        return JobResponse.model_validate(await get_owned_job(session, owner_id, job_id))
    except JobNotFoundError as exc:
        raise ApplicationError("job_not_found", str(exc), 404) from exc


@router.post("/{job_id}/cancel", response_model=JobResponse, summary="Request job cancellation")
async def request_job_cancellation(
    job_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> JobResponse:
    try:
        job = await cancel_job(session, owner_id, job_id)
    except JobNotFoundError as exc:
        raise ApplicationError("job_not_found", str(exc), 404) from exc
    if job.task_type == "music_analysis" and job.status == "cancelled":
        analysis = await session.scalar(
            select(MusicAnalysisRecord).where(MusicAnalysisRecord.job_id == job.id)
        )
        if analysis is not None and analysis.status in {"queued", "running"}:
            analysis.status = "cancelled"
            analysis.stage_errors = [{"stage": "analysis", "code": "job_cancelled"}]
            analysis.completed_at = datetime.now(UTC)
            await session.commit()
    return JobResponse.model_validate(job)
