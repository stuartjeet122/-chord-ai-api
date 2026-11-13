from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from music_api.jobs.models import OutboxEvent, ProcessingJob

_SAFE_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class JobNotFoundError(Exception):
    pass


class JobConflictError(Exception):
    pass


async def create_job(
    session: AsyncSession,
    owner_id: uuid.UUID,
    input_asset_id: uuid.UUID,
    task_type: str,
    options: dict[str, Any],
    idempotency_key: str | None,
    *,
    commit: bool = True,
) -> tuple[ProcessingJob, bool]:
    if idempotency_key is not None and not _SAFE_IDEMPOTENCY_KEY.fullmatch(idempotency_key):
        raise ValueError("Idempotency-Key must be 1-128 safe ASCII characters.")
    if idempotency_key:
        existing = await session.scalar(
            select(ProcessingJob).where(
                ProcessingJob.owner_id == owner_id,
                ProcessingJob.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            if (
                existing.input_asset_id != input_asset_id
                or existing.task_type != task_type
                or existing.options != options
            ):
                raise JobConflictError("Idempotency-Key was already used for a different job.")
            existing_id = existing.id
            await session.rollback()
            reloaded = await session.get(ProcessingJob, existing_id)
            if reloaded is None:
                raise JobConflictError("Idempotent job was no longer available.")
            return reloaded, False

    job = ProcessingJob(
        owner_id=owner_id,
        input_asset_id=input_asset_id,
        task_type=task_type,
        status="queued",
        stage="QUEUED",
        progress=0,
        options=options,
        idempotency_key=idempotency_key,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(job)
    await session.flush()
    session.add(
        OutboxEvent(
            job_id=job.id,
            event_type="audio.process",
            payload={"job_id": str(job.id)},
        )
    )
    if commit:
        await session.commit()
    return job, True


async def get_owned_job(
    session: AsyncSession,
    owner_id: uuid.UUID,
    job_id: uuid.UUID,
) -> ProcessingJob:
    job = await session.scalar(
        select(ProcessingJob).where(
            ProcessingJob.id == job_id,
            ProcessingJob.owner_id == owner_id,
        )
    )
    if job is None:
        raise JobNotFoundError("Job was not found.")
    return job


async def cancel_job(
    session: AsyncSession,
    owner_id: uuid.UUID,
    job_id: uuid.UUID,
) -> ProcessingJob:
    job = await get_owned_job(session, owner_id, job_id)
    if job.status in {"succeeded", "failed", "cancelled"}:
        return job
    job.cancel_requested = True
    job.updated_at = datetime.now(UTC)
    if job.status in {"queued", "retry_wait"}:
        job.status = "cancelled"
        job.stage = "CANCELLED"
        job.finished_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(job)
    return job
