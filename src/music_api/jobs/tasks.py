from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

from celery import Task, shared_task
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from music_api.ai.runtime import AIInferenceRuntime
from music_api.audio.service import AudioIngestionService
from music_api.config import get_settings
from music_api.db.session import Database
from music_api.jobs.models import OutboxEvent, ProcessingJob
from music_api.storage.providers import StorageError, build_storage_provider
from music_api.storage.service import StorageService

logger = logging.getLogger(__name__)
_MAX_ATTEMPTS = 4
_JOB_HEARTBEAT_TIMEOUT_SECONDS = 300


@shared_task(
    name="music_api.jobs.process_audio_job",
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
)  # type: ignore[misc]
def process_audio_job(task: Task, job_id: str) -> None:
    try:
        asyncio.run(_run_audio_job(uuid.UUID(job_id)))
    except (StorageError, SQLAlchemyError) as exc:
        if task.request.retries >= int(task.max_retries or 0):
            asyncio.run(_mark_job_failed(uuid.UUID(job_id)))
            raise
        raise task.retry(
            exc=exc,
            countdown=min(300, 2 ** (task.request.retries + 1)),
        ) from exc


async def _mark_job_failed(job_id: uuid.UUID) -> None:
    database = Database(get_settings().database_url)
    try:
        async with database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is not None and job.status == "retry_wait":
                job.status = "failed"
                job.stage = "FAILED"
                job.error_code = "temporary_processing_failure"
                job.error_message = "Audio processing failed after repeated temporary errors."
                job.finished_at = datetime.now(UTC)
                job.updated_at = job.finished_at
                await session.commit()
    finally:
        await database.close()


async def _run_audio_job(job_id: uuid.UUID) -> None:
    settings = get_settings()
    database = Database(settings.database_url)
    provider = build_storage_provider(settings)
    storage = StorageService(settings, provider)
    try:
        processor = AudioIngestionService(
            settings,
            database,
            provider,
            storage,
            ai_runtime=AIInferenceRuntime(settings),
        )
        await processor.process_job(job_id)
    finally:
        await provider.close()
        await database.close()


@shared_task(name="music_api.jobs.dispatch_outbox")  # type: ignore[misc]
def dispatch_outbox() -> None:
    asyncio.run(_dispatch_outbox())


async def _dispatch_outbox() -> None:
    from music_api.jobs.celery_app import celery_app

    settings = get_settings()
    database = Database(settings.database_url)
    try:
        async with database.session_factory() as session:
            events = list(
                (
                    await session.scalars(
                        select(OutboxEvent)
                        .where(OutboxEvent.published_at.is_(None))
                        .order_by(OutboxEvent.created_at)
                        .limit(100)
                    )
                ).all()
            )
            for event in events:
                job = await session.get(ProcessingJob, event.job_id)
                if job is None or job.status in {"succeeded", "failed", "cancelled"}:
                    event.published_at = datetime.now(UTC)
                    continue
                try:
                    result = celery_app.send_task(
                        "music_api.jobs.process_audio_job",
                        args=[str(job.id)],
                        task_id=str(event.id),
                        queue=("ai_inference" if job.task_type == "ai_inference" else "audio_cpu"),
                    )
                except Exception:
                    event.attempts += 1
                    logger.exception(
                        "Could not publish audio job from outbox",
                        extra={"job_id": str(job.id), "outbox_event_id": str(event.id)},
                    )
                    continue
                event.published_at = datetime.now(UTC)
                job.celery_task_id = result.id
                job.updated_at = datetime.now(UTC)
            await session.commit()
    finally:
        await database.close()


@shared_task(name="music_api.jobs.recover_stale_jobs")  # type: ignore[misc]
def recover_stale_jobs() -> None:
    asyncio.run(_recover_stale_jobs())


async def _recover_stale_jobs() -> None:
    settings = get_settings()
    database = Database(settings.database_url)
    cutoff = datetime.now(UTC) - timedelta(seconds=_JOB_HEARTBEAT_TIMEOUT_SECONDS)
    try:
        async with database.session_factory() as session:
            stale_jobs = list(
                (
                    await session.scalars(
                        select(ProcessingJob)
                        .where(
                            ProcessingJob.status == "running",
                            ProcessingJob.heartbeat_at < cutoff,
                        )
                        .with_for_update(skip_locked=True)
                        .limit(100)
                    )
                ).all()
            )
            for job in stale_jobs:
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.stage = "CANCELLED"
                    job.finished_at = datetime.now(UTC)
                    continue
                if job.attempt_count >= _MAX_ATTEMPTS:
                    job.status = "failed"
                    job.stage = "FAILED"
                    job.error_code = "worker_lost"
                    job.error_message = "The audio worker stopped before the job completed."
                    job.finished_at = datetime.now(UTC)
                    continue
                job.status = "queued"
                job.stage = "QUEUED"
                job.progress = 0
                job.updated_at = datetime.now(UTC)
                session.add(
                    OutboxEvent(
                        job_id=job.id,
                        event_type="audio.process",
                        payload={"job_id": str(job.id)},
                    )
                )
            await session.commit()
    finally:
        await database.close()
