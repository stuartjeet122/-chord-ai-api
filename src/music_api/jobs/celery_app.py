from __future__ import annotations

from celery import Celery

from music_api.config import get_settings

settings = get_settings()
celery_app = Celery("music_api", broker=settings.redis_url)
celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    result_backend=None,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=settings.audio_conversion_max_concurrency,
    task_default_queue="audio_cpu",
    task_routes={
        "music_api.jobs.process_audio_job": {"queue": "audio_cpu"},
        "music_api.jobs.dispatch_outbox": {"queue": "maintenance"},
        "music_api.jobs.recover_stale_jobs": {"queue": "maintenance"},
    },
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": max(3600, settings.audio_conversion_timeout_seconds + 600)
    },
    beat_schedule={
        "dispatch-audio-job-outbox": {
            "task": "music_api.jobs.dispatch_outbox",
            "schedule": 30.0,
        },
        "recover-stale-audio-jobs": {
            "task": "music_api.jobs.recover_stale_jobs",
            "schedule": 60.0,
        },
    },
    imports=("music_api.jobs.tasks",),
)
