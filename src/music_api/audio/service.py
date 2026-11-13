from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import uuid
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import SQLAlchemyError

from music_api.ai.benchmarking import TimedChord, TimedNote, chord_metrics, note_metrics
from music_api.ai.engine import InferenceError, InferenceValidationError
from music_api.ai.models import (
    AIInferenceRecord,
    BenchmarkDatasetRecord,
    BenchmarkRunRecord,
)
from music_api.ai.providers import AudioInput
from music_api.ai.runtime import AIInferenceRuntime
from music_api.ai.schemas import (
    BenchmarkChordReference,
    BenchmarkNoteReference,
    ChordPrediction,
    InferenceTask,
    NotePrediction,
)
from music_api.audio.dsp.analysis import inspect_quality
from music_api.audio.dsp.worker import (
    DspOutput,
    DspResult,
    ProcessingCancelledError,
)
from music_api.audio.dsp.worker import (
    process_audio as process_dsp_audio,
)
from music_api.audio.metadata import Artwork, extract_metadata
from music_api.audio.models import (
    AudioArtwork,
    AudioConversion,
    AudioMetadata,
    AudioProcessingOutput,
    AudioProcessingRecord,
    AudioProcessingSource,
)
from music_api.audio.toolchain import AudioToolchain, AudioToolError, ProbeResult
from music_api.config import Settings
from music_api.db.session import Database
from music_api.jobs.models import ProcessingJob
from music_api.music.dsp_analysis import (
    AnalysisCancelledError,
    analyze_audio_file,
)
from music_api.music.models import MusicAnalysisEvent, MusicAnalysisRecord
from music_api.storage.models import AssetType, AudioAsset
from music_api.storage.providers import StorageError, StorageObjectNotFoundError, StorageProvider
from music_api.storage.service import (
    InvalidAssetContentError,
    StorageQuotaExceededError,
    StorageService,
    StoredUpload,
)

logger = logging.getLogger(__name__)
_OUTPUT_MIME_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "aac": "audio/aac",
    "opus": "audio/ogg",
}


def _aggregate_benchmark_metrics(
    samples: list[dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    if not samples:
        return {}
    keys = set(samples[0])
    for sample in samples[1:]:
        keys.intersection_update(sample)
    aggregate: dict[str, float | int | None] = {}
    for key in keys:
        values: list[float | int] = []
        for sample in samples:
            value = sample[key]
            if isinstance(value, (float, int)):
                values.append(value)
        if not values:
            aggregate[key] = None
        elif key.endswith("_count"):
            aggregate[key] = sum(int(value) for value in values)
        else:
            aggregate[key] = sum(float(value) for value in values) / len(values)
    return aggregate


class JobCancelledError(Exception):
    pass


class JobLeaseLostError(Exception):
    pass


class _PathUpload:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("rb")

    async def read(self, size: int) -> bytes:
        return await asyncio.to_thread(self.handle.read, size)

    async def close(self) -> None:
        await asyncio.to_thread(self.handle.close)


class AudioIngestionService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        provider: StorageProvider,
        storage: StorageService,
        ai_runtime: AIInferenceRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.provider = provider
        self.storage = storage
        self.ai_runtime = ai_runtime
        self.tools = AudioToolchain(settings)
        self.temp_root = Path(settings.temp_dir).expanduser().resolve() / "audio-jobs"
        self.temp_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    async def process_job(self, job_id: uuid.UUID) -> None:
        claimed = await self._claim_job(job_id)
        if claimed is None:
            return
        owner_id, asset_id, task_type, options = claimed
        output_asset_id: uuid.UUID | None = None
        heartbeat = asyncio.create_task(self._heartbeat_loop(job_id))
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"job-{job_id}-",
                dir=self.temp_root,
            ) as job_directory:
                work_dir = Path(job_directory)
                source_path = work_dir / "source.audio"
                source_asset = await self._get_source_asset(asset_id, owner_id)
                await self._download_source(source_asset, source_path)
                await self._check_cancelled(job_id)
                await self._update_progress(job_id, "VALIDATING", 5)
                probe = await self.tools.inspect(source_path)

                async def report_validation(elapsed_seconds: float) -> bool:
                    fraction = min(1.0, max(0.0, elapsed_seconds / probe.duration_seconds))
                    return await self._update_progress(
                        job_id,
                        "VALIDATING",
                        min(19, 5 + int(fraction * 14)),
                    )

                try:
                    await self.tools.verify_decodable(source_path, report_validation)
                except AudioToolError as exc:
                    if exc.code == "job_cancelled":
                        await self._check_cancelled(job_id)
                    raise
                ffmpeg_version, ffprobe_version = await self.tools.versions()
                await self._check_cancelled(job_id)
                if task_type == "ai_inference":
                    await self._process_ai_inference_job(
                        job_id,
                        source_asset,
                        source_path,
                        probe,
                        options,
                    )
                    await self._complete_job(job_id, None)
                    return
                if task_type == "ai_benchmark":
                    await self._process_ai_benchmark_job(
                        job_id,
                        source_asset,
                        source_path,
                        probe,
                        options,
                        work_dir,
                    )
                    await self._complete_job(job_id, None)
                    return
                if task_type == "music_analysis":
                    await self._process_music_analysis_job(
                        job_id,
                        source_asset,
                        source_path,
                    )
                    await self._complete_job(job_id, None)
                    return
                if task_type.startswith("dsp_"):
                    output_ids = await self._process_dsp_job(
                        job_id,
                        source_asset,
                        source_path,
                        probe,
                        task_type.removeprefix("dsp_"),
                        options,
                        work_dir,
                    )
                    await self._complete_job(job_id, output_ids[0] if output_ids else None)
                    return
                await self._update_progress(job_id, "EXTRACTING_METADATA", 20)
                extracted = await asyncio.to_thread(extract_metadata, source_path, probe)
                await self._save_metadata(
                    source_asset,
                    extracted,
                    probe,
                    ffprobe_version,
                    job_id,
                    work_dir,
                )
                await self._check_cancelled(job_id)

                conversion_duration_ms: int | None = None
                if task_type == "convert":
                    await self._update_progress(job_id, "CONVERTING", 40)
                    output_path = work_dir / f"converted.{options['output_format']}"

                    async def report_progress(elapsed_seconds: float) -> bool:
                        fraction = min(1.0, max(0.0, elapsed_seconds / probe.duration_seconds))
                        return await self._update_progress(
                            job_id,
                            "CONVERTING",
                            min(89, 40 + int(fraction * 49)),
                        )

                    try:
                        conversion_version, conversion_duration_ms = await self.tools.convert(
                            source_path,
                            output_path,
                            options,
                            probe,
                            report_progress,
                        )
                    except AudioToolError as exc:
                        if exc.code == "job_cancelled":
                            await self._check_cancelled(job_id)
                        raise
                    await self._update_progress(job_id, "VERIFYING_OUTPUT", 90)
                    output_probe = await self.tools.inspect(output_path)

                    async def report_output_verification(elapsed_seconds: float) -> bool:
                        fraction = min(
                            1.0, max(0.0, elapsed_seconds / output_probe.duration_seconds)
                        )
                        return await self._update_progress(
                            job_id,
                            "VERIFYING_OUTPUT",
                            min(96, 90 + int(fraction * 6)),
                        )

                    try:
                        await self.tools.verify_decodable(
                            output_path,
                            report_output_verification,
                        )
                    except AudioToolError as exc:
                        if exc.code == "job_cancelled":
                            await self._check_cancelled(job_id)
                        raise
                    _verify_output(probe, output_probe, options)
                    if output_path.stat().st_size > self.settings.max_audio_output_size_bytes:
                        raise AudioToolError(
                            "audio_output_too_large",
                            "Converted output exceeds the configured maximum size.",
                        )
                    await self._check_cancelled(job_id)
                    await self._update_progress(job_id, "STORING", 97)
                    output_asset_id = await self._store_converted_asset(
                        output_path,
                        owner_id,
                        source_asset.id,
                        str(options["output_format"]),
                    )
                    await self._store_output_probe(output_asset_id, output_probe)
                    await self._save_conversion(
                        job_id,
                        source_asset.id,
                        output_asset_id,
                        options,
                        conversion_duration_ms,
                        conversion_version,
                    )
                await self._complete_job(job_id, output_asset_id)
        except JobCancelledError:
            if output_asset_id is not None:
                await self._delete_uncommitted_output(output_asset_id)
            if task_type == "music_analysis":
                await self._mark_music_analysis_terminal(
                    job_id,
                    "cancelled",
                    "job_cancelled",
                    "Analysis was cancelled.",
                )
            await self._finish_job(job_id, "cancelled", "CANCELLED", "job_cancelled", None)
        except JobLeaseLostError:
            logger.info(
                "Worker lease expired while processing audio job", extra={"job_id": str(job_id)}
            )
        except AudioToolError as exc:
            if exc.code == "job_cancelled":
                if output_asset_id is not None:
                    await self._delete_uncommitted_output(output_asset_id)
                await self._finish_job(job_id, "cancelled", "CANCELLED", exc.code, None)
            else:
                if task_type == "music_analysis":
                    await self._mark_music_analysis_terminal(
                        job_id,
                        "failed",
                        exc.code,
                        str(exc),
                    )
                if task_type != "music_analysis":
                    await self._save_failed_conversion(job_id, asset_id, options, exc.code)
                await self._finish_job(job_id, "failed", "FAILED", exc.code, str(exc))
        except InferenceError as exc:
            await self._finish_job(job_id, "failed", "FAILED", exc.code, str(exc))
        except StorageQuotaExceededError as exc:
            await self._finish_job(
                job_id,
                "failed",
                "FAILED",
                "storage_quota_exceeded",
                str(exc),
            )
        except (StorageError, SQLAlchemyError):
            await self._finish_job(
                job_id,
                "retry_wait",
                "RETRY_WAIT",
                "temporary_processing_failure",
                "A temporary storage or database error occurred.",
            )
            raise
        except InvalidAssetContentError as exc:
            await self._finish_job(job_id, "failed", "FAILED", "invalid_audio", str(exc))
        except Exception:
            logger.exception("Audio processing job failed", extra={"job_id": str(job_id)})
            if task_type == "music_analysis":
                await self._mark_music_analysis_terminal(
                    job_id,
                    "failed",
                    "analysis_failed",
                    "Music analysis failed.",
                )
            await self._finish_job(
                job_id,
                "failed",
                "FAILED",
                "audio_processing_failed",
                "Audio processing failed. Retry the job or contact support.",
            )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat_loop(self, job_id: uuid.UUID) -> None:
        while True:
            await asyncio.sleep(30)
            async with self.database.session_factory() as session:
                job = await session.get(ProcessingJob, job_id)
                if job is None or job.status != "running":
                    return
                job.heartbeat_at = datetime.now(UTC)
                await session.commit()

    async def _claim_job(
        self,
        job_id: uuid.UUID,
    ) -> tuple[uuid.UUID, uuid.UUID, str, dict[str, Any]] | None:
        async with self.database.session_factory() as session:
            now = datetime.now(UTC)
            result = await session.execute(
                update(ProcessingJob)
                .where(
                    ProcessingJob.id == job_id,
                    ProcessingJob.status.in_(("queued", "retry_wait")),
                    ProcessingJob.cancel_requested.is_(False),
                )
                .values(
                    status="running",
                    stage="VALIDATING",
                    progress=0,
                    attempt_count=ProcessingJob.attempt_count + 1,
                    started_at=now,
                    heartbeat_at=now,
                    updated_at=now,
                )
                .returning(
                    ProcessingJob.owner_id,
                    ProcessingJob.input_asset_id,
                    ProcessingJob.task_type,
                    ProcessingJob.options,
                )
            )
            claim = result.one_or_none()
            await session.commit()
            if claim is None:
                return None
            return claim[0], claim[1], claim[2], dict(claim[3])

    async def _get_source_asset(
        self,
        asset_id: uuid.UUID,
        owner_id: uuid.UUID,
    ) -> AudioAsset:
        async with self.database.session_factory() as session:
            asset = await session.scalar(
                select(AudioAsset).where(
                    AudioAsset.id == asset_id,
                    AudioAsset.owner_id == owner_id,
                    AudioAsset.status == "available",
                )
            )
            if asset is None:
                raise AudioToolError(
                    "asset_not_available", "The source audio asset is unavailable."
                )
            return asset

    async def _download_source(self, asset: AudioAsset, destination: Path) -> None:
        digest = hashlib.sha256()
        size = 0
        try:
            with destination.open("wb") as target:
                async for chunk in self.provider.iter_bytes(asset.storage_key):
                    size += len(chunk)
                    if size > self.settings.max_upload_size_bytes:
                        raise AudioToolError(
                            "audio_input_too_large", "Audio input exceeds size limits."
                        )
                    digest.update(chunk)
                    await asyncio.to_thread(target.write, chunk)
        except StorageObjectNotFoundError as exc:
            raise AudioToolError("asset_missing", "The source audio object is missing.") from exc
        if size != asset.size_bytes or digest.hexdigest() != asset.sha256:
            raise AudioToolError(
                "asset_integrity_failed", "The source audio object failed integrity checks."
            )

    async def _save_metadata(
        self,
        source_asset: AudioAsset,
        extracted: Any,
        probe: ProbeResult,
        ffprobe_version: str,
        job_id: uuid.UUID,
        work_dir: Path,
    ) -> None:
        await self._update_progress(job_id, "EXTRACTING_ARTWORK", 30)
        async with self.database.session_factory() as session:
            metadata = await session.get(AudioMetadata, source_asset.id)
            if metadata is None:
                metadata = AudioMetadata(
                    audio_asset_id=source_asset.id,
                    tags=extracted.tags,
                    raw_tags=extracted.raw_tags,
                    technical=extracted.technical,
                    mutagen_version=extracted.mutagen_version,
                    ffprobe_version=ffprobe_version,
                )
                session.add(metadata)
            else:
                metadata.tags = extracted.tags
                metadata.raw_tags = extracted.raw_tags
                metadata.technical = extracted.technical
                metadata.mutagen_version = extracted.mutagen_version
                metadata.ffprobe_version = ffprobe_version
                metadata.extracted_at = datetime.now(UTC)
            asset = await session.get(AudioAsset, source_asset.id)
            if asset is not None:
                asset.audio_format = probe.codec
                asset.duration_seconds = probe.duration_seconds
                asset.sample_rate = probe.sample_rate
                asset.channel_count = probe.channels
                asset.bit_depth = probe.bit_depth
            await session.commit()

        for index, artwork in enumerate(extracted.artworks):
            await self._store_artwork(source_asset.id, artwork, index, work_dir)

    async def _store_artwork(
        self,
        audio_asset_id: uuid.UUID,
        artwork: Artwork,
        index: int,
        work_dir: Path,
    ) -> None:
        digest = hashlib.sha256(artwork.data).hexdigest()
        async with self.database.session_factory() as session:
            linked = list(
                (
                    await session.scalars(
                        select(AudioArtwork).where(AudioArtwork.audio_asset_id == audio_asset_id)
                    )
                ).all()
            )
            for record in linked:
                asset = await session.get(AudioAsset, record.artwork_asset_id)
                if asset is not None and asset.sha256 == digest and asset.status != "deleted":
                    return

        extension = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
        }[artwork.mime_type]
        artwork_path = work_dir / f"artwork-{index}.{extension}"
        await asyncio.to_thread(artwork_path.write_bytes, artwork.data)
        upload = _PathUpload(artwork_path)
        try:
            staged = await self.storage.stage_upload(
                upload,
                filename=f"embedded-artwork-{index}.{extension}",
                asset_type=AssetType.EMBEDDED_ARTWORK,
            )
        finally:
            await upload.close()
        async with self.database.session_factory() as session:
            image_asset, _created = await self.storage.create_asset(
                session,
                (await self._get_owner_for_asset(audio_asset_id)),
                staged,
                parent_asset_id=audio_asset_id,
                idempotency_key=f"audio-art:{audio_asset_id}:{digest}",
            )
            linked_artwork = await session.scalar(
                select(AudioArtwork.id).where(
                    AudioArtwork.audio_asset_id == audio_asset_id,
                    AudioArtwork.artwork_asset_id == image_asset.id,
                )
            )
            if linked_artwork is None:
                session.add(
                    AudioArtwork(
                        audio_asset_id=audio_asset_id,
                        artwork_asset_id=image_asset.id,
                        picture_type=artwork.picture_type,
                        description=artwork.description,
                        mime_type=artwork.mime_type,
                    )
                )
                await session.commit()

    async def _get_owner_for_asset(self, asset_id: uuid.UUID) -> uuid.UUID:
        async with self.database.session_factory() as session:
            owner_id = await session.scalar(
                select(AudioAsset.owner_id).where(AudioAsset.id == asset_id)
            )
            if owner_id is None:
                raise AudioToolError("asset_missing", "Source asset is not available.")
            return owner_id

    async def _process_ai_benchmark_job(
        self,
        job_id: uuid.UUID,
        first_asset: AudioAsset,
        first_path: Path,
        first_probe: ProbeResult,
        options: dict[str, Any],
        work_dir: Path,
    ) -> None:
        if self.ai_runtime is None:
            raise InferenceError(
                "inference_runtime_unavailable",
                "The AI inference runtime is not configured for this worker.",
            )
        try:
            dataset_id = uuid.UUID(str(options["dataset_record_id"]))
            task = InferenceTask(str(options["task"]))
            model_ids = [str(model_id) for model_id in options["model_ids"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise InferenceError(
                "invalid_benchmark_request", "The benchmark job options are invalid."
            ) from exc
        if task not in {InferenceTask.CHORD_RECOGNITION, InferenceTask.NOTE_TRANSCRIPTION}:
            raise InferenceError(
                "unsupported_benchmark_task",
                "Only chord recognition and note transcription are benchmarked.",
            )

        async with self.database.session_factory() as session:
            dataset = await session.scalar(
                select(BenchmarkDatasetRecord).where(
                    BenchmarkDatasetRecord.id == dataset_id,
                    BenchmarkDatasetRecord.owner_id == first_asset.owner_id,
                )
            )
            run = await session.scalar(
                select(BenchmarkRunRecord)
                .where(BenchmarkRunRecord.job_id == job_id)
                .with_for_update()
            )
            if dataset is None or run is None:
                raise InferenceError(
                    "benchmark_record_missing",
                    "The benchmark dataset or run record is unavailable.",
                )
            if dataset.task != task.value or run.model_ids != model_ids:
                raise InferenceError(
                    "benchmark_request_mismatch",
                    "The benchmark job no longer matches its registered records.",
                )
            samples = list(dataset.samples)
            run.status = "running"
            run.started_at = datetime.now(UTC)
            await session.commit()

        completed_count = 0
        results: dict[str, Any] = {}
        try:
            for model_id in model_ids:
                definition = self.ai_runtime.registry.get(model_id)
                sample_metrics: list[dict[str, float | int | None]] = []
                for sample_index, sample in enumerate(samples):
                    await self._check_cancelled(job_id)
                    asset_id = uuid.UUID(sample["asset_id"])
                    sample_asset = await self._get_source_asset(asset_id, first_asset.owner_id)
                    sample_path = first_path
                    sample_probe = first_probe
                    if asset_id != first_asset.id:
                        sample_path = work_dir / f"benchmark-{sample_index}.audio"
                        await self._download_source(sample_asset, sample_path)
                        sample_probe = await self.tools.inspect(sample_path)
                        await self.tools.verify_decodable(sample_path)
                    if any(
                        float(reference["end_seconds"]) > sample_probe.duration_seconds
                        for reference in sample["references"]
                    ):
                        raise InferenceError(
                            "benchmark_annotation_out_of_range",
                            "A benchmark annotation exceeds its decoded audio duration.",
                        )

                    async def report_progress(
                        fraction: float,
                        completed_samples: int = completed_count,
                    ) -> bool:
                        overall = (completed_samples + max(0.0, min(1.0, fraction))) / (
                            len(model_ids) * len(samples)
                        )
                        return await self._update_progress(
                            job_id,
                            "BENCHMARKING",
                            min(94, 20 + int(overall * 74)),
                        )

                    cancelled = asyncio.Event()

                    async def watch_benchmark_cancellation(
                        cancel_event: asyncio.Event = cancelled,
                    ) -> None:
                        while True:
                            await asyncio.sleep(0.5)
                            try:
                                await self._check_cancelled(job_id)
                            except (JobCancelledError, JobLeaseLostError):
                                cancel_event.set()
                                return

                    cancellation_watch = asyncio.create_task(watch_benchmark_cancellation())

                    async def benchmark_progress(
                        fraction: float,
                        cancel_event: asyncio.Event = cancelled,
                    ) -> bool:
                        keep_running = await report_progress(fraction)
                        if not keep_running:
                            cancel_event.set()
                        return keep_running

                    try:
                        result, _ = await self.ai_runtime.engine.execute(
                            definition,
                            task,
                            AudioInput(
                                asset_id=sample_asset.id,
                                path=sample_path,
                                duration_seconds=sample_probe.duration_seconds,
                                sample_rate=sample_probe.sample_rate,
                                channel_count=sample_probe.channels,
                            ),
                            parameters=dict(options.get("parameters", {})),
                            progress=benchmark_progress,
                            cancelled=cancelled,
                        )
                    finally:
                        cancellation_watch.cancel()
                        with suppress(asyncio.CancelledError):
                            await cancellation_watch
                    if task == InferenceTask.CHORD_RECOGNITION:
                        chord_references = [
                            TimedChord(**reference.model_dump())
                            for reference in (
                                BenchmarkChordReference.model_validate(item)
                                for item in sample["references"]
                            )
                        ]
                        chord_predictions = [
                            ChordPrediction.model_validate(item) for item in result.predictions
                        ]
                        metrics = chord_metrics(chord_references, chord_predictions)
                    else:
                        note_references = [
                            TimedNote(**reference.model_dump())
                            for reference in (
                                BenchmarkNoteReference.model_validate(item)
                                for item in sample["references"]
                            )
                        ]
                        note_predictions = [
                            NotePrediction.model_validate(item) for item in result.predictions
                        ]
                        metrics = note_metrics(note_references, note_predictions)
                    sample_metrics.append(metrics)
                    completed_count += 1
                results[model_id] = {
                    "sample_count": len(sample_metrics),
                    "aggregate": _aggregate_benchmark_metrics(sample_metrics),
                    "samples": sample_metrics,
                }
            async with self.database.session_factory() as session:
                run = await session.scalar(
                    select(BenchmarkRunRecord).where(BenchmarkRunRecord.job_id == job_id)
                )
                if run is None:
                    raise SQLAlchemyError("Benchmark run disappeared before completion.")
                run.status = "succeeded"
                run.results = results
                run.completed_at = datetime.now(UTC)
                await session.commit()
        except Exception as exc:
            async with self.database.session_factory() as session:
                run = await session.scalar(
                    select(BenchmarkRunRecord).where(BenchmarkRunRecord.job_id == job_id)
                )
                if run is not None:
                    run.status = (
                        "cancelled"
                        if isinstance(exc, (JobCancelledError, JobLeaseLostError))
                        else "failed"
                    )
                    run.error_code = getattr(exc, "code", "benchmark_processing_failed")
                    run.error_message = str(exc)[:255]
                    run.completed_at = datetime.now(UTC)
                    await session.commit()
            raise

    async def _process_music_analysis_job(
        self,
        job_id: uuid.UUID,
        source_asset: AudioAsset,
        source_path: Path,
    ) -> None:
        async with self.database.session_factory() as session:
            analysis = await session.scalar(
                select(MusicAnalysisRecord)
                .where(MusicAnalysisRecord.job_id == job_id)
                .with_for_update()
            )
            if analysis is None or analysis.source_asset_id != source_asset.id:
                raise AudioToolError(
                    "analysis_record_missing",
                    "The versioned music analysis record is unavailable.",
                )
            analysis.status = "running"
            analysis.stage_errors = []
            analysis.result = None
            await session.execute(
                delete(MusicAnalysisEvent).where(MusicAnalysisEvent.analysis_id == analysis.id)
            )
            await session.commit()
            analysis_id = analysis.id

        await self._check_cancelled(job_id)
        await self._update_progress(job_id, "ANALYSIS_FEATURES", 30)
        cancellation = ThreadEvent()
        lease_lost = ThreadEvent()

        async def watch_cancellation() -> None:
            while True:
                await asyncio.sleep(0.5)
                try:
                    await self._check_cancelled(job_id)
                except JobLeaseLostError:
                    lease_lost.set()
                    cancellation.set()
                    return
                except JobCancelledError:
                    cancellation.set()
                    return

        watcher = asyncio.create_task(watch_cancellation())
        try:
            result = await asyncio.to_thread(
                analyze_audio_file,
                source_path,
                cancellation,
            )
        except AnalysisCancelledError as exc:
            if lease_lost.is_set():
                raise JobLeaseLostError from exc
            async with self.database.session_factory() as session:
                analysis = await session.get(MusicAnalysisRecord, analysis_id)
                if analysis is not None:
                    analysis.status = "cancelled"
                    analysis.stage_errors = [{"stage": "analysis", "code": "job_cancelled"}]
                    analysis.completed_at = datetime.now(UTC)
                    await session.commit()
            raise JobCancelledError from exc
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
        try:
            await self._check_cancelled(job_id)
        except JobCancelledError:
            async with self.database.session_factory() as session:
                analysis = await session.get(MusicAnalysisRecord, analysis_id)
                if analysis is not None:
                    analysis.status = "cancelled"
                    analysis.stage_errors = [{"stage": "analysis", "code": "job_cancelled"}]
                    analysis.completed_at = datetime.now(UTC)
                    await session.commit()
            raise
        await self._update_progress(job_id, "ANALYSIS_PERSISTENCE", 90)

        requested = set(analysis.tasks)
        expanded = (
            {"chords", "key", "tempo", "beats", "bass", "melody", "sections"}
            if "full" in requested
            else requested
        )
        optional_errors: list[dict[str, str]] = []
        if "melody" in expanded:
            optional_errors.append(
                {
                    "stage": "melody",
                    "code": "validated_model_unavailable",
                    "message": "No validated note-transcription provider is installed.",
                }
            )
        if "sections" in expanded:
            optional_errors.append(
                {
                    "stage": "sections",
                    "code": "validated_model_unavailable",
                    "message": "No validated song-structure provider is installed.",
                }
            )

        event_rows: list[MusicAnalysisEvent] = []
        if "chords" in expanded:
            event_rows.extend(
                MusicAnalysisEvent(
                    analysis_id=analysis_id,
                    event_type="chord",
                    start_seconds=event["start_seconds"],
                    end_seconds=event["end_seconds"],
                    label=event["symbol"],
                    root=event["root"],
                    quality=event["quality"],
                    confidence=None,
                    evidence={
                        **event["evidence"],
                        "sources": event["sources"],
                        "inversion": event["inversion"],
                        "bass_note": event["bass_note"],
                        "confidence_status": event["confidence_status"],
                        "roman_numeral": event["roman_numeral"],
                    },
                    manually_corrected=False,
                )
                for event in result["chords"]
            )
        if "beats" in expanded or "tempo" in expanded:
            event_rows.extend(
                MusicAnalysisEvent(
                    analysis_id=analysis_id,
                    event_type="beat",
                    start_seconds=beat["time_seconds"],
                    label="beat",
                    evidence={"strength": beat["strength"], "downbeat": None},
                    manually_corrected=False,
                )
                for beat in result["rhythm"]["beats"]
            )
        if "key" in expanded:
            event_rows.extend(
                MusicAnalysisEvent(
                    analysis_id=analysis_id,
                    event_type="key_change",
                    start_seconds=change["start_seconds"],
                    label=f"{change['tonic']} {change['mode']}",
                    evidence=change,
                    manually_corrected=False,
                )
                for change in result["key"]["changes"]
            )
        if "bass" in expanded:
            event_rows.extend(
                MusicAnalysisEvent(
                    analysis_id=analysis_id,
                    event_type="bass",
                    start_seconds=event["start_seconds"],
                    end_seconds=event["end_seconds"],
                    label=event["note"],
                    evidence=event,
                    manually_corrected=False,
                )
                for event in result["bass_events"]
            )

        filtered_result = {
            "duration_seconds": result["duration_seconds"],
            "sample_rate": result["sample_rate"],
            "methodology": result["methodology"],
        }
        for task in expanded:
            if task == "chords":
                filtered_result["chords"] = result["chords"]
            elif task == "key":
                filtered_result["key"] = result["key"]
            elif task in {"tempo", "beats"}:
                filtered_result["rhythm"] = result["rhythm"]
            elif task == "bass":
                filtered_result["bass_events"] = result["bass_events"]
            elif task == "melody":
                filtered_result["melody"] = result["melody"]
            elif task == "sections":
                filtered_result["sections"] = result["sections"]

        async with self.database.session_factory() as session:
            analysis = await session.get(MusicAnalysisRecord, analysis_id)
            if analysis is None:
                raise SQLAlchemyError("Music analysis record disappeared before completion.")
            analysis.result = filtered_result
            analysis.stage_errors = optional_errors
            analysis.status = "partial" if optional_errors else "succeeded"
            analysis.completed_at = datetime.now(UTC)
            session.add_all(event_rows)
            await session.commit()

    async def _mark_music_analysis_terminal(
        self,
        job_id: uuid.UUID,
        state: str,
        error_code: str,
        message: str,
    ) -> None:
        async with self.database.session_factory() as session:
            analysis = await session.scalar(
                select(MusicAnalysisRecord).where(MusicAnalysisRecord.job_id == job_id)
            )
            if analysis is not None and analysis.status in {"queued", "running"}:
                analysis.status = state
                analysis.stage_errors = [
                    {"stage": "analysis", "code": error_code, "message": message[:255]}
                ]
                analysis.completed_at = datetime.now(UTC)
                await session.commit()

    async def _process_ai_inference_job(
        self,
        job_id: uuid.UUID,
        source_asset: AudioAsset,
        source_path: Path,
        probe: ProbeResult,
        options: dict[str, Any],
    ) -> None:
        if self.ai_runtime is None:
            raise InferenceError(
                "inference_runtime_unavailable",
                "The AI inference runtime is not configured for this worker.",
            )
        try:
            definition = self.ai_runtime.registry.get(str(options["model_id"]))
            task = InferenceTask(str(options["task"]))
        except KeyError as exc:
            raise InferenceError(
                "invalid_inference_request", "The model ID or task is missing."
            ) from exc
        except ValueError as exc:
            raise InferenceError(
                "invalid_inference_task", "The inference task is invalid."
            ) from exc
        if not definition.enabled:
            raise InferenceError("model_disabled", f"Model '{definition.model_id}' is disabled.")

        started = datetime.now(UTC)
        async with self.database.session_factory() as session:
            record = await session.scalar(
                select(AIInferenceRecord)
                .where(AIInferenceRecord.job_id == job_id)
                .with_for_update()
            )
            if record is None:
                record = AIInferenceRecord(
                    job_id=job_id,
                    owner_id=source_asset.owner_id,
                    source_asset_id=source_asset.id,
                    model_id=definition.model_id,
                    model_version=definition.version,
                    provider=definition.provider,
                    task=task.value,
                    status="running",
                    started_at=started,
                )
                session.add(record)
            else:
                record.status = "running"
                record.result = None
                record.metrics = None
                record.validation_error = None
                record.error_code = None
                record.error_message = None
                record.execution_device = None
                record.inference_duration_ms = None
                record.started_at = started
                record.completed_at = None
            await session.commit()
            await session.refresh(record)
            record_id = record.id

        cancelled = asyncio.Event()

        async def watch_cancellation() -> None:
            while True:
                await asyncio.sleep(0.5)
                try:
                    await self._check_cancelled(job_id)
                except (JobCancelledError, JobLeaseLostError):
                    cancelled.set()
                    return

        async def report_progress(fraction: float) -> bool:
            completed = await self._update_progress(
                job_id,
                "INFERENCING",
                min(94, 20 + int(max(0.0, min(1.0, fraction)) * 74)),
            )
            if not completed:
                cancelled.set()
            return completed

        cancellation_watch = asyncio.create_task(watch_cancellation())
        try:
            result, metrics = await self.ai_runtime.engine.execute(
                definition,
                task,
                AudioInput(
                    asset_id=source_asset.id,
                    path=source_path,
                    duration_seconds=probe.duration_seconds,
                    sample_rate=probe.sample_rate,
                    channel_count=probe.channels,
                ),
                parameters=dict(options.get("parameters", {})),
                progress=report_progress,
                cancelled=cancelled,
            )
            async with self.database.session_factory() as session:
                record = await session.get(AIInferenceRecord, record_id)
                if record is None:
                    raise SQLAlchemyError("AI inference record disappeared before completion.")
                record.status = "succeeded"
                record.execution_device = result.execution_device.value
                record.result = result.model_dump(mode="json")
                record.metrics = metrics
                record.inference_duration_ms = result.processing_duration_ms
                record.completed_at = datetime.now(UTC)
                await session.commit()
        except InferenceError as exc:
            record_status = (
                "validation_failed"
                if isinstance(exc, InferenceValidationError)
                else "cancelled"
                if exc.code == "job_cancelled"
                else "failed"
            )
            async with self.database.session_factory() as session:
                record = await session.get(AIInferenceRecord, record_id)
                if record is not None:
                    record.status = record_status
                    record.error_code = exc.code
                    record.error_message = str(exc)[:255]
                    record.validation_error = (
                        {"code": exc.code, "message": str(exc)[:255]}
                        if record_status == "validation_failed"
                        else None
                    )
                    record.completed_at = datetime.now(UTC)
                    await session.commit()
            if exc.code == "job_cancelled":
                raise JobCancelledError from exc
            raise
        finally:
            cancellation_watch.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_watch

    async def _process_dsp_job(
        self,
        job_id: uuid.UUID,
        source_asset: AudioAsset,
        source_path: Path,
        probe: ProbeResult,
        operation: str,
        options: dict[str, Any],
        work_dir: Path,
    ) -> list[uuid.UUID]:
        started = datetime.now(UTC)
        monotonic_started = asyncio.get_running_loop().time()
        async with self.database.session_factory() as session:
            processing_record = await session.scalar(
                select(AudioProcessingRecord)
                .where(AudioProcessingRecord.job_id == job_id)
                .with_for_update()
            )
            if processing_record is None:
                processing_record = AudioProcessingRecord(
                    job_id=job_id,
                    source_asset_id=source_asset.id,
                    operation=operation,
                    parameters=options,
                    status="running",
                    processing_version="1.0",
                    started_at=started,
                )
                session.add(processing_record)
            else:
                processing_record.status = "running"
                processing_record.parameters = options
                processing_record.processing_version = "1.0"
                processing_record.started_at = started
                processing_record.completed_at = None
                processing_record.duration_ms = None
                processing_record.error_code = None
                processing_record.error_message = None
                processing_record.result = None
            await session.commit()
            await session.refresh(processing_record)
            record_id = processing_record.id

        output_ids: list[uuid.UUID] = []
        try:
            await self._update_progress(job_id, "DECODING", 35)
            processing_options = dict(options)
            processing_probe = probe
            decode_trim_start: float | None = None
            decode_trim_end: float | None = None
            if operation == "time_pitch":
                if probe.channels not in {1, 2}:
                    raise ValueError("Pitch and tempo processing supports mono or stereo input.")
                target_rate = int(
                    processing_options.get("sample_rate") or min(probe.sample_rate, 48_000)
                )
                original_start = float(processing_options.get("start_seconds") or 0.0)
                original_end = float(
                    processing_options.get("end_seconds") or probe.duration_seconds
                )
                if original_start < 0 or original_end <= original_start:
                    raise ValueError(
                        "The selected time range must be non-empty and within the source."
                    )
                if original_end > probe.duration_seconds:
                    raise ValueError("The selected time range exceeds the source duration.")
                selected_duration = original_end - original_start
                expected_duration = selected_duration / float(
                    processing_options.get("tempo_factor", 1.0)
                )
                if expected_duration > self.settings.max_audio_duration_seconds:
                    raise ValueError(
                        "The transformed output would exceed the maximum audio duration."
                    )
                max_wav_frames = (2**32 - 4096) // (probe.channels * 4)
                if target_rate * selected_duration > max_wav_frames:
                    raise ValueError(
                        "The decoded audio section exceeds the SoundTouch WAV "
                        "intermediate size limit."
                    )
                if target_rate * expected_duration > max_wav_frames:
                    raise ValueError(
                        "The transformed output exceeds the SoundTouch WAV intermediate size limit."
                    )
                if processing_options.get("start_seconds") is not None:
                    decode_trim_start = original_start
                    decode_trim_end = original_end
                    processing_options["_source_start_seconds"] = original_start
                    processing_options["_source_end_seconds"] = original_end
                    processing_options["_selected_section"] = True
                    processing_options["start_seconds"] = 0.0
                    processing_options["end_seconds"] = selected_duration
                else:
                    processing_options["_source_start_seconds"] = 0.0
                    processing_options["_source_end_seconds"] = probe.duration_seconds
                processing_options["sample_rate"] = target_rate
                processing_options["_timeout_seconds"] = (
                    self.settings.audio_conversion_timeout_seconds
                )
                processing_probe = replace(
                    probe,
                    sample_rate=target_rate,
                    duration_seconds=selected_duration,
                )
            if operation == "concatenate":
                processing_options, processing_probe = await self._prepare_concatenation(
                    job_id,
                    source_asset,
                    source_path,
                    probe,
                    processing_options,
                    work_dir,
                )
                decoded_source = Path(processing_options["_source_paths"][0])
            else:
                decoded_source = work_dir / "dsp-source.wav"

                async def decode_progress(elapsed_seconds: float) -> bool:
                    fraction = min(
                        1.0,
                        max(0.0, elapsed_seconds / processing_probe.duration_seconds),
                    )
                    return await self._update_progress(
                        job_id,
                        "DECODING",
                        min(49, 35 + int(fraction * 14)),
                    )

                await self.tools.convert(
                    source_path,
                    decoded_source,
                    {
                        "output_format": "wav",
                        "channels": "original",
                        "sample_rate": (
                            processing_probe.sample_rate
                            if operation == "time_pitch"
                            and processing_probe.sample_rate != probe.sample_rate
                            else None
                        ),
                        "bit_depth": 32,
                        "float32": True,
                        "normalize": False,
                        "allow_upsampling": False,
                        "trim_start_seconds": decode_trim_start,
                        "trim_end_seconds": decode_trim_end,
                    },
                    probe,
                    decode_progress,
                )
                if operation == "time_pitch":
                    processing_probe = await self.tools.inspect(decoded_source)
                    if processing_options.get("_selected_section"):
                        processing_options["end_seconds"] = processing_probe.duration_seconds

            async def report_progress(fraction: float) -> bool:
                return await self._update_progress(
                    job_id,
                    "PROCESSING",
                    min(94, 50 + int(max(0.0, min(1.0, fraction)) * 44)),
                )

            if operation == "normalize" and processing_options.get("method") == "loudness":
                normalization_started = asyncio.get_running_loop().time()
                output_path = work_dir / "loudness-normalized.wav"
                loudness_result = await self.tools.normalize_loudness(
                    decoded_source,
                    output_path,
                    target_lufs=float(processing_options["target_lufs"]),
                    true_peak_dbfs=float(processing_options["true_peak_dbfs"]),
                    loudness_range_lu=float(processing_options["loudness_range_lu"]),
                    limit=bool(processing_options.get("limit", False)),
                    sample_rate=processing_probe.sample_rate,
                    progress=report_progress,
                )
                result = DspResult(
                    [DspOutput(output_path, AssetType.NORMALIZED_AUDIO.value, {})],
                    loudness_result,
                    int((asyncio.get_running_loop().time() - normalization_started) * 1000),
                )
            else:
                result = await process_dsp_audio(
                    decoded_source,
                    work_dir,
                    source_asset_id=source_asset.id,
                    operation=operation,
                    options=processing_options,
                    probe=processing_probe,
                    progress=report_progress,
                    soundtouch_path=self.settings.soundtouch_path,
                    soundtouch_version=self.settings.soundtouch_version,
                )
            if operation == "time_pitch":
                result = await self._prepare_time_pitch_export(
                    job_id,
                    result,
                    processing_probe,
                    processing_options,
                    work_dir,
                )
            for position, output in enumerate(result.outputs):
                await self._check_cancelled(job_id)
                output_id = await self._store_dsp_output(
                    output.path,
                    output.asset_type,
                    source_asset,
                    job_id,
                    position,
                )
                output_ids.append(output_id)
                if output.asset_type in {
                    AssetType.PROCESSED_AUDIO.value,
                    AssetType.TRIMMED_AUDIO.value,
                    AssetType.NORMALIZED_AUDIO.value,
                }:
                    output_probe = await self.tools.inspect(output.path)
                    await self.tools.verify_decodable(output.path)
                    await self._store_output_probe(output_id, output_probe)

            async with self.database.session_factory() as session:
                processing_record = await session.get(AudioProcessingRecord, record_id)
                if processing_record is None:
                    raise SQLAlchemyError("DSP processing record disappeared before completion.")
                processing_record.status = "succeeded"
                processing_record.completed_at = datetime.now(UTC)
                processing_record.duration_ms = result.duration_ms
                processing_record.result = _processing_result_summary(operation, result.result)
                existing_outputs = {
                    output.position: output
                    for output in (
                        await session.scalars(
                            select(AudioProcessingOutput).where(
                                AudioProcessingOutput.processing_record_id == processing_record.id
                            )
                        )
                    ).all()
                }
                for position, (output_id, output) in enumerate(
                    zip(output_ids, result.outputs, strict=True)
                ):
                    existing_output = existing_outputs.pop(position, None)
                    if existing_output is None:
                        existing_output = AudioProcessingOutput(
                            processing_record_id=processing_record.id,
                            position=position,
                        )
                        session.add(existing_output)
                    existing_output.output_asset_id = output_id
                    existing_output.output_metadata = {
                        "operation": operation,
                        **output.metadata,
                    }
                for obsolete_output in existing_outputs.values():
                    await session.delete(obsolete_output)
                source_ids = options.get("asset_ids", [source_asset.id])
                existing_sources = {
                    source.position: source
                    for source in (
                        await session.scalars(
                            select(AudioProcessingSource).where(
                                AudioProcessingSource.processing_record_id == processing_record.id
                            )
                        )
                    ).all()
                }
                for position, source_id in enumerate(source_ids):
                    existing_source = existing_sources.pop(position, None)
                    if existing_source is None:
                        existing_source = AudioProcessingSource(
                            processing_record_id=processing_record.id,
                            position=position,
                        )
                        session.add(existing_source)
                    existing_source.source_asset_id = uuid.UUID(str(source_id))
                for obsolete_source in existing_sources.values():
                    await session.delete(obsolete_source)
                await session.commit()
            return output_ids
        except (JobCancelledError, ProcessingCancelledError):
            for output_id in output_ids:
                await self._delete_uncommitted_output(output_id)
            async with self.database.session_factory() as session:
                processing_record = await session.get(AudioProcessingRecord, record_id)
                if processing_record is not None:
                    processing_record.status = "cancelled"
                    processing_record.completed_at = datetime.now(UTC)
                    processing_record.duration_ms = int(
                        (asyncio.get_running_loop().time() - monotonic_started) * 1000
                    )
                    processing_record.error_code = "job_cancelled"
                    await session.commit()
            await self._check_cancelled(job_id)
            raise
        except (StorageError, SQLAlchemyError):
            async with self.database.session_factory() as session:
                processing_record = await session.get(AudioProcessingRecord, record_id)
                if processing_record is not None:
                    processing_record.status = "failed"
                    processing_record.completed_at = datetime.now(UTC)
                    processing_record.duration_ms = int(
                        (asyncio.get_running_loop().time() - monotonic_started) * 1000
                    )
                    processing_record.error_code = "temporary_processing_failure"
                    processing_record.error_message = (
                        "A temporary storage or database error occurred."
                    )
                    await session.commit()
            raise
        except AudioToolError as exc:
            if exc.code == "job_cancelled":
                await self._check_cancelled(job_id)
            for output_id in output_ids:
                await self._delete_uncommitted_output(output_id)
            async with self.database.session_factory() as session:
                processing_record = await session.get(AudioProcessingRecord, record_id)
                if processing_record is not None:
                    processing_record.status = "failed"
                    processing_record.completed_at = datetime.now(UTC)
                    processing_record.duration_ms = int(
                        (asyncio.get_running_loop().time() - monotonic_started) * 1000
                    )
                    processing_record.error_code = exc.code
                    processing_record.error_message = str(exc)[:255]
                    await session.commit()
            raise
        except Exception as exc:
            for output_id in output_ids:
                await self._delete_uncommitted_output(output_id)
            async with self.database.session_factory() as session:
                processing_record = await session.get(AudioProcessingRecord, record_id)
                if processing_record is not None:
                    processing_record.status = "failed"
                    processing_record.completed_at = datetime.now(UTC)
                    processing_record.duration_ms = int(
                        (asyncio.get_running_loop().time() - monotonic_started) * 1000
                    )
                    processing_record.error_code = "dsp_processing_failed"
                    processing_record.error_message = str(exc)[:255]
                    await session.commit()
            raise

    async def _prepare_time_pitch_export(
        self,
        job_id: uuid.UUID,
        result: DspResult,
        source_probe: ProbeResult,
        options: dict[str, Any],
        work_dir: Path,
    ) -> DspResult:
        if len(result.outputs) != 1:
            raise ValueError("Audio transformation must produce exactly one output.")
        intermediate = result.outputs[0].path
        await self.tools.verify_decodable(intermediate)
        intermediate_probe = await self.tools.inspect(intermediate)
        expected_duration = float(result.result["expected_duration_seconds"])
        tolerance = max(0.2, expected_duration * 0.02)
        actual_duration = intermediate_probe.duration_seconds
        if abs(actual_duration - expected_duration) > tolerance:
            raise AudioToolError(
                "audio_transform_duration_mismatch",
                "Transformed audio duration is outside the expected tolerance.",
            )
        if (
            intermediate_probe.sample_rate != source_probe.sample_rate
            or intermediate_probe.channels != source_probe.channels
        ):
            raise AudioToolError(
                "audio_transform_properties_mismatch",
                "Transformed audio changed its sample rate or channel count.",
            )
        quality = await asyncio.to_thread(inspect_quality, intermediate)
        if any(
            float(value) > 0 for value in result.result["input_quality"]["peak_amplitude"]
        ) and all(float(value) == 0 for value in quality["peak_amplitude"]):
            raise AudioToolError(
                "audio_transform_unexpected_silence",
                "Transformed audio is unexpectedly silent.",
            )
        output_format = str(options["output_format"])
        output_path = intermediate
        encoding_duration_ms = 0
        if output_format != "wav":
            output_path = work_dir / f"transformed.{output_format}"

            async def encoding_progress(elapsed_seconds: float) -> bool:
                ratio = min(1.0, max(0.0, elapsed_seconds / max(expected_duration, 0.001)))
                return await self._update_progress(
                    job_id,
                    "ENCODING",
                    min(98, 94 + int(ratio * 4)),
                )

            _, encoding_duration_ms = await self.tools.convert(
                intermediate,
                output_path,
                {
                    "output_format": output_format,
                    "sample_rate": None,
                    "channels": "original",
                    "bit_depth": 24,
                    "float32": False,
                    "bitrate_kbps": int(options["bitrate_kbps"]),
                    "normalize": False,
                    "allow_upsampling": False,
                },
                intermediate_probe,
                encoding_progress,
            )
            await self.tools.verify_decodable(output_path)
            output_probe = await self.tools.inspect(output_path)
            if abs(output_probe.duration_seconds - expected_duration) > max(tolerance, 0.5):
                raise AudioToolError(
                    "audio_transform_duration_mismatch",
                    "Encoded output duration is outside the expected tolerance.",
                )
        else:
            output_probe = intermediate_probe
        details = {
            "output_format": output_format,
            "output_validation": {
                "readable": True,
                "decodable": True,
                "sample_rate": output_probe.sample_rate,
                "channel_count": output_probe.channels,
                "bit_depth": output_probe.bit_depth,
                "duration_seconds": output_probe.duration_seconds,
                "peak_amplitude": quality["peak_amplitude"],
                "rms_amplitude": quality["rms_amplitude"],
                "clipped_sample_percentage": quality["clipped_sample_percentage"],
                "quality_measurement_stage": "float WAV intermediate before final encoding",
            },
            "encoding_duration_ms": encoding_duration_ms,
        }
        return DspResult(
            [
                DspOutput(
                    output_path,
                    result.outputs[0].asset_type,
                    {
                        **result.outputs[0].metadata,
                        "output_format": output_format,
                        "sample_rate": output_probe.sample_rate,
                        "channel_count": output_probe.channels,
                        "duration_seconds": output_probe.duration_seconds,
                    },
                )
            ],
            {**result.result, **details},
            result.duration_ms + encoding_duration_ms,
        )

    async def _prepare_concatenation(
        self,
        job_id: uuid.UUID,
        first_asset: AudioAsset,
        first_path: Path,
        first_probe: ProbeResult,
        options: dict[str, Any],
        work_dir: Path,
    ) -> tuple[dict[str, Any], ProbeResult]:
        source_ids = [uuid.UUID(str(value)) for value in options.get("asset_ids", [])]
        if (
            len(source_ids) < 2
            or source_ids[0] != first_asset.id
            or len(set(source_ids)) != len(source_ids)
        ):
            raise ValueError(
                "Concatenation source IDs must be unique and begin with the job input."
            )
        sources: list[tuple[AudioAsset, Path, ProbeResult]] = []
        for index, asset_id in enumerate(source_ids):
            asset = (
                first_asset
                if index == 0
                else await self._get_source_asset(asset_id, first_asset.owner_id)
            )
            path = first_path if index == 0 else work_dir / f"concat-source-{index}.audio"
            if index:
                await self._download_source(asset, path)
            source_probe = first_probe if index == 0 else await self.tools.inspect(path)
            await self.tools.verify_decodable(path)
            sources.append((asset, path, source_probe))
        rates = {source_probe.sample_rate for _, _, source_probe in sources}
        channel_counts = {source_probe.channels for _, _, source_probe in sources}
        target_rate_value = options.get("sample_rate")
        if len(rates) > 1 and target_rate_value is None:
            raise ValueError("Sources with different sample rates require an explicit sample_rate.")
        channels = str(options.get("channels", "preserve"))
        if len(channel_counts) > 1 and channels == "preserve":
            raise ValueError(
                "Sources with different channel counts require mono or stereo conversion."
            )
        target_rate = int(target_rate_value) if target_rate_value else first_probe.sample_rate
        target_channels = (
            1 if channels == "mono" else 2 if channels == "stereo" else first_probe.channels
        )
        if channels == "preserve" and len(channel_counts) > 1:
            raise ValueError("The source channel counts are incompatible.")
        prepared_paths: list[str] = []
        total_duration = 0.0
        for index, (_, path, source_probe) in enumerate(sources):
            destination = work_dir / f"concat-prepared-{index}.wav"
            await self.tools.convert(
                path,
                destination,
                {
                    "output_format": "wav",
                    "sample_rate": target_rate if target_rate != source_probe.sample_rate else None,
                    "channels": channels if channels != "preserve" else "original",
                    "bit_depth": 32,
                    "float32": True,
                    "normalize": False,
                    "allow_upsampling": target_rate_value is not None,
                },
                source_probe,
                lambda _elapsed: self._update_progress(job_id, "DECODING", 45),
            )
            prepared_paths.append(str(destination))
            total_duration += source_probe.duration_seconds
        crossfade = float(options.get("crossfade_seconds", 0))
        total_duration -= crossfade * (len(sources) - 1)
        if total_duration <= 0:
            raise ValueError("Crossfade duration exceeds the total concatenated duration.")
        options["_source_paths"] = prepared_paths
        return options, replace(
            first_probe,
            sample_rate=target_rate,
            channels=target_channels,
            duration_seconds=total_duration,
        )

    async def _store_dsp_output(
        self,
        output_path: Path,
        asset_type: str,
        source_asset: AudioAsset,
        job_id: uuid.UUID,
        position: int,
    ) -> uuid.UUID:
        parsed_type = AssetType(asset_type)
        upload = _PathUpload(output_path)
        try:
            staged = await self.storage.stage_upload(
                upload,
                filename=output_path.name,
                asset_type=parsed_type,
                max_size_bytes=self.settings.max_audio_output_size_bytes,
            )
        finally:
            await upload.close()
        async with self.database.session_factory() as session:
            asset, _ = await self.storage.create_asset(
                session,
                source_asset.owner_id,
                staged,
                parent_asset_id=source_asset.id,
                idempotency_key=f"dsp:{job_id}:{position}",
            )
            return asset.id

    async def _store_converted_asset(
        self,
        output_path: Path,
        owner_id: uuid.UUID,
        parent_asset_id: uuid.UUID,
        output_format: str,
    ) -> uuid.UUID:
        upload = _PathUpload(output_path)
        try:
            staged: StoredUpload = await self.storage.stage_upload(
                upload,
                filename=f"converted.{output_format}",
                asset_type=AssetType.CONVERTED_AUDIO,
                max_size_bytes=self.settings.max_audio_output_size_bytes,
            )
        finally:
            await upload.close()
        async with self.database.session_factory() as session:
            asset, _ = await self.storage.create_asset(
                session,
                owner_id,
                staged,
                parent_asset_id=parent_asset_id,
                idempotency_key=f"audio-convert:{parent_asset_id}:{staged.sha256}",
            )
            return asset.id

    async def _save_conversion(
        self,
        job_id: uuid.UUID,
        source_asset_id: uuid.UUID,
        output_asset_id: uuid.UUID,
        options: dict[str, Any],
        duration_ms: int,
        ffmpeg_version: str,
    ) -> None:
        async with self.database.session_factory() as session:
            session.add(
                AudioConversion(
                    job_id=job_id,
                    source_asset_id=source_asset_id,
                    output_asset_id=output_asset_id,
                    profile=options,
                    conversion_duration_ms=duration_ms,
                    ffmpeg_version=ffmpeg_version,
                )
            )
            await session.commit()

    async def _store_output_probe(self, asset_id: uuid.UUID, probe: ProbeResult) -> None:
        async with self.database.session_factory() as session:
            asset = await session.get(AudioAsset, asset_id)
            if asset is None:
                raise AudioToolError("asset_missing", "Converted audio asset was not committed.")
            asset.audio_format = probe.codec
            asset.duration_seconds = probe.duration_seconds
            asset.sample_rate = probe.sample_rate
            asset.channel_count = probe.channels
            asset.bit_depth = probe.bit_depth
            await session.commit()

    async def _delete_uncommitted_output(self, asset_id: uuid.UUID) -> None:
        async with self.database.session_factory() as session:
            result = await session.execute(
                select(AudioAsset.owner_id).where(AudioAsset.id == asset_id)
            )
            owner_id = result.scalar_one_or_none()
            if owner_id is None:
                return
            try:
                await self.storage.delete_asset(session, owner_id, asset_id)
            except Exception:
                logger.exception(
                    "Could not delete cancelled conversion output",
                    extra={"asset_id": str(asset_id)},
                )

    async def _save_failed_conversion(
        self,
        job_id: uuid.UUID,
        source_asset_id: uuid.UUID,
        options: dict[str, Any],
        error_code: str,
    ) -> None:
        if "output_format" not in options:
            return
        async with self.database.session_factory() as session:
            existing = await session.scalar(
                select(AudioConversion).where(AudioConversion.job_id == job_id)
            )
            if existing is None:
                try:
                    ffmpeg_version, _ = await self.tools.versions()
                except AudioToolError:
                    ffmpeg_version = "unavailable"
                session.add(
                    AudioConversion(
                        job_id=job_id,
                        source_asset_id=source_asset_id,
                        output_asset_id=None,
                        profile=options,
                        ffmpeg_version=ffmpeg_version,
                        error_code=error_code,
                    )
                )
                await session.commit()

    async def _update_progress(self, job_id: uuid.UUID, stage: str, progress: int) -> bool:
        async with self.database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None or job.status != "running":
                raise JobLeaseLostError
            if job.cancel_requested:
                return False
            job.stage = stage
            job.progress = max(job.progress, min(99, max(0, progress)))
            job.heartbeat_at = datetime.now(UTC)
            job.updated_at = datetime.now(UTC)
            await session.commit()
            return True

    async def _check_cancelled(self, job_id: uuid.UUID) -> None:
        async with self.database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None or job.status != "running":
                raise JobLeaseLostError
            if job.cancel_requested:
                raise JobCancelledError

    async def _complete_job(self, job_id: uuid.UUID, output_asset_id: uuid.UUID | None) -> None:
        async with self.database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None or job.status != "running":
                raise JobLeaseLostError
            if job.cancel_requested:
                raise JobCancelledError
            job.output_asset_id = output_asset_id
            job.status = "succeeded"
            job.stage = "COMPLETED"
            job.progress = 100
            job.finished_at = datetime.now(UTC)
            job.heartbeat_at = job.finished_at
            job.updated_at = job.finished_at
            await session.commit()

    async def _finish_job(
        self,
        job_id: uuid.UUID,
        status: str,
        stage: str,
        error_code: str,
        error_message: str | None,
    ) -> None:
        async with self.database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None or job.status != "running":
                return
            job.status = status
            job.stage = stage
            job.error_code = error_code
            job.error_message = error_message[:255] if error_message else None
            job.updated_at = datetime.now(UTC)
            job.heartbeat_at = job.updated_at
            if status != "retry_wait":
                job.finished_at = job.updated_at
            else:
                job.progress = 0
            await session.commit()


def _verify_output(
    source: ProbeResult,
    output: ProbeResult,
    options: dict[str, Any],
) -> None:
    trim_start = float(options.get("trim_start_seconds") or 0)
    trim_end = options.get("trim_end_seconds")
    expected_duration = (
        float(trim_end) - trim_start
        if trim_end is not None
        else source.duration_seconds - trim_start
    )
    if expected_duration <= 0 or abs(output.duration_seconds - expected_duration) > max(
        0.5, expected_duration * 0.02
    ):
        raise AudioToolError(
            "invalid_conversion_output",
            "Converted audio duration differs from the requested source range.",
        )
    if output.channels <= 0 or output.sample_rate <= 0:
        raise AudioToolError(
            "invalid_conversion_output", "Converted audio has invalid stream properties."
        )
    if options["output_format"] == "wav" and output.container.find("wav") < 0:
        raise AudioToolError("invalid_conversion_output", "FFmpeg did not produce WAV output.")
    source_rate = source.sample_rate
    requested_rate = options.get("sample_rate")
    expected_rate = int(requested_rate) if requested_rate else source_rate
    if expected_rate > source_rate and not bool(options.get("allow_upsampling")):
        expected_rate = source_rate
    if output.sample_rate != expected_rate:
        raise AudioToolError(
            "invalid_conversion_output", "Converted audio has an unexpected sample rate."
        )
    requested_channels = options.get("channels", "original")
    expected_channels = (
        source.channels
        if requested_channels == "original"
        else 1
        if requested_channels == "mono"
        else 2
    )
    if output.channels != expected_channels:
        raise AudioToolError(
            "invalid_conversion_output", "Converted audio has an unexpected channel count."
        )
    if options["output_format"] == "wav" and not bool(options.get("float32")):
        if output.bit_depth != int(options.get("bit_depth", 24)):
            raise AudioToolError(
                "invalid_conversion_output", "Converted WAV has an unexpected bit depth."
            )


def _processing_result_summary(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    if operation == "waveform":
        return {
            key: value
            for key, value in result.items()
            if key not in {"channels", "timestamps_seconds"}
        }
    return result
