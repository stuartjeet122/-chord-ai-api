from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from music_api.api.dependencies import (
    get_db_session,
    get_owner_id,
    get_resources,
    get_storage_service,
)
from music_api.audio.dsp.schemas import (
    AudioTransformRequest,
    ChannelRequest,
    ChunkRequest,
    ConcatenateRequest,
    FadeRequest,
    FilterRequest,
    GainRequest,
    NormalizeRequest,
    PipelineRequest,
    ProcessingHistoryResponse,
    ResampleRequest,
    SilenceDetectionRequest,
    SplitRequest,
    TrimRequest,
    WaveformRequest,
)
from music_api.audio.models import (
    AudioArtwork,
    AudioMetadata,
    AudioProcessingOutput,
    AudioProcessingRecord,
)
from music_api.audio.profiles import CONVERSION_PROFILES
from music_api.audio.schemas import (
    AudioArtworkResponse,
    AudioAssetsResponse,
    AudioMetadataResponse,
    ConversionProfileResponse,
    ConversionRequest,
    JobResponse,
)
from music_api.audio.transforms import resolve_transform_parameters
from music_api.errors import ApplicationError
from music_api.jobs.models import ProcessingJob
from music_api.jobs.service import JobConflictError, create_job
from music_api.resources import RuntimeResources
from music_api.storage.models import AssetStatus, AssetType, AudioAsset
from music_api.storage.schemas import AssetResponse
from music_api.storage.service import AssetNotFoundError, StorageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio"])
OwnerId = Annotated[uuid.UUID, Depends(get_owner_id)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
Storage = Annotated[StorageService, Depends(get_storage_service)]
Resources = Annotated[RuntimeResources, Depends(get_resources)]

_AUDIO_ASSET_TYPES = {
    AssetType.ORIGINAL_AUDIO.value,
    AssetType.CONVERTED_AUDIO.value,
    AssetType.NORMALIZED_AUDIO.value,
    AssetType.TRIMMED_AUDIO.value,
    AssetType.PROCESSED_AUDIO.value,
    AssetType.GENERATED_AUDIO.value,
    AssetType.SEPARATED_STEM.value,
}


@router.post(
    "/concatenate",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue compatible audio assets for concatenation",
)
async def queue_concatenation(
    request: ConcatenateRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    assets = [
        await _owned_audio_asset(storage, session, owner_id, asset_id)
        for asset_id in request.asset_ids
    ]
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        assets[0].id,
        "concatenate",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.get(
    "/conversion-profiles",
    response_model=ConversionProfileResponse,
    summary="List supported conversion profiles",
)
async def conversion_profiles() -> ConversionProfileResponse:
    return ConversionProfileResponse(profiles=list(CONVERSION_PROFILES))


@router.post(
    "/{asset_id}/pipeline",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a validated sequential audio processing pipeline",
)
async def queue_pipeline(
    asset_id: uuid.UUID,
    request: PipelineRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "pipeline",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/waveform",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue waveform generation",
)
async def queue_waveform(
    asset_id: uuid.UUID,
    request: WaveformRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "waveform",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.get(
    "/{asset_id}/waveform",
    response_model=None,
    summary="Retrieve the latest generated waveform",
)
async def get_waveform(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
) -> StreamingResponse:
    await _owned_audio_asset(storage, session, owner_id, asset_id)
    output = await session.scalar(
        select(AudioAsset)
        .join(AudioProcessingOutput, AudioProcessingOutput.output_asset_id == AudioAsset.id)
        .join(
            AudioProcessingRecord,
            AudioProcessingRecord.id == AudioProcessingOutput.processing_record_id,
        )
        .where(
            AudioProcessingRecord.source_asset_id == asset_id,
            AudioProcessingRecord.operation == "waveform",
            AudioProcessingRecord.status == "succeeded",
            AudioAsset.owner_id == owner_id,
            AudioAsset.status == AssetStatus.AVAILABLE.value,
        )
        .order_by(AudioProcessingRecord.created_at.desc(), AudioProcessingOutput.position)
        .limit(1)
    )
    if output is None:
        raise ApplicationError(
            "waveform_not_found",
            "No waveform has been generated for this audio asset.",
            404,
        )
    return StreamingResponse(
        storage.provider.iter_bytes(output.storage_key),
        media_type="application/json",
        headers={
            "Content-Length": str(output.size_bytes),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/{asset_id}/silence-detection",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue silence detection",
)
async def queue_silence_detection(
    asset_id: uuid.UUID,
    request: SilenceDetectionRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "silence",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/transform",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue pitch, tempo, key, and selected-section audio transformations",
)
async def queue_audio_transform(
    asset_id: uuid.UUID,
    request: AudioTransformRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    if (
        request.end_seconds is not None
        and asset.duration_seconds is not None
        and request.end_seconds > asset.duration_seconds
    ):
        raise ApplicationError(
            "invalid_transform_range",
            "end_seconds exceeds the source audio duration.",
            422,
        )
    try:
        options = resolve_transform_parameters(request)
    except ValueError as exc:
        raise ApplicationError("invalid_audio_transform", str(exc), 422) from exc
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "time_pitch",
        options,
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/trim",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue sample-accurate audio trimming",
)
async def queue_trim(
    asset_id: uuid.UUID,
    request: TrimRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    if asset.duration_seconds is not None and request.end_seconds > asset.duration_seconds:
        raise ApplicationError("invalid_trim_range", "end_seconds exceeds source duration.", 422)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "trim",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/split",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue audio segmentation",
)
async def queue_split(
    asset_id: uuid.UUID,
    request: SplitRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    if asset.duration_seconds is not None and any(
        boundary >= asset.duration_seconds for boundary in request.boundaries_seconds
    ):
        raise ApplicationError(
            "invalid_split_boundary",
            "Split boundaries must fall strictly inside the source duration.",
            422,
        )
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "split",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/chunks",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue time-aligned chunks for model preprocessing",
)
async def queue_chunks(
    asset_id: uuid.UUID,
    request: ChunkRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "chunks",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/normalize",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue peak or RMS normalization",
)
async def queue_normalize(
    asset_id: uuid.UUID,
    request: NormalizeRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "normalize",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/gain",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue gain adjustment",
)
async def queue_gain(
    asset_id: uuid.UUID,
    request: GainRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    if request.channel_gains_db is not None and asset.channel_count is not None:
        if len(request.channel_gains_db) != asset.channel_count:
            raise ApplicationError(
                "invalid_channel_gains",
                "Provide one channel gain for each source channel.",
                422,
            )
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "gain",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/fade",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue fade-in and fade-out processing",
)
async def queue_fade(
    asset_id: uuid.UUID,
    request: FadeRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    if asset.duration_seconds is not None:
        if request.fade_in_seconds + request.fade_out_seconds > asset.duration_seconds:
            raise ApplicationError(
                "invalid_fade_duration", "Fade durations exceed source duration.", 422
            )
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "fade",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/filter",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a validated digital filter",
)
async def queue_filter(
    asset_id: uuid.UUID,
    request: FilterRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "filter",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/channels",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue channel conversion",
)
async def queue_channel_operation(
    asset_id: uuid.UUID,
    request: ChannelRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    if request.mode == "swap" and asset.channel_count not in {None, 2}:
        raise ApplicationError(
            "invalid_channel_operation", "Channel swapping requires stereo.", 422
        )
    if request.mode in {"left", "right"} and asset.channel_count == 1:
        raise ApplicationError(
            "invalid_channel_operation", "Channel extraction requires stereo.", 422
        )
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "channels",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/resample",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue sample-rate conversion",
)
async def queue_resample(
    asset_id: uuid.UUID,
    request: ResampleRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(
        session,
        owner_id,
        asset.id,
        "resample",
        request.model_dump(mode="json"),
        idempotency_key,
        response,
    )


@router.post(
    "/{asset_id}/quality",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue signal quality measurements",
)
async def queue_quality(
    asset_id: uuid.UUID,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    return await _queue_dsp(session, owner_id, asset.id, "quality", {}, idempotency_key, response)


@router.get(
    "/{asset_id}/quality",
    response_model=dict[str, object],
    summary="Retrieve the latest completed signal quality report",
)
async def get_audio_quality(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
) -> dict[str, object]:
    await _owned_audio_asset(storage, session, owner_id, asset_id)
    record = await session.scalar(
        select(AudioProcessingRecord)
        .where(
            AudioProcessingRecord.source_asset_id == asset_id,
            AudioProcessingRecord.operation == "quality",
            AudioProcessingRecord.status == "succeeded",
        )
        .order_by(AudioProcessingRecord.created_at.desc())
        .limit(1)
    )
    if record is None or record.result is None:
        raise ApplicationError(
            "quality_report_not_found",
            "No quality analysis has completed for this audio asset.",
            404,
        )
    return record.result


@router.get(
    "/{asset_id}/processing-history",
    response_model=list[ProcessingHistoryResponse],
    summary="List processing history for an audio asset",
)
async def processing_history(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
) -> list[ProcessingHistoryResponse]:
    await _owned_audio_asset(storage, session, owner_id, asset_id)
    records = list(
        (
            await session.scalars(
                select(AudioProcessingRecord)
                .where(AudioProcessingRecord.source_asset_id == asset_id)
                .order_by(AudioProcessingRecord.created_at.desc())
                .limit(100)
            )
        ).all()
    )
    responses: list[ProcessingHistoryResponse] = []
    for record in records:
        outputs = list(
            (
                await session.scalars(
                    select(AudioProcessingOutput)
                    .where(AudioProcessingOutput.processing_record_id == record.id)
                    .order_by(AudioProcessingOutput.position)
                )
            ).all()
        )
        responses.append(
            ProcessingHistoryResponse(
                id=record.id,
                job_id=record.job_id,
                source_asset_id=record.source_asset_id,
                operation=record.operation,
                parameters=record.parameters,
                status=record.status,
                processing_version=record.processing_version,
                started_at=record.started_at,
                completed_at=record.completed_at,
                duration_ms=record.duration_ms,
                error_code=record.error_code,
                error_message=record.error_message,
                result=record.result,
                outputs=[
                    {
                        "asset_id": output.output_asset_id,
                        "position": output.position,
                        "metadata": output.output_metadata,
                    }
                    for output in outputs
                ],
            )
        )
    return responses


@router.post(
    "/{asset_id}/inspect",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue audio inspection",
)
async def inspect_audio(
    asset_id: uuid.UUID,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    _require_worker_broker(resources.settings.redis_enabled)
    job = await _create_audio_job(
        session,
        owner_id,
        asset.id,
        "inspect",
        {},
        idempotency_key,
        response,
    )
    return JobResponse.model_validate(job)


@router.get(
    "/{asset_id}/metadata",
    response_model=AudioMetadataResponse,
    summary="Get extracted audio metadata",
)
async def get_audio_metadata(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
) -> AudioMetadataResponse:
    await _owned_audio_asset(storage, session, owner_id, asset_id)
    metadata = await session.get(AudioMetadata, asset_id)
    if metadata is None:
        raise ApplicationError(
            "audio_metadata_not_found",
            "Audio metadata has not been extracted yet; submit an inspection job.",
            404,
        )
    records = list(
        (
            await session.scalars(
                select(AudioArtwork).where(AudioArtwork.audio_asset_id == asset_id)
            )
        ).all()
    )
    artworks = [
        AudioArtworkResponse(
            asset_id=record.artwork_asset_id,
            mime_type=record.mime_type,
            picture_type=record.picture_type,
            description=record.description,
        )
        for record in records
    ]
    return AudioMetadataResponse(
        audio_asset_id=metadata.audio_asset_id,
        tags=metadata.tags,
        raw_tags=metadata.raw_tags,
        technical=metadata.technical,
        source=metadata.source,
        mutagen_version=metadata.mutagen_version,
        ffprobe_version=metadata.ffprobe_version,
        extracted_at=metadata.extracted_at,
        artworks=artworks,
    )


@router.post(
    "/{asset_id}/convert",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue audio conversion",
)
async def convert_audio(
    asset_id: uuid.UUID,
    request: ConversionRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_audio_asset(storage, session, owner_id, asset_id)
    if (
        request.trim_end_seconds is not None
        and asset.duration_seconds is not None
        and request.trim_end_seconds > asset.duration_seconds
    ):
        raise ApplicationError(
            "invalid_conversion_options",
            "trim_end_seconds cannot exceed the source duration.",
            422,
        )
    _require_worker_broker(resources.settings.redis_enabled)
    options = request.model_dump(mode="json")
    if request.profile == "music_analysis_wav" and request.sample_rate is None:
        options["sample_rate"] = 44100
    job = await _create_audio_job(
        session,
        owner_id,
        asset.id,
        "convert",
        options,
        idempotency_key,
        response,
    )
    return JobResponse.model_validate(job)


@router.get(
    "/{asset_id}/assets",
    response_model=AudioAssetsResponse,
    summary="List derived audio assets",
)
async def list_audio_assets(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    storage: Storage,
) -> AudioAssetsResponse:
    await _owned_audio_asset(storage, session, owner_id, asset_id)
    assets = list(
        (
            await session.scalars(
                select(AudioAsset)
                .where(
                    AudioAsset.owner_id == owner_id,
                    AudioAsset.parent_asset_id == asset_id,
                    AudioAsset.asset_type.in_(
                        {
                            AssetType.CONVERTED_AUDIO.value,
                            AssetType.NORMALIZED_AUDIO.value,
                            AssetType.TRIMMED_AUDIO.value,
                            AssetType.PROCESSED_AUDIO.value,
                            AssetType.GENERATED_AUDIO.value,
                            AssetType.SEPARATED_STEM.value,
                        }
                    ),
                    AudioAsset.status != AssetStatus.DELETED.value,
                )
                .order_by(AudioAsset.created_at, AudioAsset.id)
            )
        ).all()
    )
    return AudioAssetsResponse(
        source_asset_id=asset_id,
        assets=[AssetResponse.model_validate(asset) for asset in assets],
    )


async def _owned_audio_asset(
    storage: StorageService,
    session: AsyncSession,
    owner_id: uuid.UUID,
    asset_id: uuid.UUID,
) -> AudioAsset:
    try:
        asset = await storage.get_asset(session, owner_id, asset_id)
    except AssetNotFoundError as exc:
        raise ApplicationError("asset_not_found", str(exc), 404) from exc
    if asset.asset_type not in _AUDIO_ASSET_TYPES:
        raise ApplicationError("not_an_audio_asset", "The requested asset is not audio.", 415)
    if asset.status != AssetStatus.AVAILABLE.value:
        raise ApplicationError("asset_not_available", "The audio asset is not available.", 409)
    return asset


def _require_worker_broker(redis_enabled: bool) -> None:
    if not redis_enabled:
        raise ApplicationError(
            "audio_workers_unavailable",
            "Audio background processing requires the configured Redis task broker.",
            503,
        )


async def _create_audio_job(
    session: AsyncSession,
    owner_id: uuid.UUID,
    asset_id: uuid.UUID,
    task_type: str,
    options: dict[str, object],
    idempotency_key: str | None,
    response: Response,
) -> ProcessingJob:
    try:
        job, created = await create_job(
            session,
            owner_id,
            asset_id,
            task_type,
            options,
            idempotency_key,
        )
    except JobConflictError as exc:
        raise ApplicationError("job_conflict", str(exc), 409) from exc
    except ValueError as exc:
        raise ApplicationError("invalid_idempotency_key", str(exc), 422) from exc
    if created:
        response.headers["Location"] = f"/api/v1/jobs/{job.id}"
        logger.info(
            "Accepted audio processing job",
            extra={"job_id": str(job.id), "task_type": task_type},
        )
    return job


async def _queue_dsp(
    session: AsyncSession,
    owner_id: uuid.UUID,
    asset_id: uuid.UUID,
    operation: str,
    options: dict[str, object],
    idempotency_key: str | None,
    response: Response,
) -> JobResponse:
    job = await _create_audio_job(
        session,
        owner_id,
        asset_id,
        f"dsp_{operation}",
        options,
        idempotency_key,
        response,
    )
    return JobResponse.model_validate(job)
