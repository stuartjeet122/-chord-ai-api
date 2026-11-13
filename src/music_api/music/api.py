from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from music_api.api.dependencies import (
    get_db_session,
    get_owner_id,
    get_resources,
    get_storage_service,
)
from music_api.audio.schemas import JobResponse
from music_api.errors import ApplicationError
from music_api.jobs.service import JobConflictError, create_job
from music_api.music.models import (
    MetronomeConfiguration,
    MusicAnalysisEvent,
    MusicAnalysisRecord,
    PracticePlan,
    PracticeSession,
    TempoMap,
)
from music_api.music.rhythm import (
    build_schedule_window,
    build_tempo_plan,
    build_variable_tempo_map,
    generate_tempo_ramp,
    validate_bpm,
)
from music_api.music.schemas import (
    AnalysisComparisonResponse,
    AnalysisRequest,
    AnalysisResponse,
    ChordEventResponse,
    ChordListResponse,
    MetronomeScheduleRequest,
    MetronomeScheduleResponse,
    MetronomeRequest,
    PracticePlanRequest,
    PracticeSessionRequest,
    PracticeSessionResponse,
    PracticeSessionUpdateRequest,
    TempoMapRequest,
)
from music_api.resources import RuntimeResources
from music_api.storage.models import AssetStatus, AssetType, AudioAsset
from music_api.storage.service import StorageService

router = APIRouter(prefix="/music", tags=["Music analysis"])
OwnerId = Annotated[uuid.UUID, Depends(get_owner_id)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
Resources = Annotated[RuntimeResources, Depends(get_resources)]
Storage = Annotated[StorageService, Depends(get_storage_service)]

_AUDIO_TYPES = {
    AssetType.ORIGINAL_AUDIO.value,
    AssetType.CONVERTED_AUDIO.value,
    AssetType.NORMALIZED_AUDIO.value,
    AssetType.TRIMMED_AUDIO.value,
    AssetType.PROCESSED_AUDIO.value,
    AssetType.GENERATED_AUDIO.value,
    AssetType.SEPARATED_STEM.value,
}


@router.post(
    "/{asset_id}/analyze",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a new versioned music analysis",
)
async def queue_analysis(
    asset_id: uuid.UUID,
    request: AnalysisRequest,
    response: Response,
    owner_id: OwnerId,
    session: DatabaseSession,
    resources: Resources,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JobResponse:
    asset = await _owned_asset(session, owner_id, asset_id)
    if not resources.settings.redis_enabled:
        raise ApplicationError(
            "background_worker_unavailable",
            "Music analysis requires the configured background worker and broker.",
            503,
        )
    await session.scalar(select(AudioAsset.id).where(AudioAsset.id == asset.id).with_for_update())
    options = {
        "tasks": request.tasks,
        "parameters": request.parameters,
        "source_sha256": asset.sha256,
    }
    current_version = await session.scalar(
        select(func.max(MusicAnalysisRecord.version)).where(
            MusicAnalysisRecord.owner_id == owner_id,
            MusicAnalysisRecord.source_asset_id == asset.id,
        )
    )
    version = int(current_version or 0) + 1
    try:
        job, created = await create_job(
            session,
            owner_id,
            asset.id,
            "music_analysis",
            options,
            idempotency_key,
            commit=False,
        )
    except JobConflictError as exc:
        raise ApplicationError("job_conflict", str(exc), 409) from exc
    except ValueError as exc:
        raise ApplicationError("invalid_idempotency_key", str(exc), 422) from exc
    if created:
        analysis = MusicAnalysisRecord(
            owner_id=owner_id,
            source_asset_id=asset.id,
            job_id=job.id,
            version=version,
            status="queued",
            tasks=request.tasks,
            source_sha256=asset.sha256,
            engine_version="dsp-harmonic-0.1.0",
            model_references=[],
            execution_device="cpu",
            parameters=request.parameters,
            stage_errors=[],
        )
        session.add(analysis)
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise ApplicationError(
                "analysis_version_conflict",
                "Another analysis version was created concurrently; retry this request.",
                409,
            ) from exc
        response.headers["Location"] = f"/api/v1/jobs/{job.id}"
    return JobResponse.model_validate(job)


@router.get(
    "/{asset_id}/analysis/compare",
    response_model=AnalysisComparisonResponse,
    summary="Compare completed analysis versions for an asset",
)
async def compare_analyses(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> AnalysisComparisonResponse:
    await _owned_asset(session, owner_id, asset_id)
    rows = list(
        (
            await session.scalars(
                select(MusicAnalysisRecord)
                .where(
                    MusicAnalysisRecord.owner_id == owner_id,
                    MusicAnalysisRecord.source_asset_id == asset_id,
                    MusicAnalysisRecord.status.in_(("succeeded", "partial")),
                )
                .order_by(MusicAnalysisRecord.version.desc())
                .limit(limit)
            )
        ).all()
    )
    return AnalysisComparisonResponse(
        asset_id=asset_id,
        analyses=[AnalysisResponse.model_validate(row) for row in rows],
    )


@router.get(
    "/{asset_id}/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get a specific completed analysis version",
)
async def get_analysis_version(
    asset_id: uuid.UUID,
    analysis_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> AnalysisResponse:
    analysis = await _analysis_for_asset(session, owner_id, asset_id, analysis_id)
    return AnalysisResponse.model_validate(analysis)


@router.get(
    "/{asset_id}/analysis",
    response_model=AnalysisResponse,
    summary="Get the latest completed analysis",
)
async def get_latest_analysis(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> AnalysisResponse:
    await _owned_asset(session, owner_id, asset_id)
    analysis = await session.scalar(
        select(MusicAnalysisRecord)
        .where(
            MusicAnalysisRecord.owner_id == owner_id,
            MusicAnalysisRecord.source_asset_id == asset_id,
            MusicAnalysisRecord.status.in_(("succeeded", "partial")),
        )
        .order_by(MusicAnalysisRecord.version.desc())
        .limit(1)
    )
    if analysis is None:
        raise ApplicationError("analysis_not_found", "No completed analysis was found.", 404)
    return AnalysisResponse.model_validate(analysis)


@router.get(
    "/{asset_id}/chords",
    response_model=ChordListResponse,
    summary="Retrieve timestamped chord events",
)
async def list_chords(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    start_seconds: Annotated[float | None, Query(ge=0)] = None,
    end_seconds: Annotated[float | None, Query(gt=0)] = None,
    min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    chord_type: Annotated[str | None, Query(max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChordListResponse:
    if start_seconds is not None and end_seconds is not None and end_seconds <= start_seconds:
        raise ApplicationError(
            "invalid_time_range", "end_seconds must be greater than start_seconds.", 422
        )
    await _owned_asset(session, owner_id, asset_id)
    analysis = await _latest_analysis_row(session, owner_id, asset_id)
    conditions = [
        MusicAnalysisEvent.analysis_id == analysis.id,
        MusicAnalysisEvent.event_type == "chord",
    ]
    if start_seconds is not None:
        conditions.append(MusicAnalysisEvent.end_seconds > start_seconds)
    if end_seconds is not None:
        conditions.append(MusicAnalysisEvent.start_seconds < end_seconds)
    if min_confidence is not None:
        conditions.append(MusicAnalysisEvent.confidence >= min_confidence)
    if chord_type is not None:
        conditions.append(MusicAnalysisEvent.quality == chord_type)
    count_query = select(func.count()).select_from(MusicAnalysisEvent).where(*conditions)
    total = int(await session.scalar(count_query) or 0)
    events = list(
        (
            await session.scalars(
                select(MusicAnalysisEvent)
                .where(*conditions)
                .order_by(MusicAnalysisEvent.start_seconds)
                .limit(limit)
                .offset(offset)
            )
        ).all()
    )
    items = [
        ChordEventResponse(
            event_id=event.id,
            symbol=event.label or "N",
            root=event.root,
            quality=event.quality or "unknown",
            inversion=event.evidence.get("inversion"),
            bass_note=event.evidence.get("bass_note"),
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds or event.start_seconds,
            duration_seconds=(event.end_seconds or event.start_seconds) - event.start_seconds,
            confidence=None,
            confidence_status=str(event.evidence.get("confidence_status", "unknown")),
            sources=list(event.evidence.get("sources", [])),
            evidence=event.evidence,
            roman_numeral=event.evidence.get("roman_numeral"),
            manually_corrected=event.manually_corrected,
        )
        for event in events
    ]
    return ChordListResponse(
        analysis_id=analysis.id,
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{asset_id}/tempo", summary="Retrieve tempo and beat map")
async def get_tempo(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    analysis = await _analysis_for_asset(session, owner_id, asset_id)
    return _result_section(analysis, "rhythm")


@router.get("/{asset_id}/key", summary="Retrieve key candidates and changes")
async def get_key(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    analysis = await _analysis_for_asset(session, owner_id, asset_id)
    return _result_section(analysis, "key")


@router.get("/{asset_id}/sections", summary="Retrieve song sections when available")
async def get_sections(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    analysis = await _analysis_for_asset(session, owner_id, asset_id)
    return _result_section(analysis, "sections")


@router.get("/{asset_id}/bass", summary="Retrieve bass-note events")
async def get_bass(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    analysis = await _analysis_for_asset(session, owner_id, asset_id)
    return {"items": analysis.result.get("bass_events", []) if analysis.result else []}


@router.get("/{asset_id}/melody", summary="Retrieve melody notes when available")
async def get_melody(
    asset_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    analysis = await _analysis_for_asset(session, owner_id, asset_id)
    return _result_section(analysis, "melody")


@router.post("/metronome/schedule", response_model=MetronomeScheduleResponse, summary="Generate a deterministic metronome schedule")
async def generate_metronome_schedule(
    request: MetronomeScheduleRequest,
) -> MetronomeScheduleResponse:
    schedule = build_schedule_window(
        bpm=request.bpm,
        time_signature=request.time_signature,
        duration_seconds=request.duration_seconds,
        subdivisions=request.subdivision,
        swing_ratio=request.swing_ratio,
        accent_pattern=request.accent_pattern,
        count_in_bars=request.count_in_bars,
        max_events=request.max_events,
    )
    return MetronomeScheduleResponse(
        events=[
            {
                "timestamp_seconds": float(event["timestamp_seconds"]),
                "measure_index": int(event["measure_index"]),
                "beat_index": int(event["beat_index"]),
                "subdivision_index": int(event["subdivision_index"]),
                "event_type": str(event["event_type"]),
                "accent_level": int(event["accent_level"]),
                "bpm": float(event["bpm"]),
                "phase": str(event.get("phase", "main")),
            }
            for event in schedule["events"]
        ],
        total_events=int(schedule["total_events"]),
        truncated=bool(schedule["truncated"]),
    )


@router.post("/metronome/configurations", summary="Create or update a named metronome configuration")
async def create_metronome_configuration(
    request: MetronomeRequest,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    validate_bpm(request.bpm)
    record = MetronomeConfiguration(
        owner_id=owner_id,
        name=request.name,
        bpm=request.bpm,
        numerator=int(request.time_signature.split("/", 1)[0]),
        denominator=int(request.time_signature.split("/", 1)[1]),
        beats_per_measure=int(request.time_signature.split("/", 1)[0]),
        beat_unit="quarter",
        subdivision=request.subdivision,
        swing_ratio=request.swing_ratio,
        accent_pattern=request.accent_pattern or [1],
        count_in_bars=request.count_in_bars,
        count_in_subdivision=request.count_in_subdivision,
        is_default=request.is_default,
        description=request.description,
        config_data={"time_signature": request.time_signature},
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return {
        "id": str(record.id),
        "owner_id": str(record.owner_id),
        "name": record.name,
        "bpm": record.bpm,
        "time_signature": request.time_signature,
        "subdivision": record.subdivision,
        "swing_ratio": record.swing_ratio,
        "count_in_bars": record.count_in_bars,
        "description": record.description,
    }


@router.get("/metronome/configurations/{config_id}", summary="Retrieve a metronome configuration")
async def get_metronome_configuration(
    config_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    record = await session.scalar(
        select(MetronomeConfiguration).where(
            MetronomeConfiguration.id == config_id,
            MetronomeConfiguration.owner_id == owner_id,
        )
    )
    if record is None:
        raise ApplicationError("metronome_configuration_not_found", "Metronome configuration was not found.", 404)
    return {
        "id": str(record.id),
        "name": record.name,
        "bpm": record.bpm,
        "time_signature": f"{record.numerator}/{record.denominator}",
        "subdivision": record.subdivision,
        "swing_ratio": record.swing_ratio,
        "count_in_bars": record.count_in_bars,
        "description": record.description,
        "metadata": record.config_data,
    }


@router.post("/tempo/maps", summary="Create a tempo map")
async def create_tempo_map(
    request: TempoMapRequest,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    payload = build_variable_tempo_map(
        base_bpm=request.base_bpm,
        tempo_points=request.tempo_points,
        duration_seconds=request.duration_seconds,
    )
    record = TempoMap(
        owner_id=owner_id,
        name=request.name,
        source_type=request.source_type,
        base_bpm=request.base_bpm,
        points=payload["tempo_points"],
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return {
        "id": str(record.id),
        "name": record.name,
        "source_type": record.source_type,
        "base_bpm": record.base_bpm,
        "points": record.points,
        "bpm_at_end": payload["bpm_at_end"],
    }


@router.post("/tempo/ramp", summary="Generate a tempo ramp")
async def tempo_ramp(
    start_bpm: float,
    end_bpm: float,
    duration_seconds: float,
    steps: int = 8,
) -> dict[str, Any]:
    return generate_tempo_ramp(
        start_bpm,
        end_bpm,
        duration_seconds=duration_seconds,
        steps=steps,
    )


@router.post("/practice/plans", summary="Create a practice plan")
async def create_practice_plan(
    request: PracticePlanRequest,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> dict[str, Any]:
    stages = build_tempo_plan(
        start_bpm=request.start_bpm,
        target_bpm=request.target_bpm,
        stage_duration_seconds=request.stage_duration_seconds,
        stage_count=request.stage_count,
    )
    record = PracticePlan(
        owner_id=owner_id,
        name=request.name,
        description=request.name,
        start_bpm=request.start_bpm,
        target_bpm=request.target_bpm,
        stages=stages,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return {
        "id": str(record.id),
        "name": record.name,
        "stages": record.stages,
        "start_bpm": record.start_bpm,
        "target_bpm": record.target_bpm,
    }


@router.post("/practice/sessions", response_model=PracticeSessionResponse, summary="Create a practice session")
async def create_practice_session(
    request: PracticeSessionRequest,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> PracticeSessionResponse:
    record = PracticeSession(
        owner_id=owner_id,
        song_id=request.song_id,
        start_bpm=request.start_bpm,
        target_bpm=request.target_bpm,
        time_signature=request.time_signature,
        notes=request.notes,
        session_data=request.metadata,
        status="planned",
        start_seconds=0.0,
        duration_seconds=0.0,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return PracticeSessionResponse.model_validate(record)


@router.get("/practice/sessions/{session_id}", response_model=PracticeSessionResponse, summary="Retrieve a practice session")
async def get_practice_session(
    session_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> PracticeSessionResponse:
    record = await session.scalar(
        select(PracticeSession).where(
            PracticeSession.id == session_id,
            PracticeSession.owner_id == owner_id,
        )
    )
    if record is None:
        raise ApplicationError("practice_session_not_found", "Practice session was not found.", 404)
    return PracticeSessionResponse.model_validate(record)


@router.patch("/practice/sessions/{session_id}", response_model=PracticeSessionResponse, summary="Update a practice session")
async def update_practice_session(
    session_id: uuid.UUID,
    request: PracticeSessionUpdateRequest,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> PracticeSessionResponse:
    record = await session.scalar(
        select(PracticeSession).where(
            PracticeSession.id == session_id,
            PracticeSession.owner_id == owner_id,
        )
    )
    if record is None:
        raise ApplicationError("practice_session_not_found", "Practice session was not found.", 404)
    if request.status is not None:
        record.status = request.status
    if request.start_seconds is not None:
        record.start_seconds = request.start_seconds
    if request.end_seconds is not None:
        record.end_seconds = request.end_seconds
    if request.duration_seconds is not None:
        record.duration_seconds = request.duration_seconds
    if request.notes is not None:
        record.notes = request.notes
    if request.metadata is not None:
        record.session_data = request.metadata
    await session.commit()
    await session.refresh(record)
    return PracticeSessionResponse.model_validate(record)


async def _owned_asset(
    session: AsyncSession,
    owner_id: uuid.UUID,
    asset_id: uuid.UUID,
) -> AudioAsset:
    asset = await session.scalar(
        select(AudioAsset).where(
            AudioAsset.id == asset_id,
            AudioAsset.owner_id == owner_id,
            AudioAsset.status == AssetStatus.AVAILABLE.value,
            AudioAsset.asset_type.in_(_AUDIO_TYPES),
        )
    )
    if asset is None:
        raise ApplicationError("asset_not_found", "Available audio asset was not found.", 404)
    return asset


async def _latest_analysis_row(
    session: AsyncSession,
    owner_id: uuid.UUID,
    asset_id: uuid.UUID,
) -> MusicAnalysisRecord:
    row = await session.scalar(
        select(MusicAnalysisRecord)
        .where(
            MusicAnalysisRecord.owner_id == owner_id,
            MusicAnalysisRecord.source_asset_id == asset_id,
            MusicAnalysisRecord.status.in_(("succeeded", "partial")),
        )
        .order_by(MusicAnalysisRecord.version.desc())
        .limit(1)
    )
    if row is None:
        raise ApplicationError("analysis_not_found", "No completed analysis was found.", 404)
    return row


async def _analysis_for_asset(
    session: AsyncSession,
    owner_id: uuid.UUID,
    asset_id: uuid.UUID,
    analysis_id: uuid.UUID | None = None,
) -> MusicAnalysisRecord:
    conditions = [
        MusicAnalysisRecord.owner_id == owner_id,
        MusicAnalysisRecord.source_asset_id == asset_id,
        MusicAnalysisRecord.status.in_(("succeeded", "partial")),
    ]
    if analysis_id is not None:
        conditions.append(MusicAnalysisRecord.id == analysis_id)
        row = await session.scalar(select(MusicAnalysisRecord).where(*conditions))
        if row is None:
            raise ApplicationError("analysis_not_found", "Analysis version was not found.", 404)
        return row
    return await _latest_analysis_row(session, owner_id, asset_id)


def _result_section(analysis: MusicAnalysisRecord, section: str) -> dict[str, Any]:
    if analysis.result is None or section not in analysis.result:
        return {
            "status": "unavailable",
            "reason": f"The '{section}' task was not included in this analysis version.",
        }
    value = analysis.result[section]
    return value if isinstance(value, dict) else {"items": value}
