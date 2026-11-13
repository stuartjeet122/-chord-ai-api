from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from music_api.api.dependencies import get_db_session, get_owner_id
from music_api.errors import ApplicationError
from music_api.music import theory
from music_api.music.models import MusicAnalysisRecord, MusicTheoryRecord
from music_api.music.theory import (
    analyze_progression,
    capo_calculation,
    chord_inversion,
    compare_progressions,
    identify_interval,
    identify_scales,
    key_signature,
    parse_chord,
    parse_key,
    parse_note,
    scale_information,
    transpose_between_keys,
    transpose_key,
    transpose_progression,
)
from music_api.music.theory_schemas import (
    CapoRequest,
    CapoResponse,
    ChordConstructionRequest,
    ChordInversionRequest,
    ChordInversionResponse,
    ChordParseRequest,
    ChordResponse,
    HarmonyRequest,
    HarmonyResponse,
    IntervalRequest,
    IntervalResponse,
    KeyAwareTranspositionRequest,
    KeyAwareTranspositionResponse,
    KeyRequest,
    KeyResponse,
    KeyTranspositionRequest,
    KeyTranspositionResponse,
    MelodyHarmonyRequest,
    MelodyHarmonyResponse,
    NoteRequest,
    NoteResponse,
    ProgressionComparisonRequest,
    ProgressionComparisonResponse,
    ProgressionRequest,
    ProgressionResponse,
    ScaleIdentificationRequest,
    ScaleIdentificationResponse,
    ScaleRequest,
    ScaleResponse,
    TheoryInterpretationRequest,
    TheoryInterpretationResponse,
    TranspositionRequest,
    TranspositionResponse,
)
from music_api.storage.models import AssetStatus, AssetType, AudioAsset

theory_router = APIRouter(prefix="/theory", tags=["Music theory"])
analysis_theory_router = APIRouter(prefix="/music", tags=["Music theory"])

DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
OwnerId = Annotated[uuid.UUID, Depends(get_owner_id)]
ResultT = TypeVar("ResultT")

_AUDIO_TYPES = {
    AssetType.ORIGINAL_AUDIO.value,
    AssetType.CONVERTED_AUDIO.value,
    AssetType.NORMALIZED_AUDIO.value,
    AssetType.TRIMMED_AUDIO.value,
    AssetType.PROCESSED_AUDIO.value,
    AssetType.GENERATED_AUDIO.value,
    AssetType.SEPARATED_STEM.value,
}


def _calculate(operation: Callable[[], ResultT]) -> ResultT:  # noqa: UP047
    try:
        return operation()
    except ValueError as exc:
        raise ApplicationError("invalid_theory_input", str(exc), 422) from exc


def _chord_response(symbol: str) -> dict[str, Any]:
    return parse_chord(symbol).as_dict()


def _normalize_secondary_dominant_labels(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict):
        for key, value in result.items():
            if key == "chords" and isinstance(value, list):
                for chord in value:
                    if isinstance(chord, dict):
                        numeral = chord.get("roman_numeral")
                        if isinstance(numeral, str) and numeral.startswith("V7/"):
                            chord["roman_numeral"] = f"V/{numeral.split('/', 1)[1]}"
            elif key == "interpretations" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _normalize_secondary_dominant_labels(item)
    return result


def _progression_result(
    symbols: list[str],
    key: str,
    timestamps: list[dict[str, float]] | None,
) -> dict[str, Any]:
    return _calculate(lambda: analyze_progression(symbols, key, timestamps=timestamps))


@theory_router.post("/notes/parse", response_model=NoteResponse, summary="Parse a spelled note")
async def parse_note_endpoint(request: NoteRequest) -> NoteResponse:
    parsed = _calculate(lambda: parse_note(request.note))
    return NoteResponse(
        spelling=parsed.name,
        pitch_class=parsed.pitch_class,
        octave=parsed.octave,
        midi=parsed.midi,
        frequency_hz=parsed.frequency_hz,
    )


@theory_router.post(
    "/intervals", response_model=IntervalResponse, summary="Identify a spelled interval"
)
async def interval_endpoint(request: IntervalRequest) -> IntervalResponse:
    interval = _calculate(lambda: identify_interval(request.from_pitch, request.to_pitch))
    return IntervalResponse(
        name=interval.name,
        number=interval.number,
        quality=interval.quality,
        semitones=interval.semitones,
        direction=interval.direction,
    )


@theory_router.post("/scales", response_model=ScaleResponse, summary="Construct a scale")
async def scale_endpoint(request: ScaleRequest) -> ScaleResponse:
    result = _calculate(
        lambda: scale_information(
            request.root,
            request.scale,
            None if request.pattern is None else tuple(request.pattern),
        )
    )
    return ScaleResponse.model_validate(result)


@theory_router.post(
    "/scales/identify",
    response_model=ScaleIdentificationResponse,
    summary="Return candidate scales for a note collection",
)
async def identify_scale_endpoint(
    request: ScaleIdentificationRequest,
) -> ScaleIdentificationResponse:
    return ScaleIdentificationResponse.model_validate(
        _calculate(lambda: identify_scales(request.notes))
    )


@theory_router.post(
    "/chords/parse", response_model=ChordResponse, summary="Parse and normalize a chord symbol"
)
async def parse_chord_endpoint(request: ChordParseRequest) -> ChordResponse:
    return ChordResponse.model_validate(_calculate(lambda: _chord_response(request.symbol)))


@theory_router.post(
    "/chords/construct",
    response_model=ChordResponse,
    summary="Construct a chord from root and quality",
)
async def construct_chord_endpoint(request: ChordConstructionRequest) -> ChordResponse:
    result = _calculate(lambda: theory.construct_chord(request.root, request.quality).as_dict())
    return ChordResponse.model_validate(result)


@theory_router.post(
    "/chords/inversion",
    response_model=ChordInversionResponse,
    summary="Identify a chord inversion without inferring instrument voicing",
)
async def inversion_endpoint(request: ChordInversionRequest) -> ChordInversionResponse:
    result = _calculate(lambda: chord_inversion(parse_chord(request.symbol), request.bass))
    return ChordInversionResponse.model_validate(result)


@theory_router.post(
    "/keys", response_model=KeyResponse, summary="Return key signature and relatives"
)
async def key_endpoint(request: KeyRequest) -> KeyResponse:
    return KeyResponse.model_validate(_calculate(lambda: key_signature(request.key)))


@theory_router.post(
    "/keys/transpose",
    response_model=KeyTranspositionResponse,
    summary="Transpose a key and return the target signature",
)
async def transpose_key_endpoint(request: KeyTranspositionRequest) -> KeyTranspositionResponse:
    target_key = _calculate(
        lambda: transpose_key(request.key, request.semitones, prefer_flats=request.prefer_flats)
    )
    signature = _calculate(lambda: key_signature(target_key))
    return KeyTranspositionResponse(
        source_key=request.key,
        semitones=request.semitones,
        target_key=target_key,
        target_key_signature=KeyResponse.model_validate(signature),
    )


@theory_router.post(
    "/harmony", response_model=HarmonyResponse, summary="Generate diatonic triads or seventh chords"
)
async def harmony_endpoint(request: HarmonyRequest) -> HarmonyResponse:
    chords = _calculate(lambda: theory.diatonic_harmony(request.key, sevenths=request.sevenths))
    return HarmonyResponse(key=request.key, chords=chords)


@theory_router.post(
    "/harmony/melody",
    response_model=MelodyHarmonyResponse,
    summary="List possible diatonic harmonizations for one melody note",
)
async def melody_harmony_endpoint(request: MelodyHarmonyRequest) -> MelodyHarmonyResponse:
    result = _calculate(
        lambda: theory.harmonize_melody_note(request.key, request.note, sevenths=request.sevenths)
    )
    return MelodyHarmonyResponse.model_validate(result)


@theory_router.post(
    "/progressions/analyze",
    response_model=ProgressionResponse,
    summary="Analyze a chord progression against an explicit key",
)
async def progression_endpoint(request: ProgressionRequest) -> ProgressionResponse:
    timestamps = (
        None if request.timestamps is None else [item.model_dump() for item in request.timestamps]
    )
    result = _calculate(
        lambda: analyze_progression(
            request.chord_symbols,
            request.key,
            timestamps=timestamps,
            soprano_notes=request.soprano_notes,
        )
    )
    return ProgressionResponse.model_validate(result)


@theory_router.post(
    "/progressions/compare",
    response_model=ProgressionComparisonResponse,
    summary="Compare progression symbols and transposition-equivalent structure",
)
async def compare_progressions_endpoint(
    request: ProgressionComparisonRequest,
) -> ProgressionComparisonResponse:
    result = _calculate(
        lambda: compare_progressions(
            request.first_chords,
            request.second_chords,
            key=request.key,
        )
    )
    return ProgressionComparisonResponse.model_validate(result)


@theory_router.post(
    "/transpose",
    response_model=TranspositionResponse,
    summary="Transpose chords by signed semitones",
)
async def transpose_endpoint(request: TranspositionRequest) -> TranspositionResponse:
    result = _calculate(
        lambda: transpose_progression(
            request.chord_symbols,
            request.semitones,
            prefer_flats=request.prefer_flats,
            key=request.key,
        )
    )
    return TranspositionResponse.model_validate(result)


@theory_router.post(
    "/transpose-between-keys",
    response_model=KeyAwareTranspositionResponse,
    summary="Transpose a progression from one key to another",
)
async def transpose_between_keys_endpoint(
    request: KeyAwareTranspositionRequest,
) -> KeyAwareTranspositionResponse:
    result = _calculate(
        lambda: transpose_between_keys(
            request.chord_symbols,
            request.source_key,
            request.target_key,
            prefer_flats=request.prefer_flats,
        )
    )
    return KeyAwareTranspositionResponse.model_validate(result)


@theory_router.post(
    "/capo", response_model=CapoResponse, summary="Convert written and sounding chords"
)
async def capo_endpoint(request: CapoRequest) -> CapoResponse:
    result = _calculate(
        lambda: capo_calculation(
            request.chord_symbols,
            request.capo_fret,
            direction=request.direction,
            prefer_flats=request.prefer_flats,
            key=request.key,
        )
    )
    return CapoResponse.model_validate(result)


@analysis_theory_router.post(
    "/{asset_id}/analysis/{analysis_id}/theory",
    response_model=TheoryInterpretationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Persist a separate versioned theory interpretation",
)
async def create_theory_interpretation(
    asset_id: uuid.UUID,
    analysis_id: uuid.UUID,
    request: TheoryInterpretationRequest,
    owner_id: OwnerId,
    session: DatabaseSession,
) -> TheoryInterpretationResponse:
    await _owned_audio(session, owner_id, asset_id)
    analysis = await session.scalar(
        select(MusicAnalysisRecord)
        .where(
            MusicAnalysisRecord.id == analysis_id,
            MusicAnalysisRecord.owner_id == owner_id,
            MusicAnalysisRecord.source_asset_id == asset_id,
            MusicAnalysisRecord.status.in_(("succeeded", "partial")),
        )
        .with_for_update()
    )
    if analysis is None:
        raise ApplicationError(
            "analysis_not_found", "Completed analysis version was not found.", 404
        )

    provided = request.chord_symbols is not None
    if provided:
        symbols = request.chord_symbols or []
        timestamps = None
    else:
        raw_events = (analysis.result or {}).get("chords", [])
        valid_events = [
            item
            for item in raw_events
            if isinstance(item, dict)
            and isinstance(item.get("symbol"), str)
            and item["symbol"] != "N"
        ]
        symbols = [item["symbol"] for item in valid_events]
        timestamps = [
            {
                "start_seconds": float(item["start_seconds"]),
                "end_seconds": float(item["end_seconds"]),
            }
            for item in valid_events
            if isinstance(item.get("start_seconds"), int | float)
            and isinstance(item.get("end_seconds"), int | float)
            and item["end_seconds"] > item["start_seconds"]
        ]
        if len(timestamps) != len(symbols):
            timestamps = None
    if not symbols:
        raise ApplicationError(
            "chords_unavailable",
            "The selected analysis version contains no supported detected chord events.",
            422,
        )

    selected_key = request.key
    key_candidates: list[str] = []
    if selected_key is None:
        key_result = (analysis.result or {}).get("key", {})
        selected = key_result.get("selected") if isinstance(key_result, dict) else None
        if (
            isinstance(selected, dict)
            and isinstance(selected.get("tonic"), str)
            and isinstance(selected.get("mode"), str)
        ):
            selected_key = f"{selected['tonic']} {selected['mode']}"
        else:
            candidates = []
            if isinstance(key_result, dict):
                best = key_result.get("best_candidate")
                alternatives = key_result.get("alternatives", [])
                if isinstance(best, dict):
                    candidates.append(best)
                if isinstance(alternatives, list):
                    candidates.extend(item for item in alternatives if isinstance(item, dict))
            for candidate in candidates:
                tonic = candidate.get("tonic")
                mode = candidate.get("mode")
                if isinstance(tonic, str) and isinstance(mode, str):
                    candidate_key = f"{tonic} {mode}"
                    if candidate_key not in key_candidates:
                        key_candidates.append(candidate_key)

    key_context: dict[str, str] | None = None
    if selected_key is None:
        result: dict[str, Any] = {
            "status": "ambiguous" if key_candidates else "unavailable",
            "reason": (
                "The source analysis did not select a key; each candidate is reported separately."
                if key_candidates
                else "No selected or candidate key is available in the source analysis."
            ),
            "chord_symbols": symbols,
            "interpretations": [
                _progression_result(symbols, candidate, timestamps) for candidate in key_candidates
            ],
        }
    else:
        tonic, mode = _calculate(lambda: parse_key(selected_key))
        key_context = {"tonic": tonic.name, "mode": mode}
        result = _progression_result(symbols, selected_key, timestamps)
        if provided:
            result = _normalize_secondary_dominant_labels(result)
        result["status"] = "interpreted"
        result["source_chords"] = "provided_chords" if provided else "detected_chords"

    current_version = await session.scalar(
        select(func.max(MusicTheoryRecord.version)).where(
            MusicTheoryRecord.analysis_id == analysis_id
        )
    )
    record = MusicTheoryRecord(
        analysis_id=analysis_id,
        owner_id=owner_id,
        version=int(current_version or 0) + 1,
        key_context=key_context,
        source="provided_chords" if provided else "detected_chords",
        result=result,
        created_at=datetime.now(UTC),
    )
    session.add(record)
    try:
        await session.commit()
        await session.refresh(record)
    except IntegrityError as exc:
        await session.rollback()
        raise ApplicationError(
            "theory_version_conflict",
            "Another theory interpretation was created concurrently; retry this request.",
            409,
        ) from exc
    return TheoryInterpretationResponse.model_validate(record)


@analysis_theory_router.get(
    "/{asset_id}/analysis/{analysis_id}/theory",
    response_model=TheoryInterpretationResponse,
    summary="Retrieve the latest theory interpretation for an analysis",
)
async def get_theory_interpretation(
    asset_id: uuid.UUID,
    analysis_id: uuid.UUID,
    owner_id: OwnerId,
    session: DatabaseSession,
    version: Annotated[int | None, Query(ge=1)] = None,
) -> TheoryInterpretationResponse:
    await _owned_audio(session, owner_id, asset_id)
    conditions = [
        MusicTheoryRecord.analysis_id == analysis_id,
        MusicTheoryRecord.owner_id == owner_id,
    ]
    if version is not None:
        conditions.append(MusicTheoryRecord.version == version)
    record = await session.scalar(
        select(MusicTheoryRecord)
        .join(MusicAnalysisRecord, MusicAnalysisRecord.id == MusicTheoryRecord.analysis_id)
        .where(
            *conditions,
            MusicAnalysisRecord.source_asset_id == asset_id,
            MusicAnalysisRecord.owner_id == owner_id,
        )
        .order_by(MusicTheoryRecord.version.desc())
        .limit(1)
    )
    if record is None:
        raise ApplicationError(
            "theory_analysis_not_found", "Theory interpretation was not found.", 404
        )
    return TheoryInterpretationResponse.model_validate(record)


async def _owned_audio(
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
