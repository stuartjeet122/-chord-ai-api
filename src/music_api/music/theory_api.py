from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter

from music_api.errors import ApplicationError
from music_api.music import theory
from music_api.music.theory import (
    analyze_progression,
    capo_calculation,
    compare_progressions,
    identify_interval,
    key_signature,
    parse_chord,
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
    ChordParseRequest,
    ChordResponse,
    IntervalRequest,
    IntervalResponse,
    KeyAwareTranspositionRequest,
    KeyAwareTranspositionResponse,
    KeyRequest,
    KeyResponse,
    KeyTranspositionRequest,
    KeyTranspositionResponse,
    NoteRequest,
    NoteResponse,
    ProgressionComparisonRequest,
    ProgressionComparisonResponse,
    ProgressionRequest,
    ProgressionResponse,
    ScaleRequest,
    ScaleResponse,
    TranspositionRequest,
    TranspositionResponse,
)

theory_router = APIRouter(prefix="/theory", tags=["Music theory"])
ResultT = TypeVar("ResultT")


def _calculate(operation: Callable[[], ResultT]) -> ResultT:  # noqa: UP047
    try:
        return operation()
    except ValueError as exc:
        raise ApplicationError("invalid_theory_input", str(exc), 422) from exc


@theory_router.post(
    "/notes/parse",
    response_model=NoteResponse,
    summary="Parse a musical note",
)
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
    "/intervals",
    response_model=IntervalResponse,
    summary="Identify the interval between two pitches",
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


@theory_router.post(
    "/scales",
    response_model=ScaleResponse,
    summary="Build a scale",
)
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
    "/chords/parse",
    response_model=ChordResponse,
    summary="Parse a chord symbol",
)
async def parse_chord_endpoint(request: ChordParseRequest) -> ChordResponse:
    return ChordResponse.model_validate(
        _calculate(lambda: parse_chord(request.symbol).as_dict())
    )


@theory_router.post(
    "/chords/construct",
    response_model=ChordResponse,
    summary="Construct a chord",
)
async def construct_chord_endpoint(request: ChordConstructionRequest) -> ChordResponse:
    result = _calculate(lambda: theory.construct_chord(request.root, request.quality).as_dict())
    return ChordResponse.model_validate(result)


@theory_router.post(
    "/keys",
    response_model=KeyResponse,
    summary="Get a key signature",
)
async def key_endpoint(request: KeyRequest) -> KeyResponse:
    return KeyResponse.model_validate(_calculate(lambda: key_signature(request.key)))


@theory_router.post(
    "/keys/transpose",
    response_model=KeyTranspositionResponse,
    summary="Transpose a key",
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
    "/progressions/analyze",
    response_model=ProgressionResponse,
    summary="Analyze a chord progression",
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
    summary="Compare two chord progressions",
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
    summary="Transpose a chord progression",
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
    summary="Transpose a progression between keys",
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
    "/capo",
    response_model=CapoResponse,
    summary="Calculate capo chord positions",
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
