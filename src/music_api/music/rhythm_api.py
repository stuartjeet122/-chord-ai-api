from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from music_api.errors import ApplicationError
from music_api.music.rhythm import build_schedule_window

router = APIRouter(prefix="/music", tags=["Rhythm"])


class MetronomeScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bpm: float = Field(..., gt=0, le=300, allow_inf_nan=False)
    time_signature: str = Field(default="4/4", min_length=3, max_length=16)
    duration_seconds: float = Field(..., gt=0, le=86_400, allow_inf_nan=False)
    subdivision: Literal["beat", "eighth", "sixteenth", "triplet"] = "eighth"
    swing_ratio: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    accent_pattern: list[int] | None = Field(default=None, min_length=1, max_length=16)
    count_in_bars: int = Field(default=0, ge=0, le=8)
    max_events: int = Field(default=10_000, ge=1, le=10_000)


class MetronomeEvent(BaseModel):
    timestamp_seconds: float
    measure_index: int
    beat_index: int
    subdivision_index: int
    event_type: str
    accent_level: int
    bpm: float
    phase: str


class MetronomeScheduleResponse(BaseModel):
    events: list[MetronomeEvent]
    total_events: int
    truncated: bool


@router.post(
    "/metronome/schedule",
    response_model=MetronomeScheduleResponse,
    summary="Generate a deterministic metronome schedule",
)
async def generate_metronome_schedule(
    request: MetronomeScheduleRequest,
) -> MetronomeScheduleResponse:
    try:
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
    except ValueError as exc:
        raise ApplicationError("invalid_rhythm_input", str(exc), 422) from exc

    events = [
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
    ]
    return MetronomeScheduleResponse(
        events=events,
        total_events=int(schedule["total_events"]),
        truncated=bool(schedule["truncated"]),
    )
