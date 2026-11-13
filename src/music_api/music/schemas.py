from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AnalysisTask = Literal["chords", "key", "tempo", "beats", "full"]


def _default_tasks() -> list[AnalysisTask]:
    return ["full"]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[AnalysisTask] = Field(default_factory=_default_tasks, min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tasks(self) -> AnalysisRequest:
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Analysis tasks must be unique.")
        return self


class ChordEventResponse(BaseModel):
    event_id: uuid.UUID
    symbol: str
    root: str | None
    quality: str
    inversion: str | None
    bass_note: str | None
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    confidence: None = None
    confidence_status: Literal["uncalibrated", "ambiguous", "unknown"]
    sources: list[str]
    evidence: dict[str, Any]
    roman_numeral: str | None = None
    manually_corrected: bool = False


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    asset_id: uuid.UUID = Field(validation_alias="source_asset_id")
    version: int
    status: str
    tasks: list[str]
    source_sha256: str
    engine_version: str
    model_references: list[dict[str, str]]
    execution_device: str
    parameters: dict[str, Any]
    result: dict[str, Any] | None
    stage_errors: list[dict[str, str]]
    created_at: datetime
    completed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class ChordListResponse(BaseModel):
    analysis_id: uuid.UUID
    items: list[ChordEventResponse]
    total: int
    limit: int
    offset: int


class AnalysisComparisonResponse(BaseModel):
    asset_id: uuid.UUID
    analyses: list[AnalysisResponse]


class MetronomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="default", min_length=1, max_length=128)
    bpm: float = Field(..., gt=0, le=300, allow_inf_nan=False)
    time_signature: str = Field(default="4/4", min_length=3, max_length=16)
    subdivision: Literal["beat", "eighth", "sixteenth", "triplet"] = "eighth"
    swing_ratio: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    accent_pattern: list[int] | None = Field(default=None, min_length=1, max_length=16)
    count_in_bars: int = Field(default=0, ge=0, le=16)
    count_in_subdivision: Literal["beat", "eighth", "sixteenth", "triplet"] = "beat"
    is_default: bool = False
    description: str | None = Field(default=None, max_length=512)


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


class TempoMapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="tempo-map", min_length=1, max_length=128)
    source_type: Literal["detected", "manual", "practice"] = "manual"
    base_bpm: float = Field(..., gt=0, le=300, allow_inf_nan=False)
    tempo_points: list[dict[str, float]] = Field(default_factory=list, min_length=1, max_length=64)
    duration_seconds: float = Field(..., gt=0, le=86_400, allow_inf_nan=False)


class PracticePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="practice-plan", min_length=1, max_length=128)
    start_bpm: float = Field(..., gt=0, le=300, allow_inf_nan=False)
    target_bpm: float = Field(..., gt=0, le=300, allow_inf_nan=False)
    stage_duration_seconds: float = Field(..., gt=0, le=86_400, allow_inf_nan=False)
    stage_count: int = Field(default=1, ge=1, le=32)


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


class PracticeSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    song_id: uuid.UUID | None = None
    start_bpm: float = Field(..., gt=0, le=300, allow_inf_nan=False)
    target_bpm: float | None = Field(default=None, gt=0, le=300, allow_inf_nan=False)
    time_signature: str = Field(default="4/4", min_length=3, max_length=16)
    notes: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PracticeSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    song_id: uuid.UUID | None
    start_bpm: float
    target_bpm: float | None
    time_signature: str
    status: str
    start_seconds: float
    end_seconds: float | None
    duration_seconds: float
    notes: str | None
    metadata: dict[str, Any]


class PracticeSessionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["planned", "running", "completed", "cancelled"] | None = None
    start_seconds: float | None = Field(default=None, ge=0, le=86_400, allow_inf_nan=False)
    end_seconds: float | None = Field(default=None, ge=0, le=86_400, allow_inf_nan=False)
    duration_seconds: float | None = Field(default=None, ge=0, le=86_400, allow_inf_nan=False)
    notes: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] | None = None
