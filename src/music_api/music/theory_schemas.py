from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TheoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoteRequest(TheoryRequest):
    note: str = Field(min_length=1, max_length=8)


class NoteResponse(BaseModel):
    spelling: str
    pitch_class: int
    octave: int | None
    midi: int | None
    frequency_hz: float | None


class IntervalRequest(TheoryRequest):
    from_pitch: str = Field(min_length=1, max_length=8)
    to_pitch: str = Field(min_length=1, max_length=8)


class IntervalResponse(BaseModel):
    name: str
    number: int
    quality: str
    semitones: int
    direction: Literal[-1, 1]


class ScaleRequest(TheoryRequest):
    root: str = Field(min_length=1, max_length=8)
    scale: str = Field(default="major", min_length=1, max_length=32)
    pattern: list[int] | None = Field(default=None, min_length=1, max_length=24)


class ScaleResponse(BaseModel):
    root: str
    name: str
    interval_pattern: list[int]
    notes: list[str]
    pitch_classes: list[int]
    degrees: list[str]
    tonal_context: dict[str, str] | None


class ScaleIdentificationRequest(TheoryRequest):
    notes: list[str] = Field(min_length=1, max_length=24)


class ScaleCandidateResponse(BaseModel):
    root: str
    scale: str
    matched_notes: int
    input_note_count: int
    score: float


class ScaleIdentificationResponse(BaseModel):
    status: Literal["unique", "ambiguous", "no_match"]
    candidates: list[ScaleCandidateResponse]


class ChordParseRequest(TheoryRequest):
    symbol: str = Field(min_length=1, max_length=64)


class ChordConstructionRequest(TheoryRequest):
    root: str = Field(min_length=1, max_length=8)
    quality: str = Field(min_length=1, max_length=32)


class ChordToneResponse(BaseModel):
    degree: int
    alteration: int
    semitones: int
    note: str


class ChordResponse(BaseModel):
    symbol: str
    root: str
    root_pitch_class: int
    quality: str
    extensions: list[int]
    alterations: list[str]
    added_tones: list[int]
    omitted_tones: list[int]
    bass: str | None
    bass_pitch_class: int | None
    tones: list[ChordToneResponse]
    pitch_classes: list[int]


class ChordInversionRequest(TheoryRequest):
    symbol: str = Field(min_length=1, max_length=64)
    bass: str | None = Field(default=None, max_length=8)


class ChordInversionResponse(BaseModel):
    symbol: str
    root: str
    bass: str
    inversion: str
    bass_is_chord_tone: bool
    chord_tones: list[str]
    pitch_classes: list[int]
    voicing_inferred: Literal[False]


class KeyRequest(TheoryRequest):
    key: str = Field(min_length=1, max_length=32)


class KeyResponse(BaseModel):
    tonic: str
    mode: str
    key_signature: list[str]
    signature_count: int
    relative_key: str
    parallel_key: str
    scale_notes: list[str]
    circle_of_fifths_position: int | None


class KeyTranspositionRequest(TheoryRequest):
    key: str = Field(min_length=1, max_length=32)
    semitones: int = Field(ge=-48, le=48)
    prefer_flats: bool = False


class KeyTranspositionResponse(BaseModel):
    source_key: str
    semitones: int
    target_key: str
    target_key_signature: KeyResponse


class HarmonyRequest(TheoryRequest):
    key: str = Field(min_length=1, max_length=32)
    sevenths: bool = False


class HarmonyResponse(BaseModel):
    key: str
    chords: list[dict[str, Any]]


class MelodyHarmonyRequest(TheoryRequest):
    key: str = Field(min_length=1, max_length=32)
    note: str = Field(min_length=1, max_length=8)
    sevenths: bool = False


class MelodyHarmonyResponse(BaseModel):
    note: str
    key: str
    status: str
    unique: bool
    candidates: list[dict[str, Any]]


class HarmonicTimestamp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)


class ProgressionRequest(TheoryRequest):
    key: str = Field(min_length=1, max_length=32)
    chord_symbols: list[str] = Field(min_length=1, max_length=256)
    timestamps: list[HarmonicTimestamp] | None = None
    soprano_notes: list[str] | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_timestamps(self) -> ProgressionRequest:
        if self.timestamps is not None and len(self.timestamps) != len(self.chord_symbols):
            raise ValueError("Provide one timestamp range for each chord.")
        if self.soprano_notes is not None and len(self.soprano_notes) != len(self.chord_symbols):
            raise ValueError("Provide one soprano pitch for each chord.")
        if self.timestamps is not None:
            for current, following in zip(self.timestamps, self.timestamps[1:], strict=False):
                if following.start_seconds < current.start_seconds:
                    raise ValueError("Chord timestamps must be ordered by start_seconds.")
        return self


class ProgressionResponse(BaseModel):
    key: dict[str, str]
    chords: list[dict[str, Any]]
    root_movements: list[dict[str, Any]]
    diatonic_chords: list[bool]
    harmonic_functions: list[str | None]
    cadences: list[dict[str, Any]]
    patterns: list[dict[str, Any]]
    repeated_patterns: list[dict[str, Any]]
    harmonic_rhythm: dict[str, Any] | None
    confidence: None
    confidence_methodology: str


class ProgressionComparisonRequest(TheoryRequest):
    first_chords: list[str] = Field(min_length=1, max_length=256)
    second_chords: list[str] = Field(min_length=1, max_length=256)
    key: str | None = Field(default=None, max_length=32)


class ProgressionComparisonResponse(BaseModel):
    same_length: bool
    same_symbols: bool
    same_harmonic_structure: bool
    equivalent_under_transposition: bool
    semitone_shift: int | None
    first_analysis: dict[str, Any] | None = None
    second_analysis: dict[str, Any] | None = None


class TranspositionRequest(TheoryRequest):
    chord_symbols: list[str] = Field(min_length=1, max_length=256)
    semitones: int = Field(ge=-48, le=48)
    key: str | None = Field(
        default=None, max_length=32, description="Target key used for enharmonic spelling."
    )
    prefer_flats: bool = False


class TranspositionResponse(BaseModel):
    semitones: int
    target_key: str | None
    source_symbols: list[str]
    symbols: list[str]
    chords: list[ChordResponse]
    preserved_structure: bool
    direction: str


class KeyAwareTranspositionRequest(TheoryRequest):
    chord_symbols: list[str] = Field(min_length=1, max_length=256)
    source_key: str = Field(min_length=1, max_length=32)
    target_key: str = Field(min_length=1, max_length=32)
    prefer_flats: bool = False


class KeyAwareTranspositionResponse(TranspositionResponse):
    source_key: str
    target_key: str


class CapoRequest(TheoryRequest):
    chord_symbols: list[str] = Field(min_length=1, max_length=256)
    capo_fret: int = Field(ge=0, le=24)
    direction: Literal["written_to_sounding", "sounding_to_written"] = "written_to_sounding"
    prefer_flats: bool = False
    key: str | None = Field(default=None, max_length=32)


class CapoResponse(BaseModel):
    capo_fret: int
    direction: Literal["written_to_sounding", "sounding_to_written"]
    written_chords: list[str]
    sounding_chords: list[str]
    written_key: str | None
    sounding_key: str | None
    semitone_delta: int
    convention: str


class TheoryInterpretationRequest(TheoryRequest):
    key: str | None = Field(default=None, max_length=32)
    chord_symbols: list[str] | None = Field(default=None, max_length=256)


class TheoryInterpretationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    analysis_id: uuid.UUID
    version: int
    key_context: dict[str, str] | None
    source: Literal["detected_chords", "provided_chords"]
    result: dict[str, Any]
    created_at: datetime
