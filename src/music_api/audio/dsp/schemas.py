from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeSeconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class DspRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WaveformRequest(DspRequest):
    samples_per_second: int = Field(default=100, ge=1, le=5000)
    max_points: int = Field(default=2_000_000, ge=1, le=2_000_000)


class SilenceDetectionRequest(DspRequest):
    threshold_dbfs: float = Field(default=-60, ge=-160, le=0, allow_inf_nan=False)
    relative: bool = False
    window_ms: int = Field(default=20, ge=1, le=1000)
    min_silence_ms: int = Field(default=300, ge=0, le=60_000)
    min_sound_ms: int = Field(default=0, ge=0, le=60_000)


class TrimRequest(DspRequest):
    start_seconds: NonNegativeSeconds
    end_seconds: PositiveSeconds

    @model_validator(mode="after")
    def validate_bounds(self) -> TrimRequest:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self


class AudioTransformRequest(DspRequest):
    semitones: float | None = Field(default=None, ge=-24, le=24, allow_inf_nan=False)
    source_key: str | None = Field(default=None, min_length=1, max_length=32)
    target_key: str | None = Field(default=None, min_length=1, max_length=32)
    tempo_factor: float | None = Field(default=None, ge=0.25, le=4, allow_inf_nan=False)
    source_bpm: float | None = Field(default=None, ge=20, le=400, allow_inf_nan=False)
    target_bpm: float | None = Field(default=None, ge=20, le=400, allow_inf_nan=False)
    start_seconds: NonNegativeSeconds | None = None
    end_seconds: PositiveSeconds | None = None
    edge_fade_ms: int = Field(default=10, ge=0, le=50)
    output_format: Literal["wav", "flac", "mp3", "aac", "opus"] = "wav"
    bitrate_kbps: int = Field(default=192, ge=64, le=320)

    @model_validator(mode="after")
    def validate_transform(self) -> AudioTransformRequest:
        if (self.source_key is None) != (self.target_key is None):
            raise ValueError("Provide both source_key and target_key for key transposition.")
        if self.semitones is not None and self.source_key is not None:
            raise ValueError("Use semitones or source_key/target_key, not both.")
        if (self.source_bpm is None) != (self.target_bpm is None):
            raise ValueError("Provide both source_bpm and target_bpm for BPM conversion.")
        if self.tempo_factor is not None and self.source_bpm is not None:
            raise ValueError("Use tempo_factor or source_bpm/target_bpm, not both.")
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("Provide both start_seconds and end_seconds for a selected section.")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self


class SplitRequest(DspRequest):
    mode: Literal["timestamps", "equal", "silence", "chunks"] = "equal"
    boundaries_seconds: list[NonNegativeSeconds] = Field(default_factory=list, max_length=1000)
    segment_seconds: PositiveSeconds | None = Field(default=None, le=21_600)
    overlap_seconds: NonNegativeSeconds = Field(default=0, le=3600)
    min_segment_seconds: NonNegativeSeconds | None = Field(default=None, le=21_600)
    max_segment_seconds: PositiveSeconds | None = Field(default=None, le=21_600)
    silence_threshold_dbfs: float = Field(default=-45, ge=-160, le=0, allow_inf_nan=False)
    min_silence_ms: int = Field(default=300, ge=0, le=60_000)

    @model_validator(mode="after")
    def validate_mode(self) -> SplitRequest:
        if self.mode == "timestamps" and not self.boundaries_seconds:
            raise ValueError("Timestamp splitting requires boundaries_seconds.")
        if self.mode in {"equal", "chunks"} and self.segment_seconds is None:
            raise ValueError(f"{self.mode} splitting requires segment_seconds.")
        if self.overlap_seconds >= (self.segment_seconds or float("inf")):
            raise ValueError("overlap_seconds must be shorter than segment_seconds.")
        if self.boundaries_seconds != sorted(set(self.boundaries_seconds)):
            raise ValueError("boundaries_seconds must be unique and in ascending order.")
        return self


class GainRequest(DspRequest):
    gain_db: float | None = Field(default=None, ge=-120, le=60, allow_inf_nan=False)
    linear_gain: float | None = Field(default=None, ge=0, le=1000, allow_inf_nan=False)
    channel_gains_db: list[float] | None = Field(default=None, min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_mode(self) -> GainRequest:
        if (
            sum(
                value is not None
                for value in (self.gain_db, self.linear_gain, self.channel_gains_db)
            )
            != 1
        ):
            raise ValueError("Specify exactly one gain mode.")
        if self.channel_gains_db and any(
            not -120 <= value <= 60 or not math_is_finite(value) for value in self.channel_gains_db
        ):
            raise ValueError("Channel gains must be finite values from -120 to 60 dB.")
        return self


class NormalizeRequest(DspRequest):
    method: Literal["peak", "rms", "loudness"] = "peak"
    target_dbfs: float = Field(default=-1, ge=-60, le=0, allow_inf_nan=False)
    ceiling_dbfs: float = Field(default=-1, ge=-60, le=0, allow_inf_nan=False)
    target_lufs: float = Field(default=-16, ge=-36, le=-5, allow_inf_nan=False)
    true_peak_dbfs: float = Field(default=-1.5, ge=-9, le=0, allow_inf_nan=False)
    loudness_range_lu: float = Field(default=11, ge=1, le=20, allow_inf_nan=False)
    limit: bool = False


class FadeRequest(DspRequest):
    fade_in_seconds: NonNegativeSeconds = 0
    fade_out_seconds: NonNegativeSeconds = 0
    curve: Literal["linear", "equal_power"] = "linear"


class FilterRequest(DspRequest):
    kind: Literal["highpass", "lowpass", "bandpass", "notch"]
    cutoff_hz: float | tuple[float, float]
    order: int = Field(default=4, ge=1, le=12)
    zero_phase: bool = False


class ChannelRequest(DspRequest):
    mode: Literal["mono", "stereo", "left", "right", "swap", "split"]


class ResampleRequest(DspRequest):
    sample_rate: int = Field(ge=8000, le=192000)


class ChunkRequest(DspRequest):
    chunk_seconds: PositiveSeconds = Field(le=21_600)
    overlap_seconds: NonNegativeSeconds = Field(default=0, le=3600)
    min_chunk_seconds: NonNegativeSeconds | None = Field(default=None, le=21_600)
    max_chunk_seconds: PositiveSeconds | None = Field(default=None, le=21_600)
    sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    channels: Literal["preserve", "mono", "stereo"] = "preserve"
    profile_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkRequest:
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("overlap_seconds must be shorter than chunk_seconds.")
        return self


class ConcatenateRequest(DspRequest):
    asset_ids: list[uuid.UUID] = Field(min_length=2, max_length=32)
    sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    channels: Literal["preserve", "mono", "stereo"] = "preserve"
    crossfade_seconds: NonNegativeSeconds = Field(default=0, le=30)

    @model_validator(mode="after")
    def validate_sources(self) -> ConcatenateRequest:
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids must be unique.")
        return self


class PipelineTrim(DspRequest):
    type: Literal["trim"]
    start_seconds: NonNegativeSeconds
    end_seconds: PositiveSeconds


class PipelineGain(DspRequest):
    type: Literal["gain"]
    gain_db: float = Field(ge=-120, le=60, allow_inf_nan=False)


class PipelineFade(DspRequest):
    type: Literal["fade"]
    fade_in_seconds: NonNegativeSeconds = 0
    fade_out_seconds: NonNegativeSeconds = 0
    curve: Literal["linear", "equal_power"] = "linear"


class PipelineFilter(DspRequest):
    type: Literal["filter"]
    kind: Literal["highpass", "lowpass", "bandpass", "notch"]
    cutoff_hz: float | tuple[float, float]
    order: int = Field(default=4, ge=1, le=12)
    zero_phase: bool = False


class PipelineResample(DspRequest):
    type: Literal["resample"]
    sample_rate: int = Field(ge=8000, le=192000)


class PipelineChannels(DspRequest):
    type: Literal["channels"]
    mode: Literal["mono", "stereo", "left", "right", "swap"]


class PipelineNormalize(DspRequest):
    type: Literal["normalize"]
    method: Literal["peak", "rms"] = "peak"
    target_dbfs: float = Field(default=-1, ge=-60, le=0, allow_inf_nan=False)
    ceiling_dbfs: float = Field(default=-1, ge=-60, le=0, allow_inf_nan=False)


PipelineOperation = Annotated[
    PipelineTrim
    | PipelineGain
    | PipelineFade
    | PipelineFilter
    | PipelineResample
    | PipelineChannels
    | PipelineNormalize,
    Field(discriminator="type"),
]


class PipelineRequest(DspRequest):
    operations: list[PipelineOperation] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_ordered_operations(self) -> PipelineRequest:
        for operation in self.operations:
            if (
                isinstance(operation, PipelineTrim)
                and operation.end_seconds <= operation.start_seconds
            ):
                raise ValueError("Pipeline trim end_seconds must exceed start_seconds.")
        return self


class ProcessingOutputResponse(BaseModel):
    asset_id: uuid.UUID | None
    position: int
    metadata: dict[str, object]


class ProcessingHistoryResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    source_asset_id: uuid.UUID
    operation: str
    parameters: dict[str, object]
    status: str
    processing_version: str
    started_at: datetime
    completed_at: datetime | None
    duration_ms: int | None
    error_code: str | None
    error_message: str | None
    result: dict[str, object] | None
    outputs: list[ProcessingOutputResponse]


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
