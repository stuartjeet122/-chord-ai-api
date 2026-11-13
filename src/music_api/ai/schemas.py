from __future__ import annotations

import math
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelCategory(StrEnum):
    CHORD_RECOGNITION = "chord_recognition"
    NOTE_TRANSCRIPTION = "note_transcription"
    SOURCE_SEPARATION = "source_separation"
    MUSIC_UNDERSTANDING = "music_understanding"
    AUDIO_EMBEDDING = "audio_embedding"


class ModelLifecycleState(StrEnum):
    REGISTERED = "registered"
    DOWNLOADING = "downloading"
    INITIALIZING = "initializing"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    DISABLED = "disabled"


class InferenceTask(StrEnum):
    CHORD_RECOGNITION = "chord_recognition"
    NOTE_TRANSCRIPTION = "note_transcription"
    SOURCE_SEPARATION = "source_separation"
    MUSIC_UNDERSTANDING = "music_understanding"
    AUDIO_EMBEDDING = "audio_embedding"


class ExecutionDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class ModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._-]+$")
    name: str = Field(min_length=1, max_length=255)
    provider: str = Field(min_length=1, max_length=128)
    category: ModelCategory
    version: str = Field(min_length=1, max_length=128)
    architecture: str = Field(min_length=1, max_length=255)
    supported_tasks: tuple[InferenceTask, ...]
    weights_path: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    max_duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    supported_devices: tuple[ExecutionDevice, ...]
    license: str
    download_source: str | None = None
    weight_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    weight_format: str | None = None
    commercial_use: str
    enabled: bool = True


class ModelStatusResponse(BaseModel):
    model_id: str
    state: ModelLifecycleState
    selected_device: ExecutionDevice | None = None
    initialized_at: datetime | None = None
    initialization_duration_ms: int | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None


class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str
    name: str
    provider: str
    category: ModelCategory
    version: str
    architecture: str
    supported_tasks: tuple[InferenceTask, ...]
    sample_rate: int | None
    channels: int | None
    max_duration_seconds: float | None
    supported_devices: tuple[ExecutionDevice, ...]
    license: str
    download_source: str | None
    weight_format: str | None
    commercial_use: str
    enabled: bool


class AudioFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_asset_id: uuid.UUID
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channel_count: int = Field(gt=0)
    channels: tuple[tuple[float, ...], ...] | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> AudioFrame:
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample.")
        if self.channels is not None:
            if len(self.channels) != self.channel_count:
                raise ValueError("The audio buffer must match channel_count.")
            expected_frames = self.end_sample - self.start_sample
            if any(len(channel) != expected_frames for channel in self.channels):
                raise ValueError("All audio channels must match the sample range.")
            if any(not math.isfinite(sample) for channel in self.channels for sample in channel):
                raise ValueError("Audio samples must be finite.")
        return self

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


class TimedPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_times(self) -> TimedPrediction:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Prediction end_seconds must exceed start_seconds.")
        return self


class ChordPrediction(TimedPrediction):
    symbol: str = Field(min_length=1, max_length=64)
    root: str | None = Field(default=None, max_length=8)
    quality: str | None = Field(default=None, max_length=32)
    bass: str | None = Field(default=None, max_length=8)


class NotePrediction(TimedPrediction):
    pitch_midi: int = Field(ge=0, le=127)
    velocity: int | None = Field(default=None, ge=0, le=127)
    instrument: str | None = Field(default=None, max_length=128)


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_version: str
    provider: str
    task: InferenceTask
    source_asset_id: uuid.UUID
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    processing_duration_ms: int = Field(ge=0)
    execution_device: ExecutionDevice
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    raw_output: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_result(self) -> InferenceResult:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Result end_seconds must exceed start_seconds.")
        if self.task == InferenceTask.CHORD_RECOGNITION:
            parsed_chords = [ChordPrediction.model_validate(item) for item in self.predictions]
            self.predictions = [item.model_dump(mode="json") for item in parsed_chords]
        elif self.task == InferenceTask.NOTE_TRANSCRIPTION:
            parsed_notes = [NotePrediction.model_validate(item) for item in self.predictions]
            self.predictions = [item.model_dump(mode="json") for item in parsed_notes]
        for prediction in self.predictions:
            if prediction["start_seconds"] < self.start_seconds:
                raise ValueError("Prediction starts before its inference segment.")
            if prediction["end_seconds"] > self.end_seconds:
                raise ValueError("Prediction ends after its inference segment.")
        return self


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: uuid.UUID
    model_id: str = Field(min_length=1, max_length=128)
    task: InferenceTask
    parameters: dict[str, Any] = Field(default_factory=dict)


class InferenceRecordResponse(BaseModel):
    job_id: uuid.UUID
    source_asset_id: uuid.UUID
    model_id: str
    model_version: str
    provider: str
    task: InferenceTask
    status: str
    execution_device: ExecutionDevice | None
    result: dict[str, Any] | None
    metrics: dict[str, Any] | None
    validation_error: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    inference_duration_ms: int | None
    started_at: datetime
    completed_at: datetime | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BatchInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: list[uuid.UUID] = Field(min_length=1, max_length=32)
    model_id: str = Field(min_length=1, max_length=128)
    task: InferenceTask
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_assets(self) -> BatchInferenceRequest:
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids must be unique.")
        return self


class HardwareResponse(BaseModel):
    python_version: str
    operating_system: str
    cpu_count: int
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_devices: list[dict[str, Any]]
    mps_available: bool
    configured_device: Literal["auto", "cpu", "cuda", "mps"]


class BenchmarkChordReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    symbol: str = Field(min_length=1, max_length=64)
    root: str = Field(min_length=1, max_length=8)
    quality: str = Field(min_length=1, max_length=32)
    bass: str | None = Field(default=None, max_length=8)
    inversion: str | None = Field(default=None, max_length=16)

    @model_validator(mode="after")
    def validate_interval(self) -> BenchmarkChordReference:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Reference end_seconds must exceed start_seconds.")
        return self


class BenchmarkNoteReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    pitch_midi: int = Field(ge=0, le=127)

    @model_validator(mode="after")
    def validate_interval(self) -> BenchmarkNoteReference:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Reference end_seconds must exceed start_seconds.")
        return self


class BenchmarkSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: uuid.UUID
    references: list[dict[str, Any]] = Field(min_length=1, max_length=20_000)


class BenchmarkDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._-]+$")
    version: str = Field(min_length=1, max_length=64)
    task: Literal[InferenceTask.CHORD_RECOGNITION, InferenceTask.NOTE_TRANSCRIPTION]
    license: str = Field(min_length=1, max_length=128)
    attribution: str | None = Field(default=None, max_length=512)
    samples: list[BenchmarkSample] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_annotations(self) -> BenchmarkDatasetRequest:
        asset_ids = [sample.asset_id for sample in self.samples]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("A benchmark dataset cannot repeat an audio asset.")
        reference_model = (
            BenchmarkChordReference
            if self.task == InferenceTask.CHORD_RECOGNITION
            else BenchmarkNoteReference
        )
        for sample in self.samples:
            for reference in sample.references:
                reference_model.model_validate(reference)
        return self


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=128)
    dataset_version: str = Field(min_length=1, max_length=64)
    model_ids: list[str] = Field(min_length=1, max_length=16)
    task: InferenceTask
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_models(self) -> BenchmarkRequest:
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("model_ids must be unique.")
        return self


class BenchmarkDatasetResponse(BaseModel):
    id: uuid.UUID
    dataset_id: str
    version: str
    task: InferenceTask
    license: str
    attribution: str | None
    sample_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BenchmarkRunResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    dataset_record_id: uuid.UUID
    model_ids: list[str]
    status: str
    results: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
