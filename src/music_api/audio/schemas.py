from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from music_api.storage.schemas import AssetResponse


class AudioArtworkResponse(BaseModel):
    asset_id: uuid.UUID
    mime_type: str
    picture_type: str | None
    description: str | None


class ConversionRequest(BaseModel):
    profile: Literal["source_preserving_wav", "music_analysis_wav"] = "source_preserving_wav"
    output_format: Literal["wav", "flac", "mp3", "aac", "opus"] = "wav"
    sample_rate: int | None = Field(default=None, ge=22050, le=96000)
    channels: Literal["original", "mono", "stereo"] = "original"
    bit_depth: Literal[16, 24, 32] = 24
    float32: bool = False
    bitrate_kbps: int | None = Field(default=None, ge=64, le=512)
    normalize: bool = False
    trim_start_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    trim_end_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    allow_upsampling: bool = False

    @model_validator(mode="after")
    def validate_options(self) -> ConversionRequest:
        if self.sample_rate is not None and self.sample_rate not in {
            22050,
            32000,
            44100,
            48000,
            88200,
            96000,
        }:
            raise ValueError("sample_rate must be one of 22050, 32000, 44100, 48000, 88200, 96000.")
        if self.float32 and (self.output_format != "wav" or self.bit_depth != 32):
            raise ValueError("float32 requires WAV output with bit_depth=32.")
        if self.output_format != "wav" and self.float32:
            raise ValueError("float32 output is supported only for WAV.")
        if self.bitrate_kbps is not None and self.output_format in {"wav", "flac"}:
            raise ValueError("bitrate_kbps applies only to lossy compressed formats.")
        if self.trim_start_seconds is not None and self.trim_end_seconds is not None:
            if self.trim_end_seconds <= self.trim_start_seconds:
                raise ValueError("trim_end_seconds must be greater than trim_start_seconds.")
        return self


class JobResponse(BaseModel):
    id: uuid.UUID
    task_type: str
    input_asset_id: uuid.UUID
    output_asset_id: uuid.UUID | None
    status: str
    stage: str
    progress: int
    attempt_count: int
    cancel_requested: bool
    options: dict[str, Any]
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class AudioMetadataResponse(BaseModel):
    audio_asset_id: uuid.UUID
    tags: dict[str, Any]
    raw_tags: dict[str, Any]
    technical: dict[str, Any]
    source: str
    mutagen_version: str
    ffprobe_version: str
    extracted_at: datetime
    artworks: list[AudioArtworkResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class AudioAssetsResponse(BaseModel):
    source_asset_id: uuid.UUID
    assets: list[AssetResponse]


class ConversionProfileResponse(BaseModel):
    profiles: list[dict[str, Any]]
