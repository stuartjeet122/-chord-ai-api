from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from music_api.db.base import Base


class AudioMetadata(Base):
    __tablename__ = "audio_metadata"

    audio_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="CASCADE"), primary_key=True
    )
    tags: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    raw_tags: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    technical: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="mutagen+ffprobe")
    mutagen_version: Mapped[str] = mapped_column(String(32), nullable=False)
    ffprobe_version: Mapped[str] = mapped_column(String(128), nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    artworks: Mapped[list[AudioArtwork]] = relationship(
        back_populates="audio_metadata", cascade="all, delete-orphan"
    )


class AudioArtwork(Base):
    __tablename__ = "audio_artworks"
    __table_args__ = (
        UniqueConstraint("audio_asset_id", "artwork_asset_id", name="uq_audio_artwork_assets"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audio_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_metadata.audio_asset_id", ondelete="CASCADE"), nullable=False, index=True
    )
    artwork_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="RESTRICT"), nullable=False
    )
    picture_type: Mapped[str | None] = mapped_column(String(32))
    description: Mapped[str | None] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(32), nullable=False)

    audio_metadata: Mapped[AudioMetadata] = relationship(back_populates="artworks")


class AudioConversion(Base):
    __tablename__ = "audio_conversions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    source_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    output_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="SET NULL")
    )
    profile: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    conversion_duration_ms: Mapped[int | None] = mapped_column()
    ffmpeg_version: Mapped[str] = mapped_column(String(128), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class AudioProcessingRecord(Base):
    __tablename__ = "audio_processing_records"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled')",
            name="ck_audio_processing_records_status",
        ),
        Index("ix_audio_processing_source_created", "source_asset_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    source_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="RESTRICT"), nullable=False
    )
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    processing_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(255))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class AudioProcessingOutput(Base):
    __tablename__ = "audio_processing_outputs"
    __table_args__ = (
        UniqueConstraint("processing_record_id", "position", name="uq_processing_output_position"),
        Index("ix_audio_processing_outputs_asset", "output_asset_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    processing_record_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_processing_records.id", ondelete="CASCADE"), nullable=False
    )
    output_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="SET NULL")
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    output_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class AudioProcessingSource(Base):
    __tablename__ = "audio_processing_sources"
    __table_args__ = (
        UniqueConstraint("processing_record_id", "position", name="uq_processing_source_position"),
        Index("ix_audio_processing_sources_asset", "source_asset_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    processing_record_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_processing_records.id", ondelete="CASCADE"), nullable=False
    )
    source_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="RESTRICT"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
