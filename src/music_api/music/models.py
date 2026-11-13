from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from music_api.db.base import Base


class MetronomeConfiguration(Base):
    __tablename__ = "metronome_configurations"
    __table_args__ = (
        CheckConstraint("bpm > 0", name="ck_metronome_bpm_positive"),
        CheckConstraint("numerator > 0", name="ck_metronome_numerator_positive"),
        CheckConstraint("denominator > 0", name="ck_metronome_denominator_positive"),
        CheckConstraint("count_in_bars >= 0", name="ck_metronome_count_in_non_negative"),
        Index("ix_metronome_owner_name", "owner_id", "name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    song_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("audio_assets.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    bpm: Mapped[float] = mapped_column(nullable=False)
    numerator: Mapped[int] = mapped_column(Integer, nullable=False)
    denominator: Mapped[int] = mapped_column(Integer, nullable=False)
    beats_per_measure: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    beat_unit: Mapped[str] = mapped_column(String(32), nullable=False, default="quarter")
    subdivision: Mapped[str] = mapped_column(String(16), nullable=False, default="eighth")
    swing_ratio: Mapped[float] = mapped_column(nullable=False, default=0.0)
    accent_pattern: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    count_in_bars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    count_in_subdivision: Mapped[str] = mapped_column(String(16), nullable=False, default="beat")
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    description: Mapped[str | None] = mapped_column(String(512))
    config_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class TempoMap(Base):
    __tablename__ = "tempo_maps"
    __table_args__ = (
        CheckConstraint("base_bpm > 0", name="ck_tempo_maps_base_bpm_positive"),
        Index("ix_tempo_maps_owner_song", "owner_id", "song_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    song_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("audio_assets.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    base_bpm: Mapped[float] = mapped_column(nullable=False)
    points: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class PracticePlan(Base):
    __tablename__ = "practice_plans"
    __table_args__ = (
        CheckConstraint("start_bpm > 0", name="ck_practice_plan_start_bpm_positive"),
        Index("ix_practice_plans_owner_name", "owner_id", "name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    song_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("audio_assets.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(String(512))
    start_bpm: Mapped[float] = mapped_column(nullable=False)
    target_bpm: Mapped[float] = mapped_column(nullable=False)
    stages: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class PracticeSession(Base):
    __tablename__ = "practice_sessions"
    __table_args__ = (
        CheckConstraint("start_bpm > 0", name="ck_practice_session_start_bpm_positive"),
        CheckConstraint("duration_seconds >= 0", name="ck_practice_session_duration_non_negative"),
        Index("ix_practice_sessions_owner_song", "owner_id", "song_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    song_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("audio_assets.id", ondelete="SET NULL"))
    plan_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("practice_plans.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="planned")
    start_seconds: Mapped[float] = mapped_column(nullable=False, default=0.0)
    end_seconds: Mapped[float | None] = mapped_column()
    duration_seconds: Mapped[float] = mapped_column(nullable=False, default=0.0)
    start_bpm: Mapped[float] = mapped_column(nullable=False)
    target_bpm: Mapped[float | None] = mapped_column()
    time_signature: Mapped[str] = mapped_column(String(16), nullable=False, default="4/4")
    notes: Mapped[str | None] = mapped_column(String(4096))
    session_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PracticeSessionEvent(Base):
    __tablename__ = "practice_session_events"
    __table_args__ = (
        CheckConstraint("seconds >= 0", name="ck_practice_session_event_seconds_non_negative"),
        Index("ix_practice_session_events_session", "session_id", "seconds"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("practice_sessions.id", ondelete="CASCADE"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    seconds: Mapped[float] = mapped_column(nullable=False)
    bpm: Mapped[float | None] = mapped_column()
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class MusicAnalysisRecord(Base):
    __tablename__ = "music_analyses"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'partial', 'succeeded', 'failed', 'cancelled')",
            name="ck_music_analyses_status",
        ),
        CheckConstraint("version > 0", name="ck_music_analyses_version_positive"),
        UniqueConstraint(
            "owner_id",
            "source_asset_id",
            "version",
            name="uq_music_analysis_version",
        ),
        UniqueConstraint("job_id", name="uq_music_analysis_job"),
        Index("ix_music_analyses_asset_status", "source_asset_id", "status", "version"),
        Index("ix_music_analyses_owner_created", "owner_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    source_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("audio_assets.id", ondelete="RESTRICT"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    tasks: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_references: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False)
    execution_device: Mapped[str] = mapped_column(String(16), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    stage_errors: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MusicAnalysisEvent(Base):
    __tablename__ = "music_analysis_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN "
            "('chord', 'beat', 'downbeat', 'key_change', 'bass', 'melody', 'section')",
            name="ck_music_analysis_events_type",
        ),
        CheckConstraint("start_seconds >= 0", name="ck_music_analysis_event_start"),
        CheckConstraint(
            "end_seconds IS NULL OR end_seconds > start_seconds",
            name="ck_music_analysis_event_end",
        ),
        Index("ix_music_analysis_events_timeline", "analysis_id", "event_type", "start_seconds"),
        Index("ix_music_analysis_events_root", "analysis_id", "root"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("music_analyses.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    start_seconds: Mapped[float] = mapped_column(nullable=False)
    end_seconds: Mapped[float | None] = mapped_column()
    label: Mapped[str | None] = mapped_column(String(128))
    root: Mapped[str | None] = mapped_column(String(8))
    quality: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column()
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    manually_corrected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class MusicTheoryRecord(Base):
    __tablename__ = "music_theory_analyses"
    __table_args__ = (
        CheckConstraint("version > 0", name="ck_music_theory_analyses_version_positive"),
        CheckConstraint(
            "source IN ('detected_chords', 'provided_chords')",
            name="ck_music_theory_analyses_source",
        ),
        UniqueConstraint("analysis_id", "version", name="uq_music_theory_analysis_version"),
        Index("ix_music_theory_analyses_owner_created", "owner_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("music_analyses.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    key_context: Mapped[dict[str, str] | None] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(24), nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
