"""add versioned music analysis results and timeline events

Revision ID: 20261112_07
Revises: 20261112_06
Create Date: 2026-11-12 22:30:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261112_07"
down_revision: str | None = "20261112_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "music_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("source_asset_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("tasks", sa.JSON(), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("engine_version", sa.String(length=64), nullable=False),
        sa.Column("model_references", sa.JSON(), nullable=False),
        sa.Column("execution_device", sa.String(length=16), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("stage_errors", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'partial', 'succeeded', 'failed', 'cancelled')",
            name="ck_music_analyses_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_music_analyses_version_positive"),
        sa.ForeignKeyConstraint(["job_id"], ["processing_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_asset_id"], ["audio_assets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_id",
            "source_asset_id",
            "version",
            name="uq_music_analysis_version",
        ),
        sa.UniqueConstraint("job_id", name="uq_music_analysis_job"),
    )
    op.create_index(
        "ix_music_analyses_asset_status",
        "music_analyses",
        ["source_asset_id", "status", "version"],
    )
    op.create_index("ix_music_analyses_owner_created", "music_analyses", ["owner_id", "created_at"])
    op.create_table(
        "music_analysis_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=16), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=True),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("root", sa.String(length=8), nullable=True),
        sa.Column("quality", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("manually_corrected", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "event_type IN "
            "('chord', 'beat', 'downbeat', 'key_change', 'bass', 'melody', 'section')",
            name="ck_music_analysis_events_type",
        ),
        sa.CheckConstraint("start_seconds >= 0", name="ck_music_analysis_event_start"),
        sa.CheckConstraint(
            "end_seconds IS NULL OR end_seconds > start_seconds",
            name="ck_music_analysis_event_end",
        ),
        sa.ForeignKeyConstraint(["analysis_id"], ["music_analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_music_analysis_events_timeline",
        "music_analysis_events",
        ["analysis_id", "event_type", "start_seconds"],
    )
    op.create_index(
        "ix_music_analysis_events_root", "music_analysis_events", ["analysis_id", "root"]
    )


def downgrade() -> None:
    op.drop_index("ix_music_analysis_events_root", table_name="music_analysis_events")
    op.drop_index("ix_music_analysis_events_timeline", table_name="music_analysis_events")
    op.drop_table("music_analysis_events")
    op.drop_index("ix_music_analyses_owner_created", table_name="music_analyses")
    op.drop_index("ix_music_analyses_asset_status", table_name="music_analyses")
    op.drop_table("music_analyses")
