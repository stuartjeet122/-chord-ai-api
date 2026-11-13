"""add audio DSP processing history

Revision ID: 20261112_05
Revises: 22748efa84a9
Create Date: 2026-11-12 21:35:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261112_05"
down_revision: str | None = "22748efa84a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audio_processing_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("source_asset_id", sa.Uuid(), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("processing_version", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled')",
            name="ck_audio_processing_records_status",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["processing_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_asset_id"], ["audio_assets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index(
        "ix_audio_processing_source_created",
        "audio_processing_records",
        ["source_asset_id", "created_at"],
    )
    op.create_table(
        "audio_processing_outputs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("processing_record_id", sa.Uuid(), nullable=False),
        sa.Column("output_asset_id", sa.Uuid(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("output_metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["processing_record_id"], ["audio_processing_records.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["output_asset_id"], ["audio_assets.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "processing_record_id", "position", name="uq_processing_output_position"
        ),
    )
    op.create_index(
        "ix_audio_processing_outputs_asset",
        "audio_processing_outputs",
        ["output_asset_id"],
    )
    op.create_table(
        "audio_processing_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("processing_record_id", sa.Uuid(), nullable=False),
        sa.Column("source_asset_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["processing_record_id"], ["audio_processing_records.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["source_asset_id"], ["audio_assets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "processing_record_id", "position", name="uq_processing_source_position"
        ),
    )
    op.create_index(
        "ix_audio_processing_sources_asset",
        "audio_processing_sources",
        ["source_asset_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_audio_processing_sources_asset", table_name="audio_processing_sources")
    op.drop_table("audio_processing_sources")
    op.drop_index("ix_audio_processing_outputs_asset", table_name="audio_processing_outputs")
    op.drop_table("audio_processing_outputs")
    op.drop_index("ix_audio_processing_source_created", table_name="audio_processing_records")
    op.drop_table("audio_processing_records")
