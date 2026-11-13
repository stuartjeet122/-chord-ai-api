"""add AI inference and benchmark infrastructure

Revision ID: 20261112_06
Revises: 20261112_05
Create Date: 2026-11-12 22:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261112_06"
down_revision: str | None = "20261112_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_inference_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("source_asset_id", sa.Uuid(), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("execution_device", sa.String(length=16), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("validation_error", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.Column("inference_duration_ms", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled', 'validation_failed')",
            name="ck_ai_inference_records_status",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["processing_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_asset_id"], ["audio_assets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_ai_inference_records_owner_id", "ai_inference_records", ["owner_id"])
    op.create_index(
        "ix_ai_inference_source_created",
        "ai_inference_records",
        ["source_asset_id", "created_at"],
    )
    op.create_index(
        "ix_ai_inference_model_created",
        "ai_inference_records",
        ["model_id", "created_at"],
    )
    op.create_table(
        "ai_inference_cache",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("source_asset_id", sa.Uuid(), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("preprocessing_hash", sa.String(length=64), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_asset_id"], ["audio_assets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_id", "cache_key", name="uq_ai_inference_cache_owner_key"),
    )
    op.create_index("ix_ai_inference_cache_expires", "ai_inference_cache", ["expires_at"])
    op.create_table(
        "ai_benchmark_datasets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("license", sa.String(length=128), nullable=False),
        sa.Column("attribution", sa.String(length=512), nullable=True),
        sa.Column("samples", sa.JSON(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("sample_count > 0", name="ck_ai_benchmark_datasets_sample_count"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_id", "dataset_id", "version", name="uq_ai_dataset_owner_version"
        ),
    )
    op.create_index(
        "ix_ai_benchmark_datasets_owner_created",
        "ai_benchmark_datasets",
        ["owner_id", "created_at"],
    )
    op.create_table(
        "ai_benchmark_dataset_assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_record_id", sa.Uuid(), nullable=False),
        sa.Column("source_asset_id", sa.Uuid(), nullable=False),
        sa.Column("sample_index", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_record_id"], ["ai_benchmark_datasets.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["source_asset_id"], ["audio_assets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_record_id",
            "sample_index",
            name="uq_ai_benchmark_dataset_sample_index",
        ),
        sa.UniqueConstraint(
            "dataset_record_id",
            "source_asset_id",
            name="uq_ai_benchmark_dataset_asset",
        ),
    )
    op.create_index(
        "ix_ai_benchmark_dataset_source_asset",
        "ai_benchmark_dataset_assets",
        ["source_asset_id"],
    )
    op.create_table(
        "ai_benchmark_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_record_id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("model_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_ai_benchmark_runs_status",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["processing_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["dataset_record_id"], ["ai_benchmark_datasets.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index(
        "ix_ai_benchmark_runs_owner_created",
        "ai_benchmark_runs",
        ["owner_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ai_benchmark_runs_owner_created", table_name="ai_benchmark_runs")
    op.drop_table("ai_benchmark_runs")
    op.drop_index(
        "ix_ai_benchmark_dataset_source_asset",
        table_name="ai_benchmark_dataset_assets",
    )
    op.drop_table("ai_benchmark_dataset_assets")
    op.drop_index("ix_ai_benchmark_datasets_owner_created", table_name="ai_benchmark_datasets")
    op.drop_table("ai_benchmark_datasets")
    op.drop_index("ix_ai_inference_cache_expires", table_name="ai_inference_cache")
    op.drop_table("ai_inference_cache")
    op.drop_index("ix_ai_inference_model_created", table_name="ai_inference_records")
    op.drop_index("ix_ai_inference_source_created", table_name="ai_inference_records")
    op.drop_index("ix_ai_inference_records_owner_id", table_name="ai_inference_records")
    op.drop_table("ai_inference_records")
