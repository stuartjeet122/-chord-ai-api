"""Add user-owned audio asset storage and quota tracking.

Revision ID: 0002_audio_storage
Revises: 0001_initial_schema
Create Date: 2026-11-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_audio_storage"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audio_assets",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("owner_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("asset_type", sa.String(length=32), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=1024), nullable=False),
        sa.Column("storage_provider", sa.String(length=16), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("file_extension", sa.String(length=16), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("audio_format", sa.String(length=32), nullable=True),
        sa.Column("sample_rate", sa.Integer(), nullable=True),
        sa.Column("channel_count", sa.Integer(), nullable=True),
        sa.Column("bit_depth", sa.Integer(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("parent_asset_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("processing_metadata", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("idempotency_owner_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("size_bytes > 0", name="ck_audio_assets_size_positive"),
        sa.CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_audio_assets_duration_nonnegative",
        ),
        sa.CheckConstraint(
            "sample_rate IS NULL OR sample_rate > 0",
            name="ck_audio_assets_sample_rate_positive",
        ),
        sa.CheckConstraint(
            "channel_count IS NULL OR channel_count > 0",
            name="ck_audio_assets_channel_count_positive",
        ),
        sa.CheckConstraint(
            "bit_depth IS NULL OR bit_depth > 0",
            name="ck_audio_assets_bit_depth_positive",
        ),
        sa.CheckConstraint(
            "status IN ('uploading', 'available', 'processing', 'failed', 'deleting', 'deleted')",
            name="ck_audio_assets_status",
        ),
        sa.ForeignKeyConstraint(
            ["parent_asset_id"],
            ["audio_assets.id"],
            name="fk_audio_assets_parent_asset_id_audio_assets",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audio_assets"),
        sa.UniqueConstraint("idempotency_owner_id", "idempotency_key", name="uq_asset_idempotency"),
        sa.UniqueConstraint(
            "storage_provider",
            "storage_key",
            name="uq_asset_storage_key",
        ),
    )
    op.create_index("ix_audio_assets_owner_id", "audio_assets", ["owner_id"])
    op.create_index("ix_audio_assets_asset_type", "audio_assets", ["asset_type"])
    op.create_index(
        "ix_audio_assets_owner_created",
        "audio_assets",
        ["owner_id", "created_at", "id"],
    )
    op.create_index(
        "ix_audio_assets_owner_status",
        "audio_assets",
        ["owner_id", "status"],
    )
    op.create_index("ix_audio_assets_parent", "audio_assets", ["parent_asset_id"])

    op.create_table(
        "storage_quotas",
        sa.Column("owner_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("quota_bytes", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("owner_id", name="pk_storage_quotas"),
    )
    op.create_table(
        "upload_reservations",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("owner_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("size_bytes > 0", name="ck_upload_reservations_size_positive"),
        sa.PrimaryKeyConstraint("id", name="pk_upload_reservations"),
        sa.UniqueConstraint(
            "owner_id",
            "idempotency_key",
            name="uq_upload_reservation_idempotency",
        ),
    )
    op.create_index("ix_upload_reservations_owner_id", "upload_reservations", ["owner_id"])
    op.create_index(
        "ix_upload_reservations_created",
        "upload_reservations",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_upload_reservations_created", table_name="upload_reservations")
    op.drop_index("ix_upload_reservations_owner_id", table_name="upload_reservations")
    op.drop_table("upload_reservations")
    op.drop_table("storage_quotas")
    op.drop_index("ix_audio_assets_parent", table_name="audio_assets")
    op.drop_index("ix_audio_assets_owner_status", table_name="audio_assets")
    op.drop_index("ix_audio_assets_owner_created", table_name="audio_assets")
    op.drop_index("ix_audio_assets_asset_type", table_name="audio_assets")
    op.drop_index("ix_audio_assets_owner_id", table_name="audio_assets")
    op.drop_table("audio_assets")
