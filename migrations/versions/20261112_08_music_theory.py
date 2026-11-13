"""persist separate, versioned theoretical interpretations

Revision ID: 20261112_08
Revises: 20261112_07
Create Date: 2026-11-12 22:40:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261112_08"
down_revision: str | None = "20261112_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "music_theory_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("key_context", sa.JSON(), nullable=True),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("version > 0", name="ck_music_theory_analyses_version_positive"),
        sa.CheckConstraint(
            "source IN ('detected_chords', 'provided_chords')",
            name="ck_music_theory_analyses_source",
        ),
        sa.ForeignKeyConstraint(["analysis_id"], ["music_analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_id", "version", name="uq_music_theory_analysis_version"),
    )
    op.create_index(
        "ix_music_theory_analyses_owner_created",
        "music_theory_analyses",
        ["owner_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_music_theory_analyses_owner_created", table_name="music_theory_analyses")
    op.drop_table("music_theory_analyses")
