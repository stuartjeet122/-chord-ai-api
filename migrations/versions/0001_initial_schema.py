"""Establish the initial migration baseline.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-12
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Domain tables are introduced with their owning implementation phases.
    pass


def downgrade() -> None:
    pass
