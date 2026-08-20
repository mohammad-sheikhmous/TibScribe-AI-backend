"""Persist raw/canonical clinical text and rewrite audit metadata.

Revision ID: 20260819canonical
Revises: 20260814suggestactive
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260819canonical"
down_revision: Union[str, None] = "20260814suggestactive"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("report_items", sa.Column("text_raw", sa.Text(), nullable=True))
    op.add_column("report_items", sa.Column("text_canonical", sa.Text(), nullable=True))
    op.add_column(
        "report_items",
        sa.Column(
            "canonicalization_status",
            sa.String(length=32),
            nullable=False,
            server_default="not_run",
        ),
    )
    op.add_column(
        "report_items", sa.Column("canonicalization_confidence", sa.Float(), nullable=True)
    )
    op.add_column(
        "report_items", sa.Column("canonicalization_model", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "report_items", sa.Column("canonicalization_reasons", sa.JSON(), nullable=True)
    )
    # Existing rows pre-date canonicalization: their effective text is also their raw
    # traceable baseline.  New reports always populate text_raw explicitly.
    op.execute("UPDATE report_items SET text_raw = text WHERE text_raw IS NULL")


def downgrade() -> None:
    op.drop_column("report_items", "canonicalization_reasons")
    op.drop_column("report_items", "canonicalization_model")
    op.drop_column("report_items", "canonicalization_confidence")
    op.drop_column("report_items", "canonicalization_status")
    op.drop_column("report_items", "text_canonical")
    op.drop_column("report_items", "text_raw")
