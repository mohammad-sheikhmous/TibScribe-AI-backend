"""Persist original ASR text alongside canonicalized report text.

Revision ID: 20260819textraw
Revises: 20260814suggestactive
"""
from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260819textraw"
down_revision: Union[str, None] = "20260814suggestactive"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("report_items", sa.Column("text_raw", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("report_items", "text_raw")
