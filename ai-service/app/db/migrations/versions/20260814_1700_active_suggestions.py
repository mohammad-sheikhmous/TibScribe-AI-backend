"""Retain KBS suggestion history across clinician-triggered recomputation.

Revision ID: 20260814suggestactive
Revises: 20260814gateway
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "20260814suggestactive"
down_revision: Union[str, None] = "20260814gateway"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("suggestions") as batch:
        batch.add_column(sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch.create_index("ix_suggestions_is_active", ["is_active"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("suggestions") as batch:
        batch.drop_index("ix_suggestions_is_active")
        batch.drop_column("is_active")
