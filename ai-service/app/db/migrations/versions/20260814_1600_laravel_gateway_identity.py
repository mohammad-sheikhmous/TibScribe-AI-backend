"""Laravel gateway identity/correlation fields.

Revision ID: 20260814gateway
Revises: 8f8ec9d933cd
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "20260814gateway"
down_revision: Union[str, None] = "8f8ec9d933cd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    with op.batch_alter_table("patients") as batch:
        batch.add_column(sa.Column("external_source", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("external_id", sa.String(length=100), nullable=True))
        batch.create_index("ix_patients_external_source", ["external_source"], unique=False)
        batch.create_index("ix_patients_external_id", ["external_id"], unique=False)
        batch.create_unique_constraint("uq_patient_external_identity", ["external_source", "external_id"])
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("external_session_id", sa.String(length=100), nullable=True))
        batch.add_column(sa.Column("external_doctor_id", sa.String(length=100), nullable=True))
        batch.create_index("ix_jobs_external_session_id", ["external_session_id"], unique=True)
        batch.create_index("ix_jobs_external_doctor_id", ["external_doctor_id"], unique=False)

def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_index("ix_jobs_external_doctor_id")
        batch.drop_index("ix_jobs_external_session_id")
        batch.drop_column("external_doctor_id")
        batch.drop_column("external_session_id")
    with op.batch_alter_table("patients") as batch:
        batch.drop_constraint("uq_patient_external_identity", type_="unique")
        batch.drop_index("ix_patients_external_id")
        batch.drop_index("ix_patients_external_source")
        batch.drop_column("external_id")
        batch.drop_column("external_source")
