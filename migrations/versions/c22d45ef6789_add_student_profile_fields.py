"""add_student_profile_fields

Revision ID: c22d45ef6789
Revises: a92f81c12e34
Create Date: 2026-10-23 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c22d45ef6789'
down_revision = 'a92f81c12e34'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    if 'users' in tables:
        columns = {col['name'] for col in inspector.get_columns('users')}
        with op.batch_alter_table('users', schema=None) as batch_op:
            if 'academic_year' not in columns:
                batch_op.add_column(sa.Column('academic_year', sa.String(length=20), nullable=True))
            if 'target_exam' not in columns:
                batch_op.add_column(sa.Column('target_exam', sa.String(length=30), nullable=True))


def downgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    if 'users' in tables:
        columns = {col['name'] for col in inspector.get_columns('users')}
        with op.batch_alter_table('users', schema=None) as batch_op:
            if 'target_exam' in columns:
                batch_op.drop_column('target_exam')
            if 'academic_year' in columns:
                batch_op.drop_column('academic_year')
