"""add_password_reset_tokens_table

Revision ID: a92f81c12e34
Revises: 810319bdc6d0
Create Date: 2026-10-22 14:44:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a92f81c12e34'
down_revision = '810319bdc6d0'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    if 'password_reset_tokens' not in tables:
        op.create_table(
            'password_reset_tokens',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('token_hash', sa.String(length=64), nullable=False),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
            sa.Column('used_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        with op.batch_alter_table('password_reset_tokens', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_password_reset_tokens_expires_at'), ['expires_at'], unique=False)
            batch_op.create_index(batch_op.f('ix_password_reset_tokens_token_hash'), ['token_hash'], unique=True)
            batch_op.create_index(batch_op.f('ix_password_reset_tokens_user_id'), ['user_id'], unique=False)
            batch_op.create_index('ix_pwd_reset_user_expires', ['user_id', 'expires_at'], unique=False)


def downgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    if 'password_reset_tokens' in tables:
        with op.batch_alter_table('password_reset_tokens', schema=None) as batch_op:
            batch_op.drop_index('ix_pwd_reset_user_expires')
            batch_op.drop_index(batch_op.f('ix_password_reset_tokens_user_id'))
            batch_op.drop_index(batch_op.f('ix_password_reset_tokens_token_hash'))
            batch_op.drop_index(batch_op.f('ix_password_reset_tokens_expires_at'))
        op.drop_table('password_reset_tokens')
