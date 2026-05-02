"""019 — per-row attempt counter on the outcome labeler

Required by the Gemini-stall workaround in routes/admin.py: workers
now skip rows whose outcome_label_attempts has hit the cap. Without
this column the workaround silently re-claims every failed row and
the queue never exhausts (which is exactly what produced 419k wasted
errors over 5.5h on 2026-05-02 before the operator noticed).

Adds columns to post_catalyst_outcomes:
  outcome_label_attempts        — INTEGER NOT NULL DEFAULT 0
  outcome_label_last_attempt_at — TIMESTAMPTZ
  outcome_label_last_error      — TEXT (last failure message, truncated)

Plus a partial index that keeps _claim_unlabeled fast.

Revision ID: 019_label_attempts
"""
from alembic import op


revision = "019_label_attempts"
down_revision = "018_lgbm_model"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS outcome_label_attempts INTEGER NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS outcome_label_last_attempt_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN IF NOT EXISTS outcome_label_last_error TEXT
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pco_label_pending_attempts
        ON post_catalyst_outcomes(catalyst_date)
        WHERE outcome_labeled_at IS NULL
          AND outcome_label_attempts < 3
          AND catalyst_date IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pco_label_pending_attempts")
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS outcome_label_attempts,
        DROP COLUMN IF EXISTS outcome_label_last_attempt_at,
        DROP COLUMN IF EXISTS outcome_label_last_error
    """)
