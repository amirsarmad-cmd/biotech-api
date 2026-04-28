"""016 — structured outcome labels from press release text

Per ChatGPT critique on V2: 'Right now outcome is inferred by ±20% price
action heuristic. To evaluate event-outcome prediction separately from
stock-reaction prediction, we need real labels (approved/rejected,
hit-primary-endpoint/missed) from press releases.'

Adds columns to post_catalyst_outcomes:
  outcome_labeled_json     — full JSON returned by the labeler (source URL,
                             evidence quote, reasoning, etc.)
  outcome_label_class      — APPROVED / REJECTED / MET_ENDPOINT /
                             MISSED_ENDPOINT / DELAYED / WITHDRAWN /
                             MIXED / UNKNOWN
  outcome_label_confidence — labeler's confidence 0..1
  outcome_labeled_at       — when the labeler ran (NULL = not yet labeled)

Revision ID: 016_outcome_labels (18 chars)
"""
from alembic import op


revision = '016_outcome_labels'
down_revision = '015_snapshot_table'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS outcome_labeled_json JSONB,
        ADD COLUMN IF NOT EXISTS outcome_label_class TEXT,
        ADD COLUMN IF NOT EXISTS outcome_label_confidence NUMERIC,
        ADD COLUMN IF NOT EXISTS outcome_labeled_at TIMESTAMP WITH TIME ZONE
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pco_outcome_label_class
        ON post_catalyst_outcomes(outcome_label_class)
        WHERE outcome_label_class IS NOT NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pco_outcome_labeled_pending
        ON post_catalyst_outcomes(catalyst_date)
        WHERE outcome_labeled_at IS NULL AND catalyst_date IS NOT NULL
    """)


def downgrade():
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS outcome_labeled_json,
        DROP COLUMN IF EXISTS outcome_label_class,
        DROP COLUMN IF EXISTS outcome_label_confidence,
        DROP COLUMN IF EXISTS outcome_labeled_at
    """)
