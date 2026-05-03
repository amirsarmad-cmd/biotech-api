"""020 — prediction_v2 columns on post_catalyst_outcomes

Required by the NPV-driven prediction wiring described in
docs/spec-03-prediction.md (statistical model) and the user-approved
plan in C:/Users/itsup/.claude/plans/what-do-you-think-linked-sunbeam.md
(NPV-primary hybrid).

Adds 21 nullable columns to post_catalyst_outcomes so the new compute
path can run alongside the legacy `predicted_move = p × ref_up + (1-p) ×
ref_down` formula in shadow mode (PREDICTION_V2_ENABLED=0 by default).
The legacy column `predicted_move_pct` keeps being written by the old
formula; v2 writes to the new columns. Phase 3 of the rollout flips
which column the UI reads.

Also creates the lookup index on `historical_catalyst_moves` so the
move-distribution lookup is O(log n).

Revision ID: 020_prediction_v2_columns
"""
from alembic import op


revision = "020_prediction_v2_columns"
down_revision = "019_label_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        -- legacy archive (filled in shadow alongside the legacy write)
        ADD COLUMN IF NOT EXISTS predicted_move_pct_v1_archive NUMERIC,
        -- v2 outputs
        ADD COLUMN IF NOT EXISTS predicted_move_npv_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS predicted_move_statistical_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS predicted_low_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS predicted_high_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS predicted_p_source TEXT,
        ADD COLUMN IF NOT EXISTS predicted_p_confidence TEXT,
        ADD COLUMN IF NOT EXISTS magnitude_n INTEGER,
        ADD COLUMN IF NOT EXISTS magnitude_fallback_level TEXT,
        ADD COLUMN IF NOT EXISTS regime TEXT,
        ADD COLUMN IF NOT EXISTS cap_bucket_at_prediction TEXT,
        -- priced-in (computed two ways, both stored, average used by default)
        ADD COLUMN IF NOT EXISTS priced_in_fraction NUMERIC,
        ADD COLUMN IF NOT EXISTS priced_in_method TEXT,
        ADD COLUMN IF NOT EXISTS priced_in_ratio_value NUMERIC,
        ADD COLUMN IF NOT EXISTS priced_in_options_value NUMERIC,
        -- disagreement tracking (NPV vs statistical)
        ADD COLUMN IF NOT EXISTS disagreement_pp NUMERIC,
        ADD COLUMN IF NOT EXISTS disagreement_verdict TEXT,
        ADD COLUMN IF NOT EXISTS disagreement_reasoning TEXT,
        -- which path produced the primary prediction
        ADD COLUMN IF NOT EXISTS prediction_source_v2 TEXT,
        -- abstention is now first-class
        ADD COLUMN IF NOT EXISTS abstained BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS abstain_reason TEXT
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_hcm_lookup
        ON historical_catalyst_moves (catalyst_type, market_cap_bucket, source)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_hcm_lookup")
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS predicted_move_pct_v1_archive,
        DROP COLUMN IF EXISTS predicted_move_npv_pct,
        DROP COLUMN IF EXISTS predicted_move_statistical_pct,
        DROP COLUMN IF EXISTS predicted_low_pct,
        DROP COLUMN IF EXISTS predicted_high_pct,
        DROP COLUMN IF EXISTS predicted_p_source,
        DROP COLUMN IF EXISTS predicted_p_confidence,
        DROP COLUMN IF EXISTS magnitude_n,
        DROP COLUMN IF EXISTS magnitude_fallback_level,
        DROP COLUMN IF EXISTS regime,
        DROP COLUMN IF EXISTS cap_bucket_at_prediction,
        DROP COLUMN IF EXISTS priced_in_fraction,
        DROP COLUMN IF EXISTS priced_in_method,
        DROP COLUMN IF EXISTS priced_in_ratio_value,
        DROP COLUMN IF EXISTS priced_in_options_value,
        DROP COLUMN IF EXISTS disagreement_pp,
        DROP COLUMN IF EXISTS disagreement_verdict,
        DROP COLUMN IF EXISTS disagreement_reasoning,
        DROP COLUMN IF EXISTS prediction_source_v2,
        DROP COLUMN IF EXISTS abstained,
        DROP COLUMN IF EXISTS abstain_reason
    """)
