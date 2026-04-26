"""004 — split probability, per-share NPV, scenarios

Adds fields to capture the model improvements from the methodology audit:
- catalyst_universe.p_event_occurs and p_positive_outcome — split the
  ambiguous 'probability' field into timing-certainty vs success-probability
- post_catalyst_outcomes.options_implied_move_pct — straddle-based market
  consensus, captured at backfill time for the catalyst window
- post_catalyst_outcomes.shares_outstanding_at_event — for per-share normalization
"""
from alembic import op


revision = "004_methodology_audit"
down_revision = "003_llm_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE catalyst_universe
        ADD COLUMN IF NOT EXISTS p_event_occurs NUMERIC,
        ADD COLUMN IF NOT EXISTS p_positive_outcome NUMERIC
    """)
    # Backfill from existing confidence_score: assume confidence_score IS p_positive_outcome
    # (the more useful interpretation), and p_event_occurs = max(0.7, confidence_score) since
    # most LLM-discovered catalysts are ones the LLM is confident the event will happen
    op.execute("""
        UPDATE catalyst_universe
        SET p_positive_outcome = COALESCE(p_positive_outcome, confidence_score),
            p_event_occurs = COALESCE(p_event_occurs, GREATEST(0.7, confidence_score))
        WHERE confidence_score IS NOT NULL
          AND (p_positive_outcome IS NULL OR p_event_occurs IS NULL)
    """)

    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS options_implied_move_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS options_implied_move_source TEXT,
        ADD COLUMN IF NOT EXISTS shares_outstanding_at_event NUMERIC
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE catalyst_universe
        DROP COLUMN IF EXISTS p_event_occurs,
        DROP COLUMN IF EXISTS p_positive_outcome
    """)
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS options_implied_move_pct,
        DROP COLUMN IF EXISTS options_implied_move_source,
        DROP COLUMN IF EXISTS shares_outstanding_at_event
    """)
