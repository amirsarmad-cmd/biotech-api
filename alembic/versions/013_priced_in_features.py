"""013 — priced-in features for signal classification

User analysis after observing tradeable accuracy = 31.7% (inverse 68.3%):
  'The old signal is not useless — it is telling you popular/high-confidence
   biotech catalysts often underperform after the catalyst. That is a
   sell-the-news factor. Now build the classifier around priced-in score,
   not just probability.'

Add the priced-in features needed to differentiate:
  high P(approval) + clean setup     → LONG_UNDERPRICED_POSITIVE
  high P(approval) + crowded setup   → SHORT_SELL_THE_NEWS
  low P(approval)                    → SHORT_LOW_PROBABILITY

Three columns added:

1. runup_pre_event_30d_pct — stock % move in the 30 days BEFORE the catalyst.
   Computed as (pre_event_price - price_30d_ago) / price_30d_ago * 100.
   Anything > +20% is a 'crowded long' — likely priced in.

2. priced_in_score — composite 0..1 score from runup + iv_euphoria +
   options_implied_move. Higher = more priced in.

3. signal_v2 — re-classified using the new logic. Stored separately from
   trade_signal so we can A/B compare in aggregate-v2.

Revision ID: 013_priced_in_features (22 chars, well under varchar(32))
"""
from alembic import op


revision = '013_priced_in_features'
down_revision = '012_abstention_layer'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS price_30d_before_event NUMERIC,
        ADD COLUMN IF NOT EXISTS runup_pre_event_30d_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS priced_in_score NUMERIC,
        ADD COLUMN IF NOT EXISTS signal_v2 TEXT,
        ADD COLUMN IF NOT EXISTS direction_correct_v2 BOOLEAN
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pco_signal_v2
        ON post_catalyst_outcomes(signal_v2)
        WHERE signal_v2 IS NOT NULL
    """)


def downgrade():
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS price_30d_before_event,
        DROP COLUMN IF EXISTS runup_pre_event_30d_pct,
        DROP COLUMN IF EXISTS priced_in_score,
        DROP COLUMN IF EXISTS signal_v2,
        DROP COLUMN IF EXISTS direction_correct_v2
    """)
