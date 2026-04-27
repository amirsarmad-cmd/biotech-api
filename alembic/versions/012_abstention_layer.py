"""012 — abstention layer + 3-day actuals

User feedback after seeing 52.5% direction accuracy on 358 catalysts:
  'Chasing 70% across all catalysts is overfitting. The right target is
   70% on tradeable subset with abstention. Replace 30D raw with 3D
   abnormal return vs XBI. Add trade_signal classification.'

Three changes:

1. actual_move_pct_3d / abnormal_move_pct_3d — fill the gap between 1d
   and 7d. Per the analysis, 3-day abnormal return is the canonical
   catalyst-effect window (1d for FDA/AdCom, 3-7d for trial readouts).

2. trade_signal TEXT — 'LONG' | 'SHORT' | one of the 'NO_TRADE_*' tiers.
   Set at backfill time so we can SELECT WHERE trade_signal IN ('LONG','SHORT')
   for tradeable-subset metrics.

3. tradeable BOOLEAN — convenience column (true if trade_signal in LONG/SHORT).
   error_abs_abnormal_3d_pct — magnitude error against 3d abnormal target.

Revision ID: 012_abstention_layer (16 chars, well under varchar(32))
"""
from alembic import op


revision = '012_abstention_layer'
down_revision = '011_manual_override'
branch_labels = None
depends_on = None


def upgrade():
    # 3-day actuals — fill the gap between 1d and 7d. Catalyst-effect
    # windows by event type:
    #   FDA / PDUFA / AdCom    → 1D (immediate gap fill or fade)
    #   Phase 2 / Phase 3      → 1-3D (data digestion)
    #   Partnerships / M&A     → 1D
    #   Earnings               → not catalyst-tradeable; ignore
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS actual_move_pct_3d NUMERIC,
        ADD COLUMN IF NOT EXISTS abnormal_move_pct_3d NUMERIC,
        ADD COLUMN IF NOT EXISTS sector_move_pct_3d NUMERIC,
        ADD COLUMN IF NOT EXISTS day3_price NUMERIC
    """)

    # Abstention: only score events where the model expressed a real edge.
    # trade_signal stores the classifier output. tradeable is the boolean
    # convenience for WHERE clauses + indexes. error_abs_abnormal_3d_pct
    # is the magnitude error against the canonical target.
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS trade_signal TEXT,
        ADD COLUMN IF NOT EXISTS tradeable BOOLEAN DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS error_abs_abnormal_3d_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS direction_correct_3d BOOLEAN
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pco_tradeable
        ON post_catalyst_outcomes(tradeable, trade_signal)
        WHERE tradeable = TRUE
    """)


def downgrade():
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS actual_move_pct_3d,
        DROP COLUMN IF EXISTS abnormal_move_pct_3d,
        DROP COLUMN IF EXISTS sector_move_pct_3d,
        DROP COLUMN IF EXISTS day3_price,
        DROP COLUMN IF EXISTS trade_signal,
        DROP COLUMN IF EXISTS tradeable,
        DROP COLUMN IF EXISTS error_abs_abnormal_3d_pct,
        DROP COLUMN IF EXISTS direction_correct_3d
    """)
