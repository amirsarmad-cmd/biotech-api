"""008 — abnormal return columns on post_catalyst_outcomes

ChatGPT critique #4: 'Your post-catalyst tracker should compare
stock_move - XBI_move (sector basket), not raw stock move.'

A +5% biotech move on a +5% XBI day has zero alpha — the catalyst didn't
deliver real outperformance. Without subtracting sector beta, our backtest
metrics conflate catalyst-driven moves with market drift.

Schema:
  - sector_move_pct_1d / 7d / 30d:  raw % move of XBI ETF over same window
  - abnormal_move_pct_1d / 7d / 30d: stock - sector
  - sector_basket: which basket was used (default 'XBI', could later be IBB
    or matched-cap basket)

These are computed at backfill time alongside actual_move_pct_*. Existing
rows can be retroactively populated via /admin/post-catalyst/recompute-abnormal.
"""
from alembic import op


revision = "008_abnormal_returns"
down_revision = "007_options_implied_meta"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS sector_basket TEXT DEFAULT 'XBI',
        ADD COLUMN IF NOT EXISTS sector_move_pct_1d NUMERIC,
        ADD COLUMN IF NOT EXISTS sector_move_pct_7d NUMERIC,
        ADD COLUMN IF NOT EXISTS sector_move_pct_30d NUMERIC,
        ADD COLUMN IF NOT EXISTS abnormal_move_pct_1d NUMERIC,
        ADD COLUMN IF NOT EXISTS abnormal_move_pct_7d NUMERIC,
        ADD COLUMN IF NOT EXISTS abnormal_move_pct_30d NUMERIC
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS sector_basket,
        DROP COLUMN IF EXISTS sector_move_pct_1d,
        DROP COLUMN IF EXISTS sector_move_pct_7d,
        DROP COLUMN IF EXISTS sector_move_pct_30d,
        DROP COLUMN IF EXISTS abnormal_move_pct_1d,
        DROP COLUMN IF EXISTS abnormal_move_pct_7d,
        DROP COLUMN IF EXISTS abnormal_move_pct_30d
    """)
