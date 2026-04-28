"""014 — sector-adjusted runup features

Per ChatGPT critique on the V2 priced-in classifier:
  'Raw runup is weak. Replace with runup_30d_vs_xbi.'

Without sector adjustment, a stock that ran +20% during a +25% biotech
rally gets miscategorized as 'priced in' when it actually underperformed
its sector. Catalyst-specific runup is what should drive the priced-in
filter, not whole-sector beta.

Adds:
  sector_runup_30d_pct          — XBI % move over the same 30d window
  runup_pre_event_30d_vs_xbi_pct — stock_runup - sector_runup
                                   (catalyst-specific runup; the actual signal)

The V2 classifier will prefer runup_vs_xbi when available, fall back to
raw runup_pre_event_30d_pct when XBI data is missing (older catalysts).

Revision ID: 014_sector_runup (16 chars, well under varchar(32))
"""
from alembic import op


revision = '014_sector_runup'
down_revision = '013_priced_in_features'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS sector_runup_30d_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS runup_pre_event_30d_vs_xbi_pct NUMERIC
    """)


def downgrade():
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS sector_runup_30d_pct,
        DROP COLUMN IF EXISTS runup_pre_event_30d_vs_xbi_pct
    """)
