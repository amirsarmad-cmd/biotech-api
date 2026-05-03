"""024 — Finviz scrape extensions (news / ratings / insider) + dynamic-refresh tracking.

These three sources change continuously (new headlines daily, analyst
upgrades weekly, insider Form 4 with 2-day delay). Snapshot at backfill
time + a separate daily refresh job for upcoming events keeps them fresh
without re-fetching the whole event.

Revision ID: 024_finviz_dynamic_columns
"""
from alembic import op


revision = "024_finviz_dynamic_columns"
down_revision = "023_finnhub_columns"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        ALTER TABLE catalyst_event_features
          -- News (Finviz quote-page news table — last ~100 headlines)
          ADD COLUMN IF NOT EXISTS finviz_news_count_7d_pre   INTEGER,
          ADD COLUMN IF NOT EXISTS finviz_news_count_30d_pre  INTEGER,
          ADD COLUMN IF NOT EXISTS finviz_news_source_count_30d INTEGER,  -- distinct sources
          ADD COLUMN IF NOT EXISTS finviz_news_top_sources    TEXT,        -- comma-separated top 5
          ADD COLUMN IF NOT EXISTS finviz_news_high_quality_present BOOLEAN, -- Bloomberg/Reuters/WSJ/FT
          ADD COLUMN IF NOT EXISTS finviz_news_latest_date    TEXT,

          -- Analyst ratings (Finviz quote-page ratings table — chronological events)
          ADD COLUMN IF NOT EXISTS finviz_analyst_upgrades_30d_pre   INTEGER,
          ADD COLUMN IF NOT EXISTS finviz_analyst_downgrades_30d_pre INTEGER,
          ADD COLUMN IF NOT EXISTS finviz_analyst_pt_changes_30d_pre INTEGER,
          ADD COLUMN IF NOT EXISTS finviz_analyst_latest_action      TEXT,
          ADD COLUMN IF NOT EXISTS finviz_analyst_latest_firm        TEXT,
          ADD COLUMN IF NOT EXISTS finviz_analyst_latest_date        TEXT,

          -- Insider transactions full table (more granular than Finnhub free's totals)
          ADD COLUMN IF NOT EXISTS finviz_insider_buys_named_30d         INTEGER,  -- non-grant buys
          ADD COLUMN IF NOT EXISTS finviz_insider_sells_named_30d        INTEGER,
          ADD COLUMN IF NOT EXISTS finviz_insider_buy_value_named_usd_30d NUMERIC,
          ADD COLUMN IF NOT EXISTS finviz_insider_top_buyer_title         TEXT,    -- e.g. "CEO", "Director"

          -- Bookkeeping for the daily refresh job
          ADD COLUMN IF NOT EXISTS dynamic_refresh_at TIMESTAMPTZ
        ;
        """
    )


def downgrade():
    op.execute(
        """
        ALTER TABLE catalyst_event_features
          DROP COLUMN IF EXISTS finviz_news_count_7d_pre,
          DROP COLUMN IF EXISTS finviz_news_count_30d_pre,
          DROP COLUMN IF EXISTS finviz_news_source_count_30d,
          DROP COLUMN IF EXISTS finviz_news_top_sources,
          DROP COLUMN IF EXISTS finviz_news_high_quality_present,
          DROP COLUMN IF EXISTS finviz_news_latest_date,
          DROP COLUMN IF EXISTS finviz_analyst_upgrades_30d_pre,
          DROP COLUMN IF EXISTS finviz_analyst_downgrades_30d_pre,
          DROP COLUMN IF EXISTS finviz_analyst_pt_changes_30d_pre,
          DROP COLUMN IF EXISTS finviz_analyst_latest_action,
          DROP COLUMN IF EXISTS finviz_analyst_latest_firm,
          DROP COLUMN IF EXISTS finviz_analyst_latest_date,
          DROP COLUMN IF EXISTS finviz_insider_buys_named_30d,
          DROP COLUMN IF EXISTS finviz_insider_sells_named_30d,
          DROP COLUMN IF EXISTS finviz_insider_buy_value_named_usd_30d,
          DROP COLUMN IF EXISTS finviz_insider_top_buyer_title,
          DROP COLUMN IF EXISTS dynamic_refresh_at
        ;
        """
    )
