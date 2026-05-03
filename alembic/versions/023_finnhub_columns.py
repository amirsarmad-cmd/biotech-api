"""023 — Finnhub-sourced columns on catalyst_event_features.

Three signal categories from Finnhub free tier (60 calls/min):
  1. Insider transactions / sentiment (Form 4 cleanly parsed)
  2. Analyst price-target DISPERSION (count + low/high/median, not just mean —
     dispersion is itself a calibration signal)
  3. Analyst recommendation trends (buy/hold/sell counts over time, change vs 3mo ago)
  4. News sentiment (Finnhub's bullish/bearish/buzz scores)

Revision ID: 023_finnhub_columns
"""
from alembic import op


revision = "023_finnhub_columns"
down_revision = "022_event_features_extended"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        ALTER TABLE catalyst_event_features
          -- Insider transactions (replaces SEC Form 4 stub which never got built)
          ADD COLUMN IF NOT EXISTS finnhub_insider_buys_30d_pre        INTEGER,
          ADD COLUMN IF NOT EXISTS finnhub_insider_sells_30d_pre       INTEGER,
          ADD COLUMN IF NOT EXISTS finnhub_insider_net_value_usd_30d_pre NUMERIC,
          ADD COLUMN IF NOT EXISTS finnhub_insider_sentiment_3m         NUMERIC,  -- avg mspr (monthly share purchase ratio)

          -- Analyst price-target dispersion (Finviz only gives a single Target Price)
          ADD COLUMN IF NOT EXISTS finnhub_target_high_usd     NUMERIC,
          ADD COLUMN IF NOT EXISTS finnhub_target_low_usd      NUMERIC,
          ADD COLUMN IF NOT EXISTS finnhub_target_median_usd   NUMERIC,
          ADD COLUMN IF NOT EXISTS finnhub_target_dispersion_pct NUMERIC,  -- (high-low)/median × 100

          -- Analyst recommendation trends (snapshot + delta vs 3mo ago)
          ADD COLUMN IF NOT EXISTS finnhub_buy_count            INTEGER,
          ADD COLUMN IF NOT EXISTS finnhub_hold_count           INTEGER,
          ADD COLUMN IF NOT EXISTS finnhub_sell_count           INTEGER,
          ADD COLUMN IF NOT EXISTS finnhub_buy_count_change_3m  INTEGER,

          -- News sentiment (totally new feature; we had no pre-event sentiment data)
          ADD COLUMN IF NOT EXISTS finnhub_news_bullish_pct       NUMERIC,
          ADD COLUMN IF NOT EXISTS finnhub_news_bearish_pct       NUMERIC,
          ADD COLUMN IF NOT EXISTS finnhub_news_buzz_articles_week INTEGER,
          ADD COLUMN IF NOT EXISTS finnhub_news_buzz_score          NUMERIC,
          ADD COLUMN IF NOT EXISTS finnhub_company_news_score       NUMERIC,

          ADD COLUMN IF NOT EXISTS finnhub_data_source TEXT
        ;
        """
    )


def downgrade():
    op.execute(
        """
        ALTER TABLE catalyst_event_features
          DROP COLUMN IF EXISTS finnhub_insider_buys_30d_pre,
          DROP COLUMN IF EXISTS finnhub_insider_sells_30d_pre,
          DROP COLUMN IF EXISTS finnhub_insider_net_value_usd_30d_pre,
          DROP COLUMN IF EXISTS finnhub_insider_sentiment_3m,
          DROP COLUMN IF EXISTS finnhub_target_high_usd,
          DROP COLUMN IF EXISTS finnhub_target_low_usd,
          DROP COLUMN IF EXISTS finnhub_target_median_usd,
          DROP COLUMN IF EXISTS finnhub_target_dispersion_pct,
          DROP COLUMN IF EXISTS finnhub_buy_count,
          DROP COLUMN IF EXISTS finnhub_hold_count,
          DROP COLUMN IF EXISTS finnhub_sell_count,
          DROP COLUMN IF EXISTS finnhub_buy_count_change_3m,
          DROP COLUMN IF EXISTS finnhub_news_bullish_pct,
          DROP COLUMN IF EXISTS finnhub_news_bearish_pct,
          DROP COLUMN IF EXISTS finnhub_news_buzz_articles_week,
          DROP COLUMN IF EXISTS finnhub_news_buzz_score,
          DROP COLUMN IF EXISTS finnhub_company_news_score,
          DROP COLUMN IF EXISTS finnhub_data_source
        ;
        """
    )
