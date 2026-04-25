"""post_catalyst_outcomes — track predicted vs actual catalyst outcomes

Revision ID: 002_post_catalyst
Revises: 001_universe
Create Date: 2026-04-25

Adds a single table:
  - post_catalyst_outcomes — for each historical catalyst, stores pre/post
    prices, actual % moves at multiple horizons, the prediction we had at
    the time (NPV / probability), the inferred outcome (approved/rejected/
    delayed), and prediction error. Enables a learning loop: prediction vs
    reality history → recalibration of base rates.
"""
from alembic import op
import sqlalchemy as sa

revision = "002_post_catalyst"
down_revision = "001_universe"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS post_catalyst_outcomes (
            id SERIAL PRIMARY KEY,
            catalyst_id INTEGER REFERENCES catalyst_universe(id) ON DELETE SET NULL,
            ticker TEXT NOT NULL,
            catalyst_type TEXT,
            catalyst_date TEXT NOT NULL,           -- ISO YYYY-MM-DD
            drug_name TEXT,
            indication TEXT,

            -- Pre-event price baseline (close on trading day before catalyst)
            pre_event_date TEXT,
            pre_event_price NUMERIC,

            -- Post-event prices at standard horizons
            day0_price NUMERIC,                    -- close on catalyst date itself
            day1_price NUMERIC,                    -- next trading day close
            day7_price NUMERIC,                    -- ~5 trading days later
            day30_price NUMERIC,                   -- ~22 trading days later

            -- Computed actual moves (% from pre_event_price)
            actual_move_pct_1d NUMERIC,
            actual_move_pct_7d NUMERIC,
            actual_move_pct_30d NUMERIC,

            -- Volume / volatility around the event
            preevent_avg_volume_30d NUMERIC,
            postevent_volume_1d NUMERIC,
            postevent_max_intraday_move_pct NUMERIC,

            -- Predictions captured at the time (snapshotted from npv_cache / catalyst_universe)
            predicted_prob NUMERIC,                -- catalyst_universe.confidence_score / probability
            predicted_npv_b NUMERIC,               -- rnpv_b at the time
            predicted_move_pct NUMERIC,            -- expected % move (typical reference moves)
            prediction_source TEXT,                -- 'rnpv_v2' | 'legacy_npv' | 'reference_move'
            predicted_at TIMESTAMPTZ,              -- when we made the prediction

            -- Inferred outcome
            outcome TEXT,                          -- 'approved' | 'rejected' | 'delayed' | 'mixed' | 'unknown'
            outcome_confidence NUMERIC,            -- 0-1 confidence in classification
            outcome_source TEXT,                   -- 'price_action' | 'news' | 'manual' | 'press_release'
            outcome_notes TEXT,                    -- short summary of basis

            -- Error metrics
            error_abs_pct NUMERIC,                 -- |predicted - actual_30d|
            error_signed_pct NUMERIC,              -- predicted - actual_30d (sign matters for bias)
            direction_correct BOOLEAN,             -- did predicted direction match actual?

            -- Bookkeeping
            computed_at TIMESTAMPTZ DEFAULT NOW(),
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            backfill_attempts INTEGER DEFAULT 0,
            last_error TEXT,

            UNIQUE(ticker, catalyst_type, catalyst_date)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_pco_ticker ON post_catalyst_outcomes(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pco_date ON post_catalyst_outcomes(catalyst_date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pco_catalyst ON post_catalyst_outcomes(catalyst_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pco_outcome ON post_catalyst_outcomes(outcome)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS post_catalyst_outcomes CASCADE")
