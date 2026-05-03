"""021 — catalyst_event_features (the feature store for backtest-driven algo iteration)

User-driven (2026-05-03): "lets finish our data feature store make it robust
band expand data sets. if you need other data set inputs let me know with
suggestions." — i.e. snapshot every feature any future algo could want at
the catalyst-event grain, ONCE, so when we change algorithms we don't re-fetch
data per iteration. Replaces the implicit "compute features on the fly"
pattern that left the V2 scenario backtest stuck at 302 of 5000 events.

One row per (catalyst_id, ticker, catalyst_date). All columns nullable; the
backfiller (services/feature_store.py) populates what its sources can fill.
A second pass adds LLM-enrichment columns (drug_npv, priced_in_fraction)
when Gemini is available; in the meantime, all the non-LLM features are
fillable today.

Sources, by category:
  - Stock state          → screener_stocks + yfinance history
  - Capital structure    → SEC EDGAR (sec_financials.py)
  - Price action         → yfinance history (already cached)
  - Peer-relative        → XBI / IBB price history
  - Microstructure       → screener_stocks + yfinance volume
  - Options              → POLYGON (historical chain — backfills > 1y),
                           yfinance (recent only)
  - Insider / institutional → SEC EDGAR Form 4 + Form 13F
  - LLM-enrichment       → catalyst_npv_cache (Gemini-bound)
  - Multi-source labels  → LLM gateway (gemini/anthropic/openai) + price-proxy

Revision ID: 021_catalyst_event_features
"""
from alembic import op


revision = "021_catalyst_event_features"
down_revision = "020_prediction_v2_columns"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS catalyst_event_features (
            id SERIAL PRIMARY KEY,
            catalyst_id INTEGER REFERENCES catalyst_universe(id) ON DELETE CASCADE,
            ticker TEXT NOT NULL,
            catalyst_date TEXT NOT NULL,
            catalyst_type TEXT,
            UNIQUE (catalyst_id),

            -- ─── Stock state (yfinance + screener_stocks) ─────────────────
            market_cap_at_date_m       NUMERIC,
            shares_out_at_date_m       NUMERIC,
            pre_event_price            NUMERIC,        -- mirrored from pco for join convenience

            -- ─── Capital structure (SEC EDGAR sec_financials) ─────────────
            cash_at_date_m             NUMERIC,
            debt_at_date_m             NUMERIC,
            net_cash_at_date_m         NUMERIC,
            runway_months_at_date      NUMERIC,
            sec_filing_date_used       TEXT,           -- which 10-Q/K we pulled from

            -- ─── Price action (yfinance) ───────────────────────────────────
            runup_pct_30d              NUMERIC,
            runup_pct_90d              NUMERIC,
            runup_pct_180d             NUMERIC,
            realized_vol_30d           NUMERIC,        -- annualized stdev of daily returns
            realized_vol_90d           NUMERIC,
            max_drawdown_30d           NUMERIC,        -- worst peak-to-trough in window

            -- ─── Peer-relative (XBI, IBB) ──────────────────────────────────
            xbi_runup_30d              NUMERIC,
            xbi_runup_90d              NUMERIC,
            ibb_runup_30d              NUMERIC,
            relative_strength_xbi_30d  NUMERIC,        -- ticker_runup - xbi_runup
            relative_strength_xbi_90d  NUMERIC,
            beta_to_xbi_180d           NUMERIC,

            -- ─── Microstructure ────────────────────────────────────────────
            short_interest_pct_at_date NUMERIC,
            short_ratio_days_at_date   NUMERIC,        -- short int / avg daily volume
            avg_volume_30d             NUMERIC,
            volume_ratio_t_minus_1     NUMERIC,        -- (T-1 volume) / avg_30d

            -- ─── Options (Polygon historical + yfinance recent) ────────────
            options_implied_move_pct   NUMERIC,
            atm_iv_at_date             NUMERIC,
            iv_skew_25d                NUMERIC,        -- 25d call IV - 25d put IV
            put_call_oi_ratio          NUMERIC,
            put_call_volume_ratio      NUMERIC,
            options_source             TEXT,           -- 'polygon' | 'yfinance' | NULL

            -- ─── Insider transactions (SEC EDGAR Form 4) ───────────────────
            insider_buys_count_30d_pre   INTEGER,
            insider_sells_count_30d_pre  INTEGER,
            insider_net_value_usd_30d_pre NUMERIC,    -- (buy $) - (sell $) by insiders
            insider_buyer_count_30d_pre  INTEGER,     -- distinct insiders buying

            -- ─── Institutional holdings (SEC EDGAR Form 13F) ───────────────
            institutional_pct_at_quarter NUMERIC,     -- most recent 13F before catalyst date
            institutional_change_qoq_pp  NUMERIC,     -- QoQ change in % held
            top_10_holders_concentration NUMERIC,

            -- ─── Catalyst metadata snapshot ────────────────────────────────
            p_approval_at_pred         NUMERIC,        -- catalyst_universe.confidence_score at backfill time
            days_until_catalyst_at_pred INTEGER,
            regime                     TEXT,           -- MANDATED / SEMI_MANDATED / VOLUNTARY
            cap_bucket                 TEXT,           -- micro_lt500m / small_500m_2b / mid_or_above
            product_class              TEXT,           -- from drug_programs.classify_product

            -- ─── LLM-enrichment (filled when Gemini cap returns) ───────────
            drug_npv_b_at_date         NUMERIC,
            raw_drug_npv_b_at_date     NUMERIC,        -- unrisked NPV (= "if approved" value)
            priced_in_fraction_at_date NUMERIC,
            priced_in_method_used      TEXT,           -- 'ratio' | 'options' | 'blend'
            peak_sales_b               NUMERIC,
            time_to_peak_years         NUMERIC,
            discount_rate_used         NUMERIC,
            multiple_used              NUMERIC,

            -- ─── Multi-source outcome labels (cross-validation) ────────────
            outcome_label_gemini       TEXT,           -- mirror of pco.outcome_label_class for convenience
            outcome_label_anthropic    TEXT,           -- when LLM gateway anthropic side is used
            outcome_label_openai       TEXT,           -- when LLM gateway openai side is used
            outcome_label_consensus    TEXT,           -- majority-vote when ≥2 sources agree
            outcome_label_price_proxy  TEXT,           -- 'POSITIVE'/'NEGATIVE'/'MIXED' from 7d move
            outcome_confidence         NUMERIC,        -- 0-1 composite

            -- ─── Realized outcomes (mirrored from pco for one-row queries) ─
            actual_move_pct_1d         NUMERIC,
            actual_move_pct_7d         NUMERIC,
            actual_move_pct_30d        NUMERIC,

            -- ─── Bookkeeping ──────────────────────────────────────────────
            backfill_version           INTEGER NOT NULL DEFAULT 1,
            backfill_source_status     JSONB,          -- {price_action: 'ok', sec_edgar: 'no_filing', polygon: 'no_chain', ...}
            backfilled_at              TIMESTAMPTZ DEFAULT NOW(),
            llm_enriched_at            TIMESTAMPTZ,    -- set when Gemini-bound columns populated
            updated_at                 TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cef_ticker_date
          ON catalyst_event_features (ticker, catalyst_date);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cef_catalyst_type_cap
          ON catalyst_event_features (catalyst_type, cap_bucket);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cef_consensus
          ON catalyst_event_features (outcome_label_consensus)
          WHERE outcome_label_consensus IS NOT NULL;
        """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_cef_consensus")
    op.execute("DROP INDEX IF EXISTS idx_cef_catalyst_type_cap")
    op.execute("DROP INDEX IF EXISTS idx_cef_ticker_date")
    op.execute("DROP TABLE IF EXISTS catalyst_event_features")
