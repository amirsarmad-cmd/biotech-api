"""Initial universe schema — 10 tables for biotech screener v2

Revision ID: 001_universe
Revises: 
Create Date: 2026-04-25

Creates the new schema for:
  - catalyst_universe — broad universe of upcoming non-earnings catalysts
  - earnings_dates — separate, informational
  - catalyst_npv_cache — NPV per (catalyst, slider params)
  - ai_analysis_cache — consensus + news-impact LLM results
  - stock_risk_factors — stock+drug+catalyst-specific risk analysis
  - drug_economics_cache — patient pop, pricing, penetration, timeline
  - historical_catalyst_moves — reference dataset for typical move sizes
  - stock_scores — long + short scores
  - npv_defaults — global default NPV/weights config
  - cron_runs — job log

Old screener_* tables are left untouched and will be dual-written during transition.
"""
from alembic import op
import sqlalchemy as sa

revision = "001_universe"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. catalyst_universe
    op.execute("""
        CREATE TABLE IF NOT EXISTS catalyst_universe (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            company_name TEXT,
            catalyst_type TEXT NOT NULL,
            catalyst_date DATE,
            date_precision TEXT DEFAULT 'exact',
            description TEXT,
            drug_name TEXT,
            canonical_drug_name TEXT,
            indication TEXT,
            phase TEXT,
            source TEXT,
            source_url TEXT,
            confidence_score NUMERIC,
            verified BOOLEAN DEFAULT FALSE,
            status TEXT DEFAULT 'active',
            superseded_by INTEGER REFERENCES catalyst_universe(id) ON DELETE SET NULL,
            superseded_at TIMESTAMPTZ,
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(ticker, catalyst_type, catalyst_date, drug_name)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_cu_ticker ON catalyst_universe(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cu_date ON catalyst_universe(catalyst_date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cu_status ON catalyst_universe(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cu_phase ON catalyst_universe(phase)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cu_indication ON catalyst_universe(indication)")

    # 2. earnings_dates
    op.execute("""
        CREATE TABLE IF NOT EXISTS earnings_dates (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            earnings_date DATE,
            fiscal_period TEXT,
            confirmed BOOLEAN DEFAULT FALSE,
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(ticker, earnings_date)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_earn_ticker ON earnings_dates(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_earn_date ON earnings_dates(earnings_date)")

    # 3. catalyst_npv_cache
    op.execute("""
        CREATE TABLE IF NOT EXISTS catalyst_npv_cache (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            catalyst_id INTEGER REFERENCES catalyst_universe(id) ON DELETE RESTRICT,
            params_hash TEXT NOT NULL,
            source_news_hash TEXT,
            drug_npv_b NUMERIC,
            p_approval NUMERIC,
            p_commercial NUMERIC,
            peak_sales_b NUMERIC,
            multiple NUMERIC,
            expected_pct NUMERIC,
            full_payload JSONB,
            computed_at TIMESTAMPTZ DEFAULT NOW(),
            ttl TIMESTAMPTZ,
            UNIQUE(ticker, catalyst_id, params_hash)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_npv_ticker ON catalyst_npv_cache(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_npv_catalyst ON catalyst_npv_cache(catalyst_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_npv_ttl ON catalyst_npv_cache(ttl)")

    # 4. ai_analysis_cache
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai_analysis_cache (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            catalyst_id INTEGER REFERENCES catalyst_universe(id) ON DELETE RESTRICT,
            analysis_type TEXT NOT NULL,
            source_news_hash TEXT,
            result JSONB,
            models_used JSONB,
            computed_at TIMESTAMPTZ DEFAULT NOW(),
            ttl TIMESTAMPTZ,
            UNIQUE(ticker, catalyst_id, analysis_type)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ai_ticker ON ai_analysis_cache(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ai_type ON ai_analysis_cache(analysis_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ai_ttl ON ai_analysis_cache(ttl)")

    # 5. stock_risk_factors
    op.execute("""
        CREATE TABLE IF NOT EXISTS stock_risk_factors (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            catalyst_id INTEGER REFERENCES catalyst_universe(id) ON DELETE RESTRICT,
            drug_name TEXT,
            computed_at TIMESTAMPTZ DEFAULT NOW(),
            ttl TIMESTAMPTZ,
            factors JSONB,
            prior_crls JSONB,
            active_litigation JSONB,
            insider_transactions JSONB,
            short_data JSONB,
            source_news_hash TEXT,
            news_articles_used JSONB,
            llm_provider TEXT,
            UNIQUE(ticker, catalyst_id, drug_name)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_risk_ticker ON stock_risk_factors(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_risk_recent ON stock_risk_factors(ticker, computed_at DESC)")

    # 6. drug_economics_cache
    op.execute("""
        CREATE TABLE IF NOT EXISTS drug_economics_cache (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            canonical_drug_name TEXT NOT NULL,
            indication TEXT,
            addressable_population_us BIGINT,
            addressable_population_global BIGINT,
            annual_cost_min_usd NUMERIC,
            annual_cost_max_usd NUMERIC,
            standard_of_care_cost_usd NUMERIC,
            penetration_min_pct NUMERIC,
            penetration_max_pct NUMERIC,
            penetration_mid_pct NUMERIC,
            launch_year INTEGER,
            peak_sales_year INTEGER,
            patent_expiry_date DATE,
            competitors JSONB,
            competitive_intensity TEXT,
            first_in_class BOOLEAN,
            llm_rationale TEXT,
            llm_provider TEXT,
            computed_at TIMESTAMPTZ DEFAULT NOW(),
            ttl TIMESTAMPTZ,
            UNIQUE(ticker, canonical_drug_name)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_drug_ticker ON drug_economics_cache(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_drug_indication ON drug_economics_cache(indication)")

    # 7. historical_catalyst_moves
    op.execute("""
        CREATE TABLE IF NOT EXISTS historical_catalyst_moves (
            id SERIAL PRIMARY KEY,
            catalyst_type TEXT NOT NULL,
            indication TEXT,
            market_cap_bucket TEXT,
            mean_move_pct NUMERIC,
            p25_move_pct NUMERIC,
            p75_move_pct NUMERIC,
            std_dev_pct NUMERIC,
            n_observations INTEGER,
            source TEXT,
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(catalyst_type, indication, market_cap_bucket)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_hcm_type ON historical_catalyst_moves(catalyst_type)")

    # 8. stock_scores
    op.execute("""
        CREATE TABLE IF NOT EXISTS stock_scores (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            overall_score NUMERIC,
            short_score NUMERIC,
            short_thesis JSONB,
            computed_at TIMESTAMPTZ DEFAULT NOW(),
            ttl TIMESTAMPTZ,
            UNIQUE(ticker, computed_at)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_scores_ticker ON stock_scores(ticker)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_scores_overall ON stock_scores(overall_score DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_scores_short ON stock_scores(short_score DESC)")

    # 9. npv_defaults (singleton row config)
    # Note: rating_weights default is set via UPDATE after CREATE+INSERT to avoid
    # SQLAlchemy's bindparam parser misinterpreting ':' in JSON literal.
    op.execute("""
        CREATE TABLE IF NOT EXISTS npv_defaults (
            id SERIAL PRIMARY KEY,
            scope TEXT DEFAULT 'global',
            discount_rate NUMERIC DEFAULT 0.12,
            tax_rate NUMERIC DEFAULT 0.21,
            cogs_pct NUMERIC DEFAULT 0.15,
            default_penetration_pct NUMERIC DEFAULT 0.15,
            default_time_to_peak_years INTEGER DEFAULT 5,
            rating_weights JSONB,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(scope)
        )
    """)
    # Seed singleton row + populate JSONB default
    op.execute("""
        INSERT INTO npv_defaults (scope) VALUES ('global')
        ON CONFLICT (scope) DO NOTHING
    """)
    # Use sa.text() with explicit bindparams=[] to disable colon parsing
    from sqlalchemy import text
    op.execute(
        text("UPDATE npv_defaults SET rating_weights = CAST(:weights AS JSONB) WHERE scope = 'global' AND rating_weights IS NULL")
        .bindparams(weights='{"catalyst_probability": 0.35, "news_sentiment": 0.15, "news_activity": 0.10, "market_cap": 0.10, "days_proximity": 0.30}')
    )

    # 10. cron_runs
    op.execute("""
        CREATE TABLE IF NOT EXISTS cron_runs (
            id SERIAL PRIMARY KEY,
            job_name TEXT NOT NULL,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            status TEXT,
            records_processed INTEGER,
            records_added INTEGER,
            records_updated INTEGER,
            errors JSONB,
            log TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_cron_job ON cron_runs(job_name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cron_started ON cron_runs(started_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS cron_runs CASCADE")
    op.execute("DROP TABLE IF EXISTS npv_defaults CASCADE")
    op.execute("DROP TABLE IF EXISTS stock_scores CASCADE")
    op.execute("DROP TABLE IF EXISTS historical_catalyst_moves CASCADE")
    op.execute("DROP TABLE IF EXISTS drug_economics_cache CASCADE")
    op.execute("DROP TABLE IF EXISTS stock_risk_factors CASCADE")
    op.execute("DROP TABLE IF EXISTS ai_analysis_cache CASCADE")
    op.execute("DROP TABLE IF EXISTS catalyst_npv_cache CASCADE")
    op.execute("DROP TABLE IF EXISTS earnings_dates CASCADE")
    op.execute("DROP TABLE IF EXISTS catalyst_universe CASCADE")
