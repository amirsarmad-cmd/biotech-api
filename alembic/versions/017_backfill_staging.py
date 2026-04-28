"""017 — backfill staging table for historical catalysts

Per user decision to backfill 10 years (2015-2025) one-shot from
SEC EDGAR → Finnhub → BioPharmaCatalyst.

The staging table receives raw candidate events from all three scrapers
without committing them to catalyst_universe. A separate normalize
+ dedupe + LLM-label pipeline merges accepted candidates upstream.

This separation is important because:
  - EDGAR will return ~50k 8-K filings; most are non-clinical
  - Filtering happens in two stages (regex → LLM) and we want to be
    able to retry the LLM stage without re-scraping
  - Dedupe across sources is non-trivial (different date precision,
    drug name variants); we need a buffer to resolve conflicts

Schema:
  source           — 'edgar', 'finnhub', 'biopharmacatalyst', 'manual'
  source_id        — unique within source (e.g. 'edgar:0001234567-24-001234')
  ticker           — best-effort, sometimes NULL for EDGAR (CIK only)
  cik              — for EDGAR rows
  filing_date      — when the source first reported the event
  catalyst_date    — extracted event date (may be the same as filing_date)
  date_precision   — 'exact' / 'day' / 'month' / 'quarter' / 'unknown'
  catalyst_type    — pre-classification ('FDA Decision' / 'Phase 2 Readout' / etc)
  drug_name        — extracted drug name when available
  indication       — extracted indication when available
  raw_title        — source's headline / 8-K item title
  raw_text_excerpt — first ~2000 chars of source text for LLM extraction
  source_url       — link to original document

  status           — 'pending' / 'accepted' / 'rejected' / 'duplicate' / 'unclear'
  reject_reason    — when status='rejected'
  catalyst_id      — FK into catalyst_universe when accepted (NULL otherwise)

  scraped_at       — when row was created
  processed_at     — when LLM normalization ran
  normalized_json  — full LLM output (catalyst_type, drug, indication, date_precision)

Revision ID: 017_backfill_staging (20 chars)
"""
from alembic import op


revision = '017_backfill_staging'
down_revision = '016_outcome_labels'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS catalyst_backfill_staging (
            id BIGSERIAL PRIMARY KEY,

            -- Provenance
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            UNIQUE (source, source_id),

            -- Identifiers
            ticker TEXT,
            cik TEXT,

            -- Timing
            filing_date DATE,
            catalyst_date DATE,
            date_precision TEXT DEFAULT 'unknown',

            -- Pre-classification (regex / heuristic)
            catalyst_type TEXT,
            drug_name TEXT,
            indication TEXT,

            -- Raw source data
            raw_title TEXT,
            raw_text_excerpt TEXT,
            source_url TEXT,

            -- Pipeline status
            status TEXT NOT NULL DEFAULT 'pending',
            reject_reason TEXT,
            catalyst_id INTEGER,
            normalized_json JSONB,

            -- Audit
            scraped_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            processed_at TIMESTAMP WITH TIME ZONE
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_bks_status_source
        ON catalyst_backfill_staging(status, source)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_bks_ticker_filing_date
        ON catalyst_backfill_staging(ticker, filing_date)
        WHERE ticker IS NOT NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_bks_cik
        ON catalyst_backfill_staging(cik)
        WHERE cik IS NOT NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_bks_pending
        ON catalyst_backfill_staging(scraped_at)
        WHERE status = 'pending'
    """)
    # Audit table for batch runs (keeps history of scraper invocations)
    op.execute("""
        CREATE TABLE IF NOT EXISTS catalyst_backfill_runs (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            ended_at TIMESTAMP WITH TIME ZONE,
            year_range TEXT,
            params_json JSONB,
            rows_scraped INTEGER DEFAULT 0,
            rows_inserted INTEGER DEFAULT 0,
            rows_skipped INTEGER DEFAULT 0,
            rows_errored INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running',
            error_message TEXT
        )
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS catalyst_backfill_staging")
    op.execute("DROP TABLE IF EXISTS catalyst_backfill_runs")
