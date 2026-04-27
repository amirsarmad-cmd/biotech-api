"""011 — manual override + ingestion log

User feedback after AI chat critique: 'Build a simple admin form allowing
manual catalyst entry that overrides auto-parsed data. Track parse failures
per ticker for visibility.'

Two changes:

1. catalyst_universe.is_manual_override BOOLEAN — flag rows that were
   manually added or edited via the admin form. These rows are NEVER
   overwritten by the universe seeder, even if the seeder produces a
   conflicting catalyst for the same (ticker, drug, date) tuple.
   Source for manual rows is set to 'manual' so the FE can render a
   distinct badge.

2. catalyst_ingestion_log table — append-only log of every seeder/parser
   attempt. Lets the admin see WHY parsing failed for a particular ticker
   ('Gemini 503', 'OpenAI returned no catalysts', 'Anthropic web_search
   timeout'). Without this, failures are silent and the user has no way
   to diagnose why a stock has stale/missing data.

Revision ID: 011_manual_override_and_ingestion_log
"""
from alembic import op


revision = '011_manual_override_and_ingestion_log'
down_revision = '010_recanonicalize_dedup'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add is_manual_override flag to catalyst_universe
    op.execute("""
        ALTER TABLE catalyst_universe
        ADD COLUMN IF NOT EXISTS is_manual_override BOOLEAN DEFAULT FALSE
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cu_manual
        ON catalyst_universe(is_manual_override)
        WHERE is_manual_override = TRUE
    """)

    # 2. catalyst_ingestion_log — track every seeder/parser attempt
    op.execute("""
        CREATE TABLE IF NOT EXISTS catalyst_ingestion_log (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            attempt_at TIMESTAMPTZ DEFAULT NOW(),
            source TEXT NOT NULL,           -- 'gemini' | 'openai' | 'anthropic' | 'clinicaltrials' | 'manual'
            status TEXT NOT NULL,           -- 'success' | 'no_data' | 'error' | 'rate_limited'
            catalysts_found INTEGER DEFAULT 0,
            error_class TEXT,               -- exception class name
            error_message TEXT,             -- truncated message (200 chars max)
            duration_ms INTEGER
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cil_ticker_attempt
        ON catalyst_ingestion_log(ticker, attempt_at DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cil_status
        ON catalyst_ingestion_log(status, attempt_at DESC)
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS catalyst_ingestion_log")
    op.execute("ALTER TABLE catalyst_universe DROP COLUMN IF EXISTS is_manual_override")
