"""llm_usage + llm_budgets — token usage tracking with budget enforcement

Revision ID: 003_llm_usage
Revises: 002_post_catalyst
Create Date: 2026-04-25

Adds two tables to enable per-call accounting + budget enforcement:
  - llm_usage: every LLM call recorded with tokens, cost, duration, status
  - llm_budgets: per-scope (global/provider/feature) daily/monthly limits
                 with optional hard cutoff
"""
from alembic import op
import sqlalchemy as sa

revision = "003_llm_usage"
down_revision = "002_post_catalyst"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. llm_usage — one row per LLM call
    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_usage (
            id BIGSERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT NOW() NOT NULL,
            day DATE GENERATED ALWAYS AS ((ts AT TIME ZONE 'UTC')::date) STORED,
            provider TEXT NOT NULL,         -- 'anthropic' | 'openai' | 'google'
            model TEXT,                     -- 'claude-sonnet-4-5' | 'gpt-4o' | 'gemini-2.5-flash' etc
            feature TEXT,                   -- 'npv_v2' | 'consensus' | 'risk_factors' | 'universe_seeder' | 'news_impact' | 'ai_pipeline'
            ticker TEXT,                    -- if call was ticker-scoped
            tokens_input INTEGER,
            tokens_output INTEGER,
            cost_usd NUMERIC(10, 6),        -- 6 decimal precision (sub-cent)
            duration_ms INTEGER,
            status TEXT,                    -- 'success' | 'error' | 'timeout' | 'budget_blocked'
            error_message TEXT,
            request_id TEXT                 -- correlation id (job_id, run_id, etc)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_day ON llm_usage(day DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_provider ON llm_usage(provider, day DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_feature ON llm_usage(feature, day DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_ticker ON llm_usage(ticker, day DESC) WHERE ticker IS NOT NULL")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage(ts DESC)")

    # 2. llm_budgets — per-scope spending limits
    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_budgets (
            id SERIAL PRIMARY KEY,
            scope_type TEXT NOT NULL,           -- 'global' | 'provider' | 'feature' | 'provider_feature'
            scope_value TEXT NOT NULL,          -- 'global' | 'anthropic' | 'npv_v2' | 'anthropic:npv_v2'
            daily_limit_usd NUMERIC(10, 2),
            monthly_limit_usd NUMERIC(10, 2),
            hard_cutoff BOOLEAN DEFAULT FALSE,  -- if true, calls REJECTED when over budget
            alert_at_pct NUMERIC(5, 2) DEFAULT 80.0,  -- alert threshold (e.g. 80%)
            enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            notes TEXT,
            UNIQUE(scope_type, scope_value)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_budgets_scope ON llm_budgets(scope_type, scope_value)")

    # Seed sensible defaults
    # Global daily $5 budget (matches universe_seeder DAILY_LLM_BUDGET_USD), no hard cutoff
    op.execute("""
        INSERT INTO llm_budgets (scope_type, scope_value, daily_limit_usd, monthly_limit_usd,
                                  hard_cutoff, alert_at_pct, notes)
        VALUES ('global', 'global', 25.00, 500.00, FALSE, 80.0,
                'Default global daily/monthly budget. Edit via /admin/llm/budgets.')
        ON CONFLICT (scope_type, scope_value) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS llm_budgets CASCADE")
    op.execute("DROP TABLE IF EXISTS llm_usage CASCADE")
