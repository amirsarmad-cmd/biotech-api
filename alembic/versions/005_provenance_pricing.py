"""005 — provenance JSONB + US/ex-US net pricing

Adds:
- drug_economics_cache.provenance (JSONB) — per-field source/confidence/citation
- drug_economics_cache.annual_cost_us_net_usd (NUMERIC)
- drug_economics_cache.annual_cost_exus_net_usd (NUMERIC)
- drug_economics_cache.revenue_split_us_pct (NUMERIC)
- drug_economics_cache.confidence_score (NUMERIC) — overall confidence rollup

Why: see methodology audit
- Provenance: every numeric output should carry source + confidence so users
  can distinguish FDA-confirmed values from LLM estimates
- US/ex-US split: applying US gross WAC to global population overstates peak
  sales by ~30-50% — this gives the LLM/code path a way to model net realized
  pricing in each region separately
"""
from alembic import op


revision = "005_provenance_pricing"
down_revision = "004_methodology_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE drug_economics_cache
        ADD COLUMN IF NOT EXISTS provenance JSONB,
        ADD COLUMN IF NOT EXISTS annual_cost_us_net_usd NUMERIC,
        ADD COLUMN IF NOT EXISTS annual_cost_exus_net_usd NUMERIC,
        ADD COLUMN IF NOT EXISTS revenue_split_us_pct NUMERIC,
        ADD COLUMN IF NOT EXISTS confidence_score NUMERIC
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE drug_economics_cache
        DROP COLUMN IF EXISTS provenance,
        DROP COLUMN IF EXISTS annual_cost_us_net_usd,
        DROP COLUMN IF EXISTS annual_cost_exus_net_usd,
        DROP COLUMN IF EXISTS revenue_split_us_pct,
        DROP COLUMN IF EXISTS confidence_score
    """)
