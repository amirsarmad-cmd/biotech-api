"""007 — options_implied_meta on post_catalyst_outcomes

When backfilling implied_move from Polygon historical chains, we want to
record the provenance: which expiration was used, what the ATM strike was,
what date the chain was sampled (T-1 from catalyst). This column carries
that metadata so we can debug bad backfill rows without re-fetching.

Schema rationale: JSONB so we don't have to re-migrate every time we add a
new field (e.g. method comparison: vol surface fit vs straddle-pct).
"""
from alembic import op


revision = "007_options_implied_meta"
down_revision = "006_research_corpus"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        ADD COLUMN IF NOT EXISTS options_implied_meta JSONB
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE post_catalyst_outcomes
        DROP COLUMN IF EXISTS options_implied_meta
    """)
