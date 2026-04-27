"""010 — re-canonicalize drug names with stronger normalization, then dedup

Migration 009 added a partial unique index on canonical_drug_name and
attempted to dedup. The dedup didn't merge anything because the
_canonicalize_drug function in services/universe_seeder.py was only doing
lowercase + whitespace collapse. So:
  'lonvoguran ziclumeran'             → 'lonvoguran ziclumeran'
  'lonvoguran ziclumeran (lonvo-z)'    → 'lonvoguran ziclumeran (lonvo-z)'
  'lonvoguran ziclumeran (lonvo-z, NTLA-2002)' → same (with NTLA lowered)
All three got distinct canonical strings → no dedup grouping.

This migration:
  1. Re-canonicalizes ALL active rows using the new _canonicalize_drug
     logic (strip parentheticals + brackets + trademark symbols +
     leading/trailing punctuation, then lowercase + whitespace collapse).
  2. Deduplicates: for each (ticker, catalyst_type, catalyst_date,
     new_canonical) group, keeps the most-recently-updated row and marks
     the rest as 'superseded' with superseded_by → survivor.

The order is important: we must dedup BEFORE updating canonical to the
new value, because updating to a colliding value would violate the
partial unique index from migration 009. So we compute new_canonical
inline inside the dedup CTE, mark losers as superseded (which removes
them from the partial index), then update canonical on the survivors.

Revision ID: 010
Revises: 009_dedup_catalyst_canonical
"""

from alembic import op

revision = '010_recanonicalize_dedup'
down_revision = '009_dedup_catalyst_canonical'
branch_labels = None
depends_on = None


# Postgres expression that mirrors the new _canonicalize_drug Python function.
# Order: lowercase → strip parens → strip brackets → strip TM/® symbols →
# trim leading/trailing punct/whitespace → collapse whitespace.
NEW_CANONICAL_SQL = r"""
TRIM(
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER(drug_name),
                    '\s*\([^)]*\)\s*', ' ', 'g'
                ),
                '\s*\[[^\]]*\]\s*', ' ', 'g'
            ),
            '[™®©℠]', '', 'g'
        ),
        '\s+', ' ', 'g'
    ),
    ' .,;:-_'
)
"""


def upgrade() -> None:
    # Step 1: dedup using the NEW canonical computed inline. Mark losers as
    # 'superseded'. We dedup BEFORE updating canonical_drug_name because
    # the partial unique index on canonical_drug_name from migration 009
    # would otherwise reject the UPDATE to colliding values.
    op.execute(f"""
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, catalyst_type, catalyst_date,
                                 NULLIF({NEW_CANONICAL_SQL}, '')
                    ORDER BY last_updated DESC NULLS LAST, id DESC
                ) AS rn,
                FIRST_VALUE(id) OVER (
                    PARTITION BY ticker, catalyst_type, catalyst_date,
                                 NULLIF({NEW_CANONICAL_SQL}, '')
                    ORDER BY last_updated DESC NULLS LAST, id DESC
                ) AS survivor_id
            FROM catalyst_universe
            WHERE status = 'active'
              AND drug_name IS NOT NULL
              AND drug_name != ''
        )
        UPDATE catalyst_universe cu
        SET status = 'superseded',
            superseded_by = ranked.survivor_id,
            superseded_at = NOW()
        FROM ranked
        WHERE cu.id = ranked.id
          AND ranked.rn > 1
    """)

    # Step 2: update canonical_drug_name on all surviving active rows so it
    # matches the new normalization. Now safe because losers are superseded
    # (excluded from the partial unique index).
    op.execute(f"""
        UPDATE catalyst_universe
        SET canonical_drug_name = NULLIF({NEW_CANONICAL_SQL}, '')
        WHERE status = 'active'
          AND drug_name IS NOT NULL
          AND drug_name != ''
    """)


def downgrade() -> None:
    # No-op: superseded rows can be reactivated manually if needed by
    # setting status='active' and superseded_by=NULL, but the partial
    # unique index would then reject the duplicates again.
    pass
