"""009 — fix catalyst_universe duplicate rows from drug_name formatting variants

Problem:
  catalyst_universe had UNIQUE(ticker, catalyst_type, catalyst_date, drug_name)
  but drug_name is the raw LLM output, which varies in formatting:
    'lonvoguran ziclumeran'
    'lonvoguran ziclumeran (lonvo-z)'
    'lonvoguran ziclumeran (lonvo-z, NTLA-2002)'
  Each variant got a separate row with its own ID, even though they're the
  same drug. NTLA showed 3 duplicate Phase 3 Readouts that polluted the
  catalyst list, materiality ranking, and NPV anchor selection.

Fix:
  1. Deduplicate existing rows in-place: for each (ticker, catalyst_type,
     catalyst_date, canonical_drug_name) group, keep the row with the
     latest last_updated and mark the rest as status='superseded' with
     superseded_by pointing at the survivor.
  2. Drop old UNIQUE constraint on drug_name.
  3. Add new UNIQUE constraint on canonical_drug_name (which IS already
     normalized by _canonicalize_drug — strips parentheticals, lowercases,
     etc.).
  4. Future inserts use canonical_drug_name in the conflict key, so
     formatting variations from different LLM calls upsert into the same row.

Revision ID: 009
Revises: 008_abnormal_returns
"""

from alembic import op

revision = '009_dedup_catalyst_canonical'
down_revision = '008_abnormal_returns'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Step 1: Backfill canonical_drug_name where missing (defensive — should
    # already be populated for new rows, but old rows pre-canonical-column
    # may have NULL). Use lowercased + stripped raw drug_name as fallback.
    op.execute("""
        UPDATE catalyst_universe
        SET canonical_drug_name = LOWER(TRIM(REGEXP_REPLACE(drug_name, '\\s*\\(.*?\\)\\s*', '', 'g')))
        WHERE canonical_drug_name IS NULL AND drug_name IS NOT NULL AND drug_name != ''
    """)

    # Step 2: Identify duplicate groups by (ticker, catalyst_type,
    # catalyst_date, canonical_drug_name) and supersede all but the most-
    # recently-updated row. NULL canonical_drug_name groups are NOT deduped
    # because we can't tell if they're the same drug.
    op.execute("""
        WITH ranked AS (
            SELECT
                id,
                ticker,
                catalyst_type,
                catalyst_date,
                canonical_drug_name,
                last_updated,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker, catalyst_type, catalyst_date, canonical_drug_name
                    ORDER BY last_updated DESC NULLS LAST, id DESC
                ) AS rn,
                FIRST_VALUE(id) OVER (
                    PARTITION BY ticker, catalyst_type, catalyst_date, canonical_drug_name
                    ORDER BY last_updated DESC NULLS LAST, id DESC
                ) AS survivor_id
            FROM catalyst_universe
            WHERE status = 'active'
              AND canonical_drug_name IS NOT NULL
              AND canonical_drug_name != ''
        )
        UPDATE catalyst_universe cu
        SET status = 'superseded',
            superseded_by = ranked.survivor_id,
            superseded_at = NOW()
        FROM ranked
        WHERE cu.id = ranked.id
          AND ranked.rn > 1
    """)

    # Step 3: Drop the old UNIQUE constraint. Postgres autogenerates the name
    # as <table>_<col1>_<col2>...key. Locate and drop it.
    op.execute("""
        DO $$
        DECLARE
            cn TEXT;
        BEGIN
            SELECT conname INTO cn
            FROM pg_constraint
            WHERE conrelid = 'catalyst_universe'::regclass
              AND contype = 'u'
              AND pg_get_constraintdef(oid) LIKE '%drug_name)%'
              AND pg_get_constraintdef(oid) NOT LIKE '%canonical_drug_name)%';
            IF cn IS NOT NULL THEN
                EXECUTE 'ALTER TABLE catalyst_universe DROP CONSTRAINT ' || cn;
            END IF;
        END $$;
    """)

    # Step 4: Add the new constraint on canonical_drug_name. Use a partial
    # unique INDEX (not constraint) so NULL canonical_drug_name rows are
    # allowed multiple times — those are catalysts without an identified
    # drug (e.g. company-level events). Index instead of constraint because
    # constraints can't have WHERE clauses but indexes can.
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_catalyst_universe_canonical
        ON catalyst_universe (ticker, catalyst_type, catalyst_date, canonical_drug_name)
        WHERE canonical_drug_name IS NOT NULL AND status = 'active'
    """)


def downgrade() -> None:
    # Drop the new index, restore the old constraint. We don't restore the
    # superseded rows — they're still in the table with status='superseded',
    # which is the original schema's mechanism anyway.
    op.execute("DROP INDEX IF EXISTS uq_catalyst_universe_canonical")
    op.execute("""
        ALTER TABLE catalyst_universe
        ADD CONSTRAINT catalyst_universe_ticker_catalyst_type_catalyst_date_drug_n_key
        UNIQUE (ticker, catalyst_type, catalyst_date, drug_name)
    """)
