"""022 — Extend catalyst_event_features with free + Finviz sourced columns

Adds nullable columns the spec-04 follow-up sources will fill:
  - OpenFDA FAERS (adverse events)
  - clinicaltrials.gov (trial status, enrollment)
  - Finviz Elite (insider transactions, analyst recommendations)
  - PubMed (publication counts; deferred filler)
  - USPTO patent expiry (deferred filler)
  - NIH RePORTER grants (deferred filler)

Revision ID: 022_event_features_extended
"""
from alembic import op


revision = "022_event_features_extended"
down_revision = "021_catalyst_event_features"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        ALTER TABLE catalyst_event_features
          -- OpenFDA FAERS adverse event reports (rolling counts pre-catalyst)
          ADD COLUMN IF NOT EXISTS adverse_event_count_90d_pre   INTEGER,
          ADD COLUMN IF NOT EXISTS adverse_event_count_365d_pre  INTEGER,
          ADD COLUMN IF NOT EXISTS serious_ae_pct_90d_pre        NUMERIC,
          ADD COLUMN IF NOT EXISTS faers_data_source             TEXT,

          -- clinicaltrials.gov enrollment + status snapshot at catalyst date
          ADD COLUMN IF NOT EXISTS trial_count_active_at_date    INTEGER,
          ADD COLUMN IF NOT EXISTS trial_total_enrollment        INTEGER,
          ADD COLUMN IF NOT EXISTS trial_amendments_180d_pre     INTEGER,
          ADD COLUMN IF NOT EXISTS trial_completion_lag_months   NUMERIC,
          ADD COLUMN IF NOT EXISTS ctgov_data_source             TEXT,

          -- Finviz Elite — analyst recommendations + insider snapshot
          ADD COLUMN IF NOT EXISTS analyst_recommendation_avg    NUMERIC,  -- 1=Strong Buy, 5=Strong Sell
          ADD COLUMN IF NOT EXISTS analyst_target_price_usd      NUMERIC,
          ADD COLUMN IF NOT EXISTS analyst_target_upside_pct     NUMERIC,
          ADD COLUMN IF NOT EXISTS finviz_perf_ytd_pct           NUMERIC,
          ADD COLUMN IF NOT EXISTS finviz_data_source            TEXT,

          -- PubMed publication count per drug (deferred filler)
          ADD COLUMN IF NOT EXISTS pubmed_count_drug_180d        INTEGER,
          ADD COLUMN IF NOT EXISTS pubmed_count_drug_lifetime    INTEGER,

          -- USPTO patent expiry runway (deferred filler)
          ADD COLUMN IF NOT EXISTS years_to_loe_at_date          NUMERIC,
          ADD COLUMN IF NOT EXISTS patent_count_active           INTEGER,

          -- NIH RePORTER grants for the drug / company (deferred filler)
          ADD COLUMN IF NOT EXISTS nih_grants_active_count       INTEGER,
          ADD COLUMN IF NOT EXISTS nih_grants_total_value_m      NUMERIC
        ;
        """
    )


def downgrade():
    op.execute(
        """
        ALTER TABLE catalyst_event_features
          DROP COLUMN IF EXISTS adverse_event_count_90d_pre,
          DROP COLUMN IF EXISTS adverse_event_count_365d_pre,
          DROP COLUMN IF EXISTS serious_ae_pct_90d_pre,
          DROP COLUMN IF EXISTS faers_data_source,
          DROP COLUMN IF EXISTS trial_count_active_at_date,
          DROP COLUMN IF EXISTS trial_total_enrollment,
          DROP COLUMN IF EXISTS trial_amendments_180d_pre,
          DROP COLUMN IF EXISTS trial_completion_lag_months,
          DROP COLUMN IF EXISTS ctgov_data_source,
          DROP COLUMN IF EXISTS analyst_recommendation_avg,
          DROP COLUMN IF EXISTS analyst_target_price_usd,
          DROP COLUMN IF EXISTS analyst_target_upside_pct,
          DROP COLUMN IF EXISTS finviz_perf_ytd_pct,
          DROP COLUMN IF EXISTS finviz_data_source,
          DROP COLUMN IF EXISTS pubmed_count_drug_180d,
          DROP COLUMN IF EXISTS pubmed_count_drug_lifetime,
          DROP COLUMN IF EXISTS years_to_loe_at_date,
          DROP COLUMN IF EXISTS patent_count_active,
          DROP COLUMN IF EXISTS nih_grants_active_count,
          DROP COLUMN IF EXISTS nih_grants_total_value_m
        ;
        """
    )
