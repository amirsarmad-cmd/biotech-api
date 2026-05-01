"""lgbm_models — store trained LightGBM model artifacts and training metrics

Revision ID: 018_lgbm_model
Revises: 017_backfill_staging
Create Date: 2026-05-01

Adds:
  - lgbm_models — one row per trained model. We keep all versions so we can
    roll back if a new training run regresses. Inference reads the latest
    is_active=true row.
"""
from alembic import op


revision = "018_lgbm_model"
down_revision = "017_backfill_staging"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS lgbm_models (
            id SERIAL PRIMARY KEY,
            model_version TEXT NOT NULL,
            trained_at TIMESTAMPTZ DEFAULT NOW(),
            model_str TEXT NOT NULL,
            feature_names TEXT[] NOT NULL,
            categorical_features TEXT[] DEFAULT ARRAY[]::TEXT[],
            train_n INTEGER,
            test_n INTEGER,
            train_accuracy NUMERIC,
            test_accuracy NUMERIC,
            test_ci_lower_pct NUMERIC,
            test_ci_upper_pct NUMERIC,
            metrics_json JSONB,
            min_outcome_confidence NUMERIC DEFAULT 0.7,
            is_active BOOLEAN DEFAULT TRUE,
            notes TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_lgbm_models_active ON lgbm_models(is_active, trained_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS lgbm_models CASCADE")
