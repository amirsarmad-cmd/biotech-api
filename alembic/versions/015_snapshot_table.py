"""015 — forward prediction snapshots for OOS validation

ChatGPT critique: 'Frozen prediction snapshots. Every prediction must be
stored exactly as it existed before the event. Never evaluate using
updated data. No snapshot = no backtest.'

The 459 historical rows are in-sample — V2 thresholds were tuned against
them. We can't retrofit snapshots to those (the events already happened).
But starting NOW we can snapshot every NEW classification, and after
6-12 months we have honest OOS data.

Schema:
  catalyst_outcome_id    FK to post_catalyst_outcomes (when outcome lands)
                         Nullable until outcome is observed.
  ticker                 redundant, for query performance
  catalyst_id            stable canonical catalyst ID at prediction time
  catalyst_date          predicted catalyst date at snapshot time
  catalyst_type          for per-type analysis without joins

  prediction_time        when classifier ran (immutable, not updated)
  signal_version         'v2' or future versions; for A/B comparison
  feature_version        bumps when feature pipeline changes

  signal                 LONG_UNDERPRICED_POSITIVE / SHORT_SELL_THE_NEWS / etc.
  predicted_prob         model's P(positive outcome)
  priced_in_score        composite at prediction time
  predicted_direction    +1 LONG / -1 SHORT / NULL no-trade

  full_features_json     all inputs to the classifier (runup_30d, runup_vs_xbi,
                         options_implied, IV, market_cap, etc.) — frozen
  model_version          string identifying the active classifier version
                         (e.g. 'v2_thresh_0.60_0.80')

  evaluated_at           when actual_move was scored (NULL until evaluated)
  actual_dir_3d_vs_xbi   actual direction (-1 / +1 / 0=deadband)
  direction_correct      bool (NULL while pending or in deadband)

Key indexes:
  - prediction_time DESC for recent snapshots
  - catalyst_outcome_id for joining outcomes
  - signal_version + signal for OOS aggregation
  - evaluated_at IS NULL for "what's pending" queries

Revision ID: 015_snapshot_table (19 chars)
"""
from alembic import op


revision = '015_snapshot_table'
down_revision = '014_sector_runup'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            id BIGSERIAL PRIMARY KEY,

            -- Catalyst identification
            ticker TEXT NOT NULL,
            catalyst_id INTEGER,
            catalyst_date DATE,
            catalyst_type TEXT,
            catalyst_outcome_id BIGINT,

            -- Snapshot metadata (immutable once written)
            prediction_time TIMESTAMP WITH TIME ZONE
                NOT NULL DEFAULT now(),
            signal_version TEXT NOT NULL DEFAULT 'v2',
            feature_version TEXT NOT NULL DEFAULT 'v1',
            model_version TEXT,

            -- Frozen prediction
            signal TEXT NOT NULL,
            predicted_prob NUMERIC,
            priced_in_score NUMERIC,
            predicted_direction INTEGER,

            -- Frozen features (the full input snapshot)
            full_features_json JSONB,

            -- Evaluation results (filled when outcome is observed)
            evaluated_at TIMESTAMP WITH TIME ZONE,
            actual_abnormal_3d_pct NUMERIC,
            actual_dir_3d_vs_xbi INTEGER,
            direction_correct BOOLEAN,

            -- Audit
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_snap_prediction_time
        ON prediction_snapshots(prediction_time DESC)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_snap_outcome_id
        ON prediction_snapshots(catalyst_outcome_id)
        WHERE catalyst_outcome_id IS NOT NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_snap_signal_version_signal
        ON prediction_snapshots(signal_version, signal)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_snap_pending
        ON prediction_snapshots(catalyst_date)
        WHERE evaluated_at IS NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_snap_ticker_date
        ON prediction_snapshots(ticker, catalyst_date DESC)
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS prediction_snapshots")
