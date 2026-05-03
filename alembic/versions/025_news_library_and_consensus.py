"""025 — Shared news library + per-LLM consensus label columns.

Multi-LLM consensus architecture (project_news_consensus_scope.md, revised
plan in C:\\Users\\itsup\\.claude\\plans\\prancy-gathering-rose.md):

  Pass 1  — Gemini 2.5 Flash (Pro fallback) grounded scout. Returns a
            label PLUS the URLs it consulted via response.candidates[0]
            .grounding_metadata.grounding_chunks. Discovered URLs land in
            catalyst_event_news so all downstream agents see the same
            enriched corpus.

  Pass 2  — Three readers in parallel, all reading the same library:
              Claude Sonnet 4.6
              GPT-5.5 (gpt-4o fallback)
              Gemini 3 (gemini-2.5-pro fallback)
            Each returns the same JSON-schema label.

  Pass 3  — Claude Opus 4.7 arbiter. Reads the library + Pass-1 verdict +
            all 3 Pass-2 verdicts, synthesizes the final consensus class
            + confidence + reasoning. Opus's verdict IS the consensus —
            not a flat majority vote.

This migration ships the storage layer:
  catalyst_event_news       — article store. Sources: finviz |
                              seeking_alpha_xml | stat_news_rss |
                              fierce_biotech | biopharma_dive | benzinga |
                              stat_plus | gemini_grounded.
  post_catalyst_outcomes    — gains JSONB columns for the 4 per-LLM
                              labels (sonnet, gpt55, gemini3, opus) and
                              the consensus result. Existing Gemini-only
                              outcome_label_class / outcome_labeled_at
                              are kept untouched so the in-flight
                              single-LLM labeler still works.

Revision ID: 025_news_library_and_consensus
"""
from alembic import op


revision = "025_news_library_and_consensus"
down_revision = "024_finviz_dynamic_columns"
branch_labels = None
depends_on = None


def upgrade():
    # ── Shared news library ───────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS catalyst_event_news (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            catalyst_id INTEGER REFERENCES catalyst_universe(id) ON DELETE CASCADE,
            catalyst_date TEXT,                    -- mirrored for fast filtering w/o join
            source TEXT NOT NULL,
              -- finviz | seeking_alpha_xml | stat_news_rss | fierce_biotech
              -- | biopharma_dive | benzinga | stat_plus | gemini_grounded
            url TEXT NOT NULL,
            url_hash TEXT NOT NULL,                -- sha256(normalize_url(url))
            headline TEXT NOT NULL,
            summary TEXT,
            body TEXT,                             -- optional full text (STAT+ / authenticated paths)
            published_at TIMESTAMPTZ,
            collected_at TIMESTAMPTZ DEFAULT NOW(),

            -- Where the URL came from. Lets us audit retrieval-vs-analytical
            -- divergence: if grounded URLs are systematically the ones that
            -- flip a verdict, we know discovery is doing the work.
            discovery_path TEXT,
              -- forward_collection | finviz_per_event | sa_xml_per_ticker |
              -- gemini_grounded_search

            -- For RSS feeds, how we matched the ticker (audit false positives)
            ticker_mention_method TEXT,
              -- finviz_per_event | sa_xml_ticker | cashtag | parens_listing
              -- | company_name_match | grounded_attribution

            UNIQUE (ticker, url_hash)
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_cen_ticker_date ON catalyst_event_news (ticker, catalyst_date);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cen_catalyst    ON catalyst_event_news (catalyst_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cen_published   ON catalyst_event_news (published_at);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cen_source      ON catalyst_event_news (source);")

    # ── Per-LLM + consensus label columns ─────────────────────────
    op.execute(
        """
        ALTER TABLE post_catalyst_outcomes
          -- Pass 2 readers
          ADD COLUMN IF NOT EXISTS outcome_label_sonnet_json   JSONB,
          ADD COLUMN IF NOT EXISTS outcome_label_sonnet_class  TEXT,
          ADD COLUMN IF NOT EXISTS outcome_label_gpt55_json    JSONB,
          ADD COLUMN IF NOT EXISTS outcome_label_gpt55_class   TEXT,
          ADD COLUMN IF NOT EXISTS outcome_label_gemini3_json  JSONB,
          ADD COLUMN IF NOT EXISTS outcome_label_gemini3_class TEXT,

          -- Pass 3 arbiter (this IS the consensus)
          ADD COLUMN IF NOT EXISTS outcome_label_opus_json     JSONB,
          ADD COLUMN IF NOT EXISTS outcome_label_opus_class    TEXT,

          -- Pass 1 grounding chunks (per-event audit trail of what Gemini
          -- grounded search found)
          ADD COLUMN IF NOT EXISTS outcome_label_pass1_grounding_json JSONB,

          -- Final consensus columns (mirrors of Opus's verdict + derived
          -- confidence). consensus_json is the full vote-stack with
          -- per-LLM details, persisted alongside the projected class.
          ADD COLUMN IF NOT EXISTS outcome_label_consensus_json        JSONB,
          ADD COLUMN IF NOT EXISTS outcome_label_consensus_class       TEXT,
          ADD COLUMN IF NOT EXISTS outcome_label_consensus_confidence  NUMERIC,
          ADD COLUMN IF NOT EXISTS outcome_label_consensus_at          TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS outcome_label_consensus_attempts    INTEGER DEFAULT 0,
          ADD COLUMN IF NOT EXISTS outcome_label_consensus_last_attempt_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS outcome_label_consensus_last_error  TEXT,

          -- Library coverage at the moment consensus ran. Lets us
          -- distinguish "all 4 LLMs disagreed because the library was
          -- thin" from "all 4 disagreed despite a fat library".
          ADD COLUMN IF NOT EXISTS news_library_size_at_consensus      INTEGER
        ;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pco_consensus_class
          ON post_catalyst_outcomes(outcome_label_consensus_class)
          WHERE outcome_label_consensus_class IS NOT NULL;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pco_consensus_pending
          ON post_catalyst_outcomes(catalyst_date)
          WHERE outcome_label_consensus_at IS NULL AND catalyst_date IS NOT NULL;
        """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_pco_consensus_pending")
    op.execute("DROP INDEX IF EXISTS idx_pco_consensus_class")
    op.execute(
        """
        ALTER TABLE post_catalyst_outcomes
          DROP COLUMN IF EXISTS outcome_label_sonnet_json,
          DROP COLUMN IF EXISTS outcome_label_sonnet_class,
          DROP COLUMN IF EXISTS outcome_label_gpt55_json,
          DROP COLUMN IF EXISTS outcome_label_gpt55_class,
          DROP COLUMN IF EXISTS outcome_label_gemini3_json,
          DROP COLUMN IF EXISTS outcome_label_gemini3_class,
          DROP COLUMN IF EXISTS outcome_label_opus_json,
          DROP COLUMN IF EXISTS outcome_label_opus_class,
          DROP COLUMN IF EXISTS outcome_label_pass1_grounding_json,
          DROP COLUMN IF EXISTS outcome_label_consensus_class,
          DROP COLUMN IF EXISTS outcome_label_consensus_confidence,
          DROP COLUMN IF EXISTS outcome_label_consensus_at,
          DROP COLUMN IF EXISTS outcome_label_consensus_attempts,
          DROP COLUMN IF EXISTS outcome_label_consensus_last_attempt_at,
          DROP COLUMN IF EXISTS outcome_label_consensus_last_error,
          DROP COLUMN IF EXISTS news_library_size_at_consensus
        ;
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_cen_source")
    op.execute("DROP INDEX IF EXISTS idx_cen_published")
    op.execute("DROP INDEX IF EXISTS idx_cen_catalyst")
    op.execute("DROP INDEX IF EXISTS idx_cen_ticker_date")
    op.execute("DROP TABLE IF EXISTS catalyst_event_news")
