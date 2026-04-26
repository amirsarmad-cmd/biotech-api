"""006 — research_corpus table for URL-ingested articles

User-pasted URLs (Seeking Alpha, Substack, IR pages, transcripts) get
fetched, summarized via LLM, embedded, and stored here. Retrieval at
NPV time pulls similar/relevant past articles to enrich the V2 prompt
as Layer 5 (user-supplied research) in the source stack.

Why pgvector: we already have the extension installed (used by AEGRA's
memory system). Adding it for biotech research retrieval is a no-op.

Schema rationale:
- ticker_hint nullable: some articles are about the sector or analyst
  framework, not a specific ticker — those should still be retrievable
- summary stored separately from raw_text so the prompt can use just the
  summary without bloating context
- embedding 1536-dim matches text-embedding-3-small (OpenAI's cheapest)
- url unique to prevent dupes; refetching same URL updates the row
"""
from alembic import op


revision = "006_research_corpus"
down_revision = "005_provenance_pricing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE IF NOT EXISTS research_corpus (
            id SERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            url_domain TEXT,
            ticker_hint TEXT,
            title TEXT,
            author TEXT,
            published_at TIMESTAMPTZ,
            ingested_at TIMESTAMPTZ DEFAULT NOW(),
            raw_text TEXT,
            summary TEXT,
            key_claims JSONB,
            valuation_framework TEXT,
            contrarian_points JSONB,
            tags JSONB,
            embedding vector(1536),
            llm_provider TEXT,
            llm_extraction_cost_usd NUMERIC,
            extraction_status TEXT DEFAULT 'ok',
            error_message TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_research_corpus_ticker ON research_corpus(ticker_hint)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_research_corpus_url_domain ON research_corpus(url_domain)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_research_corpus_ingested ON research_corpus(ingested_at DESC)")
    # ivfflat for cosine similarity — lists=100 fine for <100k rows
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_research_corpus_embedding
        ON research_corpus USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS research_corpus")
