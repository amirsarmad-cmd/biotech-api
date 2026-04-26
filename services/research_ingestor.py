"""research_ingestor — Layer 5 of the source stack: user-supplied research.

Solves the "Seeking Alpha scraper" problem by inverting it. Instead of trying
to scrape SA's catalog (impossible, ToS-violating, paywalled), this lets a
user paste any URL and the system:

  1. Fetches the page (with optional auth cookies passed per-request, never stored)
  2. Extracts main article content via readability heuristics
  3. Sends it to an LLM for structured summarization:
       - Summary
       - Key claims with confidence
       - Valuation framework used
       - Contrarian / pushback points
       - Tags (ticker_hint, themes, indications)
  4. Embeds the summary via OpenAI text-embedding-3-small
  5. Stores in research_corpus with the URL as natural key

Retrieval: when running an NPV analysis on a ticker, we cosine-similarity
search research_corpus for relevant prior articles (same ticker, similar
indication, similar valuation themes) and inject the top-k into the V2
prompt as Layer 5 user research.

Supported sources:
  - Seeking Alpha (free articles + Premium with cookies)
  - Substack (most articles public)
  - IR press releases / earnings transcripts
  - Trade press (STAT, Endpoints, FierceBiotech, BioPharma Dive)
  - Any HTML page with a discernible main-content block

Cost per ingestion: ~$0.001-0.005 (cheap LLM extraction + embedding).
"""
import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import requests

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Fetching + extraction
# ────────────────────────────────────────────────────────────

def _fetch_url(url: str, cookies: Optional[str] = None,
               user_agent: Optional[str] = None,
               timeout: int = 15) -> Tuple[Optional[str], Dict]:
    """Fetch a URL, return (html_or_none, metadata).
    
    metadata = {"final_url": str, "status": int, "content_type": str,
                "fetch_error": str | None}
    """
    headers = {
        "User-Agent": user_agent or
                      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    cookies_dict = None
    if cookies:
        # Parse "key1=val1; key2=val2" format
        cookies_dict = {}
        for part in cookies.split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                cookies_dict[k.strip()] = v.strip()

    try:
        resp = requests.get(url, headers=headers, cookies=cookies_dict,
                            timeout=timeout, allow_redirects=True)
        return resp.text, {
            "final_url": resp.url,
            "status": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "fetch_error": None if resp.ok else f"HTTP {resp.status_code}",
        }
    except requests.exceptions.RequestException as e:
        return None, {
            "final_url": url,
            "status": 0,
            "content_type": "",
            "fetch_error": str(e)[:200],
        }


def _extract_main_content(html: str, url: str) -> Dict:
    """Extract title + author + main article body from HTML.
    
    Tries multiple strategies:
      1. <article> tag content
      2. og:title / article:author meta tags
      3. Common content-class selectors (.post-content, .article-body)
      4. Largest <p>-rich block
    
    Returns:
      {"title": str, "author": str | None, "published_at": str | None,
       "main_text": str, "extraction_method": str}
    """
    if not html:
        return {"title": "", "author": None, "published_at": None,
                "main_text": "", "extraction_method": "empty"}
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _extract_naive(html, url)

    soup = BeautifulSoup(html, "html.parser")

    # ─── Title ─────
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"]
    elif soup.title:
        title = soup.title.string or ""
    title = (title or "").strip()[:300]

    # ─── Author ─────
    author = None
    art_author = soup.find("meta", attrs={"name": "author"}) or \
                 soup.find("meta", property="article:author")
    if art_author and art_author.get("content"):
        author = art_author["content"][:100]
    if not author:
        # Try .author / [rel=author] / .byline
        for sel in [(".author",), ("[rel=author]",), (".byline",), (".post-author",)]:
            try:
                el = soup.select_one(sel[0])
                if el and el.get_text(strip=True):
                    author = el.get_text(strip=True)[:100]
                    break
            except Exception:
                pass

    # ─── Published at ─────
    published_at = None
    pub_meta = soup.find("meta", property="article:published_time") or \
               soup.find("meta", attrs={"name": "publish_date"}) or \
               soup.find("time")
    if pub_meta:
        published_at = (pub_meta.get("content") or pub_meta.get("datetime")
                        or pub_meta.get_text(strip=True) if hasattr(pub_meta, "get_text") else None)
        if published_at:
            published_at = str(published_at)[:50]

    # ─── Main content ─────
    # Strategy 1: <article> tag
    article = soup.find("article")
    if article:
        # Strip noise
        for sel in ["nav", "aside", "footer", ".comments", ".related",
                    ".newsletter-signup", "script", "style"]:
            for el in article.select(sel):
                el.decompose()
        main_text = article.get_text(separator="\n", strip=True)
        if len(main_text) > 500:
            return {
                "title": title, "author": author, "published_at": published_at,
                "main_text": main_text[:50000],
                "extraction_method": "article_tag",
            }

    # Strategy 2: largest <p>-rich block
    candidates = []
    for el in soup.find_all(["main", "div", "section"]):
        ps = el.find_all("p", recursive=False)
        if len(ps) >= 5:
            text = el.get_text(separator="\n", strip=True)
            candidates.append((len(text), el, text))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, _, main_text = candidates[0]
        return {
            "title": title, "author": author, "published_at": published_at,
            "main_text": main_text[:50000],
            "extraction_method": "largest_p_block",
        }

    # Strategy 3: all <p> tags concatenated
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")
                  if len(p.get_text(strip=True)) > 50]
    main_text = "\n\n".join(paragraphs)
    return {
        "title": title, "author": author, "published_at": published_at,
        "main_text": main_text[:50000],
        "extraction_method": "all_p_concat",
    }


def _extract_naive(html: str, url: str) -> Dict:
    """Fallback when BeautifulSoup isn't available — strip tags with regex."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    return {
        "title": title_match.group(1).strip()[:300] if title_match else "",
        "author": None, "published_at": None,
        "main_text": text[:50000],
        "extraction_method": "naive_regex",
    }


# ────────────────────────────────────────────────────────────
# LLM extraction — structured summarization
# ────────────────────────────────────────────────────────────

def _llm_extract_structured(title: str, body: str,
                            url: str, ticker_hint: Optional[str] = None) -> Dict:
    """Extract structured analysis from article content via LLM.
    
    Returns:
      {
        "summary": str,                  # 3-5 sentence high-level
        "key_claims": [{"claim": str, "confidence": "high"|"med"|"low"}],
        "valuation_framework": str,       # what valuation method the author uses
        "contrarian_points": [str],       # pushback / counter-arguments
        "tags": [str],                    # themes (e.g. "FDA risk", "biosimilar erosion")
        "ticker_inferred": str | None,    # ticker the article is about, if extractable
      }
    """
    body_slice = body[:18000]  # leave room for prompt
    prompt = f"""You are an analyst's research assistant. Extract structured insights from
this article. Return ONLY a JSON object — no preamble, no markdown.

ARTICLE URL: {url}
{f'TICKER HINT: {ticker_hint}' if ticker_hint else ''}
TITLE: {title}
---
BODY:
{body_slice}
---

Schema:
{{
  "summary": "<3-5 sentence high-level summary of the article's argument>",
  "key_claims": [
    {{"claim": "<specific factual or analytical claim>", "confidence": "high|medium|low"}},
    ...up to 8 most material claims
  ],
  "valuation_framework": "<what valuation method does the author use? DCF, peak-sales-multiple, comparable EV/Sales, sum-of-parts, M&A premium, etc. Or 'none' if it's not a valuation piece.>",
  "contrarian_points": [
    "<pushback or counter-argument the author raises or that the article overlooks>",
    ...up to 4
  ],
  "tags": ["<short tag e.g. 'FDA risk'>", "<'commercial uptake'>", ...up to 6],
  "ticker_inferred": "<single ticker symbol the article is primarily about, or null if it's a general piece>"
}}

Rules:
- Use null for fields you can't determine; don't fabricate
- 'high' confidence: author cites a specific number/source; 'medium': defensible inference; 'low': speculation
- Return strictly valid JSON
"""
    try:
        from services.llm_helper import call_llm_json
        extracted, err = call_llm_json(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.2,
            feature="research_ingest",
            ticker=ticker_hint,
        )
        if extracted is None:
            raise RuntimeError(err or "all LLM providers failed")
        # call_llm_json already adds _llm_provider; rename for our schema
        extracted["_llm_provider"] = extracted.get("_llm_provider", "?")
        return extracted
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}")
        return {
            "summary": title[:200],
            "key_claims": [],
            "valuation_framework": "unknown",
            "contrarian_points": [],
            "tags": [],
            "ticker_inferred": None,
            "_extraction_error": str(e)[:200],
        }


# ────────────────────────────────────────────────────────────
# Embedding
# ────────────────────────────────────────────────────────────

def _embed_text(text: str) -> Optional[List[float]]:
    """Return a 1536-dim embedding via OpenAI text-embedding-3-small.
    Returns None on failure (caller should handle gracefully).
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": "text-embedding-3-small",
                  "input": text[:8000]},  # ~8k chars ~ 2k tokens, well within limit
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        logger.warning(f"embedding failed: {e}")
        return None


# ────────────────────────────────────────────────────────────
# Top-level ingestion
# ────────────────────────────────────────────────────────────

def ingest_url(url: str, ticker_hint: Optional[str] = None,
               cookies: Optional[str] = None) -> Dict:
    """End-to-end: fetch → extract → LLM summarize → embed → upsert.
    
    Returns:
      {
        "status": "ok" | "fetch_failed" | "extraction_failed" | "embed_failed",
        "url": str,
        "id": int | None,        # research_corpus row id
        "title": str,
        "summary": str,
        "key_claims": [...],
        "ticker_hint": str,
        "ticker_inferred": str | None,
        "fetch_metadata": {...},
        "error": str | None,
      }
    """
    from services.database import BiotechDatabase
    
    # 1. Fetch
    html, fetch_meta = _fetch_url(url, cookies=cookies)
    if not html or fetch_meta.get("fetch_error"):
        return {
            "status": "fetch_failed",
            "url": url,
            "error": fetch_meta.get("fetch_error", "unknown fetch error"),
            "fetch_metadata": fetch_meta,
        }

    # 2. Extract
    extracted = _extract_main_content(html, url)
    if len(extracted.get("main_text", "")) < 200:
        return {
            "status": "extraction_failed",
            "url": url,
            "error": f"content too short ({len(extracted.get('main_text', ''))} chars) — likely paywall or empty page",
            "extraction": extracted,
            "fetch_metadata": fetch_meta,
        }

    # 3. LLM structured extraction
    structured = _llm_extract_structured(
        title=extracted["title"],
        body=extracted["main_text"],
        url=url,
        ticker_hint=ticker_hint,
    )

    # 4. Embed (summary text — keeps embeddings concept-focused)
    embed_input = f"{extracted['title']}\n\n{structured.get('summary', '')}\n\n" + \
                  "\n".join(c.get("claim", "") for c in (structured.get("key_claims") or []))
    embedding = _embed_text(embed_input)
    if not embedding:
        logger.info(f"embed missing for {url}; storing without vector")

    # 5. Upsert into research_corpus
    domain = urlparse(fetch_meta.get("final_url", url)).netloc
    db = BiotechDatabase()
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            embedding_sql = f"'[{','.join(str(x) for x in embedding)}]'::vector" if embedding else "NULL"
            cur.execute(f"""
                INSERT INTO research_corpus
                  (url, url_domain, ticker_hint, title, author, published_at,
                   raw_text, summary, key_claims, valuation_framework,
                   contrarian_points, tags, embedding, llm_provider,
                   extraction_status)
                VALUES (%s, %s, %s, %s, %s, %s,
                        %s, %s, %s::jsonb, %s,
                        %s::jsonb, %s::jsonb, {embedding_sql}, %s,
                        %s)
                ON CONFLICT (url) DO UPDATE SET
                    ticker_hint     = COALESCE(EXCLUDED.ticker_hint, research_corpus.ticker_hint),
                    title           = EXCLUDED.title,
                    author          = COALESCE(EXCLUDED.author, research_corpus.author),
                    raw_text        = EXCLUDED.raw_text,
                    summary         = EXCLUDED.summary,
                    key_claims      = EXCLUDED.key_claims,
                    valuation_framework = EXCLUDED.valuation_framework,
                    contrarian_points = EXCLUDED.contrarian_points,
                    tags            = EXCLUDED.tags,
                    embedding       = EXCLUDED.embedding,
                    llm_provider    = EXCLUDED.llm_provider,
                    ingested_at     = NOW(),
                    extraction_status = EXCLUDED.extraction_status
                RETURNING id
            """, (
                fetch_meta.get("final_url", url), domain,
                ticker_hint or structured.get("ticker_inferred"),
                extracted["title"][:500], extracted.get("author"),
                extracted.get("published_at"),
                extracted["main_text"][:200000],
                structured.get("summary", "")[:5000],
                json.dumps(structured.get("key_claims") or []),
                structured.get("valuation_framework", "")[:500],
                json.dumps(structured.get("contrarian_points") or []),
                json.dumps(structured.get("tags") or []),
                structured.get("_llm_provider", "?"),
                "ok",
            ))
            row_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        logger.exception("research_corpus upsert failed")
        return {
            "status": "db_failed",
            "url": url,
            "error": str(e)[:300],
            "structured": structured,
        }

    return {
        "status": "ok",
        "id": row_id,
        "url": fetch_meta.get("final_url", url),
        "url_domain": domain,
        "title": extracted["title"],
        "author": extracted.get("author"),
        "summary": structured.get("summary", ""),
        "key_claims": structured.get("key_claims", []),
        "valuation_framework": structured.get("valuation_framework"),
        "contrarian_points": structured.get("contrarian_points", []),
        "tags": structured.get("tags", []),
        "ticker_hint": ticker_hint,
        "ticker_inferred": structured.get("ticker_inferred"),
        "extraction_method": extracted.get("extraction_method"),
        "main_text_chars": len(extracted["main_text"]),
        "has_embedding": embedding is not None,
        "fetch_metadata": fetch_meta,
        "llm_provider": structured.get("_llm_provider"),
    }


# ────────────────────────────────────────────────────────────
# Retrieval — pgvector cosine similarity
# ────────────────────────────────────────────────────────────

def find_relevant_research(ticker: str, indication: Optional[str] = None,
                            query_text: Optional[str] = None,
                            limit: int = 5) -> List[Dict]:
    """Find relevant research_corpus entries for a ticker / indication.
    
    Strategy:
      1. Direct ticker_hint matches first (highest signal)
      2. Cosine-similarity match on query_text or "ticker + indication"
      3. Combined: rank by ticker match BOOLEAN, then by cosine similarity
    
    Returns: list of {id, url, title, summary, ticker_hint, similarity}
    """
    from services.database import BiotechDatabase
    db = BiotechDatabase()

    # Build query embedding from ticker + indication + query_text
    query_for_embed = " ".join(filter(None, [
        f"Ticker: {ticker}" if ticker else None,
        f"Indication: {indication}" if indication else None,
        query_text,
    ]))
    query_emb = _embed_text(query_for_embed) if query_for_embed else None

    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            # If we have an embedding, use cosine similarity. Otherwise just
            # filter on ticker_hint.
            if query_emb:
                emb_sql = f"'[{','.join(str(x) for x in query_emb)}]'::vector"
                cur.execute(f"""
                    SELECT id, url, title, summary, ticker_hint, key_claims,
                           valuation_framework, contrarian_points, ingested_at,
                           CASE WHEN ticker_hint = %s THEN 1 ELSE 0 END AS ticker_match,
                           1 - (embedding <=> {emb_sql}) AS similarity
                    FROM research_corpus
                    WHERE embedding IS NOT NULL
                      AND extraction_status = 'ok'
                    ORDER BY ticker_match DESC, similarity DESC
                    LIMIT %s
                """, (ticker, limit * 2))
            else:
                cur.execute("""
                    SELECT id, url, title, summary, ticker_hint, key_claims,
                           valuation_framework, contrarian_points, ingested_at,
                           CASE WHEN ticker_hint = %s THEN 1 ELSE 0 END AS ticker_match,
                           NULL::float AS similarity
                    FROM research_corpus
                    WHERE ticker_hint = %s AND extraction_status = 'ok'
                    ORDER BY ingested_at DESC
                    LIMIT %s
                """, (ticker, ticker, limit))

            rows = cur.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "url": r[1],
                    "title": r[2],
                    "summary": r[3],
                    "ticker_hint": r[4],
                    "key_claims": r[5] if isinstance(r[5], (list, dict)) else [],
                    "valuation_framework": r[6],
                    "contrarian_points": r[7] if isinstance(r[7], (list, dict)) else [],
                    "ingested_at": str(r[8]) if r[8] else None,
                    "ticker_match": bool(r[9]),
                    "similarity": float(r[10]) if r[10] is not None else None,
                })
            return results[:limit]
    except Exception as e:
        logger.warning(f"find_relevant_research failed: {e}")
        return []


def list_corpus(ticker: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """List ingested articles (admin / debugging view)."""
    from services.database import BiotechDatabase
    db = BiotechDatabase()
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            if ticker:
                cur.execute("""
                    SELECT id, url, url_domain, ticker_hint, title, author,
                           ingested_at, extraction_status,
                           jsonb_array_length(COALESCE(key_claims, '[]'::jsonb)) AS n_claims
                    FROM research_corpus
                    WHERE ticker_hint = %s
                    ORDER BY ingested_at DESC
                    LIMIT %s
                """, (ticker, limit))
            else:
                cur.execute("""
                    SELECT id, url, url_domain, ticker_hint, title, author,
                           ingested_at, extraction_status,
                           jsonb_array_length(COALESCE(key_claims, '[]'::jsonb)) AS n_claims
                    FROM research_corpus
                    ORDER BY ingested_at DESC
                    LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
            return [{
                "id": r[0], "url": r[1], "url_domain": r[2],
                "ticker_hint": r[3], "title": r[4], "author": r[5],
                "ingested_at": str(r[6]) if r[6] else None,
                "extraction_status": r[7], "n_claims": r[8],
            } for r in rows]
    except Exception as e:
        logger.warning(f"list_corpus failed: {e}")
        return []
