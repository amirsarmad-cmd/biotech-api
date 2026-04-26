# ── Plotly compatibility patch ─────────────────────────────────────────────
try:
    import plotly.graph_objects as _go
    import plotly.graph_objs._indicator as _ind_mod
    _OrigNumber = None
    try:
        from plotly.graph_objs.indicator import Number as _OrigNumber
    except Exception:
        pass
    if _OrigNumber is not None:
        _orig_init = _OrigNumber.__init__
        def _patched_init(self, arg=None, format=None, **kwargs):
            if format is not None and "valueformat" not in kwargs:
                kwargs["valueformat"] = format
            _orig_init(self, arg=arg, **kwargs)
        _OrigNumber.__init__ = _patched_init
        print("✅ Plotly go.Indicator.Number patched: format->valueformat")
except Exception as _pe:
    print(f"Plotly patch skipped: {_pe}")
# ── End patch ───────────────────────────────────────────────────────────────

import logging
import json, os, sqlite3
from datetime import datetime
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SQLITE_PATH = os.getenv("SQLITE_PATH", "/tmp/biotech_screener.db")

# Detect if we can use Postgres
USE_POSTGRES = False
_pg_module = None

if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras
        _pg_module = psycopg2
        USE_POSTGRES = True
        logger.info(f"Using PostgreSQL via psycopg2 ({DATABASE_URL[:30]}...)")
    except ImportError:
        try:
            import pg8000.native as pg8000
            _pg_module = pg8000
            USE_POSTGRES = True
            logger.info("Using PostgreSQL via pg8000")
        except ImportError:
            logger.warning("DATABASE_URL is set but no Postgres driver available — falling back to SQLite")


def _ph(use_pg):
    """Placeholder — ? for SQLite, %s for Postgres."""
    return "%s" if use_pg else "?"


def _upsert_stock_sql(use_pg):
    """Return (sql, params_from_dict) for upsert on ticker+catalyst_type+catalyst_date."""
    if use_pg:
        return """INSERT INTO screener_stocks
            (ticker,company_name,industry,market_cap,catalyst_type,catalyst_date,
             probability,description,news_count,sentiment_score,overall_score,last_updated)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker, catalyst_type, catalyst_date) DO UPDATE SET
                company_name=EXCLUDED.company_name, industry=EXCLUDED.industry,
                market_cap=EXCLUDED.market_cap, probability=EXCLUDED.probability,
                description=EXCLUDED.description, news_count=EXCLUDED.news_count,
                sentiment_score=EXCLUDED.sentiment_score, overall_score=EXCLUDED.overall_score,
                last_updated=EXCLUDED.last_updated"""
    else:
        return """INSERT OR REPLACE INTO screener_stocks
            (ticker,company_name,industry,market_cap,catalyst_type,catalyst_date,
             probability,description,news_count,sentiment_score,overall_score,last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""


def _insert_shortlist_sql(use_pg):
    if use_pg:
        return """INSERT INTO screener_shortlist
            (ticker,company_name,date_added,initial_price,initial_score,
             initial_sentiment,price_history,score_history,sentiment_history,last_updated)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker) DO NOTHING"""
    return """INSERT OR IGNORE INTO screener_shortlist
            (ticker,company_name,date_added,initial_price,initial_score,
             initial_sentiment,price_history,score_history,sentiment_history,last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?)"""


class BiotechDatabase:
    def __init__(self, db_path: str = None):
        self.use_pg = USE_POSTGRES
        self.db_path = db_path or SQLITE_PATH
        self.init_database()

    def _conn(self):
        if self.use_pg:
            import psycopg2
            # psycopg2 handles postgresql:// URLs directly
            return psycopg2.connect(DATABASE_URL)
        return sqlite3.connect(self.db_path)

    # Aliases for biotech-api (main.py expects these names)
    def get_conn(self):
        """Context-manager-friendly connection wrapper."""
        class _CM:
            def __init__(self, outer): self._c = outer._conn()
            def __enter__(self): return self._c
            def __exit__(self, *a): self._c.close()
        return _CM(self)

    def ensure_schema(self):
        """Alias for init_database — idempotent DDL."""
        # init_database() runs in __init__, so this is essentially a no-op
        # but exposed for explicit calls from FastAPI lifespan
        return True

    def init_database(self):
        conn = self._conn()
        try:
            cur = conn.cursor()
            # Postgres vs SQLite DDL
            if self.use_pg:
                # Detect incompatible legacy schema (from an earlier app using same DB).
                # If screener_stocks exists without our catalyst_type column, DROP and recreate.
                cur.execute("""SELECT column_name FROM information_schema.columns 
                               WHERE table_schema='public' AND table_name='screener_stocks'""")
                existing_cols = {r[0] for r in cur.fetchall()}
                required_cols = {"ticker","catalyst_type","catalyst_date","probability",
                                  "description","market_cap","news_count","sentiment_score",
                                  "overall_score","company_name","last_updated"}
                if existing_cols and not required_cols.issubset(existing_cols):
                    logger.warning(f"screener_stocks has incompatible legacy schema "
                                   f"(has {existing_cols}, need {required_cols}) — DROPPING")
                    cur.execute("DROP TABLE IF EXISTS screener_stocks CASCADE")
                    conn.commit()
                
                cur.execute("""CREATE TABLE IF NOT EXISTS screener_stocks (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL, company_name TEXT, industry TEXT,
                    market_cap REAL DEFAULT 0, catalyst_type TEXT DEFAULT '', catalyst_date TEXT DEFAULT '',
                    probability REAL DEFAULT 0.5, description TEXT,
                    news_count INTEGER DEFAULT 0, sentiment_score REAL DEFAULT 0,
                    overall_score REAL DEFAULT 0, last_updated TEXT,
                    UNIQUE(ticker, catalyst_type, catalyst_date))""")
                # Same legacy check for screener_shortlist
                cur.execute("""SELECT column_name FROM information_schema.columns 
                               WHERE table_schema='public' AND table_name='screener_shortlist'""")
                existing_cols_sl = {r[0] for r in cur.fetchall()}
                required_cols_sl = {"ticker","company_name","date_added","initial_price",
                                      "strategies","npv_overrides"}
                if existing_cols_sl and not required_cols_sl.issubset(existing_cols_sl):
                    logger.warning(f"screener_shortlist has incompatible legacy schema — DROPPING")
                    cur.execute("DROP TABLE IF EXISTS screener_shortlist CASCADE")
                    conn.commit()
                
                cur.execute("""CREATE TABLE IF NOT EXISTS screener_shortlist (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL UNIQUE, company_name TEXT,
                    date_added TEXT NOT NULL,
                    initial_price REAL DEFAULT 0, initial_score REAL DEFAULT 0,
                    initial_sentiment REAL DEFAULT 0,
                    price_history TEXT DEFAULT '[]',
                    score_history TEXT DEFAULT '[]',
                    sentiment_history TEXT DEFAULT '[]',
                    strategies TEXT DEFAULT '[]',
                    npv_overrides TEXT DEFAULT '{}',
                    last_updated TEXT)""")
                # Add columns if missing (idempotent on Postgres)
                for col, default in (("strategies","'[]'"), ("npv_overrides","'{}'")):
                    try:
                        cur.execute(f"ALTER TABLE screener_shortlist ADD COLUMN IF NOT EXISTS {col} TEXT DEFAULT {default}")
                    except Exception:
                        pass
            else:
                cur.execute("""CREATE TABLE IF NOT EXISTS screener_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL, company_name TEXT, industry TEXT,
                    market_cap REAL DEFAULT 0, catalyst_type TEXT, catalyst_date TEXT,
                    probability REAL DEFAULT 0.5, description TEXT,
                    news_count INTEGER DEFAULT 0, sentiment_score REAL DEFAULT 0,
                    overall_score REAL DEFAULT 0, last_updated TEXT,
                    UNIQUE(ticker, catalyst_type, catalyst_date))""")
                cur.execute("""CREATE TABLE IF NOT EXISTS screener_shortlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE, company_name TEXT,
                    date_added TEXT NOT NULL,
                    initial_price REAL DEFAULT 0, initial_score REAL DEFAULT 0,
                    initial_sentiment REAL DEFAULT 0,
                    price_history TEXT DEFAULT '[]',
                    score_history TEXT DEFAULT '[]',
                    sentiment_history TEXT DEFAULT '[]',
                    strategies TEXT DEFAULT '[]',
                    npv_overrides TEXT DEFAULT '{}',
                    last_updated TEXT)""")
                for col, default in (("strategies","'[]'"), ("npv_overrides","'{}'")):
                    try:
                        cur.execute(f"ALTER TABLE screener_shortlist ADD COLUMN {col} TEXT DEFAULT {default}")
                        conn.commit()
                    except Exception:
                        pass
            conn.commit()
            backend = "postgres" if self.use_pg else f"sqlite@{self.db_path}"
            logger.info(f"DB ready ({backend})")
        except Exception as e:
            logger.error(f"DB init error: {e}")
        finally:
            conn.close()

    def add_stock(self, d: Dict) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            sql = _upsert_stock_sql(self.use_pg)
            # Normalize: Postgres rejects None in unique-constraint columns differently than SQLite.
            # Coerce catalyst_type and catalyst_date to empty string if None/missing.
            catalyst_type = d.get("catalyst_type") or ""
            catalyst_date = d.get("catalyst_date") or ""
            cur.execute(sql,
                (d.get("ticker"), d.get("company_name"), d.get("industry"),
                 d.get("market_cap",0), catalyst_type, catalyst_date,
                 d.get("probability",0.5), d.get("description"), d.get("news_count",0),
                 d.get("sentiment_score",0), d.get("overall_score",0),
                 d.get("last_updated", datetime.now().isoformat())))
            conn.commit()
            return True
        except Exception as e:
            # On Postgres, a failed INSERT aborts the whole transaction.
            # Rollback so subsequent inserts on this connection work.
            try: conn.rollback()
            except Exception: pass
            logger.error(f"add_stock({d.get('ticker','?')}/{d.get('catalyst_type','?')}): {type(e).__name__}: {e}")
            return False
        finally:
            conn.close()

    def get_all_stocks(self) -> List[Dict]:
        """Return one row per ticker — the nearest upcoming catalyst from catalyst_universe.
        
        V2: Reads from catalyst_universe (full IBB+XBI ETF coverage). For each ticker,
        picks the closest future catalyst as the 'primary'. Falls back to screener_stocks
        for legacy tickers not yet seeded into V2.
        """
        conn = self._conn()
        try:
            cur = conn.cursor()
            
            # Pick nearest future catalyst per ticker from catalyst_universe
            cur.execute("""
                WITH ranked AS (
                    SELECT 
                        ticker, company_name, catalyst_type, catalyst_date::TEXT AS catalyst_date,
                        description, drug_name, indication, phase, confidence_score,
                        ROW_NUMBER() OVER (
                            PARTITION BY ticker 
                            ORDER BY 
                              CASE WHEN catalyst_date >= CURRENT_DATE THEN 0 ELSE 1 END,
                              catalyst_date ASC,
                              confidence_score DESC NULLS LAST
                        ) AS rn
                    FROM catalyst_universe
                    WHERE status = 'active' 
                      AND catalyst_date IS NOT NULL
                      AND (drug_name IS NOT NULL OR indication IS NOT NULL)
                )
                SELECT ticker, company_name, catalyst_type, catalyst_date,
                       description, drug_name, indication, phase, confidence_score
                FROM ranked WHERE rn = 1
            """)
            v2_cols = [d[0] for d in cur.description]
            v2_rows = [dict(zip(v2_cols, r)) for r in cur.fetchall()]
            
            # Fetch market_cap and other metadata from screener_stocks (legacy table — fallback)
            v2_tickers = {r["ticker"] for r in v2_rows}
            cur.execute("SELECT ticker, market_cap, industry, news_count, sentiment_score FROM screener_stocks")
            legacy = {r[0]: {"market_cap": r[1] or 0, "industry": r[2] or "Biotechnology",
                            "news_count": r[3] or 0, "sentiment_score": r[4] or 0} 
                      for r in cur.fetchall()}
            
            # Build canonical rows
            results = []
            
            # Phase-based default probability
            def _prob_from_catalyst(catalyst_type: str, phase: str | None) -> float:
                t = (catalyst_type or "").lower()
                p = (phase or "").lower()
                if "fda decision" in t or "pdufa" in t: return 0.65
                if "phase 3" in t or "phase iii" in p: return 0.50
                if "phase 2" in t or "phase ii" in p: return 0.45
                if "phase 1" in t or "phase i" in p: return 0.40
                if "adcomm" in t: return 0.55
                if "bla" in t or "nda" in t: return 0.55
                if "partnership" in t: return 0.50
                if "clinical trial" in t: return 0.45
                return 0.40
            
            from datetime import date as _date, datetime as _dt
            today = _date.today()
            
            for v2 in v2_rows:
                ticker = v2["ticker"]
                meta = legacy.get(ticker, {"market_cap": 0, "industry": "Biotechnology",
                                          "news_count": 0, "sentiment_score": 0})
                
                # Compute days until catalyst
                try:
                    cat_date = _dt.strptime(v2["catalyst_date"], "%Y-%m-%d").date()
                    days = (cat_date - today).days
                except Exception:
                    days = 365
                
                prob = _prob_from_catalyst(v2["catalyst_type"], v2.get("phase"))
                
                # Overall score: probability × proximity factor × confidence
                # Proximity: 1.0 for today, 0.5 for 180 days out
                proximity = max(0.1, 1 - (days / 365)) if days >= 0 else 0.3
                conf = float(v2.get("confidence_score") or 0.7)
                overall_score = prob * proximity * conf * 1.4  # multiplier so scores are 0-1 range
                
                results.append({
                    "ticker": ticker,
                    "company_name": v2.get("company_name") or ticker,
                    "industry": meta["industry"],
                    "market_cap": float(meta["market_cap"]),
                    "catalyst_type": v2.get("catalyst_type") or "Unknown",
                    "catalyst_date": v2["catalyst_date"],
                    "probability": prob,
                    "description": v2.get("description") or "",
                    "drug_name": v2.get("drug_name"),
                    "indication": v2.get("indication"),
                    "phase": v2.get("phase"),
                    "news_count": int(meta["news_count"]),
                    "sentiment_score": float(meta["sentiment_score"]),
                    "overall_score": min(1.0, overall_score),
                    "last_updated": today.isoformat(),
                })
            
            # Add legacy screener_stocks tickers that aren't in V2 universe yet.
            # Filter out marker rows (yfinance backfill placeholder, no real catalyst):
            # - rows with empty catalyst_type AND empty catalyst_date
            # - rows whose description starts with 'yfinance backfill' or 'yfinance: no data'
            # These rows exist only to carry market_cap metadata.
            cur.execute("""
                SELECT * FROM screener_stocks
                WHERE ticker NOT IN %s
                  AND COALESCE(catalyst_type, '') != ''
                  AND COALESCE(catalyst_date, '') != ''
                  AND COALESCE(description, '') NOT LIKE 'yfinance backfill%%'
                  AND COALESCE(description, '') NOT LIKE 'yfinance: no data%%'
            """, (tuple(v2_tickers) if v2_tickers else ("__none__",),))
            legacy_cols = [d[0] for d in cur.description]
            for r in cur.fetchall():
                results.append(dict(zip(legacy_cols, r)))
            
            results.sort(key=lambda x: x.get("overall_score") or 0, reverse=True)
            return results
        except Exception as e:
            logger.error(f"get_all_stocks: {e}")
            # Fallback: pure legacy
            try:
                cur = conn.cursor()
                cur.execute("SELECT * FROM screener_stocks ORDER BY overall_score DESC")
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
            except Exception as e2:
                logger.error(f"get_all_stocks fallback: {e2}")
                return []
        finally:
            conn.close()

    def get_stock(self, ticker: str) -> List[Dict]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"SELECT * FROM screener_stocks WHERE ticker={ph} ORDER BY catalyst_date", (ticker,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            conn.close()

    def is_shortlisted(self, ticker: str) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"SELECT id FROM screener_shortlist WHERE ticker={ph}", (ticker,))
            return cur.fetchone() is not None
        except Exception:
            return False
        finally:
            conn.close()

    def add_to_shortlist(self, ticker, company_name="", initial_price=0,
                         initial_score=0, initial_sentiment=0) -> bool:
        conn = self._conn()
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            cur = conn.cursor()
            sql = _insert_shortlist_sql(self.use_pg)
            cur.execute(sql,
                (ticker, company_name, today, initial_price, initial_score,
                 initial_sentiment,
                 json.dumps([{"date":today,"price":initial_price}]),
                 json.dumps([{"date":today,"score":initial_score}]),
                 json.dumps([{"date":today,"sentiment":initial_sentiment}]),
                 datetime.now().isoformat()))
            conn.commit()
            return True
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            logger.error(f"add_to_shortlist: {type(e).__name__}: {e}")
            return False
        finally:
            conn.close()

    def save_strategy(self, ticker: str, strategy: Dict) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"SELECT strategies FROM screener_shortlist WHERE ticker={ph}", (ticker,))
            r = cur.fetchone()
            if r is None: return False
            try: strategies = json.loads(r[0] or "[]")
            except: strategies = []
            strategy["saved_at"] = datetime.now().isoformat()
            strategy["id"] = f"strat_{int(datetime.now().timestamp()*1000)}"
            strategies.append(strategy)
            cur.execute(f"UPDATE screener_shortlist SET strategies={ph}, last_updated={ph} WHERE ticker={ph}",
                (json.dumps(strategies), datetime.now().isoformat(), ticker))
            conn.commit()
            return True
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            logger.error(f"save_strategy: {type(e).__name__}: {e}")
            return False
        finally:
            conn.close()

    def get_strategies(self, ticker: str) -> List[Dict]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"SELECT strategies FROM screener_shortlist WHERE ticker={ph}", (ticker,))
            r = cur.fetchone()
            if r is None: return []
            try: return json.loads(r[0] or "[]")
            except: return []
        except Exception as e:
            logger.error(f"get_strategies: {e}")
            return []
        finally:
            conn.close()

    def delete_strategy(self, ticker: str, strategy_id: str) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"SELECT strategies FROM screener_shortlist WHERE ticker={ph}", (ticker,))
            r = cur.fetchone()
            if r is None: return False
            try: strategies = json.loads(r[0] or "[]")
            except: strategies = []
            strategies = [s for s in strategies if s.get("id") != strategy_id]
            cur.execute(f"UPDATE screener_shortlist SET strategies={ph}, last_updated={ph} WHERE ticker={ph}",
                (json.dumps(strategies), datetime.now().isoformat(), ticker))
            conn.commit()
            return True
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            logger.error(f"delete_strategy: {type(e).__name__}: {e}")
            return False
        finally:
            conn.close()

    def save_npv_overrides(self, ticker: str, overrides: Dict) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"SELECT id FROM screener_shortlist WHERE ticker={ph}", (ticker,))
            if cur.fetchone() is None: return False
            cur.execute(f"UPDATE screener_shortlist SET npv_overrides={ph}, last_updated={ph} WHERE ticker={ph}",
                (json.dumps(overrides), datetime.now().isoformat(), ticker))
            conn.commit()
            return True
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            logger.error(f"save_npv_overrides: {type(e).__name__}: {e}")
            return False
        finally:
            conn.close()

    def remove_from_shortlist(self, ticker: str) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"DELETE FROM screener_shortlist WHERE ticker={ph}", (ticker,))
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            logger.error(f"remove_from_shortlist: {type(e).__name__}: {e}")
            return False
        finally:
            conn.close()

    def get_shortlist(self) -> List[Dict]:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM screener_shortlist ORDER BY date_added DESC")
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                row = dict(zip(cols, r))
                for k in ("price_history","score_history","sentiment_history","strategies"):
                    try: row[k] = json.loads(row.get(k) or "[]")
                    except: row[k] = []
                try: row["npv_overrides"] = json.loads(row.get("npv_overrides") or "{}")
                except: row["npv_overrides"] = {}
                rows.append(row)
            return rows
        except Exception as e:
            logger.error(f"get_shortlist: {e}")
            return []
        finally:
            conn.close()

    def update_shortlist_snapshot(self, ticker, current_price, current_score, current_sentiment) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            ph = _ph(self.use_pg)
            cur.execute(f"SELECT price_history, score_history, sentiment_history FROM screener_shortlist WHERE ticker={ph}", (ticker,))
            r = cur.fetchone()
            if r is None: return False
            today = datetime.now().strftime("%Y-%m-%d")
            try: ph_list = json.loads(r[0] or "[]")
            except: ph_list = []
            try: sh_list = json.loads(r[1] or "[]")
            except: sh_list = []
            try: senth_list = json.loads(r[2] or "[]")
            except: senth_list = []
            ph_list.append({"date":today,"price":current_price})
            sh_list.append({"date":today,"score":current_score})
            senth_list.append({"date":today,"sentiment":current_sentiment})
            # Keep last 30 entries per list
            ph_list = ph_list[-30:]; sh_list = sh_list[-30:]; senth_list = senth_list[-30:]
            cur.execute(f"""UPDATE screener_shortlist SET
                price_history={ph}, score_history={ph}, sentiment_history={ph}, last_updated={ph}
                WHERE ticker={ph}""",
                (json.dumps(ph_list), json.dumps(sh_list), json.dumps(senth_list),
                 datetime.now().isoformat(), ticker))
            conn.commit()
            return True
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            logger.error(f"update_shortlist_snapshot: {type(e).__name__}: {e}")
            return False
        finally:
            conn.close()

    def clear_all_data(self) -> bool:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM screener_stocks")
            cur.execute("DELETE FROM screener_shortlist")
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM screener_stocks")
            nstocks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM screener_shortlist")
            nshort = cur.fetchone()[0]
            return {"stocks": nstocks, "shortlist": nshort, "backend": ("postgres" if self.use_pg else "sqlite")}
        except Exception:
            return {"stocks": 0, "shortlist": 0, "backend": "error"}
        finally:
            conn.close()

    # Back-compat stubs
    def add_to_watchlist(self, ticker, user_id="default"): return True
    def get_watchlist(self, user_id="default"): return []
    def save_daily_snapshot(self, data): return True
    def get_daily_snapshot(self, date): return []


def create_database(db_path=None): return BiotechDatabase(db_path)
