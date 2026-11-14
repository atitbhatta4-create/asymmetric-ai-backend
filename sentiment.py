"""
sentiment.py
────────────
Fear & Greed index data layer for Step 7 sentiment filter.

Provides:
  - DB table creation for historical F&G data
  - Historical data fetch from alternative.me (free, no key required)
  - Score lookup by timestamp for backtester
  - Cached live fetch for the live engine

Pre-registered thresholds (do not change after backtest starts):
  FNG_LONG_BLOCK  = 80  — block LONG when extreme greed (entering at tops)
  FNG_SHORT_BLOCK = 20  — block SHORT when extreme fear (shorting bottoms)
"""

import json, time
from datetime import datetime, timezone
from typing import Dict, Optional

FNG_LONG_BLOCK  = 80
FNG_SHORT_BLOCK = 20
FNG_API_URL     = "https://api.alternative.me/fng/?limit=0&format=json"

_live_cache: Dict = {}   # {"score": int, "expires": float}


def create_sentiment_table() -> None:
    from database import db_conn
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_fng (
                date_str TEXT PRIMARY KEY,
                fng_score INTEGER NOT NULL,
                fng_label TEXT
            )
        """)
        conn.commit()


def fetch_and_store_fng_history(verbose: bool = True) -> int:
    """
    Download all historical F&G data from alternative.me and store in DB.
    Returns count of rows inserted/updated.
    Safe to re-run — uses upsert.
    """
    import urllib.request
    from database import db_conn, USING_PG

    create_sentiment_table()

    if verbose:
        print("  Fetching F&G history from alternative.me ...", flush=True)

    try:
        with urllib.request.urlopen(FNG_API_URL, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ✗ F&G fetch failed: {e}")
        return 0

    rows = payload.get("data", [])
    if not rows:
        print("  ✗ No data returned from F&G API")
        return 0

    if verbose:
        print(f"  Got {len(rows)} F&G records. Storing ...", flush=True)

    upsert_sql = (
        "INSERT INTO sentiment_fng (date_str, fng_score, fng_label) "
        "VALUES (%s, %s, %s) "
        + ("ON CONFLICT (date_str) DO UPDATE SET fng_score=EXCLUDED.fng_score, fng_label=EXCLUDED.fng_label"
           if USING_PG else
           "ON CONFLICT(date_str) DO UPDATE SET fng_score=excluded.fng_score, fng_label=excluded.fng_label")
    )

    count = 0
    with db_conn() as conn:
        cur = conn.cursor()
        for rec in rows:
            ts_sec  = int(rec.get("timestamp", 0))
            score   = int(rec.get("value", 50))
            label   = rec.get("value_classification", "")
            date_str = datetime.utcfromtimestamp(ts_sec).strftime("%Y-%m-%d")
            cur.execute(upsert_sql, (date_str, score, label))
            count += 1
        conn.commit()

    if verbose:
        print(f"  ✓ Stored {count} F&G records (2018-present)")
    return count


def load_fng_history() -> Dict[str, int]:
    """
    Load all F&G history from DB into memory as {date_str: score}.
    Returned dict is used by backtester to avoid per-candle DB calls.
    Call once per backtest run.
    """
    from database import db_conn
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT date_str, fng_score FROM sentiment_fng ORDER BY date_str")
        rows = cur.fetchall()
    if not rows:
        return {}
    # Handle both dict-style (psycopg2 RealDictRow) and tuple-style results
    result = {}
    for row in rows:
        if isinstance(row, (list, tuple)):
            result[row[0]] = int(row[1])
        else:
            result[row["date_str"]] = int(row["fng_score"])
    return result


def fng_at_ts(fng_history: Dict[str, int], ts_ms: int) -> Optional[int]:
    """
    Return the F&G score for the day of ts_ms (UTC).
    Uses the previous day if today's score is missing (data may lag by 1 day).
    """
    if not fng_history:
        return None
    dt  = datetime.utcfromtimestamp(ts_ms / 1000)
    for delta in range(3):   # try today, yesterday, day before
        from datetime import timedelta
        date_str = (dt - timedelta(days=delta)).strftime("%Y-%m-%d")
        if date_str in fng_history:
            return fng_history[date_str]
    return None


def get_live_fng(ttl_sec: int = 3600) -> Optional[int]:
    """
    Fetch current F&G score with 1-hour cache.
    Returns None on any error (never raises — engine must not crash on API failure).
    """
    import urllib.request

    now = time.time()
    if _live_cache.get("expires", 0) > now:
        return _live_cache.get("score")

    try:
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        score = int(data["data"][0]["value"])
        _live_cache["score"]   = score
        _live_cache["expires"] = now + ttl_sec
        return score
    except Exception:
        return None


def apply_sentiment_gate(
    side: str,
    fng_score: Optional[int],
) -> Optional[str]:
    """
    Returns a block reason string if sentiment blocks this side, else None.
    Pre-registered thresholds: LONG blocked at F&G ≥ 80, SHORT blocked at F&G ≤ 20.
    """
    if fng_score is None:
        return None
    if side == "LONG" and fng_score >= FNG_LONG_BLOCK:
        return (f"SENTIMENT: F&G {fng_score} (Extreme Greed ≥{FNG_LONG_BLOCK}) "
                f"— LONG blocked (entering at market top)")
    if side == "SHORT" and fng_score <= FNG_SHORT_BLOCK:
        return (f"SENTIMENT: F&G {fng_score} (Extreme Fear ≤{FNG_SHORT_BLOCK}) "
                f"— SHORT blocked (shorting capitulation bottom)")
    return None
