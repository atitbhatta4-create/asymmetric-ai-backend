"""
Database layer — supports both PostgreSQL (prod) and SQLite (local dev).

When DATABASE_URL env var is set → uses psycopg3 + ConnectionPool (PostgreSQL).
When not set → falls back to SQLite with a thin compatibility wrapper
that makes it behave identically (dict rows, %s placeholders).
"""
import os
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if DATABASE_URL:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    _url = DATABASE_URL
    if _url.startswith("postgres://"):
        _url = _url.replace("postgres://", "postgresql://", 1)

    # Add TCP keepalives so idle pool connections survive Render's network
    # timeouts without dropping the SSL session unexpectedly.
    # keepalives_idle=30: send keepalive probe after 30s idle
    # keepalives_interval=10: retry probe every 10s
    # keepalives_count=5: declare dead after 5 missed probes (~80s total)
    _ka = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5&connect_timeout=10"
    _url = _url + ("&" if "?" in _url else "?") + _ka

    USING_PG = True

    # Connection pool — sized for ~100 concurrent active engines.
    # Render free PostgreSQL allows max 25 connections; Standard allows 97.
    # Set max_size to leave headroom for migrations / admin tools.
    _PG_POOL_MAX = int(os.getenv("PG_POOL_MAX", "30"))
    _pool = ConnectionPool(
        conninfo=_url,
        min_size=2,
        max_size=_PG_POOL_MAX,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    _pool.open(wait=True, timeout=30)

    @contextmanager
    def db_conn():
        """Context manager — borrows a connection from the pool and returns it on exit."""
        with _pool.connection() as conn:
            yield conn

    def db():
        """Legacy: returns a raw pool connection. Caller must call .close() to return it."""
        return _pool.getconn()

    def serial_pk() -> str:
        return "SERIAL PRIMARY KEY"

    def column_exists(conn, table: str, col: str) -> bool:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = %s",
            (table, col),
        )
        exists = cur.fetchone() is not None
        cur.close()
        return exists

else:
    import sqlite3
    from contextlib import contextmanager as _cm

    USING_PG = False
    _DB_PATH = os.getenv("ASYM_DB_PATH", "asymmetric_demo.db")

    def _dict_factory(cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    class _Cursor:
        """Wraps sqlite3.Cursor so it accepts %s placeholders like PostgreSQL."""

        def __init__(self, raw_cur):
            self._cur = raw_cur

        def execute(self, sql, params=None):
            sql = sql.replace("%s", "?")
            if params is not None:
                self._cur.execute(sql, params)
            else:
                self._cur.execute(sql)

        def fetchone(self):
            return self._cur.fetchone()

        def fetchall(self):
            return self._cur.fetchall()

        def close(self):
            self._cur.close()

    class _Conn:
        """Wraps sqlite3.Connection — exposes raw connection for PRAGMA access."""

        def __init__(self, raw_conn):
            self._raw = raw_conn

        def cursor(self):
            return _Cursor(self._raw.cursor())

        def commit(self):
            self._raw.commit()

        def rollback(self):
            self._raw.rollback()

        def close(self):
            self._raw.close()

    def _make_conn():
        raw = sqlite3.connect(_DB_PATH, check_same_thread=False)
        raw.row_factory = _dict_factory
        return _Conn(raw)

    @contextmanager
    def db_conn():
        conn = _make_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            raise
        finally:
            conn.close()

    def db():
        return _make_conn()

    def serial_pk() -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def column_exists(conn, table: str, col: str) -> bool:
        raw_cur = conn._raw.cursor()
        raw_cur.execute(f"PRAGMA table_info({table})")
        cols = [r["name"] for r in raw_cur.fetchall()]
        raw_cur.close()
        return col in cols


def init_outcome_log_table() -> None:
    """Create trade_outcomes table if it doesn't exist. Safe to call on every startup."""
    pk = serial_pk()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id             {pk},
                email          TEXT    NOT NULL,
                symbol         TEXT    NOT NULL,
                side           TEXT    NOT NULL,
                mode           TEXT    NOT NULL,
                trade_style    TEXT    NOT NULL,
                grade          TEXT    NOT NULL,
                label          TEXT,
                signal         TEXT,
                score          REAL,
                regime         TEXT,
                outcome        TEXT    NOT NULL,
                entry_price    REAL,
                exit_price     REAL,
                sl_pct         REAL,
                tp_pct         REAL,
                atr_pct        REAL,
                effective_size REAL,
                leverage       INTEGER,
                pnl_pct        REAL,
                pnl_value      REAL,
                equity_before  REAL,
                equity_after   REAL,
                peak_equity    REAL,
                t18_scale1     INTEGER DEFAULT 0,
                t18_scale2     INTEGER DEFAULT 0,
                trailing_used  INTEGER DEFAULT 0,
                vol_adj        REAL,
                dd_adj         REAL,
                candles_open   INTEGER,
                opened_at      TEXT,
                closed_at      TEXT    NOT NULL,
                consecutive_wins_before INTEGER DEFAULT 0
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_email_closed"
            " ON trade_outcomes(email, closed_at)"
        )
        conn.commit()


def log_trade_outcome(
    email: str, symbol: str, side: str, mode: str, trade_style: str,
    grade: str, label: str, signal: str, score: float, regime: str,
    outcome: str, entry_price: float, exit_price: float,
    sl_pct: float, tp_pct: float, atr_pct: float,
    effective_size: float, leverage: int,
    pnl_pct: float, pnl_value: float,
    equity_before: float, equity_after: float, peak_equity: float,
    t18_scale1: bool, t18_scale2: bool, trailing_used: bool,
    vol_adj: float, dd_adj: float,
    candles_open: int, opened_at: str, closed_at: str,
    consecutive_wins_before: int,
) -> None:
    """Insert one row into trade_outcomes. Never raises — failures are silent."""
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO trade_outcomes"
                "(email,symbol,side,mode,trade_style,grade,label,signal,score,regime,"
                "outcome,entry_price,exit_price,sl_pct,tp_pct,atr_pct,"
                "effective_size,leverage,pnl_pct,pnl_value,equity_before,equity_after,"
                "peak_equity,t18_scale1,t18_scale2,trailing_used,vol_adj,dd_adj,"
                "candles_open,opened_at,closed_at,consecutive_wins_before)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                "%s,%s,%s,%s,%s,%s,"
                "%s,%s,%s,%s,%s,%s,"
                "%s,%s,%s,%s,%s,%s,"
                "%s,%s,%s,%s)",
                (
                    email, symbol, side, mode, trade_style, grade, label or "",
                    signal or "-", round(score or 0.0, 4), regime or "-",
                    outcome, round(entry_price or 0.0, 6), round(exit_price or 0.0, 6),
                    round(sl_pct or 0.0, 6), round(tp_pct or 0.0, 6), round(atr_pct or 0.0, 6),
                    round(effective_size or 0.0, 4), int(leverage or 1),
                    round(pnl_pct or 0.0, 6), round(pnl_value or 0.0, 4),
                    round(equity_before or 0.0, 4), round(equity_after or 0.0, 4),
                    round(peak_equity or 0.0, 4),
                    1 if t18_scale1 else 0,
                    1 if t18_scale2 else 0,
                    1 if trailing_used else 0,
                    round(vol_adj or 1.0, 4), round(dd_adj or 1.0, 4),
                    int(candles_open or 0), opened_at or "", closed_at,
                    int(consecutive_wins_before or 0),
                ),
            )
            conn.commit()
    except Exception:
        pass  # outcome log is analytics-only — never block the engine
