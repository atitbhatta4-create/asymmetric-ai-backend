"""
Database layer — supports both PostgreSQL (prod) and SQLite (local dev/fallback).

When DATABASE_URL env var is set → tries psycopg3 + ConnectionPool (PostgreSQL).
If that fails (e.g. Render IP block from local dev) → falls back to SQLite automatically.
When DATABASE_URL is not set → SQLite directly.
"""
import os
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

_pg_connected = False   # True only if PostgreSQL pool opened successfully

if DATABASE_URL:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        import time as _time

        _url = DATABASE_URL
        if _url.startswith("postgres://"):
            _url = _url.replace("postgres://", "postgresql://", 1)

        # TCP keepalives — aggressive so Render's 60-90s idle timeout
        # never kills a pooled connection before the pre-ping catches it.
        _ka = "keepalives=1&keepalives_idle=30&keepalives_interval=5&keepalives_count=3&connect_timeout=10"
        _url = _url + ("&" if "?" in _url else "?") + _ka

        def _pre_ping(conn: "psycopg.Connection") -> None:
            try:
                conn.execute("SELECT 1")
            except Exception as exc:
                print(f"[DB] SSL drop detected in pre-ping ({type(exc).__name__}) — pool will reconnect")
                raise

        _PG_POOL_MAX = int(os.getenv("PG_POOL_MAX", "50"))

        # Try to open pool 3 times; if all fail, fall through to SQLite.
        _pool_error = None
        for _attempt in range(3):
            try:
                _pool = ConnectionPool(
                    conninfo=_url,
                    min_size=2,
                    max_size=_PG_POOL_MAX,
                    kwargs={"row_factory": dict_row},
                    open=False,
                    check=_pre_ping,
                    max_lifetime=1800,
                    reconnect_timeout=10,
                )
                _pool.open(wait=True, timeout=30)
                _pool_error = None
                break
            except Exception as _e:
                _pool_error = _e
                if _attempt < 2:
                    _time.sleep(5)

        if _pool_error is not None:
            raise _pool_error   # falls into the outer except → SQLite fallback

        USING_PG = True
        _pg_connected = True

        @contextmanager
        def db_conn():
            """Borrow a pre-pinged connection from the pool."""
            try:
                with _pool.connection() as conn:
                    yield conn
            except psycopg.OperationalError as exc:
                print(f"[DB] OperationalError after pool hand-out — {exc}")
                try:
                    _pool.check()
                except Exception:
                    pass
                print("[DB] DB reconnected after SSL drop")
                raise

        def db():
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

    except Exception as _pg_err:
        print(f"[DB] PostgreSQL unavailable ({_pg_err.__class__.__name__}) — falling back to SQLite")
        DATABASE_URL = ""   # triggers SQLite block below

if not DATABASE_URL:
    import sqlite3

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
        """Wraps sqlite3.Connection."""

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
        raw = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=60)
        raw.row_factory = _dict_factory
        raw.execute("PRAGMA journal_mode=WAL")   # allow concurrent readers + 1 writer
        raw.execute("PRAGMA busy_timeout=60000")  # 60s retry on locked writes
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
