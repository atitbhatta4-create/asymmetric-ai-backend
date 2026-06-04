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
