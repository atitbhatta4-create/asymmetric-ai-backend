"""
Database layer — supports both PostgreSQL (prod) and SQLite (local dev).

When DATABASE_URL env var is set → uses psycopg3 (PostgreSQL).
When not set → falls back to SQLite with a thin compatibility wrapper
that makes it behave identically (dict rows, %s placeholders).

PostgreSQL mode uses a persistent connection (autocommit=True) instead of
opening a new TCP connection on every db() call. A new connection costs
50-200ms over the network; the persistent proxy costs ~0ms. Reconnects
automatically if the connection drops.
"""
import os

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if DATABASE_URL:
    import psycopg
    from psycopg.rows import dict_row

    _url = DATABASE_URL
    if _url.startswith("postgres://"):
        _url = _url.replace("postgres://", "postgresql://", 1)

    USING_PG = True

    def _new_pg_conn():
        return psycopg.connect(_url, row_factory=dict_row, autocommit=True)

    _pg_conn = None   # module-level persistent connection

    class _PgProxy:
        """
        Persistent-connection proxy. Returned by db() instead of a fresh
        psycopg.connect() each time.

        close()  → no-op  (keeps TCP connection alive for next call)
        commit() → no-op  (autocommit=True — each statement commits instantly)
        cursor() → reconnects transparently if the connection was dropped
        """

        def cursor(self):
            global _pg_conn
            try:
                if _pg_conn is None or _pg_conn.closed:
                    _pg_conn = _new_pg_conn()
                return _pg_conn.cursor()
            except Exception:
                try:
                    _pg_conn = _new_pg_conn()
                    return _pg_conn.cursor()
                except Exception as e:
                    raise RuntimeError(f"DB reconnect failed: {e}") from e

        def commit(self):
            pass   # autocommit=True — every statement is already committed

        def close(self):
            pass   # no-op — keep the TCP connection alive

    _proxy = _PgProxy()

    def db():
        return _proxy

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

    def db():
        raw = sqlite3.connect(_DB_PATH, check_same_thread=False)
        raw.row_factory = _dict_factory
        return _Conn(raw)

    def serial_pk() -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    def column_exists(conn, table: str, col: str) -> bool:
        raw_cur = conn._raw.cursor()
        raw_cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in raw_cur.fetchall()]
        raw_cur.close()
        return col in cols
