from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
import hashlib
import hmac
import threading
from cryptography.fernet import Fernet
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Literal, Any, Deque
from collections import deque

import httpx
from fastapi import FastAPI, Depends, HTTPException, Response, Cookie, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from database import db, db_conn, USING_PG, column_exists, serial_pk
from config import (
    ENV, IS_PROD, REAL_TRADING, FRONTEND_ORIGINS,
    SMTP_USER, SMTP_PASS, SMTP_HOST, SMTP_PORT,
    OKX_BASE, OKX_TF_MAP, TF_MIN_INTERVAL,
    START_EQUITY, RiskMode, Side, TF, TradeStyle, TF_MAP, HIGHER_TF_MAP,
    TRADE_STYLE_PARAMS, _MID_CANDLE_THRESHOLDS, _MID_CANDLE_MIN_SCORE, _MID_CANDLE_INTERVAL,
    DUBAI_TZ, now_dubai, dubai_day_key,
)
from notifications import (
    tg_alert as _tg_alert,
    email_ai_started, email_trade_opened,
    email_trade_closed, email_ai_stopped,
)
from indicators import (
    _ema, _rsi, _atr, _adx, _rsi_series,
    _session_quality, _ema_spread_trend, _candle_pattern, _rsi_divergence,
    _classify_regime, _compute_signal_layers,
)
from routes_backtest import backtest_router
from routes_optimizer import optimizer_router
from routes_support import support_router
from routes_auth import auth_router, hash_pw
from routes_admin import admin_router

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

_SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if _SENTRY_DSN:
    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        integrations=[
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        traces_sample_rate=0.05,
        environment=os.getenv("RENDER_SERVICE_NAME", "local"),
        release=os.getenv("RENDER_GIT_COMMIT", "unknown"),
    )

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

_SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if _SENTRY_DSN:
    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        integrations=[
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        traces_sample_rate=0.05,  # 5% of requests — stays within free tier
        environment=os.getenv("RENDER_SERVICE_NAME", "local"),
        release=os.getenv("RENDER_GIT_COMMIT", "unknown"),
    )

# =========================
# Telegram alerts
# =========================
_TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID",  "").strip()

def _tg_alert(text: str) -> None:
    """Fire-and-forget Telegram message. Never raises, never blocks the engine."""
    if not (_TG_TOKEN and _TG_CHAT):
        return
    def _send():
        try:
            import urllib.request, urllib.parse
            url = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"
            payload = urllib.parse.urlencode({
                "chat_id": _TG_CHAT, "text": text, "parse_mode": "HTML"
            }).encode()
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=8):
                pass
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

# =========================
# App
# =========================
app = FastAPI(title="Asymmetric AI Backend", version="0.9.0")

# Simple in-memory rate limiter — keyed by (endpoint, ip)
_rl_hits: Dict[str, List[float]] = {}
_rl_lock = threading.Lock()

def _rate_limit(request: Request, limit: int, window: int = 60) -> None:
    key = f"{request.url.path}:{(request.client.host if request.client else 'unknown')}"
    now = time.time()
    with _rl_lock:
        hits = [t for t in _rl_hits.get(key, []) if now - t < window]
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests — slow down.")
        hits.append(now)
        _rl_hits[key] = hits

# =========================
# ENV / PROD SETTINGS
# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(backtest_router)
app.include_router(optimizer_router)
app.include_router(support_router)
app.include_router(auth_router)
app.include_router(admin_router)

# Strip /api prefix forwarded by Vercel
@app.middleware("http")
async def strip_api_prefix(request: Request, call_next):
    path = request.scope.get("path", "")
    if path == "/api":
        request.scope["path"] = "/"
    elif path.startswith("/api/"):
        request.scope["path"] = path[4:]
    return await call_next(request)


# ── Global rate limiter: 60 req/min per user (JWT) or per IP (unauthenticated) ──
# Keyed by JWT sub or IP. Excludes health/docs endpoints.
_GLOBAL_RL_HITS: Dict[str, List[float]] = {}
_GLOBAL_RL_LOCK = threading.Lock()

# Never rate-limit these (infra / docs)
_GLOBAL_RL_SKIP = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

# High-frequency dashboard read endpoints — higher ceiling (600/min per user)
_GLOBAL_RL_DASHBOARD = {
    "/balance", "/exchange/balance", "/exchange/status",
    "/auto/status", "/auto/history", "/auto/sessions", "/auto/signal",
    "/portfolio/stats", "/portfolio/pnl", "/market/tickers",
}

# Authenticated users: 300/min  |  Unauthenticated (real IP): 120/min
_RL_LIMIT_AUTH = 300
_RL_LIMIT_IP   = 120
_RL_LIMIT_DASH = 600  # dashboard polling endpoints

@app.middleware("http")
async def global_rate_limit(request: Request, call_next):
    path = request.scope.get("path", "")

    # Always skip infra paths
    if path in _GLOBAL_RL_SKIP:
        return await call_next(request)

    # Skip price/ticker/symbols endpoints — they hit the exchange cache, not our DB
    if path.startswith("/price/") or path.startswith("/market/ticker/") or path == "/market/symbols":
        return await call_next(request)

    # Identify caller: prefer JWT sub, fall back to IP
    auth_header = request.headers.get("authorization", "")
    rl_key = None
    is_authed = False
    if auth_header.startswith("Bearer "):
        try:
            import jose.jwt as _jwt
            payload = _jwt.get_unverified_claims(auth_header[7:])
            rl_key = f"user:{payload.get('sub', '')}"
            is_authed = True
        except Exception:
            pass
    if not rl_key:
        # X-Forwarded-For is set by Render's proxy with the real client IP.
        # Falling back to request.client.host gives Render's internal proxy IP,
        # which is shared by all users — they all hit the same 60/min bucket.
        xff = request.headers.get("x-forwarded-for", "")
        real_ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "unknown")
        rl_key = f"ip:{real_ip}"

    # Choose limit based on endpoint type and auth status
    if path in _GLOBAL_RL_DASHBOARD and is_authed:
        limit = _RL_LIMIT_DASH
    elif is_authed:
        limit = _RL_LIMIT_AUTH
    else:
        limit = _RL_LIMIT_IP

    now = time.time()
    with _GLOBAL_RL_LOCK:
        hits = [t for t in _GLOBAL_RL_HITS.get(rl_key, []) if now - t < 60]
        if len(hits) >= limit:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait 60 seconds before trying again."},
            )
        hits.append(now)
        _GLOBAL_RL_HITS[rl_key] = hits

    return await call_next(request)


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    _serial = serial_pk()

    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS exchange_keys (
            email TEXT PRIMARY KEY,
            exchange TEXT NOT NULL,
            api_key TEXT NOT NULL,
            api_secret TEXT NOT NULL,
            passphrase TEXT,
            created_at TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_state (
            email TEXT PRIMARY KEY,
            equity REAL NOT NULL,
            session_id INTEGER NOT NULL DEFAULT 0
        )
        """)

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS trades (
            id {_serial},
            email TEXT NOT NULL,
            time TEXT NOT NULL,
            side TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            size REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            leverage REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL,
            unreal_pnl_percent REAL NOT NULL,
            unreal_pnl_value REAL NOT NULL,
            equity_after REAL NOT NULL,
            reason TEXT,
            session_id INTEGER NOT NULL DEFAULT 0
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_settings (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_notes (
            email      TEXT PRIMARY KEY,
            notes      TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            updated_by TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_access_log (
            id         SERIAL PRIMARY KEY,
            admin_email TEXT NOT NULL,
            user_email  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            tab_viewed  TEXT NOT NULL DEFAULT 'overview'
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        )
        """)

        # OTP codes for forgot-password (6-digit, 15min expiry)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS otp_codes (
            email TEXT PRIMARY KEY,
            code  TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        )
        """)

        # 2FA TOTP secrets — one per user, optional
        cur.execute("""
        CREATE TABLE IF NOT EXISTS totp_secrets (
            email      TEXT PRIMARY KEY,
            secret     TEXT NOT NULL,
            enabled    INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """)

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS ai_sessions (
            id          {_serial},
            email       TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            mode        TEXT NOT NULL,
            trade_style TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            ended_at    TEXT,
            stop_reason TEXT
        )
        """)

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS ai_logs (
            id          {_serial},
            email       TEXT NOT NULL,
            session_id  INTEGER,
            t           TEXT NOT NULL,
            msg         TEXT NOT NULL
        )
        """)
        if not column_exists(conn, "ai_logs", "session_id"):
            cur.execute("ALTER TABLE ai_logs ADD COLUMN session_id INTEGER")

        # Performance indexes
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ai_sessions_email ON ai_sessions(email)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ai_logs_session ON ai_logs(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ai_logs_email ON ai_logs(email)")
        # Composite: covers "WHERE email=? ORDER BY t DESC" — the most common log query
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ai_logs_email_t ON ai_logs(email, t DESC)")
        # Timestamp index: covers time-range log queries and ORDER BY t
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ai_logs_t ON ai_logs(t DESC)")
        # Composite: covers "WHERE email=? AND session_id=? ORDER BY t DESC/ASC" — every
        # log viewer call filters by both email and session. Without this the DB scans all
        # rows for the email then re-filters by session.
        cur.execute("CREATE INDEX IF NOT EXISTS ix_ai_logs_email_session_t ON ai_logs(email, session_id, t DESC)")
        # Trades indexes: no index existed — every trade history query was a full scan
        cur.execute("CREATE INDEX IF NOT EXISTS ix_trades_email ON trades(email)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_trades_email_time ON trades(email, time DESC)")
        # Composite: covers "WHERE email=? AND session_id=? AND time>=?" — daily stats,
        # bad-trade counts, and session P&L summaries all filter on both columns.
        cur.execute("CREATE INDEX IF NOT EXISTS ix_trades_email_session ON trades(email, session_id, time DESC)")
        # Symbol-level lookup: "WHERE email=? AND symbol=? AND session_id=?" used in
        # correct-trade endpoint and per-symbol performance queries.
        cur.execute("CREATE INDEX IF NOT EXISTS ix_trades_email_symbol_session ON trades(email, symbol, session_id)")
        # Win-rate query: "WHERE email=? AND unreal_pnl_value >= 0" — avoids full email scan
        cur.execute("CREATE INDEX IF NOT EXISTS ix_trades_email_pnl ON trades(email, unreal_pnl_value)")
        # Sessions: email lookup (token is already primary key)
        cur.execute("CREATE INDEX IF NOT EXISTS ix_sessions_email ON sessions(email)")

        # AI runner persistence — survives backend restarts / deploys.
        # Saved when user starts the AI, cleared when AI stops for any reason.
        # On startup, all rows here are auto-resumed so live sessions never drop.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_runner_state (
            email               TEXT PRIMARY KEY,
            symbol              TEXT NOT NULL,
            trade_style         TEXT NOT NULL,
            mode                TEXT NOT NULL,
            max_trades_per_day  INTEGER NOT NULL,
            stop_after_bad_trades INTEGER NOT NULL,
            duration_days       INTEGER NOT NULL,
            trend_filter        INTEGER NOT NULL,
            chop_min_sep_pct    REAL NOT NULL,
            end_at_ts           REAL,
            started_at          INTEGER NOT NULL
        )
        """)

        if not column_exists(conn, "users", "display_name"):
            cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT DEFAULT ''")

        # Safe migrations for existing databases
        if not column_exists(conn, "trades", "reason"):
            cur.execute("ALTER TABLE trades ADD COLUMN reason TEXT")
        if not column_exists(conn, "trades", "session_id"):
            cur.execute("ALTER TABLE trades ADD COLUMN session_id INTEGER NOT NULL DEFAULT 0")
        if not column_exists(conn, "user_state", "session_id"):
            cur.execute("ALTER TABLE user_state ADD COLUMN session_id INTEGER NOT NULL DEFAULT 0")
        if not column_exists(conn, "trades", "hard_floor"):
            cur.execute("ALTER TABLE trades ADD COLUMN hard_floor REAL DEFAULT 0")
        if not column_exists(conn, "trades", "is_primary"):
            cur.execute("ALTER TABLE trades ADD COLUMN is_primary INTEGER DEFAULT 1")
        if not column_exists(conn, "user_state", "peak_equity"):
            cur.execute("ALTER TABLE user_state ADD COLUMN peak_equity REAL DEFAULT 0")
        if not column_exists(conn, "user_state", "all_time_high"):
            cur.execute("ALTER TABLE user_state ADD COLUMN all_time_high REAL DEFAULT 0")
        # Floor reset tracking — records initial starting capital and reset history for Situation 1/2/3 logic
        if not column_exists(conn, "user_state", "starting_capital"):
            cur.execute("ALTER TABLE user_state ADD COLUMN starting_capital REAL DEFAULT 0")
        if not column_exists(conn, "user_state", "floor_reset_count"):
            cur.execute("ALTER TABLE user_state ADD COLUMN floor_reset_count INTEGER DEFAULT 0")
        if not column_exists(conn, "user_state", "floor_reset_at"):
            # JSON array of Unix timestamps — used to enforce 7-day cooldown after 3 resets in 30 days
            cur.execute("ALTER TABLE user_state ADD COLUMN floor_reset_at TEXT DEFAULT '[]'")
        # Persistent floor — separate from peak so it survives ai_runner_state deletion.
        # Updated via max() only — never decreases except on explicit reset (reset_sandbox / reset-floor).
        # Fixes: floor dropping after redeploy when ai_runner_state (and its floor_equity column) is cleared.
        if not column_exists(conn, "user_state", "floor_equity"):
            cur.execute("ALTER TABLE user_state ADD COLUMN floor_equity REAL DEFAULT 0")
        # Persists why the engine last stopped — allows frontend to show correct warning on restart.
        # Cleared when engine starts successfully (auto_start writes NULL).
        if not column_exists(conn, "user_state", "last_stop_reason"):
            cur.execute("ALTER TABLE user_state ADD COLUMN last_stop_reason TEXT DEFAULT NULL")
        # Live runner state — persists adaptive_strictness, open positions, etc. across redeploys
        for col, coltype, defval in [
            ("adaptive_strictness",  "REAL",    "1.0"),
            ("last_trade_ts",        "REAL",    "0"),
            ("last_trade_bad",       "INTEGER", "0"),
            ("pending_trades_json",  "TEXT",    "''"),
            ("peak_equity",          "REAL",    "0"),
            ("floor_equity",         "REAL",    "0"),
            ("consecutive_wins",     "INTEGER", "0"),
            ("session_start_equity", "REAL",    "0"),
            ("strictness_day_key",   "TEXT",    "''"),  # Dubai date when strictness was last changed
        ]:
            if not column_exists(conn, "ai_runner_state", col):
                cur.execute(f"ALTER TABLE ai_runner_state ADD COLUMN {col} {coltype} DEFAULT {defval}")

        # Support chat tables
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS support_tickets (
            id                  {_serial},
            user_email          TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'open',
            subject             TEXT NOT NULL DEFAULT '',
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            last_user_msg_at    TIMESTAMPTZ DEFAULT NOW(),
            last_admin_reply_at TIMESTAMPTZ
        )
        """)

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS support_messages (
            id          {_serial},
            ticket_id   INTEGER NOT NULL,
            sender      TEXT NOT NULL,
            message     TEXT NOT NULL,
            sent_at     TIMESTAMPTZ DEFAULT NOW()
        )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS ix_support_tickets_user ON support_tickets(user_email)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_support_tickets_status ON support_tickets(status, last_user_msg_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_support_messages_ticket ON support_messages(ticket_id, sent_at)")

        # Seed admin settings defaults
        def _set_default(key: str, value: str):
            cur.execute("SELECT v FROM admin_settings WHERE k=%s", (key,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO admin_settings(k,v,updated_at) VALUES(%s,%s,%s)",
                    (key, value, now_utc_str()),
                )

        _set_default("signup_enabled", "true")
        _set_default("seat_capacity", "50")

        conn.commit()


init_db()
from backtester import init_backtest_tables; init_backtest_tables()
from optimizer import init_optimizer_tables; init_optimizer_tables()

# Runners are resumed after all AutoRunner code is defined — see bottom of file.

# =========================
# SECURITY — KEY ENCRYPTION
# =========================

# Fernet encryption for API keys stored in DB
# Set ENCRYPTION_KEY env var to a Fernet key (generate once: Fernet.generate_key().decode())
_RAW_ENC_KEY = os.getenv("ENCRYPTION_KEY", "")
_FERNET: Fernet | None = None
if _RAW_ENC_KEY:
    try:
        _FERNET = Fernet(_RAW_ENC_KEY.encode() if isinstance(_RAW_ENC_KEY, str) else _RAW_ENC_KEY)
    except Exception:
        pass

if not _FERNET:
    import base64, warnings
    _FALLBACK = base64.urlsafe_b64encode(hashlib.sha256(b"asym-dev-key").digest())
    _FERNET = Fernet(_FALLBACK)
    if os.getenv("ENV", "dev") == "prod":
        warnings.warn("ENCRYPTION_KEY not set — using dev fallback. Set ENCRYPTION_KEY in prod!")


def encrypt_key(value: str) -> str:
    if not value:
        return ""
    return _FERNET.encrypt(value.encode()).decode()


def decrypt_key(value: str) -> str:
    if not value:
        return ""
    try:
        return _FERNET.decrypt(value.encode()).decode()
    except Exception:
        # Fallback: might be unencrypted legacy value
        return value


def mask_key(s: str) -> str:
    s = (s or "").strip()
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + ("*" * (len(s) - 8)) + s[-4:]


def ensure_user_state(email: str) -> None:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT equity, session_id FROM user_state WHERE email = %s", (email,))
        row = cur.fetchone()
        if not row:
            cur.execute(
                "INSERT INTO user_state(email, equity, session_id) VALUES(%s, %s, %s)"
                " ON CONFLICT (email) DO NOTHING",
                (email, START_EQUITY, 0),
            )
            conn.commit()
        elif float(row["equity"]) <= 0:
            # Equity is zero — reset to configured start amount
            cur.execute("UPDATE user_state SET equity = %s WHERE email = %s", (START_EQUITY, email))
            conn.commit()


def get_equity(email: str) -> float:
    ensure_user_state(email)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT equity FROM user_state WHERE email = %s", (email,))
        row = cur.fetchone()
    return float(row["equity"])


def set_equity(email: str, equity: float) -> None:
    ensure_user_state(email)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET equity = %s WHERE email = %s", (float(equity), email))
        conn.commit()


def update_peak_ath(email: str, peak: float, equity_after: float) -> None:
    """Update peak_equity, all_time_high, and floor_equity in user_state after each trade.
    All three columns only ever increase — never decrease — so the floor survives runner
    restarts, redeploys, and ai_runner_state deletions without dropping."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT peak_equity, all_time_high, floor_equity FROM user_state WHERE email = %s",
            (email,),
        )
        row = cur.fetchone()
        if row:
            new_peak = max(float(row["peak_equity"] or 0), peak)
            new_ath = max(float(row["all_time_high"] or 0), equity_after)
            # Floor is 85% of the running peak — but only ever moves UP.
            # Stored here so it persists even after ai_runner_state is deleted on stop.
            new_floor = max(float(row["floor_equity"] or 0), round(new_peak * 0.85, 2))
            cur.execute(
                "UPDATE user_state SET peak_equity=%s, all_time_high=%s, floor_equity=%s "
                "WHERE email=%s",
                (new_peak, new_ath, new_floor, email),
            )
            conn.commit()


def get_session_id(email: str) -> int:
    ensure_user_state(email)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT session_id FROM user_state WHERE email = %s", (email,))
        row = cur.fetchone()
    return int(row["session_id"] or 0)


def set_session_id(email: str, sid: int) -> None:
    ensure_user_state(email)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET session_id = %s WHERE email = %s", (int(sid), email))
        conn.commit()


def get_exchange(email: str) -> Optional[dict]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM exchange_keys WHERE email = %s", (email,))
        row = cur.fetchone()
    if not row:
        return None
    # Decrypt keys before returning — callers always get plain text
    return {
        **dict(row),
        "api_key":    decrypt_key(row["api_key"]),
        "api_secret": decrypt_key(row["api_secret"]),
        "passphrase": decrypt_key(row.get("passphrase") or ""),
    }


_SESSION_CACHE: Dict[str, tuple] = {}    # token → (email, cached_at_ts)
_SESSION_TTL = 300                        # reuse cached token for 5 minutes

_EX_STATUS_CACHE: Dict[str, tuple] = {}  # email → (status_dict, ts)
_EX_STATUS_TTL = 60                       # re-read DB at most once per minute
_EX_TEST_CACHE: Dict[str, tuple] = {}    # email → (test_result_dict, ts)
_EX_TEST_TTL = 60                         # live Bybit test cached 60s

def require_user(session: Optional[str] = Cookie(default=None)) -> Dict[str, str]:
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    now = time.time()
    cached = _SESSION_CACHE.get(session)
    if cached:
        email, ts = cached
        if now - ts < _SESSION_TTL:
            return {"email": email}

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM sessions WHERE token = %s", (session,))
        row = cur.fetchone()
    if not row:
        _SESSION_CACHE.pop(session, None)
        raise HTTPException(status_code=401, detail="Unauthorized")
    _SESSION_CACHE[session] = (row["email"], now)
    return {"email": row["email"]}


# =========================
# ADMIN
# =========================
DEMO_EMAIL = "admin@demo.com"
DEMO_PASS = "demo123"

ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", DEMO_EMAIL).split(",")
    if e.strip()
)

ADMIN_FORCE_RESET_TOKEN = os.getenv("ADMIN_FORCE_RESET_TOKEN", "")


def require_admin(user=Depends(require_user)) -> str:
    email = user["email"].strip().lower()
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access only")
    return email


def admin_get_setting(key: str, default: str) -> str:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT v FROM admin_settings WHERE k=%s", (key,))
        row = cur.fetchone()
    return (row["v"] if row else default)


def admin_set_setting(key: str, value: str) -> None:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO admin_settings(k,v,updated_at) VALUES(%s,%s,%s)
            ON CONFLICT(k) DO UPDATE SET
                v=excluded.v,
                updated_at=excluded.updated_at
            """,
            (key, value, now_utc_str()),
        )
        conn.commit()


def signup_is_enabled() -> bool:
    if os.environ.get("INVITE_ONLY", "").lower() == "true":
        return False
    return admin_get_setting("signup_enabled", "true").lower() == "true"


def seat_capacity() -> int:
    try:
        return int(admin_get_setting("seat_capacity", "50"))
    except Exception:
        return 50


def seats_used() -> int:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM users")
        n = int(cur.fetchone()["n"])
    return n


# Seed demo admin user if missing
with db_conn() as conn:
    cur = conn.cursor()
    cur.execute("SELECT email FROM users WHERE email = %s", (DEMO_EMAIL,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users(email, password_hash, created_at) VALUES(%s, %s, %s)",
            (DEMO_EMAIL, hash_pw(DEMO_PASS), now_utc_str()),
        )
        conn.commit()

# =========================
# BASIC INFO
# =========================
@app.get("/config")
def config():
    return {
        "market": "okx",
        "timezone": "Asia/Dubai (UTC+4)",
        "env": ENV,
        "frontend_origins": FRONTEND_ORIGINS,
        "db": "postgresql" if USING_PG else "sqlite",
        "real_trading": REAL_TRADING,
    }


@app.get("/health")
def health():
    return {
        "health": "green",
        "market": "okx",
        "db": "postgresql" if USING_PG else "sqlite",
        "env": ENV,
    }


# Auth endpoints moved to routes_auth.py (auth_router)

# =========================
# PROFILE + CONFIG
# =========================

class ProfileIn(BaseModel):
    display_name: str = Field(..., max_length=40)

@app.get("/profile")
def get_profile(user=Depends(require_user)):
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT display_name FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
    name = (row["display_name"] if row and row["display_name"] else "").strip()
    if not name:
        name = email.split("@")[0]
    return {"email": email, "display_name": name}

@app.post("/profile")
def update_profile(payload: ProfileIn, user=Depends(require_user)):
    email = user["email"]
    name = payload.display_name.strip()[:40]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET display_name = %s WHERE email = %s", (name, email))
        conn.commit()
    return {"ok": True, "display_name": name}

# =========================
# EXCHANGE CONNECT
# =========================
class ExchangeConnectIn(BaseModel):
    exchange: Literal["binance", "okx", "bybit"]
    api_key: str = Field(..., min_length=10, max_length=256)
    api_secret: str = Field(..., min_length=10, max_length=256)
    passphrase: Optional[str] = Field(default=None, max_length=256)


@app.get("/exchange/status")
def exchange_status(user=Depends(require_user)):
    email = user["email"]
    cached = _EX_STATUS_CACHE.get(email)
    if cached:
        result, ts = cached
        if time.time() - ts < _EX_STATUS_TTL:
            return result
    row = get_exchange(email)
    if not row:
        result = {"connected": False, "exchange": None, "api_key_masked": None, "created_at": None}
    else:
        result = {
            "connected": True,
            "exchange": row["exchange"],
            "api_key_masked": mask_key(row["api_key"]),
            "created_at": row["created_at"],
        }
    _EX_STATUS_CACHE[email] = (result, time.time())
    return result


@app.post("/exchange/connect")
def exchange_connect(payload: ExchangeConnectIn, user=Depends(require_user)):
    email = user["email"]
    if not payload.api_key.strip() or not payload.api_secret.strip():
        raise HTTPException(status_code=400, detail="API key/secret required")

    created_at = now_utc_str()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO exchange_keys(email, exchange, api_key, api_secret, passphrase, created_at)
            VALUES(%s, %s, %s, %s, %s, %s)
            ON CONFLICT(email) DO UPDATE SET
                exchange=excluded.exchange,
                api_key=excluded.api_key,
                api_secret=excluded.api_secret,
                passphrase=excluded.passphrase,
                created_at=excluded.created_at
            """,
            (
                email,
                payload.exchange,
                encrypt_key(payload.api_key.strip()),
                encrypt_key(payload.api_secret.strip()),
                encrypt_key((payload.passphrase or "").strip()),
                created_at,
            ),
        )
        conn.commit()

    # Clear caches so new keys are visible immediately
    _EX_STATUS_CACHE.pop(email, None)
    _EX_TEST_CACHE.pop(email, None)
    _REAL_BAL_CACHE.pop(email, None)
    return {"ok": True}


@app.post("/exchange/disconnect")
def exchange_disconnect(user=Depends(require_user)):
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM exchange_keys WHERE email = %s", (email,))
        conn.commit()
    _EX_STATUS_CACHE.pop(email, None)
    _EX_TEST_CACHE.pop(email, None)
    _REAL_BAL_CACHE.pop(email, None)
    return {"ok": True}


# ── Per-exchange fee table (futures/perpetuals, USD-margined) ─────────────────
# maker = limit order (you add liquidity)   taker = market order (you take liquidity)
# SL exits are always market (taker). TP exits can be limit (maker).
# Slippage estimate is conservative round-trip for <$20k position size.
EXCHANGE_FEES = {
    "bybit": {
        "maker":     0.00020,   # 0.020%
        "taker":     0.00055,   # 0.055%
        "slippage":  0.00025,   # ~0.025% round-trip (BTC liquid, alts slightly more)
        "min_qty":   0.001,     # BTC minimum order size
    },
    "okx": {
        "maker":     0.00020,   # 0.020%
        "taker":     0.00050,   # 0.050%
        "slippage":  0.00030,   # ~0.030% round-trip
        "min_qty":   0.001,
    },
    "binance": {
        "maker":     0.00020,   # 0.020% (futures)
        "taker":     0.00040,   # 0.040% (futures, lower than spot)
        "slippage":  0.00020,   # ~0.020% round-trip (most liquid exchange)
        "min_qty":   0.001,
    },
}

def exchange_fee_cost(exchange_id: str, size_dollar: float, outcome: str) -> float:
    """
    Calculate total fee + slippage cost for one complete trade (entry + exit).
    outcome: "TP_HIT" (limit exit) | "SL_HIT" | "TRAIL_STOP" | other (market exit)
    Returns dollar cost to deduct from PnL.
    """
    fees = EXCHANGE_FEES.get(exchange_id.lower(), EXCHANGE_FEES["bybit"])
    entry_fee   = size_dollar * fees["maker"]           # entry always limit
    exit_fee    = size_dollar * (fees["maker"] if outcome == "TP_HIT" else fees["taker"])
    slippage    = size_dollar * fees["slippage"]
    return round(entry_fee + exit_fee + slippage, 4)


def _make_ccxt_exchange(row: dict):
    """Build a ccxt exchange instance from stored (decrypted) keys."""
    import ccxt
    exchange_id = (row.get("exchange") or "bybit").lower()
    config = {
        "apiKey":   row["api_key"],
        "secret":   row["api_secret"],
        "options":  {"defaultType": "linear"},
        "enableRateLimit": True,
        "timeout":  10000,   # 10-second timeout on every ccxt network call
    }
    if row.get("passphrase"):
        config["password"] = row["passphrase"]  # required for OKX
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    return cls(config)


def _ccxt_call(fn, *args, label: str = "exchange", retries: int = 1, **kwargs):
    """
    Wrap any ccxt call with timeout detection, 1 retry, and clear error logging.
    Raises a RuntimeError with a specific EXCHANGE_TIMEOUT prefix on timeout.
    Never crashes the engine — caller catches RuntimeError and continues.
    """
    import ccxt
    last_exc = None
    for attempt in range(1, retries + 2):  # attempts = retries + 1
        try:
            return fn(*args, **kwargs)
        except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
            last_exc = e
            is_timeout = isinstance(e, ccxt.RequestTimeout) or "timeout" in str(e).lower()
            tag = "EXCHANGE_TIMEOUT" if is_timeout else "EXCHANGE_NETWORK_ERROR"
            print(f"[{tag}] {label} attempt {attempt}: {e}", flush=True)
            if attempt <= retries:
                import time as _t
                _t.sleep(2)
        except ccxt.AuthenticationError as e:
            raise RuntimeError(f"EXCHANGE_AUTH_ERROR: API key rejected by exchange — {e}")
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"EXCHANGE_ERROR ({label}): {e}")
        except Exception as e:
            raise RuntimeError(f"EXCHANGE_UNEXPECTED ({label}): {type(e).__name__}: {e}")
    raise RuntimeError(f"EXCHANGE_TIMEOUT: {label} did not respond within 10 seconds after {retries + 1} attempts")


@app.get("/exchange/test")
def exchange_test(user=Depends(require_user)):
    """Test API key against the real exchange — cached 60s so page loads stay fast."""
    email = user["email"]
    cached = _EX_TEST_CACHE.get(email)
    if cached:
        result, ts = cached
        if time.time() - ts < _EX_TEST_TTL:
            return result
    row = get_exchange(email)
    if not row:
        raise HTTPException(status_code=400, detail="No exchange connected.")
    ex_id = (row.get("exchange") or "bybit").lower()
    try:
        if ex_id == "bybit":
            bal, err_msg = _bybit_direct_balance(row["api_key"], row["api_secret"])
            if bal is None:
                raise ValueError(f"Bybit balance fetch failed: {err_msg}")
            result = {"ok": True, "exchange": row["exchange"], "canTrade": True,
                      "accountType": "UNIFIED", "usdt_free": round(bal, 4),
                      "usdt_total": round(bal, 4), "note": "Live connection verified ✓"}
        elif ex_id == "okx":
            bal = _okx_direct_balance(row["api_key"], row["api_secret"], row.get("passphrase") or "")
            if bal is None:
                raise ValueError("Could not fetch OKX balance — check API key, secret, passphrase, and permissions.")
            result = {"ok": True, "exchange": row["exchange"], "canTrade": True,
                      "accountType": "UNIFIED", "usdt_free": round(bal, 4),
                      "usdt_total": round(bal, 4), "note": "Live connection verified ✓"}
        elif ex_id == "binance":
            bal = _binance_direct_balance(row["api_key"], row["api_secret"])
            if bal is None:
                raise ValueError("Could not fetch Binance Futures balance — check API key, secret, and Futures permissions.")
            result = {"ok": True, "exchange": row["exchange"], "canTrade": True,
                      "accountType": "FUTURES", "usdt_free": round(bal, 4),
                      "usdt_total": round(bal, 4), "note": "Live connection verified ✓"}
        else:
            raise ValueError(f"Unsupported exchange: {ex_id}")
    except Exception as e:
        result = {"ok": False, "canTrade": False,
                  "error": str(e)[:400],
                  "note": "Connection failed — see error below for exact reason."}
    _EX_TEST_CACHE[email] = (result, time.time())
    return result


@app.get("/exchange/balance")
def exchange_balance(user=Depends(require_user)):
    """Fetch real USDT balance from the connected exchange."""
    email = user["email"]
    row = get_exchange(email)
    if not row:
        raise HTTPException(status_code=400, detail="No exchange connected.")
    ex_id = (row.get("exchange") or "bybit").lower()
    try:
        if ex_id == "bybit":
            bal, err = _bybit_direct_balance(row["api_key"], row["api_secret"])
            if bal is not None:
                return {"ok": True, "balances": [{"asset": "USDT", "free": round(bal, 4), "locked": 0.0}], "note": "Live balance from exchange ✓"}
            raise ValueError(err)
        ex = _make_ccxt_exchange(row)
        balance = ex.fetch_balance({"accountType": "UNIFIED"})
        balances = []
        for asset in ["USDT", "BTC", "ETH", "SOL", "BNB"]:
            info = balance.get(asset, {})
            free  = float(info.get("free") or 0)
            total = float(info.get("total") or 0)
            if total > 0:
                balances.append({"asset": asset, "free": round(free, 6), "locked": round(total - free, 6)})
        return {"ok": True, "balances": balances, "note": "Live balance from exchange ✓"}
    except Exception as e:
        eq = get_equity(email)
        return {
            "ok": False,
            "balances": [{"asset": "USDT", "free": eq, "locked": 0.0}],
            "error": str(e)[:200],
            "note": "Could not reach exchange — showing paper balance.",
        }


BYBIT_HOSTS = ["https://api.bybit.com", "https://api.bytick.com"]

def _bybit_signed_get(api_key: str, api_secret: str, path: str, query: str) -> tuple[Optional[dict], str]:
    """Make a signed Bybit GET request, trying both hosts. Returns (parsed JSON or None, last_error)."""
    ts = str(int(time.time() * 1000))
    recv_window = "5000"
    sign_str = ts + api_key + recv_window + query
    sig = hmac.new(api_secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY":     api_key,
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-SIGN":        sig,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type":       "application/json",
    }
    last_error = "unknown"
    for host in BYBIT_HOSTS:
        try:
            resp = httpx.get(f"{host}{path}?{query}", headers=headers, timeout=10)
            print(f"[bybit] {host}{path} → HTTP {resp.status_code}")
            if resp.status_code == 200:
                return resp.json(), ""
            last_error = f"HTTP {resp.status_code} from {host}: {resp.text[:200]}"
            print(f"[bybit] non-200: {last_error}")
        except Exception as e:
            last_error = f"{type(e).__name__} from {host}: {e}"
            print(f"[bybit] exception: {last_error}")
    return None, last_error


def _bybit_direct_balance(api_key: str, api_secret: str) -> tuple:
    """
    Direct Bybit V5 signed request for USDT balance.
    Checks Unified Trading wallet first, then Funding wallet.
    Returns (balance_float, None) on success, (None, error_str) on failure.
    """
    try:
        # 1. Try Unified Trading wallet (required for trading)
        data, net_err = _bybit_signed_get(api_key, api_secret, "/v5/account/wallet-balance", "accountType=UNIFIED")
        if data is None:
            return (None, f"Could not reach Bybit API{' (' + net_err + ')' if net_err else ''}")
        ret_code = data.get("retCode")
        ret_msg  = data.get("retMsg", "")
        print(f"[bybit-balance] UNIFIED retCode={ret_code} retMsg={ret_msg}")
        if ret_code in (10003, 10004):
            return (None, f"Invalid API key or secret (retCode={ret_code}: {ret_msg})")
        if ret_code != 0:
            return (None, f"Bybit API error retCode={ret_code}: {ret_msg}")
        for acc in data.get("result", {}).get("list", []):
            for coin in acc.get("coin", []):
                if coin.get("coin") == "USDT":
                    val = float(coin.get("walletBalance") or coin.get("equity") or 0)
                    print(f"[bybit-balance] USDT in Unified Trading: {val}")
                    return (val, None)

        # 2. USDT not in Unified — check Funding wallet
        print("[bybit-balance] No USDT in Unified, checking Funding wallet...")
        fund_data, _ = _bybit_signed_get(api_key, api_secret, "/v5/asset/transfer/query-account-coins-balance", "accountType=FUND&coin=USDT")
        if fund_data and fund_data.get("retCode") == 0:
            for item in fund_data.get("result", {}).get("balance", []):
                if item.get("coin") == "USDT":
                    val = float(item.get("walletBalance") or item.get("transferBalance") or 0)
                    print(f"[bybit-balance] USDT in Funding wallet: {val}")
                    if val > 0:
                        return (None, f"Your {val:.2f} USDT is in the Funding wallet — transfer it to Unified Trading in Bybit before the engine can trade.")
        return (None, "No USDT found in Unified Trading or Funding wallet. Please deposit USDT to your Bybit account.")
    except Exception as ex:
        return (None, f"Exception: {ex}")


def _okx_direct_balance(api_key: str, api_secret: str, passphrase: str) -> Optional[float]:
    """Direct OKX V5 signed request for USDT balance."""
    import base64
    from datetime import timezone as _tz
    ts = datetime.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.') + \
         str(datetime.now(_tz.utc).microsecond // 1000).zfill(3) + 'Z'
    path = "/api/v5/account/balance"
    sign_str = ts + "GET" + path + ""
    sig = base64.b64encode(
        hmac.new(api_secret.encode(), sign_str.encode(), hashlib.sha256).digest()
    ).decode()
    headers = {
        "OK-ACCESS-KEY":        api_key,
        "OK-ACCESS-SIGN":       sig,
        "OK-ACCESS-TIMESTAMP":  ts,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type":         "application/json",
    }
    try:
        resp = httpx.get(f"https://www.okx.com{path}", headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            for detail in data.get("data", [{}])[0].get("details", []):
                if detail.get("ccy") == "USDT":
                    return float(detail.get("cashBal") or detail.get("availBal") or 0)
    except Exception:
        pass
    return None


def _binance_direct_balance(api_key: str, api_secret: str) -> Optional[float]:
    """Direct Binance Futures V2 signed request for USDT balance."""
    ts = str(int(time.time() * 1000))
    params = f"timestamp={ts}"
    sig = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    try:
        resp = httpx.get(
            f"https://fapi.binance.com/fapi/v2/balance?{params}&signature={sig}",
            headers=headers, timeout=10,
        )
        data = resp.json()
        if isinstance(data, list):
            for item in data:
                if item.get("asset") == "USDT":
                    return float(item.get("balance") or item.get("availableBalance") or 0)
    except Exception:
        pass
    return None


_REAL_BAL_CACHE: Dict[str, tuple] = {}      # email → (balance, fetched_at_ts)
_REAL_BAL_TTL = 30                           # seconds before background refresh fires
_REAL_BAL_REFRESHING: set = set()            # emails with a refresh already in flight

def _do_balance_refresh(email: str) -> None:
    """Background thread: fetch live balance and silently update cache."""
    try:
        row = get_exchange(email)
        if not row:
            return
        ex_id = (row.get("exchange") or "bybit").lower()
        if ex_id == "bybit":
            bal, _ = _bybit_direct_balance(row["api_key"], row["api_secret"])
        elif ex_id == "okx":
            bal = _okx_direct_balance(row["api_key"], row["api_secret"], row.get("passphrase") or "")
        elif ex_id == "binance":
            bal = _binance_direct_balance(row["api_key"], row["api_secret"])
        else:
            bal = None
        if bal is not None:
            _REAL_BAL_CACHE[email] = (bal, time.time())
    finally:
        _REAL_BAL_REFRESHING.discard(email)

def get_real_usdt_balance(email: str, force: bool = False) -> Optional[float]:
    """Return cached balance instantly; refresh from Bybit in background when stale.
    force=True blocks for a fresh fetch (used before runner init)."""
    cached = _REAL_BAL_CACHE.get(email)

    if not force and cached:
        bal, fetched_at = cached
        if time.time() - fetched_at >= _REAL_BAL_TTL and email not in _REAL_BAL_REFRESHING:
            _REAL_BAL_REFRESHING.add(email)
            threading.Thread(target=_do_balance_refresh, args=(email,), daemon=True).start()
        return bal  # always instant when any cached value exists

    # No cache yet or force=True — block for fresh data
    row = get_exchange(email)
    if not row:
        return None
    ex_id = (row.get("exchange") or "bybit").lower()
    if ex_id == "bybit":
        bal, _ = _bybit_direct_balance(row["api_key"], row["api_secret"])
    elif ex_id == "okx":
        bal = _okx_direct_balance(row["api_key"], row["api_secret"], row.get("passphrase") or "")
    elif ex_id == "binance":
        bal = _binance_direct_balance(row["api_key"], row["api_secret"])
    else:
        bal = None
    if bal is not None:
        _REAL_BAL_CACHE[email] = (bal, time.time())
    return bal


# =========================
# MARKET (PUBLIC) — PRICE (multi-source fallback)
# OKX geo-blocks Render US; Binance US is primary, OKX as secondary fallback.
# =========================
def to_okx_inst(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if "-" in s:
        return s
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
    return s


async def _fetch_price_binance_us(symbol: str) -> float:
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": symbol.upper()},
            headers={"accept": "application/json"},
        )
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:120]}")
    price = r.json().get("price")
    if price is None:
        raise ValueError("no price field")
    return float(price)


async def _fetch_price_mexc(symbol: str) -> float:
    """MEXC global exchange — no geo-blocking, same format as Binance."""
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(
            "https://api.mexc.com/api/v3/ticker/price",
            params={"symbol": symbol.upper()},
            headers={"accept": "application/json"},
        )
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:120]}")
    price = r.json().get("price")
    if price is None:
        raise ValueError("no price field")
    return float(price)


async def _fetch_price_okx(symbol: str) -> float:
    inst = to_okx_inst(symbol)
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(
            f"{OKX_BASE}/api/v5/market/ticker",
            params={"instId": inst},
            headers={"accept": "application/json"},
        )
    if r.status_code == 429:
        raise ValueError(f"OKX 429 rate-limit — backing off")
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:120]}")
    arr = (r.json() or {}).get("data") or []
    if not arr:
        raise ValueError(f"no ticker for {inst}")
    last = arr[0].get("last")
    if last is None:
        raise ValueError(f"missing last price for {inst}")
    return float(last)


async def okx_price(symbol: str) -> float:
    """Fetch spot price with fallback: Binance US → MEXC → OKX."""
    errors: list = []
    for name, fn in [
        ("BinanceUS", _fetch_price_binance_us),
        ("MEXC",      _fetch_price_mexc),
        ("OKX",       _fetch_price_okx),
    ]:
        try:
            return await fn(symbol)
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"[price] {name} failed for {symbol}: {e}")
    raise HTTPException(
        status_code=503,
        detail=f"All price sources failed for {symbol} — {'; '.join(errors)}"
    )


# ── Market data cache ─────────────────────────────────────────────────────
# All users on the same symbol+tf share ONE OKX fetch per TTL period.
# 1000 BTCUSDT/15m users → 1 API call per minute instead of 1000.
_KLINES_CACHE: Dict[tuple, tuple] = {}   # (symbol, tf) → (fetched_at, klines)
_KLINES_CACHE_LOCK = threading.Lock()
_KLINES_CACHE_TTL = 55  # seconds — refresh just under 1 minute


def _fetch_klines_raw(symbol: str, tf: str, limit: int) -> List[Dict[str, Any]]:
    """Direct OKX fetch — no cache."""
    inst = to_okx_inst(symbol)
    bar = OKX_TF_MAP.get(str(tf))
    if not bar:
        print(f"[klines] unknown tf={tf!r} for {symbol}")
        return []
    try:
        r = httpx.get(
            f"{OKX_BASE}/api/v5/market/candles",
            params={"instId": inst, "bar": bar, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code == 429:
            print(f"[klines] OKX 429 rate-limit for {inst}/{bar} — sleeping 60s")
            time.sleep(60)
            return []
        if r.status_code != 200:
            print(f"[klines] OKX HTTP {r.status_code} for {inst}/{bar}: {r.text[:120]}")
            return []
        rows = (r.json() or {}).get("data") or []
        if not rows:
            print(f"[klines] OKX returned empty data for {inst}/{bar} limit={limit}")
            return []
        out: List[Dict[str, Any]] = []
        for k in rows:
            out.append({"t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])})
        out.reverse()
        return out
    except Exception as e:
        print(f"[klines] exception fetching {inst}/{bar}: {e}")
        return []


def _fetch_klines_sync(symbol: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Cached OKX fetch — all users sharing same symbol+tf get the same data."""
    key = (symbol.upper(), tf)
    now = time.time()
    with _KLINES_CACHE_LOCK:
        if key in _KLINES_CACHE:
            fetched_at, cached = _KLINES_CACHE[key]
            # Only use cache if it's fresh AND has enough candles for this request
            if now - fetched_at < _KLINES_CACHE_TTL and len(cached) >= limit:
                return cached
    # Cache miss, expired, or insufficient length — fetch fresh
    fresh = _fetch_klines_raw(symbol, tf, limit)
    if fresh:
        with _KLINES_CACHE_LOCK:
            _KLINES_CACHE[key] = (now, fresh)
    return fresh


# =========================
# MARKET (PUBLIC) — BYBIT
# =========================
BYBIT_BASE = "https://api.bybit.com"
BYBIT_TF_MAP = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}


def to_bybit_symbol(symbol: str) -> str:
    """Bybit uses BTCUSDT format (no dash)."""
    s = (symbol or "").upper().strip()
    if "-" in s:
        return s.replace("-", "")
    return s


async def bybit_price(symbol: str) -> float:
    sym = to_bybit_symbol(symbol)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{BYBIT_BASE}/v5/market/tickers",
            params={"category": "linear", "symbol": sym},
            headers={"accept": "application/json"},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Bybit price error: {r.text[:200]}")
    data = (r.json() or {})
    items = (data.get("result") or {}).get("list") or []
    if not items:
        raise HTTPException(status_code=400, detail=f"Bybit: no ticker for {sym}")
    last = items[0].get("lastPrice")
    if last is None:
        raise HTTPException(status_code=400, detail=f"Bybit: missing lastPrice for {sym}")
    return float(last)


def _fetch_klines_bybit_sync(symbol: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    sym = to_bybit_symbol(symbol)
    interval = BYBIT_TF_MAP.get(str(tf))
    if not interval:
        return []
    try:
        r = httpx.get(
            f"{BYBIT_BASE}/v5/market/kline",
            params={"category": "linear", "symbol": sym, "interval": interval, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code != 200:
            print(f"[klines-bybit] HTTP {r.status_code} for {sym}/{interval}: {r.text[:120]}")
            return []
        items = ((r.json() or {}).get("result") or {}).get("list") or []
        if not items:
            return []
        out: List[Dict[str, Any]] = []
        # Bybit returns: [startTime, open, high, low, close, volume, turnover]
        for k in items:
            out.append({
                "t": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        out.reverse()
        return out
    except Exception:
        return []


# alias used by trade engine (OKX has no geo-restriction on Render)
binance_price = okx_price


async def _route_price(symbol: str, exchange: str) -> float:
    # OKX is the only exchange whose public API is accessible from Render US servers.
    # Binance and Bybit both geo-block US infrastructure.
    # We use OKX as the data source for all exchanges — prices are essentially identical.
    return await okx_price(symbol)


def _route_klines(symbol: str, tf: str, exchange: str, limit: int) -> List[Dict[str, Any]]:
    return _fetch_klines_sync(symbol, tf, limit=limit)


_EXCHANGE_ALLOWLIST = {"bybit", "okx", "binance"}

def _validate_exchange_param(ex: Optional[str]) -> str:
    """Normalise and allowlist-check the ?exchange= query param."""
    v = (ex or "okx").lower().strip()
    if v not in _EXCHANGE_ALLOWLIST:
        raise HTTPException(status_code=400, detail=f"Unknown exchange '{v}'. Use: bybit, okx, binance")
    return v

import re as _re
_SYMBOL_RE = _re.compile(r'^[A-Z0-9]{2,20}$')


def _validate_symbol(v: str) -> str:
    """Normalise a symbol from a Pydantic model field — uppercase + strip only.
    Endpoint-level checks (USDT suffix, length) happen after this."""
    return v.upper().strip()


def _validate_path_symbol(symbol: str) -> str:
    """Validate and normalise a symbol coming in as a URL path segment."""
    sym = symbol.upper().strip()
    if not _SYMBOL_RE.match(sym):
        raise HTTPException(status_code=400, detail=f"Invalid symbol '{sym}' — use format like BTCUSDT")
    return sym


@app.get("/price/{symbol}")
async def get_price(symbol: str, exchange: Optional[str] = Query(default=None)):
    sym = _validate_path_symbol(symbol)
    ex  = _validate_exchange_param(exchange)
    price = await _route_price(sym, ex)
    return {"symbol": sym, "price": price, "exchange": ex}


def _parse_binance_style_24h(d: dict) -> dict:
    """Parse Binance-format 24h ticker (used by Binance US and MEXC)."""
    last   = float(d.get("lastPrice") or 0)
    open24 = float(d.get("openPrice") or last or 1)
    high   = float(d.get("highPrice") or 0)
    low    = float(d.get("lowPrice") or 0)
    vol    = float(d.get("quoteVolume") or 0)
    chg    = ((last - open24) / open24 * 100) if open24 else 0.0
    return {"price": last, "open24h": open24, "high24h": high, "low24h": low,
            "change24h": round(chg, 3), "volume24h": round(vol, 2)}


async def _fetch_ticker_24h(symbol: str) -> dict:
    """24h stats: Binance US → MEXC → OKX."""
    sym = symbol.upper().strip()
    for name, url in [
        ("BinanceUS", "https://api.binance.us/api/v3/ticker/24hr"),
        ("MEXC",      "https://api.mexc.com/api/v3/ticker/24hr"),
    ]:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(url, params={"symbol": sym},
                                     headers={"accept": "application/json"})
            if r.status_code == 200:
                return _parse_binance_style_24h(r.json())
        except Exception as e:
            print(f"[ticker24h] {name} failed for {sym}: {e}")
    # OKX fallback
    try:
        inst = to_okx_inst(sym)
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                f"{OKX_BASE}/api/v5/market/ticker",
                params={"instId": inst},
                headers={"accept": "application/json"},
            )
        if r.status_code == 200:
            d = ((r.json() or {}).get("data") or [{}])[0]
            last   = float(d.get("last") or 0)
            open24 = float(d.get("open24h") or last or 1)
            high   = float(d.get("high24h") or 0)
            low    = float(d.get("low24h") or 0)
            vol    = float(d.get("volCcy24h") or 0)
            chg    = ((last - open24) / open24 * 100) if open24 else 0.0
            return {"price": last, "open24h": open24, "high24h": high, "low24h": low,
                    "change24h": round(chg, 3), "volume24h": round(vol, 2)}
    except Exception as e:
        print(f"[ticker24h] OKX failed for {sym}: {e}")
    return {"price": 0, "open24h": 0, "high24h": 0, "low24h": 0, "change24h": 0.0, "volume24h": 0}


@app.get("/market/ticker/{symbol}")
async def market_ticker(symbol: str, exchange: Optional[str] = Query(default=None)):
    """24h stats for a symbol: price, change%, high, low, volume in USDT."""
    sym = _validate_path_symbol(symbol)
    stats = await _fetch_ticker_24h(sym)
    return {"symbol": sym, **stats, "exchange": (exchange or "okx").lower()}


@app.get("/market/tickers")
async def market_tickers(symbols: str = Query(..., max_length=500)):
    """Batch 24h stats. Pass ?symbols=BTCUSDT,ETHUSDT,... (max 20 symbols)"""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
    results = await asyncio.gather(*[_fetch_ticker_24h(s) for s in syms], return_exceptions=True)
    out = []
    for sym, res in zip(syms, results):
        if isinstance(res, dict):
            out.append({"symbol": sym, **res})
        else:
            out.append({"symbol": sym, "price": 0, "change24h": 0, "high24h": 0, "low24h": 0, "volume24h": 0})
    return {"tickers": out}


@app.get("/market/orderbook/{symbol}")
async def market_orderbook(symbol: str, depth: int = Query(default=15, ge=5, le=50)):
    """Live order book bids/asks."""
    inst = to_okx_inst(symbol.upper().strip())
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{OKX_BASE}/api/v5/market/books",
            params={"instId": inst, "sz": depth},
            headers={"accept": "application/json"},
        )
    d = ((r.json() or {}).get("data") or [{}])[0]
    def parse_side(rows):
        return [{"price": float(row[0]), "size": float(row[1])} for row in (rows or [])]
    return {
        "symbol": symbol.upper().strip(),
        "asks": parse_side(d.get("asks", [])),   # ascending (lowest ask first)
        "bids": parse_side(d.get("bids", [])),   # descending (highest bid first)
    }


@app.get("/market/trades/{symbol}")
async def market_trades(symbol: str, limit: int = Query(default=20, ge=5, le=50)):
    """Recent trades."""
    inst = to_okx_inst(symbol.upper().strip())
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{OKX_BASE}/api/v5/market/trades",
            params={"instId": inst, "limit": limit},
            headers={"accept": "application/json"},
        )
    rows = (r.json() or {}).get("data") or []
    trades = [
        {"price": float(t.get("px", 0)), "size": float(t.get("sz", 0)),
         "side": t.get("side", "").upper(), "ts": int(t.get("ts", 0))}
        for t in rows
    ]
    return {"symbol": symbol.upper().strip(), "trades": trades}


# ── Symbol list cache ─────────────────────────────────────────────────────────
# Stores the full USDT perpetual coin list per exchange so the market search
# dropdown shows the correct coins for the user's connected exchange.
# Refreshed every 5 minutes — symbol lists rarely change more often than that.
_symbols_cache: Dict[str, Any] = {}  # key = exchange name, value = {"ts": float, "symbols": list}
_SYMBOLS_TTL = 300  # 5 minutes in seconds

async def _fetch_symbols_for_exchange(exchange: str) -> List[str]:
    """
    Return all active USDT perpetual trading pairs for the given exchange.

    Why OKX is the fallback:
        Render deploys on US servers. Binance and Bybit geo-block US IPs on
        some endpoints. OKX does not. So we always try the native exchange API
        first, and silently fall back to OKX's list if it fails.
        Coins are ~95% the same across all three exchanges.

    Returns a sorted list of symbols in plain format e.g. ["BTCUSDT", "ETHUSDT"]
    """
    ex = exchange.lower()
    now = time.time()

    # Return cached list if it's still fresh (within 5 minutes)
    cached = _symbols_cache.get(ex)
    if cached and (now - cached["ts"]) < _SYMBOLS_TTL:
        return cached["symbols"]

    symbols: List[str] = []

    async with httpx.AsyncClient(timeout=12) as client:

        # ── Binance Futures ────────────────────────────────────────────────────
        # exchangeInfo returns every pair with its type and status.
        # We filter to: USDT-quoted + PERPETUAL contract + currently TRADING.
        if ex == "binance":
            try:
                r = await client.get(
                    "https://fapi.binance.com/fapi/v1/exchangeInfo",
                    headers={"accept": "application/json"},
                )
                if r.status_code == 200:
                    data = r.json()
                    symbols = sorted([
                        s["symbol"] for s in (data.get("symbols") or [])
                        if s.get("quoteAsset") == "USDT"        # USDT pairs only
                        and s.get("status") == "TRADING"        # actively trading
                        and s.get("contractType") == "PERPETUAL" # perps only, not delivery
                    ])
            except Exception:
                pass  # geo-block or timeout — fall through to OKX fallback below

        # ── Bybit Linear Perpetuals ────────────────────────────────────────────
        # category=linear covers USDT-margined perps (not inverse BTC-margined).
        elif ex == "bybit":
            try:
                r = await client.get(
                    f"{BYBIT_BASE}/v5/market/instruments-info",
                    params={"category": "linear", "limit": 1000},
                    headers={"accept": "application/json"},
                )
                if r.status_code == 200:
                    items = ((r.json() or {}).get("result") or {}).get("list") or []
                    symbols = sorted([
                        i["symbol"] for i in items
                        if i.get("quoteCoin") == "USDT"         # USDT pairs only
                        and i.get("status") == "Trading"        # actively trading
                        and i.get("symbol", "").endswith("USDT") # double-check suffix
                    ])
            except Exception:
                pass  # fall through to OKX fallback below

        # ── OKX fallback (or primary if exchange == "okx") ────────────────────
        # instType=SWAP covers perpetual contracts on OKX.
        # OKX uses "BTC-USDT-SWAP" format — we convert to plain "BTCUSDT" so all
        # three exchanges return symbols in the same format to the frontend.
        if not symbols:
            try:
                r = await client.get(
                    f"{OKX_BASE}/api/v5/public/instruments",
                    params={"instType": "SWAP"},
                    headers={"accept": "application/json"},
                )
                if r.status_code == 200:
                    items = (r.json() or {}).get("data") or []
                    symbols = sorted([
                        i["instId"].replace("-USDT-SWAP", "USDT")  # "BTC-USDT-SWAP" → "BTCUSDT"
                        for i in items
                        if i.get("instId", "").endswith("-USDT-SWAP")
                        and i.get("state") == "live"
                    ])
            except Exception:
                pass

    # Store the result in cache with the current timestamp
    _symbols_cache[ex] = {"ts": now, "symbols": symbols}
    return symbols


@app.get("/market/symbols")
async def market_symbols(exchange: str = Query(default="okx")):
    """
    Return all tradeable USDT perpetual symbols for the given exchange.

    The frontend market search calls this endpoint on page load to populate
    the coin search dropdown. Without this, the search would only show a
    hardcoded list instead of the actual coins available on the user's exchange.

    Usage:
        GET /market/symbols?exchange=binance   → Binance USDT perp list
        GET /market/symbols?exchange=bybit     → Bybit USDT linear list
        GET /market/symbols?exchange=okx       → OKX USDT swap list (default)

    Response:
        { "exchange": "binance", "count": 320, "symbols": ["BTCUSDT", ...] }
    """
    syms = await _fetch_symbols_for_exchange(exchange.lower())
    return {
        "exchange": exchange.lower(),
        "count": len(syms),
        "symbols": syms,
    }


@app.get("/klines/{symbol}")
def klines(
    symbol: str,
    tf: TF = Query("1h"),
    limit: int = Query(200, ge=20, le=1000),
    exchange: Optional[str] = Query(default=None),
):
    rows = _route_klines(symbol, tf, exchange or "okx", limit)
    if not rows:
        raise HTTPException(status_code=400, detail="No kline data (bad symbol/tf or exchange unavailable).")
    return {"symbol": symbol.upper().strip(), "tf": tf, "klines": rows, "exchange": (exchange or "okx").lower()}


# =========================
# TRADING + RISK ENGINE (sandbox)
# =========================
@dataclass
class Trade:
    time: str
    side: Side
    symbol: str
    mode: RiskMode
    size: float
    sl: float
    tp: float
    leverage: float
    entry_price: float
    current_price: float
    unreal_pnl_percent: float
    unreal_pnl_value: float
    equity_after: float
    reason: Optional[str] = None
    is_primary: int = 1


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def presets_for_mode(mode: RiskMode) -> Dict[str, float]:
    return {
        "ULTRA_SAFE": dict(size=0.30, sl=0.35, tp=0.60, leverage=2),
        "SAFE": dict(size=0.45, sl=0.45, tp=0.80, leverage=3),
        "NORMAL": dict(size=0.60, sl=0.55, tp=1.00, leverage=5),
        "MINI_ASYM": dict(size=0.65, sl=0.55, tp=1.10, leverage=6),
        "AGGRESSIVE": dict(size=0.85, sl=0.70, tp=1.50, leverage=8),
    }[mode]


def default_max_trades_per_day(mode: RiskMode) -> int:
    return {"ULTRA_SAFE": 1, "SAFE": 2, "NORMAL": 3, "MINI_ASYM": 3, "AGGRESSIVE": 5}[mode]


def build_reason(mode: RiskMode, equity: float, computed: Dict[str, float], extra: Optional[str] = None) -> str:
    from_start = equity - START_EQUITY
    dir_word = "up" if from_start >= 0 else "down"
    pct = abs(from_start / START_EQUITY * 100)
    size_dollar = equity * computed["size"]
    effective = size_dollar * computed["leverage"]
    reduced = pct >= 10.0

    lines = [
        f"{mode}  •  Manual trade",
        "",
        "Account",
        f"  Equity:      ${equity:,.2f}  ({dir_word} {pct:.1f}% from ${START_EQUITY:.0f} start)",
        "",
        "Position",
        f"  Size:        {computed['size'] * 100:.1f}% of equity  →  ${size_dollar:.2f} at risk",
        f"  Leverage:    {computed['leverage']:.0f}×  →  ${effective:.2f} total exposure",
        f"  Stop loss:   {computed['sl']:.2f}% from entry",
        f"  Take profit: {computed['tp']:.2f}% from entry",
    ]
    if reduced:
        lines.append("")
        lines.append("Risk note: Equity up 10%+ — size & leverage reduced ~10% to protect profits.")

    if extra:
        lines.append("")
        lines.append(extra)

    return "\n".join(lines)


def mini_asym_risk_engine(mode: RiskMode, equity: float) -> Dict[str, Any]:
    p = presets_for_mode(mode).copy()
    growth = (equity - START_EQUITY) / START_EQUITY
    if growth >= 0.10:
        p["size"] *= 0.9
        p["leverage"] *= 0.9
    computed = {
        "size": clamp(p["size"], 0.10, 1.50),
        "sl": clamp(p["sl"], 0.10, 2.50),
        "tp": clamp(p["tp"], 0.20, 5.00),
        "leverage": clamp(p["leverage"], 1.0, 10.0),
    }
    return {"allowed": True, "computed": computed, "reason": build_reason(mode, equity, computed)}


class TradeIn(BaseModel):
    symbol: str
    side: Side
    mode: RiskMode
    size: float = Field(..., gt=0, le=1_000_000)
    sl: float = Field(..., gt=0)
    tp: float = Field(..., gt=0)
    leverage: float = Field(..., ge=1, le=125)

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


class RiskPreviewIn(BaseModel):
    symbol: str
    side: Side
    mode: RiskMode
    size: float = Field(..., gt=0, le=1_000_000)
    sl: float = Field(..., gt=0)
    tp: float = Field(..., gt=0)
    leverage: float = Field(..., ge=1, le=125)

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


@app.post("/risk/preview")
def risk_preview(payload: RiskPreviewIn, user=Depends(require_user)):
    email = user["email"]
    equity = get_equity(email)
    out = mini_asym_risk_engine(payload.mode, equity)
    c = out["computed"]
    return {
        "allowed": True,
        "reason": out.get("reason"),
        "computed": {
            "symbol": payload.symbol.upper().strip(),
            "side": payload.side,
            "mode": payload.mode,
            "size": float(c["size"]),
            "sl": float(c["sl"]),
            "tp": float(c["tp"]),
            "leverage": float(c["leverage"]),
        },
    }


@app.get("/balance")
def balance(user=Depends(require_user)):
    email = user["email"]

    real_bal = get_real_usdt_balance(email)
    if real_bal is not None:
        # Exchange connected — always show real balance on equity card
        equity = real_bal
        db_equity = get_equity(email)
        if abs(db_equity - real_bal) > 0.01:
            set_equity(email, real_bal)
    elif REAL_TRADING:
        # Real trading mode but exchange unreachable — show 0, not fake balance
        equity = 0.0
    else:
        equity = get_equity(email)

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT time FROM trades WHERE email = %s ORDER BY id DESC LIMIT 1", (email,))
        last_row = cur.fetchone()
        cur.execute(
            "SELECT peak_equity, all_time_high, floor_equity, starting_capital FROM user_state WHERE email = %s",
            (email,),
        )
        state_row = cur.fetchone()
        # If peak_equity was never written (column newly added, runner not yet started),
        # recover the best historical peak from trade equity_after values.
        # This prevents "floor = equity × 0.85" on fresh deploys before runner starts.
        _db_peak = float(state_row["peak_equity"] or 0) if state_row else 0.0
        if _db_peak == 0:
            cur.execute(
                "SELECT MAX(equity_after) AS max_eq FROM trades WHERE email = %s",
                (email,),
            )
            _tp = cur.fetchone()
            _trade_peak = float((_tp or {}).get("max_eq") or 0)
            if _trade_peak > 0:
                _db_peak = _trade_peak
                # Persist so next call is instant
                _new_floor = round(_db_peak * 0.85, 2)
                cur.execute(
                    "UPDATE user_state SET peak_equity=%s, floor_equity=%s WHERE email=%s",
                    (_db_peak, _new_floor, email),
                )
                conn.commit()

    # start_eq is the balance when trading began — used for locked_profit and floor % display.
    # Using current equity as start_eq was wrong: it made locked_profit always 0 or negative.
    _sc = float(state_row["starting_capital"] or 0) if state_row else 0.0
    if real_bal is not None:
        start_eq = _sc if _sc > 0 else equity
    elif REAL_TRADING:
        start_eq = 0.0
    else:
        start_eq = START_EQUITY
    peak = _db_peak
    # floor_equity is the persistent, historically-max floor stored in user_state.
    # It only ever increases (via max in update_peak_ath) and survives runner restarts.
    stored_floor = float(state_row["floor_equity"] or 0) if state_row else 0.0
    # Include live runner's peak and floor — more up-to-date than DB between saves
    with AUTO_LOCK:
        _r = AUTO_RUNNERS.get(email)
    if _r and _r.is_running():
        peak = max(peak, _r.peak_equity)
        stored_floor = max(stored_floor, _r.floor_equity)
    if peak < equity:
        peak = equity
    # Hard floor = max of: (current peak × 0.85) and the historically stored floor.
    # Using max() means the floor can NEVER go lower than its best historical value,
    # even if peak_equity had a transient undercount or was reset by a bug.
    hard_floor = round(max(peak * 0.85, stored_floor), 2)
    locked_profit = round(hard_floor - start_eq, 2)
    distance_to_floor = round(equity - hard_floor, 2)
    distance_pct = round(distance_to_floor / hard_floor * 100, 2) if hard_floor > 0 else 0.0
    ath = float(state_row["all_time_high"] or 0) if state_row else 0.0
    if ath < equity:
        ath = equity
    return {
        "total": equity,
        "equity_after_last_trade": equity,
        "last_trade": last_row["time"] if last_row else None,
        "start_equity": start_eq,
        "peak_equity": round(peak, 2),
        "hard_floor": hard_floor,
        "locked_profit": locked_profit,
        "distance_to_floor": distance_to_floor,
        "distance_pct": distance_pct,
        "all_time_high": round(ath, 2),
    }


@app.get("/trades")
def trades(
    user=Depends(require_user),
    symbol: Optional[str] = Query(default=None),
    current_session: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=2000),
):
    email = user["email"]
    where = ["email = %s"]
    params: List[Any] = [email]

    if current_session:
        sid = get_session_id(email)
        where.append("session_id = %s")
        params.append(sid)

    if symbol:
        where.append("UPPER(symbol) = %s")
        params.append(symbol.strip().upper())

    q = f"SELECT * FROM trades WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT %s"
    params.append(int(limit))

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(q, tuple(params))
        rows = cur.fetchall()

    return {"trades": [dict(r) for r in rows]}


@app.get("/portfolio/stats")
def portfolio_stats(user=Depends(require_user)):
    import statistics as _st
    from datetime import timezone as _tz, timedelta as _td

    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM trades WHERE email = %s ORDER BY time ASC", (email,))
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return {"empty": True}

    # ── helpers ──────────────────────────────────────────────────────────────
    def _parse_dt(s):
        s = str(s).strip()
        iso = s if "T" in s else s.replace(" ", "T")
        if not iso.endswith("Z"):
            iso += "Z"
        try:
            from datetime import datetime as _dt
            return _dt.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            return None

    def _dubai_hour(s):
        dt = _parse_dt(s)
        if not dt:
            return -1
        return (dt + _td(hours=4)).hour

    def _session(h):
        if 2 <= h < 6:   return "Dead"
        if 6 <= h < 12:  return "Asia"
        if 12 <= h < 16: return "London"
        if 16 <= h < 21: return "London+NY"
        if 21 <= h <= 23: return "NY"
        return "NY-close"

    def _style(reason):
        r = str(reason or "")
        if "Scalp" in r or "SCALP" in r: return "SCALP"
        if "Swing" in r or "SWING" in r: return "SWING"
        return "DAY_TRADE"

    def _grade(trade: dict):
        stored = trade.get("grade")
        if stored in ("A", "B"):
            return stored
        return "B" if "Grade B" in str(trade.get("reason") or "") else "A"

    # ── base arrays ───────────────────────────────────────────────────────────
    pnls     = [float(t["unreal_pnl_value"])   for t in rows]
    pnl_pcts = [float(t["unreal_pnl_percent"]) for t in rows]
    total    = len(rows)
    wins     = [t for t in rows if float(t["unreal_pnl_value"]) >= 0]
    losses   = [t for t in rows if float(t["unreal_pnl_value"]) <  0]
    win_rate = len(wins) / total if total else 0
    total_pnl = sum(pnls)

    # ── equity + drawdown curve ───────────────────────────────────────────────
    first_eq = float(rows[0]["equity_after"]) - pnls[0]
    eq_curve = [{"time": str(rows[0]["time"]), "equity": round(first_eq, 2)}]
    for t in rows:
        eq_curve.append({"time": str(t["time"]), "equity": round(float(t["equity_after"]), 2)})

    peak = eq_curve[0]["equity"]
    max_dd = 0.0
    dd_curve = [{"time": eq_curve[0]["time"], "dd": 0.0}]
    for pt in eq_curve[1:]:
        if pt["equity"] > peak:
            peak = pt["equity"]
        dd = (peak - pt["equity"]) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        dd_curve.append({"time": pt["time"], "dd": round(-dd * 100, 3)})

    # ── avg R/R ───────────────────────────────────────────────────────────────
    win_p  = [abs(p) for p in pnl_pcts if p >= 0]
    loss_p = [abs(p) for p in pnl_pcts if p <  0]
    avg_rr = ((_st.mean(win_p) if win_p else 0) / (_st.mean(loss_p) if loss_p else 1))

    # ── monthly P&L ───────────────────────────────────────────────────────────
    monthly: dict = {}
    for t in rows:
        dt = _parse_dt(t["time"])
        if not dt:
            continue
        key = dt.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"month": key, "pnl": 0.0, "trades": 0, "wins": 0}
        monthly[key]["pnl"]    += float(t["unreal_pnl_value"])
        monthly[key]["trades"] += 1
        if float(t["unreal_pnl_value"]) >= 0:
            monthly[key]["wins"] += 1
    monthly_list = sorted(monthly.values(), key=lambda x: x["month"])
    for m in monthly_list:
        m["pnl"] = round(m["pnl"], 2)
    best_month  = max(monthly_list, key=lambda x: x["pnl"])  if monthly_list else None
    worst_month = min(monthly_list, key=lambda x: x["pnl"]) if monthly_list else None

    # ── current streak ────────────────────────────────────────────────────────
    streak_type  = "W" if pnls[-1] >= 0 else "L"
    streak_count = 0
    for p in reversed(pnls):
        if (p >= 0) == (streak_type == "W"):
            streak_count += 1
        else:
            break

    # ── session distribution ──────────────────────────────────────────────────
    sessions: dict = {}
    for t in rows:
        h = _dubai_hour(t["time"])
        if h < 0:
            continue
        s = _session(h)
        if s not in sessions:
            sessions[s] = {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_history": []}
        sessions[s]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        sessions[s]["pnl"]  += pv
        sessions[s]["pnl_history"].append(round(pv, 2))
        if pv >= 0:
            sessions[s]["wins"] += 1
    for s in sessions.values():
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)

    # ── grade distribution ────────────────────────────────────────────────────
    grades: dict = {"A": 0, "B": 0}
    for t in rows:
        g = _grade(t)
        grades[g] = grades.get(g, 0) + 1

    # ── symbol breakdown ──────────────────────────────────────────────────────
    syms: dict = {}
    for t in rows:
        sym = t["symbol"]
        if sym not in syms:
            syms[sym] = {"symbol": sym, "trades": 0, "wins": 0, "pnl": 0.0}
        syms[sym]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        syms[sym]["pnl"] += pv
        if pv >= 0:
            syms[sym]["wins"] += 1
    sym_list = sorted(syms.values(), key=lambda x: -x["trades"])
    for s in sym_list:
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)

    # ── mode breakdown ────────────────────────────────────────────────────────
    modes: dict = {}
    for t in rows:
        m = t["mode"]
        if m not in modes:
            modes[m] = {"mode": m, "trades": 0, "wins": 0, "pnl": 0.0}
        modes[m]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        modes[m]["pnl"] += pv
        if pv >= 0:
            modes[m]["wins"] += 1
    mode_list = sorted(modes.values(), key=lambda x: -x["trades"])
    for m in mode_list:
        m["win_rate"] = round(m["wins"] / m["trades"] * 100, 1) if m["trades"] else 0
        m["pnl"] = round(m["pnl"], 2)

    # ── style breakdown ───────────────────────────────────────────────────────
    stls: dict = {}
    hold_map = {"SCALP": "~15m", "DAY_TRADE": "~1–2h", "SWING": "~4–12h"}
    for t in rows:
        st = _style(t.get("reason"))
        if st not in stls:
            stls[st] = {"style": st, "trades": 0, "wins": 0, "pnl": 0.0}
        stls[st]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        stls[st]["pnl"] += pv
        if pv >= 0:
            stls[st]["wins"] += 1
    style_list = sorted(stls.values(), key=lambda x: -x["trades"])
    for s in style_list:
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)
        s["avg_hold"] = hold_map.get(s["style"], "~1h")

    # ── time heatmap (Dubai hour) ─────────────────────────────────────────────
    heatmap: dict = {h: {"wins": 0, "losses": 0} for h in range(24)}
    for t in rows:
        h = _dubai_hour(t["time"])
        if h < 0:
            continue
        if float(t["unreal_pnl_value"]) >= 0:
            heatmap[h]["wins"]   += 1
        else:
            heatmap[h]["losses"] += 1

    # ── best / worst trade ────────────────────────────────────────────────────
    best_trade  = max(rows, key=lambda t: float(t["unreal_pnl_value"]))
    worst_trade = min(rows, key=lambda t: float(t["unreal_pnl_value"]))

    # ── consistency & health scores ───────────────────────────────────────────
    if len(pnls) > 1:
        mean_abs = _st.mean(abs(p) for p in pnls)
        std_pnl  = _st.stdev(pnls)
        cv       = std_pnl / mean_abs if mean_abs > 0 else 2.0
        consistency_score = max(0, min(100, int(100 - cv * 25)))
    else:
        consistency_score = 50

    wr_pts   = min(40, int(win_rate * 80))
    dd_pts   = max(0, int(30 * (1 - max_dd / 0.20)))
    con_pts  = int(consistency_score * 0.30)
    health_score = min(100, wr_pts + dd_pts + con_pts)

    # ── daily returns → Sharpe / Sortino / Calmar ────────────────────────────
    daily: dict = {}
    for t in rows:
        dt = _parse_dt(t["time"])
        if not dt:
            continue
        key = dt.strftime("%Y-%m-%d")
        daily[key] = daily.get(key, 0.0) + float(t["unreal_pnl_percent"])
    daily_rets = list(daily.values())

    sharpe = sortino = calmar = 0.0
    if len(daily_rets) >= 2:
        mean_dr  = _st.mean(daily_rets)
        std_dr   = _st.stdev(daily_rets)
        neg_rets = [r for r in daily_rets if r < 0]
        sor_std  = _st.stdev(neg_rets) if len(neg_rets) > 1 else std_dr
        if std_dr:
            sharpe  = round(mean_dr * (365 ** 0.5) / std_dr, 2)
        if sor_std:
            sortino = round(mean_dr * (365 ** 0.5) / sor_std, 2)
        eq_start = eq_curve[0]["equity"]
        eq_end   = eq_curve[-1]["equity"]
        days     = max(1, len(daily_rets))
        ann_ret  = ((eq_end / eq_start) ** (365 / days) - 1) * 100 if eq_start > 0 else 0
        if max_dd > 0.001:
            calmar = round(ann_ret / (max_dd * 100), 2)

    # ── mistake detection ─────────────────────────────────────────────────────
    mistakes = []
    eligible_sess = [(k, v) for k, v in sessions.items() if v["trades"] >= 2]
    if eligible_sess:
        worst_s = min(eligible_sess, key=lambda x: x[1]["win_rate"])
        if worst_s[1]["win_rate"] < 40:
            mistakes.append(
                f"Most losses during {worst_s[0]} session "
                f"({worst_s[1]['win_rate']:.0f}% win rate)"
            )
    eligible_modes = [m for m in mode_list if m["trades"] >= 2]
    if eligible_modes:
        worst_m = min(eligible_modes, key=lambda x: x["win_rate"])
        if worst_m["win_rate"] < win_rate * 100 * 0.8:
            mistakes.append(
                f"{worst_m['mode']} underperformed "
                f"({worst_m['win_rate']:.0f}% vs {win_rate*100:.0f}% avg)"
            )
    if not mistakes:
        mistakes.append("No significant loss pattern detected yet — keep trading!")

    # ── AI summary sentence ───────────────────────────────────────────────────
    streak_str = f"{streak_count} consecutive {'win' if streak_type=='W' else 'loss'}{'s' if streak_count!=1 else ''}"
    best_s_name = max(sessions.items(), key=lambda x: x[1]["win_rate"])[0] if sessions else None
    summary_parts = [
        f"{streak_str}",
        f"{win_rate*100:.0f}% win rate over {total} trades",
        f"{'Up' if total_pnl >= 0 else 'Down'} ${abs(total_pnl):.2f} all time",
    ]
    if best_s_name:
        summary_parts.append(f"Best during {best_s_name} session")
    ai_summary = " · ".join(summary_parts)

    return {
        "empty": False,
        "summary": {
            "total_pnl":        round(total_pnl, 2),
            "total_trades":     total,
            "win_rate":         round(win_rate * 100, 1),
            "avg_rr":           round(avg_rr, 2),
            "best_month":       best_month,
            "worst_month":      worst_month,
            "current_streak":   {"type": streak_type, "count": streak_count},
            "max_drawdown":     round(max_dd * 100, 2),
            "health_score":     health_score,
            "consistency_score": consistency_score,
        },
        "equity_curve":        eq_curve,
        "drawdown_curve":      dd_curve,
        "monthly_pnl":         monthly_list,
        "session_distribution": sessions,
        "grade_distribution":  grades,
        "symbol_breakdown":    sym_list[:10],
        "mode_breakdown":      mode_list,
        "style_breakdown":     style_list,
        "time_heatmap":        heatmap,
        "best_trade":          dict(best_trade),
        "worst_trade":         dict(worst_trade),
        "ai_summary":          ai_summary,
        "mistake_detection":   mistakes,
        "ratios":              {"sharpe": sharpe, "sortino": sortino, "calmar": calmar},
        "most_traded_symbol":  sym_list[0]["symbol"] if sym_list else "-",
    }


@app.get("/portfolio/pnl")
def portfolio_pnl(user=Depends(require_user)):
    """Today / week / month PnL cards + full monthly history for the PnL section."""
    from datetime import timezone as _tz, timedelta as _td, datetime as _dt

    email = user["email"]
    now_utc = _dt.now(_tz.utc)
    dubai_now = now_utc + _td(hours=4)

    # Dubai midnight boundaries
    today_dubai  = dubai_now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start   = today_dubai - _td(days=today_dubai.weekday())
    month_start  = today_dubai.replace(day=1)

    # Convert back to UTC ISO for DB comparison
    def _to_utc(dt_dubai):
        return (dt_dubai - _td(hours=4)).strftime("%Y-%m-%dT%H:%M:%S")

    today_utc = _to_utc(today_dubai)
    week_utc  = _to_utc(week_start)
    month_utc = _to_utc(month_start)

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT time, unreal_pnl_value, equity_after FROM trades "
            "WHERE email = %s ORDER BY time ASC",
            (email,),
        )
        rows = cur.fetchall()

    if not rows:
        return {"empty": True}

    def _agg(subset):
        if not subset:
            return {"pnl": 0.0, "pnl_pct": 0.0, "trades": 0, "wins": 0}
        pnls = [float(r["unreal_pnl_value"]) for r in subset]
        wins = sum(1 for p in pnls if p >= 0)
        # Use equity_before of first trade as base for % calc
        first_eq_after = float(subset[0]["equity_after"])
        first_pnl      = float(subset[0]["unreal_pnl_value"])
        base_eq        = first_eq_after - first_pnl
        total_pnl      = sum(pnls)
        pnl_pct        = round(total_pnl / base_eq * 100, 2) if base_eq > 0 else 0.0
        return {"pnl": round(total_pnl, 2), "pnl_pct": pnl_pct,
                "trades": len(subset), "wins": wins}

    today_rows = [r for r in rows if str(r["time"]) >= today_utc]
    week_rows  = [r for r in rows if str(r["time"]) >= week_utc]
    month_rows = [r for r in rows if str(r["time"]) >= month_utc]

    # ── monthly history ───────────────────────────────────────────────────────
    monthly: dict = {}
    for r in rows:
        ts  = str(r["time"])
        key = ts[:7]  # "YYYY-MM"
        if key not in monthly:
            monthly[key] = {"period": key, "pnl": 0.0, "trades": 0, "wins": 0,
                            "_first_pnl": float(r["unreal_pnl_value"]),
                            "_first_eq_after": float(r["equity_after"])}
        monthly[key]["pnl"]    += float(r["unreal_pnl_value"])
        monthly[key]["trades"] += 1
        if float(r["unreal_pnl_value"]) >= 0:
            monthly[key]["wins"] += 1

    history = []
    for m in sorted(monthly.values(), key=lambda x: x["period"]):
        base = m["_first_eq_after"] - m["_first_pnl"]
        pnl  = round(m["pnl"], 2)
        pct  = round(pnl / base * 100, 2) if base > 0 else 0.0
        history.append({
            "period": m["period"],
            "pnl":    pnl,
            "pnl_pct": pct,
            "trades": m["trades"],
            "wins":   m["wins"],
            "losses": m["trades"] - m["wins"],
        })

    best_month  = max(history, key=lambda x: x["pnl"])  if history else None
    worst_month = min(history, key=lambda x: x["pnl"]) if history else None

    # ── yearly grouping ───────────────────────────────────────────────────────
    years_dict: dict = {}
    for m in history:
        yr = m["period"][:4]
        if yr not in years_dict:
            years_dict[yr] = {"year": yr, "pnl": 0.0, "trades": 0, "wins": 0, "months": []}
        years_dict[yr]["pnl"]    += m["pnl"]
        years_dict[yr]["trades"] += m["trades"]
        years_dict[yr]["wins"]   += m["wins"]
        years_dict[yr]["months"].append(m)
    for y in years_dict.values():
        y["pnl"]    = round(y["pnl"], 2)
        y["losses"] = y["trades"] - y["wins"]

    yearly = sorted(years_dict.values(), key=lambda x: x["year"])

    return {
        "today":       _agg(today_rows),
        "week":        _agg(week_rows),
        "month":       _agg(month_rows),
        "history":     history,
        "yearly":      yearly,
        "best_month":  best_month,
        "worst_month": worst_month,
        "has_multiple_years": len(years_dict) > 1,
    }


@app.get("/portfolio/pnl/month/{year_month}")
def portfolio_pnl_month(year_month: str, user=Depends(require_user)):
    """Return individual trades for a specific month (YYYY-MM) — drill-down view."""
    if len(year_month) != 7 or year_month[4] != "-":
        raise HTTPException(status_code=400, detail="Format must be YYYY-MM")
    email = user["email"]
    yr, mo = int(year_month[:4]), int(year_month[5:])
    next_yr, next_mo = (yr + 1, 1) if mo == 12 else (yr, mo + 1)
    end_str = f"{next_yr}-{next_mo:02d}-01"
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT time, side, symbol, mode, unreal_pnl_value, unreal_pnl_percent, "
            "entry_price, current_price, equity_after, leverage, reason "
            "FROM trades WHERE email = %s AND time >= %s AND time < %s "
            "ORDER BY time ASC",
            (email, year_month + "-01", end_str),
        )
        trades = [dict(r) for r in cur.fetchall()]
    return {"month": year_month, "trades": trades, "count": len(trades)}


@app.get("/portfolio/pnl/csv/{year}")
def portfolio_pnl_csv(year: str, user=Depends(require_user)):
    """Download all trades for a given year as CSV (tax export)."""
    from fastapi.responses import StreamingResponse
    import io, csv as _csv

    if not year.isdigit() or len(year) != 4:
        raise HTTPException(status_code=400, detail="Year must be 4 digits")

    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT time, side, symbol, mode, size, leverage, "
            "entry_price, current_price, unreal_pnl_value, unreal_pnl_percent, "
            "equity_after, reason "
            "FROM trades WHERE email = %s AND time >= %s AND time < %s "
            "ORDER BY time ASC",
            (email, f"{year}-01-01", f"{int(year)+1}-01-01"),
        )
        rows = cur.fetchall()

    buf = io.StringIO()
    w   = _csv.writer(buf)
    w.writerow(["Date (UTC)", "Side", "Symbol", "Mode", "Size %", "Leverage",
                "Entry Price", "Exit Price", "PnL ($)", "PnL (%)", "Equity After"])
    for r in rows:
        w.writerow([
            str(r["time"])[:19],
            r["side"], r["symbol"], r["mode"],
            r["size"], r["leverage"],
            r["entry_price"], r["current_price"],
            round(float(r["unreal_pnl_value"]), 4),
            round(float(r["unreal_pnl_percent"]), 4),
            round(float(r["equity_after"]), 4),
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{year}.csv"},
    )


async def _place_trade_internal(
    email: str,
    symbol: str,
    side: Side,
    mode: RiskMode,
    extra_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if not get_exchange(email):
        raise HTTPException(status_code=400, detail="Connect exchange first (Exchange tab).")

    equity_before = get_equity(email)
    sid = get_session_id(email)

    out = mini_asym_risk_engine(mode, equity_before)
    c = out["computed"]
    reason_text = build_reason(mode, equity_before, c, extra=extra_reason)

    entry = await binance_price(symbol)

    move = ((int(time.time()) % 140) - 70) / 1000.0
    pnl_pct = move * (c["leverage"] / 5.0)
    pnl_value = equity_before * c["size"] * pnl_pct
    equity_after = equity_before + pnl_value

    tr = Trade(
        time=now_utc_str(),
        side=side,
        symbol=symbol.upper(),
        mode=mode,
        size=float(c["size"]),
        sl=float(c["sl"]),
        tp=float(c["tp"]),
        leverage=float(c["leverage"]),
        entry_price=float(entry),
        current_price=float(entry * (1.0 + pnl_pct)),
        unreal_pnl_percent=float(pnl_pct * 100.0),
        unreal_pnl_value=float(pnl_value),
        equity_after=float(equity_after),
        reason=reason_text,
    )

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trades(
                email, time, side, symbol, mode, size, sl, tp, leverage,
                entry_price, current_price, unreal_pnl_percent, unreal_pnl_value, equity_after,
                reason, session_id
            ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                email, tr.time, tr.side, tr.symbol, tr.mode,
                float(tr.size), float(tr.sl), float(tr.tp), float(tr.leverage),
                float(tr.entry_price), float(tr.current_price),
                float(tr.unreal_pnl_percent), float(tr.unreal_pnl_value),
                float(tr.equity_after), tr.reason, int(sid),
            ),
        )
        conn.commit()

    set_equity(email, equity_after)
    # Trail peak and floor — ensures manual trades raise the floor the same way AI trades do.
    # Missing this call was a bug: a profitable manual trade would raise equity but not peak_equity,
    # so the floor shown in balance stayed at the old (lower) value.
    update_peak_ath(email, max(equity_before, equity_after), equity_after)
    return {"ok": True, "trade": asdict(tr), "pnl_pct": float(tr.unreal_pnl_percent)}


# =========================
# AUTO AI TRADER (V3)
# =========================
AUTO_LOCK = threading.Lock()
AUTO_RUNNERS: Dict[str, "AutoRunner"] = {}

# ── Engine start queue (Task 11) ──────────────────────────────────────────────
# Prevents server overload when many users start engines simultaneously.
# Max 10 starts per 60s rolling window, minimum 6s gap between any two starts.
_ENGINE_START_LOCK  = threading.Lock()
_engine_start_times: List[float] = []   # timestamps of accepted starts in last 60s
_last_engine_start_ts: float = 0.0      # timestamp of most recent accepted start
_MAX_STARTS_PER_MIN = 10
_ENGINE_START_GAP   = 6                 # seconds


# ── Runner state persistence (survive deploys) ────────────────────────────────

def _save_runner_state(runner: "AutoRunner") -> None:
    """Persist runner config AND live state to DB so it survives redeploys.
    started_at and session_start_equity are set only on INSERT — they are never
    overwritten on UPDATE so the original session start values survive redeploys."""
    import json as _json
    try:
        pending_json = _json.dumps(list(runner.pending_trades))
        with db_conn() as conn:
            cur = conn.cursor()
            if USING_PG:
                cur.execute("""
                    INSERT INTO ai_runner_state
                        (email, symbol, trade_style, mode, max_trades_per_day,
                         stop_after_bad_trades, duration_days, trend_filter,
                         chop_min_sep_pct, end_at_ts, started_at,
                         adaptive_strictness, last_trade_ts, last_trade_bad,
                         pending_trades_json, peak_equity, floor_equity,
                         consecutive_wins, session_start_equity, strictness_day_key)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(email) DO UPDATE SET
                        symbol=%s, trade_style=%s, mode=%s,
                        max_trades_per_day=%s, stop_after_bad_trades=%s,
                        duration_days=%s, trend_filter=%s,
                        chop_min_sep_pct=%s, end_at_ts=%s,
                        adaptive_strictness=%s, last_trade_ts=%s, last_trade_bad=%s,
                        pending_trades_json=%s, peak_equity=%s, floor_equity=%s,
                        consecutive_wins=%s, strictness_day_key=%s
                """, (
                    # INSERT values (20)
                    runner.email, runner.symbol, runner.trade_style, runner.mode,
                    runner.max_trades_per_day, runner.stop_after_bad_trades,
                    runner.duration_days, int(runner.trend_filter),
                    runner.chop_min_sep_pct, runner.end_at_ts, int(time.time()),
                    runner.adaptive_strictness, runner.last_trade_ts,
                    int(runner._last_trade_bad), pending_json,
                    runner.peak_equity, runner.floor_equity, runner.consecutive_wins,
                    runner.session_start_equity, runner.strictness_day_key,
                    # UPDATE SET values (17 — started_at and session_start_equity kept from original INSERT)
                    runner.symbol, runner.trade_style, runner.mode,
                    runner.max_trades_per_day, runner.stop_after_bad_trades,
                    runner.duration_days, int(runner.trend_filter),
                    runner.chop_min_sep_pct, runner.end_at_ts,
                    runner.adaptive_strictness, runner.last_trade_ts,
                    int(runner._last_trade_bad), pending_json,
                    runner.peak_equity, runner.floor_equity, runner.consecutive_wins,
                    runner.strictness_day_key,
                ))
            else:
                # SQLite: preserve started_at and session_start_equity if row exists
                cur.execute(
                    "SELECT started_at, session_start_equity FROM ai_runner_state WHERE email = ?",
                    (runner.email,),
                )
                existing = cur.fetchone()
                orig_started_at = existing["started_at"] if existing else int(time.time())
                orig_start_eq = (
                    existing["session_start_equity"]
                    if existing and existing["session_start_equity"]
                    else runner.session_start_equity
                )
                cur.execute("""
                    INSERT OR REPLACE INTO ai_runner_state
                        (email, symbol, trade_style, mode, max_trades_per_day,
                         stop_after_bad_trades, duration_days, trend_filter,
                         chop_min_sep_pct, end_at_ts, started_at,
                         adaptive_strictness, last_trade_ts, last_trade_bad,
                         pending_trades_json, peak_equity, floor_equity,
                         consecutive_wins, session_start_equity, strictness_day_key)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    runner.email, runner.symbol, runner.trade_style, runner.mode,
                    runner.max_trades_per_day, runner.stop_after_bad_trades,
                    runner.duration_days, int(runner.trend_filter),
                    runner.chop_min_sep_pct, runner.end_at_ts, orig_started_at,
                    runner.adaptive_strictness, runner.last_trade_ts,
                    int(runner._last_trade_bad), pending_json,
                    runner.peak_equity, runner.floor_equity, runner.consecutive_wins,
                    orig_start_eq, runner.strictness_day_key,
                ))
            conn.commit()
    except Exception as e:
        print(f"[runner-state] save failed for {runner.email}: {e}")


def _clear_runner_state(email: str) -> None:
    """Remove persisted runner state — called when AI stops for any reason."""
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM ai_runner_state WHERE email = %s", (email,))
            conn.commit()
    except Exception as e:
        print(f"[runner-state] clear failed for {email}: {e}")


def _load_all_runner_states() -> list:
    """Return all saved runner configs (called once on startup)."""
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM ai_runner_state")
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[runner-state] load failed: {e}")
        return []


def get_ai_restart_lock(email: str) -> int:
    """Returns Unix timestamp until which AI restart is locked (0 = not locked)."""
    try:
        return int(admin_get_setting(f"ai_restart_lock:{email}", "0"))
    except Exception:
        return 0


def set_ai_restart_lock(email: str, until_ts: int) -> None:
    admin_set_setting(f"ai_restart_lock:{email}", str(until_ts))


@dataclass
class AutoState:
    running: bool
    email: str
    symbol: str
    tf: str
    interval_sec: int
    mode: RiskMode
    side: Side
    last_signal: str
    blocked_reason: Optional[str]
    last_run_at: str
    last_trade_at: str
    max_trades_per_day: int
    trades_today: int
    stop_after_bad_trades: int
    bad_trades_today: int
    reset_in_sec: int
    duration_days: int
    end_at: Optional[str]
    trend_filter: bool
    chop_min_sep_pct: float
    signal_breakdown: Dict
    adaptive_strictness: float
    pending_trades: List[Dict]
    signal_score: float
    trade_style: str
    market_grade: str
    market_regime: str


class AutoStartIn(BaseModel):
    symbol: str
    trade_style: TradeStyle = "DAY_TRADE"
    mode: RiskMode = "MINI_ASYM"
    max_trades_per_day: Optional[int] = Field(default=None, ge=1, le=50)
    stop_after_bad_trades: int = Field(default=2, ge=1, le=20)
    duration_days: int = Field(default=0, ge=0, le=365)
    trend_filter: bool = True
    chop_min_sep_pct: float = Field(default=0.005, ge=0.001, le=0.05)

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


class FloorResetIn(BaseModel):
    # confirm=False → dry-run (returns situation/status without applying any changes)
    # confirm=True  → apply the reset
    confirm: bool = False
    # Must be exactly "CONFIRM" (case-insensitive) for Situation 2 when confirm=True
    typed_confirm: str = ""


class CorrectTradeIn(BaseModel):
    # symbol that was traded (e.g. "NEAR/USDT:USDT" or "NEARUSDT")
    symbol: str = Field(..., min_length=3, max_length=30)
    # total real P&L from the exchange for all legs combined (positive = profit)
    real_pnl: float = Field(..., ge=-100_000, le=100_000)
    # how many recent trade records to correct — 2 for Grade B (T1+T2), 1 for Grade A
    num_trades: int = Field(default=2, ge=1, le=10)
    # correct outcome label written to reason text
    new_outcome: Literal["TP_HIT", "SL_HIT", "TRAIL_STOP", "NATURAL_CLOSE"] = "TP_HIT"


class _MidCandleSkip(Exception):
    """Raised inside _run_mid_candle_monitor to break out of nested blocks cleanly."""


class AutoRunner:
    def __init__(self, email, symbol, trade_style, mode,
                 max_trades_per_day, stop_after_bad_trades, duration_days,
                 trend_filter, chop_min_sep_pct):
        self.email = email
        self.symbol = symbol.upper().strip()
        self.trade_style: TradeStyle = trade_style
        sp = TRADE_STYLE_PARAMS.get(trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
        self.tf = sp["tf"]
        self.interval_sec = sp["interval"]
        self.mode = mode
        self.max_trades_per_day = int(max(0, max_trades_per_day))
        # Minimum of 1 — 0 would disable bad-trade protection entirely.
        # Users can raise the limit but cannot turn it off completely via parameter.
        self.stop_after_bad_trades = int(max(1, stop_after_bad_trades))
        self.duration_days = int(max(0, duration_days))
        self.trend_filter = bool(trend_filter)
        self.chop_min_sep_pct = float(max(0.0, chop_min_sep_pct))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        # Single background log-writer thread drains a queue — replaces one-thread-per-entry.
        import queue as _queue_mod
        self._log_queue: "_queue_mod.Queue[Optional[Dict]]" = _queue_mod.Queue()
        self._log_writer = threading.Thread(target=self._log_writer_loop, daemon=True)
        self._log_writer.start()
        self.last_signal: str = "-"
        self.last_side: Side = "LONG"
        # Allow first trade after 60s (enough to initialize) — not a full interval wait.
        # Previous: set to now() which blocked SWING for 4h after every restart.
        self.last_trade_ts: float = time.time() - self.interval_sec + 60
        self.last_run_ts: float = 0.0
        self.blocked_reason: Optional[str] = None
        self.day_key = dubai_day_key()
        self.end_at_ts: Optional[float] = None
        if self.duration_days > 0:
            end_dt = now_dubai() + timedelta(days=self.duration_days)
            self.end_at_ts = end_dt.timestamp()
        self.history: Deque[Dict[str, str]] = deque(maxlen=200)
        self.ai_session_id: Optional[int] = None  # DB row id for this AI run
        self.pending_trades: List[Dict] = []
        self.adaptive_strictness: float = 1.0
        self.strictness_day_key: str = dubai_day_key()  # tracks which Dubai day strictness belongs to
        self.last_breakdown: Dict = {}
        self.last_score: float = 0.0
        self.last_atr_pct: float = 0.01   # fallback 1% ATR until first signal
        self.market_grade: str = "-"
        self.signal_type: str = "NORMAL"  # "NORMAL" or "REVERSAL" (T16)
        self.sl_atr_override: Optional[float] = None   # T16 reversal uses 1.5
        self.tp_atr_override: Optional[float] = None   # T16 reversal uses 3.0
        self.market_regime: str = "-"     # Phase 4: TRENDING / MARGINAL / CHOPPY / VOLATILE
        self._last_trade_bad: bool = False
        self._last_holding_log_ts: float = 0.0   # throttle repeated holding logs
        self._last_drawdown_log_ts: float = 0.0  # throttle repeated drawdown tier logs
        self._last_dd_tier: int = 0              # 0=none 1=65% 2=40% 3=25% 4=stop
        self._last_mc_log_ts: float = 0.0        # throttle mid-candle noise logs
        # Momentum scaling (pyramiding) — tracks scale-in state per active trade
        # Allowed modes: MINI_ASYM, NORMAL, AGGRESSIVE only
        self._scaling_enabled: bool = mode in ("MINI_ASYM", "NORMAL", "AGGRESSIVE")
        self.session_start_equity: float = get_equity(email)
        # Peak must be the highest equity ever seen — read from user_state which survives
        # AI stop/restart (ai_runner_state is deleted on stop, user_state is permanent).
        # This prevents floor from drifting down when equity dips between sessions.
        _saved_peak = 0.0
        try:
            with db_conn() as conn:
                _pcur = conn.cursor()
                _pcur.execute("SELECT peak_equity FROM user_state WHERE email = %s", (email,))
                _prow = _pcur.fetchone()
                _saved_peak = float(_prow["peak_equity"] or 0) if _prow else 0.0
        except Exception:
            _saved_peak = 0.0
        self.peak_equity: float = max(get_equity(email), _saved_peak)
        self.consecutive_wins: int = 0    # reset strictness after 3 wins in a row
        # Hard floor = 85% of peak equity — trails UP with new highs, never decreases.
        self.floor_equity: float = self.peak_equity * 0.85
        # Persist the correct peak immediately so /balance endpoint shows the right floor
        # even before the first trade is recorded.
        update_peak_ath(email, self.peak_equity, self.peak_equity)
        # Load today's real trade counts from DB — prevents bypass by stop+restart
        self.trades_today, self.bad_trades_today = self._load_today_stats()
        # Restore live state (strictness, open positions, drawdown level) from last session
        self._restore_live_state()
        # Startup audit log — printed to the AI log every time the engine starts.
        # Shows the exact peak and floor values the bot is using so any floor bug
        # is immediately visible without digging through DB records.
        # Format: "STARTUP: DB peak=$X | current equity=$Y | final peak=$Z | floor=$W"
        #   DB peak       = what user_state.peak_equity had before this session
        #   current equity = live wallet balance right now
        #   final peak     = the value actually used (max of DB peak + ai_runner_state peak)
        #   floor          = 85% of final peak — bot will stop if equity drops below this
        try:
            _cur_eq = get_equity(email)
            self.log(
                f"STARTUP: DB peak=${_saved_peak:.2f} | current equity=${_cur_eq:.2f} | "
                f"final peak=${self.peak_equity:.2f} | floor=${self.floor_equity:.2f}"
            )
        except Exception:
            pass

    def _restore_live_state(self) -> None:
        """Load adaptive_strictness, pending_trades, peak/floor equity, and
        session_start_equity from DB. Called on init so a redeploy picks up
        exactly where the previous session left off."""
        import json as _json
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT adaptive_strictness, last_trade_ts, last_trade_bad, "
                    "pending_trades_json, peak_equity, floor_equity, consecutive_wins, "
                    "session_start_equity, strictness_day_key "
                    "FROM ai_runner_state WHERE email = %s" if USING_PG else
                    "SELECT adaptive_strictness, last_trade_ts, last_trade_bad, "
                    "pending_trades_json, peak_equity, floor_equity, consecutive_wins, "
                    "session_start_equity, strictness_day_key "
                    "FROM ai_runner_state WHERE email = ?",
                    (self.email,),
                )
                row = cur.fetchone()
            if not row:
                return
            row = dict(row)
            if row.get("adaptive_strictness"):
                self.adaptive_strictness = float(row["adaptive_strictness"])
                saved_sdk = row.get("strictness_day_key") or ""
                today_sdk = dubai_day_key()
                if saved_sdk and saved_sdk != today_sdk and self.adaptive_strictness > 1.0:
                    # Strictness was set on a previous Dubai day — apply midnight step-down now.
                    # This fires on ANY restart (redeploy, crash, manual stop) after a day boundary.
                    STEP_DOWN = {2.50: 1.50, 1.50: 1.25}
                    if self.adaptive_strictness <= 1.25:
                        self.adaptive_strictness = 1.0
                        self.consecutive_wins = 0
                        self.log(f"Startup [{today_sdk}]: strictness was from {saved_sdk} — cleared to 1.0x.")
                    else:
                        self.adaptive_strictness = STEP_DOWN.get(
                            round(self.adaptive_strictness, 2), 1.25
                        )
                        self.log(f"Startup [{today_sdk}]: strictness was from {saved_sdk} — stepped down to {self.adaptive_strictness:.2f}x.")
                self.strictness_day_key = today_sdk if (saved_sdk != today_sdk) else saved_sdk
            if row.get("last_trade_ts"):
                self.last_trade_ts = float(row["last_trade_ts"])
            if row.get("last_trade_bad"):
                self._last_trade_bad = bool(int(row["last_trade_bad"]))
            if row.get("consecutive_wins"):
                self.consecutive_wins = int(row["consecutive_wins"])
            # Restore peak/floor equity — always take the MAX so floor never drifts down.
            # ai_runner_state may have been saved with a lower value if equity had dipped.
            if row.get("peak_equity") and float(row["peak_equity"]) > 0:
                self.peak_equity = max(self.peak_equity, float(row["peak_equity"]))
                self.floor_equity = self.peak_equity * 0.85
            if row.get("floor_equity") and float(row["floor_equity"]) > 0:
                self.floor_equity = max(self.floor_equity, float(row["floor_equity"]))
            # ── Hard floor persistence fix ─────────────────────────────────────
            # Problem: __init__ calls update_peak_ath() BEFORE calling _restore_live_state().
            # That means the early write saves the pre-restore peak (possibly lower than the
            # true peak stored in ai_runner_state from the previous session).
            # When the engine stops, _clear_runner_state() deletes ai_runner_state entirely.
            # Next restart: user_state.peak_equity has the wrong (lower) peak → floor = 85%
            # of current equity instead of 85% of the real all-time peak.
            #
            # Fix: call update_peak_ath() HERE, after we have already elevated self.peak_equity
            # from ai_runner_state. This writes the correct peak to user_state so it survives
            # even after ai_runner_state is deleted on stop.
            update_peak_ath(self.email, self.peak_equity, self.peak_equity)
            # Restore original session start equity — keeps P&L% and risk calcs relative
            # to the real session start, not the redeploy time
            if row.get("session_start_equity") and float(row["session_start_equity"]) > 0:
                self.session_start_equity = float(row["session_start_equity"])
            # Restore open positions — bot continues managing them immediately
            raw = row.get("pending_trades_json") or "[]"
            trades = _json.loads(raw) if isinstance(raw, str) else []
            if trades:
                self.pending_trades = trades
                self.log(f"Restored {len(trades)} open position(s) from previous session.")
        except Exception as e:
            print(f"[runner-state] restore failed for {self.email}: {e}")
        # Restore log history and active session_id from DB so they survive redeploys
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                # Find the most recent open session (ended_at IS NULL)
                cur.execute(
                    "SELECT id FROM ai_sessions WHERE email=%s AND ended_at IS NULL "
                    "ORDER BY id DESC LIMIT 1",
                    (self.email,),
                )
                sess_row = cur.fetchone()
                if sess_row:
                    self.ai_session_id = int(sess_row["id"])
                    cur.execute(
                        "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                        "ORDER BY id DESC LIMIT 200",
                        (self.email, self.ai_session_id),
                    )
                else:
                    cur.execute(
                        "SELECT t, msg FROM ai_logs WHERE email=%s ORDER BY id DESC LIMIT 200",
                        (self.email,),
                    )
                rows = cur.fetchall()
            for row in rows:
                self.history.append({"t": row["t"], "msg": row["msg"]})
        except Exception as e:
            print(f"[runner-state] log restore failed for {self.email}: {e}")

        # FIX 3: Reconcile exchange positions on every startup/restart
        # Runs in a background thread so it never blocks the runner from starting.
        import threading as _thr
        _thr.Thread(
            target=self._reconcile_exchange_positions,
            daemon=True,
            name=f"reconcile-{self.email[:8]}",
        ).start()

    def is_running(self):
        return not self.stop_event.is_set()

    def _load_today_stats(self) -> tuple:
        """
        Count today's trades and bad trades from the DB (Dubai timezone).
        Prevents stop+restart from bypassing daily risk limits.
        """
        try:
            # Dubai midnight → UTC equivalent for DB query
            dubai_midnight = now_dubai().replace(hour=0, minute=0, second=0, microsecond=0)
            utc_midnight_str = dubai_midnight.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            # Use current session_id so Reset Sandbox clears the daily lock in demo mode
            sid = get_session_id(self.email)
            with db_conn() as conn:
                cur = conn.cursor()
                # Only count primary trades (T1 / Grade A) — T2 is_primary=0.
                # Older rows without is_primary default to 1 (column default), so they
                # are still counted — no data loss from the schema migration.
                cur.execute(
                    "SELECT COUNT(*) as cnt, "
                    "SUM(CASE WHEN unreal_pnl_percent < 0 THEN 1 ELSE 0 END) as bad "
                    "FROM trades WHERE email = %s AND time >= %s AND session_id = %s "
                    "AND is_primary = 1",
                    (self.email, utc_midnight_str, sid),
                )
                row = cur.fetchone()
            trades = int((row or {}).get("cnt") or 0)
            bad = int((row or {}).get("bad") or 0)
            return trades, bad
        except Exception:
            return 0, 0

    def log(self, msg):
        entry = {"t": now_dubai().strftime("%Y-%m-%d %H:%M:%S"), "msg": msg}
        self.history.appendleft(entry)
        self._log_queue.put(entry)  # handed off to single writer thread — no new thread per entry

    def _log_writer_loop(self) -> None:
        """Single background thread that drains the log queue in batches.
        Replaces one-thread-per-entry pattern — cuts thread count by ~95%."""
        import queue as _q
        while True:
            try:
                entry = self._log_queue.get(timeout=5)
            except _q.Empty:
                if self.stop_event.is_set():
                    break
                continue
            if entry is None:  # sentinel: writer told to exit
                break
            batch = [entry]
            # Drain any additional queued entries without waiting
            while True:
                try:
                    nxt = self._log_queue.get_nowait()
                    if nxt is None:
                        break
                    batch.append(nxt)
                    if len(batch) >= 20:
                        break
                except _q.Empty:
                    break
            try:
                with db_conn() as conn:
                    cur = conn.cursor()
                    for e in batch:
                        cur.execute(
                            "INSERT INTO ai_logs(email, session_id, t, msg) VALUES(%s, %s, %s, %s)",
                            (self.email, self.ai_session_id, e["t"], e["msg"]),
                        )
                    conn.commit()
            except Exception:
                pass

    def _open_ai_session(self) -> None:
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                if USING_PG:
                    # RETURNING id avoids SELECT lastval() which returns a dict row
                    # in psycopg3 — dict[0] raised TypeError and prevented conn.commit()
                    # causing the INSERT to silently roll back every time.
                    cur.execute(
                        "INSERT INTO ai_sessions(email, symbol, mode, trade_style, started_at) "
                        "VALUES(%s, %s, %s, %s, %s) RETURNING id",
                        (self.email, self.symbol, self.mode, self.trade_style,
                         now_dubai().strftime("%Y-%m-%d %H:%M:%S")),
                    )
                    row = cur.fetchone()
                    self.ai_session_id = row["id"] if isinstance(row, dict) else row[0]
                else:
                    cur.execute(
                        "INSERT INTO ai_sessions(email, symbol, mode, trade_style, started_at) "
                        "VALUES(%s, %s, %s, %s, %s)",
                        (self.email, self.symbol, self.mode, self.trade_style,
                         now_dubai().strftime("%Y-%m-%d %H:%M:%S")),
                    )
                    # SQLite: _Cursor wrapper has no lastrowid — use SELECT instead
                    cur.execute("SELECT last_insert_rowid()")
                    row = cur.fetchone()
                    self.ai_session_id = list(row.values())[0] if isinstance(row, dict) else row[0]
                conn.commit()
        except Exception as e:
            print(f"[ai-session] open failed: {e}")

    def _close_ai_session(self, reason: str) -> None:
        if not self.ai_session_id:
            return
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE ai_sessions SET ended_at=%s, stop_reason=%s WHERE id=%s",
                    (now_dubai().strftime("%Y-%m-%d %H:%M:%S"), reason, self.ai_session_id),
                )
                conn.commit()
        except Exception as e:
            print(f"[ai-session] close failed: {e}")

    def start(self):
        if self.ai_session_id:
            # Resuming existing session after server restart — don't create a new one
            self.log("AI resumed after server restart.")
        else:
            self._open_ai_session()
            self.log("AI started.")
        self.thread.start()
        # Mid-candle monitor runs in parallel — fires early trades on big price moves
        threading.Thread(target=self._run_mid_candle_monitor, daemon=True).start()

    def stop(self, reason="Stopped by user."):
        self.log(reason)
        self._close_ai_session(reason)
        self.stop_event.set()
        self._log_queue.put(None)  # sentinel: tell log writer to exit after draining
        if self.pending_trades:
            self.log(f"Pending trade(s) abandoned at stop (entry={self.pending_trades[0].get('entry_price', '?')}).")
            self.pending_trades = []

    def _reset_if_new_day(self):
        k = dubai_day_key()
        if k != self.day_key:
            self.day_key = k
            self.trades_today = 0
            self.bad_trades_today = 0
            self.log("Daily counters reset (Dubai timezone).")

            # Reset adaptive strictness at Dubai midnight so a bad day never
            # bleeds into the next. Step down gradually if very elevated.
            STEP_DOWN = {2.50: 1.50, 1.50: 1.25}
            if self.adaptive_strictness <= 1.25:
                self.adaptive_strictness = 1.0
                self.consecutive_wins = 0
                self.log("Midnight reset — strictness cleared to 1.0x. Fresh start.")
            else:
                self.adaptive_strictness = STEP_DOWN.get(
                    round(self.adaptive_strictness, 2), 1.25
                )
                self.log(f"Midnight reset — strictness stepped down to {self.adaptive_strictness:.2f}x")
            self.strictness_day_key = dubai_day_key()
            # Persist immediately so a server restart won't revert the reset
            _save_runner_state(self)

    def _reset_in_sec(self):
        now_dt = now_dubai()
        next_midnight = (now_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int((next_midnight - now_dt).total_seconds())

    def status(self) -> AutoState:
        last_run = datetime.utcfromtimestamp(self.last_run_ts).strftime("%Y-%m-%d %H:%M:%S") if self.last_run_ts else "-"
        last_trade = datetime.utcfromtimestamp(self.last_trade_ts).strftime("%Y-%m-%d %H:%M:%S") if self.last_trade_ts else "-"
        end_at = datetime.fromtimestamp(self.end_at_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if self.end_at_ts else None
        return AutoState(
            running=self.is_running(), email=self.email, symbol=self.symbol,
            tf=self.tf, interval_sec=self.interval_sec, mode=self.mode,
            side=self.last_side, last_signal=self.last_signal or "-",
            blocked_reason=self.blocked_reason, last_run_at=last_run,
            last_trade_at=last_trade, max_trades_per_day=self.max_trades_per_day,
            trades_today=self.trades_today, stop_after_bad_trades=self.stop_after_bad_trades,
            bad_trades_today=self.bad_trades_today, reset_in_sec=self._reset_in_sec(),
            duration_days=self.duration_days, end_at=end_at,
            trend_filter=self.trend_filter, chop_min_sep_pct=self.chop_min_sep_pct,
            signal_breakdown=self.last_breakdown,
            adaptive_strictness=self.adaptive_strictness,
            pending_trades=list(self.pending_trades),
            signal_score=self.last_score,
            trade_style=self.trade_style,
            market_grade=self.market_grade,
            market_regime=self.market_regime,
        )

    def _fetch_price_sync(self, symbol: str) -> float:
        """Sync OKX price fetch for use inside runner thread. 1 retry on transient errors."""
        inst = to_okx_inst(symbol)
        last_err: Exception = RuntimeError("price fetch failed")
        for _attempt in range(2):
            try:
                r = httpx.get(
                    f"{OKX_BASE}/api/v5/market/ticker",
                    params={"instId": inst},
                    timeout=8,
                    headers={"accept": "application/json"},
                )
                if r.status_code != 200:
                    raise RuntimeError(f"OKX HTTP {r.status_code} for {inst}")
                arr = (r.json() or {}).get("data") or []
                if not arr:
                    raise RuntimeError(f"No price data for {symbol}")
                return float(arr[0]["last"])
            except Exception as _pe:
                last_err = _pe
                if _attempt == 0:
                    time.sleep(2)
        raise RuntimeError(f"Price fetch failed after 2 attempts: {last_err}")

    def _close_one_trade(self, pt: Dict, exit_price: float, equity_before: float,
                         candle_high: float = 0.0, candle_low: float = 0.0) -> float:
        """
        Close a single pending trade dict. Returns equity_after.
        candle_high/candle_low: intrabar extremes — used to detect SL/TP hits
        that occurred DURING the interval, not just at close price.
        """
        entry = float(pt["entry_price"])
        side: Side = pt["side"]
        mode: RiskMode = pt["mode"]
        size_mult = float(pt.get("size_mult", 1.0))
        tp_mult   = float(pt.get("tp_mult", 1.0))
        is_primary = pt.get("is_primary", True)
        label = pt.get("label", "")

        # Base risk preset (size% and leverage from mode)
        risk = mini_asym_risk_engine(mode, equity_before)
        c = risk["computed"]

        # SL/TP locked at trade entry — never recalculate from fresh ATR mid-trade.
        # This ensures T1 and T2 keep their original separate SLs across candles
        # and after server restarts. Only the trailing stop can move SL upward.
        sp = TRADE_STYLE_PARAMS.get(self.trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
        atr = self.last_atr_pct
        if pt.get("sl_pct_open"):
            sl_pct = float(pt["sl_pct_open"])
            tp_pct = float(pt["tp_pct_open"]) * tp_mult
        else:
            # Legacy trades opened before this fix — fall back to ATR recalc
            sl_pct = min(atr * sp["sl_atr"], sp["sl_max"] / 100.0)
            tp_pct = min(atr * sp["tp_atr"], sp["tp_max"] / 100.0) * tp_mult
        sl_pct = max(sl_pct, 0.002)
        tp_pct = max(tp_pct, 0.004)

        # Break-even SL: T2 after T1 wins → SL moves to entry (risk-free trade)
        # Only activates when breakeven flag is set (via breakeven_next promotion in
        # _close_pending_trades — never fires on the same candle T1 wins).
        if pt.get("breakeven"):
            sl_pct = 0.0001  # near-zero SL at entry price — can only lose a tiny bit

        # ── Volatility-adjusted sizing ────────────────────────────────────
        # When ATR is elevated vs the typical range for this timeframe, reduce
        # position size so dollar-risk-per-trade stays roughly constant.
        #
        # BUG FIXED: atr_normal was derived from sl_max/sl_atr — a constant
        # formula from static parameters, not from observed market ATR.
        # SCALP: sl_max=1.5/sl_atr=0.8 → atr_normal=1.875%. Threshold = 2.8%.
        # BTC 15m real ATR = 0.3–0.5%, so the trigger (2.8%) was never reached.
        # Volatility sizing was effectively dead code in normal conditions.
        #
        # Fix: use empirical per-timeframe baselines from BTC historical ATR.
        # These are the typical quiet-session ATRs — 1.5× of these = elevated.
        _ATR_BASELINE = {
            "15m": 0.0040,   # 0.40% — typical BTC 15m ATR in normal conditions
            "1h":  0.0090,   # 0.90% — typical BTC 1h ATR
            "4h":  0.0180,   # 1.80% — typical BTC 4h ATR
            "1d":  0.0350,   # 3.50% — typical BTC 1d ATR
        }
        atr_normal = _ATR_BASELINE.get(self.tf, 0.0050)
        vol_ratio_atr = atr / atr_normal if atr_normal > 0 else 1.0
        vol_size_mult = 1.0
        if vol_ratio_atr > 1.5:
            vol_size_mult = max(0.4, 1.0 / vol_ratio_atr)  # scale down, floor at 40%
            # Throttle: log only when vol tier changes or every 30 minutes.
            # Without this, the log fires every 30s (twice per cycle for T1+T2),
            # flooding the session history with hundreds of identical lines.
            _vol_tier_now = round(vol_size_mult * 10)
            _vol_now_ts   = time.time()
            _vol_tier_last = getattr(self, "_last_vol_tier", -1)
            _vol_log_last  = getattr(self, "_last_vol_log_ts", 0.0)
            if _vol_tier_now != _vol_tier_last or (_vol_now_ts - _vol_log_last) >= 1800:
                self.log(
                    f"High volatility (ATR {atr*100:.2f}% = {vol_ratio_atr:.1f}× normal "
                    f"{atr_normal*100:.2f}%) → size reduced to {vol_size_mult*100:.0f}%"
                )
                self._last_vol_log_ts = _vol_now_ts
                self._last_vol_tier   = _vol_tier_now

        # ── Hard equity floor — never trade if account below 85% of session start ──
        if equity_before < self.floor_equity:
            self.log(
                f"HARD FLOOR HIT — equity ${equity_before:.2f} below 85% floor "
                f"(${self.floor_equity:.2f}). AI stopped to protect capital."
            )
            self.blocked_reason = "HARD_FLOOR"
            self.stop_event.set()
            email_ai_stopped(self.email, self.symbol, "HARD_FLOOR", equity_before)
            _tg_alert(
                f"🛑 <b>Hard floor hit — AI stopped</b>\n"
                f"{self.email} | {self.symbol}\n"
                f"Equity <b>${equity_before:.2f}</b> fell below floor <b>${self.floor_equity:.2f}</b>"
            )
            try:
                with db_conn() as _hf2:
                    _hf2.cursor().execute(
                        "UPDATE user_state SET last_stop_reason=%s, "
                        "peak_equity=%s, floor_equity=%s WHERE email=%s",
                        ("HARD_FLOOR", self.peak_equity, self.floor_equity, self.email),
                    )
                    _hf2.commit()
            except Exception as _hf_db_err:
                _tg_alert(
                    f"⚠️ <b>DB write failed — hard floor state not saved</b>\n"
                    f"{self.email} | {self.symbol}\n"
                    f"<code>{str(_hf_db_err)[:200]}</code>"
                )
            return equity_before

        # ── Drawdown protection (4-tier, from peak equity) ────────────────
        # Tiers shrink position size progressively — engine slows down as drawdown grows.
        # At -15% from peak: full stop. Hard floor (85% of session start) is a second wall.
        if equity_before > self.peak_equity:
            self.peak_equity = equity_before
            self.floor_equity = round(self.peak_equity * 0.85, 2)  # trail floor up with every new high
            # Persist immediately — called on every candle evaluation (holding or closing)
            # so the DB always has the latest peak/floor even between trade closes.
            # This ensures the balance endpoint shows the correct floor when runner is stopped.
            update_peak_ath(self.email, self.peak_equity, self.peak_equity)
        drawdown_pct = (self.peak_equity - equity_before) / self.peak_equity if self.peak_equity > 0 else 0.0
        dd_size_mult = 1.0

        # Determine which drawdown tier we are in (0 = no drawdown).
        # Used for throttled logging — only log when tier changes or every 30 min.
        _dd_now = time.time()
        _dd_log_interval = 1800  # 30 minutes between repeated drawdown reminders

        if drawdown_pct >= 0.15:
            # -15%+ from peak: stop trading, protect what's left — always log this
            self.log(f"Drawdown {drawdown_pct*100:.1f}% from peak — STOPPING to protect capital.")
            self.blocked_reason = "MAX_DRAWDOWN"
            self.stop_event.set()
            return equity_before
        elif drawdown_pct >= 0.10:
            dd_size_mult = 0.25   # -10%: quarter size (recovery mode)
            _dd_tier = 3
        elif drawdown_pct >= 0.07:
            dd_size_mult = 0.40   # -7%: 40% size
            _dd_tier = 2
        elif drawdown_pct >= 0.04:
            dd_size_mult = 0.65   # -4%: 65% size
            _dd_tier = 1
        else:
            _dd_tier = 0

        # Log only when: tier just changed (e.g. worsened from 40% → 25% size),
        # OR it has been 30+ minutes since the last drawdown log.
        # This prevents the log from filling with one identical line every 15 seconds.
        _tier_changed = (_dd_tier != self._last_dd_tier)
        _dd_due = (_dd_now - self._last_drawdown_log_ts) >= _dd_log_interval
        if _dd_tier > 0 and (_tier_changed or _dd_due):
            _labels = {1: "65%", 2: "40%", 3: "25%"}
            self.log(f"Drawdown {drawdown_pct*100:.1f}% from peak → size at {_labels[_dd_tier]}")
            self._last_drawdown_log_ts = _dd_now
        self._last_dd_tier = _dd_tier

        # Apply all size multipliers
        effective_size = c["size"] * size_mult * vol_size_mult * dd_size_mult

        # Hard per-trade max loss cap — only applies to legacy trades without locked SL.
        # For trades with sl_pct_open, the cap was already applied at entry time.
        if not pt.get("sl_pct_open"):
            max_loss_pct = {"SCALP": 0.015, "DAY_TRADE": 0.020, "SWING": 0.030}.get(self.trade_style, 0.020)
            max_sl_for_cap = max_loss_pct / (effective_size * c["leverage"]) if (effective_size * c["leverage"]) > 0 else sl_pct
            if sl_pct > max_sl_for_cap:
                sl_pct = max_sl_for_cap
                tp_pct = sl_pct * (sp["tp_atr"] / sp["sl_atr"]) * tp_mult

        # ── Intrabar SL/TP detection ──────────────────────────────────────
        # Use candle high/low to check if SL or TP was touched DURING the
        # interval, not just at close. This is more realistic for paper trading.
        sl_price = entry * (1 - sl_pct) if side == "LONG" else entry * (1 + sl_pct)
        tp_price = entry * (1 + tp_pct) if side == "LONG" else entry * (1 - tp_pct)

        # Breakeven SL uses CLOSE price only — not the wick.
        # A normal SL (real capital at risk) triggers on intrabar wick: your broker
        # fills the stop the moment price touches it, even on a wick.
        # A breakeven SL is different: its purpose is "exit if the trade goes to a
        # net loss." A wick below entry that closes back above is NOT a net loss.
        # Using wick for breakeven causes T2 to be stopped out of strong trending
        # trades by a single tick below entry, leaving profit on the table.
        # Fix: skip intrabar check when breakeven is active — the close-based
        # branch below handles it correctly (exits only if close < entry).
        _is_breakeven_sl = pt.get("breakeven", False) or sl_pct <= 0.0002
        intrabar_sl = (not _is_breakeven_sl) and candle_high > 0 and (
            (side == "LONG"  and candle_low  <= sl_price) or
            (side == "SHORT" and candle_high >= sl_price)
        )
        intrabar_tp = candle_high > 0 and (
            (side == "LONG"  and candle_high >= tp_price) or
            (side == "SHORT" and candle_low  <= tp_price)
        )

        # ── Progressive trailing stop ─────────────────────────────────────
        # Simulates a real trailing stop using intrabar high/low:
        #   Price reaches 40% of TP  → SL trails to break-even (entry)
        #   Price reaches 70% of TP  → SL trails to +50% of SL locked in
        #   Price reaches 100% of TP → TP_HIT (handled above)
        #
        # RATCHET FIX: best_move uses LIFETIME peak, not just current candle.
        # Previously trail_locked_pct reset to 0.0 every interval — if price
        # reached 70% of TP in candle 1 then retraced in candle 2, the profit
        # lock was forgotten and the original SL came back into effect.
        # Fix: persist peak move in pt["trail_best_move"] — can only increase.
        candle_best = candle_high if (side == "LONG" and candle_high > 0) else candle_low if candle_high > 0 else exit_price
        candle_best_move = (candle_best - entry) / entry if side == "LONG" else (entry - candle_best) / entry
        # Ratchet: only ever move forward
        lifetime_best_move = max(float(pt.get("trail_best_move", 0.0)), candle_best_move)
        pt["trail_best_move"] = lifetime_best_move   # persist for next interval
        best_move = lifetime_best_move

        trailing_activated = False
        trail_locked_pct   = 0.0   # how much profit is locked in

        # T2 holds its original SL until T1 wins — trailing stop only activates
        # after breakeven is set (meaning T1 hit TP and T2 became risk-free).
        # Before that, T2's SL never moves — it lives or dies at sl_pct_open.
        _t2_waiting = pt.get("breakeven_after_t1") and not pt.get("breakeven") and not pt.get("breakeven_next")

        if candle_high > 0 and not intrabar_tp and not _t2_waiting:
            if best_move >= tp_pct * 0.70:
                # Deep in profit — trail SL to lock in half the SL distance as profit
                trail_locked_pct  = sl_pct * 0.50
                trailing_activated = True
            elif best_move >= tp_pct * 0.40:
                # At break-even trigger — SL moves to entry
                trail_locked_pct  = 0.0
                trailing_activated = True

        # Ratchet: if a profit lock was established in a prior candle, keep it
        # even if the current candle didn't re-reach the trigger threshold.
        # Skip ratchet for T2 waiting on T1 — ignore any stale trail state.
        _prev_locked = float(pt.get("trail_locked_pct", 0.0)) if not _t2_waiting else 0.0
        if _prev_locked > 0.0 and not trailing_activated:
            trail_locked_pct   = _prev_locked
            trailing_activated = True
        # Locked profit can only increase (ratchet forward, never backward)
        trail_locked_pct = max(trail_locked_pct, _prev_locked)
        pt["trail_locked_pct"] = trail_locked_pct   # persist for next interval

        if trailing_activated:
            # New SL is at (entry + locked profit) — anything above this is safe
            if side == "LONG":
                sl_price = entry * (1 + trail_locked_pct)
            else:
                sl_price = entry * (1 - trail_locked_pct)
            sl_pct = trail_locked_pct  # effective SL distance from entry

        # SL and TP both touched in same candle → TP wins if trailing activated
        # (trade moved to profit first), otherwise SL wins (worst case)
        if intrabar_sl and intrabar_tp:
            intrabar_tp = trailing_activated  # trailing means TP is more likely
            intrabar_sl = not trailing_activated

        if intrabar_sl:
            if trailing_activated:
                # Trailing SL hit — price went up past the lock-in level then pulled back.
                # Exit at the locked-in profit level, NOT at a loss.
                final_move = trail_locked_pct   # positive = profit
                outcome = "TRAIL_STOP"
            else:
                final_move = -sl_pct
                outcome = "SL_HIT"
        elif intrabar_tp:
            final_move = tp_pct
            outcome = "TP_HIT"
        else:
            # No intrabar trigger — use actual close price
            raw_move = (exit_price - entry) / entry if side == "LONG" else (entry - exit_price) / entry
            if trailing_activated and raw_move < trail_locked_pct:
                # Trail SL was hit — exit at the locked-in level
                final_move = trail_locked_pct
                outcome = "TRAIL_STOP"
            elif raw_move <= -sl_pct:
                final_move = -sl_pct
                outcome = "SL_HIT"
            elif raw_move >= tp_pct:
                final_move = tp_pct
                outcome = "TP_HIT"
            else:
                # Multi-candle hold: close price is between SL and TP — keep position open.
                # Update trailing/breakeven state (already written to pt dict above),
                # then return None so _close_pending_trades keeps this trade alive.
                candles_held_so_far = int(
                    (time.time() - pt.get("open_ts", time.time())) / self.interval_sec
                )
                # Throttle: log at most once per candle interval to keep logs readable.
                # The 30-second wake-up cycle would otherwise spam one entry every 30s.
                _now = time.time()
                if _now - self._last_holding_log_ts >= self.interval_sec:
                    self._last_holding_log_ts = _now
                    self.log(
                        f"Holding ({side} @ {entry:.4f}) — candle {candles_held_so_far + 1} | "
                        f"close {exit_price:.4f} ({((exit_price - entry)/entry if side=='LONG' else (entry - exit_price)/entry)*100:+.3f}%) | "
                        f"SL={sl_price:.4f}  TP={tp_price:.4f}"
                    )

                # ── Momentum Scaling (Pyramiding) — Grade A only, both paper and real ──
                # Scale into winning positions in two stages.
                # Guards: in profit, drawdown < 7%, strictness ≤ 1.10.
                # Grade B excluded (T1/T2 already multi-entry).
                if (pt.get("scaling_enabled")
                        and best_move > 0
                        and not pt.get("breakeven_after_t1")
                        and drawdown_pct < 0.07
                        and self.adaptive_strictness <= 1.10):
                    _atr_now = float(atr) if atr else 0.0
                    if _atr_now > 0 and not pt.get("scale1_done") and best_move >= 1.5 * _atr_now:
                        pt["scale1_done"] = True
                        pt["size_mult"] = round(float(pt.get("size_mult", 0.70)) + 0.20, 4)
                        pt["breakeven"] = True
                        if REAL_TRADING:
                            _si_usdt = pt.get("usdt_size_base", 0) * 0.20
                            _si_lev  = pt.get("leverage_used", 6)
                            if _si_usdt > 0:
                                _si_ok = _place_real_scalein(self, self.symbol, side, _si_usdt, _si_lev)
                                if _si_ok:
                                    _move_real_sl_to_breakeven(self.email, self.symbol, entry)
                        self.log(
                            f"SCALE IN 1: +20% added at +{best_move*100:.2f}% "
                            f"(threshold 1.5×ATR={1.5*_atr_now*100:.2f}%). "
                            f"SL moved to breakeven. Total size {pt['size_mult']*100:.0f}% of mode."
                        )
                    elif _atr_now > 0 and pt.get("scale1_done") and not pt.get("scale2_done") and best_move >= 3.0 * _atr_now:
                        pt["scale2_done"] = True
                        pt["size_mult"] = round(float(pt.get("size_mult", 0.90)) + 0.10, 4)
                        pt["breakeven"] = False
                        pt["trail_locked_pct"] = round(1.5 * _atr_now, 6)
                        if REAL_TRADING:
                            _si2_usdt = pt.get("usdt_size_base", 0) * 0.10
                            _si2_lev  = pt.get("leverage_used", 6)
                            _sl2_price = (entry * (1 - 1.5 * _atr_now) if side == "LONG"
                                          else entry * (1 + 1.5 * _atr_now))
                            if _si2_usdt > 0:
                                _si2_ok = _place_real_scalein(self, self.symbol, side, _si2_usdt, _si2_lev)
                                if _si2_ok:
                                    try:
                                        _ex_row = get_exchange(self.email)
                                        if _ex_row:
                                            _ex2 = _make_ccxt_exchange(_ex_row)
                                            _set_position_sl(_ex2, _ex_row.get("exchange","bybit"), self.symbol, side, _sl2_price)
                                    except Exception:
                                        pass
                        self.log(
                            f"SCALE IN 2: +10% added at +{best_move*100:.2f}% "
                            f"(threshold 3.0×ATR={3.0*_atr_now*100:.2f}%). "
                            f"SL locked at +{1.5*_atr_now*100:.2f}%. Total size {pt['size_mult']*100:.0f}% of mode."
                        )

                return None

        # Use the actual SL/TP price for display and DB when those levels were hit,
        # not the raw candle close — otherwise exit price and PnL are inconsistent.
        if outcome == "SL_HIT":
            actual_exit = entry * (1 - sl_pct) if side == "LONG" else entry * (1 + sl_pct)
        elif outcome == "TP_HIT":
            actual_exit = entry * (1 + tp_pct) if side == "LONG" else entry * (1 - tp_pct)
        elif outcome == "TRAIL_STOP":
            actual_exit = entry * (1 + trail_locked_pct) if side == "LONG" else entry * (1 - trail_locked_pct)
        else:
            actual_exit = exit_price

        raw_move = final_move  # always consistent with PnL

        pnl_pct_leveraged = final_move * c["leverage"]
        size_dollar = equity_before * effective_size
        pnl_value = size_dollar * pnl_pct_leveraged

        # ── Realistic fee + slippage deduction ───────────────────────────────
        # Paper trading deducts real exchange fees so results match live trading.
        # Fee model: entry (limit/maker) + exit (limit for TP, market for SL/trail)
        # + slippage estimate per exchange. Prevents paper results > real results.
        ex_row = get_exchange(self.email)
        ex_id  = (ex_row.get("exchange") or "bybit").lower() if ex_row else "bybit"
        fee_cost  = exchange_fee_cost(ex_id, size_dollar, outcome)
        pnl_value = pnl_value - fee_cost

        equity_after = equity_before + pnl_value

        from_start = equity_after - self.session_start_equity
        dir_word = "up" if from_start >= 0 else "down"
        outcome_label = (
            "Take profit hit"      if outcome == "TP_HIT"
            else "Stop loss hit"   if outcome == "SL_HIT"
            else "Trailing stop"   if outcome == "TRAIL_STOP"
            else "Natural close"
        )
        grade_label = pt.get("grade", "A")
        style_label = self.trade_style.replace("_", " ").title()
        tp_display = tp_pct * 100 / tp_mult   # show full ATR TP in display
        reason_text = "\n".join([
            f"{mode}  •  AI Trade  •  {style_label} ({self.tf})",
            f"Grade {grade_label}{' — ' + label if label else ''}",
            "",
            f"Signal       {pt.get('signal', '-')}  →  {side}",
            f"Entry        ${entry:,.4f}",
            f"Exit         ${actual_exit:,.4f}  ({raw_move * 100:+.3f}% price move)",
            f"Outcome      {outcome_label}  →  {pnl_pct_leveraged * 100:+.2f}% PnL  (${pnl_value:+.2f})",
            f"Fees+Slip    -${fee_cost:.2f}  ({ex_id})",
            "",
            "Position (ATR-based)",
            f"  Size:        {effective_size * 100:.1f}% of equity  →  ${size_dollar:.2f}",
            f"  Leverage:    {c['leverage']:.0f}×",
            f"  SL (1×ATR):  {sl_pct * 100:.3f}%   |   TP ({tp_mult:.1f}×ATR): {tp_pct * 100:.3f}%",
            f"  ATR:         {atr * 100:.3f}%",
            f"  Trailing:    {'activated — locked ' + f'{trail_locked_pct*100:.2f}%' if trailing_activated else 'not triggered'}",
            f"  Vol adj:     {vol_size_mult*100:.0f}%  |  DD adj: {dd_size_mult*100:.0f}%",
            "",
            "Account",
            f"  Before:      ${equity_before:,.2f}",
            f"  After:       ${equity_after:,.2f}  ({dir_word} {abs(from_start / max(1, self.session_start_equity) * 100):.2f}% from session start)",
        ])

        if REAL_TRADING:
            _is_b_t1 = (pt.get("grade") == "B" and pt.get("label") == "T1")
            if _is_b_t1:
                # For real Grade B T1, confirm TP via Bybit order check.
                # Bug 3 fix: detection priority —
                #   Step 1: Check if T1_TP limit order was filled on Bybit (primary).
                #   Step 2: If order check fails (no id, network error, markets issue),
                #           fall back to intrabar_tp from paper sim as secondary signal.
                #           intrabar_tp = candle high touched T1_TP price level.
                #   Never assume SL_HIT when T1_TP order ID is missing — check both paths.
                _t1_tp_id = pt.get("t1_tp_id")
                _t1_filled = False
                if _t1_tp_id:
                    _t1_filled = _check_bybit_order_filled(self.email, _t1_tp_id, self.symbol)
                    if not _t1_filled:
                        # Primary check failed — use paper sim intrabar_tp as fallback
                        # (candle high reached the TP price level = TP likely triggered on exchange)
                        if intrabar_tp:
                            _t1_filled = True
                            self.log(
                                "T1 TP: Bybit order check inconclusive — intrabar_tp confirms "
                                "candle touched TP level, treating as TP_HIT"
                            )
                else:
                    # No T1_TP order id stored (placement failed earlier) —
                    # use intrabar_tp as the only available signal
                    if intrabar_tp:
                        _t1_filled = True
                        self.log(
                            "T1 TP: no order id stored — intrabar_tp indicates TP level was touched, "
                            "treating as TP_HIT"
                        )

                if _t1_filled:
                    self.log(f"REAL T1 HIT TP | {side} {self.symbol} — T2 still running, SL→breakeven next candle")
                    # Cancel the T1 stop order — T1 is closed at TP so its stop is redundant.
                    # T2's stop (t2_sl_id) remains active at the original SL price and will be
                    # moved to breakeven by _move_real_sl_to_breakeven on the next candle.
                    _t1_sl_id = pt.get("t1_sl_id")
                    if _t1_sl_id:
                        try:
                            _t1c_row = get_exchange(self.email)
                            if _t1c_row:
                                _t1c_ex = _make_ccxt_exchange(_t1c_row)
                                _ccxt_call(_t1c_ex.cancel_order, _t1_sl_id, self.symbol,
                                           label="cancel_t1_sl_on_tp", retries=0)
                                self.log(f"T1 SL stop {_t1_sl_id} cancelled — T1 closed at TP, T2 SL still active")
                        except Exception as _cte:
                            self.log(f"T1 SL cancel note (non-fatal, T2 SL still protects): {_cte}")
                else:
                    # Bug 6 fix: T1 confirmed NOT at TP — only now run T2 SL verification.
                    # This path was incorrectly reached when _check_bybit_order_filled returned
                    # False due to markets not loaded, causing T2 to be force-closed.
                    self.log(f"REAL T1 CLOSED ({outcome}) — not TP_HIT | verifying T2 SL...")
                    # T1 exited at SL (not TP). The T1 stop order was consumed by Bybit.
                    # T2's separate stop should still be active — verify and re-place if missing.
                    _t2_pt_for_sl = next((p for p in self.pending_trades if p.get("label") == "T2"), None)
                    if _t2_pt_for_sl:
                        _t2_sl_id   = _t2_pt_for_sl.get("t2_sl_id")
                        _t2_sl_pct  = float(_t2_pt_for_sl.get("sl_pct_open", 0.015))
                        _t2_entry   = float(_t2_pt_for_sl.get("entry_price", 0))
                        _t2_side    = _t2_pt_for_sl.get("side", side)
                        _t2_sl_px   = _t2_entry * (1 - _t2_sl_pct) if _t2_side == "LONG" else _t2_entry * (1 + _t2_sl_pct)
                        try:
                            _t2v_row = get_exchange(self.email)
                            if _t2v_row:
                                _t2v_ex    = _make_ccxt_exchange(_t2v_row)
                                _t2v_ex_id = (_t2v_row.get("exchange") or "bybit").lower()
                                # Bug 5 fix: pre-load markets on the new exchange instance
                                # before _verify_and_restore_t2_sl calls ex.market(symbol)
                                if not _t2v_ex.markets:
                                    try:
                                        _t2v_ex.load_markets()
                                    except Exception as _lm:
                                        self.log(f"T2 verify: market load note: {_lm}")
                                _verify_and_restore_t2_sl(
                                    self, _t2v_ex, _t2v_ex_id, self.symbol,
                                    _t2_side, _t2_sl_id, _t2_sl_px,
                                )
                        except Exception as _vte:
                            self.log(f"CRITICAL: T2 SL verify error: {_vte} — MANUAL CHECK NEEDED on {self.symbol}")
                    # T2 is still live — do NOT cancel/close; let it run to its own TP or SL
            else:
                # Grade A, or Grade B T2 (TP or SL hit): cancel open orders then confirm close
                _cancel_all_real_orders(self.email, self.symbol)
                try:
                    close_real_order(email=self.email, symbol=self.symbol, side=side)
                    self.log(f"REAL POSITION CLOSED | {side} {self.symbol} | {outcome}")
                except Exception as _ce:
                    self.log(f"REAL CLOSE note: {_ce}")
            # Sync real balance after any close event — always bypass cache for accuracy
            real_bal_after = get_real_usdt_balance(self.email, force=True)
            if real_bal_after is not None:
                equity_after = real_bal_after
                real_pnl = real_bal_after - equity_before
                self.log(f"Post-trade balance synced: ${real_bal_after:.2f} USDT (real P&L: ${real_pnl:+.2f})")
                # Real balance change is authoritative — overrides paper simulation.
                # Paper sim diverges when Bybit fills at a different price than the
                # simulated candle close (wick fill, force-close via _ensure_sl_or_close,
                # slippage, or partial fills). We always trust what Bybit paid us.
                if size_dollar > 0:
                    pnl_value = real_pnl
                    # Use account equity as denominator so T1 and T2 show the same scale.
                    # Dividing by size_dollar inflates T2 % because T2's position is only 40%
                    # of base size — a $6 loss on a $5.60 T2 position shows -107% (impossible).
                    pnl_pct_leveraged = real_pnl / equity_before if equity_before > 0 else 0.0
                _old_outcome = outcome
                if real_pnl > 0 and outcome in ("SL_HIT", "TRAIL_STOP"):
                    outcome = "TP_HIT"
                elif real_pnl < -(equity_before * 0.005) and outcome == "TP_HIT":
                    outcome = "SL_HIT"
                if outcome != _old_outcome:
                    self.log(
                        f"Outcome auto-corrected {_old_outcome} → {outcome} "
                        f"(real balance change ${real_pnl:+.4f} vs paper sim)"
                    )
                # Always append real P&L to reason_text when it differs from paper sim
                # by more than $0.05 — the trade card was built with paper values so
                # users would see wrong numbers without this correction note.
                _paper_pnl = size_dollar * (final_move * c["leverage"]) - fee_cost if size_dollar > 0 else 0.0
                if abs(real_pnl - _paper_pnl) > 0.05:
                    _outcome_label_real = (
                        "TP_HIT" if outcome == "TP_HIT"
                        else "SL_HIT" if outcome == "SL_HIT"
                        else outcome
                    )
                    reason_text += (
                        f"\n\n[REAL P&L] Paper sim: ${_paper_pnl:+.2f} → "
                        f"Actual Bybit: ${real_pnl:+.2f} | Outcome: {_outcome_label_real}"
                        f"\n  Balance: ${equity_before:.2f} → ${real_bal_after:.2f}"
                    )

        sid = get_session_id(self.email)
        tr = Trade(
            time=now_utc_str(), side=side, symbol=self.symbol, mode=mode,
            size=float(effective_size),
            sl=round(sl_pct * 100, 4), tp=round(tp_pct * 100, 4),
            leverage=float(c["leverage"]), entry_price=entry,
            current_price=actual_exit,
            unreal_pnl_percent=float(pnl_pct_leveraged * 100.0),
            unreal_pnl_value=float(pnl_value),
            equity_after=float(equity_after), reason=reason_text,
            is_primary=int(is_primary),
        )

        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO trades(
                    email, time, side, symbol, mode, size, sl, tp, leverage,
                    entry_price, current_price, unreal_pnl_percent, unreal_pnl_value,
                    equity_after, reason, session_id, hard_floor, is_primary
                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (self.email, tr.time, tr.side, tr.symbol, tr.mode,
                 float(tr.size), float(tr.sl), float(tr.tp), float(tr.leverage),
                 float(tr.entry_price), float(tr.current_price),
                 float(tr.unreal_pnl_percent), float(tr.unreal_pnl_value),
                 float(tr.equity_after), tr.reason, int(sid), float(self.floor_equity),
                 int(is_primary)),
            )
            conn.commit()

        set_equity(self.email, equity_after)
        # Update in-memory peak immediately so the next trade's drawdown calc is accurate.
        # Without this, a profitable real trade raises equity_after above equity_before but
        # self.peak_equity stays stale — floor_equity never ratchets up until the NEXT trade.
        if equity_after > self.peak_equity:
            self.peak_equity = equity_after
            self.floor_equity = round(self.peak_equity * 0.85, 2)
        update_peak_ath(self.email, float(self.peak_equity), float(equity_after))

        # A "bad trade" requires an ACTUAL loss > 0.5% of current equity.
        # Trail stops at entry, breakeven closes, and tiny fee losses are NOT bad trades —
        # they must not trigger strictness increases, double cooldown, or the bad-trade counter.
        _bad_trade_threshold = equity_before * 0.005
        _real_loss = pnl_value < -_bad_trade_threshold

        if is_primary:
            self.trades_today += 1
            self._last_trade_bad = _real_loss
            if _real_loss:
                self.bad_trades_today += 1

        # Mini-Asym adaptive strictness — only update on primary trade
        if is_primary and mode == "MINI_ASYM":
            if _real_loss:
                self.consecutive_wins = 0
                # Graduated steps: 1.0 → 1.10 → 1.25 → 1.50 → 2.50
                # First loss is gentle (+0.10); escalates with each consecutive loss.
                _STRICT_STEPS = [1.0, 1.10, 1.25, 1.50, 2.50]
                _next_strict = 2.50
                for _s in _STRICT_STEPS:
                    if _s > self.adaptive_strictness + 0.001:
                        _next_strict = _s
                        break
                self.adaptive_strictness = _next_strict
                self.strictness_day_key = dubai_day_key()
                self.log(f"MINI_ASYM strictness ↑ {self.adaptive_strictness:.2f} (after loss)")
            elif pnl_value > 0:
                # Only genuine profit advances the win streak and relaxes strictness
                self.consecutive_wins += 1
                if self.consecutive_wins >= 3:
                    # 3 wins in a row → reset strictness fully, AI is in good form
                    self.adaptive_strictness = 1.0
                    self.consecutive_wins = 0
                    self.strictness_day_key = dubai_day_key()
                    self.log("MINI_ASYM strictness reset to 1.0 (3 consecutive wins)")
                else:
                    self.adaptive_strictness = max(1.0, self.adaptive_strictness - 0.10)
            # else: breakeven or sub-threshold fee loss — no strictness change, no win streak

        mt = self.max_trades_per_day if self.max_trades_per_day > 0 else "∞"
        self.log(
            f"TRADE CLOSED ({side}){' [' + label + ']' if label else ''} | {outcome} | "
            f"entry={entry:.2f} exit={actual_exit:.2f} | "
            f"pnl={pnl_pct_leveraged * 100:.2f}% (${pnl_value:.2f}) | "
            f"trades={self.trades_today}/{mt}"
        )

        email_trade_closed(
            to=self.email, symbol=self.symbol, side=side, mode=mode,
            entry=entry, exit_price=exit_price, outcome=outcome,
            pnl_pct=pnl_pct_leveraged * 100.0,
            pnl_value=pnl_value, equity_after=equity_after,
        )

        return equity_after

    def _close_pending_trades(self) -> None:
        """Evaluate pending trades each interval (multi-candle).
        A trade stays alive until SL, TP, or trailing stop is hit.
        Only then is it removed from pending_trades and written to DB."""
        if not self.pending_trades:
            return
        try:
            klines = _fetch_klines_sync(self.symbol, self.tf, limit=10)
            if klines:
                last_candle = klines[-1]
                candle_high = last_candle["high"]
                candle_low  = last_candle["low"]
                exit_price  = last_candle["close"]
            else:
                exit_price  = self._fetch_price_sync(self.symbol)
                candle_high = exit_price
                candle_low  = exit_price

            still_open: List[Dict] = []
            t1_won = False
            t1_entry_price: Optional[float] = None

            for pt in self.pending_trades:
                # Promote deferred breakeven: T1 won last candle → T2 is now risk-free.
                # Deferred one candle so T2 is never evaluated with a 0.010% SL in the
                # same pass that T1 won (which would kill it at near-entry immediately).
                if pt.get("breakeven_next"):
                    pt["breakeven"] = True
                    del pt["breakeven_next"]

                eq = get_equity(self.email)
                result = self._close_one_trade(pt, exit_price, eq,
                                               candle_high=candle_high, candle_low=candle_low)

                if result is None:
                    # NATURAL_CLOSE — price between SL and TP, hold another candle
                    still_open.append(pt)
                else:
                    # Definitively closed — determine if T1 won for Grade B breakeven logic.
                    # Real trading: Bybit API confirms T1 TP order fill (never trust price sim).
                    # Paper trading / Grade A: price-simulation profit check.
                    if pt.get("is_primary"):
                        if REAL_TRADING and pt.get("grade") == "B" and pt.get("t1_tp_id"):
                            if _check_bybit_order_filled(self.email, pt["t1_tp_id"], self.symbol):
                                t1_won = True
                                t1_entry_price = float(pt.get("entry_price", 0))
                        elif result > eq:
                            t1_won = True
                            t1_entry_price = float(pt.get("entry_price", 0))

            self.pending_trades = still_open

            # After the loop, still_open contains T2 — safe to activate its breakeven now
            if t1_won:
                for pt in still_open:
                    if pt.get("breakeven_after_t1") and not pt.get("breakeven"):
                        pt["breakeven_next"] = True
                        if REAL_TRADING and t1_entry_price:
                            try:
                                ok = _move_real_sl_to_breakeven(self.email, self.symbol, t1_entry_price)
                                if ok:
                                    self.log(f"REAL T2 SL→BREAKEVEN @ {t1_entry_price:.4f}")
                                else:
                                    self.log("REAL T2 SL breakeven move failed — T2 continues with original SL")
                            except Exception as _slm:
                                self.log(f"REAL T2 SL move error: {_slm}")

            # Persist live state so a redeploy picks up the updated positions/strictness
            _save_runner_state(self)

        except Exception as e:
            self.log(f"Error evaluating trade(s): {e}")
            # On error, leave pending_trades intact — retry next interval

    def _signal_and_filters(self):
        klines = _fetch_klines_sync(self.symbol, self.tf, limit=260)
        higher_tf = HIGHER_TF_MAP.get(self.tf, "4h")
        higher_klines: List[Dict] = []
        if higher_tf != self.tf:
            try:
                higher_klines = _fetch_klines_sync(self.symbol, higher_tf, limit=100)
            except Exception:
                higher_klines = []
        # 15m confirmation layer — DAY_TRADE and SWING only (SCALP IS 15m already)
        mtf_klines: List[Dict] = []
        if self.trade_style in ("DAY_TRADE", "SWING") and self.tf != "15m":
            try:
                mtf_klines = _fetch_klines_sync(self.symbol, "15m", limit=50)
            except Exception:
                mtf_klines = []
        res = _compute_signal_layers(klines, self.mode, self.adaptive_strictness, higher_klines, self.trade_style, mtf_klines)
        self.last_breakdown = res.get("breakdown", {})
        self.last_score = res.get("score", 0.0)
        self.market_grade = res.get("grade", "-")
        self.signal_type = res.get("signal_type", "NORMAL")
        self.sl_atr_override = res.get("sl_atr_override")
        self.tp_atr_override = res.get("tp_atr_override")
        if res.get("signal_type") == "REVERSAL" and res.get("reversal_log"):
            self.log(res["reversal_log"])
        if res.get("market_regime"):
            self.market_regime = res["market_regime"]
        if res.get("atr_pct"):
            self.last_atr_pct = res["atr_pct"]
        slope_pct = res.get("htf_bear_slope_pct")
        return res

    def _secs_until_dubai_midnight(self) -> int:
        """Seconds remaining until next Dubai midnight."""
        now_dt = now_dubai()
        next_midnight = (now_dt + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1, int((next_midnight - now_dt).total_seconds()))

    def _sleep_until_dubai_midnight(self) -> None:
        """Sleep in small chunks until Dubai midnight, checking stop_event."""
        secs = self._secs_until_dubai_midnight()
        self.log(f"Daily limit reached — pausing {secs // 3600}h {(secs % 3600) // 60}m until Dubai midnight.")
        chunk = 60  # wake up every minute to check stop_event
        while secs > 0 and not self.stop_event.is_set():
            time.sleep(min(chunk, secs))
            secs -= chunk
        if not self.stop_event.is_set():
            # New day — apply strictness step-down (same logic as _reset_if_new_day)
            STEP_DOWN = {2.50: 1.50, 1.50: 1.25}
            if self.adaptive_strictness <= 1.25:
                self.adaptive_strictness = 1.0
                self.consecutive_wins = 0
                self.log("Midnight reset (sleep path) — strictness cleared to 1.0x. Fresh start.")
            else:
                self.adaptive_strictness = STEP_DOWN.get(
                    round(self.adaptive_strictness, 2), 1.25
                )
                self.log(f"Midnight reset (sleep path) — strictness stepped down to {self.adaptive_strictness:.2f}x")
            self.strictness_day_key = dubai_day_key()
            # Reload counters from DB and resume
            self.trades_today, self.bad_trades_today = self._load_today_stats()
            self.day_key = dubai_day_key()
            self._last_trade_bad = False
            self.last_trade_ts = time.time()
            _save_runner_state(self)
            self.log("New Dubai day — daily limits reset. Resuming AI.")

    def _run_mid_candle_monitor(self):
        """
        Background thread: checks for significant intra-candle price moves every
        5 minutes. If price has moved > threshold% from the current candle's open,
        runs a full signal analysis. Fires a trade immediately if score >= 0.75.

        Fires LONG on upward moves, SHORT on downward moves.
        Sets last_trade_ts after firing so the main loop's cooldown naturally
        prevents double-firing at the next scheduled candle close.
        Stops cleanly when stop_event is set.
        """
        threshold = _MID_CANDLE_THRESHOLDS.get(self.trade_style, 0.015)
        cooldown_sec = max(60, self.interval_sec)

        # Wait one full check interval before starting — give the main loop time
        # to compute the first signal and initialise last_atr_pct / market_grade.
        for _ in range(_MID_CANDLE_INTERVAL // 10):
            if self.stop_event.is_set():
                return
            time.sleep(10)

        while not self.stop_event.is_set():
            try:
                # ── Pre-checks — skip quickly if conditions aren't right ───────────
                if self.pending_trades:
                    pass  # fall through to next sleep; main loop will handle them
                else:
                    # Respect daily limits and cooldown
                    _daily_bad_ok = (self.stop_after_bad_trades <= 0 or
                                     self.bad_trades_today < self.stop_after_bad_trades)
                    _daily_trade_ok = (self.max_trades_per_day <= 0 or
                                       self.trades_today < self.max_trades_per_day)
                    now_ts = time.time()
                    _effective_cd = cooldown_sec * (2 if self._last_trade_bad else 1)
                    _cooldown_ok = (now_ts - self.last_trade_ts) >= _effective_cd

                    if _daily_bad_ok and _daily_trade_ok and _cooldown_ok:
                        # ── Fetch current candle open and live price ───────────────
                        klines = _fetch_klines_sync(self.symbol, self.tf, limit=3)
                        if klines:
                            candle_open  = float(klines[-1]["open"])
                            current_px   = self._fetch_price_sync(self.symbol)
                            move_pct     = (current_px - candle_open) / candle_open
                            abs_move     = abs(move_pct)

                            if abs_move >= threshold:
                                # ── Full 4-layer signal analysis ──────────────────
                                res = self._signal_and_filters()
                                score      = res.get("score", 0.0)
                                signal_ok  = res.get("ok", False)
                                signal_side: Side = res.get("side", "LONG")

                                if not signal_ok:
                                    pass  # blocked — silent
                                elif score < _MID_CANDLE_MIN_SCORE:
                                    pass  # score too low — silent
                                else:
                                    # Direction gate: price move must align with signal
                                    expected_side: Side = "LONG" if move_pct > 0 else "SHORT"
                                    if signal_side != expected_side:
                                        pass  # direction mismatch — silent
                                    else:
                                        # ── Fire trade ────────────────────────────
                                        entry_price = self._fetch_price_sync(self.symbol)
                                        grade = self.market_grade
                                        c   = presets_for_mode(self.mode)
                                        st  = TRADE_STYLE_PARAMS.get(self.trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
                                        sl_pct_open = min(self.last_atr_pct * st["sl_atr"], st["sl_max"] / 100.0)
                                        tp_pct_open = min(self.last_atr_pct * st["tp_atr"], st["tp_max"] / 100.0)

                                        # BTC correlation check for mid-candle (skip BTC itself)
                                        _btc_mc_mult = 1.0
                                        if "BTC" not in self.symbol.upper():
                                            _btc_mc = _get_btc_momentum(self.tf)
                                            if _btc_mc is not None:
                                                _mc_against = ((signal_side == "LONG"  and _btc_mc < -0.02) or
                                                               (signal_side == "SHORT" and _btc_mc >  0.02))
                                                if _mc_against:
                                                    if abs(_btc_mc) >= 0.04:
                                                        raise _MidCandleSkip()  # strong conflict — skip
                                                    _btc_mc_mult = 0.60

                                        if REAL_TRADING:
                                            real_bal = get_real_usdt_balance(self.email, force=True)
                                            if real_bal is None:
                                                self.log("MID_CANDLE REAL ORDER SKIPPED — could not fetch balance")
                                                raise _MidCandleSkip()
                                            equity_now = real_bal
                                            set_equity(self.email, real_bal)
                                        else:
                                            equity_now = get_equity(self.email)

                                        real_order_id   = None
                                        real_b_result   = None
                                        _mc_real_result = None
                                        if REAL_TRADING:
                                            try:
                                                # Apply vol and drawdown size adjustments — same
                                                # logic as _run_loop fix; no mtf mult here because
                                                # mid-candle trades don't go through MTF confirmation.
                                                _atr_bl_mc = {"15m": 0.0040, "1h": 0.0090, "4h": 0.0180, "1d": 0.0350}
                                                _atr_nm_mc = _atr_bl_mc.get(self.tf, 0.0050)
                                                _vol_sm_mc = (max(0.4, _atr_nm_mc / self.last_atr_pct)
                                                              if self.last_atr_pct > _atr_nm_mc else 1.0)
                                                _dd_now_mc = (max(0.0, (self.peak_equity - equity_now) / self.peak_equity)
                                                              if self.peak_equity > 0 else 0.0)
                                                _dd_sm_mc  = (0.25 if _dd_now_mc >= 0.10 else
                                                              0.40 if _dd_now_mc >= 0.07 else
                                                              0.65 if _dd_now_mc >= 0.04 else 1.0)
                                                _mc_fund_mult = 1.0
                                                _mc_fr = _get_funding_rate(self.email, self.symbol)
                                                if _mc_fr is not None:
                                                    _mc_fr_against = ((signal_side == "LONG"  and _mc_fr >  0.001) or
                                                                      (signal_side == "SHORT" and _mc_fr < -0.001))
                                                    if _mc_fr_against:
                                                        _mc_fund_mult = 0.50 if abs(_mc_fr) >= 0.003 else 0.70
                                                        self.log(f"MID_CANDLE funding {_mc_fr*100:.4f}%/8h against {signal_side} — size × {_mc_fund_mult:.2f}")
                                                size_pct  = float(c["size"]) * _vol_sm_mc * _dd_sm_mc * _btc_mc_mult * _mc_fund_mult
                                                usdt_size = equity_now * size_pct
                                                self.log(
                                                    f"MID-CANDLE REAL ORDER SIZE: base={float(c['size'])*100:.0f}% "
                                                    f"× vol={_vol_sm_mc:.2f} × dd={_dd_sm_mc:.2f} "
                                                    f"× btc={_btc_mc_mult:.2f} × fund={_mc_fund_mult:.2f} "
                                                    f"= {size_pct*100:.1f}% → ${usdt_size:.2f}"
                                                )
                                                if grade == "B":
                                                    real_b_result = place_real_grade_b_order(
                                                        email=self.email, symbol=self.symbol,
                                                        side=signal_side, usdt_size=usdt_size,
                                                        leverage=int(c["leverage"]),
                                                        sl_pct=sl_pct_open,
                                                        t1_tp_pct=tp_pct_open * 0.80,
                                                        t2_tp_pct=tp_pct_open * 1.00,
                                                    )
                                                else:
                                                    _mc_real_result = place_real_order(
                                                        email=self.email, symbol=self.symbol,
                                                        side=signal_side, usdt_size=usdt_size,
                                                        leverage=int(c["leverage"]),
                                                        sl_pct=sl_pct_open, tp_pct=tp_pct_open,
                                                    )
                                                    real_order_id = _mc_real_result.get("order_id")
                                            except Exception as _re:
                                                self.log(f"MID_CANDLE REAL ORDER FAILED — {_re}")
                                                _tg_alert(
                                                    f"❌ <b>Real order failed (mid-candle)</b>\n"
                                                    f"{self.email} | {self.symbol} {signal_side}\n"
                                                    f"<code>{str(_re)[:250]}</code>"
                                                )
                                                raise _MidCandleSkip()

                                        base_trade = {
                                            "entry_price":    entry_price,
                                            "side":           signal_side,
                                            "mode":           self.mode,
                                            "signal":         self.last_signal,
                                            "open_ts":        time.time(),
                                            "sl_pct_open":    sl_pct_open,
                                            "tp_pct_open":    tp_pct_open,
                                            "mid_candle":     True,
                                            "usdt_size_base": usdt_size if REAL_TRADING else 0,
                                            "leverage_used":  int(c["leverage"]),
                                        }
                                        if grade == "B" and REAL_TRADING:
                                            self.pending_trades = [
                                                {**base_trade, "grade": "B", "size_mult": 0.60,
                                                 "tp_mult": 0.80, "is_primary": True, "label": "T1",
                                                 "order_id": real_b_result.get("order_id"),
                                                 "t1_tp_id": real_b_result.get("t1_tp_id"),
                                                 "t1_sl_id": None},   # filled after stop placement
                                                {**base_trade, "grade": "B", "size_mult": 0.40,
                                                 "tp_mult": 1.00, "is_primary": False, "label": "T2",
                                                 "breakeven_after_t1": True, "t2_sl_id": None},
                                            ]
                                        elif grade == "B":
                                            self.pending_trades = [
                                                {**base_trade, "grade": "B", "size_mult": 0.60,
                                                 "tp_mult": 0.80, "is_primary": True, "label": "T1"},
                                                {**base_trade, "grade": "B", "size_mult": 0.40,
                                                 "tp_mult": 1.00, "is_primary": False, "label": "T2",
                                                 "breakeven_after_t1": True},
                                            ]
                                        else:
                                            self.pending_trades = [
                                                {**base_trade, "grade": "A", "size_mult": 1.00,
                                                 "tp_mult": 1.00, "is_primary": True,
                                                 **({"order_id": real_order_id} if real_order_id else {})},
                                            ]

                                        # FIX 1: Log position BEFORE SL attempt
                                        self.last_trade_ts = time.time()
                                        _save_runner_state(self)

                                        self.log(
                                            f"MID_CANDLE TRADE OPENED Grade {grade} ({signal_side}) "
                                            f"@ {entry_price:.4f} | score={score:.2f} | "
                                            f"{abs_move*100:.1f}% price move triggered"
                                        )
                                        _mc_sl_px = entry_price * (1 + sl_pct_open) if signal_side == "SHORT" else entry_price * (1 - sl_pct_open)
                                        _mc_tp_px = entry_price * (1 - tp_pct_open) if signal_side == "SHORT" else entry_price * (1 + tp_pct_open)
                                        try:
                                            email_trade_opened(
                                                to=self.email, symbol=self.symbol, side=signal_side, mode=self.mode,
                                                grade=grade, entry=entry_price, sl=_mc_sl_px, tp=_mc_tp_px,
                                                score=score, equity=equity_now,
                                            )
                                        except Exception:
                                            pass

                                        # Place SL — Grade B: two separate stops; Grade A: position SL
                                        if REAL_TRADING:
                                            if grade == "B":
                                                _mc_b_ex    = real_b_result["ex"]
                                                _mc_b_ex_id = real_b_result["ex_id"]
                                                _mc_b_sl    = real_b_result["sl_price"]
                                                if _mc_b_ex_id == "bybit":
                                                    _mc_t1_sl = _place_grade_b_stop_order(
                                                        self, _mc_b_ex, _mc_b_ex_id, self.symbol,
                                                        signal_side, _mc_b_sl,
                                                        real_b_result["t1_qty"], "T1_SL",
                                                    )
                                                    _mc_t2_sl = _place_grade_b_stop_order(
                                                        self, _mc_b_ex, _mc_b_ex_id, self.symbol,
                                                        signal_side, _mc_b_sl,
                                                        real_b_result["t2_qty"], "T2_SL",
                                                    )
                                                    self.pending_trades[0]["t1_sl_id"] = _mc_t1_sl
                                                    self.pending_trades[1]["t2_sl_id"] = _mc_t2_sl
                                                    _save_runner_state(self)
                                                    if not _mc_t1_sl or not _mc_t2_sl:
                                                        self.log(f"MID_CANDLE Grade B partial stop failed — position SL fallback")
                                                        _mc_sl_ok = _ensure_sl_or_close(self, _mc_b_ex, _mc_b_ex_id, self.symbol, signal_side, _mc_b_sl)
                                                        if not _mc_sl_ok:
                                                            self.pending_trades = []
                                                            _save_runner_state(self)
                                                else:
                                                    _mc_sl_ok = _ensure_sl_or_close(self, _mc_b_ex, _mc_b_ex_id, self.symbol, signal_side, _mc_b_sl)
                                                    if not _mc_sl_ok:
                                                        self.pending_trades = []
                                                        _save_runner_state(self)
                                            else:
                                                _mc_sl_ok = _ensure_sl_or_close(
                                                    self, _mc_real_result["ex"], _mc_real_result["ex_id"],
                                                    self.symbol, signal_side, _mc_real_result["sl_price"],
                                                )
                                                if not _mc_sl_ok:
                                                    self.pending_trades = []
                                                    _save_runner_state(self)

            except _MidCandleSkip:
                pass
            except Exception as _mc_err:
                self.log(f"MID_CANDLE monitor error: {_mc_err}")

            # Sleep in 10-second chunks so stop_event is respected quickly
            for _ in range(_MID_CANDLE_INTERVAL // 10):
                if self.stop_event.is_set():
                    return
                time.sleep(10)

    def _reconcile_exchange_positions(self) -> None:
        """
        FIX 3: Compare exchange open positions with engine state.
        - Position on exchange but not in engine → load it and attach SL immediately.
        - Engine has pending_trades but no position on exchange → clear (closed externally).
        Only runs in real-trading mode and only for the runner's symbol.
        """
        if not REAL_TRADING:
            return
        try:
            row = get_exchange(self.email)
            if not row:
                return
            ex     = _make_ccxt_exchange(row)
            ex_id  = (row.get("exchange") or "bybit").lower()
            positions = _ccxt_call(ex.fetch_positions, [self.symbol],
                                   label=f"reconcile fetch_positions {self.symbol}", retries=0)

            # Symbol format mismatch guard:
            # ccxt normalises position symbols to its unified format (e.g. "NEAR/USDT:USDT")
            # but self.symbol is stored in the native exchange format (e.g. "NEARUSDT").
            # A plain == compare always fails → reconcile incorrectly sees "no position"
            # and wipes the engine state on every restart. We check BOTH fields:
            #   p["symbol"]          → ccxt unified format  ("NEAR/USDT:USDT")
            #   p["info"]["symbol"]  → native exchange format ("NEARUSDT")
            _sym_uc = self.symbol.upper()
            def _pos_match(p: dict) -> bool:
                unified = (p.get("symbol") or "").upper()
                native  = ((p.get("info") or {}).get("symbol") or "").upper()
                has_size = abs(float(p.get("contracts") or 0)) > 0
                return has_size and (unified == _sym_uc or native == _sym_uc)
            live_pos = next((p for p in positions if _pos_match(p)), None)

            if live_pos and not self.pending_trades:
                # Position exists on exchange but engine is blind — recover it.
                # Use Bybit's actual stopLoss price if set — avoids widening a tight
                # breakeven/trailing SL that was already protecting profit.
                contracts  = abs(float(live_pos.get("contracts") or 0))
                entry_px   = float(live_pos.get("entryPrice") or live_pos.get("averagePrice") or 0)
                pos_side   = "LONG" if (live_pos.get("side") or "").lower() == "long" else "SHORT"

                # Try to read the exchange-side SL price from Bybit position data.
                # Bybit returns it as info["stopLoss"] (string). Fall back to 1.5% default
                # only if the exchange has no SL set at all.
                _exch_sl_str = (live_pos.get("info") or {}).get("stopLoss") or ""
                _exch_sl_px  = float(_exch_sl_str) if _exch_sl_str and float(_exch_sl_str) > 0 else 0.0
                if _exch_sl_px > 0 and entry_px > 0:
                    sl_pct_use = abs(_exch_sl_px - entry_px) / entry_px
                    sl_price   = _exch_sl_px
                    self.log(
                        f"RECONCILE: Using existing exchange SL @ {sl_price:.4f} "
                        f"({sl_pct_use*100:.2f}% from entry) — not overwriting."
                    )
                else:
                    sl_pct_use = 0.015  # 1.5% default — no SL was set on exchange
                    sl_price   = entry_px * (1 - sl_pct_use) if pos_side == "LONG" else entry_px * (1 + sl_pct_use)

                self.log(
                    f"RECONCILE: Found untracked {pos_side} {self.symbol} "
                    f"{contracts} contracts @ {entry_px:.4f} on {ex_id}. "
                    f"Loading into engine and attaching SL @ {sl_price:.4f}."
                )
                self.pending_trades = [{
                    "entry_price":  entry_px,
                    "side":         pos_side,
                    "mode":         self.mode,
                    "signal":       "RECOVERED",
                    "open_ts":      time.time(),
                    "sl_pct_open":  sl_pct_use,
                    "tp_pct_open":  sl_pct_use * 2,
                    "grade":        "A",
                    "size_mult":    1.00,
                    "tp_mult":      1.00,
                    "is_primary":   True,
                    "recovered":    True,
                }]
                _save_runner_state(self)
                # Only call _ensure_sl_or_close if the exchange has no SL set.
                # If _exch_sl_px > 0 the position is already protected — don't overwrite it.
                if _exch_sl_px <= 0:
                    _ensure_sl_or_close(self, ex, ex_id, self.symbol, pos_side, sl_price)

            elif not live_pos and self.pending_trades:
                # Engine thinks a position is open but exchange has none.
                # Trade closed during downtime (restart/deploy gap) — SL or TP was hit on
                # the exchange while the bot was offline. Sync real balance from Bybit so
                # peak/floor equity calculations stay correct after the missed P&L.
                self.log(
                    f"RECONCILE: Engine had {len(self.pending_trades)} pending trade(s) "
                    f"but no open position on {ex_id} for {self.symbol}. "
                    f"Trade likely closed during restart gap — syncing real balance."
                )
                try:
                    real_bal  = _get_available_usdt(ex, ex_id)
                    if real_bal > 0:
                        old_eq = get_equity(self.email)
                        set_equity(self.email, real_bal)
                        self.equity = real_bal
                        update_peak_ath(self.email, real_bal, real_bal)
                        self.log(
                            f"RECONCILE: Equity synced from {ex_id} → ${real_bal:.2f} USDT "
                            f"(was ${old_eq:.2f} before sync)."
                        )
                except Exception as _bal_e:
                    self.log(f"RECONCILE: Balance sync failed (non-fatal): {_bal_e}")
                self.pending_trades = []
                _save_runner_state(self)

        except Exception as e:
            self.log(f"RECONCILE error (non-fatal): {e}")

    def _run_loop(self):
        cooldown_sec = max(60, self.interval_sec)
        _last_reconcile_ts = time.time()   # startup reconcile runs via _restore_live_state thread
        while not self.stop_event.is_set():
            try:
                try:
                    self._reset_if_new_day()
                except Exception as _mrf_err:
                    _tg_alert(
                        f"⚠️ <b>Midnight reset failed</b>\n"
                        f"{self.email} | {self.symbol}\n"
                        f"<code>{str(_mrf_err)[:200]}</code>"
                    )
                self.last_run_ts = time.time()

                # FIX 3: Reconcile exchange positions every hour
                if REAL_TRADING and (time.time() - _last_reconcile_ts) >= 3600:
                    self._reconcile_exchange_positions()
                    _last_reconcile_ts = time.time()

                if self.end_at_ts and time.time() >= self.end_at_ts:
                    self.blocked_reason = "DURATION_ENDED"
                    if self.pending_trades:
                        self._close_pending_trades()
                    self.log("Session complete — duration ended. AI stopped.")
                    self.stop_event.set()
                    email_ai_stopped(self.email, self.symbol, "DURATION_END", get_equity(self.email))
                    break

                if not get_exchange(self.email):
                    self.blocked_reason = "EXCHANGE_NOT_CONNECTED"
                    self.last_signal = "BLOCKED: EXCHANGE_NOT_CONNECTED"
                    self.log("Blocked: exchange not connected.")
                    continue

                # ── Step 1: Close any pending trades first (real exit after one interval) ──
                if self.pending_trades:
                    self._close_pending_trades()
                    continue

                # ── Step 2a: Hard floor + max drawdown guard ─────────────────────────────
                current_equity = get_equity(self.email)
                if current_equity < self.floor_equity:
                    self.log(
                        f"HARD FLOOR — equity ${current_equity:.2f} is below 85% of start "
                        f"(${self.floor_equity:.2f}). AI stopped to protect remaining capital."
                    )
                    self.blocked_reason = "HARD_FLOOR"
                    self.stop_event.set()
                    email_ai_stopped(self.email, self.symbol, "HARD_FLOOR", current_equity)
                    _tg_alert(
                        f"🛑 <b>Hard floor hit — AI stopped</b>\n"
                        f"{self.email} | {self.symbol}\n"
                        f"Equity <b>${current_equity:.2f}</b> fell below floor <b>${self.floor_equity:.2f}</b>"
                    )
                    if _SENTRY_DSN:
                        with sentry_sdk.new_scope() as scope:
                            scope.set_user({"email": self.email})
                            scope.set_tag("symbol", self.symbol)
                            scope.set_tag("mode", self.mode)
                            scope.set_level("warning")
                            sentry_sdk.capture_message(
                                f"HARD_FLOOR triggered: {self.email} | equity ${current_equity:.2f} < floor ${self.floor_equity:.2f}"
                            )
                    try:
                        with db_conn() as _hf_conn:
                            _hf_conn.cursor().execute(
                                "UPDATE user_state SET last_stop_reason=%s, "
                                "peak_equity=%s, floor_equity=%s WHERE email=%s",
                                ("HARD_FLOOR", self.peak_equity, self.floor_equity, self.email),
                            )
                            _hf_conn.commit()
                    except Exception as _hfl_db_err:
                        _tg_alert(
                            f"⚠️ <b>DB write failed — hard floor state not saved</b>\n"
                            f"{self.email} | {self.symbol}\n"
                            f"<code>{str(_hfl_db_err)[:200]}</code>"
                        )
                    break

                if self.peak_equity > 0:
                    dd = (self.peak_equity - current_equity) / self.peak_equity
                    if dd >= 0.15:
                        self.log(
                            f"MAX DRAWDOWN — equity down {dd*100:.1f}% from peak "
                            f"(${self.peak_equity:.2f} → ${current_equity:.2f}). AI stopped."
                        )
                        self.blocked_reason = "MAX_DRAWDOWN"
                        self.stop_event.set()
                        email_ai_stopped(self.email, self.symbol, "MAX_DRAWDOWN", current_equity)
                        break

                # ── Step 2b: Daily limit checks ─────────────────────────────────────────
                if self.max_trades_per_day > 0 and self.trades_today >= self.max_trades_per_day:
                    self.blocked_reason = f"MAX_TRADES_DAY: {self.trades_today}/{self.max_trades_per_day}"
                    self.last_signal = "BLOCKED: MAX_TRADES_DAY"
                    if self.duration_days > 0:
                        self._sleep_until_dubai_midnight()
                    else:
                        self.log(f"Max trades/day reached ({self.trades_today}/{self.max_trades_per_day}). Stopping.")
                        self.stop_event.set()
                        break
                    continue

                if self.stop_after_bad_trades > 0 and self.bad_trades_today >= self.stop_after_bad_trades:
                    self.blocked_reason = f"MAX_BAD_TRADES: {self.bad_trades_today}/{self.stop_after_bad_trades}"
                    self.last_signal = "BLOCKED: MAX_BAD_TRADES"
                    if self.duration_days > 0:
                        self._sleep_until_dubai_midnight()
                    else:
                        self.log(f"Bad trade limit reached ({self.bad_trades_today}/{self.stop_after_bad_trades}). Stopping.")
                        self.stop_event.set()
                        email_ai_stopped(self.email, self.symbol, "MAX_BAD_TRADES", get_equity(self.email))
                        break
                    continue

                # ── Step 3: 4-layer signal analysis ─────────────────────────────────────
                res = self._signal_and_filters()

                # Retry up to 3× on NO_DATA — price feed may just be transiently down
                if res.get("blocked") == "NO_DATA":
                    recovered = False
                    for attempt in range(1, 4):
                        self.log(f"Price data unavailable — retrying in 30s (attempt {attempt} of 3)")
                        for _ in range(30):
                            if self.stop_event.is_set():
                                break
                            time.sleep(1)
                        if self.stop_event.is_set():
                            break
                        res = self._signal_and_filters()
                        if res.get("blocked") != "NO_DATA":
                            self.log(f"Price data recovered on attempt {attempt} — continuing normally.")
                            recovered = True
                            break
                    if not recovered:
                        self.log("Price data unavailable after 3 retries — skipping this interval. Will resume next scheduled check.")
                        self.blocked_reason = "NO_DATA"
                        continue

                self.last_signal = res.get("signal") or "-"
                # Always reflect the HTF-computed direction, even when blocked.
                # Without this, last_side stays at "LONG" init default forever
                # and the status bar shows the wrong side while direction is blocked.
                if res.get("side"):
                    self.last_side = res["side"]
                if not res.get("ok"):
                    self.blocked_reason = res.get("blocked") or "BLOCKED"
                    _side_lbl = res.get("side", "?")
                    _score_lbl = f"{res.get('score', 0.0):.2f}"
                    _raw = res.get("blocked") or "signal not ready"
                    # "BLOCKED (entry): Price 4.3% from EMA21" → "Price 4.3% from EMA21"
                    _reason_lbl = _raw.split("): ", 1)[-1] if "): " in _raw else _raw
                    self.log(f"Market check — {_side_lbl} score={_score_lbl} | {_reason_lbl}")
                    continue

                self.blocked_reason = None
                desired_side: Side = res["side"]

                now_ts = time.time()
                # After a bad trade wait 2× the normal cooldown before entering again
                effective_cooldown = cooldown_sec * 2 if self._last_trade_bad else cooldown_sec
                should_trade = (now_ts - self.last_trade_ts) >= effective_cooldown

                if should_trade:
                    # ── Step 4: Open trade(s) — Grade A = 1 trade, Grade B = T1+T2 ──
                    entry_price = self._fetch_price_sync(self.symbol)
                    self._last_trade_bad = False
                    grade = self.market_grade
                    c = presets_for_mode(self.mode)
                    st = TRADE_STYLE_PARAMS.get(self.trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
                    # T16 reversal overrides SL/TP multipliers (1.5×ATR SL, 3×ATR TP)
                    _sl_atr = self.sl_atr_override if self.sl_atr_override else st["sl_atr"]
                    _tp_atr = self.tp_atr_override if self.tp_atr_override else st["tp_atr"]

                    # ── BTC Correlation filter (all symbols except BTC itself) ───────
                    _btc_mult = 1.0
                    if "BTC" not in self.symbol.upper():
                        _btc_move = _get_btc_momentum(self.tf)
                        if _btc_move is not None:
                            _btc_against = ((desired_side == "LONG"  and _btc_move < -0.02) or
                                            (desired_side == "SHORT" and _btc_move >  0.02))
                            if _btc_against:
                                _btc_dir = f"BTC {_btc_move*100:.1f}% vs {desired_side}"
                                if abs(_btc_move) >= 0.04:
                                    self.log(f"BTC macro conflict: {_btc_dir} — strong headwind, trade skipped")
                                    self.last_trade_ts = now_ts  # apply cooldown so we re-check next interval
                                    continue
                                _btc_mult = 0.60
                                self.log(f"BTC macro conflict: {_btc_dir} — size reduced to 60%")
                    sl_pct_open = min(self.last_atr_pct * _sl_atr, st["sl_max"] / 100.0)
                    tp_pct_open = min(self.last_atr_pct * _tp_atr, st["tp_max"] / 100.0)

                    if REAL_TRADING:
                        real_bal = get_real_usdt_balance(self.email, force=True)
                        if real_bal is None:
                            self.log("REAL ORDER SKIPPED — could not fetch exchange balance.")
                            continue
                        equity_now = real_bal
                        set_equity(self.email, real_bal)
                        self.log(f"Exchange balance synced: ${real_bal:.2f} USDT")
                    else:
                        equity_now = get_equity(self.email)

                    real_order_id  = None
                    real_b_result  = None   # Grade B real only
                    _real_result   = None   # Grade A real only
                    _mtf_sm = res.get("mtf_size_mult", 1.0)
                    if not res.get("mtf_confirmed", True):
                        self.log(f"15m not confirmed — size reduced to {_mtf_sm*100:.0f}% for lower conviction")
                    if REAL_TRADING:
                        try:
                            # Apply vol and drawdown size adjustments to the real order —
                            # mirrors the paper calculation in _close_one_trade (effective_size).
                            # Without this, Bybit gets the full c["size"] (e.g. 65% of equity
                            # for MINI_ASYM) even when high volatility reduces paper size to 40%.
                            # That caused real positions to be 2–6× larger than paper expected.
                            _atr_bl = {"15m": 0.0040, "1h": 0.0090, "4h": 0.0180, "1d": 0.0350}
                            _atr_nm = _atr_bl.get(self.tf, 0.0050)
                            _vol_sm = (max(0.4, _atr_nm / self.last_atr_pct)
                                       if self.last_atr_pct > _atr_nm else 1.0)
                            _dd_now = (max(0.0, (self.peak_equity - equity_now) / self.peak_equity)
                                       if self.peak_equity > 0 else 0.0)
                            _dd_sm  = (0.25 if _dd_now >= 0.10 else
                                       0.40 if _dd_now >= 0.07 else
                                       0.65 if _dd_now >= 0.04 else 1.0)
                            # Funding rate check — reduce size if rate works against direction
                            _fund_mult = 1.0
                            _fr = _get_funding_rate(self.email, self.symbol)
                            if _fr is not None:
                                _fr_against = ((desired_side == "LONG"  and _fr >  0.001) or
                                               (desired_side == "SHORT" and _fr < -0.001))
                                if _fr_against:
                                    _fr_abs   = abs(_fr)
                                    _fund_mult = 0.50 if _fr_abs >= 0.003 else 0.70
                                    self.log(f"Funding rate {_fr*100:.4f}%/8h against {desired_side} — size × {_fund_mult:.2f}")
                            size_pct  = float(c["size"]) * _mtf_sm * _vol_sm * _dd_sm * _btc_mult * _fund_mult
                            usdt_size = equity_now * size_pct
                            self.log(
                                f"REAL ORDER SIZE: base={float(c['size'])*100:.0f}% "
                                f"× mtf={_mtf_sm:.2f} × vol={_vol_sm:.2f} × dd={_dd_sm:.2f}"
                                f"× btc={_btc_mult:.2f} × fund={_fund_mult:.2f} "
                                f"= {size_pct*100:.1f}% → ${usdt_size:.2f}"
                            )
                            if grade == "B":
                                # Real Grade B: entry + two partial reduceOnly TP limit orders
                                real_b_result = place_real_grade_b_order(
                                    email=self.email, symbol=self.symbol, side=desired_side,
                                    usdt_size=usdt_size, leverage=int(c["leverage"]),
                                    sl_pct=sl_pct_open,
                                    t1_tp_pct=tp_pct_open * 0.80,
                                    t2_tp_pct=tp_pct_open * 1.00,
                                )
                                self.log(
                                    f"REAL GRADE B | entry={real_b_result['order_id']} "
                                    f"T1_TP={real_b_result['t1_tp_id']} T2_TP={real_b_result['t2_tp_id']} "
                                    f"SL={real_b_result['sl_price']:.4f} "
                                    f"T1@{real_b_result['t1_tp_price']:.4f} T2@{real_b_result['t2_tp_price']:.4f}"
                                )
                            else:
                                # T18: open at 70% so we can scale into winners
                                _t18_active = self._scaling_enabled
                                _open_usdt  = usdt_size * (0.70 if _t18_active else 1.00)
                                _real_result = place_real_order(
                                    email=self.email, symbol=self.symbol, side=desired_side,
                                    usdt_size=_open_usdt, leverage=int(c["leverage"]),
                                    sl_pct=sl_pct_open, tp_pct=tp_pct_open,
                                )
                                real_order_id = _real_result.get("order_id")
                                _t18_lbl = f" (T18: 70% of ${usdt_size:.2f})" if _t18_active else ""
                                self.log(f"REAL ORDER PLACED | id={real_order_id} | {desired_side} {self.symbol} size=${_open_usdt:.2f} lev={c['leverage']}×{_t18_lbl}")
                        except Exception as _re:
                            self.log(f"REAL ORDER FAILED — trade skipped: {_re}")
                            _tg_alert(
                                f"❌ <b>Real order failed</b>\n"
                                f"{self.email} | {self.symbol} {desired_side}\n"
                                f"<code>{str(_re)[:250]}</code>"
                            )
                            continue

                        # ── FIX 1: Entry succeeded — log position to DB BEFORE SL ──
                        # If SL fails after this point, the position is still tracked.
                        base_trade = {
                            "entry_price": entry_price,
                            "side": desired_side,
                            "mode": self.mode,
                            "signal": self.last_signal,
                            "open_ts": time.time(),
                            "sl_pct_open": sl_pct_open,
                            "tp_pct_open": tp_pct_open,
                            "usdt_size_base": usdt_size,
                            "leverage_used":  int(c["leverage"]),
                        }
                        if grade == "B":
                            self.pending_trades = [
                                {**base_trade, "grade": "B", "size_mult": 0.60, "tp_mult": 0.80,
                                 "is_primary": True,  "label": "T1",
                                 "order_id": real_b_result.get("order_id"),
                                 "t1_tp_id": real_b_result.get("t1_tp_id"),
                                 "t1_sl_id": None},
                                {**base_trade, "grade": "B", "size_mult": 0.40, "tp_mult": 1.00,
                                 "is_primary": False, "label": "T2", "breakeven_after_t1": True,
                                 "t2_sl_id": None},
                            ]
                            self.log(f"TRADE OPENED Grade B REAL ({desired_side}) @ {entry_price:.4f} | T1 60%+80%TP T2 40%+100%TP+BE | score={self.last_score:.2f}")
                        else:
                            _real_t18   = self._scaling_enabled
                            _real_init_sm = round(0.70 * _mtf_sm if _real_t18 else 1.00 * _mtf_sm, 4)
                            _scale_lbl  = " | T18: entry 70%" if _real_t18 else ""
                            self.pending_trades = [
                                {**base_trade, "grade": "A", "size_mult": _real_init_sm, "tp_mult": 1.00,
                                 "is_primary": True, "order_id": real_order_id,
                                 "scale1_done": False, "scale2_done": False,
                                 "scaling_enabled": _real_t18},
                            ]
                            self.log(f"TRADE OPENED Grade A REAL ({desired_side}) @ {entry_price:.4f} | score={self.last_score:.2f}{_scale_lbl}")
                        _sl_px = entry_price * (1 + sl_pct_open) if desired_side == "SHORT" else entry_price * (1 - sl_pct_open)
                        _tp_px = entry_price * (1 - tp_pct_open) if desired_side == "SHORT" else entry_price * (1 + tp_pct_open)
                        try:
                            email_trade_opened(
                                to=self.email, symbol=self.symbol, side=desired_side, mode=self.mode,
                                grade=grade, entry=entry_price, sl=_sl_px, tp=_tp_px,
                                score=self.last_score, equity=equity_now,
                            )
                        except Exception:
                            pass
                        self.last_trade_ts = now_ts
                        _save_runner_state(self)   # position logged to DB before SL attempt

                        # ── Place SL — Grade B: two separate stops (T1+T2 independent)
                        #              Grade A: single position-level SL ──────────────
                        if grade == "B":
                            _b_ex    = real_b_result["ex"]
                            _b_ex_id = real_b_result["ex_id"]
                            _b_sl    = real_b_result["sl_price"]
                            if _b_ex_id == "bybit":
                                # Each stop covers only its own portion.
                                # When T1 SL fires (60% closes), T2 SL stays active.
                                _t1_sl_id = _place_grade_b_stop_order(
                                    self, _b_ex, _b_ex_id, self.symbol, desired_side,
                                    _b_sl, real_b_result["t1_qty"], "T1_SL",
                                )
                                _t2_sl_id = _place_grade_b_stop_order(
                                    self, _b_ex, _b_ex_id, self.symbol, desired_side,
                                    _b_sl, real_b_result["t2_qty"], "T2_SL",
                                )
                                self.pending_trades[0]["t1_sl_id"] = _t1_sl_id
                                self.pending_trades[1]["t2_sl_id"] = _t2_sl_id
                                _save_runner_state(self)
                                if not _t1_sl_id or not _t2_sl_id:
                                    # Partial failure — apply position SL as fallback
                                    self.log(f"Grade B partial stop failed (T1={_t1_sl_id} T2={_t2_sl_id}) — applying position SL as fallback")
                                    _sl_ok = _ensure_sl_or_close(self, _b_ex, _b_ex_id, self.symbol, desired_side, _b_sl)
                                    if not _sl_ok:
                                        self.pending_trades = []
                                        _save_runner_state(self)
                                    else:
                                        # Bug 2 fix: position-level SL (tradingStop) can cancel
                                        # existing reduce-only limit orders on some Bybit configs.
                                        # Re-verify T1_TP and T2_TP are still active after fallback SL.
                                        _t1_tp_id_chk = real_b_result.get("t1_tp_id")
                                        _t2_tp_id_chk = real_b_result.get("t2_tp_id")
                                        if _t1_tp_id_chk or _t2_tp_id_chk:
                                            try:
                                                _chk_open = _ccxt_call(
                                                    _b_ex.fetch_open_orders, self.symbol,
                                                    label="verify_tp_after_fallback_sl", retries=0,
                                                )
                                                _open_ids = {(o.get("id") or "") for o in _chk_open}
                                                _open_ids |= {(o.get("info") or {}).get("orderId", "") for o in _chk_open}
                                                _missing = []
                                                if _t1_tp_id_chk and _t1_tp_id_chk not in _open_ids:
                                                    _missing.append(("T1_TP", _t1_tp_id_chk))
                                                if _t2_tp_id_chk and _t2_tp_id_chk not in _open_ids:
                                                    _missing.append(("T2_TP", _t2_tp_id_chk))
                                                if _missing:
                                                    self.log(
                                                        f"Bug 2 guard: fallback SL cancelled TP orders "
                                                        f"{[m[0] for m in _missing]} — re-placing them"
                                                    )
                                                    _close_side_repl = "sell" if desired_side == "LONG" else "buy"
                                                    _rp_params = {"reduceOnly": True, "positionIdx": 0}
                                                    for _tp_label, _old_tp_id in _missing:
                                                        _tp_price_repl = (
                                                            real_b_result["t1_tp_price"] if _tp_label == "T1_TP"
                                                            else real_b_result["t2_tp_price"]
                                                        )
                                                        _tp_qty_repl = (
                                                            real_b_result["t1_qty"] if _tp_label == "T1_TP"
                                                            else real_b_result["t2_qty"]
                                                        )
                                                        try:
                                                            _new_tp = _ccxt_call(
                                                                _b_ex.create_order,
                                                                self.symbol, "limit",
                                                                _close_side_repl, _tp_qty_repl,
                                                                _tp_price_repl,
                                                                params=_rp_params,
                                                                label=f"replace_{_tp_label}",
                                                            )
                                                            _new_id = _new_tp.get("id")
                                                            self.log(f"{_tp_label} re-placed: order={_new_id} @ {_tp_price_repl:.4f}")
                                                            if _tp_label == "T1_TP" and self.pending_trades:
                                                                self.pending_trades[0]["t1_tp_id"] = _new_id
                                                            elif _tp_label == "T2_TP" and len(self.pending_trades) > 1:
                                                                self.pending_trades[1]["t2_tp_id"] = _new_id
                                                        except Exception as _rpe:
                                                            self.log(f"WARN: {_tp_label} re-place failed: {_rpe}")
                                                    _save_runner_state(self)
                                                else:
                                                    self.log("TP orders verified active after fallback SL placement ✓")
                                            except Exception as _tpv_err:
                                                self.log(f"TP verify after fallback SL: non-fatal error: {_tpv_err}")
                            else:
                                # OKX/Binance: position-level SL (no qty-specific conditional stops)
                                _sl_ok = _ensure_sl_or_close(self, _b_ex, _b_ex_id, self.symbol, desired_side, _b_sl)
                                if not _sl_ok:
                                    self.pending_trades = []
                                    _save_runner_state(self)
                        else:
                            _sl_ok = _ensure_sl_or_close(
                                self, _real_result["ex"], _real_result["ex_id"],
                                self.symbol, desired_side, _real_result["sl_price"],
                            )
                            if not _sl_ok:
                                self.pending_trades = []
                                _save_runner_state(self)

                    else:
                        # ── Demo / paper trading — no real orders ──
                        base_trade = {
                            "entry_price": entry_price,
                            "side": desired_side,
                            "mode": self.mode,
                            "signal": self.last_signal,
                            "open_ts": time.time(),
                            "sl_pct_open": sl_pct_open,
                            "tp_pct_open": tp_pct_open,
                        }
                        _mtf_label = "" if res.get("mtf_confirmed", True) else f" | 15m unconfirmed→size {_mtf_sm*100:.0f}%"
                        if grade == "B":
                            self.pending_trades = [
                                {**base_trade, "grade": "B", "size_mult": round(0.60 * _mtf_sm, 4), "tp_mult": 0.80,
                                 "is_primary": True,  "label": "T1"},
                                {**base_trade, "grade": "B", "size_mult": round(0.40 * _mtf_sm, 4), "tp_mult": 1.00,
                                 "is_primary": False, "label": "T2", "breakeven_after_t1": True},
                            ]
                            self.log(f"TRADE OPENED Grade B ({desired_side}) @ {entry_price:.4f} | T1 60%+80%TP, T2 40%+100%TP+BE | score={self.last_score:.2f}{_mtf_label}")
                        else:
                            # Scaling enabled: open at 70% to allow scale-ins. Builds to 100% on winners.
                            _grade_a_sm = round((0.70 if self._scaling_enabled else 1.00) * _mtf_sm, 4)
                            _scale_label = " | scaling: entry 70%" if self._scaling_enabled else ""
                            self.pending_trades = [
                                {**base_trade, "grade": "A", "size_mult": _grade_a_sm, "tp_mult": 1.00,
                                 "is_primary": True,
                                 "scale1_done": False, "scale2_done": False,
                                 "scaling_enabled": self._scaling_enabled},
                            ]
                            self.log(f"TRADE OPENED Grade A ({desired_side}) @ {entry_price:.4f} | score={self.last_score:.2f} | signal={self.last_signal}{_mtf_label}{_scale_label}")
                        _sl_px = entry_price * (1 + sl_pct_open) if desired_side == "SHORT" else entry_price * (1 - sl_pct_open)
                        _tp_px = entry_price * (1 - tp_pct_open) if desired_side == "SHORT" else entry_price * (1 + tp_pct_open)
                        try:
                            email_trade_opened(
                                to=self.email, symbol=self.symbol, side=desired_side, mode=self.mode,
                                grade=grade, entry=entry_price, sl=_sl_px, tp=_tp_px,
                                score=self.last_score, equity=equity_now,
                            )
                        except Exception:
                            pass
                        self.last_trade_ts = now_ts
                        _save_runner_state(self)   # persist open position immediately
                else:
                    wait_sec = int(effective_cooldown - (now_ts - self.last_trade_ts))
                    cd_label = "2× cooldown (after loss)" if self._last_trade_bad else "cooldown"
                    self.log(f"Signal OK (score={self.last_score:.2f}) | {cd_label} — {wait_sec}s remaining")

            except Exception as e:
                self.blocked_reason = "ERROR"
                self.last_signal = f"ERROR: {str(e)[:120]}"
                self.log(self.last_signal)
                _err_str = str(e)
                if "EXCHANGE_AUTH_ERROR" in _err_str:
                    _tg_alert(
                        f"🔑 <b>API key rejected by exchange</b>\n"
                        f"{self.email} | {self.symbol}\n"
                        f"<code>{_err_str[:300]}</code>"
                    )
                else:
                    _tg_alert(
                        f"🔴 <b>Engine crash</b>\n"
                        f"{self.email} | {self.symbol}\n"
                        f"<code>{_err_str[:300]}</code>"
                    )
                if _SENTRY_DSN:
                    with sentry_sdk.new_scope() as scope:
                        scope.set_user({"email": self.email})
                        scope.set_tag("symbol", self.symbol)
                        scope.set_tag("mode", self.mode)
                        sentry_sdk.capture_exception(e)
            finally:
                # Sleep in 30-second chunks so the loop wakes up quickly when
                # the mid-candle monitor opens a trade and sets pending_trades.
                _rem = self.interval_sec
                while _rem > 0 and not self.stop_event.is_set():
                    time.sleep(min(30, _rem))
                    _rem -= 30
                    if self.pending_trades:   # mid-candle trade opened — process it now
                        break

        # ── Loop exited — engine stopped (drawdown / duration / bad trades / user stop) ──
        # Clear persisted state so this runner is NOT resumed on the next deploy.
        _clear_runner_state(self.email)
        # Remove from global registry so status endpoints return running=False
        with AUTO_LOCK:
            if AUTO_RUNNERS.get(self.email) is self:
                del AUTO_RUNNERS[self.email]


def _resume_runners_on_startup() -> None:
    """
    Called once on backend startup. Re-creates any AutoRunner that was active
    before the last deploy/restart. Skips the restart lock (it was a server
    restart, not a user choice to stop+restart).
    Duration sessions: if the scheduled end_at_ts has already passed, skip them
    (session is over). Otherwise resume with remaining duration.
    IMPORTANT: never call _clear_runner_state here except for genuinely expired
    sessions — any other failure just skips silently so state is preserved for
    the next restart attempt.
    """
    # Give the DB connection pool time to fully initialise on Render before
    # we try to read state and reconstruct runners.
    time.sleep(8)

    rows = _load_all_runner_states()
    if not rows:
        print("[startup-resume] no saved runner states found")
        return

    now_ts = time.time()
    resumed = 0
    for row in rows:
        email = row["email"]
        try:
            # Skip if end_at_ts is in the past (duration session that already expired)
            end_at_ts = row.get("end_at_ts")
            if end_at_ts and float(end_at_ts) <= now_ts:
                _clear_runner_state(email)
                print(f"[startup-resume] {email} — duration session expired, state cleared")
                continue

            # Skip without clearing if exchange is not connected — it may be a
            # transient startup timing issue; preserving state lets the next
            # redeploy retry successfully.
            if not get_exchange(email):
                print(f"[startup-resume] {email} — no exchange connected, skipping (state preserved)")
                continue

            runner = AutoRunner(
                email=email,
                symbol=row["symbol"],
                trade_style=row["trade_style"],
                mode=row["mode"],
                max_trades_per_day=int(row["max_trades_per_day"]),
                stop_after_bad_trades=int(row["stop_after_bad_trades"]),
                duration_days=int(row["duration_days"]),
                trend_filter=bool(row["trend_filter"]),
                chop_min_sep_pct=float(row["chop_min_sep_pct"]),
            )

            # Restore the original end timestamp so duration sessions expire correctly
            if end_at_ts:
                runner.end_at_ts = float(end_at_ts)

            with AUTO_LOCK:
                # Don't clobber a runner that somehow already exists
                if email not in AUTO_RUNNERS:
                    AUTO_RUNNERS[email] = runner
                    runner.start()
                    resumed += 1
                    print(f"[startup-resume] {email} — resumed {row['symbol']} {row['mode']} {row['trade_style']}")

        except Exception as e:
            # Log but do NOT clear state — preserve it so the next redeploy can retry
            print(f"[startup-resume] {email} — error (state preserved): {e}")

    print(f"[startup-resume] done — {resumed} AI runner(s) resumed after restart")
    return resumed


def _resume_watchdog() -> None:
    """
    Runs after the initial startup resume. Retries any saved runner states
    that didn't start (e.g. because exchange check failed at boot time).
    Checks every 2 minutes for up to 20 minutes, then gives up.
    """
    max_attempts = 10
    interval_sec = 120

    for attempt in range(1, max_attempts + 1):
        time.sleep(interval_sec)

        rows = _load_all_runner_states()
        if not rows:
            print("[watchdog] no saved states — watchdog exiting")
            return

        now_ts = time.time()
        pending = []
        with AUTO_LOCK:
            for row in rows:
                email = row["email"]
                # Already running — no action needed
                if email in AUTO_RUNNERS and AUTO_RUNNERS[email].is_running():
                    continue
                # Expired duration session — skip
                end_at_ts = row.get("end_at_ts")
                if end_at_ts and float(end_at_ts) <= now_ts:
                    continue
                pending.append(row)

        if not pending:
            print("[watchdog] all runners active — watchdog exiting")
            return

        print(f"[watchdog] attempt {attempt}/{max_attempts} — {len(pending)} runner(s) not yet started")
        resumed = 0
        for row in pending:
            email = row["email"]
            try:
                if not get_exchange(email):
                    print(f"[watchdog] {email} — exchange still not ready, will retry")
                    continue

                runner = AutoRunner(
                    email=email,
                    symbol=row["symbol"],
                    trade_style=row["trade_style"],
                    mode=row["mode"],
                    max_trades_per_day=int(row["max_trades_per_day"]),
                    stop_after_bad_trades=int(row["stop_after_bad_trades"]),
                    duration_days=int(row["duration_days"]),
                    trend_filter=bool(row["trend_filter"]),
                    chop_min_sep_pct=float(row["chop_min_sep_pct"]),
                )
                end_at_ts = row.get("end_at_ts")
                if end_at_ts:
                    runner.end_at_ts = float(end_at_ts)

                with AUTO_LOCK:
                    if email not in AUTO_RUNNERS or not AUTO_RUNNERS[email].is_running():
                        AUTO_RUNNERS[email] = runner
                        runner.start()
                        resumed += 1
                        print(f"[watchdog] {email} — resumed {row['symbol']} {row['mode']} {row['trade_style']}")

            except Exception as e:
                print(f"[watchdog] {email} — error: {e}")
                _tg_alert(
                    f"🚨 <b>Engine failed to start</b>\n"
                    f"User: {email}\n"
                    f"Error: <code>{str(e)[:250]}</code>"
                )

        if resumed:
            print(f"[watchdog] {resumed} runner(s) started this attempt")

    print("[watchdog] max attempts reached — giving up (state preserved in DB)")


# =========================
# TRADING QUALITY HELPERS
# =========================

def _get_btc_momentum(tf: str) -> Optional[float]:
    """BTC % change over last 3 closed candles on given timeframe.
    Positive = rising, negative = falling. Returns None on error."""
    try:
        klines = _fetch_klines_sync("BTCUSDT", tf, limit=6)
        if not klines or len(klines) < 3:
            return None
        old_close = float(klines[-3]["close"])
        new_close = float(klines[-1]["close"])
        return (new_close - old_close) / old_close
    except Exception:
        return None


def _get_funding_rate(email: str, symbol: str) -> Optional[float]:
    """Current funding rate for a perp future as a decimal (e.g. 0.0001 = 0.01%).
    Positive = longs pay (costly for LONG). Negative = shorts pay (costly for SHORT).
    Returns None on any error."""
    try:
        row = get_exchange(email)
        if not row:
            return None
        ex = _make_ccxt_exchange(row)
        if not ex.markets:
            ex.load_markets()
        result = _ccxt_call(ex.fetch_funding_rate, symbol, label=f"funding_{symbol}", retries=0)
        if result:
            return float(result.get("fundingRate") or 0.0)
        return None
    except Exception:
        return None


def _place_real_scalein(runner, symbol: str, side: str, add_usdt: float, leverage: int) -> bool:
    """Add to an existing real position (T18 scale-in).
    Places a market order in same direction — no TP/SL since position already has them.
    Returns True on success."""
    try:
        row = get_exchange(runner.email)
        if not row:
            return False
        ex = _make_ccxt_exchange(row)
        if not ex.markets:
            ex.load_markets()
        ex_id  = (row.get("exchange") or "bybit").lower()
        mkt    = ex.market(symbol)
        _prec  = mkt.get("precision", {})
        _lims  = mkt.get("limits", {})
        qty_step = float(_prec.get("amount") or 0)
        min_qty  = float((_lims.get("amount") or {}).get("min") or 0)

        current_px = runner._fetch_price_sync(symbol)
        qty_raw    = (add_usdt * leverage) / max(current_px, 1e-9)
        qty = (round(int(qty_raw / qty_step) * qty_step, 8)
               if qty_step > 0 else round(qty_raw, 3))
        if qty < max(min_qty, 1e-9):
            runner.log(f"SCALE-IN skipped — qty {qty:.4f} below exchange minimum {min_qty}")
            return False

        if ex_id == "bybit":
            bybit_sym = mkt["id"]
            _ccxt_call(
                ex.privatePostV5OrderCreate,
                {
                    "category":    "linear",
                    "symbol":      bybit_sym,
                    "side":        "Buy" if side == "LONG" else "Sell",
                    "orderType":   "Market",
                    "qty":         str(qty),
                    "positionIdx": 0,
                },
                label=f"scalein_{symbol}", retries=1,
            )
        else:
            ccxt_side = "buy" if side == "LONG" else "sell"
            _ccxt_call(
                ex.create_order, symbol, "market", ccxt_side, qty, None,
                {"reduceOnly": False},
                label=f"scalein_{symbol}", retries=1,
            )
        runner.log(f"REAL SCALE-IN placed: {side} {symbol} +${add_usdt:.2f} qty={qty:.4f} lev={leverage}×")
        return True
    except Exception as e:
        runner.log(f"REAL SCALE-IN failed — {e}")
        return False


# =========================
# REAL ORDER PLACEMENT (Bybit/OKX via ccxt)
# =========================

def place_real_order(
    email: str,
    symbol: str,
    side: str,       # "LONG" | "SHORT"
    usdt_size: float,
    leverage: int,
    sl_pct: float,
    tp_pct: float,
) -> dict:
    """
    Place a real leveraged market order with SL/TP on Bybit.
    Returns order details or raises on failure.

    SAFETY: This function is only called when IS_PROD=True AND
    the user has explicitly enabled live trading in their settings.
    Paper trading uses _place_trade_internal() instead.
    """
    row = get_exchange(email)
    if not row:
        raise ValueError("No exchange connected")

    ex = _make_ccxt_exchange(row)
    ex_id = (row.get("exchange") or "bybit").lower()

    # Convert LONG/SHORT to buy/sell
    order_side = "buy" if side == "LONG" else "sell"
    sl_mult    = (1 - sl_pct) if side == "LONG" else (1 + sl_pct)
    tp_mult_v  = (1 + tp_pct) if side == "LONG" else (1 - tp_pct)

    # Cancel any orphaned open orders before placing a new one.
    # Previous failed Grade B attempts can leave T1/T2 limit orders that lock margin.
    try:
        open_orders = _ccxt_call(ex.fetch_open_orders, symbol, label=f"pre_order_cleanup {ex_id}", retries=0)
        for o in open_orders:
            try:
                _ccxt_call(ex.cancel_order, o["id"], symbol, label="cancel_orphan", retries=0)
                print(f"[order-cleanup] Cancelled orphaned order {o['id']} on {symbol}", flush=True)
            except RuntimeError:
                pass
    except RuntimeError:
        pass

    # Get current price — 10s timeout, 1 retry
    ticker = _ccxt_call(ex.fetch_ticker, symbol, label=f"fetch_ticker {ex_id}")
    price  = float(ticker["last"])

    # Check AVAILABLE balance (not walletBalance) — Bybit sizes orders against
    # availableToWithdraw, not total equity. Using walletBalance causes error 110007
    # when unrealized PnL or locked margin reduces the actual free amount.
    available = _get_available_usdt(ex, ex_id)
    if available < 2.0:
        raise ValueError(f"SKIP_LOW_MARGIN: available USDT ${available:.2f} too low to open a position — waiting for funds to free up.")
    if usdt_size > available * 0.95:
        usdt_size = round(available * 0.90, 4)  # cap at 90% of truly available
        print(f"[balance-cap] Capped usdt_size to ${usdt_size:.2f} (available: ${available:.2f})", flush=True)

    # Set leverage — 110043 means already correct, safe to continue
    try:
        _ccxt_call(ex.set_leverage, leverage, symbol, label=f"set_leverage {ex_id}")
    except RuntimeError as lev_e:
        err_str = str(lev_e)
        if "110043" in err_str or "leverage not modified" in err_str.lower():
            print(f"[INFO] Leverage already correct at {leverage}× on {symbol} — continuing with order")
        else:
            raise ValueError(f"Leverage set failed ({leverage}× on {symbol}): {lev_e}")

    # Quantity in base currency
    notional = usdt_size * leverage
    qty = round(notional / price, 6)

    # Minimum notional guard
    if notional < 5.0:
        raise ValueError(
            f"INSUFFICIENT_BALANCE: Available ${available:.2f} USDT — notional ${notional:.2f} below $5 minimum. "
            f"Deposit more USDT to your exchange account."
        )

    # SL / TP prices
    sl_price = round(price * sl_mult,   4)
    tp_price = round(price * tp_mult_v, 4)

    # ── Entry order — clean, no SL/TP in params ───────────────────────────────
    # SL is set SEPARATELY via _ensure_sl_or_close() after entry so that:
    # 1. Entry and SL are decoupled — SL failure can be retried without re-entering
    # 2. We always know the position exists before trying to protect it
    # 3. Bybit sometimes accepts entry but rejects inline stopLoss params silently
    if ex_id == "bybit":
        params = {"positionIdx": 0}
    elif ex_id == "okx":
        params = {"tdMode": "cross"}
    else:  # binance
        params = {"reduceOnly": False}

    order = _ccxt_call(
        ex.create_order, symbol, "market", order_side, qty, params=params,
        label=f"create_order {ex_id} {side}",
    )

    # Return entry result — caller (_run_loop) logs position immediately then
    # calls _ensure_sl_or_close() to attach SL with retry logic (FIX 1 + FIX 2).
    # TP is best-effort for Grade A: caller attaches after SL is confirmed.
    return {
        "order_id":  order.get("id"),
        "symbol":    symbol,
        "side":      side,
        "qty":       qty,
        "price":     price,
        "sl_price":  sl_price,
        "tp_price":  tp_price,
        "leverage":  leverage,
        "exchange":  ex_id,
        "ex":        ex,
        "ex_id":     ex_id,
    }


def close_real_order(email: str, symbol: str, side: str) -> dict:
    """Close an open position at market price."""
    row = get_exchange(email)
    if not row:
        raise ValueError("No exchange connected")

    ex = _make_ccxt_exchange(row)
    close_side = "sell" if side == "LONG" else "buy"

    # Fetch the live position so we know the exact contract size to close.
    # Must match on BOTH ccxt unified symbol ("NEAR/USDT:USDT") AND native exchange
    # symbol ("NEARUSDT") — same issue as in _reconcile_exchange_positions.
    # A plain == compare against the native symbol always fails and makes the
    # close think there is no position, leaving it open on the exchange.
    positions = _ccxt_call(ex.fetch_positions, [symbol], label="fetch_positions close")
    _sym_uc = symbol.upper()
    pos = next(
        (p for p in positions
         if ((p.get("symbol") or "").upper() == _sym_uc or
             ((p.get("info") or {}).get("symbol") or "").upper() == _sym_uc)
         and abs(float(p.get("contracts") or 0)) > 0),
        None,
    )
    if not pos:
        return {"ok": False, "note": "No open position found"}

    qty = abs(float(pos["contracts"]))
    order = _ccxt_call(
        ex.create_order, symbol, "market", close_side, qty,
        params={"reduceOnly": True, "positionIdx": 0},
        label="close_order",
    )
    return {"ok": True, "order_id": order.get("id"), "qty": qty}


def _set_position_sl(ex, ex_id: str, symbol: str, side: str, sl_price: float) -> None:
    """
    Set stop-loss on an already-open position.
    Separate from entry so SL is always explicitly verified and set.
    Raises RuntimeError on failure so caller can retry or close.
    """
    # Bug 5 fix: each caller may pass a fresh exchange instance with markets = None.
    # ex.market(symbol) raises "bybit markets not loaded" without this.
    if not ex.markets:
        ex.load_markets()
    if ex_id == "bybit":
        mkt = ex.market(symbol)
        bybit_sym = mkt["id"]
        _ccxt_call(
            ex.privatePostV5PositionTradingStop,
            {
                "category":    "linear",
                "symbol":      bybit_sym,
                "stopLoss":    str(round(sl_price, 4)),
                "slTriggerBy": "MarkPrice",
                "positionIdx": 0,
            },
            label=f"set_sl bybit {symbol}", retries=0,
        )
    elif ex_id == "binance":
        close_side = "sell" if side == "LONG" else "buy"
        _ccxt_call(
            ex.create_order, symbol, "STOP_MARKET", close_side, None, None,
            {"stopPrice": sl_price, "closePosition": True},
            label=f"set_sl binance {symbol}", retries=0,
        )
    else:  # OKX
        close_side = "sell" if side == "LONG" else "buy"
        _ccxt_call(
            ex.create_order, symbol, "market", close_side, None,
            params={"tdMode": "cross", "slTriggerPx": str(sl_price),
                    "slOrdPx": "-1", "reduceOnly": True},
            label=f"set_sl okx {symbol}", retries=0,
        )


def _ensure_sl_or_close(runner: "AutoRunner", ex, ex_id: str,
                         symbol: str, side: str, sl_price: float) -> bool:
    """
    Set SL on open position, retry 3×. If all attempts fail: close position.
    Returns True  = SL placed successfully.
    Returns False = SL failed, position closed for safety.
    Never leaves a naked position.

    Bug 3 pre-check: if price has already moved past the SL level, Bybit will
    reject the tradingStop with retCode 10001 on every retry (wastes 6+ seconds
    and generates confusing log spam). Detect this case immediately and close
    instead of retrying — the position needs to close, not be stopped.
    """
    # Pre-check: if mark price already past SL, skip retries and close immediately.
    # For LONG: SL must be below mark price (can't set SL above current price).
    # For SHORT: SL must be above mark price.
    try:
        _ticker  = _ccxt_call(ex.fetch_ticker, symbol, label=f"sl_precheck {symbol}", retries=0)
        _mark    = float(_ticker.get("last") or _ticker.get("mark") or 0)
        _sl_past = _mark > 0 and (
            (side == "LONG"  and sl_price >= _mark) or
            (side == "SHORT" and sl_price <= _mark)
        )
        if _sl_past:
            runner.log(
                f"SL {sl_price:.4f} already past mark price {_mark:.4f} for {side} "
                f"— price moved through SL while order was being placed. Closing immediately."
            )
            try:
                result = close_real_order(runner.email, symbol, side)
                runner.log(f"Position closed for safety: {result}")
            except Exception as _ce:
                runner.log(
                    f"CRITICAL: price past SL AND close failed: {_ce}. "
                    f"MANUAL ACTION REQUIRED on {ex_id} — open {side} {symbol}."
                )
            return False
    except Exception as _pce:
        runner.log(f"SL pre-check note (non-fatal, will try setting SL): {_pce}")

    for attempt in range(1, 4):
        try:
            _set_position_sl(ex, ex_id, symbol, side, sl_price)
            runner.log(f"SL confirmed at {sl_price:.4f} (attempt {attempt}).")
            return True
        except Exception as sl_err:
            runner.log(f"SL attempt {attempt}/3 failed: {sl_err}")
            if attempt < 3:
                time.sleep(2)

    # All 3 attempts failed — close position immediately, never leave naked
    runner.log("SL PLACEMENT FAILED AFTER 3 ATTEMPTS — closing position immediately for safety.")
    try:
        result = close_real_order(runner.email, symbol, side)
        runner.log(f"Position closed for safety: {result}")
    except Exception as close_err:
        runner.log(
            f"CRITICAL: SL failed AND close failed: {close_err}. "
            f"MANUAL ACTION REQUIRED on {ex_id} — open {side} {symbol} with NO stop-loss."
        )
    return False


def _place_grade_b_stop_order(
    runner: "AutoRunner",
    ex,
    ex_id: str,
    symbol: str,
    side: str,
    sl_price: float,
    qty: float,
    label: str,
) -> Optional[str]:
    """
    Place a qty-specific conditional stop-market order on Bybit for Grade B T1 or T2.

    Why this is needed (Grade B T2 SL bug — confirmed May 23 NEAR trade):
    ─────────────────────────────────────────────────────────────────────
    The old approach used a single position-level SL via tradingStop (covers the
    full remaining position at trigger time). When T1 closes at SL, that single
    SL order is consumed. T2 continues running but has NO stop protection.
    Result: T2 loss was $2.00 instead of the expected $0.36.

    This function places ONE partial stop order for the given qty (T1=60% or T2=40%).
    The two stops are independent — T1's stop firing does NOT cancel T2's stop.

    Bybit-only: OKX and Binance do not support reduceOnly conditional orders with
    specific quantities easily. Callers fall back to _ensure_sl_or_close on those.

    Parameters:
        label: "T1_SL" or "T2_SL" — used in log messages for traceability.
    Returns:
        Bybit orderId string on success, None on failure.
    """
    if ex_id != "bybit":
        # Only Bybit supports this pattern cleanly — caller uses position-level SL fallback
        return None

    close_side_str = "Sell" if side == "LONG" else "Buy"
    # triggerDirection: 2 = fire when price falls BELOW trigger (LONG stop-loss)
    #                   1 = fire when price rises ABOVE trigger (SHORT stop-loss)
    trigger_dir = "2" if side == "LONG" else "1"

    try:
        # Bug 5 fix: load markets if this is a fresh exchange instance
        if not ex.markets:
            ex.load_markets()

        mkt = ex.market(symbol)
        bybit_sym = mkt["id"]   # e.g. "NEAR/USDT:USDT" → "NEARUSDT"

        # Bug 4 fix: round qty to the coin's step size and enforce minimum.
        # Bybit returns retCode 10001 "Qty invalid" when qty has too many decimals
        # or is below the minimum order size (e.g. NEAR min=1, BTC min=0.001).
        import math as _math
        _prec  = mkt.get("precision") or {}
        _lims  = mkt.get("limits") or {}
        qty_step = float(_prec.get("amount") or 0)
        min_qty  = float((_lims.get("amount") or {}).get("min") or 0)
        if qty_step > 0:
            # Floor to nearest step (never round up — avoids over-closing)
            qty = _math.floor(qty / qty_step) * qty_step
            qty = round(qty, 8)
        if min_qty > 0 and qty < min_qty:
            runner.log(
                f"Grade B {label}: qty {qty:.6f} below Bybit minimum {min_qty} "
                f"for {symbol} — fallback to position SL"
            )
            return None

        result = _ccxt_call(
            ex.privatePostV5OrderCreate,
            {
                "category":         "linear",
                "symbol":           bybit_sym,
                "side":             close_side_str,
                "orderType":        "Market",          # market fill when triggered
                "qty":              str(qty),
                "triggerPrice":     str(round(sl_price, 4)),
                "triggerDirection": trigger_dir,
                "triggerBy":        "MarkPrice",       # same trigger type as position SL
                "reduceOnly":       True,              # only fires if position exists
                "positionIdx":      0,                 # one-way mode
            },
            label=f"grade_b {label} stop {ex_id}",
        )
        # Bybit raw response: {"retCode": 0, "result": {"orderId": "xxx"}}
        order_id = (result.get("result") or {}).get("orderId")
        if order_id:
            runner.log(f"Grade B {label} stop placed: order={order_id} @ {sl_price:.4f} qty={qty}")
        else:
            runner.log(f"Grade B {label} stop — unexpected response (no orderId): {result}")
        return order_id or None
    except Exception as e:
        runner.log(f"WARN: Grade B {label} stop placement failed: {e}")
        return None


def _verify_and_restore_t2_sl(
    runner: "AutoRunner",
    ex,
    ex_id: str,
    symbol: str,
    side: str,
    t2_sl_id: Optional[str],
    t2_sl_price: float,
) -> None:
    """
    After T1 exits at SL (not TP), verify T2's stop order is still active on Bybit.
    Re-places the SL immediately if the stop order is missing or was consumed.

    Why this is needed:
    Even with two separate stop orders, edge cases can leave T2 unprotected:
    - Price gaps past SL (both stops trigger simultaneously) — T1's 60% fills but
      T2's 40% might be rejected as a duplicate if fills happen in the same ms.
    - The stop order gets cancelled by the exchange for any reason.
    This verification ensures T2 always has protection within one engine cycle.

    Logs: "T2 SL verified after T1 exit" OR "T2 SL missing — placed new SL"
    """
    t2_active = False

    if t2_sl_id:
        # Check if the T2 stop order is still in Bybit's open/untriggered orders
        try:
            open_orders = _ccxt_call(
                ex.fetch_open_orders, symbol,
                label=f"verify_t2_sl open_orders {ex_id}", retries=0,
            )
            for o in open_orders:
                # Match by ccxt id or by native Bybit orderId in info dict
                if (o.get("id") == t2_sl_id or
                        (o.get("info") or {}).get("orderId") == t2_sl_id):
                    t2_active = True
                    break
        except Exception as _fe:
            runner.log(f"T2 SL verify fetch failed: {_fe} — will re-place SL to be safe")

    if t2_active:
        runner.log(f"T2 SL verified after T1 exit — order {t2_sl_id} still active ✓")
    else:
        runner.log("T2 SL missing after T1 exit — placing new position SL immediately")
        # Re-apply position-level SL for the remaining T2 position (40%).
        # _ensure_sl_or_close retries 3× and closes the position if all attempts fail.
        _ensure_sl_or_close(runner, ex, ex_id, symbol, side, t2_sl_price)


# ── Real Grade B helpers ────────────────────────────────────────────────────

def _get_available_usdt(ex, ex_id: str) -> float:
    """
    Return the USDT balance actually available for new orders.
    For Bybit UNIFIED: reads availableBalance (the correct field for new orders).
      - availableBalance = walletBalance - orderIM - positionIM (what you can trade with)
      - availableToWithdraw is WRONG — it is 0 when below Bybit's withdrawal minimum,
        even when the full balance is free to trade.
    For OKX/Binance: uses ccxt free field.
    """
    try:
        params = {"accountType": "UNIFIED"} if ex_id == "bybit" else {}
        bal = _ccxt_call(ex.fetch_balance, params=params, label=f"fetch_available_balance {ex_id}", retries=0)

        if ex_id == "bybit":
            try:
                accounts = (bal.get("info") or {}).get("result", {}).get("list", [])
                for acc in accounts:
                    coins = acc.get("coin", [])
                    for coin in coins:
                        if coin.get("coin") == "USDT":
                            # Log all fields so we can diagnose any future issue
                            wallet_bal  = float(coin.get("walletBalance")       or 0)
                            avail_bal   = float(coin.get("availableBalance")     or 0)
                            avail_wtdrw = float(coin.get("availableToWithdraw")  or 0)
                            order_im    = float(coin.get("totalOrderIM")         or 0)
                            pos_im      = float(coin.get("totalPositionIM")      or 0)
                            print(
                                f"[balance-debug] Bybit USDT | walletBalance=${wallet_bal:.4f} "
                                f"availableBalance=${avail_bal:.4f} "
                                f"availableToWithdraw=${avail_wtdrw:.4f} "
                                f"orderIM=${order_im:.4f} posIM=${pos_im:.4f} "
                                f"→ using for margin check: ${avail_bal:.4f}",
                                flush=True,
                            )
                            # Use availableBalance — the correct field for new order margin
                            if avail_bal > 0:
                                return avail_bal
                            # If availableBalance is also 0, fall back to walletBalance
                            # (handles edge case where Bybit returns 0 due to rounding/timing)
                            if wallet_bal > 0:
                                print(
                                    f"[balance-debug] availableBalance=0 but walletBalance=${wallet_bal:.4f} "
                                    f"— using walletBalance as fallback",
                                    flush=True,
                                )
                                return wallet_bal
            except Exception as _be:
                print(f"[balance-debug] Bybit raw parse failed: {_be}", flush=True)

        # ccxt free field (OKX, Binance, and Bybit fallback)
        free = float((bal.get("USDT") or {}).get("free") or 0)
        if free <= 0:
            free = float((bal.get("free") or {}).get("USDT") or 0)
        print(f"[balance-debug] {ex_id} ccxt free USDT: ${free:.4f}", flush=True)
        return free
    except Exception as e:
        print(f"[WARN] _get_available_usdt failed ({ex_id}): {e}", flush=True)
        return 0.0


def _cancel_all_real_orders(email: str, symbol: str) -> None:
    """Cancel all open orders for symbol. Best-effort cleanup — never raises."""
    try:
        row = get_exchange(email)
        if not row:
            return
        ex = _make_ccxt_exchange(row)
        open_orders = _ccxt_call(ex.fetch_open_orders, symbol, label="fetch_open_orders cancel")
        for o in open_orders:
            try:
                _ccxt_call(ex.cancel_order, o["id"], symbol, label="cancel_order", retries=0)
            except RuntimeError:
                pass
    except Exception:
        pass


def _move_real_sl_to_breakeven(email: str, symbol: str, entry_price: float) -> bool:
    """
    Move position-level SL to entry price (breakeven) when T1 hits TP.
    Uses Bybit's /v5/position/trading-stop — one atomic call, zero gap.
    Safer than cancel+replace because the position is never unprotected.
    Returns True on success.
    """
    try:
        row = get_exchange(email)
        if not row:
            return False
        ex = _make_ccxt_exchange(row)
        mkt = ex.market(symbol)
        bybit_sym = mkt["id"]   # "BTC/USDT:USDT" → "BTCUSDT"
        _ccxt_call(
            ex.privatePostV5PositionTradingStop,
            {
                "category":    "linear",
                "symbol":      bybit_sym,
                "stopLoss":    str(round(entry_price, 4)),
                "slTriggerBy": "MarkPrice",
                "positionIdx": 0,
            },
            label="move_sl_breakeven", retries=1,
        )
        return True
    except Exception:
        return False


def _check_bybit_order_filled(email: str, order_id: str, symbol: str) -> bool:
    """
    Return True if the Bybit order is confirmed fully filled.
    Returns False on any error or if order is still open / canceled / unknown.
    Used to verify T1 TP hit on real Grade B trades before moving T2 SL to breakeven.

    Bug 3 fix: previous version created a fresh exchange with no markets loaded.
    ccxt Bybit's fetch_order requires markets to resolve the symbol category (linear/spot),
    so it silently failed and returned False even when T1_TP was actually filled.
    Fix: load markets before fetch_order, and accept both "closed" and "filled" statuses.
    """
    try:
        row = get_exchange(email)
        if not row:
            return False
        ex = _make_ccxt_exchange(row)
        # Load markets if this is a fresh exchange instance — required for symbol resolution
        if not ex.markets:
            try:
                ex.load_markets()
            except Exception:
                pass  # non-fatal: fetch_order may still work without full market map
        order = _ccxt_call(ex.fetch_order, order_id, symbol, label="fetch_order T1_check", retries=1)
        # ccxt normalises Bybit "Filled" → "closed"; accept both defensively
        return order.get("status") in ("closed", "filled")
    except Exception:
        return False


def place_real_grade_b_order(
    email: str,
    symbol: str,
    side: str,          # "LONG" | "SHORT"
    usdt_size: float,   # full position size in USDT (T1 + T2 combined)
    leverage: int,
    sl_pct: float,
    t1_tp_pct: float,   # T1 TP distance (80% of full ATR TP)
    t2_tp_pct: float,   # T2 TP distance (100% of full ATR TP)
) -> dict:
    """
    Place a real Grade B position on Bybit.

    Order layout:
      Entry:  one market order for full qty (T1 60% + T2 40%)
      SL:     position-level stop attached to entry (moved to breakeven after T1 fires)
      T1 TP:  reduceOnly limit order for 60% of qty at t1_tp_pct from entry
      T2 TP:  reduceOnly limit order for 40% of qty at t2_tp_pct from entry

    When T1's limit TP fires on Bybit, 60% of the position closes automatically.
    The engine then calls _move_real_sl_to_breakeven() to protect the remaining 40%.
    T2 then runs risk-free to its own TP or the breakeven SL.

    Paper tracking is unchanged — Grade B T1/T2 paper logic handles PnL accounting.
    """
    row = get_exchange(email)
    if not row:
        raise ValueError("No exchange connected")

    ex = _make_ccxt_exchange(row)
    ex_id      = (row.get("exchange") or "bybit").lower()
    order_side = "buy"  if side == "LONG" else "sell"
    close_side = "sell" if side == "LONG" else "buy"
    sl_mult  = (1 - sl_pct)    if side == "LONG" else (1 + sl_pct)
    t1_mult  = (1 + t1_tp_pct) if side == "LONG" else (1 - t1_tp_pct)
    t2_mult  = (1 + t2_tp_pct) if side == "LONG" else (1 - t2_tp_pct)

    # Cancel orphaned open orders first — previous failed Grade B attempts can
    # leave T1/T2 limit orders on the exchange that lock margin and cause 110007.
    try:
        open_orders = _ccxt_call(ex.fetch_open_orders, symbol, label=f"pre_order_cleanup grade_b {ex_id}", retries=0)
        for o in open_orders:
            try:
                _ccxt_call(ex.cancel_order, o["id"], symbol, label="cancel_orphan_b", retries=0)
                print(f"[order-cleanup] Cancelled orphaned order {o['id']} on {symbol}", flush=True)
            except RuntimeError:
                pass
    except RuntimeError:
        pass

    ticker = _ccxt_call(ex.fetch_ticker, symbol, label=f"fetch_ticker grade_b {ex_id}")
    price  = float(ticker["last"])

    # Cap size against available balance — NOT walletBalance (Bybit 110007 fix)
    available = _get_available_usdt(ex, ex_id)
    if available < 2.0:
        raise ValueError(f"SKIP_LOW_MARGIN: available USDT ${available:.2f} too low to open Grade B position — waiting for funds to free up.")
    if usdt_size > available * 0.95:
        usdt_size = round(available * 0.90, 4)
        print(f"[balance-cap] Grade B capped usdt_size to ${usdt_size:.2f} (available: ${available:.2f})", flush=True)

    try:
        _ccxt_call(ex.set_leverage, leverage, symbol, label=f"set_leverage grade_b {ex_id}")
    except RuntimeError as lev_e:
        err_str = str(lev_e)
        if "110043" in err_str or "leverage not modified" in err_str.lower():
            print(f"[INFO] Leverage already correct at {leverage}× on {symbol} — continuing with order")
        else:
            raise ValueError(f"Leverage set failed ({leverage}×): {lev_e}")

    notional = usdt_size * leverage
    qty      = round(notional / price, 6)
    if notional < 5.0:
        raise ValueError(
            f"INSUFFICIENT_BALANCE: Available ${available:.2f} USDT — notional ${notional:.2f} below $5 minimum. "
            f"Deposit more USDT to your exchange account."
        )

    sl_price    = round(price * sl_mult, 4)
    t1_tp_price = round(price * t1_mult, 4)
    t2_tp_price = round(price * t2_mult, 4)
    t1_qty = round(qty * 0.60, 6)
    t2_qty = round(qty * 0.40, 6)

    # ── Entry order — clean, no SL in params (FIX 1) ──────────────────────────
    # SL is set SEPARATELY after entry via _ensure_sl_or_close() so that:
    # - Entry and SL failures are independent (entry failure = skip; SL failure = retry or close)
    # - If entry succeeds, caller logs position IMMEDIATELY before SL attempt
    # - T1/T2 limit order failures are non-fatal (position still protected by SL)
    if ex_id == "bybit":
        entry_params = {"positionIdx": 0}
    elif ex_id == "okx":
        entry_params = {"tdMode": "cross"}
    else:
        entry_params = {"reduceOnly": False}

    if ex_id == "bybit":
        reduce_params = {"reduceOnly": True, "positionIdx": 0}
    else:
        reduce_params = {"reduceOnly": True}

    # ── STEP 1: Entry — if this fails, raise immediately (position not opened) ──
    entry_order = _ccxt_call(
        ex.create_order, symbol, "market", order_side, qty, params=entry_params,
        label=f"grade_b entry {side} {ex_id}",
    )

    # ── STEP 2: T1/T2 limit orders — best-effort, non-fatal ──────────────────
    # Position is now open. Caller will log it IMMEDIATELY after this function
    # returns. T1/T2 failures are logged as warnings — position is still protected
    # by SL which caller sets via _ensure_sl_or_close() after logging.
    t1_order_id = None
    t2_order_id = None
    try:
        t1_order = _ccxt_call(
            ex.create_order, symbol, "limit", close_side, t1_qty, t1_tp_price,
            params=reduce_params, label=f"grade_b T1_TP {ex_id}",
        )
        t1_order_id = t1_order.get("id")
    except RuntimeError as e:
        print(f"[WARN] Grade B T1 TP placement failed (entry placed): {e} — SL will protect.", flush=True)

    try:
        t2_order = _ccxt_call(
            ex.create_order, symbol, "limit", close_side, t2_qty, t2_tp_price,
            params=reduce_params, label=f"grade_b T2_TP {ex_id}",
        )
        t2_order_id = t2_order.get("id")
    except RuntimeError as e:
        print(f"[WARN] Grade B T2 TP placement failed (entry placed): {e} — SL will protect.", flush=True)

    return {
        "order_id":    entry_order.get("id"),
        "t1_tp_id":    t1_order_id,
        "t2_tp_id":    t2_order_id,
        "qty":         qty,
        "t1_qty":      t1_qty,
        "t2_qty":      t2_qty,
        "price":       price,
        "sl_price":    sl_price,
        "t1_tp_price": t1_tp_price,
        "t2_tp_price": t2_tp_price,
        "leverage":    leverage,
        "ex":          ex,
        "ex_id":       ex_id,
    }


@app.post("/trade")
async def place_trade(payload: TradeIn, user=Depends(require_user)):
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r and r.is_running():
            raise HTTPException(status_code=400, detail="Manual trading disabled while AI is running.")
    return await _place_trade_internal(email, payload.symbol, payload.side, payload.mode, extra_reason="Manual trade request.")


@app.post("/reset")
def reset_sandbox(user=Depends(require_user)):
    email = user["email"]
    sid = get_session_id(email)
    new_sid = sid + 1
    set_session_id(email, new_sid)
    set_equity(email, START_EQUITY)
    # Full demo reset — peak and floor both go back to the starting equity baseline.
    # floor_equity column must also be reset, otherwise the stored floor from a
    # previous profitable session would persist and confuse the new session's risk display.
    _reset_floor = round(float(START_EQUITY) * 0.85, 2)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE user_state SET peak_equity=%s, floor_equity=%s WHERE email=%s",
            (START_EQUITY, _reset_floor, email),
        )
        conn.commit()
    set_ai_restart_lock(email, 0)
    return {"ok": True, "equity": START_EQUITY, "new_session_id": new_sid}


@app.get("/auto/status")
def auto_status(user=Depends(require_user)):
    email = user["email"]
    now_ts = int(time.time())
    lock_until = get_ai_restart_lock(email)
    lock_sec = max(0, lock_until - now_ts)
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if not r:
            _last_stop = None
            try:
                with db_conn() as _sc:
                    _sc_cur = _sc.cursor()
                    _sc_cur.execute(
                        "SELECT last_stop_reason FROM user_state WHERE email=%s", (email,)
                    )
                    _sc_row = _sc_cur.fetchone()
                    _last_stop = (_sc_row or {}).get("last_stop_reason")
            except Exception:
                pass
            return {
                "ok": True, "running": False,
                "blocked_reason": _last_stop,
                "restart_locked": lock_sec > 0,
                "restart_lock_sec": lock_sec,
            }
        st = r.status()
        return {
            "ok": True, **asdict(st),
            "restart_locked": lock_sec > 0,
            "restart_lock_sec": lock_sec,
        }


@app.get("/auto/history")
def auto_history(user=Depends(require_user), limit: int = Query(default=40, ge=1, le=200)):
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            return {
                "ok": True,
                "session_id": r.ai_session_id,
                "events": list(r.history)[: int(limit)],
            }
    # No runner in memory (after redeploy) — read latest open session from DB
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM ai_sessions WHERE email=%s AND ended_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (email,),
        )
        sess = cur.fetchone()
        session_id = sess["id"] if sess else None
        if session_id:
            cur.execute(
                "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                "ORDER BY id DESC LIMIT %s",
                (email, session_id, int(limit)),
            )
        else:
            cur.execute(
                "SELECT t, msg FROM ai_logs WHERE email=%s ORDER BY id DESC LIMIT %s",
                (email, int(limit)),
            )
        rows = cur.fetchall()
    return {
        "ok": True,
        "session_id": session_id,
        "events": [{"t": r["t"], "msg": r["msg"]} for r in rows],
    }


def _classify_log_type(msg: str) -> str:
    m = msg.upper()
    if any(x in m for x in ["TRADE OPENED", "TRADE CLOSED", "MID_CANDLE TRADE", "REAL ORDER"]):
        return "TRADE"
    if m.startswith("BLOCKED") or m.startswith("EARLY BEAR"):
        return "BLOCKED"
    if "MID_CANDLE" in m:
        return "MID_CANDLE"
    if m.startswith("HOLDING"):
        return "HOLDING"
    if "ERROR" in m or "FAILED" in m:
        return "ERROR"
    if any(x in m for x in ["RESET", "MIDNIGHT", "DAILY COUNTERS", "STRICTNESS"]):
        return "RESET"
    return "SYSTEM"


@app.get("/auto/sessions")
def auto_sessions(
    user=Depends(require_user),
    limit: int = Query(default=20, ge=1, le=100),
    log_limit: int = Query(default=500, ge=1, le=2000),
    symbol: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None, max_length=200),
    log_type: Optional[str] = Query(default=None, max_length=30),
):
    """Return AI sessions for this user, newest first, with classified logs.
    Supports optional symbol filter, keyword search, and log_type filter."""
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        if symbol:
            cur.execute(
                "SELECT id, symbol, mode, trade_style, started_at, ended_at, stop_reason "
                "FROM ai_sessions WHERE email=%s AND UPPER(symbol)=UPPER(%s) ORDER BY id DESC LIMIT %s",
                (email, symbol.upper().strip(), int(limit)),
            )
        else:
            cur.execute(
                "SELECT id, symbol, mode, trade_style, started_at, ended_at, stop_reason "
                "FROM ai_sessions WHERE email=%s ORDER BY id DESC LIMIT %s",
                (email, int(limit)),
            )
        sessions = [dict(r) for r in cur.fetchall()]
        for sess in sessions:
            if search:
                cur.execute(
                    "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                    "AND LOWER(msg) LIKE LOWER(%s) ORDER BY id ASC LIMIT %s",
                    (email, sess["id"], f"%{search.strip()}%", int(log_limit)),
                )
            else:
                cur.execute(
                    "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                    "ORDER BY id ASC LIMIT %s",
                    (email, sess["id"], int(log_limit)),
                )
            events = [
                {"t": r["t"], "msg": r["msg"], "log_type": _classify_log_type(r["msg"])}
                for r in cur.fetchall()
            ]
            _VALID_LOG_TYPES = {"ALL", "TRADE", "SIGNAL", "RISK", "RESET", "SYSTEM", "ERROR"}
            if log_type and log_type.upper() != "ALL":
                lt = log_type.upper()
                if lt not in _VALID_LOG_TYPES:
                    lt = "SYSTEM"
                events = [e for e in events if e["log_type"] == lt]
            sess["events"] = events
            sess["total_events"] = len(events)
    return {"ok": True, "sessions": sessions}


@app.get("/auto/signal")
def auto_signal(user=Depends(require_user)):
    """Live 4-layer signal breakdown for the running AutoRunner."""
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if not r:
            return {"ok": False, "running": False}
        return {
            "ok": True,
            "running": r.is_running(),
            "symbol": r.symbol,
            "mode": r.mode,
            "signal": r.last_signal,
            "side": r.last_side,
            "blocked": r.blocked_reason,
            "adaptive_strictness": r.adaptive_strictness,
            "signal_score": r.last_score,
            "trade_style": r.trade_style,
            "market_grade": r.market_grade,
            "pending_trades": r.pending_trades,
            "breakdown": r.last_breakdown,
        }


@app.post("/auto/start")
def auto_start(payload: AutoStartIn, user=Depends(require_user)):
    email = user["email"]
    symbol = payload.symbol.upper().strip()
    if not (symbol.endswith("USDT") or symbol.endswith("-USDT")):
        raise HTTPException(status_code=400, detail="Use symbol like BTCUSDT / ETHUSDT / SOLUSDT")
    # Mode ceiling is the maximum allowed — user may reduce but never exceed.
    _mode_max = default_max_trades_per_day(payload.mode)
    if payload.max_trades_per_day is not None:
        max_trades = min(int(payload.max_trades_per_day), _mode_max)
    else:
        max_trades = _mode_max

    # ── Check 1: Duration-session restart lock ──────────────────────────────
    now_ts = int(time.time())
    lock_until = get_ai_restart_lock(email)
    if lock_until > now_ts:
        remaining = lock_until - now_ts
        h, m = remaining // 3600, (remaining % 3600) // 60
        raise HTTPException(
            status_code=400,
            detail=f"AI restart locked for {h}h {m}m — you stopped a duration session early. "
                   f"This protects your account from impulsive restarts. "
                   f"Use Reset Sandbox (demo) to clear, or wait until Dubai midnight.",
        )

    # ── Check 2: Bad-trade daily limit — use mode minimum, not payload value ─
    # Always enforce the mode's default minimum so the payload can't bypass it.
    mode_min_stop_after = {"ULTRA_SAFE": 1, "SAFE": 1, "NORMAL": 2, "MINI_ASYM": 2, "AGGRESSIVE": 3}
    effective_stop_after = max(int(payload.stop_after_bad_trades), mode_min_stop_after.get(payload.mode, 2))
    try:
        dubai_midnight = now_dubai().replace(hour=0, minute=0, second=0, microsecond=0)
        utc_midnight_str = dubai_midnight.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sid = get_session_id(email)
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT SUM(CASE WHEN unreal_pnl_percent < 0 THEN 1 ELSE 0 END) as bad "
                "FROM trades WHERE email = %s AND time >= %s AND session_id = %s",
                (email, utc_midnight_str, sid),
            )
            row = cur.fetchone()
        bad_today = int((row or {}).get("bad") or 0)
        if bad_today >= effective_stop_after:
            raise HTTPException(
                status_code=400,
                detail=f"Daily risk limit reached — {bad_today} bad trades today "
                       f"(limit: {effective_stop_after} for {payload.mode}). "
                       f"AI is locked until Dubai midnight to protect your account.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # DB error — allow start, runner will recount

    # Sync real exchange balance into DB before creating runner.
    # Without this, runner reads DB default ($1,000) for session_start_equity
    # and floor_equity, even if the real account has a different balance.
    if REAL_TRADING:
        try:
            _pre_bal = get_real_usdt_balance(email, force=True)
            if _pre_bal is not None:
                set_equity(email, _pre_bal)
                # ── Guard: stale demo peak causes instant hard floor on real accounts ──
                # Problem: user registers on demo branch (START_EQUITY=$1,000), gets
                # user_state.peak_equity=$1,000. Switches to real branch with $25.
                # Runner reads peak=$1,000 → floor=$850 → real equity $25 < $850 → instant hard floor.
                # Fix: if stored peak is 5× or more above the real balance (clearly a different
                # scale — demo default vs small real account), reset it to the real balance.
                # We do NOT reset if the peak is close to real balance (might be a legitimate
                # real-account peak that the user should respect for drawdown protection).
                try:
                    with db_conn() as conn:
                        _gpk_cur = conn.cursor()
                        _gpk_cur.execute(
                            "SELECT peak_equity, floor_equity FROM user_state WHERE email=%s", (email,)
                        )
                        _gpk_row = _gpk_cur.fetchone()
                        _gpk_peak = float((_gpk_row or {}).get("peak_equity") or 0)
                        _gpk_floor = float((_gpk_row or {}).get("floor_equity") or 0)
                        # 5× threshold: $1,000 peak vs $25 real = 40× → stale demo value.
                        # Real account that grew from $25 to $30 then fell to $25 = 1.2× → preserved.
                        _is_stale_demo_peak = _gpk_peak > _pre_bal * 5 and _gpk_peak > 50
                        if _is_stale_demo_peak:
                            _corrected_floor = round(_pre_bal * 0.85, 2)
                            _gpk_cur.execute(
                                "UPDATE user_state SET peak_equity=%s, floor_equity=%s WHERE email=%s",
                                (_pre_bal, _corrected_floor, email),
                            )
                            conn.commit()
                            print(
                                f"[auto-start] REAL peak corrected: demo peak ${_gpk_peak:.2f} "
                                f"→ real balance ${_pre_bal:.2f} (was causing instant hard floor)"
                            )
                except Exception as _gpke:
                    print(f"[auto-start] real peak guard failed: {_gpke}")
        except Exception as _bal_err:
            print(f"[auto-start] real balance pre-sync failed (non-fatal): {_bal_err}")

    # Record starting_capital on the very first AI start — never overwritten after initial set.
    # This anchors the Situation 1/2/3 logic in the floor-reset endpoint:
    #   Situation 1 = equity >= starting_capital (profitable overall — easy reset)
    #   Situation 2 = equity in [50%, 100%) of starting_capital (losses — must type CONFIRM)
    #   Situation 3 = equity < 50% of starting_capital (heavy loss — contact support)
    try:
        with db_conn() as conn:
            _sc_cur = conn.cursor()
            _sc_cur.execute("SELECT starting_capital FROM user_state WHERE email=%s", (email,))
            _sc_row = _sc_cur.fetchone()
            if _sc_row and float(_sc_row.get("starting_capital") or 0) == 0:
                _sc_equity = get_equity(email)
                _sc_cur.execute(
                    "UPDATE user_state SET starting_capital=%s WHERE email=%s",
                    (_sc_equity, email),
                )
                conn.commit()
    except Exception as _sce:
        print(f"[auto-start] starting_capital set failed: {_sce}")

    # ── Engine start queue ────────────────────────────────────────────────────
    with _ENGINE_START_LOCK:
        global _last_engine_start_ts
        _now = time.time()
        _engine_start_times[:] = [t for t in _engine_start_times if _now - t < 60]
        _count = len(_engine_start_times)
        _gap   = _now - _last_engine_start_ts

        if _count >= _MAX_STARTS_PER_MIN:
            _oldest   = min(_engine_start_times)
            _retry_in = int(60 - (_now - _oldest)) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Engine start queued — {_MAX_STARTS_PER_MIN} engines are starting this minute. "
                    f"You are position {_count - _MAX_STARTS_PER_MIN + 1} in queue. "
                    f"Retry in {_retry_in}s."
                ),
            )
        if _gap < _ENGINE_START_GAP:
            _retry_in = int(_ENGINE_START_GAP - _gap) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Engine start queued — system staggers starts to protect stability. "
                    f"Retry in {_retry_in}s."
                ),
            )
        _engine_start_times.append(_now)
        _last_engine_start_ts = _now

    try:
        with AUTO_LOCK:
            old = AUTO_RUNNERS.get(email)
            if old:
                old.stop("Stopped: restarted with new settings.")
                del AUTO_RUNNERS[email]

            runner = AutoRunner(
                email=email, symbol=symbol, trade_style=payload.trade_style,
                mode=payload.mode,
                max_trades_per_day=int(max_trades),
                stop_after_bad_trades=int(payload.stop_after_bad_trades),
                duration_days=int(payload.duration_days),
                trend_filter=bool(payload.trend_filter),
                chop_min_sep_pct=float(payload.chop_min_sep_pct),
            )
            AUTO_RUNNERS[email] = runner
            runner.start()
            _save_runner_state(runner)   # persist so deploy/restart auto-resumes this runner
    except Exception as _start_err:
        import traceback as _tb
        _err_detail = f"Engine start failed for {email} ({symbol}): {_start_err}\n{_tb.format_exc()}"
        print(f"[auto-start] CRITICAL: {_err_detail}")
        _tg_alert(
            f"🚨 *Engine start FAILED* ({symbol})\n"
            f"User: `{email}`\n"
            f"Error: `{_start_err}`\n"
            f"This caused a 500 on /auto/start — engine NOT running."
        )
        raise HTTPException(status_code=500, detail=f"Engine failed to start: {_start_err}")

    try:
        with db_conn() as _clr_conn:
            _clr_conn.cursor().execute(
                "UPDATE user_state SET last_stop_reason=NULL WHERE email=%s", (email,)
            )
            _clr_conn.commit()
    except Exception:
        pass

    email_ai_started(
        to=email, symbol=symbol, mode=payload.mode, trade_style=payload.trade_style,
        duration_days=int(payload.duration_days),
        max_trades=int(max_trades), stop_after_bad=int(payload.stop_after_bad_trades),
    )

    sp = TRADE_STYLE_PARAMS.get(payload.trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
    return {"ok": True, "running": True, "symbol": symbol,
            "trade_style": payload.trade_style, "tf": sp["tf"],
            "interval_sec": sp["interval"], "mode": payload.mode,
            "max_trades_per_day": int(max_trades)}


@app.post("/auto/stop")
def auto_stop(user=Depends(require_user)):
    email = user["email"]
    restart_locked = False
    lock_sec = 0
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            was_duration = r.duration_days > 0
            r.stop("Stopped by user.")
            del AUTO_RUNNERS[email]
            # Duration sessions: lock restarts until Dubai midnight to prevent bypass
            if was_duration:
                now_dt = now_dubai()
                next_midnight = (now_dt + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                lock_until = int(next_midnight.timestamp())
                set_ai_restart_lock(email, lock_until)
                lock_sec = max(0, lock_until - int(time.time()))
                restart_locked = True
    # Clear persisted state — user intentionally stopped, do NOT resume on next deploy
    _clear_runner_state(email)
    return {"ok": True, "running": False, "restart_locked": restart_locked, "restart_lock_sec": lock_sec}


@app.post("/auto/reset-strictness")
def auto_reset_strictness(user=Depends(require_user)):
    """Reset adaptive_strictness to 1.0x — clears stuck/legacy values from DB and live runner."""
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            r.adaptive_strictness = 1.0
            r.consecutive_wins = 0
            r.log("Strictness manually reset to 1.0x by user.")
            _save_runner_state(r)
    # Also patch DB directly so it survives restarts
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            if USING_PG:
                cur.execute(
                    "UPDATE ai_runner_state SET adaptive_strictness = 1.0, consecutive_wins = 0 WHERE email = %s",
                    (email,),
                )
            else:
                cur.execute(
                    "UPDATE ai_runner_state SET adaptive_strictness = 1.0, consecutive_wins = 0 WHERE email = ?",
                    (email,),
                )
            conn.commit()
    except Exception as e:
        print(f"[reset-strictness] DB patch failed: {e}")
    return {"ok": True, "adaptive_strictness": 1.0, "message": "Strictness reset to 1.0x"}


@app.post("/auto/correct-real-trade")
def auto_correct_real_trade(payload: CorrectTradeIn, user=Depends(require_user)):
    """
    Correct trade records when paper simulation diverges from real Bybit P&L.

    Use this when a trade is recorded as SL_HIT / loss but the exchange shows a profit
    (or vice versa). Finds the N most recent trades for the symbol in the current session,
    splits real_pnl proportionally by position size, and patches DB + live runner state.

    Also resets bad_trades_today=0 and adaptive_strictness=1.0 when real_pnl > 0.

    Body:
      symbol      — trading symbol, e.g. "NEAR/USDT:USDT"
      real_pnl    — total real P&L from exchange for all legs combined ($, positive = profit)
      num_trades  — 2 for Grade B (T1+T2), 1 for Grade A (default 2)
      new_outcome — "TP_HIT" or "SL_HIT" (default "TP_HIT")
    """
    email = user["email"]
    symbol = payload.symbol.strip()
    real_pnl = payload.real_pnl
    num_trades = max(1, min(payload.num_trades, 5))
    new_outcome = payload.new_outcome.upper()
    if new_outcome not in ("TP_HIT", "SL_HIT", "TRAIL_STOP"):
        raise HTTPException(status_code=400, detail="new_outcome must be TP_HIT, SL_HIT, or TRAIL_STOP")

    # ── Find the N most recent trade records for this symbol in current session ──
    sid = get_session_id(email)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, size, unreal_pnl_value, unreal_pnl_percent, reason "
            "FROM trades WHERE email=%s AND symbol=%s AND session_id=%s "
            "ORDER BY id DESC LIMIT %s",
            (email, symbol, int(sid), num_trades),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No recent trades found for {symbol} in current session (session_id={sid})."
        )

    # ── Split real_pnl proportionally by size across all legs ─────────────────
    total_size = sum(float(r["size"]) for r in rows)
    corrections = []
    for r in rows:
        frac = float(r["size"]) / total_size if total_size > 0 else 1.0 / len(rows)
        leg_pnl = round(real_pnl * frac, 4)
        # For percent: pnl / (size * equity_before) — we don't have equity_before stored
        # so use the existing percent field's magnitude scaled to the new pnl sign/size
        # as an approximation; the dollar value is what the frontend displays.
        old_pct = float(r["unreal_pnl_percent"])
        new_pct = round(leg_pnl / float(r["size"]) * 100, 4) if float(r["size"]) > 0 else old_pct
        corrections.append({"id": r["id"], "leg_pnl": leg_pnl, "new_pct": new_pct,
                             "old_pnl": float(r["unreal_pnl_value"]), "reason": r["reason"]})

    # ── Apply DB corrections ──────────────────────────────────────────────────
    now_str = now_dubai().strftime("%Y-%m-%d %H:%M:%S")
    with db_conn() as conn:
        cur = conn.cursor()
        for c_item in corrections:
            correction_note = (
                f"\n\n[MANUAL CORRECTION {now_str}] "
                f"Paper: {c_item['old_pnl']:+.4f} → Real: {c_item['leg_pnl']:+.4f} | "
                f"Outcome corrected to {new_outcome}"
            )
            new_reason = (c_item["reason"] or "") + correction_note
            cur.execute(
                "UPDATE trades SET unreal_pnl_value=%s, unreal_pnl_percent=%s, reason=%s "
                "WHERE id=%s",
                (c_item["leg_pnl"], c_item["new_pct"], new_reason, c_item["id"]),
            )
        conn.commit()

    # ── Update live runner state if AI is running ──────────────────────────────
    runner_updated = False
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            if real_pnl > 0:
                # Trade was actually a WIN — undo the bad-trade penalty
                r.bad_trades_today = 0
                r.adaptive_strictness = 1.0
                r.consecutive_wins = 0
                r._last_trade_bad = False
            r.log(
                f"[MANUAL CORRECTION] {symbol} trade P&L corrected: "
                f"${real_pnl:+.4f} total ({len(corrections)} leg(s)). "
                f"bad_trades=0, strictness=1.0"
            )
            _save_runner_state(r)
            runner_updated = True

    # ── Also patch DB strictness/bad_trades in case runner is stopped ─────────
    if real_pnl > 0:
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE ai_runner_state SET adaptive_strictness=1.0, "
                    "consecutive_wins=0, last_trade_bad=0 WHERE email=%s",
                    (email,),
                )
                conn.commit()
        except Exception as _sde:
            print(f"[correct-trade] DB strictness patch failed: {_sde}")

    # ── Update peak equity if current balance is a new high ───────────────────
    current_equity = get_equity(email)
    update_peak_ath(email, current_equity, current_equity)

    return {
        "ok": True,
        "trades_corrected": len(corrections),
        "symbol": symbol,
        "real_pnl_total": real_pnl,
        "legs": [{"trade_id": c["id"], "pnl": c["leg_pnl"], "pnl_pct": c["new_pct"]} for c in corrections],
        "bad_trades_reset": real_pnl > 0,
        "strictness_reset": real_pnl > 0,
        "runner_updated": runner_updated,
        "peak_equity_updated": True,
    }


@app.post("/auto/reset-floor")
def auto_reset_floor(payload: FloorResetIn, user=Depends(require_user)):
    """
    Hard-floor reset endpoint.

    Call with confirm=False first to get a dry-run status (situation, cooldown, current equity).
    Call with confirm=True to actually apply the reset.

    Situation 1 — equity >= starting_capital: one-click reset, no extra confirmation.
    Situation 2 — equity in [50%, 100%) of starting_capital: must type 'CONFIRM'.
    Situation 3 — equity < 50% of starting_capital: blocked entirely, contact support.

    Cooldown: 3 or more resets within the last 30 days → 7-day cooldown from the most recent reset.
    After reset: peak resets to current equity, floor = 85% of that, engine is NOT auto-restarted
    (user clicks Start again with their preferred settings).
    """
    email = user["email"]
    now_ts = time.time()
    now_str = now_dubai().strftime("%Y-%m-%d %H:%M:%S")

    # ── Read current user_state from DB ─────────────────────────────────────
    with db_conn() as conn:
        _rc = conn.cursor()
        _rc.execute(
            "SELECT peak_equity, all_time_high, starting_capital, "
            "floor_reset_count, floor_reset_at FROM user_state WHERE email=%s",
            (email,),
        )
        _row = _rc.fetchone()

    if not _row:
        raise HTTPException(status_code=400, detail="No user state found. Start the AI first.")

    # Live balance — real exchange for real accounts, DB value for demo
    current_equity = get_equity(email)
    starting_capital = float(_row.get("starting_capital") or 0)
    current_ath = float(_row.get("all_time_high") or 0)

    # Parse the JSON list of prior reset timestamps
    try:
        reset_timestamps: List[float] = json.loads(_row.get("floor_reset_at") or "[]")
    except Exception:
        reset_timestamps = []

    # Keep only resets from the last 30 days for cooldown calculation
    thirty_days_ago = now_ts - (30 * 24 * 3600)
    recent_resets = [t for t in reset_timestamps if t > thirty_days_ago]

    # ── Cooldown check — 3+ resets in 30 days triggers a 7-day lock ────────
    cooldown_remaining_sec = 0
    if len(recent_resets) >= 3:
        most_recent = max(recent_resets)
        cooldown_until = most_recent + (7 * 24 * 3600)
        if now_ts < cooldown_until:
            cooldown_remaining_sec = int(cooldown_until - now_ts)

    # ── Determine situation ──────────────────────────────────────────────────
    # starting_capital = 0 means the AI was never started (no record), treat as Situation 1.
    if starting_capital <= 0:
        situation = 1
    elif current_equity >= starting_capital:
        # Equity is at or above what the user started with — profitable overall
        situation = 1
    elif current_equity >= starting_capital * 0.50:
        # Lost money but more than half remains — recoverable with caution
        situation = 2
    else:
        # Lost more than half of starting capital — blocked, manual review needed
        situation = 3

    # Build dry-run response — always return this info so frontend can show the dialog
    status = {
        "ok": True,
        "situation": situation,
        "current_equity": round(current_equity, 2),
        "starting_capital": round(starting_capital, 2),
        "new_peak_if_reset": round(current_equity, 2),
        "new_floor_if_reset": round(current_equity * 0.85, 2),
        "resets_in_30d": len(recent_resets),
        "cooldown_remaining_sec": cooldown_remaining_sec,
        "reset_applied": False,
        "message": "",
    }

    # ── Dry-run: return status without applying ──────────────────────────────
    if not payload.confirm:
        if situation == 1:
            status["message"] = (
                f"Situation 1 — your equity (${current_equity:.2f}) is at or above your "
                f"starting capital (${starting_capital:.2f}). "
                f"One-click reset available."
            )
        elif situation == 2:
            status["message"] = (
                f"Situation 2 — your equity (${current_equity:.2f}) is below your starting "
                f"capital (${starting_capital:.2f}). You must type 'CONFIRM' to proceed."
            )
        else:
            status["message"] = (
                f"Situation 3 — your equity (${current_equity:.2f}) is below 50% of your "
                f"starting capital (${starting_capital:.2f}). Reset is blocked. "
                f"Please contact support."
            )
        return status

    # ── Validation before applying ───────────────────────────────────────────
    if cooldown_remaining_sec > 0:
        h = cooldown_remaining_sec // 3600
        m = (cooldown_remaining_sec % 3600) // 60
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reset cooldown — you have reset the floor 3 times in 30 days. "
                f"Cooldown expires in {h}h {m}m. "
                f"This protects your account from repeatedly overriding risk protection."
            ),
        )

    if situation == 3:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Floor reset blocked — your equity (${current_equity:.2f}) is below 50% of "
                f"your starting capital (${starting_capital:.2f}). "
                f"Please contact support to review your account."
            ),
        )

    if situation == 2 and payload.typed_confirm.strip().upper() != "CONFIRM":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Your equity (${current_equity:.2f}) is below your starting capital "
                f"(${starting_capital:.2f}). Please type 'CONFIRM' to accept this loss "
                f"and reset the safety floor."
            ),
        )

    # ── Apply the reset ──────────────────────────────────────────────────────
    new_peak = current_equity
    new_floor = round(current_equity * 0.85, 2)
    # all_time_high never goes down — only update if new_peak exceeds it
    new_ath = max(current_ath, new_peak)
    updated_resets = recent_resets + [now_ts]  # add this reset to history

    with db_conn() as conn:
        _ac = conn.cursor()
        # Write peak AND floor_equity — floor_equity is the persistent floor that survives
        # runner restarts. We explicitly set it here (not max) because this is an intentional reset.
        _ac.execute(
            "UPDATE user_state SET "
            "peak_equity=%s, all_time_high=%s, floor_equity=%s, "
            "floor_reset_count=%s, floor_reset_at=%s "
            "WHERE email=%s",
            (
                new_peak, new_ath, new_floor,
                len(updated_resets),
                json.dumps(updated_resets),
                email,
            ),
        )
        # Log the reset permanently in ai_logs so it appears in the user's AI log history
        sid = get_session_id(email)
        _ac.execute(
            "INSERT INTO ai_logs(email, session_id, t, msg) VALUES(%s, %s, %s, %s)",
            (
                email, sid, now_str,
                (
                    f"FLOOR RESET (Situation {situation}) — "
                    f"new peak=${new_peak:.2f}, new floor=${new_floor:.2f}, "
                    f"starting_capital=${starting_capital:.2f}, "
                    f"resets_in_30d={len(updated_resets)}"
                ),
            ),
        )
        conn.commit()

    # ── Update live runner in memory if somehow still present ────────────────
    # Normally the runner has already removed itself when HARD_FLOOR hit,
    # but we patch in-memory state defensively in case of any race condition.
    with AUTO_LOCK:
        _r = AUTO_RUNNERS.get(email)
    if _r:
        _r.peak_equity = new_peak
        _r.floor_equity = new_floor
        _r.blocked_reason = None
        _r.log(
            f"FLOOR RESET applied by user — new peak=${new_peak:.2f}, "
            f"new floor=${new_floor:.2f} (Situation {situation})"
        )
        _save_runner_state(_r)

    status["reset_applied"] = True
    status["resets_in_30d"] = len(updated_resets)
    status["message"] = (
        f"Floor reset applied (Situation {situation}). "
        f"New peak: ${new_peak:.2f}, new floor: ${new_floor:.2f}. "
        f"Click Start to resume trading with the new floor."
    )
    return status


# Admin + analytics endpoints moved to routes_admin.py (admin_router)

# =========================
# STARTUP: RESUME AI RUNNERS
# =========================
# Must run AFTER all classes (AutoRunner, helpers) are defined.
# Re-starts any AI session that was live before the last deploy/restart.
import threading as _threading
_threading.Thread(target=_resume_runners_on_startup, daemon=True).start()
_threading.Thread(target=_resume_watchdog, daemon=True).start()

# ── Deploy/restart notification ───────────────────────────────────────────────
# Fires on every successful startup so you know the service is alive.
# Absence of this message = service crashed before reaching this point.
if REAL_TRADING:
    _tg_alert("🚀 *Real trading service restarted* — backend is live and accepting connections.")
