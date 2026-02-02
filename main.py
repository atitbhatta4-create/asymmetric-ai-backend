from __future__ import annotations

import os
import secrets
import time
import hashlib
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Literal, Any, Deque
from collections import deque

import httpx
from fastapi import FastAPI, Depends, HTTPException, Response, Cookie, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# App
# =========================
app = FastAPI(title="Asymmetric AI Backend", version="0.9.0")

# =========================
# ENV / PROD SETTINGS
# =========================
ENV = os.getenv("ENV", "dev").lower().strip()  # dev | prod
IS_PROD = ENV == "prod"

# FRONTEND_ORIGINS should be comma-separated (IMPORTANT for cookies with Vercel)
# Example:
# FRONTEND_ORIGINS="https://asymmetric-ai-frontend-23yi.vercel.app"
DEFAULT_ORIGINS = ["https://asymmetric-ai-frontend-23yi.vercel.app"]

FRONTEND_ORIGINS_RAW = os.getenv("FRONTEND_ORIGINS", "").strip()
if FRONTEND_ORIGINS_RAW:
    FRONTEND_ORIGINS = [x.strip() for x in FRONTEND_ORIGINS_RAW.split(",") if x.strip()]
else:
    FRONTEND_ORIGINS = DEFAULT_ORIGINS

# =========================
# CORS (NO WILDCARDS when using credentials)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ✅ IMPORTANT: Support /api/* paths used by frontend
# This keeps all your existing routes unchanged.
# /api/auth/login -> /auth/login
# /api/health -> /health
# =========================
@app.middleware("http")
async def strip_api_prefix(request: Request, call_next):
    path = request.scope.get("path", "")
    if path == "/api":
        request.scope["path"] = "/"
    elif path.startswith("/api/"):
        request.scope["path"] = path[4:]  # remove "/api"
    return await call_next(request)

@app.get("/")
def root():
    return {"ok": True, "service": "Asymmetric AI Backend", "env": ENV}

# =========================
# SQLITE (DEMO PERSISTENCE)
# =========================
DB_PATH = os.getenv("ASYM_DB_PATH", "asymmetric_demo.db")
DB_LOCK = threading.Lock()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _column_exists(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols


def init_db() -> None:
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS exchange_keys (
                email TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                api_key TEXT NOT NULL,
                api_secret TEXT NOT NULL,
                passphrase TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_state (
                email TEXT PRIMARY KEY,
                equity REAL NOT NULL,
                session_id INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_settings (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS password_resets (
                token TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                used INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        # safe migrations
        if not _column_exists(cur, "trades", "reason"):
            cur.execute("ALTER TABLE trades ADD COLUMN reason TEXT")
        if not _column_exists(cur, "trades", "session_id"):
            cur.execute("ALTER TABLE trades ADD COLUMN session_id INTEGER NOT NULL DEFAULT 0")
        if not _column_exists(cur, "user_state", "session_id"):
            cur.execute("ALTER TABLE user_state ADD COLUMN session_id INTEGER NOT NULL DEFAULT 0")

        # seed defaults if missing
        def _set_default(key: str, value: str) -> None:
            cur.execute("SELECT v FROM admin_settings WHERE k=?", (key,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO admin_settings(k,v,updated_at) VALUES(?,?,?)",
                    (key, value, now_utc_str()),
                )

        _set_default("signup_enabled", "true")
        _set_default("seat_capacity", "50")

        conn.commit()
        conn.close()


init_db()

# =========================
# CONFIG (Binance) - kept for compatibility
# =========================
BINANCE_ENV = os.getenv("BINANCE_ENV", "live").lower().strip()
if BINANCE_ENV not in ("live", "testnet"):
    BINANCE_ENV = "live"
BINANCE_BASE = "https://api.binance.com" if BINANCE_ENV == "live" else "https://testnet.binance.vision"

# =========================
# HELPERS
# =========================
START_EQUITY = 1000.0
RiskMode = Literal["ULTRA_SAFE", "SAFE", "NORMAL", "MINI_ASYM", "AGGRESSIVE"]
Side = Literal["LONG", "SHORT"]
TF = Literal["15m", "1h", "4h", "1d"]

# used for validation only in your code
TF_MAP: Dict[str, str] = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}

DUBAI_TZ = timezone(timedelta(hours=4))


def now_dubai() -> datetime:
    return datetime.now(tz=DUBAI_TZ)


def dubai_day_key(dt: Optional[datetime] = None) -> str:
    d = dt or now_dubai()
    return d.strftime("%Y-%m-%d")


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def mask_key(s: str) -> str:
    s = (s or "").strip()
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + ("*" * (len(s) - 8)) + s[-4:]


def ensure_user_state(email: str) -> None:
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT equity, session_id FROM user_state WHERE email = ?", (email,))
        row = cur.fetchone()
        if not row:
            cur.execute(
                "INSERT INTO user_state(email, equity, session_id) VALUES(?, ?, ?)",
                (email, START_EQUITY, 0),
            )
            conn.commit()
        conn.close()


def get_equity(email: str) -> float:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT equity FROM user_state WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()
    return float(row["equity"])


def set_equity(email: str, equity: float) -> None:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET equity = ? WHERE email = ?", (float(equity), email))
        conn.commit()
        conn.close()


def get_session_id(email: str) -> int:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT session_id FROM user_state WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()
    return int(row["session_id"] or 0)


def set_session_id(email: str, sid: int) -> None:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET session_id = ? WHERE email = ?", (int(sid), email))
        conn.commit()
        conn.close()


def get_exchange(email: str) -> Optional[sqlite3.Row]:
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM exchange_keys WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()
    return row


def require_user(session: Optional[str] = Cookie(default=None)) -> Dict[str, str]:
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email FROM sessions WHERE token = ?", (session,))
        row = cur.fetchone()
        conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Unauthorized")
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
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT v FROM admin_settings WHERE k=?", (key,))
        row = cur.fetchone()
        conn.close()
    return (row["v"] if row else default)


def admin_set_setting(key: str, value: str) -> None:
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO admin_settings(k,v,updated_at) VALUES(?,?,?)
            ON CONFLICT(k) DO UPDATE SET
                v=excluded.v,
                updated_at=excluded.updated_at
            """,
            (key, value, now_utc_str()),
        )
        conn.commit()
        conn.close()


def signup_is_enabled() -> bool:
    return admin_get_setting("signup_enabled", "true").lower() == "true"


def seat_capacity() -> int:
    try:
        return int(admin_get_setting("seat_capacity", "50"))
    except Exception:
        return 50


def seats_used() -> int:
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM users")
        n = int(cur.fetchone()["n"])
        conn.close()
    return n


# Seed demo admin user if missing
with DB_LOCK:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT email FROM users WHERE email = ?", (DEMO_EMAIL,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users(email, password_hash, created_at) VALUES(?, ?, ?)",
            (DEMO_EMAIL, hash_pw(DEMO_PASS), now_utc_str()),
        )
        conn.commit()
    conn.close()

# =========================
# BASIC INFO
# =========================
@app.get("/config")
def config():
    return {
        "binance_env": BINANCE_ENV,
        "binance_base": BINANCE_BASE,
        "timezone": "Asia/Dubai (UTC+4)",
        "env": ENV,
        "frontend_origins": FRONTEND_ORIGINS,
    }


@app.get("/health")
def health():
    return {"health": "green", "binance_env": BINANCE_ENV, "db_path": DB_PATH, "env": ENV}


@app.get("/api/health")
def api_health():
    return health()


# =========================
# AUTH MODELS
# =========================
class AuthIn(BaseModel):
    email: str
    password: str


class SessionOut(BaseModel):
    ok: bool
    email: Optional[str] = None


# =========================
# COOKIE SETTINGS (FIXED FOR VERCEL)
# =========================
def set_session_cookie(response: Response, token: str) -> None:
    """
    Dev: samesite=lax, secure=False
    Prod (Vercel -> Render cross-site cookies): samesite=none, secure=True
    """
    base = dict(
        key="session",
        value=token,
        httponly=True,
        max_age=60 * 60 * 24 * 7,
        path="/",
    )
    if IS_PROD:
        response.set_cookie(**base, samesite="none", secure=True)
    else:
        response.set_cookie(**base, samesite="lax", secure=False)


def clear_session_cookie(response: Response) -> None:
    if IS_PROD:
        response.delete_cookie("session", path="/", samesite="none", secure=True)
    else:
        response.delete_cookie("session", path="/", samesite="lax", secure=False)


# =========================
# AUTH ENDPOINTS
# =========================
@app.post("/auth/signup")
def signup(payload: AuthIn):
    email = payload.email.strip().lower()
    if not email or not payload.password:
        raise HTTPException(status_code=400, detail="Email and password required")

    if not signup_is_enabled():
        raise HTTPException(status_code=403, detail="Signup is currently disabled by admin.")
    if seats_used() >= seat_capacity():
        raise HTTPException(status_code=403, detail="Seats are full. Signup is closed.")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="User already exists")

        cur.execute(
            "INSERT INTO users(email, password_hash, created_at) VALUES(?, ?, ?)",
            (email, hash_pw(payload.password), now_utc_str()),
        )
        conn.commit()
        conn.close()

    return {"ok": True}


@app.post("/auth/login", response_model=SessionOut)
def login(payload: AuthIn, response: Response):
    email = payload.email.strip().lower()

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()

    if not row or row["password_hash"] != hash_pw(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(32)
    now_ts = int(time.time())

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("INSERT INTO sessions(token, email, created_at) VALUES(?, ?, ?)", (token, email, now_ts))
        conn.commit()
        conn.close()

    set_session_cookie(response, token)
    ensure_user_state(email)
    return {"ok": True, "email": email}


@app.post("/auth/logout")
def logout(response: Response, session: Optional[str] = Cookie(default=None)):
    if session:
        with DB_LOCK:
            conn = db()
            cur = conn.cursor()
            cur.execute("DELETE FROM sessions WHERE token = ?", (session,))
            conn.commit()
            conn.close()

    clear_session_cookie(response)
    return {"ok": True}


@app.get("/session", response_model=SessionOut)
def session_me(session: Optional[str] = Cookie(default=None)):
    if not session:
        return {"ok": False, "email": None}
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email FROM sessions WHERE token = ?", (session,))
        row = cur.fetchone()
        conn.close()
    return {"ok": bool(row), "email": row["email"] if row else None}


# =========================
# FORGOT PASSWORD (DEMO)
# =========================
class ForgotIn(BaseModel):
    email: str


class ResetIn(BaseModel):
    token: str
    new_password: str


@app.post("/auth/forgot")
def auth_forgot(payload: ForgotIn):
    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email=?", (email,))
        u = cur.fetchone()
        if not u:
            conn.close()
            return {"ok": True}

        token = secrets.token_urlsafe(32)
        cur.execute(
            "INSERT INTO password_resets(token,email,created_at,used) VALUES(?,?,?,0)",
            (token, email, int(time.time())),
        )
        conn.commit()
        conn.close()

    return {"ok": True, "reset_token": token}


@app.post("/auth/reset")
def auth_reset(payload: ResetIn):
    token = payload.token.strip()
    new_pw = payload.new_password or ""
    if not token or len(new_pw) < 4:
        raise HTTPException(status_code=400, detail="token + new_password required (min 4 chars)")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email, used, created_at FROM password_resets WHERE token=?", (token,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid token")

        if int(row["used"]) == 1:
            conn.close()
            raise HTTPException(status_code=400, detail="Token already used")

        created_at = int(row["created_at"])
        if int(time.time()) - created_at > 1800:
            conn.close()
            raise HTTPException(status_code=400, detail="Token expired")

        email = row["email"]
        cur.execute("UPDATE users SET password_hash=? WHERE email=?", (hash_pw(new_pw), email))
        cur.execute("UPDATE password_resets SET used=1 WHERE token=?", (token,))
        cur.execute("DELETE FROM sessions WHERE email=?", (email,))
        conn.commit()
        conn.close()

    return {"ok": True}


class AdminForceResetIn(BaseModel):
    token: str
    email: str
    new_password: str


@app.post("/admin/force-reset-password")
def admin_force_reset(payload: AdminForceResetIn):
    if not ADMIN_FORCE_RESET_TOKEN or payload.token.strip() != ADMIN_FORCE_RESET_TOKEN:
        raise HTTPException(status_code=403, detail="Not allowed")

    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=400, detail="email is not an admin")
    if not payload.new_password or len(payload.new_password) < 4:
        raise HTTPException(status_code=400, detail="new_password min 4 chars")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email=?", (email,))
        u = cur.fetchone()
        if not u:
            cur.execute(
                "INSERT INTO users(email, password_hash, created_at) VALUES(?,?,?)",
                (email, hash_pw(payload.new_password), now_utc_str()),
            )
        else:
            cur.execute("UPDATE users SET password_hash=? WHERE email=?", (hash_pw(payload.new_password), email))
        cur.execute("DELETE FROM sessions WHERE email=?", (email,))
        conn.commit()
        conn.close()

    return {"ok": True}


# =========================
# EXCHANGE CONNECT
# =========================
class ExchangeConnectIn(BaseModel):
    exchange: Literal["binance", "okx"]
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None


@app.get("/exchange/status")
def exchange_status(user=Depends(require_user)):
    email = user["email"]
    row = get_exchange(email)
    if not row:
        return {"connected": False, "exchange": None, "api_key_masked": None, "created_at": None}
    return {
        "connected": True,
        "exchange": row["exchange"],
        "api_key_masked": mask_key(row["api_key"]),
        "created_at": row["created_at"],
    }


@app.post("/exchange/connect")
def exchange_connect(payload: ExchangeConnectIn, user=Depends(require_user)):
    email = user["email"]
    if not payload.api_key.strip() or not payload.api_secret.strip():
        raise HTTPException(status_code=400, detail="API key/secret required")

    created_at = now_utc_str()
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO exchange_keys(email, exchange, api_key, api_secret, passphrase, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
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
                payload.api_key.strip(),
                payload.api_secret.strip(),
                (payload.passphrase or "").strip(),
                created_at,
            ),
        )
        conn.commit()
        conn.close()

    return {"ok": True, "binance_env": BINANCE_ENV}


@app.post("/exchange/disconnect")
def exchange_disconnect(user=Depends(require_user)):
    email = user["email"]
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("DELETE FROM exchange_keys WHERE email = ?", (email,))
        conn.commit()
        conn.close()
    return {"ok": True}


@app.get("/exchange/test")
def exchange_test(user=Depends(require_user)):
    email = user["email"]
    row = get_exchange(email)
    if not row:
        raise HTTPException(status_code=400, detail="No exchange connected.")
    return {"ok": True, "canTrade": False, "accountType": "DEMO", "note": "Demo mode only."}


@app.get("/exchange/balance")
def exchange_balance(user=Depends(require_user)):
    email = user["email"]
    row = get_exchange(email)
    if not row:
        raise HTTPException(status_code=400, detail="No exchange connected.")
    eq = get_equity(email)
    return {"ok": True, "balances": [{"asset": "USDT", "free": eq, "locked": 0.0}], "note": "Demo balances only."}
# =========================
# MARKET (PUBLIC) - OKX
# =========================

OKX_BASE = "https://www.okx.com"

# OKX timeframes
OKX_TF_MAP = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}

def to_okx_inst(symbol: str) -> str:
    # BTCUSDT -> BTC-USDT
    s = (symbol or "").upper().strip()
    if "-" in s:
        return s
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USDT"
    return s

async def okx_price(symbol: str) -> float:
    inst = to_okx_inst(symbol)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{OKX_BASE}/api/v5/market/ticker",
            params={"instId": inst},
            headers={"accept": "application/json"},
        )

    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"OKX price error: {r.text[:200]}")

    data = r.json()
    arr = (data or {}).get("data") or []
    if not arr:
        raise HTTPException(status_code=400, detail=f"OKX price error: no ticker for {inst}")

    last = arr[0].get("last")
    if last is None:
        raise HTTPException(status_code=400, detail=f"OKX price error: missing last for {inst}")

    return float(last)

@app.get("/price/{symbol}")
async def get_price(symbol: str):
    return {"symbol": symbol.upper().strip(), "price": await okx_price(symbol)}

def _fetch_klines_sync(symbol: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    inst = to_okx_inst(symbol)
    bar = OKX_TF_MAP.get(str(tf))
    if not bar:
        return []

    try:
        r = httpx.get(
            f"{OKX_BASE}/api/v5/market/candles",
            params={"instId": inst, "bar": bar, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code != 200:
            return []

        data = r.json()
        rows = (data or {}).get("data") or []
        if not rows:
            return []

        # OKX candle row:
        # [ ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm ]
        out: List[Dict[str, Any]] = []
        for k in rows:
            out.append(
                {
                    "t": int(k[0]),          # ms
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )

        # OKX usually returns newest -> oldest
        out.reverse()
        return out

    except Exception:
        return []

@app.get("/klines/{symbol}")
def klines(symbol: str, tf: TF = Query("1h"), limit: int = Query(200, ge=20, le=1000)):
    rows = _fetch_klines_sync(symbol, str(tf), limit=limit)
    if not rows:
        raise HTTPException(status_code=400, detail="No kline data (bad symbol/tf or OKX blocked).")
    return {"symbol": symbol.upper().strip(), "tf": str(tf), "klines": rows}

# ✅ keep compatibility with your existing trading engine
binance_price = okx_price


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
    growth = (equity - START_EQUITY) / START_EQUITY
    lines: List[str] = []
    lines.append(f"Mode: {mode}")
    lines.append(f"Equity: ${equity:.2f} (start ${START_EQUITY:.2f}, growth {growth*100:.2f}%)")
    lines.append("Risk rules applied:")
    if growth >= 0.10:
        lines.append("- Equity growth ≥ 10% → reduce size and leverage by ~10% to protect profits.")
    else:
        lines.append("- Equity growth < 10% (or negative) → use default preset for the selected mode.")
    lines.append("Computed parameters:")
    lines.append(f"- Size%: {computed['size']:.2f}")
    lines.append(f"- SL%:   {computed['sl']:.2f}")
    lines.append(f"- TP%:   {computed['tp']:.2f}")
    lines.append(f"- Lev:   {computed['leverage']:.2f}")
    if extra:
        lines.append("")
        lines.append(extra)
    lines.append("Note: Demo sandbox trade PnL is simulated. No real orders placed.")
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
    size: float
    sl: float
    tp: float
    leverage: float


class RiskPreviewIn(BaseModel):
    symbol: str
    side: Side
    mode: RiskMode
    size: float
    sl: float
    tp: float
    leverage: float


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
    equity = get_equity(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT time FROM trades WHERE email = ? ORDER BY id DESC LIMIT 1", (email,))
        row = cur.fetchone()
        conn.close()
    return {"total": equity, "equity_after_last_trade": equity, "last_trade": row["time"] if row else None}


@app.get("/trades")
def trades(
    user=Depends(require_user),
    symbol: Optional[str] = Query(default=None),
    current_session: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=2000),
):
    email = user["email"]
    where = ["email = ?"]
    params: List[Any] = [email]

    if current_session:
        sid = get_session_id(email)
        where.append("session_id = ?")
        params.append(sid)

    if symbol:
        where.append("UPPER(symbol) = ?")
        params.append(symbol.strip().upper())

    q = f"SELECT * FROM trades WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute(q, tuple(params))
        rows = cur.fetchall()
        conn.close()

    return {"trades": [dict(r) for r in rows]}


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

    entry = await binance_price(symbol)  # alias -> bybit_price

    move = ((int(time.time()) % 140) - 70) / 1000.0  # -0.07..+0.07
    pnl_pct = move * (c["leverage"] / 5.0)
    pnl_value = equity_before * (c["size"] / 100.0) * pnl_pct
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

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trades(
                email, time, side, symbol, mode, size, sl, tp, leverage,
                entry_price, current_price, unreal_pnl_percent, unreal_pnl_value, equity_after,
                reason, session_id
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                email,
                tr.time,
                tr.side,
                tr.symbol,
                tr.mode,
                float(tr.size),
                float(tr.sl),
                float(tr.tp),
                float(tr.leverage),
                float(tr.entry_price),
                float(tr.current_price),
                float(tr.unreal_pnl_percent),
                float(tr.unreal_pnl_value),
                float(tr.equity_after),
                tr.reason,
                int(sid),
            ),
        )
        conn.commit()
        conn.close()

    set_equity(email, equity_after)
    return {"ok": True, "trade": asdict(tr), "pnl_pct": float(tr.unreal_pnl_percent)}


# =========================
# AUTO AI TRADER (V3)
# =========================
AUTO_LOCK = threading.Lock()
AUTO_RUNNERS: Dict[str, "AutoRunner"] = {}


def _ema(series: List[float], period: int) -> List[float]:
    if not series:
        return []
    k = 2.0 / (period + 1.0)
    out = [series[0]]
    for x in series[1:]:
        out.append((x - out[-1]) * k + out[-1])
    return out


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


class AutoStartIn(BaseModel):
    symbol: str
    tf: TF = "15m"
    interval_sec: int = 60
    mode: RiskMode = "MINI_ASYM"
    max_trades_per_day: Optional[int] = None
    stop_after_bad_trades: int = 2
    duration_days: int = 0
    trend_filter: bool = True
    chop_min_sep_pct: float = 0.005


class AutoRunner:
    def __init__(
        self,
        email: str,
        symbol: str,
        tf: str,
        interval_sec: int,
        mode: RiskMode,
        max_trades_per_day: int,
        stop_after_bad_trades: int,
        duration_days: int,
        trend_filter: bool,
        chop_min_sep_pct: float,
    ) -> None:
        self.email = email
        self.symbol = symbol.upper().strip()
        self.tf = tf
        self.interval_sec = int(max(5, min(3600, interval_sec)))
        self.mode = mode
        self.max_trades_per_day = int(max(0, max_trades_per_day))
        self.stop_after_bad_trades = int(max(0, stop_after_bad_trades))
        self.duration_days = int(max(0, duration_days))
        self.trend_filter = bool(trend_filter)
        self.chop_min_sep_pct = float(max(0.0, chop_min_sep_pct))

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)

        self.last_signal: str = "-"
        self.last_side: Side = "LONG"
        self.last_trade_ts: float = 0.0
        self.last_run_ts: float = 0.0
        self.blocked_reason: Optional[str] = None

        self.day_key = dubai_day_key()
        self.trades_today = 0
        self.bad_trades_today = 0

        self.end_at_ts: Optional[float] = None
        if self.duration_days > 0:
            end_dt = now_dubai() + timedelta(days=self.duration_days)
            self.end_at_ts = end_dt.timestamp()

        self.history: Deque[Dict[str, str]] = deque(maxlen=120)
        self._prev_sig: str = ""

    def is_running(self) -> bool:
        return not self.stop_event.is_set()

    def log(self, msg: str) -> None:
        self.history.appendleft({"t": now_utc_str(), "msg": msg})

    def start(self) -> None:
        self.log("AI started.")
        self.thread.start()

    def stop(self, reason: str = "Stopped by user.") -> None:
        self.log(reason)
        self.stop_event.set()

    def _reset_if_new_day(self) -> None:
        k = dubai_day_key()
        if k != self.day_key:
            self.day_key = k
            self.trades_today = 0
            self.bad_trades_today = 0
            self.log("Daily counters reset (Dubai timezone).")

    def _reset_in_sec(self) -> int:
        now_dt = now_dubai()
        next_midnight = (now_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int((next_midnight - now_dt).total_seconds())

    def status(self) -> AutoState:
        last_run = datetime.utcfromtimestamp(self.last_run_ts).strftime("%Y-%m-%d %H:%M:%S") if self.last_run_ts else "-"
        last_trade = datetime.utcfromtimestamp(self.last_trade_ts).strftime("%Y-%m-%d %H:%M:%S") if self.last_trade_ts else "-"
        end_at = datetime.fromtimestamp(self.end_at_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if self.end_at_ts else None
        return AutoState(
            running=self.is_running(),
            email=self.email,
            symbol=self.symbol,
            tf=self.tf,
            interval_sec=self.interval_sec,
            mode=self.mode,
            side=self.last_side,
            last_signal=self.last_signal or "-",
            blocked_reason=self.blocked_reason,
            last_run_at=last_run,
            last_trade_at=last_trade,
            max_trades_per_day=self.max_trades_per_day,
            trades_today=self.trades_today,
            stop_after_bad_trades=self.stop_after_bad_trades,
            bad_trades_today=self.bad_trades_today,
            reset_in_sec=self._reset_in_sec(),
            duration_days=self.duration_days,
            end_at=end_at,
            trend_filter=self.trend_filter,
            chop_min_sep_pct=self.chop_min_sep_pct,
        )

    def _signal_and_filters(self) -> Dict[str, Any]:
        closes = [k["close"] for k in _fetch_klines_sync(self.symbol, self.tf, limit=260)]
        if len(closes) < 210:
            return {"ok": False, "blocked": "NO_DATA", "signal": "NO_DATA"}

        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        ema50 = _ema(closes, 50)
        ema200 = _ema(closes, 200)

        sig = "EMA9>EMA21" if ema9[-1] > ema21[-1] else "EMA9<EMA21"
        desired_side: Side = "LONG" if sig == "EMA9>EMA21" else "SHORT"

        if self.trend_filter:
            if desired_side == "LONG" and not (ema50[-1] > ema200[-1]):
                return {"ok": False, "blocked": "TREND_FILTER: LONG requires EMA50>EMA200", "signal": sig}
            if desired_side == "SHORT" and not (ema50[-1] < ema200[-1]):
                return {"ok": False, "blocked": "TREND_FILTER: SHORT requires EMA50<EMA200", "signal": sig}

        price = closes[-1]
        sep_pct = abs(ema50[-1] - ema200[-1]) / max(1e-9, price)
        if sep_pct < self.chop_min_sep_pct:
            return {"ok": False, "blocked": f"CHOP_FILTER: sep {sep_pct*100:.3f}% < {self.chop_min_sep_pct*100:.3f}%", "signal": sig}

        return {"ok": True, "blocked": None, "signal": sig, "side": desired_side, "sep_pct": sep_pct}

    def _run_loop(self) -> None:
        cooldown_sec = max(60, self.interval_sec)

        while not self.stop_event.is_set():
            try:
                self._reset_if_new_day()
                self.last_run_ts = time.time()

                if self.end_at_ts and time.time() >= self.end_at_ts:
                    self.blocked_reason = "DURATION_ENDED"
                    self.log("Blocked: duration ended. Stopping AI.")
                    self.stop_event.set()
                    break

                if not get_exchange(self.email):
                    self.blocked_reason = "EXCHANGE_NOT_CONNECTED"
                    self.last_signal = "BLOCKED: EXCHANGE_NOT_CONNECTED"
                    self.log("Blocked: exchange not connected.")
                    time.sleep(self.interval_sec)
                    continue

                if self.max_trades_per_day > 0 and self.trades_today >= self.max_trades_per_day:
                    self.blocked_reason = f"MAX_TRADES_DAY: {self.trades_today}/{self.max_trades_per_day}"
                    self.last_signal = "BLOCKED: MAX_TRADES_DAY"
                    self.log(f"Blocked: max trades/day reached ({self.trades_today}/{self.max_trades_per_day}).")
                    time.sleep(self.interval_sec)
                    continue

                if self.stop_after_bad_trades > 0 and self.bad_trades_today >= self.stop_after_bad_trades:
                    self.blocked_reason = f"MAX_BAD_TRADES: {self.bad_trades_today}/{self.stop_after_bad_trades}"
                    self.last_signal = "BLOCKED: MAX_BAD_TRADES"
                    self.log("Blocked: max bad trades reached. Stopping.")
                    self.stop_event.set()
                    break

                res = self._signal_and_filters()
                self.last_signal = res.get("signal") or "-"
                if not res.get("ok"):
                    self.blocked_reason = res.get("blocked") or "BLOCKED"
                    self.log(f"Blocked: {self.blocked_reason} | signal={self.last_signal}")
                    time.sleep(self.interval_sec)
                    continue

                self.blocked_reason = None
                desired_side: Side = res["side"]
                self.last_side = desired_side

                now_ts = time.time()
                should_trade = False
                if self.last_trade_ts == 0.0:
                    should_trade = True
                elif self.last_signal != self._prev_sig and (now_ts - self.last_trade_ts) >= cooldown_sec:
                    should_trade = True

                self._prev_sig = self.last_signal

                if should_trade:
                    import asyncio

                    out = asyncio.run(
                        _place_trade_internal(
                            self.email,
                            self.symbol,
                            desired_side,
                            self.mode,
                            extra_reason=f"Auto AI Trader (V3): tf={self.tf}, interval={self.interval_sec}s, signal={self.last_signal}",
                        )
                    )
                    self.last_trade_ts = now_ts
                    self.trades_today += 1

                    pnl_pct = float(out.get("pnl_pct", 0.0))
                    if pnl_pct < 0:
                        self.bad_trades_today += 1
                        self.log(f"TRADE PLACED ({desired_side}) | pnl={pnl_pct:.3f}% | BAD {self.bad_trades_today}/{self.stop_after_bad_trades}")
                    else:
                        mt = self.max_trades_per_day if self.max_trades_per_day > 0 else 999999
                        self.log(f"TRADE PLACED ({desired_side}) | pnl={pnl_pct:.3f}% | trades_today={self.trades_today}/{mt}")
                else:
                    self.log(f"No trade: waiting | signal={self.last_signal}")

            except Exception as e:
                self.blocked_reason = "ERROR"
                self.last_signal = f"ERROR: {str(e)[:120]}"
                self.log(self.last_signal)
            finally:
                time.sleep(self.interval_sec)


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
    return {"ok": True, "equity": START_EQUITY, "new_session_id": new_sid}


@app.get("/auto/status")
def auto_status(user=Depends(require_user)):
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if not r:
            return {"ok": True, "running": False}
        st = r.status()
        return {"ok": True, **asdict(st)}


@app.get("/auto/history")
def auto_history(user=Depends(require_user), limit: int = Query(default=40, ge=1, le=200)):
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if not r:
            return {"ok": True, "events": []}
        return {"ok": True, "events": list(r.history)[: int(limit)]}


@app.post("/auto/start")
def auto_start(payload: AutoStartIn, user=Depends(require_user)):
    email = user["email"]
    symbol = payload.symbol.upper().strip()
    if not symbol.endswith("USDT"):
        raise HTTPException(status_code=400, detail="Use symbol like BTCUSDT / ETHUSDT / SOLUSDT")
    if payload.tf not in TF_MAP:
        raise HTTPException(status_code=400, detail="Bad timeframe")
    if payload.interval_sec < 5 or payload.interval_sec > 3600:
        raise HTTPException(status_code=400, detail="interval_sec must be 5..3600")

    max_trades = payload.max_trades_per_day if payload.max_trades_per_day is not None else default_max_trades_per_day(payload.mode)

    with AUTO_LOCK:
        old = AUTO_RUNNERS.get(email)
        if old:
            old.stop("Stopped: restarted with new settings.")
            del AUTO_RUNNERS[email]

        runner = AutoRunner(
            email=email,
            symbol=symbol,
            tf=payload.tf,
            interval_sec=payload.interval_sec,
            mode=payload.mode,
            max_trades_per_day=int(max_trades),
            stop_after_bad_trades=int(payload.stop_after_bad_trades),
            duration_days=int(payload.duration_days),
            trend_filter=bool(payload.trend_filter),
            chop_min_sep_pct=float(payload.chop_min_sep_pct),
        )
        AUTO_RUNNERS[email] = runner
        runner.start()

    return {
        "ok": True,
        "running": True,
        "symbol": symbol,
        "tf": payload.tf,
        "interval_sec": payload.interval_sec,
        "mode": payload.mode,
        "max_trades_per_day": int(max_trades),
    }


@app.post("/auto/stop")
def auto_stop(user=Depends(require_user)):
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            r.stop("Stopped by user.")
            del AUTO_RUNNERS[email]
    return {"ok": True, "running": False}


# =========================
# ADMIN ENDPOINTS
# =========================
class AdminSettingsIn(BaseModel):
    signup_enabled: bool
    seat_capacity: int


@app.get("/admin/status")
def admin_status(admin=Depends(require_admin)):
    return {
        "ok": True,
        "admin": admin,
        "signup_enabled": signup_is_enabled(),
        "seat_capacity": seat_capacity(),
        "seats_used": seats_used(),
        "seats_remaining": max(0, seat_capacity() - seats_used()),
        "auto_runners": list(AUTO_RUNNERS.keys()),
    }


@app.get("/admin/settings")
def admin_settings(admin=Depends(require_admin)):
    return {
        "signup_enabled": signup_is_enabled(),
        "seat_capacity": seat_capacity(),
        "seats_used": seats_used(),
        "seats_remaining": max(0, seat_capacity() - seats_used()),
    }


@app.post("/admin/settings")
def admin_update_settings(payload: AdminSettingsIn, admin=Depends(require_admin)):
    sc = int(max(1, min(100000, payload.seat_capacity)))
    admin_set_setting("signup_enabled", "true" if payload.signup_enabled else "false")
    admin_set_setting("seat_capacity", str(sc))
    return {"ok": True, **admin_settings(admin)}


@app.post("/admin/stop-all-ai")
def admin_stop_all_ai(admin=Depends(require_admin)):
    stopped = []
    with AUTO_LOCK:
        for email, runner in list(AUTO_RUNNERS.items()):
            runner.stop("Stopped by admin.")
            stopped.append(email)
            del AUTO_RUNNERS[email]
    return {"ok": True, "stopped_ai_for": stopped}


@app.post("/admin/reset-user")
def admin_reset_user(email: str, admin=Depends(require_admin)):
    email = (email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")
    ensure_user_state(email)
    set_equity(email, START_EQUITY)
    set_session_id(email, get_session_id(email) + 1)
    return {"ok": True, "reset_user": email, "equity": START_EQUITY}


@app.get("/admin/users")
def admin_users(
    admin=Depends(require_admin),
    q: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=2000),
):
    qn = (q or "").strip().lower()

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()

        if qn:
            cur.execute(
                """
                SELECT u.email, u.created_at,
                       COALESCE(s.equity, ?) AS equity,
                       COALESCE(s.session_id, 0) AS session_id
                FROM users u
                LEFT JOIN user_state s ON s.email = u.email
                WHERE LOWER(u.email) LIKE ?
                ORDER BY u.created_at DESC
                LIMIT ?
                """,
                (START_EQUITY, f"%{qn}%", int(limit)),
            )
        else:
            cur.execute(
                """
                SELECT u.email, u.created_at,
                       COALESCE(s.equity, ?) AS equity,
                       COALESCE(s.session_id, 0) AS session_id
                FROM users u
                LEFT JOIN user_state s ON s.email = u.email
                ORDER BY u.created_at DESC
                LIMIT ?
                """,
                (START_EQUITY, int(limit)),
            )

        rows = cur.fetchall()
        out = []
        for r in rows:
            email = r["email"]
            cur.execute("SELECT 1 FROM exchange_keys WHERE email=? LIMIT 1", (email,))
            ex = bool(cur.fetchone())

            cur.execute("SELECT COUNT(*) AS n FROM trades WHERE email=?", (email,))
            tcount = int(cur.fetchone()["n"])

            with AUTO_LOCK:
                rr = AUTO_RUNNERS.get(email)
                running = bool(rr and rr.is_running())

            out.append(
                {
                    "email": email,
                    "created_at": r["created_at"],
                    "equity": float(r["equity"]),
                    "session_id": int(r["session_id"]),
                    "exchange_connected": ex,
                    "trades_count": tcount,
                    "ai_running": running,
                }
            )

        conn.close()

    return {"ok": True, "users": out}


@app.get("/admin/user/{email}")
def admin_user_details(
    email: str,
    admin=Depends(require_admin),
    trades_limit: int = Query(default=50, ge=1, le=500),
):
    email = (email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()

        cur.execute("SELECT email, created_at FROM users WHERE email=?", (email,))
        u = cur.fetchone()
        if not u:
            conn.close()
            raise HTTPException(status_code=404, detail="User not found")

        cur.execute("SELECT equity, session_id FROM user_state WHERE email=?", (email,))
        st = cur.fetchone()
        equity = float(st["equity"]) if st else START_EQUITY
        session_id = int(st["session_id"]) if st else 0

        cur.execute("SELECT exchange, created_at FROM exchange_keys WHERE email=?", (email,))
        ex = cur.fetchone()

        cur.execute("SELECT COUNT(*) AS n FROM trades WHERE email=?", (email,))
        tcount = int(cur.fetchone()["n"])

        cur.execute("SELECT * FROM trades WHERE email=? ORDER BY id DESC LIMIT ?", (email, int(trades_limit)))
        trows = [dict(r) for r in cur.fetchall()]

        conn.close()

    with AUTO_LOCK:
        rr = AUTO_RUNNERS.get(email)
        running = bool(rr and rr.is_running())
        auto_status_obj = asdict(rr.status()) if running and rr else None

    return {
        "ok": True,
        "user": {"email": u["email"], "created_at": u["created_at"]},
        "state": {"equity": equity, "session_id": session_id},
        "exchange": {
            "connected": bool(ex),
            "exchange": ex["exchange"] if ex else None,
            "connected_at": ex["created_at"] if ex else None,
        },
        "trades": {"count": tcount, "recent": trows},
        "ai": {"running": running, "status": auto_status_obj},
    }
