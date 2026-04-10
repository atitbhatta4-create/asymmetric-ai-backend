from __future__ import annotations

import os
import secrets
import time
import hashlib
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Literal, Any, Deque
from collections import deque

import httpx
from fastapi import FastAPI, Depends, HTTPException, Response, Cookie, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, USING_PG, column_exists, serial_pk

# =========================
# App
# =========================
app = FastAPI(title="Asymmetric AI Backend", version="0.9.0")

# =========================
# ENV / PROD SETTINGS
# =========================
ENV = os.getenv("ENV", "dev").lower().strip()  # dev | prod
IS_PROD = ENV == "prod"

FRONTEND_ORIGINS_RAW = os.getenv("FRONTEND_ORIGINS", "")
if FRONTEND_ORIGINS_RAW.strip():
    FRONTEND_ORIGINS = [x.strip() for x in FRONTEND_ORIGINS_RAW.split(",") if x.strip()]
else:
    FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

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

# Strip /api prefix forwarded by Vercel
@app.middleware("http")
async def strip_api_prefix(request: Request, call_next):
    path = request.scope.get("path", "")
    if path == "/api":
        request.scope["path"] = "/"
    elif path.startswith("/api/"):
        request.scope["path"] = path[4:]
    return await call_next(request)

# =========================
# DB LOCK (thread safety)
# =========================
DB_LOCK = threading.Lock()


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    _serial = serial_pk()

    with DB_LOCK:
        conn = db()
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
        CREATE TABLE IF NOT EXISTS password_resets (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        )
        """)

        # Safe migrations for existing databases
        if not column_exists(conn, "trades", "reason"):
            cur.execute("ALTER TABLE trades ADD COLUMN reason TEXT")
        if not column_exists(conn, "trades", "session_id"):
            cur.execute("ALTER TABLE trades ADD COLUMN session_id INTEGER NOT NULL DEFAULT 0")
        if not column_exists(conn, "user_state", "session_id"):
            cur.execute("ALTER TABLE user_state ADD COLUMN session_id INTEGER NOT NULL DEFAULT 0")

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
        conn.close()


init_db()

# =========================
# CONFIG
# =========================
OKX_BASE = "https://www.okx.com"
OKX_TF_MAP = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}

# =========================
# HELPERS
# =========================
START_EQUITY = 1000.0
RiskMode = Literal["ULTRA_SAFE", "SAFE", "NORMAL", "MINI_ASYM", "AGGRESSIVE"]
Side = Literal["LONG", "SHORT"]
TF = Literal["15m", "1h", "4h", "1d"]
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
        cur.execute("SELECT equity, session_id FROM user_state WHERE email = %s", (email,))
        row = cur.fetchone()
        if not row:
            cur.execute(
                "INSERT INTO user_state(email, equity, session_id) VALUES(%s, %s, %s)",
                (email, START_EQUITY, 0),
            )
            conn.commit()
        conn.close()


def get_equity(email: str) -> float:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT equity FROM user_state WHERE email = %s", (email,))
        row = cur.fetchone()
        conn.close()
    return float(row["equity"])


def set_equity(email: str, equity: float) -> None:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET equity = %s WHERE email = %s", (float(equity), email))
        conn.commit()
        conn.close()


def get_session_id(email: str) -> int:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT session_id FROM user_state WHERE email = %s", (email,))
        row = cur.fetchone()
        conn.close()
    return int(row["session_id"] or 0)


def set_session_id(email: str, sid: int) -> None:
    ensure_user_state(email)
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET session_id = %s WHERE email = %s", (int(sid), email))
        conn.commit()
        conn.close()


def get_exchange(email: str) -> Optional[dict]:
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM exchange_keys WHERE email = %s", (email,))
        row = cur.fetchone()
        conn.close()
    return row


def require_user(session: Optional[str] = Cookie(default=None)) -> Dict[str, str]:
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email FROM sessions WHERE token = %s", (session,))
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
        cur.execute("SELECT v FROM admin_settings WHERE k=%s", (key,))
        row = cur.fetchone()
        conn.close()
    return (row["v"] if row else default)


def admin_set_setting(key: str, value: str) -> None:
    with DB_LOCK:
        conn = db()
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
    cur.execute("SELECT email FROM users WHERE email = %s", (DEMO_EMAIL,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users(email, password_hash, created_at) VALUES(%s, %s, %s)",
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
        "market": "okx",
        "timezone": "Asia/Dubai (UTC+4)",
        "env": ENV,
        "frontend_origins": FRONTEND_ORIGINS,
        "db": "postgresql" if USING_PG else "sqlite",
    }


@app.get("/health")
def health():
    return {
        "health": "green",
        "market": "okx",
        "db": "postgresql" if USING_PG else "sqlite",
        "env": ENV,
    }


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
# COOKIE SETTINGS
# =========================
def set_session_cookie(response: Response, token: str) -> None:
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
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="User already exists")

        cur.execute(
            "INSERT INTO users(email, password_hash, created_at) VALUES(%s, %s, %s)",
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
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        conn.close()

    if not row or row["password_hash"] != hash_pw(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(32)
    now = int(time.time())

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("INSERT INTO sessions(token, email, created_at) VALUES(%s, %s, %s)", (token, email, now))
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
            cur.execute("DELETE FROM sessions WHERE token = %s", (session,))
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
        cur.execute("SELECT email FROM sessions WHERE token = %s", (session,))
        row = cur.fetchone()
        conn.close()
    return {"ok": bool(row), "email": row["email"] if row else None}


# =========================
# FORGOT PASSWORD
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
        cur.execute("SELECT email FROM users WHERE email=%s", (email,))
        u = cur.fetchone()
        if not u:
            conn.close()
            return {"ok": True}

        token = secrets.token_urlsafe(32)
        cur.execute(
            "INSERT INTO password_resets(token,email,created_at,used) VALUES(%s,%s,%s,0)",
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
        cur.execute("SELECT email, used, created_at FROM password_resets WHERE token=%s", (token,))
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
        cur.execute("UPDATE users SET password_hash=%s WHERE email=%s", (hash_pw(new_pw), email))
        cur.execute("UPDATE password_resets SET used=1 WHERE token=%s", (token,))
        cur.execute("DELETE FROM sessions WHERE email=%s", (email,))
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
        cur.execute("SELECT email FROM users WHERE email=%s", (email,))
        u = cur.fetchone()
        if not u:
            cur.execute(
                "INSERT INTO users(email, password_hash, created_at) VALUES(%s,%s,%s)",
                (email, hash_pw(payload.new_password), now_utc_str()),
            )
        else:
            cur.execute("UPDATE users SET password_hash=%s WHERE email=%s", (hash_pw(payload.new_password), email))
        cur.execute("DELETE FROM sessions WHERE email=%s", (email,))
        conn.commit()
        conn.close()

    return {"ok": True}


# =========================
# EXCHANGE CONNECT
# =========================
class ExchangeConnectIn(BaseModel):
    exchange: Literal["binance", "okx", "bybit"]
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
                payload.api_key.strip(),
                payload.api_secret.strip(),
                (payload.passphrase or "").strip(),
                created_at,
            ),
        )
        conn.commit()
        conn.close()

    return {"ok": True}


@app.post("/exchange/disconnect")
def exchange_disconnect(user=Depends(require_user)):
    email = user["email"]
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("DELETE FROM exchange_keys WHERE email = %s", (email,))
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
# MARKET (PUBLIC) — OKX
# =========================
def to_okx_inst(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if "-" in s:
        return s
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
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
    arr = (r.json() or {}).get("data") or []
    if not arr:
        raise HTTPException(status_code=400, detail=f"OKX: no ticker for {inst}")
    last = arr[0].get("last")
    if last is None:
        raise HTTPException(status_code=400, detail=f"OKX: missing last price for {inst}")
    return float(last)


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
        rows = (r.json() or {}).get("data") or []
        if not rows:
            return []
        out: List[Dict[str, Any]] = []
        for k in rows:
            out.append({"t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])})
        out.reverse()
        return out
    except Exception:
        return []


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


@app.get("/price/{symbol}")
async def get_price(symbol: str, exchange: Optional[str] = Query(default=None)):
    sym = symbol.upper().strip()
    price = await _route_price(sym, exchange or "okx")
    return {"symbol": sym, "price": price, "exchange": (exchange or "okx").lower()}


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
    size_dollar = equity * computed["size"] / 100.0
    effective = size_dollar * computed["leverage"]
    reduced = pct >= 10.0

    lines = [
        f"{mode}  •  Manual trade",
        "",
        "Account",
        f"  Equity:      ${equity:,.2f}  ({dir_word} {pct:.1f}% from ${START_EQUITY:.0f} start)",
        "",
        "Position",
        f"  Size:        {computed['size']:.2f}% of equity  →  ${size_dollar:.2f} at risk",
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
        cur.execute("SELECT time FROM trades WHERE email = %s ORDER BY id DESC LIMIT 1", (email,))
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

    entry = await binance_price(symbol)

    move = ((int(time.time()) % 140) - 70) / 1000.0
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
        conn.close()

    set_equity(email, equity_after)
    return {"ok": True, "trade": asdict(tr), "pnl_pct": float(tr.unreal_pnl_percent)}


# =========================
# AUTO AI TRADER (V3)
# =========================
AUTO_LOCK = threading.Lock()
AUTO_RUNNERS: Dict[str, "AutoRunner"] = {}


def get_ai_restart_lock(email: str) -> int:
    """Returns Unix timestamp until which AI restart is locked (0 = not locked)."""
    try:
        return int(admin_get_setting(f"ai_restart_lock:{email}", "0"))
    except Exception:
        return 0


def set_ai_restart_lock(email: str, until_ts: int) -> None:
    admin_set_setting(f"ai_restart_lock:{email}", str(until_ts))


def _ema(series: List[float], period: int) -> List[float]:
    if not series:
        return []
    k = 2.0 / (period + 1.0)
    out = [series[0]]
    for x in series[1:]:
        out.append((x - out[-1]) * k + out[-1])
    return out


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    return 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al))


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
           for i in range(1, len(closes))]
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period * 2 + 1:
        return None
    pdms, ndms, trs = [], [], []
    for i in range(1, len(closes)):
        hd = highs[i] - highs[i - 1]
        ld = lows[i - 1] - lows[i]
        pdms.append(hd if hd > ld and hd > 0 else 0.0)
        ndms.append(ld if ld > hd and ld > 0 else 0.0)
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))

    def _wilder(s: List[float]) -> List[float]:
        v = sum(s[:period])
        out = [v]
        for x in s[period:]:
            v = v - v / period + x
            out.append(v)
        return out

    s_tr, s_p, s_n = _wilder(trs), _wilder(pdms), _wilder(ndms)
    dxs = []
    for i in range(len(s_tr)):
        pdi = 100 * s_p[i] / s_tr[i] if s_tr[i] else 0
        ndi = 100 * s_n[i] / s_tr[i] if s_tr[i] else 0
        dxs.append(100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else 0)
    if len(dxs) < period:
        return None
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


# Per-mode signal filter thresholds — all 4 layers
MODE_SIGNAL_PARAMS: Dict[str, Dict] = {
    "ULTRA_SAFE": dict(adx_min=30, atr_min=0.003, atr_max=0.020, rsi_min=42, rsi_max=58, pullback_max=0.010, vol_factor=1.50, mom_n=3),
    "SAFE":       dict(adx_min=25, atr_min=0.002, atr_max=0.025, rsi_min=40, rsi_max=62, pullback_max=0.015, vol_factor=1.30, mom_n=2),
    "NORMAL":     dict(adx_min=20, atr_min=0.002, atr_max=0.030, rsi_min=38, rsi_max=65, pullback_max=0.020, vol_factor=1.20, mom_n=2),
    "MINI_ASYM":  dict(adx_min=18, atr_min=0.001, atr_max=0.035, rsi_min=35, rsi_max=68, pullback_max=0.025, vol_factor=1.10, mom_n=1),
    "AGGRESSIVE": dict(adx_min=15, atr_min=0.001, atr_max=0.040, rsi_min=32, rsi_max=70, pullback_max=0.030, vol_factor=1.00, mom_n=1),
}


def _compute_signal_layers(
    klines: List[Dict],
    mode: RiskMode,
    adaptive_strictness: float = 1.0,
) -> Dict:
    """
    Full 4-layer signal analysis. Returns breakdown dict used by both
    AutoRunner._signal_and_filters() and the /auto/signal endpoint.
    """
    if len(klines) < 220:
        return {"ok": False, "blocked": "NO_DATA", "signal": "NO_DATA", "breakdown": {}}

    closes  = [k["close"]  for k in klines]
    highs   = [k["high"]   for k in klines]
    lows    = [k["low"]    for k in klines]
    volumes = [k["volume"] for k in klines]

    ema9   = _ema(closes, 9)
    ema21  = _ema(closes, 21)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    rsi    = _rsi(closes, 14)
    atr    = _atr(highs, lows, closes, 14)
    adx    = _adx(highs, lows, closes, 14)

    price   = closes[-1]
    atr_pct = (atr / price) if atr else 0.0
    avg_vol = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else 0.0
    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 0.0

    sig          = "EMA9>EMA21" if ema9[-1] > ema21[-1] else "EMA9<EMA21"
    desired_side: Side = "LONG" if sig == "EMA9>EMA21" else "SHORT"

    # Apply Mini-Asym adaptive strictness
    p = MODE_SIGNAL_PARAMS.get(mode, MODE_SIGNAL_PARAMS["NORMAL"]).copy()
    if mode == "MINI_ASYM" and adaptive_strictness != 1.0:
        s = adaptive_strictness
        p["adx_min"]      = p["adx_min"] * s
        p["rsi_min"]      = min(50, p["rsi_min"] + (s - 1) * 8)
        p["rsi_max"]      = max(50, p["rsi_max"] - (s - 1) * 8)
        p["pullback_max"] = p["pullback_max"] / max(0.5, s)
        p["vol_factor"]   = p["vol_factor"] * s

    # ── Layer 1: Market regime (ADX + ATR) ──────────────────────────────
    adx_ok = adx is not None and adx >= p["adx_min"]
    atr_ok = p["atr_min"] <= atr_pct <= p["atr_max"]
    if not adx_ok:
        reg_reason = f"ADX {adx:.1f} < {p['adx_min']:.0f} — market choppy" if adx else "ADX unavailable"
    elif not atr_ok:
        reg_reason = f"ATR {atr_pct*100:.2f}% outside {p['atr_min']*100:.1f}–{p['atr_max']*100:.0f}%"
    else:
        reg_reason = ""
    breakdown_regime = {
        "adx": round(adx or 0, 1), "adx_min": round(p["adx_min"], 1),
        "atr_pct": round(atr_pct * 100, 3),
        "atr_range": f"{p['atr_min']*100:.1f}–{p['atr_max']*100:.0f}%",
        "ok": adx_ok and atr_ok, "reason": reg_reason,
    }

    # ── Layer 2: Direction bias (EMA50 vs EMA200 + price position) ──────
    bull = ema50[-1] > ema200[-1]
    if desired_side == "LONG":
        dir_ok     = bull and price > ema50[-1]
        dir_reason = "" if dir_ok else ("EMA50 < EMA200 — downtrend" if not bull else "Price below EMA50")
    else:
        dir_ok     = (not bull) and price < ema50[-1]
        dir_reason = "" if dir_ok else ("EMA50 > EMA200 — uptrend" if bull else "Price above EMA50")
    breakdown_direction = {
        "trend": "BULL" if bull else "BEAR", "signal": sig, "side": desired_side,
        "ok": dir_ok, "reason": dir_reason,
    }

    # ── Layer 3: Entry location (pullback to EMA21 + RSI) ───────────────
    pb_pct  = abs(price - ema21[-1]) / max(1e-9, ema21[-1])
    pb_ok   = pb_pct <= p["pullback_max"]
    rsi_ok  = rsi is not None and p["rsi_min"] <= rsi <= p["rsi_max"]
    if not pb_ok:
        ent_reason = f"Price {pb_pct*100:.1f}% from EMA21 — want <{p['pullback_max']*100:.0f}%"
    elif not rsi_ok:
        ent_reason = f"RSI {rsi:.0f} outside {p['rsi_min']}–{p['rsi_max']}"
    else:
        ent_reason = ""
    breakdown_entry = {
        "price_vs_ema21_pct": round(pb_pct * 100, 2),
        "pullback_max_pct": round(p["pullback_max"] * 100, 1),
        "rsi": round(rsi or 0, 1), "rsi_range": f"{p['rsi_min']}–{p['rsi_max']}",
        "ok": pb_ok and rsi_ok, "reason": ent_reason,
    }

    # ── Layer 4: Momentum (candles direction + volume) ───────────────────
    n = p["mom_n"]
    if desired_side == "LONG":
        mom_ok = all(klines[-(i + 1)]["close"] > klines[-(i + 1)]["open"] for i in range(n))
    else:
        mom_ok = all(klines[-(i + 1)]["close"] < klines[-(i + 1)]["open"] for i in range(n))
    vol_ok = vol_ratio >= p["vol_factor"]
    if not mom_ok:
        mom_reason = f"Last {n} candle(s) not aligned with {desired_side}"
    elif not vol_ok:
        mom_reason = f"Volume {vol_ratio:.1f}x avg — need {p['vol_factor']:.1f}x"
    else:
        mom_reason = ""
    breakdown_momentum = {
        "candles_n": n, "candles_ok": mom_ok,
        "volume_ratio": round(vol_ratio, 2), "vol_factor": round(p["vol_factor"], 2),
        "ok": mom_ok and vol_ok, "reason": mom_reason,
    }

    breakdown = {
        "regime":    breakdown_regime,
        "direction": breakdown_direction,
        "entry":     breakdown_entry,
        "momentum":  breakdown_momentum,
    }

    failed = [k for k, v in breakdown.items() if not v["ok"]]
    if failed:
        reasons = " | ".join(breakdown[k]["reason"] for k in failed if breakdown[k]["reason"])
        return {
            "ok": False,
            "blocked": f"BLOCKED ({', '.join(failed)}): {reasons}",
            "signal": sig, "side": desired_side,
            "breakdown": breakdown, "atr_pct": atr_pct,
        }

    return {
        "ok": True, "blocked": None,
        "signal": sig, "side": desired_side,
        "breakdown": breakdown, "atr_pct": atr_pct,
    }


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
    pending_trade: Optional[Dict]


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
    def __init__(self, email, symbol, tf, interval_sec, mode,
                 max_trades_per_day, stop_after_bad_trades, duration_days,
                 trend_filter, chop_min_sep_pct):
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
        # Set to now so first trade requires a full cooldown — prevents blind trade on start
        self.last_trade_ts: float = time.time()
        self.last_run_ts: float = 0.0
        self.blocked_reason: Optional[str] = None
        self.day_key = dubai_day_key()
        self.end_at_ts: Optional[float] = None
        if self.duration_days > 0:
            end_dt = now_dubai() + timedelta(days=self.duration_days)
            self.end_at_ts = end_dt.timestamp()
        self.history: Deque[Dict[str, str]] = deque(maxlen=120)
        self._prev_sig: str = ""
        self.pending_trade: Optional[Dict] = None
        self.adaptive_strictness: float = 1.0
        self.last_breakdown: Dict = {}
        self.session_start_equity: float = get_equity(email)
        # Load today's real trade counts from DB — prevents bypass by stop+restart
        self.trades_today, self.bad_trades_today = self._load_today_stats()

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
            with DB_LOCK:
                conn = db()
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) as cnt, "
                    "SUM(CASE WHEN unreal_pnl_percent < 0 THEN 1 ELSE 0 END) as bad "
                    "FROM trades WHERE email = %s AND time >= %s AND session_id = %s",
                    (self.email, utc_midnight_str, sid),
                )
                row = cur.fetchone()
                conn.close()
            trades = int((row or {}).get("cnt") or 0)
            bad = int((row or {}).get("bad") or 0)
            return trades, bad
        except Exception:
            return 0, 0

    def log(self, msg):
        self.history.appendleft({"t": now_utc_str(), "msg": msg})

    def start(self):
        self.log("AI started.")
        self.thread.start()

    def stop(self, reason="Stopped by user."):
        self.log(reason)
        self.stop_event.set()
        if self.pending_trade:
            self.log(f"Pending trade abandoned at stop (entry={self.pending_trade.get('entry_price', '?')}).")
            self.pending_trade = None

    def _reset_if_new_day(self):
        k = dubai_day_key()
        if k != self.day_key:
            self.day_key = k
            self.trades_today = 0
            self.bad_trades_today = 0
            self.log("Daily counters reset (Dubai timezone).")

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
            pending_trade=self.pending_trade,
        )

    def _fetch_price_sync(self, symbol: str) -> float:
        """Sync OKX price fetch for use inside runner thread."""
        inst = to_okx_inst(symbol)
        r = httpx.get(
            f"{OKX_BASE}/api/v5/market/ticker",
            params={"instId": inst},
            timeout=8,
            headers={"accept": "application/json"},
        )
        arr = (r.json() or {}).get("data") or []
        if not arr:
            raise RuntimeError(f"No price data for {symbol}")
        return float(arr[0]["last"])

    def _close_pending_trade(self) -> None:
        """Fetch real exit price, apply SL/TP, record trade to DB, update adaptive strictness."""
        pt = self.pending_trade
        if not pt:
            return
        try:
            exit_price = self._fetch_price_sync(self.symbol)
            entry = float(pt["entry_price"])
            side: Side = pt["side"]
            mode: RiskMode = pt["mode"]

            equity_before = get_equity(self.email)
            risk = mini_asym_risk_engine(mode, equity_before)
            c = risk["computed"]

            sl_pct = c["sl"] / 100.0
            tp_pct = c["tp"] / 100.0

            raw_move = (exit_price - entry) / entry if side == "LONG" else (entry - exit_price) / entry

            if raw_move <= -sl_pct:
                final_move = -sl_pct
                outcome = "SL_HIT"
            elif raw_move >= tp_pct:
                final_move = tp_pct
                outcome = "TP_HIT"
            else:
                final_move = raw_move
                outcome = "NATURAL_CLOSE"

            pnl_pct_leveraged = final_move * c["leverage"]
            pnl_value = equity_before * (c["size"] / 100.0) * pnl_pct_leveraged
            equity_after = equity_before + pnl_value

            from_start = equity_after - START_EQUITY
            dir_word = "up" if from_start >= 0 else "down"
            size_dollar = equity_before * c["size"] / 100.0
            outcome_label = (
                "Take profit hit" if outcome == "TP_HIT"
                else "Stop loss hit" if outcome == "SL_HIT"
                else "Natural close"
            )
            reason_text = "\n".join([
                f"{mode}  •  AI Trade  •  {self.tf} timeframe",
                "",
                f"Signal       {pt.get('signal', '-')}  →  {side}",
                f"Entry        ${entry:,.4f}",
                f"Exit         ${exit_price:,.4f}  ({raw_move * 100:+.3f}% price move)",
                f"Outcome      {outcome_label}  →  {pnl_pct_leveraged * 100:+.2f}% PnL  (${pnl_value:+.2f})",
                "",
                "Position",
                f"  Size:        {c['size']:.2f}% of equity  →  ${size_dollar:.2f}",
                f"  Leverage:    {c['leverage']:.0f}×",
                f"  Stop loss:   {c['sl']:.2f}%   |   Take profit: {c['tp']:.2f}%",
                "",
                "Account",
                f"  Before:      ${equity_before:,.2f}",
                f"  After:       ${equity_after:,.2f}  ({dir_word} {abs(from_start / START_EQUITY * 100):.2f}% from start)",
                f"  Strictness:  {self.adaptive_strictness:.2f}×  (Mini-Asym adapts after each trade)",
            ])

            sid = get_session_id(self.email)
            tr = Trade(
                time=now_utc_str(), side=side, symbol=self.symbol, mode=mode,
                size=float(c["size"]), sl=float(c["sl"]), tp=float(c["tp"]),
                leverage=float(c["leverage"]), entry_price=entry,
                current_price=exit_price,
                unreal_pnl_percent=float(pnl_pct_leveraged * 100.0),
                unreal_pnl_value=float(pnl_value),
                equity_after=float(equity_after), reason=reason_text,
            )

            with DB_LOCK:
                conn = db()
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO trades(
                        email, time, side, symbol, mode, size, sl, tp, leverage,
                        entry_price, current_price, unreal_pnl_percent, unreal_pnl_value,
                        equity_after, reason, session_id
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (self.email, tr.time, tr.side, tr.symbol, tr.mode,
                     float(tr.size), float(tr.sl), float(tr.tp), float(tr.leverage),
                     float(tr.entry_price), float(tr.current_price),
                     float(tr.unreal_pnl_percent), float(tr.unreal_pnl_value),
                     float(tr.equity_after), tr.reason, int(sid)),
                )
                conn.commit()
                conn.close()

            set_equity(self.email, equity_after)
            self.trades_today += 1
            if pnl_value < 0:
                self.bad_trades_today += 1

            # Mini-Asym adaptive strictness: tighten after loss, relax after win
            if mode == "MINI_ASYM":
                if pnl_value < 0:
                    self.adaptive_strictness = min(2.5, self.adaptive_strictness + 0.25)
                    self.log(f"MINI_ASYM strictness ↑ {self.adaptive_strictness:.2f} (after loss)")
                else:
                    self.adaptive_strictness = max(1.0, self.adaptive_strictness - 0.10)

            mt = self.max_trades_per_day if self.max_trades_per_day > 0 else "∞"
            self.log(
                f"TRADE CLOSED ({side}) | {outcome} | "
                f"entry={entry:.2f} exit={exit_price:.2f} | "
                f"pnl={pnl_pct_leveraged * 100:.2f}% (${pnl_value:.2f}) | "
                f"trades={self.trades_today}/{mt}"
            )

        except Exception as e:
            self.log(f"Error closing trade: {e}")
        finally:
            self.pending_trade = None

    def _signal_and_filters(self):
        klines = _fetch_klines_sync(self.symbol, self.tf, limit=260)
        res = _compute_signal_layers(klines, self.mode, self.adaptive_strictness)
        self.last_breakdown = res.get("breakdown", {})
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
            # New day — reload counters from DB and resume
            self.trades_today, self.bad_trades_today = self._load_today_stats()
            self.day_key = dubai_day_key()
            self._prev_sig = ""  # reset signal memory so first trade waits for a real change
            self.last_trade_ts = time.time()
            self.log("New Dubai day — daily limits reset. Resuming AI.")

    def _run_loop(self):
        cooldown_sec = max(60, self.interval_sec)
        while not self.stop_event.is_set():
            try:
                self._reset_if_new_day()
                self.last_run_ts = time.time()

                if self.end_at_ts and time.time() >= self.end_at_ts:
                    self.blocked_reason = "DURATION_ENDED"
                    if self.pending_trade:
                        self._close_pending_trade()
                    self.log("Session complete — duration ended. AI stopped.")
                    self.stop_event.set()
                    break

                if not get_exchange(self.email):
                    self.blocked_reason = "EXCHANGE_NOT_CONNECTED"
                    self.last_signal = "BLOCKED: EXCHANGE_NOT_CONNECTED"
                    self.log("Blocked: exchange not connected.")
                    time.sleep(self.interval_sec)
                    continue

                # ── Step 1: Close any pending trade first (real exit after one interval) ──
                if self.pending_trade:
                    self._close_pending_trade()
                    time.sleep(self.interval_sec)
                    continue

                # ── Step 2: Daily limit checks ──────────────────────────────────────────
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
                        break
                    continue

                # ── Step 3: 4-layer signal analysis ─────────────────────────────────────
                res = self._signal_and_filters()
                self.last_signal = res.get("signal") or "-"
                if not res.get("ok"):
                    self.blocked_reason = res.get("blocked") or "BLOCKED"
                    self.log(f"Blocked: {self.blocked_reason[:100]}")
                    time.sleep(self.interval_sec)
                    continue

                self.blocked_reason = None
                desired_side: Side = res["side"]
                self.last_side = desired_side

                now_ts = time.time()
                # Trade only on genuine signal change + cooldown elapsed
                should_trade = (
                    self.last_signal != self._prev_sig
                    and (now_ts - self.last_trade_ts) >= cooldown_sec
                )
                self._prev_sig = self.last_signal

                if should_trade:
                    # ── Step 4: Open trade — fetch real entry price ───────────────────
                    entry_price = self._fetch_price_sync(self.symbol)
                    self.pending_trade = {
                        "entry_price": entry_price,
                        "side": desired_side,
                        "mode": self.mode,
                        "signal": self.last_signal,
                        "open_ts": time.time(),
                    }
                    self.last_trade_ts = now_ts
                    self.log(f"TRADE OPENED ({desired_side}) @ {entry_price:.4f} | signal={self.last_signal}")
                else:
                    layers = [k for k, v in res.get("breakdown", {}).items()]
                    self.log(f"Waiting | signal={self.last_signal} | layers={layers}")

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
    set_ai_restart_lock(email, 0)  # Clear restart lock for demo purposes
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
            return {
                "ok": True, "running": False,
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
        if not r:
            return {"ok": True, "events": []}
        return {"ok": True, "events": list(r.history)[: int(limit)]}


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
            "pending_trade": r.pending_trade,
            "breakdown": r.last_breakdown,
        }


@app.post("/auto/start")
def auto_start(payload: AutoStartIn, user=Depends(require_user)):
    email = user["email"]
    symbol = payload.symbol.upper().strip()
    if not (symbol.endswith("USDT") or symbol.endswith("-USDT")):
        raise HTTPException(status_code=400, detail="Use symbol like BTCUSDT / ETHUSDT / SOLUSDT")
    if payload.tf not in TF_MAP:
        raise HTTPException(status_code=400, detail="Bad timeframe")
    if payload.interval_sec < 5 or payload.interval_sec > 3600:
        raise HTTPException(status_code=400, detail="interval_sec must be 5..3600")

    max_trades = payload.max_trades_per_day if payload.max_trades_per_day is not None else default_max_trades_per_day(payload.mode)

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
        with DB_LOCK:
            conn = db()
            cur = conn.cursor()
            cur.execute(
                "SELECT SUM(CASE WHEN unreal_pnl_percent < 0 THEN 1 ELSE 0 END) as bad "
                "FROM trades WHERE email = %s AND time >= %s AND session_id = %s",
                (email, utc_midnight_str, sid),
            )
            row = cur.fetchone()
            conn.close()
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

    with AUTO_LOCK:
        old = AUTO_RUNNERS.get(email)
        if old:
            old.stop("Stopped: restarted with new settings.")
            del AUTO_RUNNERS[email]

        runner = AutoRunner(
            email=email, symbol=symbol, tf=payload.tf,
            interval_sec=payload.interval_sec, mode=payload.mode,
            max_trades_per_day=int(max_trades),
            stop_after_bad_trades=int(payload.stop_after_bad_trades),
            duration_days=int(payload.duration_days),
            trend_filter=bool(payload.trend_filter),
            chop_min_sep_pct=float(payload.chop_min_sep_pct),
        )
        AUTO_RUNNERS[email] = runner
        runner.start()

    return {"ok": True, "running": True, "symbol": symbol, "tf": payload.tf,
            "interval_sec": payload.interval_sec, "mode": payload.mode,
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
    return {"ok": True, "running": False, "restart_locked": restart_locked, "restart_lock_sec": lock_sec}


# =========================
# ADMIN ENDPOINTS
# =========================
class AdminSettingsIn(BaseModel):
    signup_enabled: bool
    seat_capacity: int


@app.get("/admin/status")
def admin_status(admin=Depends(require_admin)):
    return {
        "ok": True, "admin": admin,
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
                       COALESCE(s.equity, %s) AS equity,
                       COALESCE(s.session_id, 0) AS session_id
                FROM users u
                LEFT JOIN user_state s ON s.email = u.email
                WHERE LOWER(u.email) LIKE %s
                ORDER BY u.created_at DESC
                LIMIT %s
                """,
                (START_EQUITY, f"%{qn}%", int(limit)),
            )
        else:
            cur.execute(
                """
                SELECT u.email, u.created_at,
                       COALESCE(s.equity, %s) AS equity,
                       COALESCE(s.session_id, 0) AS session_id
                FROM users u
                LEFT JOIN user_state s ON s.email = u.email
                ORDER BY u.created_at DESC
                LIMIT %s
                """,
                (START_EQUITY, int(limit)),
            )

        rows = cur.fetchall()
        out = []
        for r in rows:
            email = r["email"]
            cur.execute("SELECT 1 FROM exchange_keys WHERE email=%s LIMIT 1", (email,))
            ex = bool(cur.fetchone())
            cur.execute("SELECT COUNT(*) AS n FROM trades WHERE email=%s", (email,))
            tcount = int(cur.fetchone()["n"])
            with AUTO_LOCK:
                rr = AUTO_RUNNERS.get(email)
                running = bool(rr and rr.is_running())
            out.append({
                "email": email,
                "created_at": r["created_at"],
                "equity": float(r["equity"]),
                "session_id": int(r["session_id"]),
                "exchange_connected": ex,
                "trades_count": tcount,
                "ai_running": running,
            })
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

        cur.execute("SELECT email, created_at FROM users WHERE email=%s", (email,))
        u = cur.fetchone()
        if not u:
            conn.close()
            raise HTTPException(status_code=404, detail="User not found")

        cur.execute("SELECT equity, session_id FROM user_state WHERE email=%s", (email,))
        st = cur.fetchone()
        equity = float(st["equity"]) if st else START_EQUITY
        session_id = int(st["session_id"]) if st else 0

        cur.execute("SELECT exchange, created_at FROM exchange_keys WHERE email=%s", (email,))
        ex = cur.fetchone()

        cur.execute("SELECT COUNT(*) AS n FROM trades WHERE email=%s", (email,))
        tcount = int(cur.fetchone()["n"])

        cur.execute("SELECT * FROM trades WHERE email=%s ORDER BY id DESC LIMIT %s", (email, int(trades_limit)))
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
        "exchange": {"connected": bool(ex), "exchange": ex["exchange"] if ex else None, "connected_at": ex["created_at"] if ex else None},
        "trades": {"count": tcount, "recent": trows},
        "ai": {"running": running, "status": auto_status_obj},
    }
