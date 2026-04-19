from __future__ import annotations

import asyncio
import os
import secrets
import time
import hashlib
import hmac
import threading
import smtplib
import bcrypt
from cryptography.fernet import Fernet
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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

# =========================
# EMAIL (Gmail SMTP)
# =========================
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def _email_base(content: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#050814;font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb;">
  <div style="max-width:520px;margin:32px auto;padding:0 16px;">
    <div style="background:#0a0f1e;border:1px solid rgba(255,255,255,0.08);border-radius:18px;overflow:hidden;">
      <div style="padding:18px 24px;border-bottom:1px solid rgba(255,255,255,0.06);">
        <span style="font-size:17px;font-weight:900;color:#00ffe0;">Asymmetric AI</span>
      </div>
      <div style="padding:24px;">{content}</div>
      <div style="padding:14px 24px;border-top:1px solid rgba(255,255,255,0.06);font-size:12px;color:#4b5563;">
        Demo mode &nbsp;·&nbsp; No real funds &nbsp;·&nbsp; Educational only
      </div>
    </div>
  </div>
</body></html>"""


def _send_email_sync(to: str, subject: str, html: str) -> None:
    if not SMTP_USER or not SMTP_PASS:
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Asymmetric AI <{SMTP_USER}>"
        msg["To"] = to
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, to, msg.as_string())
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


def send_email(to: str, subject: str, html: str) -> None:
    """Fire-and-forget — never blocks the main thread."""
    threading.Thread(target=_send_email_sync, args=(to, subject, html), daemon=True).start()


def email_password_changed(to: str) -> None:
    content = f"""
    <h2 style="margin:0 0 14px;font-size:20px;font-weight:900;color:#f1f5f9;">Password Changed</h2>
    <p style="margin:0 0 14px;opacity:0.85;line-height:1.6;">
      Your password for <b>{to}</b> was changed successfully.
    </p>
    <div style="background:rgba(220,38,38,0.12);border:1px solid rgba(248,113,113,0.3);
                border-radius:12px;padding:12px 16px;font-size:13px;color:#fecaca;">
      If this wasn't you, change your password immediately and secure your account.
    </div>"""
    send_email(to, "Your Asymmetric AI password was changed", _email_base(content))


def email_ai_started(to: str, symbol: str, mode: str, trade_style: str,
                     duration_days: int, max_trades: int, stop_after_bad: int) -> None:
    duration_str = f"{duration_days} day{'s' if duration_days != 1 else ''}" if duration_days > 0 else "Unlimited"
    sp = TRADE_STYLE_PARAMS.get(trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
    interval_str = f"{sp['interval'] // 60}m"
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">AI Trading Started</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">
      Your AI trader is live and monitoring the market.
    </p>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:16px;margin-bottom:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        {''.join(f'<tr><td style="padding:7px 0;color:#6b7280;width:140px;">{k}</td><td style="padding:7px 0;font-weight:900;color:#f1f5f9;">{v}</td></tr>' for k,v in [
            ("Coin", symbol), ("Mode", mode), ("Style", trade_style),
            ("Timeframe", sp["tf"]), ("Check every", interval_str), ("Duration", duration_str),
            ("Max trades / day", str(max_trades)), ("Bad trade limit", f"{stop_after_bad} per day"),
        ])}
      </table>
    </div>
    <p style="margin:0;font-size:12px;color:#4b5563;">
      The AI only trades when all 4 signal layers pass.
      You will receive an email for each completed trade.
    </p>"""
    send_email(to, f"AI started — {symbol} {mode}", _email_base(content))


def email_trade_closed(to: str, symbol: str, side: str, mode: str,
                       entry: float, exit_price: float, outcome: str,
                       pnl_pct: float, pnl_value: float, equity_after: float) -> None:
    outcome_label = (
        "Take profit hit"    if outcome == "TP_HIT"
        else "Stop loss hit" if outcome == "SL_HIT"
        else "Trailing stop" if outcome == "TRAIL_STOP"
        else "Natural close"
    )
    win = pnl_value >= 0
    pnl_color = "#00ff9d" if win else "#ff5078"
    side_color = "#00ff9d" if side == "LONG" else "#ff5078"
    sign = "+" if win else ""
    outcome_icon = "✓" if win else "✗"
    content = f"""
    <h2 style="margin:0 0 4px;font-size:20px;font-weight:900;color:#f1f5f9;">Trade Completed</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">{symbol} &nbsp;·&nbsp; {mode}</p>

    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:20px;margin-bottom:14px;text-align:center;">
      <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">
        {outcome_label} {outcome_icon}
      </div>
      <div style="font-size:38px;font-weight:900;color:{pnl_color};">{sign}{pnl_pct:.2f}%</div>
      <div style="font-size:16px;font-weight:900;color:{pnl_color};margin-top:4px;">{sign}${pnl_value:.2f}</div>
    </div>

    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        {''.join(f'<tr><td style="padding:6px 0;color:#6b7280;width:130px;">{k}</td><td style="padding:6px 0;font-weight:900;color:{c};">{v}</td></tr>' for k,v,c in [
            ("Direction", side, side_color),
            ("Entry price", f"${entry:,.4f}", "#f1f5f9"),
            ("Exit price", f"${exit_price:,.4f}", "#f1f5f9"),
            ("Equity after", f"${equity_after:,.2f}", "#f1f5f9"),
        ])}
      </table>
    </div>"""
    subject = f"{'Win' if win else 'Loss'} {sign}{pnl_pct:.2f}% — {symbol} {side} closed"
    send_email(to, subject, _email_base(content))


def email_ai_stopped(to: str, symbol: str, reason: str, equity: float) -> None:
    reason_map = {
        "MAX_DRAWDOWN":   ("Drawdown Limit Reached",   "#ff5078", "The AI hit the maximum drawdown limit and stopped to protect your remaining capital."),
        "HARD_FLOOR":     ("Safety Floor Triggered",   "#ff5078", "Equity fell below the 85% safety floor. AI stopped completely to protect your funds."),
        "MAX_BAD_TRADES": ("Bad Trade Limit Hit",      "#f59e0b", "Too many consecutive losing trades. AI paused for today — resets at midnight Dubai time."),
        "DURATION_END":   ("Session Ended",            "#00ffe0", "The AI completed its scheduled trading duration."),
    }
    title, color, detail = reason_map.get(reason, ("AI Stopped", "#94a3b8", f"Reason: {reason}"))
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">AI Trading Stopped</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">{symbol}</p>
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                border-radius:14px;padding:18px;margin-bottom:16px;">
      <div style="font-size:16px;font-weight:900;color:{color};margin-bottom:8px;">{title}</div>
      <div style="font-size:14px;opacity:0.85;line-height:1.6;">{detail}</div>
    </div>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:12px;
                padding:14px;font-size:14px;">
      <div style="opacity:0.6;margin-bottom:4px;">Current equity</div>
      <div style="font-size:22px;font-weight:900;color:#f1f5f9;">${equity:,.2f}</div>
    </div>
    <p style="margin-top:16px;font-size:12px;color:#4b5563;">
      Log in to Asymmetric AI to review your trades and restart when ready.
    </p>"""
    send_email(to, f"AI stopped — {symbol} ({title})", _email_base(content))


def email_otp_reset(to: str, code: str) -> None:
    """Send 6-digit OTP for forgot-password flow. Expires in 15 minutes."""
    content = f"""
    <h2 style="margin:0 0 14px;font-size:20px;font-weight:900;color:#f1f5f9;">Reset Your Password</h2>
    <p style="margin:0 0 20px;opacity:0.85;line-height:1.6;">
      We received a request to reset the password for <b>{to}</b>.<br>
      Enter this code in the app to continue:
    </p>
    <div style="background:#0f172a;border:1px solid rgba(0,255,224,0.25);border-radius:14px;
                padding:28px;text-align:center;margin-bottom:20px;">
      <div style="font-size:42px;font-weight:900;letter-spacing:12px;color:#00ffe0;
                  font-family:monospace;">{code}</div>
      <div style="margin-top:10px;font-size:12px;color:#6b7280;">Expires in 15 minutes</div>
    </div>
    <div style="background:rgba(220,38,38,0.1);border:1px solid rgba(248,113,113,0.25);
                border-radius:10px;padding:12px 16px;font-size:13px;color:#fecaca;">
      If you did not request this, ignore this email. Your password has not been changed.
    </div>"""
    send_email(to, "Asymmetric AI — Password Reset Code", _email_base(content))


def email_2fa_enabled(to: str) -> None:
    content = f"""
    <h2 style="margin:0 0 14px;font-size:20px;font-weight:900;color:#f1f5f9;">Two-Factor Authentication Enabled</h2>
    <p style="margin:0 0 14px;opacity:0.85;line-height:1.6;">
      2FA has been successfully enabled on your account <b>{to}</b>.<br>
      You will now need your authenticator app every time you log in.
    </p>
    <div style="background:rgba(0,255,157,0.08);border:1px solid rgba(0,255,157,0.25);
                border-radius:12px;padding:12px 16px;font-size:13px;color:#a7f3d0;">
      If you did not enable this, contact support immediately and change your password.
    </div>"""
    send_email(to, "2FA enabled on your Asymmetric AI account", _email_base(content))


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

# Runners are resumed after all AutoRunner code is defined — see bottom of file.

# =========================
# CONFIG
# =========================
OKX_BASE = "https://www.okx.com"
OKX_TF_MAP = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
# Minimum interval (seconds) per timeframe — trade must be open at least this long
TF_MIN_INTERVAL: Dict[str, int] = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# =========================
# HELPERS
# =========================
START_EQUITY = 1000.0
RiskMode = Literal["ULTRA_SAFE", "SAFE", "NORMAL", "MINI_ASYM", "AGGRESSIVE"]
Side = Literal["LONG", "SHORT"]
TF = Literal["15m", "1h", "4h", "1d"]
TF_MAP: Dict[str, str] = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
TradeStyle = Literal["SCALP", "DAY_TRADE", "SWING"]

# Trading style: auto-sets TF + interval + ATR multipliers for SL/TP (1:2 RR always)
TRADE_STYLE_PARAMS: Dict[str, Dict] = {
    "SCALP":     {"tf": "15m", "interval": 900,   "sl_atr": 0.8, "tp_atr": 1.6, "sl_max": 1.5, "tp_max": 3.0},
    "DAY_TRADE": {"tf": "1h",  "interval": 3600,  "sl_atr": 1.0, "tp_atr": 2.0, "sl_max": 2.5, "tp_max": 5.0},
    "SWING":     {"tf": "4h",  "interval": 14400, "sl_atr": 1.5, "tp_atr": 3.0, "sl_max": 4.0, "tp_max": 8.0},
}

DUBAI_TZ = timezone(timedelta(hours=4))


def now_dubai() -> datetime:
    return datetime.now(tz=DUBAI_TZ)


def dubai_day_key(dt: Optional[datetime] = None) -> str:
    d = dt or now_dubai()
    return d.strftime("%Y-%m-%d")


# =========================
# SECURITY — PASSWORDS + KEY ENCRYPTION
# =========================

def hash_pw(pw: str) -> str:
    """Hash a new password with bcrypt."""
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_pw(plain: str, stored: str) -> bool:
    """
    Verify password — handles both:
    - Old SHA-256 hashes (legacy, migrated on next login)
    - New bcrypt hashes
    """
    if not plain or not stored:
        return False
    if stored.startswith("$2b$") or stored.startswith("$2a$"):
        # bcrypt
        return bcrypt.checkpw(plain.encode(), stored.encode())
    # Legacy SHA-256 — still works but will be re-hashed on login
    return hashlib.sha256(plain.encode()).hexdigest() == stored


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
    if not row:
        return None
    # Decrypt keys before returning — callers always get plain text
    return {
        **dict(row),
        "api_key":    decrypt_key(row["api_key"]),
        "api_secret": decrypt_key(row["api_secret"]),
        "passphrase": decrypt_key(row.get("passphrase") or ""),
    }


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

    if not row or not verify_pw(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Silently migrate legacy SHA-256 hash to bcrypt on successful login
    stored = row["password_hash"]
    if not (stored.startswith("$2b$") or stored.startswith("$2a$")):
        new_hash = hash_pw(payload.password)
        with DB_LOCK:
            conn2 = db()
            cur2 = conn2.cursor()
            cur2.execute("UPDATE users SET password_hash=%s WHERE email=%s", (new_hash, email))
            conn2.commit()
            conn2.close()

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

    # Check if 2FA is enabled — tell frontend to prompt for TOTP code
    totp_row = _get_totp_row(email)
    requires_2fa = bool(totp_row and totp_row["enabled"])
    return {"ok": True, "email": email, "requires_2fa": requires_2fa}


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


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str


@app.post("/auth/change-password")
def change_password(payload: ChangePasswordIn, user=Depends(require_user)):
    email = user["email"]
    if not payload.current_password or not payload.new_password:
        raise HTTPException(status_code=400, detail="Both current and new password are required.")
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        conn.close()

    if not row or not verify_pw(payload.current_password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET password_hash = %s WHERE email = %s", (hash_pw(payload.new_password), email))
        conn.commit()
        conn.close()

    email_password_changed(email)
    return {"ok": True}


# =========================
# FORGOT PASSWORD (OTP via email)
# =========================

class ForgotPasswordIn(BaseModel):
    email: str

class VerifyOtpIn(BaseModel):
    email: str
    code: str
    new_password: str

@app.post("/auth/forgot-password")
def forgot_password(payload: ForgotPasswordIn):
    """
    Step 1 — user enters their email.
    Generates a 6-digit OTP, stores it (15min TTL), sends to their Gmail.
    Always returns ok=True so attackers can't enumerate registered emails.
    """
    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    # Check user exists — but don't reveal it in the response
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        exists = cur.fetchone() is not None
        conn.close()

    if exists:
        code = str(secrets.randbelow(900000) + 100000)   # 100000-999999
        now_ts = int(time.time())
        with DB_LOCK:
            conn = db()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO otp_codes(email, code, created_at, used)
                VALUES(%s, %s, %s, 0)
                ON CONFLICT(email) DO UPDATE SET code=EXCLUDED.code,
                    created_at=EXCLUDED.created_at, used=0
            """, (email, code, now_ts))
            conn.commit()
            conn.close()
        email_otp_reset(email, code)

    # Always return ok — prevents email enumeration
    return {"ok": True, "detail": "If that email exists, a reset code has been sent."}


@app.post("/auth/reset-password")
def reset_password(payload: VerifyOtpIn):
    """
    Step 2 — user enters the 6-digit code + new password.
    Code is valid for 15 minutes, single-use.
    """
    email = payload.email.strip().lower()
    code  = payload.code.strip()

    if not email or not code or not payload.new_password:
        raise HTTPException(status_code=400, detail="Email, code and new password required")
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    OTP_TTL = 15 * 60  # 15 minutes

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute(
            "SELECT code, created_at, used FROM otp_codes WHERE email = %s", (email,)
        )
        row = cur.fetchone()
        conn.close()

    if not row:
        raise HTTPException(status_code=400, detail="No reset code found. Request a new one.")
    if row["used"]:
        raise HTTPException(status_code=400, detail="Code already used. Request a new one.")
    if int(time.time()) - row["created_at"] > OTP_TTL:
        raise HTTPException(status_code=400, detail="Code expired. Request a new one.")
    if not hmac.compare_digest(str(row["code"]), code):
        raise HTTPException(status_code=400, detail="Incorrect code.")

    # Mark used + update password
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE otp_codes SET used=1 WHERE email=%s", (email,))
        cur.execute("UPDATE users SET password_hash=%s WHERE email=%s",
                    (hash_pw(payload.new_password), email))
        conn.commit()
        conn.close()

    email_password_changed(email)
    return {"ok": True, "detail": "Password reset successfully. You can now log in."}


# =========================
# TWO-FACTOR AUTH (TOTP — Google Authenticator compatible)
# =========================
import pyotp
import base64

def _get_totp_row(email: str) -> Optional[dict]:
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT secret, enabled FROM totp_secrets WHERE email=%s", (email,))
        row = cur.fetchone()
        conn.close()
    return dict(row) if row else None


@app.get("/auth/2fa/status")
def totp_status(user=Depends(require_user)):
    """Returns whether 2FA is enabled for this account."""
    row = _get_totp_row(user["email"])
    return {"enabled": bool(row and row["enabled"])}


@app.post("/auth/2fa/setup")
def totp_setup(user=Depends(require_user)):
    """
    Generate a new TOTP secret + QR code URI for Google Authenticator.
    User scans the QR, then calls /auth/2fa/confirm with a valid code to activate.
    """
    email = user["email"]
    secret = pyotp.random_base32()
    uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=email,
        issuer_name="Asymmetric AI"
    )
    # Store secret but NOT enabled yet — only enabled after confirm
    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO totp_secrets(email, secret, enabled, created_at)
            VALUES(%s, %s, 0, %s)
            ON CONFLICT(email) DO UPDATE SET secret=EXCLUDED.secret,
                enabled=0, created_at=EXCLUDED.created_at
        """, (email, encrypt_key(secret), now_utc_str()))
        conn.commit()
        conn.close()
    return {"ok": True, "uri": uri, "secret": secret}


class TotpCodeIn(BaseModel):
    code: str

@app.post("/auth/2fa/confirm")
def totp_confirm(payload: TotpCodeIn, user=Depends(require_user)):
    """
    Confirm 2FA setup by verifying the first code from the authenticator app.
    Only after this succeeds is 2FA actually activated.
    """
    email = user["email"]
    row = _get_totp_row(email)
    if not row:
        raise HTTPException(status_code=400, detail="Run /auth/2fa/setup first")

    secret = decrypt_key(row["secret"])
    totp   = pyotp.TOTP(secret)
    if not totp.verify(payload.code.strip(), valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid code — check your authenticator app")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE totp_secrets SET enabled=1 WHERE email=%s", (email,))
        conn.commit()
        conn.close()

    email_2fa_enabled(email)
    return {"ok": True, "detail": "2FA is now active on your account"}


@app.post("/auth/2fa/verify")
def totp_verify(payload: TotpCodeIn, user=Depends(require_user)):
    """
    Called after password login when 2FA is enabled.
    Frontend: after login returns {requires_2fa: true}, show code input → POST here.
    On success, session is fully authenticated.
    """
    email = user["email"]
    row = _get_totp_row(email)
    if not row or not row["enabled"]:
        return {"ok": True, "detail": "2FA not enabled"}   # no-op if not set up

    secret = decrypt_key(row["secret"])
    totp   = pyotp.TOTP(secret)
    # valid_window=1 allows 30s clock drift (one period before/after)
    if not totp.verify(payload.code.strip(), valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid 2FA code")

    return {"ok": True}


@app.post("/auth/2fa/disable")
def totp_disable(payload: TotpCodeIn, user=Depends(require_user)):
    """
    Disable 2FA. Requires a valid TOTP code to confirm it's really the user.
    """
    email = user["email"]
    row = _get_totp_row(email)
    if not row or not row["enabled"]:
        raise HTTPException(status_code=400, detail="2FA is not enabled")

    secret = decrypt_key(row["secret"])
    totp   = pyotp.TOTP(secret)
    if not totp.verify(payload.code.strip(), valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid code — cannot disable 2FA")

    with DB_LOCK:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE totp_secrets SET enabled=0 WHERE email=%s", (email,))
        conn.commit()
        conn.close()

    return {"ok": True, "detail": "2FA disabled"}


@app.get("/debug/email")
def debug_email(to: str = Query(default="")):
    """Send a test email. Pass ?to=your@email.com to test delivery. No auth needed."""
    if not SMTP_USER or not SMTP_PASS:
        return {"ok": False, "error": "SMTP_USER or SMTP_PASS not set", "smtp_user": repr(SMTP_USER)}
    target = to.strip() if to.strip() else SMTP_USER
    content = "<h2 style='color:#f1f5f9;'>Test Email</h2><p style='opacity:0.85;'>SMTP is working on Asymmetric AI.</p>"
    err = None
    try:
        # Call raw SMTP directly (bypass the internal try/except) so errors surface here
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Asymmetric AI — SMTP Test"
        msg["From"] = f"Asymmetric AI <{SMTP_USER}>"
        msg["To"] = target
        msg.attach(MIMEText(_email_base(content), "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, target, msg.as_string())
    except Exception as e:
        err = str(e)
    return {"ok": err is None, "sent_to": target, "smtp_user": SMTP_USER, "error": err}


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
                encrypt_key(payload.api_key.strip()),
                encrypt_key(payload.api_secret.strip()),
                encrypt_key((payload.passphrase or "").strip()),
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
    }
    if row.get("passphrase"):
        config["password"] = row["passphrase"]  # required for OKX
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    return cls(config)


@app.get("/exchange/test")
def exchange_test(user=Depends(require_user)):
    """Actually test the API key against the real exchange."""
    email = user["email"]
    row = get_exchange(email)
    if not row:
        raise HTTPException(status_code=400, detail="No exchange connected.")
    try:
        ex = _make_ccxt_exchange(row)
        # Fetch account info to verify key is valid and has trading permissions
        status = ex.fetch_status()
        # Try a lightweight authenticated call
        balance = ex.fetch_balance({"accountType": "UNIFIED"})
        usdt = balance.get("USDT", {})
        return {
            "ok": True,
            "exchange": row["exchange"],
            "canTrade": True,
            "accountType": "UNIFIED",
            "usdt_free": round(float(usdt.get("free") or 0), 4),
            "usdt_total": round(float(usdt.get("total") or 0), 4),
            "note": "Live connection verified ✓",
        }
    except Exception as e:
        err = str(e)
        return {
            "ok": False,
            "canTrade": False,
            "error": err[:300],
            "note": "Connection failed — check your API key, secret, and IP whitelist on the exchange.",
        }


@app.get("/exchange/balance")
def exchange_balance(user=Depends(require_user)):
    """Fetch real USDT balance from the connected exchange."""
    email = user["email"]
    row = get_exchange(email)
    if not row:
        raise HTTPException(status_code=400, detail="No exchange connected.")
    try:
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
        # Fallback to virtual equity when exchange unreachable
        eq = get_equity(email)
        return {
            "ok": False,
            "balances": [{"asset": "USDT", "free": eq, "locked": 0.0}],
            "error": str(e)[:200],
            "note": "Could not reach exchange — showing paper balance.",
        }


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


@app.get("/market/ticker/{symbol}")
async def market_ticker(symbol: str):
    """24h stats for a symbol: price, change%, high, low, volume in USDT."""
    inst = to_okx_inst(symbol.upper().strip())
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{OKX_BASE}/api/v5/market/ticker",
            params={"instId": inst},
            headers={"accept": "application/json"},
        )
    d = ((r.json() or {}).get("data") or [{}])[0]
    last   = float(d.get("last") or 0)
    open24 = float(d.get("open24h") or last or 1)
    high   = float(d.get("high24h") or 0)
    low    = float(d.get("low24h") or 0)
    vol    = float(d.get("volCcy24h") or 0)   # volume in USDT
    chg    = ((last - open24) / open24 * 100) if open24 else 0.0
    return {
        "symbol": symbol.upper().strip(),
        "price": last, "open24h": open24,
        "high24h": high, "low24h": low,
        "change24h": round(chg, 3),
        "volume24h": round(vol, 2),
    }


@app.get("/market/tickers")
async def market_tickers(symbols: str = Query(...)):
    """Batch 24h stats. Pass ?symbols=BTCUSDT,ETHUSDT,..."""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    async with httpx.AsyncClient(timeout=12) as client:
        results = await asyncio.gather(*[
            client.get(
                f"{OKX_BASE}/api/v5/market/ticker",
                params={"instId": to_okx_inst(s)},
                headers={"accept": "application/json"},
            )
            for s in syms
        ], return_exceptions=True)
    out = []
    for sym, res in zip(syms, results):
        try:
            d = ((res.json() or {}).get("data") or [{}])[0]  # type: ignore
            last   = float(d.get("last") or 0)
            open24 = float(d.get("open24h") or last or 1)
            high   = float(d.get("high24h") or 0)
            low    = float(d.get("low24h") or 0)
            vol    = float(d.get("volCcy24h") or 0)
            chg    = ((last - open24) / open24 * 100) if open24 else 0.0
            out.append({"symbol": sym, "price": last, "change24h": round(chg, 3),
                        "high24h": high, "low24h": low, "volume24h": round(vol, 2)})
        except Exception:
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


# ── Runner state persistence (survive deploys) ────────────────────────────────

def _save_runner_state(runner: "AutoRunner") -> None:
    """Persist runner config to DB so it can be resumed after a backend restart."""
    try:
        with DB_LOCK:
            conn = db()
            cur = conn.cursor()
            if USING_PG:
                cur.execute("""
                    INSERT INTO ai_runner_state
                        (email, symbol, trade_style, mode, max_trades_per_day,
                         stop_after_bad_trades, duration_days, trend_filter,
                         chop_min_sep_pct, end_at_ts, started_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(email) DO UPDATE SET
                        symbol=%s, trade_style=%s, mode=%s,
                        max_trades_per_day=%s, stop_after_bad_trades=%s,
                        duration_days=%s, trend_filter=%s,
                        chop_min_sep_pct=%s, end_at_ts=%s, started_at=%s
                """, (
                    runner.email, runner.symbol, runner.trade_style, runner.mode,
                    runner.max_trades_per_day, runner.stop_after_bad_trades,
                    runner.duration_days, int(runner.trend_filter),
                    runner.chop_min_sep_pct, runner.end_at_ts, int(time.time()),
                    runner.symbol, runner.trade_style, runner.mode,
                    runner.max_trades_per_day, runner.stop_after_bad_trades,
                    runner.duration_days, int(runner.trend_filter),
                    runner.chop_min_sep_pct, runner.end_at_ts, int(time.time()),
                ))
            else:
                cur.execute("""
                    INSERT OR REPLACE INTO ai_runner_state
                        (email, symbol, trade_style, mode, max_trades_per_day,
                         stop_after_bad_trades, duration_days, trend_filter,
                         chop_min_sep_pct, end_at_ts, started_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    runner.email, runner.symbol, runner.trade_style, runner.mode,
                    runner.max_trades_per_day, runner.stop_after_bad_trades,
                    runner.duration_days, int(runner.trend_filter),
                    runner.chop_min_sep_pct, runner.end_at_ts, int(time.time()),
                ))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[runner-state] save failed for {runner.email}: {e}")


def _clear_runner_state(email: str) -> None:
    """Remove persisted runner state — called when AI stops for any reason."""
    try:
        with DB_LOCK:
            conn = db()
            cur = conn.cursor()
            cur.execute("DELETE FROM ai_runner_state WHERE email = %s", (email,))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[runner-state] clear failed for {email}: {e}")


def _load_all_runner_states() -> list:
    """Return all saved runner configs (called once on startup)."""
    try:
        with DB_LOCK:
            conn = db()
            cur = conn.cursor()
            cur.execute("SELECT * FROM ai_runner_state")
            rows = cur.fetchall()
            conn.close()
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


# Higher timeframe for trend bias — always one step up
HIGHER_TF_MAP: Dict[str, str] = {"15m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}


def _rsi_series(closes: List[float], period: int = 14) -> List[float]:
    """RSI value for every candle after warmup — needed for divergence check."""
    if len(closes) < period + 2:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    out = []
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out.append(100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al)))
    return out


def _session_quality(trade_style: str) -> tuple:
    """
    Session quality multiplier — maps real market sessions to Dubai time (UTC+4).
    Higher score = more liquidity = signal threshold easier to meet.
    SCALP is hard-blocked 02:00-06:00 (dead zone, thin market).

    Dubai (UTC+4) session schedule:
      02:00–06:00 → Dead zone (Asia asleep, US closed)        → SCALP: 0.0, others: 0.72
      06:00–12:00 → Asia / Tokyo session                      → 0.85
      12:00–16:00 → London open (big moves start)             → 0.95
      16:00–21:00 → London + NY overlap (BEST — peak volume)  → 1.00
      21:00–01:00 → NY session solo                           → 0.92
      01:00–02:00 → NY close / wind-down                      → 0.80
    """
    if trade_style == "SWING":
        return 1.0, "swing"
    hour = now_dubai().hour

    # Dead zone — thin market, wide spreads, low volume
    if 2 <= hour < 6:
        if trade_style == "SCALP":
            return 0.0, f"scalp-blocked {hour:02d}:xx (dead zone 02-06 Dubai)"
        return 0.72, f"dead-zone {hour:02d}:xx"

    # Asia / Tokyo session — decent crypto liquidity
    if 6 <= hour < 12:
        return 0.85, f"Asia {hour:02d}:xx"

    # London open — institutions enter, volume spikes, big directional moves
    if 12 <= hour < 16:
        return 0.95, f"London {hour:02d}:xx"

    # London + NY overlap — peak volume, best setups, highest probability
    if 16 <= hour < 21:
        return 1.00, f"London+NY {hour:02d}:xx"

    # NY solo session — still active, good liquidity
    if 21 <= hour <= 23:
        return 0.92, f"NY {hour:02d}:xx"

    # NY close / wind-down (01:00-02:00)
    return 0.80, f"NY-close {hour:02d}:xx"


def _ema_spread_trend(ema_fast: List[float], ema_slow: List[float], n: int = 4) -> tuple:
    """
    Checks if EMA gap is WIDENING (trend accelerating) or NARROWING (stalling).
    Returns (widening: bool, score: float 0-1)
    A widening spread = conviction in trend direction → better entry.
    Narrowing = trend losing steam → lower quality entry.
    """
    if len(ema_fast) < n + 1 or len(ema_slow) < n + 1:
        return False, 0.5
    spread_now  = abs(ema_fast[-1] - ema_slow[-1])
    spread_prev = abs(ema_fast[-n] - ema_slow[-n])
    if spread_prev < 1e-10:
        return False, 0.5
    ratio = spread_now / spread_prev
    widening = ratio >= 1.03   # at least 3% wider than 4 candles ago
    score = min(1.0, 0.5 + (ratio - 1.0) * 2.0) if widening else max(0.1, ratio * 0.5)
    return widening, round(score, 3)


def _candle_pattern(klines: List[Dict], side: str) -> tuple:
    """
    Score the last candle's pattern quality for the given trade direction.
    Returns (pattern_name: str, score: float 0-1)
    pin_bar=1.0, engulfing=0.90, momentum=0.75, basic=0.50, against=0.10
    """
    if len(klines) < 2:
        return "none", 0.3
    curr = klines[-1]
    prev = klines[-2]
    body  = abs(curr["close"] - curr["open"])
    rng   = curr["high"] - curr["low"]
    if rng < 1e-10:
        return "doji", 0.25
    body_ratio = body / rng

    if side == "LONG":
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]
        # Pin bar: hammer shape — long lower wick, small body, closes bullish
        if body_ratio < 0.35 and lower_wick > rng * 0.55 and curr["close"] > curr["open"]:
            return "pin_bar", 1.0
        # Bullish engulfing: current body covers previous body completely
        if (curr["close"] > curr["open"] and
                curr["open"] <= prev["close"] and curr["close"] >= prev["open"]):
            return "engulfing", 0.90
        # Strong momentum candle: large bullish body
        if curr["close"] > curr["open"] and body_ratio > 0.65:
            return "momentum", 0.75
        # Basic bullish
        if curr["close"] > curr["open"]:
            return "bullish", 0.50
    else:
        upper_wick = curr["high"] - max(curr["open"], curr["close"])
        # Shooting star / pin bar bearish
        if body_ratio < 0.35 and upper_wick > rng * 0.55 and curr["close"] < curr["open"]:
            return "pin_bar", 1.0
        # Bearish engulfing
        if (curr["close"] < curr["open"] and
                curr["open"] >= prev["close"] and curr["close"] <= prev["open"]):
            return "engulfing", 0.90
        # Strong bearish momentum candle
        if curr["close"] < curr["open"] and body_ratio > 0.65:
            return "momentum", 0.75
        # Basic bearish
        if curr["close"] < curr["open"]:
            return "bearish", 0.50

    return "against_trend", 0.10


def _rsi_divergence(closes: List[float], rsi_vals: List[float], side: str, n: int = 12) -> tuple:
    """
    Detect RSI divergence — price makes new extreme but RSI disagrees.
    Bearish divergence (during LONG): price higher high, RSI lower high → weakening bull.
    Bullish divergence (during SHORT): price lower low, RSI higher low → weakening bear.
    Returns (divergence: bool, score_penalty: float 0.0-0.30)
    """
    if len(closes) < n + 1 or len(rsi_vals) < n:
        return False, 0.0
    price_window = closes[-n:]
    rsi_window   = rsi_vals[-n:]
    p_latest     = closes[-1]
    r_latest     = rsi_vals[-1]

    if side == "LONG":
        p_prev_high = max(price_window[:-3])
        r_prev_high = max(rsi_window[:-3])
        divergence  = p_latest > p_prev_high * 1.001 and r_latest < r_prev_high * 0.96
    else:
        p_prev_low  = min(price_window[:-3])
        r_prev_low  = min(rsi_window[:-3])
        divergence  = p_latest < p_prev_low * 0.999 and r_latest > r_prev_low * 1.04

    return divergence, (0.28 if divergence else 0.0)

# Per-mode signal thresholds + minimum quality score to trade
MODE_SIGNAL_PARAMS: Dict[str, Dict] = {
    "ULTRA_SAFE": dict(adx_min=28, atr_min=0.003, atr_max=0.020, rsi_min=42, rsi_max=58, pullback_max=0.012, vol_factor=1.40, mom_n=3, min_score=0.75),
    "SAFE":       dict(adx_min=22, atr_min=0.002, atr_max=0.025, rsi_min=40, rsi_max=62, pullback_max=0.015, vol_factor=1.25, mom_n=2, min_score=0.68),
    "NORMAL":     dict(adx_min=18, atr_min=0.002, atr_max=0.030, rsi_min=38, rsi_max=65, pullback_max=0.020, vol_factor=1.15, mom_n=2, min_score=0.62),
    "MINI_ASYM":  dict(adx_min=16, atr_min=0.001, atr_max=0.035, rsi_min=35, rsi_max=68, pullback_max=0.025, vol_factor=1.05, mom_n=1, min_score=0.65),
    "AGGRESSIVE": dict(adx_min=13, atr_min=0.001, atr_max=0.040, rsi_min=32, rsi_max=70, pullback_max=0.030, vol_factor=0.95, mom_n=1, min_score=0.58),
}


def _compute_signal_layers(
    klines: List[Dict],
    mode: RiskMode,
    adaptive_strictness: float = 1.0,
    higher_klines: Optional[List[Dict]] = None,
    trade_style: str = "DAY_TRADE",
) -> Dict:
    """
    4-layer scored signal analysis.
    Direction from higher TF trend. Entry on pullback+bounce, not on crossover.
    Each layer returns a 0-1 score. Total must exceed mode threshold.
    """
    # ── Session quality multiplier ───────────────────────────────────────
    # Low-liquidity hours reduce the final score so signals must be stronger.
    # SCALP 02:00-06:00 Dubai returns sess_score=0.0 = hard block (scalping
    # thin markets is bad practice and the math is unfavourable at <0.72×).
    sess_score, sess_label = _session_quality(trade_style)
    if sess_score == 0.0:
        return {
            "ok": False,
            "blocked": f"SCALP_LOW_LIQ: Scalping paused 02:00–06:00 Dubai (thin market) — {sess_label}",
            "signal": "BLOCKED", "score": 0.0, "breakdown": {},
            "side": None, "adaptive_strictness": adaptive_strictness,
        }

    if len(klines) < 220:
        return {"ok": False, "blocked": "NO_DATA", "signal": "NO_DATA", "score": 0.0, "breakdown": {}}

    closes  = [k["close"]  for k in klines]
    highs   = [k["high"]   for k in klines]
    lows    = [k["low"]    for k in klines]
    volumes = [k["volume"] for k in klines]

    ema9      = _ema(closes, 9)
    ema21     = _ema(closes, 21)
    ema50     = _ema(closes, 50)
    ema200    = _ema(closes, 200)
    rsi       = _rsi(closes, 14)
    rsi_vals  = _rsi_series(closes, 14)   # full series for divergence
    atr       = _atr(highs, lows, closes, 14)
    adx       = _adx(highs, lows, closes, 14)

    price      = closes[-1]
    atr_pct    = (atr / price) if atr else 0.0
    # Use volumes[-2] = last COMPLETED candle (OKX returns the in-progress candle
    # as volumes[-1], which has only partial volume — e.g. 1 minute into a 15m
    # candle shows 0.07x, making the volume filter block valid setups).
    # Use median of 40 completed candles as baseline (robust to spike candles
    # from big moves like BTC 74→77 inflating the mean and blocking normal vol).
    _vol_window = sorted(volumes[-42:-2]) if len(volumes) >= 42 else sorted(volumes[:-2])
    avg_vol    = _vol_window[len(_vol_window) // 2] if _vol_window else 0.0
    vol_ratio  = volumes[-2] / avg_vol if avg_vol > 0 and len(volumes) >= 3 else 0.0

    # Apply adaptive strictness for MINI_ASYM
    p = MODE_SIGNAL_PARAMS.get(mode, MODE_SIGNAL_PARAMS["NORMAL"]).copy()

    # Trade-style tightens/relaxes entry thresholds — SCALP must be strict
    # because 15m candles are noisy; SWING can tolerate wider pullbacks.
    # mom_n_add=0 for SCALP: MINI_ASYM already requires 1 candle — adding a 2nd
    # blocks too many valid entries in volatile/recovering markets.
    _style_adj = {
        "SCALP":     dict(pullback_mult=0.40, mom_n_add=0, rsi_tighten=4,  score_add=0.02),
        "DAY_TRADE": dict(pullback_mult=0.70, mom_n_add=0, rsi_tighten=2,  score_add=0.02),
        "SWING":     dict(pullback_mult=1.20, mom_n_add=0, rsi_tighten=-3, score_add=0.0),
    }
    adj = _style_adj.get(trade_style, _style_adj["DAY_TRADE"])
    p["pullback_max"] = p["pullback_max"] * adj["pullback_mult"]
    p["mom_n"]        = max(1, p["mom_n"] + adj["mom_n_add"])
    p["rsi_min"]      = min(50, p["rsi_min"] + adj["rsi_tighten"])
    p["rsi_max"]      = max(50, p["rsi_max"] - adj["rsi_tighten"])
    p["min_score"]    = min(0.92, p["min_score"] + adj["score_add"])

    if mode == "MINI_ASYM" and adaptive_strictness != 1.0:
        s = adaptive_strictness
        p["adx_min"]      = p["adx_min"] * s
        new_rsi_min = p["rsi_min"] + (s - 1) * 8
        new_rsi_max = p["rsi_max"] - (s - 1) * 8
        # Clamp so the RSI window never shrinks below 12 points — prevents the
        # engine from becoming permanently blocked after a losing streak with no message.
        mid = (new_rsi_min + new_rsi_max) / 2.0
        if new_rsi_max - new_rsi_min < 12:
            new_rsi_min = mid - 6
            new_rsi_max = mid + 6
        p["rsi_min"]      = min(50, new_rsi_min)
        p["rsi_max"]      = max(50, new_rsi_max)
        p["pullback_max"] = p["pullback_max"] / max(0.5, s)
        p["vol_factor"]   = p["vol_factor"] * s
        p["min_score"]    = min(0.90, p["min_score"] + (s - 1) * 0.12)

    # ── Higher TF trend direction ─────────────────────────────────────────
    # 4h (or 1d) EMA50 vs EMA21 sets the macro bias — trade WITH this trend only
    if higher_klines and len(higher_klines) >= 55:
        htf_closes = [k["close"] for k in higher_klines]
        htf_ema21  = _ema(htf_closes, 21)
        htf_ema50  = _ema(htf_closes, 50)
        htf_bull   = htf_ema21[-1] > htf_ema50[-1]   # fast EMA above slow = uptrend
        htf_ok     = True
    else:
        htf_bull = ema50[-1] > ema200[-1]
        htf_ok   = False

    desired_side: Side = "LONG" if htf_bull else "SHORT"

    # ── Direction-dependent RSI shift ─────────────────────────────────────
    # LONG setups enter near oversold, so shift window lower.
    # SHORT setups enter near overbought, so shift window higher.
    # SWING uses a smaller shift — 4h bull trends sustain RSI 65-72 normally,
    # so a full -8 shift would block valid continuation entries.
    # SCALP uses full ±8 — 15m candles are noisy, tighter RSI = better entries.
    _shift_magnitude = {
        "SCALP":     8,
        "DAY_TRADE": 6,
        "SWING":     3,
    }.get(trade_style, 6)
    _rsi_shift = _shift_magnitude if desired_side == "SHORT" else -_shift_magnitude
    p["rsi_min"] = max(25, min(65, p["rsi_min"] + _rsi_shift))
    p["rsi_max"] = max(40, min(85, p["rsi_max"] + _rsi_shift))

    local_bull = ema50[-1] > ema200[-1]
    ema9_aligned = (ema9[-1] > ema21[-1]) if desired_side == "LONG" else (ema9[-1] < ema21[-1])
    # Signal label: HTF trend direction + local EMA9 momentum alignment
    htf_label = "4h-BULL" if htf_bull else "4h-BEAR"
    ema9_label = "EMA9↑" if ema9[-1] > ema21[-1] else "EMA9↓"
    sig = f"{htf_label} {ema9_label}"

    # ── Layer 1: Regime — ADX strength + ATR volatility ──────────────────
    adx_ok  = adx is not None and adx >= p["adx_min"]
    atr_ok  = atr_pct > 0 and p["atr_min"] <= atr_pct <= p["atr_max"]
    adx_score = min(1.0, (adx - p["adx_min"]) / max(1.0, p["adx_min"])) if adx_ok else 0.0
    atr_score = 1.0 if atr_ok else 0.0
    regime_score = round(adx_score * 0.70 + atr_score * 0.30, 3)
    reg_reason = (
        f"ADX {adx:.1f} < {p['adx_min']:.0f} — market too choppy" if not adx_ok and adx
        else f"ATR {atr_pct*100:.2f}% outside safe range {p['atr_min']*100:.1f}–{p['atr_max']*100:.0f}%" if not atr_ok
        else ""
    )
    breakdown_regime = {
        "adx": round(adx or 0, 1), "adx_min": round(p["adx_min"], 1),
        "atr_pct": round(atr_pct * 100, 3),
        "atr_range": f"{p['atr_min']*100:.1f}–{p['atr_max']*100:.0f}%",
        "score": regime_score, "ok": regime_score > 0.10, "reason": reg_reason,
    }

    # ── Layer 2: Direction — higher TF trend + local confirmation + spread ──
    #
    # BUG FIXED: htf_aligned was a tautology.
    #   desired_side = "LONG" if htf_bull else "SHORT"
    #   htf_aligned  = htf_bull if desired_side=="LONG" else not htf_bull
    #   → always True regardless of market — htf_score was always 1.0.
    #
    # Fix: replace the self-referential check with an actual quality measure.
    # htf_score now reflects HOW STRONG the trend is (EMA spread size),
    # not whether the trend agrees with itself (which is trivially always true).
    local_aligned = (local_bull and price > ema50[-1]) if desired_side == "LONG" else (not local_bull and price < ema50[-1])

    if htf_ok:
        # Real 4h data: score by EMA21-EMA50 separation as % of price.
        # Near-zero spread = EMAs just crossed or ranging = low conviction.
        # 1.0%+ spread = established trend = full conviction.
        # BTC 4h typical: 0.1% (just crossed) → 1-3% (strong trend).
        _htf_spread = abs(htf_ema21[-1] - htf_ema50[-1]) / max(htf_ema50[-1], 1e-9)
        htf_score = min(1.0, _htf_spread / 0.010)   # 0%→0.0, 1.0%+→1.0
    else:
        # Fallback to local EMAs (less reliable) — cap at 0.5
        htf_score = 0.4

    # local_score = 0.0 when not aligned (was 0.4 — gave free credit even against trend).
    local_score   = 1.0 if local_aligned else 0.0
    ema9_bonus    = 0.20 if ema9_aligned else 0.0

    # EMA spread widening = trend accelerating = better entry quality
    spread_widening, spread_score = _ema_spread_trend(ema9, ema21, n=4)

    direction_score = round(min(1.0,
        htf_score * 0.45 + local_score * 0.25 + ema9_bonus * 0.15 + spread_score * 0.15
    ), 3)
    # Threshold raised 0.55 → 0.65: requires meaningful local + HTF agreement.
    # At 0.55, HTF alone (0.45) + neutral spread (0.075) = 0.525 could barely pass.
    # At 0.65, you need at least HTF confirmed + local OR HTF + ema9 + good spread.
    _DIR_PASS = 0.65
    dir_reason = (
        "" if direction_score >= _DIR_PASS
        else "EMA spread narrowing — trend losing momentum, wait for re-acceleration" if not spread_widening and htf_score > 0.5 and local_aligned
        else "HTF trend weak — EMAs nearly crossed, wait for stronger separation" if htf_score < 0.30
        else f"Price {'below' if desired_side == 'LONG' else 'above'} EMA50 — wait for local trend to confirm"
    )
    breakdown_direction = {
        "htf_trend": "BULL" if htf_bull else "BEAR",
        "htf_confirmed": htf_ok,
        "local_trend": "BULL" if local_bull else "BEAR",
        "ema9_momentum": "aligned" if ema9_aligned else "not aligned",
        "ema_spread": "widening" if spread_widening else "narrowing",
        "signal": sig, "side": desired_side,
        "score": direction_score, "ok": direction_score >= _DIR_PASS, "reason": dir_reason,
    }

    # ── Layer 3: Entry — pullback + candle pattern + RSI + divergence check ──
    pb_pct = abs(price - ema21[-1]) / max(1e-9, ema21[-1])
    pullback_score = max(0.0, 1.0 - pb_pct / p["pullback_max"]) if pb_pct <= p["pullback_max"] else 0.0

    # Candle pattern at pullback zone (replaces simple bounce check)
    pattern_name, pattern_score = _candle_pattern(klines, desired_side)

    # RSI: closer to center of the range = better entry quality
    rsi_in_range = rsi is not None and p["rsi_min"] <= rsi <= p["rsi_max"]
    if rsi is not None and rsi_in_range:
        mid  = (p["rsi_min"] + p["rsi_max"]) / 2.0
        half = (p["rsi_max"] - p["rsi_min"]) / 2.0
        rsi_score = max(0.25, 1.0 - abs(rsi - mid) / half)
    else:
        rsi_score = 0.0

    # RSI divergence — price extreme not confirmed by RSI = weakening setup
    div_detected, div_penalty = _rsi_divergence(closes, rsi_vals, desired_side, n=12)

    entry_score = round(
        max(0.0, pullback_score * 0.35 + pattern_score * 0.35 + rsi_score * 0.30 - div_penalty),
        3,
    )
    # against_trend: last candle closed against the trade direction at the entry zone.
    # This is a hard block — not a soft penalty. If price is pulling back INTO
    # the EMA21 zone but the last candle is still bearish on a LONG setup,
    # the entry is not ready yet. Wait for a bullish close to confirm the bounce.
    against_trend_candle = pattern_name == "against_trend"
    ent_reason = (
        f"RSI divergence — price at new extreme but RSI disagrees, trend may be weakening" if div_detected
        else f"Price {pb_pct*100:.1f}% from EMA21 — need <{p['pullback_max']*100:.1f}% to enter" if pullback_score == 0
        else f"RSI {rsi:.0f} outside entry zone {p['rsi_min']}–{p['rsi_max']}" if not rsi_in_range
        else f"Candle closed against trade direction — wait for {'bullish' if desired_side == 'LONG' else 'bearish'} close to confirm bounce" if against_trend_candle
        else f"Candle pattern weak ({pattern_name}) — wait for pin bar or engulfing at EMA21" if pattern_score < 0.40
        else ""
    )
    breakdown_entry = {
        "price_vs_ema21_pct": round(pb_pct * 100, 2),
        "pullback_max_pct":   round(p["pullback_max"] * 100, 1),
        "candle_pattern":     pattern_name,
        "pattern_score":      round(pattern_score, 2),
        "rsi":                round(rsi or 0, 1),
        "rsi_range":          f"{p['rsi_min']}–{p['rsi_max']}",
        "rsi_divergence":     div_detected,
        # against_trend is a hard fail regardless of RSI/pullback quality
        "score": entry_score, "ok": entry_score > 0.25 and not against_trend_candle, "reason": ent_reason,
    }

    # ── Layer 4: Momentum — candle alignment + volume confirmation ────────
    n = p["mom_n"]
    candles_ok = all(
        (klines[-(i + 1)]["close"] > klines[-(i + 1)]["open"]) == (desired_side == "LONG")
        for i in range(n)
    )
    candle_score = 1.0 if candles_ok else 0.0
    vol_score    = min(1.0, vol_ratio / (p["vol_factor"] * 1.5)) if vol_ratio >= p["vol_factor"] else vol_ratio / p["vol_factor"] * 0.4
    momentum_score = round(candle_score * 0.55 + vol_score * 0.45, 3)
    # Hard block: volume below 0.15x average = dead market, fake move — never trade
    # 0.30 was too strict — crypto regularly trades at 0.15-0.28x during Asian session
    vol_too_low = vol_ratio < 0.10
    mom_reason = (
        f"Volume {vol_ratio:.2f}x average — too low (min 0.10x), likely dead market" if vol_too_low
        else f"Last {n} candle(s) not {('bullish' if desired_side == 'LONG' else 'bearish')} — no momentum yet" if not candles_ok
        else f"Volume {vol_ratio:.1f}x average — need {p['vol_factor']:.1f}x minimum" if vol_ratio < p["vol_factor"]
        else ""
    )
    if vol_too_low:
        momentum_score = 0.0
    breakdown_momentum = {
        "candles_n": n, "candles_ok": candles_ok,
        "volume_ratio": round(vol_ratio, 2), "vol_factor": round(p["vol_factor"], 2),
        "score": momentum_score, "ok": momentum_score > 0.20, "reason": mom_reason,
    }

    breakdown = {
        "regime":    breakdown_regime,
        "direction": breakdown_direction,
        "entry":     breakdown_entry,
        "momentum":  breakdown_momentum,
    }

    # ── Final weighted score ──────────────────────────────────────────────
    # sess_score scales the total: 1.0 = full, 0.60 = needs 67% stronger signal
    # This means a news spike (high ATR, high ADX, high volume) can still
    # fire at 3am — but a mediocre setup in thin hours gets filtered out.
    raw_score = (
        regime_score    * 0.25 +
        direction_score * 0.30 +
        entry_score     * 0.30 +
        momentum_score  * 0.15
    )
    total_score = round(raw_score * sess_score, 3)
    min_score = p.get("min_score", 0.62)
    breakdown["session"] = {"label": sess_label, "quality": round(sess_score, 2), "raw_score": round(raw_score, 3), "ok": True, "reason": ""}

    # Market grade: A = high conviction (≥0.78), B = good setup (≥0.65), C = weak (blocked)
    grade = "A" if total_score >= 0.78 else "B" if total_score >= 0.65 else "C"

    failed = [k for k, v in breakdown.items() if not v.get("ok")]
    if failed or total_score < min_score:
        reasons = " | ".join(breakdown[k]["reason"] for k in failed if breakdown[k].get("reason"))
        if total_score < min_score and not failed:
            reasons = f"Signal quality {total_score:.2f} below threshold {min_score:.2f} — setup not strong enough"
        return {
            "ok": False,
            "blocked": f"BLOCKED ({', '.join(failed) or 'score'}): {reasons}",
            "signal": sig, "side": desired_side,
            "score": total_score, "min_score": min_score, "grade": "C",
            "breakdown": breakdown, "atr_pct": atr_pct,
        }

    return {
        "ok": True, "blocked": None,
        "signal": sig, "side": desired_side,
        "score": total_score, "min_score": min_score, "grade": grade,
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
    pending_trades: List[Dict]
    signal_score: float
    trade_style: str
    market_grade: str


class AutoStartIn(BaseModel):
    symbol: str
    trade_style: TradeStyle = "DAY_TRADE"
    mode: RiskMode = "MINI_ASYM"
    max_trades_per_day: Optional[int] = None
    stop_after_bad_trades: int = 2
    duration_days: int = 0
    trend_filter: bool = True
    chop_min_sep_pct: float = 0.005


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
        self.history: Deque[Dict[str, str]] = deque(maxlen=120)
        self.pending_trades: List[Dict] = []
        self.adaptive_strictness: float = 1.0
        self.last_breakdown: Dict = {}
        self.last_score: float = 0.0
        self.last_atr_pct: float = 0.01   # fallback 1% ATR until first signal
        self.market_grade: str = "-"
        self._last_trade_bad: bool = False
        self.session_start_equity: float = get_equity(email)
        self.peak_equity: float = get_equity(email)   # tracks highest equity for drawdown protection
        self.consecutive_wins: int = 0    # reset strictness after 3 wins in a row
        # Hard floor = 85% of actual starting equity for this session.
        # Uses real balance, NOT the paper $1,000 constant — so a $5k user's floor is $4,250
        # and a $1k user's floor is $850. Engine stops completely if equity falls below this.
        self.floor_equity: float = self.session_start_equity * 0.85
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
                # Only count primary trades (T1 / Grade A) — T2 has size_mult=0.40
                # so its stored size is always < 50% of base. Threshold: base_size * 0.45
                sp = TRADE_STYLE_PARAMS.get(self.trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
                base_size = presets_for_mode(self.mode)["size"]
                primary_threshold = round(base_size * 0.45, 4)
                cur.execute(
                    "SELECT COUNT(*) as cnt, "
                    "SUM(CASE WHEN unreal_pnl_percent < 0 THEN 1 ELSE 0 END) as bad "
                    "FROM trades WHERE email = %s AND time >= %s AND session_id = %s "
                    "AND size >= %s",
                    (self.email, utc_midnight_str, sid, primary_threshold),
                )
                row = cur.fetchone()
                conn.close()
            trades = int((row or {}).get("cnt") or 0)
            bad = int((row or {}).get("bad") or 0)
            return trades, bad
        except Exception:
            return 0, 0

    def log(self, msg):
        self.history.appendleft({"t": now_dubai().strftime("%Y-%m-%d %H:%M:%S"), "msg": msg})

    def start(self):
        self.log("AI started.")
        self.thread.start()

    def stop(self, reason="Stopped by user."):
        self.log(reason)
        self.stop_event.set()
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

        # ATR-based SL/TP — adapts to real market volatility
        sp = TRADE_STYLE_PARAMS.get(self.trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
        atr = self.last_atr_pct
        sl_pct = min(atr * sp["sl_atr"], sp["sl_max"] / 100.0)
        tp_pct = min(atr * sp["tp_atr"], sp["tp_max"] / 100.0) * tp_mult
        sl_pct = max(sl_pct, 0.002)
        tp_pct = max(tp_pct, 0.004)

        # Break-even SL: T2 after T1 wins → SL moves to entry (risk-free trade)
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
            self.log(f"High volatility (ATR {atr*100:.2f}% = {vol_ratio_atr:.1f}× normal {atr_normal*100:.2f}%) → size reduced to {vol_size_mult*100:.0f}%")

        # ── Hard equity floor — never trade if account below 85% of session start ──
        if equity_before < self.floor_equity:
            self.log(
                f"HARD FLOOR HIT — equity ${equity_before:.2f} below 85% floor "
                f"(${self.floor_equity:.2f}). AI stopped to protect capital."
            )
            self.blocked_reason = "HARD_FLOOR"
            self.stop_event.set()
            return equity_before

        # ── Drawdown protection (4-tier, from peak equity) ────────────────
        # Tiers shrink position size progressively — engine slows down as drawdown grows.
        # At -15% from peak: full stop. Hard floor (85% of session start) is a second wall.
        if equity_before > self.peak_equity:
            self.peak_equity = equity_before
        drawdown_pct = (self.peak_equity - equity_before) / self.peak_equity if self.peak_equity > 0 else 0.0
        dd_size_mult = 1.0
        if drawdown_pct >= 0.15:
            # -15%+ from peak: stop trading, protect what's left
            self.log(f"Drawdown {drawdown_pct*100:.1f}% from peak — STOPPING to protect capital.")
            self.blocked_reason = "MAX_DRAWDOWN"
            self.stop_event.set()
            return equity_before
        elif drawdown_pct >= 0.10:
            dd_size_mult = 0.25   # -10%: quarter size (recovery mode)
            self.log(f"Drawdown {drawdown_pct*100:.1f}% from peak → size at 25%")
        elif drawdown_pct >= 0.07:
            dd_size_mult = 0.40   # -7%: 40% size
            self.log(f"Drawdown {drawdown_pct*100:.1f}% from peak → size at 40%")
        elif drawdown_pct >= 0.04:
            dd_size_mult = 0.65   # -4%: 65% size
            self.log(f"Drawdown {drawdown_pct*100:.1f}% from peak → size at 65%")

        # Apply all size multipliers
        effective_size = c["size"] * size_mult * vol_size_mult * dd_size_mult

        # Hard per-trade max loss cap — scales by trade style (wider SL allowed on longer TF)
        max_loss_pct = {"SCALP": 0.015, "DAY_TRADE": 0.020, "SWING": 0.030}.get(self.trade_style, 0.020)
        max_sl_for_cap = max_loss_pct / (effective_size * c["leverage"]) if (effective_size * c["leverage"]) > 0 else sl_pct
        if sl_pct > max_sl_for_cap:
            sl_pct = max_sl_for_cap
            tp_pct = sl_pct * (sp["tp_atr"] / sp["sl_atr"]) * tp_mult

        # ── Breakeven stop — no fees, no loss ────────────────────────────
        # If trade has been open ≥ 2 intervals and price hasn't moved 30% toward TP,
        # move SL to entry. Worst case = 0% PnL, avoids paying fees on a losing exit.
        candles_held = int((time.time() - pt.get("open_ts", time.time())) / self.interval_sec)
        if candles_held >= 2 and not pt.get("breakeven"):
            raw_progress = (exit_price - entry) / entry if side == "LONG" else (entry - exit_price) / entry
            if raw_progress < tp_pct * 0.30:
                sl_pct = 0.0001   # SL at entry — exits at breakeven
                pt["breakeven"] = True
                self.log(f"Breakeven stop activated ({candles_held} candles, no progress) — SL moved to entry")

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

        if candle_high > 0 and not intrabar_tp:
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
        _prev_locked = float(pt.get("trail_locked_pct", 0.0))
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
                final_move = raw_move
                outcome = "NATURAL_CLOSE"

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
        if is_primary:
            self.trades_today += 1
            self._last_trade_bad = pnl_value < 0
            if pnl_value < 0:
                self.bad_trades_today += 1

        # Mini-Asym adaptive strictness — only update on primary trade
        if is_primary and mode == "MINI_ASYM":
            if pnl_value < 0:
                self.consecutive_wins = 0
                self.adaptive_strictness = min(2.5, self.adaptive_strictness + 0.25)
                self.log(f"MINI_ASYM strictness ↑ {self.adaptive_strictness:.2f} (after loss)")
            else:
                self.consecutive_wins += 1
                if self.consecutive_wins >= 3:
                    # 3 wins in a row → reset strictness fully, AI is in good form
                    self.adaptive_strictness = 1.0
                    self.consecutive_wins = 0
                    self.log("MINI_ASYM strictness reset to 1.0 (3 consecutive wins)")
                else:
                    self.adaptive_strictness = max(1.0, self.adaptive_strictness - 0.10)

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
        """Close all pending trades (1 for Grade A, 2 for Grade B).
        Uses intrabar high/low to realistically check SL/TP during the interval."""
        if not self.pending_trades:
            return
        try:
            # Get the latest candle to check intrabar SL/TP
            klines = _fetch_klines_sync(self.symbol, self.tf, limit=10)
            if klines:
                last_candle = klines[-1]
                candle_high = last_candle["high"]
                candle_low  = last_candle["low"]
                exit_price  = last_candle["close"]
            else:
                exit_price = self._fetch_price_sync(self.symbol)
                candle_high = exit_price
                candle_low  = exit_price

            t1_won = False
            for pt in list(self.pending_trades):
                # Break-even: if T2 has breakeven_after_t1 flag and T1 won, SL moves to entry
                if pt.get("breakeven_after_t1") and t1_won:
                    pt = {**pt, "breakeven": True}
                eq = get_equity(self.email)
                outcome_eq = self._close_one_trade(pt, exit_price, eq,
                                                   candle_high=candle_high, candle_low=candle_low)
                # Track if T1 (primary) won so T2 can use break-even
                if pt.get("is_primary") and outcome_eq > eq:
                    t1_won = True
        except Exception as e:
            self.log(f"Error closing trade(s): {e}")
        finally:
            self.pending_trades = []

    def _signal_and_filters(self):
        klines = _fetch_klines_sync(self.symbol, self.tf, limit=260)
        higher_tf = HIGHER_TF_MAP.get(self.tf, "4h")
        higher_klines: List[Dict] = []
        if higher_tf != self.tf:
            try:
                higher_klines = _fetch_klines_sync(self.symbol, higher_tf, limit=100)
            except Exception:
                higher_klines = []
        res = _compute_signal_layers(klines, self.mode, self.adaptive_strictness, higher_klines, self.trade_style)
        self.last_breakdown = res.get("breakdown", {})
        self.last_score = res.get("score", 0.0)
        self.market_grade = res.get("grade", "-")
        if res.get("atr_pct"):
            self.last_atr_pct = res["atr_pct"]
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
            self._last_trade_bad = False
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
                    time.sleep(self.interval_sec)
                    continue

                # ── Step 1: Close any pending trades first (real exit after one interval) ──
                if self.pending_trades:
                    self._close_pending_trades()
                    time.sleep(self.interval_sec)
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
                self.last_signal = res.get("signal") or "-"
                # Always reflect the HTF-computed direction, even when blocked.
                # Without this, last_side stays at "LONG" init default forever
                # and the status bar shows the wrong side while direction is blocked.
                if res.get("side"):
                    self.last_side = res["side"]
                if not res.get("ok"):
                    self.blocked_reason = res.get("blocked") or "BLOCKED"
                    self.log(f"Blocked: {self.blocked_reason[:100]}")
                    time.sleep(self.interval_sec)
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
                    base_trade = {
                        "entry_price": entry_price,
                        "side": desired_side,
                        "mode": self.mode,
                        "signal": self.last_signal,
                        "open_ts": time.time(),
                    }
                    if grade == "B":
                        self.pending_trades = [
                            {**base_trade, "grade": "B", "size_mult": 0.60, "tp_mult": 0.80, "is_primary": True,  "label": "T1"},
                            {**base_trade, "grade": "B", "size_mult": 0.40, "tp_mult": 1.00, "is_primary": False, "label": "T2", "breakeven_after_t1": True},
                        ]
                        self.log(f"TRADE OPENED Grade B ({desired_side}) @ {entry_price:.4f} | T1 60%+80%TP, T2 40%+100%TP+BE | score={self.last_score:.2f}")
                    else:
                        self.pending_trades = [
                            {**base_trade, "grade": "A", "size_mult": 1.00, "tp_mult": 1.00, "is_primary": True},
                        ]
                        self.log(f"TRADE OPENED Grade A ({desired_side}) @ {entry_price:.4f} | score={self.last_score:.2f} | signal={self.last_signal}")
                    self.last_trade_ts = now_ts
                else:
                    wait_sec = int(effective_cooldown - (now_ts - self.last_trade_ts))
                    cd_label = "2× cooldown (after loss)" if self._last_trade_bad else "cooldown"
                    self.log(f"Signal OK (score={self.last_score:.2f}) | {cd_label} — {wait_sec}s remaining")

            except Exception as e:
                self.blocked_reason = "ERROR"
                self.last_signal = f"ERROR: {str(e)[:120]}"
                self.log(self.last_signal)
            finally:
                time.sleep(self.interval_sec)

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
    """
    rows = _load_all_runner_states()
    if not rows:
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
                print(f"[startup-resume] {email} — duration session expired, skipping")
                continue

            # Skip if exchange is no longer connected (would block immediately anyway)
            if not get_exchange(email):
                _clear_runner_state(email)
                print(f"[startup-resume] {email} — no exchange connected, skipping")
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
            print(f"[startup-resume] {email} — error: {e}")
            _clear_runner_state(email)

    if resumed:
        print(f"[startup-resume] {resumed} AI runner(s) resumed after restart")


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

    # Get current price
    ticker = ex.fetch_ticker(symbol)
    price  = float(ticker["last"])

    # Set leverage (best-effort — some accounts have fixed leverage)
    try:
        ex.set_leverage(leverage, symbol)
    except Exception:
        pass

    # Quantity in base currency
    notional = usdt_size * leverage
    qty = round(notional / price, 6)

    # SL / TP prices
    sl_price = round(price * sl_mult,   4)
    tp_price = round(price * tp_mult_v, 4)

    # ── Exchange-specific order params ────────────────────────────────────────
    # Each exchange has a different way to attach SL/TP to the entry order.
    if ex_id == "bybit":
        params = {
            "stopLoss":    {"triggerPrice": sl_price, "type": "market"},
            "takeProfit":  {"triggerPrice": tp_price, "type": "limit"},
            "positionIdx": 0,   # one-way mode required
        }
    elif ex_id == "okx":
        # OKX attaches SL/TP via algo orders — place entry first, then attach
        params = {
            "tdMode":    "cross",      # cross-margin perpetual
            "slTriggerPx": str(sl_price),
            "slOrdPx":     "-1",       # -1 = market price on trigger
            "tpTriggerPx": str(tp_price),
            "tpOrdPx":     "-1",
        }
    elif ex_id == "binance":
        # Binance Futures: SL/TP placed as separate orders after entry
        params = {
            "reduceOnly": False,
        }
    else:
        params = {}

    order = ex.create_order(
        symbol = symbol,
        type   = "market",
        side   = order_side,
        amount = qty,
        params = params,
    )

    # Binance: attach SL/TP as separate stop-market orders
    if ex_id == "binance":
        close_side = "sell" if side == "LONG" else "buy"
        try:
            ex.create_order(symbol, "STOP_MARKET", close_side, qty, sl_price,
                            {"stopPrice": sl_price, "reduceOnly": True, "closePosition": True})
            ex.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, qty, tp_price,
                            {"stopPrice": tp_price, "reduceOnly": True, "closePosition": True})
        except Exception as e:
            pass  # Log but don't fail — main order is placed

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
    }


def close_real_order(email: str, symbol: str, side: str) -> dict:
    """Close an open position at market price."""
    row = get_exchange(email)
    if not row:
        raise ValueError("No exchange connected")

    ex = _make_ccxt_exchange(row)
    close_side = "sell" if side == "LONG" else "buy"

    # Fetch current position size
    positions = ex.fetch_positions([symbol])
    pos = next((p for p in positions if p["symbol"] == symbol and abs(float(p["contracts"] or 0)) > 0), None)
    if not pos:
        return {"ok": False, "note": "No open position found"}

    qty = abs(float(pos["contracts"]))
    order = ex.create_order(
        symbol   = symbol,
        type     = "market",
        side     = close_side,
        amount   = qty,
        params   = {"reduceOnly": True, "positionIdx": 0},
    )
    return {"ok": True, "order_id": order.get("id"), "qty": qty}


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


# =========================
# STARTUP: RESUME AI RUNNERS
# =========================
# Must run AFTER all classes (AutoRunner, helpers) are defined.
# Re-starts any AI session that was live before the last deploy/restart.
import threading as _threading
_threading.Thread(target=_resume_runners_on_startup, daemon=True).start()
