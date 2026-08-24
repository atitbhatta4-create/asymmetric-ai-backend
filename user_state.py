"""
user_state.py — Per-user equity, session, floor, exchange keys, and admin settings.

All DB-backed state helpers live here. No FastAPI routes — pure functions only.
main.py, routes_auth.py, and routes_admin.py import from here.
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Dict, Optional

from cryptography.fernet import Fernet

from config import START_EQUITY, now_utc_str
from database import db_conn

# ── Fernet encryption for API keys stored in DB ───────────────────────────────
# Set ENCRYPTION_KEY env var to a Fernet key.
# Generate once: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
_RAW_ENC_KEY = os.getenv("ENCRYPTION_KEY", "")
_FERNET: Fernet | None = None
if _RAW_ENC_KEY:
    try:
        _FERNET = Fernet(_RAW_ENC_KEY.encode() if isinstance(_RAW_ENC_KEY, str) else _RAW_ENC_KEY)
    except Exception:
        pass

if not _FERNET:
    if os.getenv("ENV", "dev") == "prod":
        raise RuntimeError(
            "ENCRYPTION_KEY env var is not set in production. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
            "and add it to your Render environment variables."
        )
    _FALLBACK = base64.urlsafe_b64encode(hashlib.sha256(b"asym-dev-key").digest())
    _FERNET = Fernet(_FALLBACK)


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


# ── User equity & session ─────────────────────────────────────────────────────

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


# ── Hard floor (R1 hybrid ratchet) ────────────────────────────────────────────

def _compute_floor(peak: float, start_capital: float) -> float:
    """R1 Hybrid floor: flat 15% from peak until equity doubles start capital,
    then ratchet tightens to 10% from peak (locks in compounded gains)."""
    if start_capital > 0 and peak >= 2.0 * start_capital:
        return round(max(start_capital * 0.85, peak * 0.90), 2)
    return round(peak * 0.85, 2)


def update_peak_ath(email: str, peak: float, equity_after: float) -> None:
    """Update peak_equity, all_time_high, and floor_equity after each trade.
    All three columns only ever increase — never decrease — so the floor survives
    runner restarts, redeploys, and ai_runner_state deletions."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT peak_equity, all_time_high, floor_equity, starting_capital FROM user_state WHERE email = %s",
            (email,),
        )
        row = cur.fetchone()
        if row:
            new_peak  = max(float(row["peak_equity"] or 0), peak)
            new_ath   = max(float(row["all_time_high"] or 0), equity_after)
            _sc       = float(row["starting_capital"] or 0)
            new_floor = max(float(row["floor_equity"] or 0), _compute_floor(new_peak, _sc))
            cur.execute(
                "UPDATE user_state SET peak_equity=%s, all_time_high=%s, floor_equity=%s WHERE email=%s",
                (new_peak, new_ath, new_floor, email),
            )
            conn.commit()


# ── Onboarding completion ─────────────────────────────────────────────────────

def get_onboarding_complete(email: str) -> bool:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT onboarding_complete FROM user_state WHERE email = %s", (email,))
        row = cur.fetchone()
        return bool(row["onboarding_complete"]) if row else False


def mark_onboarding_complete(email: str) -> None:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET onboarding_complete = TRUE WHERE email = %s", (email,))
        conn.commit()


# ── First-trade milestone email ───────────────────────────────────────────────

_FIRST_TRADE_SENT: set = set()


def _check_first_trade_email(email: str, symbol: str, side: str, grade: str, equity: float) -> None:
    """Send first-trade milestone email once per user lifetime."""
    if email in _FIRST_TRADE_SENT:
        return
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM trade_outcomes WHERE email=%s", (email,))
            row = cur.fetchone()
            total = int(row["n"] if row else 0)
        if total <= 1:
            from notifications import email_first_trade
            email_first_trade(email, symbol, side, grade, equity)
        _FIRST_TRADE_SENT.add(email)
    except Exception:
        pass


# ── Exchange key store ────────────────────────────────────────────────────────

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


# ── Admin settings (key/value store) ─────────────────────────────────────────

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
