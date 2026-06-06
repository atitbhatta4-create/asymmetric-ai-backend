"""
routes_auth.py — Auth, session, password, and 2FA endpoints.

Wire up in main.py:
    from routes_auth import auth_router
    app.include_router(auth_router)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import smtplib
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional

import bcrypt
import pyotp
from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from config import IS_PROD, SMTP_USER, SMTP_PASS, SMTP_HOST, SMTP_PORT
from database import db_conn
from notifications import (
    _email_base,
    email_2fa_enabled,
    email_otp_reset,
    email_password_changed,
    send_email,
)

auth_router = APIRouter(tags=["auth"])

# ── Local helpers ──────────────────────────────────────────────────────────────
def _now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ── Auth session cache ─────────────────────────────────────────────────────────
_SESSION_CACHE: Dict[str, tuple] = {}
_SESSION_TTL = 300  # 5 minutes

# ── Admin config ───────────────────────────────────────────────────────────────
_ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "admin@demo.com").split(",")
    if e.strip()
)
_ADMIN_FORCE_RESET_TOKEN = os.getenv("ADMIN_FORCE_RESET_TOKEN", "")

# ── Per-endpoint rate limiter (separate from global middleware) ─────────────────
_rl_hits: Dict[str, list] = {}
_rl_lock = threading.Lock()


def _rate_limit(request: Request, limit: int, window: int = 60) -> None:
    xff = request.headers.get("x-forwarded-for", "")
    real_ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "unknown")
    key = f"{request.url.path}:{real_ip}"
    now = time.time()
    with _rl_lock:
        hits = [t for t in _rl_hits.get(key, []) if now - t < window]
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests — slow down.")
        hits.append(now)
        _rl_hits[key] = hits


# ── Password helpers ───────────────────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_pw(plain: str, stored: str) -> bool:
    if not plain or not stored:
        return False
    if stored.startswith("$2b$") or stored.startswith("$2a$"):
        return bcrypt.checkpw(plain.encode(), stored.encode())
    return hashlib.sha256(plain.encode()).hexdigest() == stored


# ── TOTP helper ────────────────────────────────────────────────────────────────
def _get_totp_row(email: str) -> Optional[dict]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT secret, enabled FROM totp_secrets WHERE email=%s", (email,))
        row = cur.fetchone()
    return dict(row) if row else None


# ── Cookie helpers ─────────────────────────────────────────────────────────────
def set_session_cookie(response: Response, token: str) -> None:
    base = dict(key="session", value=token, httponly=True, max_age=60 * 60 * 24 * 7, path="/")
    if IS_PROD:
        response.set_cookie(**base, samesite="none", secure=True)
    else:
        response.set_cookie(**base, samesite="lax", secure=False)


def clear_session_cookie(response: Response) -> None:
    if IS_PROD:
        response.delete_cookie("session", path="/", samesite="none", secure=True)
    else:
        response.delete_cookie("session", path="/", samesite="lax", secure=False)


# ── Auth dependencies ──────────────────────────────────────────────────────────
def _require_user(session: Optional[str] = Cookie(default=None)) -> Dict:
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


def _require_admin(user=Depends(_require_user)) -> str:
    email = user["email"].strip().lower()
    if email not in _ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access only")
    return email


# ── Models ─────────────────────────────────────────────────────────────────────
class AuthIn(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., max_length=256)


class SignupIn(AuthIn):
    password: str = Field(..., min_length=8, max_length=256)


class SessionOut(BaseModel):
    ok: bool
    email: Optional[str] = None


class ForgotIn(BaseModel):
    email: str


class ResetIn(BaseModel):
    token: str
    new_password: str


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str


class ForgotPasswordIn(BaseModel):
    email: str


class VerifyOtpIn(BaseModel):
    email: str
    code: str
    new_password: str


class TotpCodeIn(BaseModel):
    code: str


class AdminForceResetIn(BaseModel):
    token: str
    email: str
    new_password: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@auth_router.post("/auth/signup")
def signup(request: Request, payload: SignupIn):
    _rate_limit(request, limit=3, window=3600)
    email = payload.email.strip().lower()
    if not email or not payload.password:
        raise HTTPException(status_code=400, detail="Email and password required")

    from main import signup_is_enabled, seat_capacity, seats_used
    if not signup_is_enabled():
        raise HTTPException(status_code=403, detail="Signup is currently disabled by admin.")
    if seats_used() >= seat_capacity():
        raise HTTPException(status_code=403, detail="Seats are full. Signup is closed.")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="User already exists")
        cur.execute(
            "INSERT INTO users(email, password_hash, created_at) VALUES(%s, %s, %s)",
            (email, hash_pw(payload.password), _now_utc_str()),
        )
        conn.commit()
    return {"ok": True}


@auth_router.post("/auth/login", response_model=SessionOut)
def login(request: Request, payload: AuthIn, response: Response):
    _rate_limit(request, limit=15, window=900)
    email = payload.email.strip().lower()

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        row = cur.fetchone()

    if not row or not verify_pw(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Silently migrate legacy SHA-256 hash to bcrypt on successful login
    stored = row["password_hash"]
    if not (stored.startswith("$2b$") or stored.startswith("$2a$")):
        new_hash = hash_pw(payload.password)
        with db_conn() as conn2:
            cur2 = conn2.cursor()
            cur2.execute("UPDATE users SET password_hash=%s WHERE email=%s", (new_hash, email))
            conn2.commit()

    token = secrets.token_urlsafe(32)
    now = int(time.time())

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO sessions(token, email, created_at) VALUES(%s, %s, %s)", (token, email, now))
        conn.commit()

    set_session_cookie(response, token)

    from main import ensure_user_state
    ensure_user_state(email)

    totp_row = _get_totp_row(email)
    requires_2fa = bool(totp_row and totp_row["enabled"])
    return {"ok": True, "email": email, "requires_2fa": requires_2fa}


@auth_router.post("/auth/logout")
def logout(response: Response, session: Optional[str] = Cookie(default=None)):
    if session:
        _SESSION_CACHE.pop(session, None)
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM sessions WHERE token = %s", (session,))
            conn.commit()
    clear_session_cookie(response)
    return {"ok": True}


@auth_router.get("/session", response_model=SessionOut)
def session_me(session: Optional[str] = Cookie(default=None)):
    if not session:
        return {"ok": False, "email": None}
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM sessions WHERE token = %s", (session,))
        row = cur.fetchone()
    return {"ok": bool(row), "email": row["email"] if row else None}


@auth_router.post("/auth/change-password")
def change_password(payload: ChangePasswordIn, user=Depends(_require_user)):
    email = user["email"]
    if not payload.current_password or not payload.new_password:
        raise HTTPException(status_code=400, detail="Both current and new password are required.")
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        row = cur.fetchone()

    if not row or not verify_pw(payload.current_password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET password_hash = %s WHERE email = %s", (hash_pw(payload.new_password), email))
        conn.commit()

    email_password_changed(email)
    return {"ok": True}


@auth_router.post("/auth/forgot-password")
def forgot_password(request: Request, payload: ForgotPasswordIn):
    _rate_limit(request, limit=3, window=3600)
    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        exists = cur.fetchone() is not None

    if exists:
        code = str(secrets.randbelow(900000) + 100000)
        now_ts = int(time.time())
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO otp_codes(email, code, created_at, used)
                VALUES(%s, %s, %s, 0)
                ON CONFLICT(email) DO UPDATE SET code=EXCLUDED.code,
                    created_at=EXCLUDED.created_at, used=0
            """, (email, code, now_ts))
            conn.commit()
        email_otp_reset(email, code)

    return {"ok": True, "detail": "If that email exists, a reset code has been sent."}


@auth_router.post("/auth/reset-password")
def reset_password(request: Request, payload: VerifyOtpIn):
    _rate_limit(request, limit=3, window=3600)
    email = payload.email.strip().lower()
    code  = payload.code.strip()

    if not email or not code or not payload.new_password:
        raise HTTPException(status_code=400, detail="Email, code and new password required")
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    OTP_TTL = 15 * 60

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT code, created_at, used FROM otp_codes WHERE email = %s", (email,))
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=400, detail="No reset code found. Request a new one.")
    if row["used"]:
        raise HTTPException(status_code=400, detail="Code already used. Request a new one.")
    if int(time.time()) - row["created_at"] > OTP_TTL:
        raise HTTPException(status_code=400, detail="Code expired. Request a new one.")
    if not hmac.compare_digest(str(row["code"]), code):
        raise HTTPException(status_code=400, detail="Incorrect code.")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE otp_codes SET used=1 WHERE email=%s", (email,))
        cur.execute("UPDATE users SET password_hash=%s WHERE email=%s",
                    (hash_pw(payload.new_password), email))
        conn.commit()

    email_password_changed(email)
    return {"ok": True, "detail": "Password reset successfully. You can now log in."}


# Legacy token-based reset (kept for compatibility)
@auth_router.post("/auth/forgot")
def auth_forgot(payload: ForgotIn):
    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email=%s", (email,))
        u = cur.fetchone()
        if not u:
            return {"ok": True}

        token = secrets.token_urlsafe(32)
        cur.execute(
            "INSERT INTO password_resets(token,email,created_at,used) VALUES(%s,%s,%s,0)",
            (token, email, int(time.time())),
        )
        conn.commit()

    return {"ok": True, "reset_token": token}


@auth_router.post("/auth/reset")
def auth_reset(payload: ResetIn):
    token = payload.token.strip()
    new_pw = payload.new_password or ""
    if not token or len(new_pw) < 4:
        raise HTTPException(status_code=400, detail="token + new_password required (min 4 chars)")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email, used, created_at FROM password_resets WHERE token=%s", (token,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Invalid token")
        if int(row["used"]) == 1:
            raise HTTPException(status_code=400, detail="Token already used")
        if int(time.time()) - int(row["created_at"]) > 1800:
            raise HTTPException(status_code=400, detail="Token expired")

        email = row["email"]
        cur.execute("UPDATE users SET password_hash=%s WHERE email=%s", (hash_pw(new_pw), email))
        cur.execute("UPDATE password_resets SET used=1 WHERE token=%s", (token,))
        cur.execute("DELETE FROM sessions WHERE email=%s", (email,))
        conn.commit()

    return {"ok": True}


# ── 2FA (TOTP) ─────────────────────────────────────────────────────────────────

@auth_router.get("/auth/2fa/status")
def totp_status(user=Depends(_require_user)):
    row = _get_totp_row(user["email"])
    return {"enabled": bool(row and row["enabled"])}


@auth_router.post("/auth/2fa/setup")
def totp_setup(user=Depends(_require_user)):
    from main import encrypt_key
    email = user["email"]
    secret = pyotp.random_base32()
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name="Asymmetric AI")
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO totp_secrets(email, secret, enabled, created_at)
            VALUES(%s, %s, 0, %s)
            ON CONFLICT(email) DO UPDATE SET secret=EXCLUDED.secret,
                enabled=0, created_at=EXCLUDED.created_at
        """, (email, encrypt_key(secret), _now_utc_str()))
        conn.commit()
    return {"ok": True, "uri": uri, "secret": secret}


@auth_router.post("/auth/2fa/confirm")
def totp_confirm(payload: TotpCodeIn, user=Depends(_require_user)):
    from main import decrypt_key
    email = user["email"]
    row = _get_totp_row(email)
    if not row:
        raise HTTPException(status_code=400, detail="Run /auth/2fa/setup first")

    secret = decrypt_key(row["secret"])
    totp   = pyotp.TOTP(secret)
    if not totp.verify(payload.code.strip(), valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid code — check your authenticator app")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE totp_secrets SET enabled=1 WHERE email=%s", (email,))
        conn.commit()

    email_2fa_enabled(email)
    return {"ok": True, "detail": "2FA is now active on your account"}


@auth_router.post("/auth/2fa/verify")
def totp_verify(payload: TotpCodeIn, user=Depends(_require_user)):
    from main import decrypt_key
    email = user["email"]
    row = _get_totp_row(email)
    if not row or not row["enabled"]:
        return {"ok": True, "detail": "2FA not enabled"}

    secret = decrypt_key(row["secret"])
    totp   = pyotp.TOTP(secret)
    if not totp.verify(payload.code.strip(), valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid 2FA code")

    return {"ok": True}


@auth_router.post("/auth/2fa/disable")
def totp_disable(payload: TotpCodeIn, user=Depends(_require_user)):
    from main import decrypt_key
    email = user["email"]
    row = _get_totp_row(email)
    if not row or not row["enabled"]:
        raise HTTPException(status_code=400, detail="2FA is not enabled")

    secret = decrypt_key(row["secret"])
    totp   = pyotp.TOTP(secret)
    if not totp.verify(payload.code.strip(), valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid code — cannot disable 2FA")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE totp_secrets SET enabled=0 WHERE email=%s", (email,))
        conn.commit()

    return {"ok": True, "detail": "2FA disabled"}


# ── Debug / admin-access endpoints ─────────────────────────────────────────────

@auth_router.get("/debug/email")
def debug_email(to: str = Query(default="")):
    if not SMTP_USER or not SMTP_PASS:
        return {"ok": False, "error": "SMTP_USER or SMTP_PASS not set", "smtp_user": repr(SMTP_USER)}
    target = to.strip() if to.strip() else SMTP_USER
    content = "<h2 style='color:#f1f5f9;'>Test Email</h2><p style='opacity:0.85;'>SMTP is working on Asymmetric AI.</p>"
    err = None
    try:
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


@auth_router.post("/admin/force-reset-password")
def admin_force_reset(payload: AdminForceResetIn):
    if not _ADMIN_FORCE_RESET_TOKEN or payload.token.strip() != _ADMIN_FORCE_RESET_TOKEN:
        raise HTTPException(status_code=403, detail="Not allowed")

    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")
    if email not in _ADMIN_EMAILS:
        raise HTTPException(status_code=400, detail="email is not an admin")
    if not payload.new_password or len(payload.new_password) < 4:
        raise HTTPException(status_code=400, detail="new_password min 4 chars")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email=%s", (email,))
        u = cur.fetchone()
        if not u:
            cur.execute(
                "INSERT INTO users(email, password_hash, created_at) VALUES(%s,%s,%s)",
                (email, hash_pw(payload.new_password), _now_utc_str()),
            )
        else:
            cur.execute("UPDATE users SET password_hash=%s WHERE email=%s", (hash_pw(payload.new_password), email))
        cur.execute("DELETE FROM sessions WHERE email=%s", (email,))
        conn.commit()

    return {"ok": True}
