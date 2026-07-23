"""
auth_helpers.py — Shared FastAPI auth dependencies.

Single source of truth for require_user / require_admin and the session cache.
All route modules import from here — eliminates the previous 6-way duplication.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

from fastapi import Cookie, Depends, HTTPException

from database import db_conn

_SESSION_CACHE: Dict[str, tuple] = {}
_SESSION_TTL     = 300           # reuse cached lookup for 5 minutes
_SESSION_MAX_AGE = 7 * 24 * 3600 # hard expiry — matches cookie max_age

ADMIN_EMAILS: set = set(
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "admin@demo.com").split(",")
    if e.strip()
)


def require_user(session: Optional[str] = Cookie(default=None)) -> Dict:
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
        # Enforce 7-day hard expiry at DB level — expired rows are rejected
        # even if the cookie is still present (e.g. stolen/old device).
        cur.execute(
            "SELECT email, created_at FROM sessions WHERE token = %s",
            (session,),
        )
        row = cur.fetchone()
    if not row or (now - int(row["created_at"])) > _SESSION_MAX_AGE:
        _SESSION_CACHE.pop(session, None)
        if row:
            # Expired — clean it up so it can never be reused
            with db_conn() as conn2:
                conn2.cursor().execute("DELETE FROM sessions WHERE token = %s", (session,))
                conn2.commit()
        raise HTTPException(status_code=401, detail="Unauthorized")
    _SESSION_CACHE[session] = (row["email"], now)
    return {"email": row["email"]}


def require_admin(user=Depends(require_user)) -> str:
    email = user["email"].strip().lower()
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access only")
    return email


def evict_session(token: str) -> None:
    """Remove a token from the shared session cache (call on logout)."""
    _SESSION_CACHE.pop(token, None)
