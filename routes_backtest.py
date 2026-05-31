"""
routes_backtest.py — FastAPI router for /backtest/* endpoints.

Admin-only: starting and viewing backtests requires admin access.
Any authenticated user can view their own run history.

Wire up in main.py:
    from routes_backtest import backtest_router
    app.include_router(backtest_router)
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException
from pydantic import BaseModel, Field

from backtester import (
    compare_runs,
    get_run,
    list_runs,
    start_backtest,
)
from database import db_conn

backtest_router = APIRouter(prefix="/backtest", tags=["backtest"])

# ── Auth (standalone, no import from main.py to avoid circular imports) ────────
_ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "admin@demo.com").split(",")
    if e.strip()
)
_SESSION_CACHE: Dict[str, tuple] = {}
_SESSION_TTL   = 300  # 5 minutes


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


# ── Pydantic models ────────────────────────────────────────────────────────────
class BacktestRunIn(BaseModel):
    symbol:       str   = Field(..., max_length=20)
    mode:         str   = Field(..., pattern="^(ULTRA_SAFE|SAFE|NORMAL|MINI_ASYM|AGGRESSIVE)$")
    style:        str   = Field(..., pattern="^(SCALP|DAY_TRADE|SWING)$")
    exchange:     str   = Field("bybit", pattern="^(bybit|okx|binance)$")
    date_from:    str   = Field(..., description="YYYY-MM-DD")
    date_to:      str   = Field(..., description="YYYY-MM-DD")
    start_equity: float = Field(1000.0, ge=100, le=1_000_000)


class CompareIn(BaseModel):
    run_id_a: str
    run_id_b: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@backtest_router.post("/run")
def backtest_run(payload: BacktestRunIn, admin: str = Depends(_require_admin)):
    """
    Start a new backtest in the background. Returns run_id immediately.
    Admin only — resource-intensive operation.
    """
    try:
        run_id = start_backtest(
            email        = admin,
            symbol       = payload.symbol.upper().strip(),
            mode         = payload.mode,
            style        = payload.style,
            exchange     = payload.exchange,
            date_from    = payload.date_from,
            date_to      = payload.date_to,
            start_equity = payload.start_equity,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"run_id": run_id, "status": "pending", "message": "Backtest started"}


@backtest_router.get("/results/{run_id}")
def backtest_results(run_id: str, admin: str = Depends(_require_admin)):
    """
    Get status and results for a specific backtest run.
    Polls this endpoint until status == 'done' or 'error'.
    """
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@backtest_router.get("/history")
def backtest_history(
    limit: int = 20,
    admin: str = Depends(_require_admin),
):
    """List admin's recent backtest runs (most recent first)."""
    return list_runs(admin, min(limit, 50))


@backtest_router.post("/compare")
def backtest_compare(payload: CompareIn, admin: str = Depends(_require_admin)):
    """Compare two completed backtest runs side-by-side."""
    try:
        return compare_runs(payload.run_id_a, payload.run_id_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
