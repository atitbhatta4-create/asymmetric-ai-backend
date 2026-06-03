"""
routes_optimizer.py — FastAPI router for /optimizer/* endpoints.
Admin-only. Wire up in main.py:
    from routes_optimizer import optimizer_router
    app.include_router(optimizer_router)
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from database import db_conn
from optimizer import (
    apply_opt_params,
    get_opt_run,
    get_opt_results_csv,
    list_applied_params,
    list_opt_runs,
    start_optimizer,
)

optimizer_router = APIRouter(prefix="/optimizer", tags=["optimizer"])

# ── Standalone auth (no circular import from main.py) ─────────────────────────
_ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "admin@demo.com").split(",")
    if e.strip()
)
_SESSION_CACHE: Dict[str, tuple] = {}
_SESSION_TTL = 300


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
class OptRunIn(BaseModel):
    symbol:       str   = Field(..., max_length=20)
    mode:         str   = Field(..., pattern="^(ULTRA_SAFE|SAFE|NORMAL|MINI_ASYM|AGGRESSIVE)$")
    style:        str   = Field(..., pattern="^(SCALP|DAY_TRADE|SWING)$")
    exchange:     str   = Field("bybit", pattern="^(bybit|okx|binance)$")
    date_from:    str   = Field(..., description="YYYY-MM-DD")
    date_to:      str   = Field(..., description="YYYY-MM-DD")
    start_equity: float = Field(1000.0, ge=100, le=1_000_000)


class ApplyIn(BaseModel):
    result_id: int


# ── Endpoints ──────────────────────────────────────────────────────────────────
@optimizer_router.post("/run")
def optimizer_run(payload: OptRunIn, admin: str = Depends(_require_admin)):
    """Start a parameter optimization run. Returns run_id immediately."""
    try:
        run_id = start_optimizer(
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
    total = 5 * 3 * 3 * 3  # OPT_GRID sizes
    return {"run_id": run_id, "status": "pending", "total_combos": total}


@optimizer_router.get("/status/{run_id}")
def optimizer_status(run_id: str, admin: str = Depends(_require_admin)):
    """Poll this every 3s for live progress."""
    run = get_opt_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id":      run["run_id"],
        "status":      run["status"],
        "progress":    run["progress"],
        "done_combos": run.get("done_combos", 0),
        "total_combos":run.get("total_combos", 135),
        "error":       run.get("error"),
        "results":     run.get("results", []) if run["status"] == "done" else [],
    }


@optimizer_router.get("/results/{run_id}")
def optimizer_results(run_id: str, admin: str = Depends(_require_admin)):
    """Full results including ranked parameter table."""
    run = get_opt_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@optimizer_router.post("/apply")
def optimizer_apply(payload: ApplyIn, admin: str = Depends(_require_admin)):
    """Apply a specific result's params as the active override for that symbol/mode/style."""
    try:
        return apply_opt_params(payload.result_id, admin)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@optimizer_router.get("/history")
def optimizer_history(
    limit: int = 10,
    admin: str = Depends(_require_admin),
):
    return list_opt_runs(admin, limit=limit)


@optimizer_router.get("/applied")
def optimizer_applied(admin: str = Depends(_require_admin)):
    """List all currently applied parameter overrides."""
    return list_applied_params()


@optimizer_router.get("/download/{run_id}")
def optimizer_download(run_id: str, admin: str = Depends(_require_admin)):
    """Download all optimizer results for a run as a CSV file."""
    csv_content = get_opt_results_csv(run_id)
    if csv_content is None:
        raise HTTPException(status_code=404, detail="Run not found or has no results")
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="optimizer_{run_id[:8]}.csv"'},
    )
