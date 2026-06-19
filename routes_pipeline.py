"""
routes_pipeline.py — FastAPI router for /pipeline/* endpoints.

Admin-only: trigger walk-forward validation, view results, get HTML report.
Wire up in main.py:
    from routes_pipeline import pipeline_router
    app.include_router(pipeline_router)
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from adaptive_pipeline import (
    PIPELINE_COINS,
    DEFAULT_MODE,
    DEFAULT_STYLE,
    generate_html_report,
    generate_pdf_report,
    get_pipeline_results,
    get_pipeline_status,
    list_pipeline_runs,
    start_pipeline,
    check_live_degradation,
)
from database import db_conn

pipeline_router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# ── Auth (same pattern as routes_backtest.py) ──────────────────────────────────
_ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "admin@demo.com").split(",")
    if e.strip()
)
_SESSION_CACHE: Dict[str, tuple] = {}
_SESSION_TTL   = 300


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
class PipelineRunIn(BaseModel):
    coins:         Optional[List[str]] = Field(None, description="Leave null for all 10 default coins")
    mode:          str                 = Field(DEFAULT_MODE,  pattern="^(ULTRA_SAFE|SAFE|NORMAL|MINI_ASYM|AGGRESSIVE)$")
    style:         str                 = Field(DEFAULT_STYLE, pattern="^(SCALP|DAY_TRADE|SWING)$")
    exchange:      str                 = Field("bybit", pattern="^(bybit|okx|binance)$")
    start_equity:  float               = Field(1000.0, ge=100, le=1_000_000)
    train_years:   int                 = Field(2, ge=1, le=5)
    test_years:    int                 = Field(2, ge=1, le=5)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@pipeline_router.post("/run")
def pipeline_run(payload: PipelineRunIn, admin: str = Depends(_require_admin)):
    """
    Start a walk-forward validation pipeline run in the background.
    Optimises on train period, validates on test period, reports degradation.
    Returns run_id immediately — poll /pipeline/status/{run_id} for progress.
    Admin only.
    """
    coins = [c.upper().strip() for c in payload.coins] if payload.coins else PIPELINE_COINS
    run_id = start_pipeline(
        triggered_by  = f"manual:{admin}",
        coins         = coins,
        mode          = payload.mode,
        style         = payload.style,
        exchange      = payload.exchange,
        start_equity  = payload.start_equity,
        train_years   = payload.train_years,
        test_years    = payload.test_years,
    )
    return {
        "run_id":  run_id,
        "status":  "running",
        "coins":   len(coins),
        "message": "Pipeline started — poll /pipeline/status/{run_id} for progress",
    }


@pipeline_router.get("/status/{run_id}")
def pipeline_status(run_id: str, admin: str = Depends(_require_admin)):
    """Poll for pipeline run progress. Status: running → complete | error."""
    return get_pipeline_status(run_id)


@pipeline_router.get("/results/{run_id}")
def pipeline_results(run_id: str, admin: str = Depends(_require_admin)):
    """Full per-coin results table for a completed pipeline run."""
    rows = get_pipeline_results(run_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Run not found or no results yet")
    return rows


@pipeline_router.get("/report/{run_id}", response_class=HTMLResponse)
def pipeline_report(run_id: str, admin: str = Depends(_require_admin)):
    """HTML report — open in browser and Ctrl+P → Save as PDF."""
    html = generate_html_report(run_id)
    return HTMLResponse(content=html)


@pipeline_router.get("/report/{run_id}/pdf")
def pipeline_report_pdf(run_id: str, admin: str = Depends(_require_admin)):
    """
    Download the pipeline report as a real PDF file.
    Opens a Save dialog in the browser automatically.
    """
    pdf_bytes = generate_pdf_report(run_id)
    if not pdf_bytes:
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable — fpdf2 not installed on this server. Use /report/{run_id} (HTML) instead.",
        )
    filename = f"pipeline_report_{run_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@pipeline_router.get("/history")
def pipeline_history(limit: int = 20, admin: str = Depends(_require_admin)):
    """List recent pipeline runs (most recent first)."""
    return list_pipeline_runs(min(limit, 50))


@pipeline_router.post("/check-degradation")
def pipeline_check_degradation(admin: str = Depends(_require_admin)):
    """
    Manually trigger a live-vs-backtest degradation check.
    Sends Telegram alert for any metric degraded >25%.
    Normally runs automatically every 24h via the monthly scheduler.
    """
    alerts = check_live_degradation()
    return {
        "alerts_fired": len(alerts),
        "details": alerts,
        "message": "No degradation detected" if not alerts else f"{len(alerts)} alert(s) sent to Telegram",
    }
