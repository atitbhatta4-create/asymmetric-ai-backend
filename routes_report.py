"""
routes_report.py — Admin endpoints to trigger & monitor the comprehensive
strategy report on the Render server (no Mac needed).

Endpoints:
  POST /admin/run-comprehensive-report  — start background run
  GET  /admin/report-status             — progress / ETA
  GET  /admin/report-download           — pre-signed R2 URL to download PDF

  POST /admin/run-batch-optimizer       — run MINI_ASYM optimizer for all 10 coins (DAY_TRADE + SWING)
  GET  /admin/batch-optimizer-status    — progress + best params per coin when done
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from auth_helpers import require_admin as _require_admin
from database import db_conn

report_router = APIRouter(tags=["report"])

# ── In-memory run state ────────────────────────────────────────────────────────
_state: dict = {
    "running":     False,
    "started_at":  None,
    "progress":    "idle",
    "done_count":  0,
    "total_count": 100,   # 5 modes × 2 styles × 10 coins
    "done":        False,
    "error":       None,
    "r2_key":      None,
    "elapsed_min": 0.0,
}
_lock = threading.Lock()


def _upd(key, val):
    with _lock:
        _state[key] = val


# ── Background worker ──────────────────────────────────────────────────────────
def _run_bg():
    t0 = time.time()
    _upd("running",    True)
    _upd("started_at", datetime.now(timezone.utc).isoformat())
    _upd("done",       False)
    _upd("error",      None)
    _upd("r2_key",     None)
    _upd("done_count", 0)
    _upd("progress",   "Fetching candle data…")

    try:
        import run_comprehensive_report as rpt

        def _cb(msg: str, done: int):
            _upd("progress",    msg)
            _upd("done_count",  done)
            _upd("elapsed_min", round((time.time() - t0) / 60, 1))

        pdf_bytes = rpt.run_full(progress_cb=_cb)

        key = _upload_r2(pdf_bytes)
        _upd("r2_key",  key)
        _upd("done",    True)
        _upd("progress", f"Complete ✓ — {key}")

    except Exception as exc:
        _upd("error",    str(exc))
        _upd("progress", f"FAILED: {exc}")
    finally:
        _upd("running",     False)
        _upd("elapsed_min", round((time.time() - t0) / 60, 1))


def _upload_r2(pdf_bytes: bytes) -> str:
    import boto3
    import httpx
    from botocore.config import Config

    key_id  = os.getenv("R2_ACCESS_KEY_ID", "")
    secret  = os.getenv("R2_SECRET_ACCESS_KEY", "")
    endpoint= os.getenv("R2_ENDPOINT", "")
    bucket  = os.getenv("R2_BUCKET", "asymmetric-ai-backups")

    if not all([key_id, secret, endpoint]):
        raise RuntimeError("R2 credentials missing — set R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    key = f"reports/comprehensive_{ts}.pdf"

    # Use presigned URL + httpx to bypass boto3/urllib3 SSL issues on Render
    presigned = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": "application/pdf"},
        ExpiresIn=600,
    )
    resp = httpx.put(presigned, content=pdf_bytes,
                     headers={"Content-Type": "application/pdf"}, timeout=300,
                     verify=False)
    resp.raise_for_status()
    print(f"[report] PDF uploaded → r2://{bucket}/{key}  ({len(pdf_bytes)//1024} KB)")
    return key


# ── Endpoints ──────────────────────────────────────────────────────────────────
@report_router.post("/admin/run-comprehensive-report")
def trigger_report(admin: str = Depends(_require_admin)):
    with _lock:
        if _state["running"]:
            raise HTTPException(409, "Report already running — check /admin/report-status")

    threading.Thread(target=_run_bg, daemon=False).start()
    return {"status": "started", "message": "Running in background. Poll /admin/report-status for progress."}


@report_router.get("/admin/report-status")
def report_status(admin: str = Depends(_require_admin)):
    with _lock:
        s = dict(_state)

    pct = round(s["done_count"] / s["total_count"] * 100, 1) if s["total_count"] else 0
    eta = None
    if s["running"] and s["done_count"] > 0 and s["elapsed_min"] > 0:
        rate = s["done_count"] / s["elapsed_min"]
        remaining = s["total_count"] - s["done_count"]
        eta = round(remaining / rate, 0) if rate > 0 else None

    return {**s, "progress_pct": pct, "eta_minutes": eta}


@report_router.get("/admin/report-download")
def report_download(admin: str = Depends(_require_admin)):
    with _lock:
        key  = _state.get("r2_key")
        done = _state.get("done")

    if not done or not key:
        raise HTTPException(404, "Report not ready — check /admin/report-status first")

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT", ""),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.getenv("R2_BUCKET", "asymmetric-ai-backups"), "Key": key},
        ExpiresIn=3600,
    )
    return {"url": url, "key": key, "expires_in_seconds": 3600}


# ── Batch Optimizer ────────────────────────────────────────────────────────────
# Runs MINI_ASYM + DAY_TRADE + SWING for all 10 coins sequentially on Render.
# Results stored in optimizer_runs / optimizer_results tables (same as single runs).
# Status endpoint returns best params per coin when done — use to update coin_params.py.

_BATCH_COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]
_BATCH_STYLES      = ["DAY_TRADE", "SWING"]
_BATCH_MODE        = "MINI_ASYM"
_BATCH_DATE_FROM   = "2020-01-01"
_BATCH_DATE_TO     = "2026-05-31"
_BATCH_EQUITY      = 1000.0

_bstate: dict = {
    "running":      False,
    "current_coin": None,
    "current_style": None,
    "done_count":   0,
    "total_count":  len(_BATCH_COINS) * len(_BATCH_STYLES),
    "done":         False,
    "error":        None,
    "run_ids":      [],   # list of {symbol, style, run_id}
    "summary":      None, # filled when done: list of best-result per coin/style
    "elapsed_min":  0.0,
    "started_at":   None,
}
_bstate_lock = threading.Lock()


def _bupd(key, val):
    with _bstate_lock:
        _bstate[key] = val


def _wait_for_opt_run(run_id: str, timeout_sec: int = 1200) -> bool:
    """Poll DB until optimizer run completes. Returns True=done, False=error/timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT status FROM optimizer_runs WHERE run_id=%s", (run_id,)
                )
                row = cur.fetchone()
            if row and row["status"] in ("done", "error"):
                return row["status"] == "done"
        except Exception:
            pass
        time.sleep(15)
    return False


def _collect_batch_summary(run_ids: list) -> list:
    """For each completed run, pull the #1 Sharpe result and format it."""
    summary = []
    for item in run_ids:
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT params_json, total_trades, win_rate, total_return, "
                    "sharpe_ratio, max_drawdown, reward_risk, passed_filters "
                    "FROM optimizer_results WHERE run_id=%s AND passed_filters=1 "
                    "ORDER BY sharpe_ratio DESC LIMIT 1",
                    (item["run_id"],),
                )
                row = cur.fetchone()
        except Exception as exc:
            summary.append({"symbol": item["symbol"], "style": item["style"],
                            "run_id": item["run_id"], "error": str(exc)})
            continue

        if not row:
            summary.append({"symbol": item["symbol"], "style": item["style"],
                            "run_id": item["run_id"],
                            "error": "No results passed filters (too few trades)"})
            continue

        params = json.loads(row["params_json"])
        # adx_delta is relative to MINI_ASYM base (adx_min=16)
        effective_adx = 16 + params.get("adx_delta", 0)
        # score_delta is relative to MINI_ASYM base (min_score=0.63)
        effective_score = round(0.63 + params.get("score_delta", 0.0), 3)

        summary.append({
            "symbol":        item["symbol"],
            "style":         item["style"],
            "run_id":        item["run_id"],
            "sharpe":        round(row["sharpe_ratio"], 3),
            "total_return":  f"{row['total_return']*100:.1f}%",
            "win_rate":      f"{row['win_rate']*100:.1f}%",
            "max_drawdown":  f"{row['max_drawdown']*100:.1f}%",
            "total_trades":  row["total_trades"],
            "best_params":   params,
            "coin_params_update": {
                "tp_multiplier":    params.get("tp_mult", 2.0),
                "sl_multiplier":    params.get("sl_mult", 1.0),
                "effective_adx_min":      effective_adx,
                "effective_score_threshold": effective_score,
            },
        })
    return summary


def _run_batch_bg(admin_email: str):
    t0 = time.time()
    _bupd("running",     True)
    _bupd("started_at",  datetime.now(timezone.utc).isoformat())
    _bupd("done",        False)
    _bupd("error",       None)
    _bupd("done_count",  0)
    _bupd("run_ids",     [])
    _bupd("summary",     None)

    from optimizer import start_optimizer

    run_ids: list = []
    done = 0

    try:
        for style in _BATCH_STYLES:
            for sym in _BATCH_COINS:
                _bupd("current_coin",  sym)
                _bupd("current_style", style)
                _bupd("elapsed_min",   round((time.time() - t0) / 60, 1))

                print(f"[batch_opt] Starting {sym} MINI_ASYM {style}…")
                run_id = start_optimizer(
                    email=admin_email,
                    symbol=sym,
                    mode=_BATCH_MODE,
                    style=style,
                    exchange="bybit",
                    date_from=_BATCH_DATE_FROM,
                    date_to=_BATCH_DATE_TO,
                    start_equity=_BATCH_EQUITY,
                )
                run_ids.append({"symbol": sym, "style": style, "run_id": run_id})
                _bupd("run_ids", list(run_ids))

                ok = _wait_for_opt_run(run_id)
                done += 1
                _bupd("done_count",  done)
                _bupd("elapsed_min", round((time.time() - t0) / 60, 1))
                print(f"[batch_opt] {sym} {style} {'done' if ok else 'FAILED'} "
                      f"({done}/{_bstate['total_count']})")

        # All runs complete — collect best results
        summary = _collect_batch_summary(run_ids)
        _bupd("summary",      summary)
        _bupd("done",         True)
        _bupd("current_coin", None)
        print(f"[batch_opt] All done in {round((time.time()-t0)/60,1)} min")

    except Exception as exc:
        _bupd("error", str(exc))
        print(f"[batch_opt] ERROR: {exc}")
    finally:
        _bupd("running",     False)
        _bupd("elapsed_min", round((time.time() - t0) / 60, 1))


@report_router.post("/admin/run-batch-optimizer")
def trigger_batch_optimizer(admin: str = Depends(_require_admin)):
    with _bstate_lock:
        if _bstate["running"]:
            raise HTTPException(
                409,
                "Batch optimizer already running — check /admin/batch-optimizer-status"
            )
    threading.Thread(target=_run_batch_bg, args=(admin,), daemon=False).start()
    return {
        "status": "started",
        "message": (
            f"Running MINI_ASYM optimizer for {len(_BATCH_COINS)} coins × "
            f"{len(_BATCH_STYLES)} styles = {len(_BATCH_COINS)*len(_BATCH_STYLES)} runs. "
            "Poll /admin/batch-optimizer-status for progress. Estimated time: 1-2 hours."
        ),
        "coins": _BATCH_COINS,
        "styles": _BATCH_STYLES,
        "date_range": f"{_BATCH_DATE_FROM} → {_BATCH_DATE_TO}",
    }


@report_router.get("/admin/batch-optimizer-status")
def batch_optimizer_status(admin: str = Depends(_require_admin)):
    with _bstate_lock:
        s = dict(_bstate)

    total = s["total_count"]
    done  = s["done_count"]
    pct   = round(done / total * 100, 1) if total else 0
    eta   = None
    if s["running"] and done > 0 and s["elapsed_min"] > 0:
        rate = done / s["elapsed_min"]
        eta  = round((total - done) / rate, 1) if rate > 0 else None

    return {
        **s,
        "progress_pct":  pct,
        "eta_minutes":   eta,
        "next_step": (
            "Review summary → update coin_params.py with best tp_multiplier/sl_multiplier "
            "→ then trigger /admin/run-comprehensive-report (S4 is now enabled in that report)"
        ) if s["done"] else None,
    }
