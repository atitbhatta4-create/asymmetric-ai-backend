"""
routes_report.py — Admin endpoints to trigger & monitor the comprehensive
strategy report on the Render server (no Mac needed).

Endpoints:
  POST /admin/run-comprehensive-report  — start background run
  GET  /admin/report-status             — progress / ETA
  GET  /admin/report-download           — pre-signed R2 URL to download PDF
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from routes_admin import _require_admin

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
                     headers={"Content-Type": "application/pdf"}, timeout=300)
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
