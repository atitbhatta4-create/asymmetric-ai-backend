"""
backup_r2.py — Daily PostgreSQL backup to Cloudflare R2.

Exports all critical tables as JSON, compresses, uploads to R2.
Keeps last 30 backups (auto-deletes older ones).
Never raises — failure is logged and Telegram-alerted, never crashes the server.

Called daily from adaptive_pipeline._monthly_scheduler_loop().
Can also be run manually from Render shell:
    python3 backup_r2.py
"""
from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timezone

import boto3
import httpx
from botocore.config import Config

from database import db_conn

# ── Config ────────────────────────────────────────────────────────────────────
R2_ACCESS_KEY  = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY  = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT    = os.getenv("R2_ENDPOINT", "")
R2_BUCKET      = os.getenv("R2_BUCKET", "asymmetric-ai-backups")
KEEP_BACKUPS   = 30  # days of backups to keep

CRITICAL_TABLES = [
    "users",
    "user_state",
    "ai_sessions",
    "ai_logs",
    "trade_outcomes",
    "pipeline_runs",
    "pipeline_coin_results",
    "pipeline_alerts",
]


def _get_r2_client():
    if not all([R2_ACCESS_KEY, R2_SECRET_KEY, R2_ENDPOINT]):
        raise RuntimeError("R2 credentials not configured (R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT)")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _dump_table(cur, table: str) -> list:
    try:
        cur.execute(f"SELECT * FROM {table}")
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[backup] Warning: could not dump table {table}: {exc}")
        return []


def _serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, '__float__'):
        return float(obj)
    return str(obj)


def run_backup() -> str:
    """
    Run a full backup of all critical tables.
    Returns the R2 object key on success.
    Raises on failure.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    key = f"backups/{ts}_asymmetric_ai.json.gz"

    print(f"[backup] Starting backup -> {key}")
    t0 = time.time()

    # Dump all tables
    payload = {"backup_time": ts, "tables": {}}
    with db_conn() as conn:
        cur = conn.cursor()
        for table in CRITICAL_TABLES:
            rows = _dump_table(cur, table)
            payload["tables"][table] = rows
            print(f"[backup]   {table}: {len(rows)} rows")

    # Compress
    raw_json  = json.dumps(payload, default=_serialize, ensure_ascii=False)
    compressed = gzip.compress(raw_json.encode("utf-8"), compresslevel=6)
    size_kb = len(compressed) / 1024
    print(f"[backup] Compressed size: {size_kb:.1f} KB")

    # Upload to R2 — use presigned URL + httpx to bypass boto3/urllib3 SSL issues
    client = _get_r2_client()
    presigned = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": R2_BUCKET, "Key": key, "ContentType": "application/gzip"},
        ExpiresIn=600,
    )
    resp = httpx.put(presigned, content=compressed,
                     headers={"Content-Type": "application/gzip"}, timeout=120)
    resp.raise_for_status()
    elapsed = time.time() - t0
    print(f"[backup] Uploaded {key} in {elapsed:.1f}s")

    # Clean up old backups (keep last KEEP_BACKUPS)
    _cleanup_old_backups(client)

    return key


def _cleanup_old_backups(client) -> None:
    try:
        resp = client.list_objects_v2(Bucket=R2_BUCKET, Prefix="backups/")
        objects = sorted(
            resp.get("Contents", []),
            key=lambda o: o["LastModified"],
            reverse=True,
        )
        to_delete = objects[KEEP_BACKUPS:]
        for obj in to_delete:
            client.delete_object(Bucket=R2_BUCKET, Key=obj["Key"])
            print(f"[backup] Deleted old backup: {obj['Key']}")
    except Exception as exc:
        print(f"[backup] Cleanup warning: {exc}")


def run_backup_safe() -> bool:
    """
    Safe wrapper — never raises. Returns True on success, False on failure.
    Sends Telegram alert on failure.
    """
    try:
        key = run_backup()
        print(f"[backup] SUCCESS: {key}")
        return True
    except Exception as exc:
        msg = f"[BACKUP FAILED] Daily R2 backup failed: {exc}"
        print(msg)
        try:
            from notifications import tg_alert
            tg_alert(msg)
        except Exception:
            pass
        return False


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    ok = run_backup_safe()
    sys.exit(0 if ok else 1)
