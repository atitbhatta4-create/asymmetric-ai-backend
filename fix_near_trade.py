"""
One-time fix script for the NEAR trade recorded as SL_HIT when it was actually profitable.
Run this ONCE on the real backend server console (Render/Railway shell).

What it does:
  - Finds the 2 most recent NEAR trades in user_state DB
  - Corrects unreal_pnl_value and unreal_pnl_percent to reflect real Bybit P&L (+$0.5612)
  - Appends a correction note to the reason field
  - Does NOT touch equity_after (already correct at $21.05)
  - Does NOT touch live runner state (midnight reset already fixed strictness + bad_trades)

Usage:
  python fix_near_trade.py
"""

import os
import sys

REAL_PNL   = 0.5612   # total real P&L from Bybit for both legs combined
EMAIL      = "mes29571@gmail.com"
SYMBOL     = "NEAR/USDT:USDT"
NUM_TRADES = 2

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    sys.exit("ERROR: DATABASE_URL env var not set. Run this on the backend server.")

import psycopg
from psycopg.rows import dict_row

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"Connecting to DB...")
with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
    cur = conn.cursor()

    # Find session_id for this user
    cur.execute("SELECT session_id FROM user_state WHERE email = %s", (EMAIL,))
    row = cur.fetchone()
    if not row:
        sys.exit(f"ERROR: No user_state found for {EMAIL}")
    sid = row["session_id"]
    print(f"Session ID: {sid}")

    # Find the 2 most recent NEAR trades
    cur.execute(
        "SELECT id, size, unreal_pnl_value, unreal_pnl_percent, equity_after, reason "
        "FROM trades WHERE email=%s AND symbol=%s AND session_id=%s "
        "ORDER BY id DESC LIMIT %s",
        (EMAIL, SYMBOL, sid, NUM_TRADES),
    )
    trades = cur.fetchall()

    if not trades:
        # Try without session filter — might be an older session
        cur.execute(
            "SELECT id, size, unreal_pnl_value, unreal_pnl_percent, equity_after, reason "
            "FROM trades WHERE email=%s AND symbol=%s "
            "ORDER BY id DESC LIMIT %s",
            (EMAIL, SYMBOL, NUM_TRADES),
        )
        trades = cur.fetchall()

    if not trades:
        sys.exit(f"ERROR: No NEAR trades found for {EMAIL}")

    print(f"\nFound {len(trades)} trade(s) to correct:")
    for t in trades:
        print(f"  ID={t['id']} size={t['size']:.4f} pnl=${t['unreal_pnl_value']:.4f} equity_after=${t['equity_after']:.2f}")

    # Split real P&L proportionally by size
    total_size = sum(float(t["size"]) for t in trades)
    corrections = []
    for t in trades:
        frac = float(t["size"]) / total_size if total_size > 0 else 1.0 / len(trades)
        leg_pnl = round(REAL_PNL * frac, 4)
        new_pct = round(leg_pnl / float(t["size"]) * 100, 4) if float(t["size"]) > 0 else 0.0
        corrections.append({
            "id": t["id"],
            "old_pnl": float(t["unreal_pnl_value"]),
            "leg_pnl": leg_pnl,
            "new_pct": new_pct,
            "reason": t["reason"] or "",
        })

    print(f"\nApplying corrections:")
    for c in corrections:
        note = (
            f"\n\n[MANUAL CORRECTION 2026-05-25] "
            f"Paper sim: {c['old_pnl']:+.4f} → Real Bybit: {c['leg_pnl']:+.4f} | "
            f"Outcome corrected to TP_HIT"
        )
        new_reason = c["reason"] + note
        cur.execute(
            "UPDATE trades SET unreal_pnl_value=%s, unreal_pnl_percent=%s, reason=%s WHERE id=%s",
            (c["leg_pnl"], c["new_pct"], new_reason, c["id"]),
        )
        print(f"  Trade ID={c['id']}: pnl ${c['old_pnl']:+.4f} → ${c['leg_pnl']:+.4f} ✓")

    conn.commit()
    print(f"\nDone. {len(corrections)} trade(s) corrected.")
    print("Dashboard stats will now show the correct NEAR P&L.")
