#!/usr/bin/env python3
"""
run_rr_analysis.py
──────────────────
Compute average win, average loss, and R:R by exit type for every live/demo
trade in the last 90 days. Compare to the backtest reference on the same
period.

Run on Render shell (DATABASE_URL auto-set):
  python3 run_rr_analysis.py

Optional flags:
  --days N        trailing days to include (default 90)
  --account demo  analyse only demo account  (default: both)
  --account real  analyse only real account
"""

from __future__ import annotations
import os, sys, argparse, statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASYM_DB_PATH", "asymmetric_demo.db")

from database import db_conn

REAL_EMAIL = "mes29571@gmail.com"
DEMO_EMAIL = "admin@demo.com"

# ── Backtest reference R:R (Option 3, OOS 2024–2026, all 10 coins combined)
# From predeploy_report.txt Task 4, floor ON. T1/T2 split with Grade B exits.
# T1 TP = 1.0×ATR, T2 trails to ~2-3×ATR. Avg based on the 10-coin pool.
BT_REF = {
    "wr_pct":           21.0,   # ~21% win rate across coins (Option 3 OOS avg)
    "avg_win_usd":      89.11,  # from live_vs_backtest.py (Oct 2026 backtest)
    "avg_loss_usd":    -30.00,  # estimated from BT R:R target
    "rr":               2.97,   # avg_win / avg_loss = 89.11 / 30.00
    "note":             "Option 3 OOS 2024-2026, 10-coin avg, floor ON",
}

SEP  = "=" * 70
SEP2 = "-" * 70

def _live_exit_type(reason: str, outcome: str) -> str:
    """Map trade outcome/reason to a readable exit type bucket."""
    r = (reason or "").upper()
    o = (outcome or "").upper()
    # Prefer outcome column (trade_outcomes table) over reason (trades table)
    if "T1_TP" in o and "T2_TRAIL" in o:   return "T1+T2_TRAIL"
    if "T1_TP" in o and "T2_BE" in o:       return "T1+BE_STOP"
    if "T1_TP" in o and "T2_SL" in o:       return "T1+SL"
    if "T1_TP_T2_TP" in o:                  return "T1+T2_TP"
    if "SL_HIT" in o or "STOP LOSS" in r:   return "SL"
    if "TRAIL" in o or "TRAILING" in r:     return "TRAIL"
    if "TP_HIT" in o or "TAKE PROFIT" in r: return "TP"
    if "BE" in o:                            return "BE_STOP"
    return "OTHER"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int, default=90)
    parser.add_argument("--account", choices=["demo", "real", "both"], default="both")
    args = parser.parse_args()

    emails = []
    if args.account in ("both", "demo"):  emails.append(DEMO_EMAIL)
    if args.account in ("both", "real"):  emails.append(REAL_EMAIL)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    placeholders = ",".join(["%s"] * len(emails))

    # ── Pull from trade_outcomes (rich analytics table) ──────────────────────
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT email, symbol, side, outcome, pnl_value, pnl_pct,
                       equity_before, grade, score, trailing_used, candles_open,
                       opened_at, closed_at
                FROM trade_outcomes
                WHERE email IN ({placeholders}) AND closed_at >= %s
                ORDER BY closed_at ASC
                """,
                emails + [cutoff],
            )
            rows_outcomes = cur.fetchall()
    except Exception as e:
        print(f"\nERROR querying trade_outcomes: {e}")
        print("Trying fallback to trades table …")
        rows_outcomes = []

    # ── Fallback: trades table (less detail but always present) ──────────────
    rows_trades = []
    if not rows_outcomes:
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT email, symbol, side, reason, pnl, entry_price,
                           exit_price, sl, tp, time, grade, score
                    FROM trades
                    WHERE email IN ({placeholders}) AND time >= %s
                      AND mode = 'MINI_ASYM'
                    ORDER BY time ASC
                    """,
                    emails + [cutoff],
                )
                rows_trades = cur.fetchall()
        except Exception as e:
            print(f"ERROR querying trades table: {e}")
            sys.exit(1)

    # ── Normalise rows into a common format ───────────────────────────────────
    trades = []
    if rows_outcomes:
        for r in rows_outcomes:
            trades.append({
                "email":    r["email"],
                "symbol":   r["symbol"],
                "side":     r["side"],
                "exit_key": r["outcome"] or "",
                "pnl_usd":  float(r["pnl_value"] or 0),
                "pnl_pct":  float(r["pnl_pct"]   or 0),
                "grade":    r["grade"] or "-",
                "score":    float(r["score"]      or 0),
                "trail":    bool(r["trailing_used"]),
                "source":   "trade_outcomes",
            })
        source_label = "trade_outcomes"
    else:
        for r in rows_trades:
            trades.append({
                "email":    r["email"],
                "symbol":   r["symbol"],
                "side":     r["side"],
                "exit_key": r["reason"] or "",
                "pnl_usd":  float(r["pnl"] or 0),
                "pnl_pct":  0.0,
                "grade":    r.get("grade") or "-",
                "score":    float(r.get("score") or 0),
                "trail":    False,
                "source":   "trades",
            })
        source_label = "trades (fallback)"

    if not trades:
        print(f"No trades found in last {args.days} days for {emails}.")
        sys.exit(0)

    print(f"\n{SEP}")
    print(f"  R:R ANALYSIS — Live vs Backtest")
    print(f"  Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Window    : last {args.days} days")
    print(f"  Accounts  : {', '.join(emails)}")
    print(f"  Source    : {source_label}  ({len(trades)} trades)")
    print(SEP)

    # ── Overall stats ─────────────────────────────────────────────────────────
    wins   = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] < 0]

    total   = len(trades)
    wr_pct  = len(wins) / total * 100 if total else 0
    avg_win  = statistics.mean([t["pnl_usd"] for t in wins])   if wins   else 0
    avg_loss = statistics.mean([t["pnl_usd"] for t in losses]) if losses else 0
    rr_live  = abs(avg_win / avg_loss) if avg_loss else 0

    print(f"\n  OVERALL  ({total} trades, {args.days}-day window)")
    print(SEP2)
    print(f"  {'Metric':<30} {'Live':>10}  {'Backtest ref':>14}")
    print(SEP2)
    print(f"  {'Win rate %':<30} {wr_pct:>9.1f}%  {BT_REF['wr_pct']:>13.1f}%")
    print(f"  {'Avg win ($)':<30} {avg_win:>+10.2f}  {BT_REF['avg_win_usd']:>+13.2f}")
    print(f"  {'Avg loss ($)':<30} {avg_loss:>+10.2f}  {BT_REF['avg_loss_usd']:>+13.2f}")
    print(f"  {'R:R (avg_win / |avg_loss|)':<30} {rr_live:>10.2f}  {BT_REF['rr']:>13.2f}")
    print(SEP2)
    print(f"  Backtest ref: {BT_REF['note']}")

    # ── Gap explanation ───────────────────────────────────────────────────────
    print(f"\n  R:R GAP ANALYSIS")
    print(SEP2)
    if avg_win > 0 and avg_loss < 0:
        gap = BT_REF["rr"] - rr_live
        print(f"  Live R:R {rr_live:.2f} vs backtest {BT_REF['rr']:.2f} — gap = {gap:.2f}")
        print()
        # T1/T2 split diagnosis
        t1_exits = [t for t in trades if "T1" in t["exit_key"].upper()]
        sl_exits  = [t for t in trades if "SL" in t["exit_key"].upper() or "STOP LOSS" in t["exit_key"].upper()]
        trail_ex  = [t for t in trades if "TRAIL" in t["exit_key"].upper()]
        t1_pct    = len(t1_exits) / total * 100 if total else 0
        sl_pct    = len(sl_exits)  / total * 100 if total else 0
        trail_pct = len(trail_ex)  / total * 100 if total else 0
        print(f"  Exit mix: T1={t1_pct:.0f}%  SL={sl_pct:.0f}%  Trail={trail_pct:.0f}%  Other={100-t1_pct-sl_pct-trail_pct:.0f}%")
        print()
        print(f"  Likely causes of the gap:")
        if rr_live < 2.0:
            print(f"  → T2 trades not running to full trail (trail exits early or T2 SL at BE)")
            print(f"    Backtest: T2 trails to 2-3×ATR on winners. Live may hit BE stop sooner.")
        if sl_pct > 70:
            print(f"  → {sl_pct:.0f}% of trades exit at SL — high SL rate compresses avg win")
        if t1_pct > 60:
            print(f"  → {t1_pct:.0f}% of exits at T1 TP (1×ATR). T2 (40% size) closing at breakeven")
            print(f"     means effective R:R on the full trade is dragged down.")
        print(f"  → TP multiplier: backtest uses 2.0×ATR for T1; live confirms same.")
        print(f"     If T2 rarely reaches trail: avg win ≈ 1×ATR (T1 only), avg loss ≈ 1×ATR → R:R ≈ 1")
    else:
        print(f"  Insufficient data — need at least 1 win and 1 loss to compute R:R.")

    # ── Exit type breakdown ───────────────────────────────────────────────────
    print(f"\n  BY EXIT TYPE")
    print(SEP2)
    print(f"  {'Exit type':<18}  {'Count':>6}  {'WR':>7}  {'Avg win':>9}  {'Avg loss':>10}  {'R:R':>6}")
    print(SEP2)

    by_exit: dict = defaultdict(list)
    for t in trades:
        bucket = _live_exit_type(t["exit_key"], t["exit_key"])
        by_exit[bucket].append(t)

    for bucket in sorted(by_exit.keys()):
        group  = by_exit[bucket]
        g_wins  = [t for t in group if t["pnl_usd"] > 0]
        g_loss  = [t for t in group if t["pnl_usd"] < 0]
        g_wr    = len(g_wins) / len(group) * 100 if group else 0
        g_aw    = statistics.mean([t["pnl_usd"] for t in g_wins])  if g_wins  else 0
        g_al    = statistics.mean([t["pnl_usd"] for t in g_loss])  if g_loss  else 0
        g_rr    = abs(g_aw / g_al) if g_al else 0
        rr_s    = f"{g_rr:.2f}" if g_al and g_aw else "—"
        print(
            f"  {bucket:<18}  {len(group):>6}  {g_wr:>6.1f}%  "
            f"{g_aw:>+9.2f}  {g_al:>+10.2f}  {rr_s:>6}"
        )

    # ── Per-account breakdown ─────────────────────────────────────────────────
    print(f"\n  PER ACCOUNT")
    print(SEP2)
    for email in emails:
        acct = [t for t in trades if t["email"] == email]
        if not acct:
            continue
        a_wins = [t for t in acct if t["pnl_usd"] > 0]
        a_loss = [t for t in acct if t["pnl_usd"] < 0]
        a_wr   = len(a_wins) / len(acct) * 100 if acct else 0
        a_aw   = statistics.mean([t["pnl_usd"] for t in a_wins]) if a_wins else 0
        a_al   = statistics.mean([t["pnl_usd"] for t in a_loss]) if a_loss else 0
        a_rr   = abs(a_aw / a_al) if a_al else 0
        total_pnl = sum(t["pnl_usd"] for t in acct)
        label  = "DEMO" if email == DEMO_EMAIL else "REAL"
        print(f"  {label} ({email}): {len(acct)} trades  WR={a_wr:.1f}%  "
              f"avg_win={a_aw:+.2f}  avg_loss={a_al:+.2f}  R:R={a_rr:.2f}  "
              f"total_pnl={total_pnl:+.2f}")

    # ── Per-symbol breakdown ──────────────────────────────────────────────────
    print(f"\n  PER SYMBOL")
    print(SEP2)
    print(f"  {'Symbol':<12}  {'N':>5}  {'WR':>7}  {'Avg win':>9}  {'Avg loss':>10}  {'R:R':>6}  {'Total PnL':>10}")
    print(SEP2)
    by_sym: dict = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    for sym in sorted(by_sym.keys()):
        sg     = by_sym[sym]
        sw     = [t for t in sg if t["pnl_usd"] > 0]
        sl_    = [t for t in sg if t["pnl_usd"] < 0]
        s_wr   = len(sw) / len(sg) * 100 if sg else 0
        s_aw   = statistics.mean([t["pnl_usd"] for t in sw])  if sw  else 0
        s_al   = statistics.mean([t["pnl_usd"] for t in sl_]) if sl_ else 0
        s_rr   = abs(s_aw / s_al) if s_al else 0
        s_pnl  = sum(t["pnl_usd"] for t in sg)
        rr_s   = f"{s_rr:.2f}" if s_al and s_aw else "—"
        print(
            f"  {sym:<12}  {len(sg):>5}  {s_wr:>6.1f}%  "
            f"{s_aw:>+9.2f}  {s_al:>+10.2f}  {rr_s:>6}  {s_pnl:>+10.2f}"
        )

    print(f"\n{SEP}")
    print(f"  Sample sizes: {len(wins)} wins  {len(losses)} losses  {total - len(wins) - len(losses)} breakeven")
    if total < 50:
        print(f"  WARNING: {total} trades is below 50 — R:R averages are noisy.")
        print(f"           Treat these numbers as directional, not definitive.")
    print(SEP)


if __name__ == "__main__":
    main()
