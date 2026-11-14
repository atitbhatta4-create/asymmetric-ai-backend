#!/usr/bin/env python3
"""
run_mr4h_live_check.py
──────────────────────
Live signal performance tracker for 4h MR trades.

After ENABLE_MR_4H_SIGNAL=True is deployed, run this after 7+ days
to see how 4h MR trades performed vs all other signal types.

Compares:
  - MR_4H trades (new 4h mean reversion)
  - MEAN_REVERSION trades (legacy 1h MR, should drop to 0 when flag ON)
  - All other signals (trending / reversal)

Usage:
  python3 run_mr4h_live_check.py           # last 7 days
  python3 run_mr4h_live_check.py --days 14  # last 14 days
  python3 run_mr4h_live_check.py --days 30  # last 30 days

Run on Render shell (DATABASE_URL auto-set).
"""

import os, sys, argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from database import db_conn

SEP  = "=" * 68
SEP2 = "-" * 68

BACKTEST_REF = {
    # From mr4h_full_results.json — OOS 2024-2026 baseline
    "oos_wins":        6,
    "roll_pct":        52.0,
    "seed_gate_wins":  7,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7, help="Look-back window in days")
    return p.parse_args()


def fmt(v, decimals=2):
    if v is None: return "   N/A"
    return f"{v:+.{decimals}f}" if v < 0 else f" {v:.{decimals}f}"


def main():
    args = parse_args()
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    since_str = since.strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{SEP}")
    print(f"  4h MR LIVE SIGNAL CHECK")
    print(f"  Window : last {args.days} days  (since {since_str})")
    print(f"  Run at : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(SEP)

    with db_conn() as conn:
        cur = conn.cursor()

        # ── All trades in window ──────────────────────────────────────────────
        cur.execute("""
            SELECT signal, outcome, pnl, entry_price, exit_price, symbol, side, grade
            FROM trade_outcomes
            WHERE created_at >= %s
            ORDER BY created_at DESC
        """, (since,))
        rows = cur.fetchall()

    if not rows:
        print(f"\n  No trades found in the last {args.days} days.")
        print(SEP)
        return

    # ── Classify by signal type ───────────────────────────────────────────────
    groups = {
        "MR_4H":          [],
        "MEAN_REVERSION": [],
        "REVERSAL":       [],
        "OTHER":          [],
    }

    for row in rows:
        sig = (row[0] or "-").upper()
        if "MR_4H" in sig:
            groups["MR_4H"].append(row)
        elif "MEAN_REVERSION" in sig:
            groups["MEAN_REVERSION"].append(row)
        elif "REVERSAL" in sig:
            groups["REVERSAL"].append(row)
        else:
            groups["OTHER"].append(row)

    def stats(trades):
        if not trades:
            return None
        pnls   = [float(r[2]) for r in trades if r[2] is not None]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr     = len(wins) / len(pnls) * 100 if pnls else 0
        avg_w  = sum(wins)   / len(wins)   if wins   else 0
        avg_l  = sum(losses) / len(losses) if losses else 0
        rr     = abs(avg_w / avg_l) if avg_l else None
        total  = sum(pnls)
        return {
            "count": len(trades), "wr": wr,
            "avg_win": avg_w, "avg_loss": avg_l,
            "rr": rr, "total_pnl": total,
            "wins": len(wins), "losses": len(losses),
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'Signal':<18} {'Trades':>7} {'W%':>6} {'Avg Win':>9} {'Avg Loss':>10} {'R:R':>6} {'Total PnL':>10}")
    print(f"  {SEP2}")

    for label, trades in groups.items():
        s = stats(trades)
        if s is None:
            print(f"  {label:<18} {'—':>7}")
            continue
        rr_str = f"{s['rr']:.2f}" if s['rr'] else "  N/A"
        print(f"  {label:<18} {s['count']:>7} {s['wr']:>5.1f}% "
              f"{s['avg_win']:>+9.2f} {s['avg_loss']:>+10.2f} "
              f"{rr_str:>6} {s['total_pnl']:>+10.2f}")

    total_all = sum(float(r[2]) for r in rows if r[2] is not None)
    print(f"  {SEP2}")
    print(f"  {'TOTAL':<18} {len(rows):>7}  {'':>6} {'':>9} {'':>10} {'':>6} {total_all:>+10.2f}")

    # ── MR_4H deep dive ───────────────────────────────────────────────────────
    mr4h = groups["MR_4H"]
    if mr4h:
        print(f"\n{SEP}")
        print(f"  MR_4H TRADE DETAIL  ({len(mr4h)} trades)")
        print(f"  {'Symbol':<12} {'Side':>5} {'Grade':>6} {'PnL':>9} {'Outcome':<24}")
        print(f"  {SEP2}")
        for row in mr4h:
            sig, outcome, pnl, entry, exit_, sym, side, grade = row
            pnl_str = f"{float(pnl):+.2f}" if pnl is not None else "   N/A"
            print(f"  {sym:<12} {side:>5} {grade:>6} {pnl_str:>9} {outcome or '—':<24}")

        s = stats(mr4h)
        print(f"\n  Summary: {s['wins']}W / {s['losses']}L  |  "
              f"WR {s['wr']:.0f}%  |  "
              f"R:R {s['rr']:.2f if s['rr'] else 'N/A'}  |  "
              f"Total {s['total_pnl']:+.2f}")

    # ── Backtest comparison ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  BACKTEST REFERENCE (OOS 2024-2026)")
    print(f"  OOS wins: {BACKTEST_REF['oos_wins']}/10 coins beat baseline")
    print(f"  Rolling : {BACKTEST_REF['roll_pct']}% windows")
    print(f"  Seed    : {BACKTEST_REF['seed_gate_wins']}/10 beat random median")

    mr4h_s = stats(mr4h)
    if mr4h_s and mr4h_s["count"] >= 3:
        print(f"\n  LIVE MR_4H so far ({args.days} days):")
        print(f"  Trades: {mr4h_s['count']}  WR: {mr4h_s['wr']:.0f}%  "
              f"R:R: {mr4h_s['rr']:.2f if mr4h_s['rr'] else 'N/A'}  "
              f"PnL: {mr4h_s['total_pnl']:+.2f}")
        print()
        if mr4h_s["count"] < 10:
            print(f"  ⚠  Only {mr4h_s['count']} trades — too few for statistical confidence.")
            print(f"     Check again after {10 - mr4h_s['count']} more MR_4H trades.")
        else:
            print(f"  {'✓' if mr4h_s['wr'] >= 40 else '✗'} WR ≥40%: {mr4h_s['wr']:.0f}%")
            print(f"  {'✓' if (mr4h_s['rr'] or 0) >= 1.5 else '✗'} R:R ≥1.5: "
                  f"{mr4h_s['rr']:.2f if mr4h_s['rr'] else 'N/A'}")
    else:
        print(f"\n  MR_4H live trades: {len(mr4h)} — check again after 7+ days with flag ON.")

    # ── 1h MR check ──────────────────────────────────────────────────────────
    if groups["MEAN_REVERSION"]:
        print(f"\n  ⚠  {len(groups['MEAN_REVERSION'])} MEAN_REVERSION (1h) trades found.")
        print(f"     If ENABLE_MR_4H_SIGNAL=True, this should be 0.")
        print(f"     Check that enable_mr_signal=(not ENABLE_MR_4H_SIGNAL) is wired correctly.")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
