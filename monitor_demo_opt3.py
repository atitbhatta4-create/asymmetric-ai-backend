#!/usr/bin/env python3
"""
monitor_demo_opt3.py
────────────────────
Reads trade_outcomes for the MAIN (demo) service and reports:
  - Every Grade B trade: label (T1/T2), outcome, trailing used, candles held
  - T1/T2 pairing: flags any T1 without a matching T2 (or vice versa)
  - Min-order-size fallback: T1 without T2 pair when grade=B signals both legs
  - Weekly P&L, win rate, trade count
  - Side-by-side backtest comparison for the same period (runs a backtest job)

Usage (run on server with DATABASE_URL set):
  python3 monitor_demo_opt3.py [--days 7]
"""

import sys, os, argparse, time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from database import db_conn

DEMO_EMAIL = "admin@demo.com"
REAL_EMAIL = "mes29571@gmail.com"

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_outcomes(email: str, days: int):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM trade_outcomes WHERE email=%s AND closed_at >= %s ORDER BY closed_at",
            (email, cutoff),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows] if rows else []


def pair_t1_t2(rows):
    """Group rows into pairs by opened_at + symbol. Returns list of pair dicts."""
    by_key = defaultdict(list)
    for r in rows:
        if r.get("grade") == "B":
            key = (r["symbol"], r.get("opened_at", "")[:16])  # minute precision
            by_key[key].append(r)

    pairs = []
    lone  = []
    for key, legs in by_key.items():
        t1 = next((x for x in legs if x.get("label") == "T1"), None)
        t2 = next((x for x in legs if x.get("label") == "T2"), None)
        if t1 and t2:
            pairs.append({"t1": t1, "t2": t2})
        else:
            for leg in legs:
                lone.append(leg)
    return pairs, lone


def fmt_pct(v):
    if v is None: return "—"
    return f"{v*100:+.2f}%" if abs(v) < 10 else f"{v:+.2f}%"


# ── Report ────────────────────────────────────────────────────────────────────

def report(email: str, label: str, days: int):
    rows = fetch_outcomes(email, days)
    if not rows:
        print(f"\n[{label}] No trades in the last {days} days.")
        return

    print(f"\n{'='*72}")
    print(f"  {label}  —  last {days} days  ({len(rows)} closed legs)")
    print(f"{'='*72}")

    # ── All trades summary ────────────────────────────────────────────────────
    wins  = sum(1 for r in rows if r.get("pnl_pct", 0) > 0)
    total = len(rows)
    total_pnl = sum(r.get("pnl_value", 0) or 0 for r in rows)
    wr = wins / total * 100 if total else 0
    print(f"\n  Legs: {total}   Win rate: {wr:.0f}%   Net PnL: ${total_pnl:+.2f}")

    # ── Grade breakdown ───────────────────────────────────────────────────────
    grade_a = [r for r in rows if r.get("grade") == "A"]
    grade_b = [r for r in rows if r.get("grade") == "B"]
    print(f"  Grade A legs: {len(grade_a)}   Grade B legs: {len(grade_b)}")

    # ── T1/T2 pairing check ───────────────────────────────────────────────────
    pairs, lone = pair_t1_t2(rows)
    print(f"\n── Grade B T1/T2 Pairs ──────────────────────────────────────────────")
    if not pairs and not lone:
        print("  No Grade B trades yet.")
    else:
        print(f"  Paired: {len(pairs)}   Lone legs (anomaly): {len(lone)}")

    for p in pairs:
        t1, t2 = p["t1"], p["t2"]
        t1_out  = t1.get("outcome", "?")
        t2_out  = t2.get("outcome", "?")
        t1_pnl  = fmt_pct(t1.get("pnl_pct"))
        t2_pnl  = fmt_pct(t2.get("pnl_pct"))
        trail   = "✓ trail" if t1.get("trailing_used") or t2.get("trailing_used") else "no trail"
        candles = t1.get("candles_open", "?")
        sym     = t1.get("symbol", "?")
        side    = t1.get("side", "?")
        opened  = t1.get("opened_at", "?")[:16]

        # Anomaly checks
        flags = []
        if t2_out not in ("SL_HIT", "TP_HIT", "TRAIL_STOP", "E1_STALL", "E2_TIME"):
            flags.append(f"⚠ T2 outcome={t2_out}")
        if t1_out == "TP_HIT" and "breakeven" not in str(t2.get("outcome", "")).lower():
            pass  # T2 may still be open — not necessarily an anomaly

        flag_str = "  " + "  ".join(flags) if flags else ""
        print(f"  {sym:<10} {side:<5} {opened}  "
              f"T1:{t1_out:<12} {t1_pnl}  "
              f"T2:{t2_out:<12} {t2_pnl}  "
              f"{trail}  {candles}c{flag_str}")

    # ── Lone legs (should not happen with proper Grade B) ────────────────────
    if lone:
        print(f"\n── Lone Grade B legs (missing pair — investigate) ───────────────────")
        for r in lone:
            print(f"  {r.get('symbol'):<10} {r.get('label','?'):<4} "
                  f"opened={r.get('opened_at','?')[:16]}  "
                  f"outcome={r.get('outcome','?')}  pnl={fmt_pct(r.get('pnl_pct'))}")

    # ── Min-order-size fallback detection ─────────────────────────────────────
    # If a Grade B trade opened as Grade A (single leg), that's the fallback firing.
    # Proxy: Grade A trade in a session where Grade B would be expected
    # We can't detect this purely from outcomes, but flag Grade A trades
    # where score < 0.78 (they should have been Grade B).
    grade_a_low_score = [r for r in grade_a if (r.get("score") or 1.0) < 0.78]
    if grade_a_low_score:
        print(f"\n── Possible min-order fallback (Grade A with score < 0.78) ──────────")
        for r in grade_a_low_score:
            print(f"  {r.get('symbol'):<10} score={r.get('score', '?'):.2f}  "
                  f"opened={r.get('opened_at','?')[:16]}  "
                  f"pnl={fmt_pct(r.get('pnl_pct'))}")

    # ── Trailing stats ────────────────────────────────────────────────────────
    trail_count = sum(1 for r in rows if r.get("trailing_used"))
    print(f"\n── Trailing stats ───────────────────────────────────────────────────")
    print(f"  Trailing activated: {trail_count}/{total} legs ({trail_count/total*100:.0f}%)")

    # ── Outcome distribution ──────────────────────────────────────────────────
    from collections import Counter
    outcomes = Counter(r.get("outcome", "?") for r in rows)
    print(f"\n── Outcome distribution ─────────────────────────────────────────────")
    for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {outcome:<20} {count:>4}  ({pct:.0f}%)")

    # ── Per-week table ────────────────────────────────────────────────────────
    if days > 7:
        print(f"\n── Weekly P&L breakdown ─────────────────────────────────────────────")
        by_week = defaultdict(list)
        for r in rows:
            try:
                dt = datetime.strptime(r["closed_at"][:10], "%Y-%m-%d")
                week = dt.strftime("%Y-W%W")
            except Exception:
                week = "unknown"
            by_week[week].append(r)

        for week in sorted(by_week):
            week_rows = by_week[week]
            w_wins  = sum(1 for r in week_rows if (r.get("pnl_pct") or 0) > 0)
            w_total = len(week_rows)
            w_pnl   = sum(r.get("pnl_value", 0) or 0 for r in week_rows)
            w_wr    = w_wins / w_total * 100 if w_total else 0
            print(f"  {week}  legs={w_total:>3}  WR={w_wr:.0f}%  PnL=${w_pnl:+.2f}")


def compare_to_backtest(email: str, days: int):
    """Run a quick backtest on the same period and print Sharpe + trade count."""
    from backtester import start_backtest, get_run, init_backtest_tables
    init_backtest_tables()

    date_to   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_from = (datetime.now(timezone.utc) - timedelta(days=days + 14)).strftime("%Y-%m-%d")

    # Only check BTC and ETH for a quick comparison
    coins = ["BTCUSDT", "ETHUSDT"]
    print(f"\n── Backtest comparison ({date_from} → {date_to}) ──────────────────────")

    for coin in coins:
        run_id = start_backtest(
            email=f"monitor-opt3-{coin.lower()}@asym-internal",
            symbol=coin, mode="MINI_ASYM", style="DAY_TRADE",
            exchange="bybit", date_from=date_from, date_to=date_to,
            start_equity=1000.0,
            enable_t16=True, enable_t18=False, enable_s4=False,
            enable_mr_4h_gate=False, enable_regime_gate=False,
            enable_e2=False, enable_step2_score=False,
            enable_grade_b_all=True,  # Option 3 on for comparison
            enable_floor=True,
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            time.sleep(5)
            rec = get_run(run_id)
            if rec and rec.get("status") == "done":
                res = rec.get("results") or {}
                print(f"  {coin:<10} BT Sharpe={res.get('sharpe_ratio', '—'):.3f}  "
                      f"trades={res.get('total_trades', '?')}  "
                      f"WR={res.get('win_rate', 0)*100:.0f}%")
                break
            if rec and rec.get("status") == "error":
                print(f"  {coin:<10} backtest error: {rec.get('error','?')}")
                break
        else:
            print(f"  {coin:<10} backtest timed out")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days",    type=int, default=7,  help="Look back N days (default 7)")
    p.add_argument("--no-bt",   action="store_true",  help="Skip backtest comparison")
    p.add_argument("--real",    action="store_true",  help="Also report real account")
    args = p.parse_args()

    report(DEMO_EMAIL, "MAIN (demo)", args.days)
    if args.real:
        report(REAL_EMAIL, "REAL", args.days)

    if not args.no_bt:
        compare_to_backtest(DEMO_EMAIL, args.days)

    print()


if __name__ == "__main__":
    main()
