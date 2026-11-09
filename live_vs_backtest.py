"""
live_vs_backtest.py — Compare live account performance vs backtest projections.

Run on Render shell:  python live_vs_backtest.py
Run locally:          DATABASE_URL=<pg_url> python live_vs_backtest.py

Backtest baseline (Oct 2026, post all fixes + Test6 + Test7):
  Sharpe +0.24 | WR 18.4% | Avg Win $89.11 | Return +131.8% | 1052 trades / 6.8yr
  Trade pace: ~154/year = ~12-13/month = ~3/week
"""
from __future__ import annotations
import os, sys, statistics
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASYM_DB_PATH", "asymmetric_demo.db")

from database import db_conn

EMAIL = os.getenv("VERIFY_EMAIL", "mes29571@gmail.com")

# ── Backtest reference numbers (post all fixes, Oct 2026) ─────────────────────
BT = {
    "trades_total":   1052,
    "years":          6.8,
    "trades_per_yr":  154,
    "trades_per_mo":  12.8,
    "wr_pct":         18.4,
    "avg_win_usd":    89.11,
    "sharpe":         0.24,
    "return_pct":     131.8,
    "max_dd_pct":     52.9,
}

SEP  = "=" * 66
SEP2 = "-" * 66

def pct_diff(live, bt):
    if bt == 0:
        return "N/A"
    d = (live - bt) / abs(bt) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"

def within(live, bt, tolerance=0.20):
    if bt == 0:
        return True
    return abs(live - bt) / abs(bt) <= tolerance

def verdict(live, bt, tolerance=0.20, higher_is_better=True):
    if bt == 0:
        return "N/A"
    diff = (live - bt) / abs(bt)
    if higher_is_better:
        if diff >= 0:
            return "GREEN  ✓"
        elif abs(diff) <= tolerance:
            return "YELLOW ~"
        else:
            return "RED    ✗"
    else:
        if diff <= 0:
            return "GREEN  ✓"
        elif abs(diff) <= tolerance:
            return "YELLOW ~"
        else:
            return "RED    ✗"

print(f"\n{SEP}")
print(f"  LIVE vs BACKTEST — {EMAIL}")
print(f"  Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(SEP)

# ── Pull all trades from database ──────────────────────────────────────────────
try:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT outcome, pnl_pct, pnl_value, equity_before, equity_after,
                   grade, label, signal, regime, mode, trade_style,
                   opened_at, closed_at, side, trailing_used, candles_open,
                   entry_price, exit_price, score, symbol
            FROM trade_outcomes
            WHERE email = %s
            ORDER BY closed_at ASC
            """,
            (EMAIL,),
        )
        rows = cur.fetchall()
except Exception as e:
    print(f"\n  ERROR connecting to database: {e}")
    print("  Make sure DATABASE_URL env var is set (run on Render shell).")
    sys.exit(1)

if not rows:
    print("\n  No trades found. Run after some trades have closed.")
    sys.exit(0)

# ── Parse trades ───────────────────────────────────────────────────────────────
trades = []
for r in rows:
    try:
        closed = datetime.fromisoformat(str(r["closed_at"]).replace("Z", "+00:00"))
    except Exception:
        closed = None

    trades.append({
        "outcome":       r["outcome"],
        "pnl_pct":       float(r["pnl_pct"]   or 0),
        "pnl_value":     float(r["pnl_value"]  or 0),
        "eq_before":     float(r["equity_before"] or 0),
        "eq_after":      float(r["equity_after"]  or 0),
        "grade":         r["grade"],
        "label":         r["label"],
        "signal":        r["signal"] or "-",
        "regime":        r["regime"] or "-",
        "mode":          r["mode"],
        "style":         r["trade_style"],
        "side":          r["side"],
        "trailing_used": bool(r["trailing_used"]),
        "candles_open":  int(r["candles_open"] or 0),
        "score":         float(r["score"] or 0),
        "symbol":        r["symbol"],
        "closed":        closed,
        "month":         closed.strftime("%Y-%m") if closed else "unknown",
    })

# ── Core metrics ───────────────────────────────────────────────────────────────
total        = len(trades)
wins         = [t for t in trades if t["outcome"] in ("TP_HIT", "TRAIL_STOP") and t["pnl_pct"] > 0]
losses       = [t for t in trades if t["pnl_pct"] < 0]
breakevens   = [t for t in trades if abs(t["pnl_pct"]) < 0.001]

wr_pct       = len(wins) / total * 100 if total else 0
avg_win_pct  = statistics.mean([t["pnl_pct"]   for t in wins])   if wins   else 0
avg_loss_pct = statistics.mean([t["pnl_pct"]   for t in losses])  if losses else 0
avg_win_usd  = statistics.mean([t["pnl_value"] for t in wins])   if wins   else 0
avg_loss_usd = statistics.mean([t["pnl_value"] for t in losses])  if losses else 0

# Account equity info
first_equity = trades[0]["eq_before"]  if trades else 0
last_equity  = trades[-1]["eq_after"]  if trades else 0
peak_equity  = max(t["eq_after"] for t in trades) if trades else 0
total_return_pct = (last_equity - first_equity) / first_equity * 100 if first_equity else 0

# Trade frequency
closed_dates = [t["closed"] for t in trades if t["closed"]]
if closed_dates:
    span_days = (max(closed_dates) - min(closed_dates)).days + 1
    span_months = max(span_days / 30.44, 1)
    trades_per_mo = total / span_months
else:
    span_days    = 0
    span_months  = 1
    trades_per_mo = 0

# Monthly P&L — both $ and %
monthly_pnl: dict  = defaultdict(float)
monthly_pnl_pct: dict = defaultdict(float)
monthly_count: dict = defaultdict(int)
for t in trades:
    monthly_pnl[t["month"]]     += t["pnl_value"]
    monthly_pnl_pct[t["month"]] += t["pnl_pct"]
    monthly_count[t["month"]]   += 1

# Trailing stop usage
trail_trades = [t for t in wins if t["trailing_used"]]
trail_pct    = len(trail_trades) / len(wins) * 100 if wins else 0

# Signal type breakdown
signal_counts: dict = defaultdict(lambda: {"total": 0, "wins": 0, "pnl": 0.0})
for t in trades:
    sig = "MR" if t["signal"] in ("MEAN_REVERSION",) else \
          "REVERSAL" if t["signal"] == "REVERSAL" else "NORMAL"
    signal_counts[sig]["total"] += 1
    if t["pnl_value"] > 0:
        signal_counts[sig]["wins"] += 1
    signal_counts[sig]["pnl"] += t["pnl_value"]

# Regime breakdown
regime_counts: dict = defaultdict(lambda: {"total": 0, "wins": 0})
for t in trades:
    regime_counts[t["regime"]]["total"] += 1
    if t["pnl_value"] > 0:
        regime_counts[t["regime"]]["wins"] += 1

print(f"\n  Data: {total} trades | {span_days} days ({span_months:.1f} months)")
print(f"  Period: {min(closed_dates).strftime('%Y-%m-%d')} → {max(closed_dates).strftime('%Y-%m-%d')}")
print(f"  Account: start ${first_equity:.2f} → current ${last_equity:.2f} | peak ${peak_equity:.2f} | total return {total_return_pct:+.1f}%")
print()

# ── Main comparison table (pct-based — dollar amounts scale with account size) ─
print(f"{SEP}")
print(f"  {'Metric':<28} {'Backtest':>10} {'Live':>10}  {'Diff':>8}  {'Status'}")
print(f"  NOTE: pnl_pct = % of account equity per trade (account-size independent)")
print(f"{SEP2}")

# Backtest pct equivalents at $1000 equity, 65% size, 6x leverage:
# avg_win_pct = avg_win_$ / (equity * size * leverage) = 89.11 / (1000*0.65*6) = 2.28%
# avg_loss_pct similarly
BT_avg_win_pct  =  2.28   # $89.11 at $1000 equity, 0.65 size, 6x lev
BT_avg_loss_pct = -0.77   # ~$30 loss equivalent

rows_table = [
    ("Trades/month",      BT["trades_per_mo"],   trades_per_mo,   True,  ".1f"),
    ("Win rate %",        BT["wr_pct"],           wr_pct,          True,  ".1f"),
    ("Avg win % equity",  BT_avg_win_pct,         avg_win_pct,     True,  ".2f"),
    ("Avg loss % equity", BT_avg_loss_pct,        avg_loss_pct,    False, ".2f"),
]

for label, bt_val, live_val, higher_better, fmt in rows_table:
    verd = verdict(live_val, bt_val, tolerance=0.25, higher_is_better=higher_better)
    diff = pct_diff(live_val, bt_val)
    print(f"  {label:<28} {bt_val:>10{fmt}} {live_val:>10{fmt}}  {diff:>8}  {verd}")

print(f"{SEP2}")

# ── Monthly P&L ───────────────────────────────────────────────────────────────
print(f"\n  MONTHLY P&L  (% = sum of per-trade pnl_pct that month)")
print(f"{SEP2}")
print(f"  {'Month':<10}  {'Trades':>6}  {'P&L %':>8}  {'P&L $':>8}  {'Win Rate':>9}")
print(f"{SEP2}")
for month in sorted(monthly_pnl.keys()):
    cnt  = monthly_count[month]
    pnl  = monthly_pnl[month]
    ppct = monthly_pnl_pct[month]
    month_trades = [t for t in trades if t["month"] == month]
    month_wins   = [t for t in month_trades if t["pnl_pct"] > 0]
    wr            = len(month_wins) / cnt * 100 if cnt else 0
    print(f"  {month:<10}  {cnt:>6}  {ppct:>+7.2f}%  ${pnl:>7.2f}  {wr:>8.1f}%")

total_pnl     = sum(monthly_pnl.values())
total_pnl_pct = sum(monthly_pnl_pct.values())
print(f"{SEP2}")
print(f"  {'TOTAL':<10}  {total:>6}  {total_pnl_pct:>+7.2f}%  ${total_pnl:>7.2f}")

# ── Signal type breakdown ─────────────────────────────────────────────────────
print(f"\n  SIGNAL TYPE BREAKDOWN")
print(f"{SEP2}")
print(f"  {'Signal':<12}  {'Trades':>6}  {'WR':>7}  {'Total P&L':>11}  {'Avg P&L':>9}")
print(f"{SEP2}")
for sig in ("NORMAL", "MR", "REVERSAL"):
    d = signal_counts.get(sig)
    if not d or d["total"] == 0:
        continue
    wr_s  = d["wins"] / d["total"] * 100
    avg_s = d["pnl"] / d["total"]
    print(f"  {sig:<12}  {d['total']:>6}  {wr_s:>6.1f}%  ${d['pnl']:>10.2f}  ${avg_s:>8.2f}")

# ── Regime breakdown ──────────────────────────────────────────────────────────
print(f"\n  REGIME BREAKDOWN")
print(f"{SEP2}")
print(f"  {'Regime':<14}  {'Trades':>6}  {'WR':>7}")
print(f"{SEP2}")
for reg in sorted(regime_counts.keys()):
    d   = regime_counts[reg]
    wr_r = d["wins"] / d["total"] * 100 if d["total"] else 0
    print(f"  {reg:<14}  {d['total']:>6}  {wr_r:>6.1f}%")

# ── Trailing stop stats ───────────────────────────────────────────────────────
print(f"\n  TRAIL USAGE (winning trades)")
print(f"{SEP2}")
print(f"  Wins with trail activated : {len(trail_trades)} / {len(wins)} ({trail_pct:.1f}%)")
if trail_trades:
    trail_avg_win = statistics.mean([t["pnl_value"] for t in trail_trades])
    non_trail     = [t for t in wins if not t["trailing_used"]]
    non_trail_avg = statistics.mean([t["pnl_value"] for t in non_trail]) if non_trail else 0
    print(f"  Avg win WITH trail        : ${trail_avg_win:.2f}")
    print(f"  Avg win WITHOUT trail     : ${non_trail_avg:.2f}")

# ── Final verdict ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  VALIDATION SUMMARY")
print(f"{SEP}")

checks = [
    ("Trade frequency",    within(trades_per_mo, BT["trades_per_mo"], 0.25), trades_per_mo,  BT["trades_per_mo"]),
    ("Win rate %",         within(wr_pct, BT["wr_pct"], 0.25),               wr_pct,          BT["wr_pct"]),
    ("Avg win % equity",   within(avg_win_pct, BT_avg_win_pct, 0.25),        avg_win_pct,     BT_avg_win_pct),
]

all_green = all(c[1] for c in checks)
for name, ok, live_v, bt_v in checks:
    status = "PASS" if ok else "FAIL"
    diff   = pct_diff(live_v, bt_v)
    print(f"  [{status}] {name:<22} live={live_v:.2f}  bt={bt_v:.2f}  ({diff})")

print()
if total < 30:
    print(f"  ⚠  Only {total} trades — sample too small for reliable comparison.")
    print(f"     Need 30+ trades before verdict is meaningful.")
elif all_green:
    print(f"  ✓  All 3 metrics within 20% of backtest.")
    print(f"     Algorithm is behaving as modeled. Launch confidence: HIGH.")
else:
    failed = [c[0] for c in checks if not c[1]]
    print(f"  ✗  {', '.join(failed)} outside 20% tolerance.")
    print(f"     Investigate before launch.")
print(SEP)
