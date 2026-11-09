#!/usr/bin/env python3
"""
run_parity_test.py
──────────────────
Trade-by-trade parity check: replay a backtest over the same period as
recent live/demo trades, then compare each live trade to its backtest
counterpart.

Default run (historical — matches what live was doing during the 90-day window):
  python run_parity_test.py

For current (corrected) flags:
  python run_parity_test.py --current-flags

Match rule (all four must hold):
  • Same side (LONG / SHORT)
  • Entry within 1 candle (≤1 h)
  • SL and TP each within 0.1% of entry
  • Same exit reason (outcome family: WIN / LOSS / TRAIL)

Writes parity_report.txt alongside this script.
"""

import sys
import os
import time
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from database import db_conn
from backtester import start_backtest, get_run

# ── Accounts ──────────────────────────────────────────────────────────────────
REAL_EMAIL = "mes29571@gmail.com"
DEMO_EMAIL = "admin@demo.com"
INTERNAL   = {"baseline@asym-ai-internal", "parity@asym-ai-internal"}

# ── Historical flags (what live was actually doing during the 90-day window) ──
HISTORICAL_FLAGS = {
    "enable_t16":        True,
    "enable_t18":        True,   # ENABLE_T18 was True
    "enable_s4":         False,
    "enable_mr_4h_gate": True,   # ENABLE_MR_4H_GATE was True
    "t1_tp_mult":        0.80,   # T1 TP bug: was 0.80x
}

# ── Current (corrected) flags ─────────────────────────────────────────────────
CURRENT_FLAGS = {
    "enable_t16":        True,
    "enable_t18":        False,  # ENABLE_T18 = False after STEP 0
    "enable_s4":         False,
    "enable_mr_4h_gate": False,  # ENABLE_MR_4H_GATE = False after STEP 0
    "t1_tp_mult":        1.00,   # T1 TP fixed to 1.00x
}

# ── Backtest config ────────────────────────────────────────────────────────────
MODE     = "MINI_ASYM"
STYLE    = "DAY_TRADE"
EXCHANGE = "bybit"
START_EQ = 1000.0

# ── Match tolerances ──────────────────────────────────────────────────────────
CANDLE_MS         = 3_600_000          # 1 h in ms
MATCH_WINDOW_MS   = 1 * CANDLE_MS     # entry within 1 candle
TOL_SL_PCT        = 0.001              # 0.1%
TOL_TP_PCT        = 0.001              # 0.1%

# ── Exit reason families ──────────────────────────────────────────────────────
# Maps live `reason` field and backtest `outcome` to a canonical family.
def _live_family(reason: str) -> str:
    if not reason:
        return "UNKNOWN"
    r = reason.upper()
    if any(k in r for k in ("SL", "STOP", "LOSS")):
        return "LOSS"
    if any(k in r for k in ("TP", "TARGET", "PROFIT")):
        return "WIN"
    if any(k in r for k in ("TRAIL", "BE", "BREAKEVEN")):
        return "TRAIL"
    return "UNKNOWN"

def _bt_family(outcome: str) -> str:
    if not outcome:
        return "UNKNOWN"
    o = outcome.upper()
    if "SL_HIT" in o or "T2_SL" in o:
        return "LOSS"
    if "TP_HIT" in o or "T1_TP_T2_TP" in o:
        return "WIN"
    if "TRAIL" in o or "BE" in o or "SCALE_BE" in o or "SCALE_TRAIL" in o:
        return "TRAIL"
    return "UNKNOWN"

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_trades(emails, days):
    """Return trades for the given email list, last `days` days, sorted by time."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    placeholders = ",".join(["%s"] * len(emails))
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM trades WHERE email IN ({placeholders}) AND time >= %s"
            f" ORDER BY time ASC",
            list(emails) + [cutoff],
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]

# ─────────────────────────────────────────────────────────────────────────────
# Backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(symbol, date_from, date_to, flags, timeout_min=30):
    """Start a backtest and return its trade_log list."""
    run_id = start_backtest(
        email="parity@asym-ai-internal",
        symbol=symbol,
        mode=MODE,
        style=STYLE,
        exchange=EXCHANGE,
        date_from=date_from,
        date_to=date_to,
        start_equity=START_EQ,
        enable_t16=flags["enable_t16"],
        enable_t18=flags["enable_t18"],
        enable_s4=flags["enable_s4"],
        enable_mr_4h_gate=flags["enable_mr_4h_gate"],
        t1_tp_mult=flags["t1_tp_mult"],
    )
    label = f"{symbol} {date_from} → {date_to}"
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        rec = get_run(run_id)
        if not rec:
            time.sleep(5)
            continue
        status = rec.get("status", "")
        if status == "done":
            print(f"  ✓ {label}")
            rj = rec.get("result_json") or {}
            return rj.get("trade_log") or []
        if status == "error":
            raise RuntimeError(f"{label} failed: {rec.get('error', 'unknown')}")
        prog = rec.get("progress", 0)
        print(f"  … {label}  {prog}%  ", end="\r", flush=True)
        time.sleep(10)
    raise TimeoutError(f"{label} timed out after {timeout_min} min")

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────

def ts_ms(time_str):
    s = str(time_str).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    return 0

def fmt_ms(ms):
    if not ms:
        return "—"
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")

def fmt_ts(time_str):
    return str(time_str)[:16] if time_str else "—"

# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def sl_pct(trade):
    e = trade.get("entry_price", 0.0)
    s = trade.get("sl", 0.0)
    return abs(e - s) / e if e > 0 and s > 0 else None

def tp_pct(trade):
    e = trade.get("entry_price", 0.0)
    t = trade.get("tp", 0.0)
    return abs(e - t) / e if e > 0 and t > 0 else None

def find_bt_match(live_ts, live_side, available_bt):
    """
    Return (bt_trade, delta_ms) for the best matching backtest trade:
      - same side
      - entry within MATCH_WINDOW_MS
    Returns (None, inf) if nothing qualifies.
    """
    best, best_d = None, float("inf")
    for bt in available_bt:
        if bt["side"].upper() != live_side.upper():
            continue
        d = abs(bt["open_ts"] - live_ts)
        if d <= MATCH_WINDOW_MS and d < best_d:
            best_d = d
            best = bt
    return best, best_d

def compare(live, bt):
    """
    Returns list of (field, live_val, bt_val, note) tuples for every mismatch.
    """
    issues = []

    # SL %
    l_sl = sl_pct(live)
    b_sl = bt.get("sl_pct")
    if l_sl is not None and b_sl is not None:
        if abs(l_sl - b_sl) > TOL_SL_PCT:
            issues.append((
                "SL_PCT",
                f"{l_sl*100:.3f}%",
                f"{b_sl*100:.3f}%",
                f"diff {abs(l_sl-b_sl)*100:.3f}%",
            ))

    # TP %
    l_tp = tp_pct(live)
    b_tp = bt.get("tp_pct")
    if l_tp is not None and b_tp is not None:
        if abs(l_tp - b_tp) > TOL_TP_PCT:
            issues.append((
                "TP_PCT",
                f"{l_tp*100:.3f}%",
                f"{b_tp*100:.3f}%",
                f"diff {abs(l_tp-b_tp)*100:.3f}%",
            ))

    # Exit reason family
    l_fam = _live_family(live.get("reason", ""))
    b_fam = _bt_family(bt.get("outcome", ""))
    if l_fam != "UNKNOWN" and b_fam != "UNKNOWN" and l_fam != b_fam:
        issues.append((
            "EXIT_FAMILY",
            f"{l_fam} ({live.get('reason','')})",
            f"{b_fam} ({bt.get('outcome','')})",
            "different exit type",
        ))

    return issues

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest vs live parity test")
    parser.add_argument("--days",          type=int,  default=90,
                        help="Trailing days of live trades to compare (default: 90)")
    parser.add_argument("--current-flags", action="store_true",
                        help="Use current corrected flags instead of historical")
    args = parser.parse_args()

    flags      = CURRENT_FLAGS if args.current_flags else HISTORICAL_FLAGS
    flag_label = "CURRENT (corrected)" if args.current_flags else "HISTORICAL (as-live)"

    lines = []
    def pr(s=""):
        lines.append(s)
        print(s)

    pr("=" * 76)
    pr("ASYMMETRIC AI — BACKTEST vs LIVE PARITY REPORT")
    pr(f"Generated  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    pr(f"Window     : last {args.days} days")
    pr(f"Accounts   : {REAL_EMAIL}  +  {DEMO_EMAIL}")
    pr(f"Flag set   : {flag_label}")
    pr(f"  enable_t16={flags['enable_t16']}  enable_t18={flags['enable_t18']}"
       f"  mr_gate={flags['enable_mr_4h_gate']}  t1_tp={flags['t1_tp_mult']}×")
    pr(f"Match rule : same side · entry ≤1 candle · SL/TP ≤0.1% · same exit family")
    pr("=" * 76)
    pr()

    # 1. Fetch live trades (real + demo)
    live_trades = fetch_trades([REAL_EMAIL, DEMO_EMAIL], args.days)
    if not live_trades:
        pr("No live/demo trades found in the last %d days. Exiting." % args.days)
        return

    pr(f"Found {len(live_trades)} trade(s) across both accounts.")
    pr()

    # 2. Deduplicate by (symbol, time, side) — real + demo log the same signal twice
    seen = set()
    deduped = []
    for t in live_trades:
        key = (t["symbol"].upper(), str(t["time"])[:16], t.get("side", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    live_trades = deduped
    pr(f"After dedup (real+demo): {len(live_trades)} unique trade(s).")
    pr()

    # 3. Group by symbol
    by_symbol = defaultdict(list)
    for t in live_trades:
        by_symbol[t["symbol"].upper()].append(t)

    # 3. Per-symbol: backtest + compare
    total_live     = 0
    total_matched  = 0
    total_mismatch = 0
    total_unmatched= 0

    all_mismatches = []  # for summary section

    for symbol in sorted(by_symbol):
        sym_trades = sorted(by_symbol[symbol], key=lambda t: t["time"])
        t_min_ms   = ts_ms(sym_trades[0]["time"])
        t_max_ms   = ts_ms(sym_trades[-1]["time"])
        # Pad enough so total window is always ≥ 7 days (backtester minimum)
        span_days  = max(0, (t_max_ms - t_min_ms) // 86_400_000)
        pad_days   = max(4, 8 - span_days)
        pad_ms     = pad_days * 86_400_000
        date_from  = datetime.utcfromtimestamp((t_min_ms - pad_ms) / 1000).strftime("%Y-%m-%d")
        date_to    = datetime.utcfromtimestamp((t_max_ms + pad_ms) / 1000).strftime("%Y-%m-%d")

        pr(f"{'─'*76}")
        pr(f"SYMBOL: {symbol}   live trades: {len(sym_trades)}   period: {date_from} → {date_to}")
        pr(f"{'─'*76}")

        try:
            bt_trades = run_backtest(symbol, date_from, date_to, flags)
        except Exception as exc:
            pr(f"  ERROR: backtest failed — {exc}")
            total_unmatched += len(sym_trades)
            total_live += len(sym_trades)
            continue

        if not bt_trades:
            pr(f"  Backtest returned 0 trades.")
            for lt in sym_trades:
                pr(f"  NO_BT_SIGNAL  {fmt_ts(lt['time'])}  {lt['side']}")
                total_unmatched += 1
            total_live += len(sym_trades)
            continue

        used_bt_indices = set()

        for lt in sym_trades:
            total_live += 1
            lt_ms   = ts_ms(lt["time"])
            lt_side = lt.get("side", "").upper()

            avail = [(i, bt) for i, bt in enumerate(bt_trades) if i not in used_bt_indices]
            best_bt = None
            best_d  = float("inf")
            best_i  = -1
            for idx, bt in avail:
                if bt["side"].upper() != lt_side:
                    continue
                d = abs(bt["open_ts"] - lt_ms)
                if d <= MATCH_WINDOW_MS and d < best_d:
                    best_d  = d
                    best_bt = bt
                    best_i  = idx

            if best_bt is None:
                total_unmatched += 1
                pr(
                    f"  NO_MATCH   {fmt_ts(lt['time'])}  {lt_side}"
                    f"  entry={lt.get('entry_price','?')}"
                    f"  reason={lt.get('reason','?')}"
                )
                continue

            used_bt_indices.add(best_i)
            total_matched += 1
            issues = compare(lt, best_bt)

            delta_min = int(best_d / 60_000)
            tag = "OK      " if not issues else "MISMATCH"
            pr(
                f"  [{tag}]  live={fmt_ts(lt['time'])}  bt={fmt_ms(best_bt['open_ts'])}"
                f"  Δ{delta_min}min  {lt_side}  live_exit={lt.get('reason','')}  bt_exit={best_bt.get('outcome','')}"
            )
            if issues:
                total_mismatch += 1
                for field, lv, bv, note in issues:
                    pr(f"            ↳ {field:12s}  live={lv}  bt={bv}  ({note})")
                all_mismatches.append({
                    "symbol":    symbol,
                    "time":      fmt_ts(lt["time"]),
                    "side":      lt_side,
                    "issues":    issues,
                    "live_exit": lt.get("reason", ""),
                    "bt_exit":   best_bt.get("outcome", ""),
                })

        # Backtest trades with no live counterpart
        orphan_bt = [bt for i, bt in enumerate(bt_trades) if i not in used_bt_indices]
        if orphan_bt:
            pr(f"  Backtest-only ({len(orphan_bt)} trade(s) with no live match):")
            for bt in orphan_bt:
                pr(
                    f"    {fmt_ms(bt['open_ts'])}  {bt['side']}"
                    f"  outcome={bt['outcome']}"
                    f"  pnl={bt['pnl']:+.2f}"
                )
        pr()

    # ── Summary ───────────────────────────────────────────────────────────────
    pr("=" * 76)
    pr("SUMMARY")
    pr(f"  Live/demo trades     : {total_live}")
    pr(f"  Matched to backtest  : {total_matched}")
    pr(f"  Unmatched            : {total_unmatched}  (backtest had no signal within 1 candle & same side)")
    pr(f"  Mismatches found     : {total_mismatch}")
    pct = (total_matched - total_mismatch) / total_matched * 100 if total_matched else 0.0
    pr(f"  Parity rate          : {pct:.1f}%")
    pr()

    if all_mismatches:
        pr("MISMATCH DETAIL:")
        for m in all_mismatches:
            pr(f"  {m['symbol']}  {m['time']}  {m['side']}")
            pr(f"    live_exit={m['live_exit']}  bt_exit={m['bt_exit']}")
            for field, lv, bv, note in m["issues"]:
                pr(f"    {field}: live={lv}  bt={bv}  ({note})")
        pr()

    if total_unmatched > 0:
        pr("UNMATCHED TRADES — possible causes:")
        pr("  • Live entered but backtest score was just below threshold")
        pr("  • Session quality gate blocked backtest entry at that hour")
        pr("  • Daily bias (BULL/BEAR/NEUTRAL) differed — daily klines diverged")
        pr("  • RANGING regime blocked backtest but live traded anyway")
        pr("  • enable_mr_4h_gate blocked backtest signal")
        pr()

    pr("=" * 76)

    report_path = os.path.join(os.path.dirname(__file__), "parity_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
