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

Match rule:
  • Same side (LONG / SHORT)
  • Entry price within 0.3%  ← primary match key
    (live "time" column = CLOSE time, not entry time, so time-based
     matching is unreliable for trades lasting > 1 candle)
  • SL within 0.15% of entry
  • Same exit reason family (WIN / LOSS / TRAIL)

NOTE — TP_PCT is intentionally NOT compared:
  The backtester applies backtest_default_tp_mult=2.0 for DAY_TRADE
  (mode_params.py), so backtest TP = 2× live TP by design.
  This is a known, accepted divergence — not a bug.

Parity rate = matched / total_live  (entry-price matches ÷ all live trades)
Acceptable threshold: ≥ 80%

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
    "enable_t16":          True,
    "enable_t18":          False,  # ENABLE_T18 = False after STEP 0
    "enable_s4":           False,
    "enable_mr_4h_gate":   False,  # ENABLE_MR_4H_GATE = False after STEP 0
    "t1_tp_mult":          1.00,   # T1 TP fixed to 1.00x
    "enable_grade_b_all":  True,   # Option 3 deployed to MAIN 2026-11-11
}

# ── Backtest config ────────────────────────────────────────────────────────────
MODE     = "MINI_ASYM"
STYLE    = "DAY_TRADE"
EXCHANGE = "bybit"
START_EQ = 1000.0

# ── Match tolerances ──────────────────────────────────────────────────────────
TOL_ENTRY_PCT     = 0.003              # 0.3% — entry price match window
TOL_SL_PCT        = 0.0015             # 0.15% — SL match (ATR-based, should be close)

# ── Exit reason families ──────────────────────────────────────────────────────
# Maps live `reason` field and backtest `outcome` to a canonical family.
def _live_family(reason: str) -> str:
    if not reason:
        return "UNKNOWN"
    r = reason.upper()
    # Use full phrases — the reason field contains "SL (1×ATR): X%" in position
    # details which would falsely match "SL". Full phrases avoid this.
    if "TRAILING STOP" in r or "TRAIL STOP" in r:
        return "TRAIL"
    if "TAKE PROFIT HIT" in r or "PROFIT HIT" in r:
        return "WIN"
    if "STOP LOSS HIT" in r or "STOP-LOSS" in r:
        return "LOSS"
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
    """Return MINI_ASYM DAY_TRADE trades for the given email list, last `days` days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    placeholders = ",".join(["%s"] * len(emails))
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM trades WHERE email IN ({placeholders}) AND time >= %s"
            f" AND mode = 'MINI_ASYM'"
            f" ORDER BY time ASC",
            list(emails) + [cutoff],
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def _dedup_entries(sym_trades):
    """
    Collapse T1/T2 Grade B pairs into one entry event per signal.

    Live fires Grade B as two separate trade records (T1 60% + T2 40%) with
    the SAME entry_price and side. The backtest has ONE record per signal.
    Dedup by (side, entry_price) so T2 duplicates don't inflate the
    unmatched count. We keep the first record (T1) as the representative.
    """
    seen = set()
    deduped = []
    for t in sym_trades:
        ep  = round(float(t.get("entry_price") or 0.0), 4)
        key = (t.get("side", "").upper(), ep)
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped

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
        enable_grade_b_all=flags.get("enable_grade_b_all", False),
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
            rj = rec.get("results") or {}
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
    # trades.sl stores sl_pct * 100 (e.g. 2.5 for a 2.5% stop), not an absolute price
    s = trade.get("sl", 0.0)
    return s / 100.0 if s > 0 else None

def tp_pct(trade):
    # trades.tp stores tp_pct * 100 (e.g. 5.0 for a 5% target)
    t = trade.get("tp", 0.0)
    return t / 100.0 if t > 0 else None

def find_bt_match(live_entry_price, live_side, available_bt):
    """
    Match by entry_price (±TOL_ENTRY_PCT) + same side.

    Rationale: live trades["time"] = close time, NOT entry time. Using
    close time vs backtest open_ts (entry time) fails for any trade lasting
    more than 1 candle. Entry price is a reliable proxy — the same entry
    price rarely repeats within a 2-3 month window.

    Returns (best_bt, price_diff_pct) or (None, inf).
    """
    best, best_d = None, float("inf")
    for _idx, bt in available_bt:  # available_bt is list of (i, bt) tuples
        if bt["side"].upper() != live_side.upper():
            continue
        if not bt.get("entry") or live_entry_price <= 0:
            continue
        d = abs(bt["entry"] - live_entry_price) / live_entry_price
        if d <= TOL_ENTRY_PCT and d < best_d:
            best_d = d
            best = bt
    return best, best_d

def compare(live, bt):
    """
    Returns list of (field, live_val, bt_val, note) tuples for every mismatch.

    SL_PCT excluded: live stores the FINAL sl (trailing-moved), not entry sl.
    TP_PCT excluded: backtest_default_tp_mult=2.0 makes BT TP = 2× live by design.
    Only EXIT_FAMILY is compared — the most meaningful behavioral check.
    """
    issues = []

    # Exit reason family — WIN / LOSS / TRAIL
    l_fam = _live_family(live.get("reason", ""))
    b_fam = _bt_family(bt.get("outcome", ""))
    if l_fam != "UNKNOWN" and b_fam != "UNKNOWN" and l_fam != b_fam:
        issues.append((
            "EXIT_FAMILY",
            f"{l_fam}",
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
    pr(f"Match rule : same side · entry_price ±0.3% · same exit family (WIN/LOSS/TRAIL)")
    pr("=" * 76)
    pr()

    # 1. Fetch live trades (real + demo)
    live_trades = fetch_trades([REAL_EMAIL, DEMO_EMAIL], args.days)
    if not live_trades:
        pr("No live/demo trades found in the last %d days. Exiting." % args.days)
        return

    pr(f"Found {len(live_trades)} trade(s) across both accounts (MINI_ASYM only).")
    pr()

    # 2. Group by symbol
    by_symbol = defaultdict(list)
    for t in live_trades:
        by_symbol[t["symbol"].upper()].append(t)

    # 3. Per-symbol: backtest + compare
    total_raw_trades = 0   # all individual live trade records
    total_live       = 0   # deduplicated entry events (T1/T2 collapsed)
    total_matched    = 0
    total_mismatch   = 0
    total_unmatched  = 0

    all_mismatches = []  # for summary section

    for symbol in sorted(by_symbol):
        sym_trades = sorted(by_symbol[symbol], key=lambda t: t["time"])
        t_min_ms = ts_ms(sym_trades[0]["time"])
        t_max_ms = ts_ms(sym_trades[-1]["time"])
        pad_ms   = 14 * 86_400_000   # 14-day pad ensures ≥250 candles
        # Minimum 30-day window so backtester always has enough candles
        span_ms  = max(t_max_ms - t_min_ms, 30 * 86_400_000)
        date_from = datetime.utcfromtimestamp((t_min_ms - pad_ms) / 1000).strftime("%Y-%m-%d")
        date_to   = datetime.utcfromtimestamp((t_min_ms + span_ms + pad_ms) / 1000).strftime("%Y-%m-%d")

        raw_count   = len(sym_trades)
        entry_events = _dedup_entries(sym_trades)  # T1/T2 collapsed
        total_raw_trades += raw_count

        pr(f"{'─'*76}")
        pr(f"SYMBOL: {symbol}   live trades: {raw_count} ({len(entry_events)} unique entries)   period: {date_from} → {date_to}")
        pr(f"{'─'*76}")

        try:
            bt_trades = run_backtest(symbol, date_from, date_to, flags)
        except Exception as exc:
            pr(f"  ERROR: backtest failed — {exc}")
            total_unmatched += len(entry_events)
            total_live += len(entry_events)
            continue

        if not bt_trades:
            pr(f"  Backtest returned 0 trades.")
            for lt in entry_events:
                pr(f"  NO_BT_SIGNAL  {fmt_ts(lt['time'])}  {lt['side']}  entry={float(lt.get('entry_price') or 0):.4f}")
                total_unmatched += 1
            total_live += len(entry_events)
            continue

        used_bt_indices = set()

        for lt in entry_events:
            total_live += 1
            lt_side       = lt.get("side", "").upper()
            lt_entry_price = float(lt.get("entry_price") or 0.0)

            avail = [(i, bt) for i, bt in enumerate(bt_trades) if i not in used_bt_indices]
            best_bt, best_d = find_bt_match(lt_entry_price, lt_side, avail)
            best_i = next(
                (i for i, bt in avail if bt is best_bt), -1
            ) if best_bt is not None else -1

            if best_bt is None:
                total_unmatched += 1
                pr(
                    f"  NO_MATCH   {fmt_ts(lt['time'])}  {lt_side}"
                    f"  entry={lt_entry_price:.4f}"
                    f"  reason={lt.get('reason','?')}"
                )
                continue

            used_bt_indices.add(best_i)
            total_matched += 1
            issues = compare(lt, best_bt)

            ep_diff_pct = best_d * 100
            tag = "OK      " if not issues else "MISMATCH"
            pr(
                f"  [{tag}]  live={fmt_ts(lt['time'])}  bt_open={fmt_ms(best_bt['open_ts'])}"
                f"  entry_Δ{ep_diff_pct:.2f}%  {lt_side}"
                f"  live_exit={lt.get('reason','')}  bt_exit={best_bt.get('outcome','')}"
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
    pr(f"  Raw live trades      : {total_raw_trades}  (all T1+T2 records, MINI_ASYM only)")
    pr(f"  Unique entry events  : {total_live}  (T1/T2 pairs collapsed to 1)")
    pr(f"  Matched to backtest  : {total_matched}  (entry_price ±0.3%, same side)")
    pr(f"  Unmatched            : {total_unmatched}  (no backtest entry at same price/side)")
    pr(f"  Exit-family mismatch : {total_mismatch}  (WIN/LOSS/TRAIL differs)")
    entry_pct = total_matched / total_live * 100 if total_live else 0.0
    clean_pct = (total_matched - total_mismatch) / total_live * 100 if total_live else 0.0
    pr(f"  Entry match rate     : {entry_pct:.1f}%  (matched / unique events — target ≥80%)")
    pr(f"  Clean parity rate    : {clean_pct:.1f}%  (no exit mismatch / unique events)")
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
