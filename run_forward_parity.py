#!/usr/bin/env python3
"""
run_forward_parity.py
─────────────────────
Forward parity check: runs every day (or on demand) and compares every
new DEMO trade against a fresh backtest for the same symbol+period.

Designed to be called by a cron job, Render scheduled job, or manually:

  python run_forward_parity.py              # check last 48 h
  python run_forward_parity.py --hours 72  # check last 72 h

Log is appended to forward_parity_log.txt.
Exits with code 1 if any mismatch is found (useful for CI / alert hooks).

Match rule (same as run_parity_test.py):
  • Same side · entry ≤ 1 candle · SL/TP ≤ 0.1% · same exit family
Flags used: current corrected flags (what the demo engine runs today).
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
DEMO_EMAIL = "admin@demo.com"
REAL_EMAIL = "mes29571@gmail.com"

# ── Current corrected flags (keep in sync with live engine) ──────────────────
CURRENT_FLAGS = {
    "enable_t16":        True,
    "enable_t18":        False,
    "enable_s4":         False,
    "enable_mr_4h_gate": False,
    "t1_tp_mult":        1.00,
}

MODE     = "MINI_ASYM"
STYLE    = "DAY_TRADE"
EXCHANGE = "bybit"
START_EQ = 1000.0

CANDLE_MS       = 3_600_000
MATCH_WINDOW_MS = 1 * CANDLE_MS
TOL_SL_PCT      = 0.001
TOL_TP_PCT      = 0.001
LOG_FILE        = os.path.join(os.path.dirname(__file__), "forward_parity_log.txt")

# ─────────────────────────────────────────────────────────────────────────────

def _live_family(reason):
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

def _bt_family(outcome):
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
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M") if ms else "—"

def sl_pct_of(trade):
    e, s = trade.get("entry_price", 0.0), trade.get("sl", 0.0)
    return abs(e - s) / e if e > 0 and s > 0 else None

def tp_pct_of(trade):
    e, t = trade.get("entry_price", 0.0), trade.get("tp", 0.0)
    return abs(e - t) / e if e > 0 and t > 0 else None

def fetch_recent_trades(hours):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM trades WHERE email IN (%s,%s) AND time >= %s ORDER BY time ASC",
            (DEMO_EMAIL, REAL_EMAIL, cutoff),
        )
        return [dict(r) for r in cur.fetchall()]

def run_backtest(symbol, date_from, date_to, timeout_min=20):
    run_id = start_backtest(
        email="parity@asym-ai-internal",
        symbol=symbol,
        mode=MODE,
        style=STYLE,
        exchange=EXCHANGE,
        date_from=date_from,
        date_to=date_to,
        start_equity=START_EQ,
        enable_t16=CURRENT_FLAGS["enable_t16"],
        enable_t18=CURRENT_FLAGS["enable_t18"],
        enable_s4=CURRENT_FLAGS["enable_s4"],
        enable_mr_4h_gate=CURRENT_FLAGS["enable_mr_4h_gate"],
        t1_tp_mult=CURRENT_FLAGS["t1_tp_mult"],
    )
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        rec = get_run(run_id)
        if not rec:
            time.sleep(5)
            continue
        if rec.get("status") == "done":
            rj = rec.get("result_json") or {}
            return rj.get("trade_log") or []
        if rec.get("status") == "error":
            raise RuntimeError(rec.get("error", "unknown"))
        time.sleep(10)
    raise TimeoutError(f"{symbol} backtest timed out")

def compare_trade(live, bt):
    issues = []
    l_sl, b_sl = sl_pct_of(live), bt.get("sl_pct")
    if l_sl is not None and b_sl is not None and abs(l_sl - b_sl) > TOL_SL_PCT:
        issues.append(("SL_PCT", f"{l_sl*100:.3f}%", f"{b_sl*100:.3f}%",
                        f"diff {abs(l_sl-b_sl)*100:.3f}%"))
    l_tp, b_tp = tp_pct_of(live), bt.get("tp_pct")
    if l_tp is not None and b_tp is not None and abs(l_tp - b_tp) > TOL_TP_PCT:
        issues.append(("TP_PCT", f"{l_tp*100:.3f}%", f"{b_tp*100:.3f}%",
                        f"diff {abs(l_tp-b_tp)*100:.3f}%"))
    l_fam = _live_family(live.get("reason", ""))
    b_fam = _bt_family(bt.get("outcome", ""))
    if l_fam != "UNKNOWN" and b_fam != "UNKNOWN" and l_fam != b_fam:
        issues.append(("EXIT_FAMILY",
                        f"{l_fam}({live.get('reason','')})",
                        f"{b_fam}({bt.get('outcome','')})",
                        "different exit type"))
    return issues

def append_log(text):
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")

def main():
    parser = argparse.ArgumentParser(description="Forward parity check (daily)")
    parser.add_argument("--hours", type=int, default=48,
                        help="Look-back window in hours (default: 48)")
    args = parser.parse_args()

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    banner  = f"\n{'='*72}\nFORWARD PARITY CHECK  {now_str}  (last {args.hours}h)\n{'='*72}"
    print(banner)
    append_log(banner)

    trades = fetch_recent_trades(args.hours)
    if not trades:
        msg = "No demo/real trades in the last %dh. Parity: N/A." % args.hours
        print(msg)
        append_log(msg)
        sys.exit(0)

    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"].upper()].append(t)

    found_mismatch = False

    for symbol in sorted(by_symbol):
        sym_trades = sorted(by_symbol[symbol], key=lambda t: t["time"])
        t_min_ms   = ts_ms(sym_trades[0]["time"])
        t_max_ms   = ts_ms(sym_trades[-1]["time"])
        pad_ms     = 2 * 86_400_000
        date_from  = datetime.utcfromtimestamp((t_min_ms - pad_ms) / 1000).strftime("%Y-%m-%d")
        date_to    = datetime.utcfromtimestamp((t_max_ms + pad_ms) / 1000).strftime("%Y-%m-%d")

        hdr = f"\n{symbol}  ({len(sym_trades)} trades)  {date_from} → {date_to}"
        print(hdr)
        append_log(hdr)

        try:
            bt_trades = run_backtest(symbol, date_from, date_to)
        except Exception as exc:
            msg = f"  ERROR: {exc}"
            print(msg)
            append_log(msg)
            continue

        used = set()
        for lt in sym_trades:
            lt_ms   = ts_ms(lt["time"])
            lt_side = lt.get("side", "").upper()

            best_bt, best_d, best_i = None, float("inf"), -1
            for i, bt in enumerate(bt_trades):
                if i in used:
                    continue
                if bt["side"].upper() != lt_side:
                    continue
                d = abs(bt["open_ts"] - lt_ms)
                if d <= MATCH_WINDOW_MS and d < best_d:
                    best_d, best_bt, best_i = d, bt, i

            live_time = str(lt["time"])[:16]
            if best_bt is None:
                line = f"  NO_MATCH  {live_time}  {lt_side}  entry={lt.get('entry_price')}  reason={lt.get('reason')}"
                print(line)
                append_log(line)
                # An unmatched demo trade is worth flagging
                found_mismatch = True
                continue

            used.add(best_i)
            issues = compare_trade(lt, best_bt)
            δ = int(best_d / 60_000)
            tag = "OK" if not issues else "MISMATCH"
            line = (
                f"  [{tag}]  live={live_time}  bt={fmt_ms(best_bt['open_ts'])}"
                f"  Δ{δ}min  {lt_side}"
                f"  live={lt.get('reason','')}  bt={best_bt.get('outcome','')}"
            )
            print(line)
            append_log(line)
            if issues:
                found_mismatch = True
                for field, lv, bv, note in issues:
                    detail = f"      ↳ {field}: live={lv}  bt={bv}  ({note})"
                    print(detail)
                    append_log(detail)

    result = "RESULT: MISMATCH(ES) FOUND — investigate before trusting backtest." if found_mismatch \
             else "RESULT: PARITY OK — demo and backtest agree."
    print(f"\n{result}")
    append_log(result)

    sys.exit(1 if found_mismatch else 0)

if __name__ == "__main__":
    main()
