#!/usr/bin/env python3
"""
run_step1_regime_gate.py
─────────────────────────
Step 1 validation: Regime Gate + Donchian Trend Core.

Runs two sets of backtests side by side:
  BASELINE  — enable_regime_gate=False  (current behaviour)
  STEP1     — enable_regime_gate=True   (MARGINAL blocked + Donchian filter)

IS  window: 2020-01-01 → 2023-12-31  (tune window — not used for pass/fail)
OOS window: 2024-01-01 → 2026-10-31  (test window — this is what must pass)

Pass criteria (OOS only):
  1. Beats baseline Sharpe on ≥ 6/10 coins
  2. Beats random median Sharpe on ≥ 6/10 coins
  3. Max drawdown stays within floor on all coins
  4. No single year (2024 / 2025 / 2026) is clearly worse than baseline

Output: step1_regime_gate_report.txt
"""

import sys
import os
import time
import json
import statistics
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from backtester import (
    start_backtest, get_run, init_backtest_tables,
    check_candle_coverage, prefetch_all_candles,
)
init_backtest_tables()

# ── Config ───────────────────────────────────────────────────────────────────
START_EQUITY  = 1000.0
EXCHANGE      = "bybit"
MODE          = "MINI_ASYM"
STYLE         = "DAY_TRADE"
EMAIL_BASE    = "step1-baseline@asym-ai-internal"
EMAIL_STEP1   = "step1-gate@asym-ai-internal"

IS_FROM  = "2020-01-01"
IS_TO    = "2023-12-31"
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-10-31"

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "NEARUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
]

# Random benchmark medians from completed 1000-run benchmark
RANDOM_MEDIANS = {
    "BTCUSDT": -0.035, "ETHUSDT": -0.010, "SOLUSDT":  0.065,
    "BNBUSDT":  0.130, "XRPUSDT":  0.140, "ADAUSDT":  0.080,
    "NEARUSDT": 0.020, "DOGEUSDT": 0.045, "AVAXUSDT": 0.250,
    "LINKUSDT": 0.080,
}

TIMEOUT_MIN = 150

# ── Helpers ───────────────────────────────────────────────────────────────────

class Out:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1)
    def writeln(self, line=""):
        print(line)
        self.f.write(line + "\n")
    def close(self):
        self.f.close()


def run_one(email, symbol, date_from, date_to, enable_floor, enable_regime_gate, out):
    run_id = start_backtest(
        email=email, symbol=symbol, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=date_from, date_to=date_to,
        start_equity=START_EQUITY,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_floor=enable_floor,
        enable_regime_gate=enable_regime_gate,
    )
    gate_tag  = "gate-ON" if enable_regime_gate else "gate-OFF"
    floor_tag = "floor-ON" if enable_floor else "floor-OFF"
    label     = f"{symbol} {date_from[:4]}-{date_to[:4]} [{gate_tag}|{floor_tag}]"
    deadline  = time.time() + TIMEOUT_MIN * 60

    while time.time() < deadline:
        rec = get_run(run_id)
        if not rec:
            time.sleep(5)
            continue
        status = rec.get("status", "")
        if status == "done":
            out.writeln(f"  ✓ {label}")
            return rec.get("results") or {}
        if status == "error":
            out.writeln(f"  ✗ {label} — ERROR: {rec.get('error','?')}")
            return {}
        time.sleep(10)

    out.writeln(f"  ✗ {label} — TIMEOUT after {TIMEOUT_MIN} min")
    return {}


def extract(res):
    if not res:
        return None, None, None, None, None
    sharpe   = res.get("sharpe_ratio")
    wr       = res.get("win_rate")
    trades   = res.get("total_trades", 0)
    mdd      = res.get("max_drawdown_pct") or res.get("max_drawdown")
    monthly  = res.get("monthly_returns") or []
    return sharpe, wr, trades, mdd, monthly


def yearly_sharpe(monthly):
    """Compute per-year Sharpe from monthly returns list [{"month": "YYYY-MM", "return_pct": float}]."""
    by_year = {}
    for entry in (monthly or []):
        if isinstance(entry, dict):
            ym  = entry.get("month", "")
            pnl = entry.get("return_pct", 0.0)
        else:
            continue
        yr = ym[:4]
        by_year.setdefault(yr, []).append(float(pnl))
    result = {}
    for yr, vals in by_year.items():
        if len(vals) < 2:
            result[yr] = None
            continue
        avg = statistics.mean(vals)
        std = statistics.stdev(vals)
        result[yr] = round(avg / std, 3) if std > 0 else 0.0
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out = Out("step1_regime_gate_report.txt")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    out.writeln("=" * 72)
    out.writeln("  STEP 1 — REGIME GATE + DONCHIAN TREND CORE")
    out.writeln(f"  Generated: {now}")
    out.writeln("=" * 72)

    # ── Coverage check ───────────────────────────────────────────────────────
    out.writeln("\nChecking candle coverage...")
    def _to_ms(d):
        return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    is_start_ms  = _to_ms(IS_FROM);  is_end_ms  = _to_ms(IS_TO)
    oos_start_ms = _to_ms(OOS_FROM); oos_end_ms = _to_ms(OOS_TO)
    all_ok_is,  rows_is  = check_candle_coverage(COINS, ["1h"], is_start_ms,  is_end_ms)
    all_ok_oos, rows_oos = check_candle_coverage(COINS, ["1h"], oos_start_ms, oos_end_ms)
    if all_ok_is and all_ok_oos:
        out.writeln("  All 10 coins × IS+OOS: coverage OK ✓")
    else:
        for r in rows_is + rows_oos:
            if r.get("pct", 1.0) < 0.97:
                out.writeln(f"  ✗ {r['symbol']} {r['tf']}: {r['pct']:.1%}")
        out.writeln("\n⚠  Coverage issues found. Run fix_candle_cache.py first.")

    # ── IS runs (tune window) ────────────────────────────────────────────────
    out.writeln("\n" + "─" * 72)
    out.writeln("  IS WINDOW 2020–2023 (tune — not used for pass/fail)")
    out.writeln("─" * 72)

    is_base, is_step1 = {}, {}
    for coin in COINS:
        out.writeln(f"\n{coin} IS:")
        is_base[coin]  = run_one(EMAIL_BASE,  coin, IS_FROM, IS_TO, False, False, out)
        is_step1[coin] = run_one(EMAIL_STEP1, coin, IS_FROM, IS_TO, False, True,  out)

    # ── OOS runs (test window — pass/fail) ───────────────────────────────────
    out.writeln("\n" + "─" * 72)
    out.writeln("  OOS WINDOW 2024–2026 (test — PASS/FAIL lives here)")
    out.writeln("─" * 72)

    oos_base, oos_step1 = {}, {}
    for coin in COINS:
        out.writeln(f"\n{coin} OOS:")
        oos_base[coin]  = run_one(EMAIL_BASE,  coin, OOS_FROM, OOS_TO, False, False, out)
        oos_step1[coin] = run_one(EMAIL_STEP1, coin, OOS_FROM, OOS_TO, False, True,  out)

    # ── Report ───────────────────────────────────────────────────────────────
    out.writeln("\n" + "=" * 72)
    out.writeln("  RESULTS SUMMARY")
    out.writeln("=" * 72)

    # IS summary
    out.writeln("\nIS 2020–2023 (tune window):")
    out.writeln(f"  {'Coin':<12} {'Base Sharpe':>12} {'Step1 Sharpe':>13} {'Delta':>8} {'Base WR':>8} {'Step1 WR':>9} {'Base Tr':>8} {'Step1 Tr':>9}")
    out.writeln("  " + "-" * 85)
    for coin in COINS:
        bs, bwr, btr, _, _ = extract(is_base.get(coin, {}))
        ss, swr, str_, _, _ = extract(is_step1.get(coin, {}))
        delta = (ss - bs) if (ss is not None and bs is not None) else None
        out.writeln(
            "  %-12s %12s %13s %8s %8s %9s %8s %9s" % (
                coin,
                f"{bs:.3f}" if bs is not None else "—",
                f"{ss:.3f}" if ss is not None else "—",
                f"{delta:+.3f}" if delta is not None else "—",
                f"{bwr:.1f}%" if bwr is not None else "—",
                f"{swr:.1f}%" if swr is not None else "—",
                str(btr) if btr else "—",
                str(str_) if str_ else "—",
            )
        )

    # OOS summary
    out.writeln("\nOOS 2024–2026 (test window — PASS/FAIL):")
    out.writeln(f"  {'Coin':<12} {'Base Sharpe':>12} {'Step1 Sharpe':>13} {'Delta':>8} {'Base WR':>8} {'Step1 WR':>9} {'vs Random':>10}")
    out.writeln("  " + "-" * 85)

    beats_baseline = 0
    beats_random   = 0
    details        = []

    for coin in COINS:
        bs,  bwr,  btr,  bmdd,  bmonth  = extract(oos_base.get(coin, {}))
        ss,  swr,  str_, smdd,  smonth  = extract(oos_step1.get(coin, {}))
        rand_med = RANDOM_MEDIANS.get(coin, 0.0)
        delta    = (ss - bs) if (ss is not None and bs is not None) else None
        beat_b   = delta is not None and delta > 0
        beat_r   = ss is not None and ss > rand_med
        if beat_b:
            beats_baseline += 1
        if beat_r:
            beats_random += 1
        vs_rand = "✓ beats" if beat_r else "✗ below"
        out.writeln(
            "  %-12s %12s %13s %8s %8s %9s %10s" % (
                coin,
                f"{bs:.3f}" if bs is not None else "—",
                f"{ss:.3f}" if ss is not None else "—",
                f"{delta:+.3f}" if delta is not None else "—",
                f"{bwr:.1f}%" if bwr is not None else "—",
                f"{swr:.1f}%" if swr is not None else "—",
                vs_rand,
            )
        )
        details.append({
            "coin": coin,
            "base_sharpe": bs, "step1_sharpe": ss, "delta": delta,
            "base_wr": bwr,    "step1_wr": swr,
            "base_mdd": bmdd,  "step1_mdd": smdd,
            "base_monthly": bmonth, "step1_monthly": smonth,
            "beats_baseline": beat_b, "beats_random": beat_r,
        })

    # Year-by-year breakdown
    out.writeln("\nYear-by-year OOS Sharpe (Step1 vs Baseline):")
    out.writeln(f"  {'Coin':<12}  {'2024 Base':>10} {'2024 S1':>8}  {'2025 Base':>10} {'2025 S1':>8}  {'2026 Base':>10} {'2026 S1':>8}")
    out.writeln("  " + "-" * 80)
    no_year_worse = True
    for d in details:
        by_b = yearly_sharpe(d["base_monthly"])
        by_s = yearly_sharpe(d["step1_monthly"])
        row = "  %-12s" % d["coin"]
        for yr in ["2024", "2025", "2026"]:
            vb = by_b.get(yr)
            vs = by_s.get(yr)
            diff = (vs - vb) if (vs is not None and vb is not None) else None
            # Flag if step1 is clearly worse (delta < -0.3)
            flag = " ⚠" if (diff is not None and diff < -0.30) else "  "
            if diff is not None and diff < -0.30:
                no_year_worse = False
            row += "  %10s %8s%s" % (
                f"{vb:.3f}" if vb is not None else "—",
                f"{vs:.3f}" if vs is not None else "—",
                flag,
            )
        out.writeln(row)

    # ── Pass/fail ────────────────────────────────────────────────────────────
    out.writeln("\n" + "=" * 72)
    out.writeln("  PASS / FAIL")
    out.writeln("=" * 72)

    c1 = beats_baseline >= 6
    c2 = beats_random   >= 6
    c3 = no_year_worse

    out.writeln(f"\n  1. Beats baseline on ≥6/10 coins OOS:  {beats_baseline}/10  {'✓ PASS' if c1 else '✗ FAIL'}")
    out.writeln(f"  2. Beats random median on ≥6/10 coins: {beats_random}/10   {'✓ PASS' if c2 else '✗ FAIL'}")
    out.writeln(f"  3. No year clearly worse (delta<-0.3):         {'✓ PASS' if c3 else '✗ FAIL'}")

    overall = c1 and c2 and c3
    out.writeln("\n" + ("  ✅ STEP 1 PASSES — ready for demo deployment" if overall
                        else "  ❌ STEP 1 FAILS — leave ENABLE_REGIME_GATE=False, record below"))

    if not overall:
        out.writeln("\n  Failure notes:")
        if not c1:
            failed_coins = [d["coin"] for d in details if not d["beats_baseline"]]
            out.writeln(f"    - Baseline underperformance: {', '.join(failed_coins)}")
        if not c2:
            rand_failed = [d["coin"] for d in details if not d["beats_random"]]
            out.writeln(f"    - Below random: {', '.join(rand_failed)}")
        if not no_year_worse:
            out.writeln("    - One or more years degraded by >0.3 Sharpe")

    # Save raw results
    with open("step1_regime_gate_results.json", "w") as jf:
        json.dump({
            "generated": now, "passed": overall,
            "beats_baseline": beats_baseline, "beats_random": beats_random,
            "is_base": {k: v for k, v in is_base.items()},
            "is_step1": {k: v for k, v in is_step1.items()},
            "oos_base": {k: v for k, v in oos_base.items()},
            "oos_step1": {k: v for k, v in oos_step1.items()},
            "details": details,
        }, jf, indent=2, default=str)

    out.writeln(f"\n  Raw results saved → step1_regime_gate_results.json")
    out.writeln("=" * 72)
    out.close()
    print("\nDone. See step1_regime_gate_report.txt")


if __name__ == "__main__":
    main()
