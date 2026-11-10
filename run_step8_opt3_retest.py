#!/usr/bin/env python3
"""
run_step8_opt3_retest.py
────────────────────────
Step 8 retest on top of Option 3 (enable_grade_b_all=True, floor=ON).

Tests E2 time exit (enable_e2=True) vs baseline (False).
E2: score-scaled max hold — 15/22/30 candles depending on signal score.
Forces close on stalled trades that haven't hit SL/TP.

IS: 2020-01-01 → 2023-12-31   (tune — not used for pass/fail)
OOS: 2024-01-01 → 2026-11-01  (test — PASS/FAIL lives here)
All 10 coins, 12 parallel workers.

Pass criteria (OOS only):
  1. Beats baseline Sharpe on ≥ 6/10 coins
  2. Beats random median Sharpe on ≥ 6/10 coins
  3. No single year (2024 / 2025 / 2026) clearly worse than baseline (delta < -0.3)

Output: step8_opt3_report.txt
"""

import sys, os, time, json, statistics, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from backtester import start_backtest, get_run, init_backtest_tables
init_backtest_tables()

COINS    = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
            "ADAUSDT","AVAXUSDT","DOGEUSDT","LINKUSDT","NEARUSDT"]
MODE     = "MINI_ASYM"
STYLE    = "DAY_TRADE"
EXCHANGE = "bybit"
START_EQ = 1000.0
TIMEOUT  = 300

IS_FROM  = "2020-01-01"
IS_TO    = "2023-12-31"
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-11-01"

RANDOM_MEDIANS = {
    "BTCUSDT": -0.030, "ETHUSDT":  0.100, "SOLUSDT":  0.210,
    "BNBUSDT": -0.030, "XRPUSDT":  0.120, "ADAUSDT":  0.120,
    "NEARUSDT": 0.080, "DOGEUSDT": 0.160, "AVAXUSDT": 0.310,
    "LINKUSDT": 0.070,
}

_lock  = threading.Lock()
_done  = [0]
_total = 2 * 2 * len(COINS)   # IS + OOS × baseline + e2 × 10 coins


def run_one(label: str, coin: str, date_from: str, date_to: str,
            enable_e2: bool) -> dict:
    email  = f"s8opt3-{label.lower()}-{coin.lower()}@asym-internal"
    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=date_from, date_to=date_to,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False, enable_regime_gate=False,
        enable_e2=enable_e2,
        enable_step2_score=False,
        enable_grade_b_all=True,   # Option 3
        enable_floor=True,
    )
    deadline = time.time() + TIMEOUT * 60
    while time.time() < deadline:
        time.sleep(5)
        rec = get_run(run_id)
        if not rec:
            continue
        st = rec.get("status", "")
        if st == "done":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total}] ✓ {label} {coin} {date_from[:4]}-{date_to[:4]}", flush=True)
            return rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total}] ✗ {label} {coin} ERROR", flush=True)
            return {}
    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total}] ✗ {label} {coin} TIMEOUT", flush=True)
    return {}


def extract(res):
    if not res:
        return None, None, None, None
    return (
        res.get("sharpe_ratio"),
        res.get("win_rate"),
        res.get("total_trades", 0),
        res.get("monthly_returns") or [],
    )


def yearly_sharpe(monthly):
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


def main():
    t0 = time.time()
    print(f"Step 8 E2 time exit retest on top of Option 3")
    print(f"IS: {IS_FROM}–{IS_TO}  |  OOS: {OOS_FROM}–{OOS_TO}")
    print(f"Total runs: {_total}  |  12 parallel workers\n")

    tasks = []
    for coin in COINS:
        tasks.append(("BASE", coin, IS_FROM,  IS_TO,  False))
        tasks.append(("E2",   coin, IS_FROM,  IS_TO,  True))
        tasks.append(("BASE", coin, OOS_FROM, OOS_TO, False))
        tasks.append(("E2",   coin, OOS_FROM, OOS_TO, True))

    results = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = {
            pool.submit(run_one, lbl, coin, dfrom, dto, e2): (lbl, coin, dfrom[:4])
            for lbl, coin, dfrom, dto, e2 in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                print(f"  ✗ {key} crashed: {e}", flush=True)
                results[key] = {}

    elapsed = time.time() - t0
    print(f"\nAll runs done in {elapsed/60:.1f} min\n")

    out_lines = []
    def w(line=""):
        print(line)
        out_lines.append(line)

    w("=" * 76)
    w("  STEP 8 — E2 TIME EXIT  |  Retest on top of Option 3")
    w(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    w("=" * 76)

    # OOS table
    w("\nOOS 2024–2026:")
    w(f"  {'Coin':<12} {'Base':>8} {'E2':>8} {'Delta':>8} "
      f"{'2024':>7} {'2025':>7} {'2026':>7} {'vsRand':>8} {'Beat?':>6}")
    w("  " + "─" * 72)

    beats_base = 0
    beats_rand = 0
    year_ok    = True
    details    = {}

    for coin in COINS:
        rb = results.get(("BASE", coin, OOS_FROM[:4]), {})
        re = results.get(("E2",   coin, OOS_FROM[:4]), {})
        sb, _, _, _  = extract(rb)
        se, wr, tr, mo = extract(re)
        ys     = yearly_sharpe(mo)
        yb     = yearly_sharpe(extract(rb)[3])
        delta  = (se - sb) if (se is not None and sb is not None) else None
        rand   = RANDOM_MEDIANS.get(coin)
        vs_r   = (se - rand) if (se is not None and rand is not None) else None
        bb = delta is not None and delta > 0
        br = vs_r  is not None and vs_r  > 0
        if bb: beats_base += 1
        if br: beats_rand += 1

        for yr in ("2024", "2025", "2026"):
            d = None
            if ys.get(yr) is not None and yb.get(yr) is not None:
                d = ys[yr] - yb[yr]
            if d is not None and d < -0.3:
                year_ok = False

        w("  %-12s %8s %8s %8s %7s %7s %7s %8s %6s" % (
            coin,
            f"{sb:.3f}" if sb is not None else "—",
            f"{se:.3f}" if se is not None else "—",
            f"{delta:+.3f}" if delta is not None else "—",
            f"{ys.get('2024'):.2f}" if ys.get("2024") is not None else "—",
            f"{ys.get('2025'):.2f}" if ys.get("2025") is not None else "—",
            f"{ys.get('2026'):.2f}" if ys.get("2026") is not None else "—",
            f"{vs_r:+.3f}" if vs_r is not None else "—",
            "✓" if bb else "✗",
        ))
        details[coin] = {"sb": sb, "se": se, "delta": delta, "bb": bb, "br": br}

    # IS reference
    w("\nIS 2020–2023 (reference):")
    w(f"  {'Coin':<12} {'Base':>8} {'E2':>8} {'Delta':>8}")
    w("  " + "─" * 40)
    for coin in COINS:
        rb = results.get(("BASE", coin, IS_FROM[:4]), {})
        re = results.get(("E2",   coin, IS_FROM[:4]), {})
        sb = extract(rb)[0]
        se = extract(re)[0]
        d  = (se - sb) if (se is not None and sb is not None) else None
        w("  %-12s %8s %8s %8s" % (
            coin,
            f"{sb:.3f}" if sb is not None else "—",
            f"{se:.3f}" if se is not None else "—",
            f"{d:+.3f}"  if d  is not None else "—",
        ))

    c1 = beats_base >= 6
    c2 = beats_rand >= 6
    c3 = year_ok
    passed = c1 and c2 and c3

    w("\n" + "=" * 76)
    w("  PASS / FAIL")
    w("=" * 76)
    w(f"  1. Beats baseline OOS ≥6/10:    {beats_base}/10   {'✓ PASS' if c1 else '✗ FAIL'}")
    w(f"  2. Beats random median ≥6/10:   {beats_rand}/10   {'✓ PASS' if c2 else '✗ FAIL'}")
    w(f"  3. No year clearly worse:        {'✓ PASS' if c3 else '✗ FAIL'}")
    w()
    w(f"  {'✅ STEP 8 PASSES' if passed else '❌ STEP 8 FAILS'} — E2 time exit on top of Option 3")
    if passed:
        w("  → Run 100-seed validation, then deploy to demo with ENABLE_E2=True.")
    else:
        w("  → Leave ENABLE_E2=False. Filter tests complete — move to grid backtest.")
    w()
    w(f"  Elapsed: {elapsed/60:.1f} min")
    w("=" * 76)

    with open("step8_opt3_report.txt", "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nReport → step8_opt3_report.txt")

    with open("step8_opt3_results.json", "w") as f:
        json.dump({"generated": datetime.now(timezone.utc).isoformat(),
                   "passed": passed, "beats_base": beats_base,
                   "beats_rand": beats_rand, "details": details},
                  f, indent=2, default=str)
    print(f"Results → step8_opt3_results.json")


if __name__ == "__main__":
    main()
