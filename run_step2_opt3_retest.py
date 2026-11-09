#!/usr/bin/env python3
"""
run_step2_opt3_retest.py
────────────────────────
Step 2 retest on top of Option 3 (enable_grade_b_all=True, floor=ON).

Tests score floor variants: BASELINE (no floor change), 0.68, 0.70, 0.72.
IS: 2020-01-01 → 2023-12-31   (tune window — not used for pass/fail)
OOS: 2024-01-01 → 2026-11-01  (test window — PASS/FAIL lives here)
All 10 coins, 12 parallel workers.

Pass criteria (OOS only, vs Option 3 baseline):
  1. Beats baseline Sharpe on ≥ 6/10 coins
  2. Beats random median Sharpe on ≥ 6/10 coins
  3. No single year (2024 / 2025 / 2026) clearly worse than baseline (delta < -0.3)

Output: step2_opt3_report.txt
"""

import sys, os, time, json, statistics, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from backtester import start_backtest, get_run, init_backtest_tables
init_backtest_tables()

# ── Config ────────────────────────────────────────────────────────────────────
COINS      = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
              "ADAUSDT","AVAXUSDT","DOGEUSDT","LINKUSDT","NEARUSDT"]
MODE       = "MINI_ASYM"
STYLE      = "DAY_TRADE"
EXCHANGE   = "bybit"
START_EQ   = 1000.0
IS_FROM    = "2020-01-01"
IS_TO      = "2023-12-31"
OOS_FROM   = "2024-01-01"
OOS_TO     = "2026-11-01"
TIMEOUT    = 180  # minutes per run

# Random medians from completed 100-seed benchmark (OOS 2024-2026, Option 3 exits)
RANDOM_MEDIANS = {
    "BTCUSDT": -0.030, "ETHUSDT":  0.100, "SOLUSDT":  0.210,
    "BNBUSDT": -0.030, "XRPUSDT":  0.120, "ADAUSDT":  0.120,
    "NEARUSDT": 0.080, "DOGEUSDT": 0.160, "AVAXUSDT": 0.310,
    "LINKUSDT": 0.070,
}

# Variants: (label, score_floor)
# BASELINE = Option 3 with no additional score floor
VARIANTS = [
    ("BASELINE", 0.0),
    ("FLOOR068", 0.68),
    ("FLOOR070", 0.70),
    ("FLOOR072", 0.72),
]

_lock   = threading.Lock()
_done   = [0]
_total_runs = len(COINS) * 2 * len(VARIANTS)  # IS + OOS per coin per variant


def run_one(label: str, coin: str, date_from: str, date_to: str,
            score_floor: float) -> dict:
    email  = f"s2opt3-{label.lower()}-{coin.lower()}@asym-internal"
    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=date_from, date_to=date_to,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False, enable_regime_gate=False,
        enable_e2=False,
        enable_step2_score=False,   # we use score_floor directly
        enable_grade_b_all=True,    # Option 3
        enable_floor=True,
        score_floor=score_floor,
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
                print(f"  [{_done[0]}/{_total_runs}] ✓ {label} {coin} {date_from[:4]}-{date_to[:4]}", flush=True)
            return rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total_runs}] ✗ {label} {coin} ERROR: {rec.get('error','?')}", flush=True)
            return {}
    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total_runs}] ✗ {label} {coin} TIMEOUT", flush=True)
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
    print(f"Step 2 retest on top of Option 3")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"IS: {IS_FROM} – {IS_TO}  |  OOS: {OOS_FROM} – {OOS_TO}")
    print(f"Total runs: {_total_runs}  |  12 parallel workers\n")

    # Submit all tasks in parallel
    tasks = []
    for label, sf in VARIANTS:
        for coin in COINS:
            tasks.append((label, coin, IS_FROM, IS_TO, sf, "IS"))
            tasks.append((label, coin, OOS_FROM, OOS_TO, sf, "OOS"))

    results = {}   # (label, coin, window) -> result dict
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {
            pool.submit(run_one, label, coin, dfrom, dto, sf): (label, coin, win)
            for label, coin, dfrom, dto, sf, win in tasks
        }
        for fut in as_completed(futs):
            key = futs[fut]
            results[key] = fut.result()

    elapsed = time.time() - t0
    print(f"\nAll runs done in {elapsed/60:.1f} min — building report...\n")

    # ── Report ────────────────────────────────────────────────────────────────
    out_lines = []
    def w(line=""):
        print(line)
        out_lines.append(line)

    w("=" * 76)
    w("  STEP 2 RETEST ON TOP OF OPTION 3  —  Score Floor Variants")
    w(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    w("=" * 76)

    # Per-variant summary
    variant_summaries = {}
    for label, sf in VARIANTS:
        base_label = "BASELINE"
        w(f"\n{'─'*76}")
        w(f"  VARIANT: {label}  (score_floor={sf if sf > 0 else 'none (Option 3 defaults)'})")
        w(f"{'─'*76}")

        # OOS table
        w(f"\n  OOS 2024–{OOS_TO[:4]}:")
        w(f"  {'Coin':<12} {'Sharpe':>8} {'WR%':>6} {'Trades':>7} "
          f"{'2024':>7} {'2025':>7} {'2026':>7} "
          f"{'vsBASE':>8} {'vsRand':>7}")
        w("  " + "─" * 72)

        beats_base  = 0
        beats_rand  = 0
        year_deltas = {"2024": [], "2025": [], "2026": []}
        coin_details = []

        for coin in COINS:
            res_this  = results.get((label,       coin, "OOS"), {})
            res_base  = results.get(("BASELINE",  coin, "OOS"), {})
            sh, wr, tr, mo = extract(res_this)
            bs, _, _, _    = extract(res_base)
            ys = yearly_sharpe(mo)
            rand_med = RANDOM_MEDIANS.get(coin)

            vs_base = (sh - bs) if (sh is not None and bs is not None) else None
            vs_rand = (sh - rand_med) if (sh is not None and rand_med is not None) else None
            bb = vs_base is not None and vs_base > 0
            br = vs_rand is not None and vs_rand > 0
            if bb: beats_base += 1
            if br: beats_rand += 1

            for yr in ("2024", "2025", "2026"):
                yr_sh  = ys.get(yr)
                base_ys = yearly_sharpe(extract(res_base)[3])
                yr_base = base_ys.get(yr)
                if yr_sh is not None and yr_base is not None:
                    year_deltas[yr].append(yr_sh - yr_base)

            w(
                "  %-12s %8s %6s %7s %7s %7s %7s %8s %7s" % (
                    coin,
                    f"{sh:.3f}"  if sh is not None else "—",
                    f"{wr:.1f}"  if wr is not None else "—",
                    str(tr or "—"),
                    f"{ys.get('2024'):.2f}"  if ys.get("2024") is not None else "—",
                    f"{ys.get('2025'):.2f}"  if ys.get("2025") is not None else "—",
                    f"{ys.get('2026'):.2f}"  if ys.get("2026") is not None else "—",
                    f"{vs_base:+.3f}" if vs_base is not None else "—",
                    f"{vs_rand:+.3f}" if vs_rand is not None else "—",
                )
            )
            coin_details.append({"coin": coin, "sharpe": sh, "vs_base": vs_base,
                                  "vs_rand": vs_rand, "beats_base": bb, "beats_rand": br})

        # Pass/fail
        no_year_worse = all(
            all(d >= -0.3 for d in deltas)
            for deltas in year_deltas.values() if deltas
        )
        c1 = beats_base >= 6
        c2 = beats_rand >= 6
        c3 = no_year_worse
        passed = c1 and c2 and c3
        w(f"\n  Pass/Fail (OOS):")
        w(f"    1. Beats BASELINE on ≥6/10: {beats_base}/10  {'✓' if c1 else '✗'}")
        w(f"    2. Beats random median ≥6/10: {beats_rand}/10  {'✓' if c2 else '✗'}")
        w(f"    3. No year clearly worse:  {'✓' if c3 else '✗'}")
        w(f"  {'✅ PASS' if passed else '❌ FAIL'} — {label}")

        variant_summaries[label] = {
            "passed": passed, "beats_base": beats_base, "beats_rand": beats_rand,
            "no_year_worse": no_year_worse, "details": coin_details,
        }

    # ── Final recommendation ──────────────────────────────────────────────────
    w("\n" + "=" * 76)
    w("  SUMMARY")
    w("=" * 76)
    w(f"  {'Variant':<12} {'Pass':>5} {'vsBASE':>8} {'vsRand':>8}")
    w("  " + "─" * 36)
    for label, _ in VARIANTS:
        s = variant_summaries[label]
        w("  %-12s %5s %8s %8s" % (
            label,
            "PASS" if s["passed"] else "FAIL",
            f"{s['beats_base']}/10",
            f"{s['beats_rand']}/10",
        ))

    winners = [lbl for lbl, _ in VARIANTS if variant_summaries[lbl]["passed"]]
    w("")
    if winners:
        w(f"  Passing variants: {', '.join(winners)}")
        w(f"  → Recommend 100-seed validation on: {winners[0]}")
    else:
        w("  No variant passed. Leave score floor off. Move to Step 1 retest.")
    w("")
    w(f"  Total elapsed: {elapsed/60:.1f} min")
    w("=" * 76)

    # Save report + results
    report_path = os.path.join(os.path.dirname(__file__), "step2_opt3_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nReport saved → {report_path}")

    json_path = os.path.join(os.path.dirname(__file__), "step2_opt3_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "variants": {lbl: variant_summaries[lbl] for lbl, _ in VARIANTS},
            "raw": {str(k): v for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"Results saved → {json_path}")


if __name__ == "__main__":
    main()
