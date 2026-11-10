#!/usr/bin/env python3
"""
run_e2_validation.py
────────────────────
Full E2 time exit validation on top of Option 3 (enable_grade_b_all=True, floor=ON).

Parts:
  A. 100-seed Sharpe distribution — does E2 improvement hold across random entries?
     100 seeds × 10 coins OOS, compare to E2=False distribution
  B. Plateau test — hold times ×0.8 and ×1.2, confirms E2 isn't threshold-sensitive
  C. Rolling windows — 10 × 6-month windows, E2 vs baseline
  D. E2_TIME trade analysis — avg PnL, win rate, % that would trail

OOS: 2024-01-01 → 2026-11-01   IS: 2020-01-01 → 2023-12-31
Output: e2_validation_report.txt
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
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-11-01"
IS_FROM  = "2020-01-01"
IS_TO    = "2023-12-31"
TIMEOUT  = 300

N_SEEDS  = 100
SEEDS    = list(range(1, N_SEEDS + 1))

ROLLING = [
    ("2024-01-01","2024-06-30"), ("2024-04-01","2024-09-30"),
    ("2024-07-01","2024-12-31"), ("2024-10-01","2025-03-31"),
    ("2025-01-01","2025-06-30"), ("2025-04-01","2025-09-30"),
    ("2025-07-01","2025-12-31"), ("2025-10-01","2026-03-31"),
    ("2026-01-01","2026-06-30"), ("2026-04-01","2026-10-31"),
]

# E2=True seed=42 Sharpes from step8_opt3 OOS results
E2_SEED42_SHARPES = {
    "BTCUSDT":  0.250, "ETHUSDT":  0.600, "SOLUSDT":  0.480,
    "BNBUSDT": -0.700, "XRPUSDT":  1.010, "ADAUSDT":  0.960,
    "AVAXUSDT": 0.970, "DOGEUSDT": 0.240, "LINKUSDT": 0.570,
    "NEARUSDT": 0.750,
}

# E2=False (Option3 baseline) seed=42 Sharpes from opt3_full_results
E2_FALSE_SHARPES = {
    "BTCUSDT":  0.140, "ETHUSDT":  0.670, "SOLUSDT":  0.800,
    "BNBUSDT": -0.940, "XRPUSDT":  0.850, "ADAUSDT":  1.040,
    "AVAXUSDT": 0.870, "DOGEUSDT": 0.250, "LINKUSDT": 0.420,
    "NEARUSDT": 0.600,
}

_lock    = threading.Lock()
_done    = [0]
_total_a = N_SEEDS * len(COINS)   # Part A: 100-seed E2 distribution
_total_b = 2 * len(COINS)         # Part B: plateau (×0.8, ×1.2)
_total_c = 2 * len(ROLLING) * len(COINS)  # Part C: rolling baseline+E2
_total   = _total_a + _total_b + _total_c

print(f"E2 full validation — {_total} total runs  ({_total_a} seeds + {_total_b} plateau + {_total_c} rolling)")
print(f"OOS: {OOS_FROM}–{OOS_TO}  |  12 workers\n", flush=True)


def _run(tag, coin, dfrom, dto, e2=True, seed=None, e2_mult=1.0):
    seed_str = f"-s{seed}" if seed else ""
    email = f"e2val-{tag}-{coin.lower()}{seed_str}@asym-internal"
    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=dfrom, date_to=dto,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False, enable_regime_gate=False,
        enable_e2=e2,
        enable_step2_score=False,
        enable_grade_b_all=True,
        enable_floor=True,
        random_seed=seed,
        e2_hold_mult=e2_mult,
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
                if _done[0] % 50 == 0 or _done[0] <= 5:
                    print(f"  [{_done[0]}/{_total}] ✓ {tag} {coin}", flush=True)
            return rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
            return {}
    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total}] ✗ {tag} {coin} TIMEOUT", flush=True)
    return {}


def sharpe(res): return res.get("sharpe_ratio") if res else None
def win_rate(res): return res.get("win_rate") if res else None
def n_trades(res): return res.get("total_trades", 0) if res else 0


def main():
    t0 = time.time()

    # Build all tasks
    tasks_a, tasks_b, tasks_c = [], [], []

    # Part A: 100-seed E2 distribution
    for coin in COINS:
        for seed in SEEDS:
            tasks_a.append(("seedE2", coin, OOS_FROM, OOS_TO, True,  seed, 1.0))

    # Part B: plateau — ×0.8 and ×1.2
    for coin in COINS:
        tasks_b.append(("plt08", coin, OOS_FROM, OOS_TO, True, None, 0.8))
        tasks_b.append(("plt12", coin, OOS_FROM, OOS_TO, True, None, 1.2))

    # Part C: rolling windows — baseline and E2
    for dfrom, dto in ROLLING:
        for coin in COINS:
            tasks_c.append((f"rbase_{dfrom[:7]}", coin, dfrom, dto, False, None, 1.0))
            tasks_c.append((f"re2_{dfrom[:7]}",   coin, dfrom, dto, True,  None, 1.0))

    all_tasks = tasks_a + tasks_b + tasks_c

    results_a = {}  # (coin, seed) -> result
    results_b = {}  # (tag, coin) -> result
    results_c = {}  # (window, coin, e2) -> result

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {}
        for tag, coin, dfrom, dto, e2, seed, mult in all_tasks:
            key = (tag, coin, seed)
            futs[pool.submit(_run, tag, coin, dfrom, dto, e2, seed, mult)] = (tag, coin, seed, dfrom)
        for fut in as_completed(futs):
            tag, coin, seed, dfrom = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  ✗ {tag} {coin} crashed: {e}", flush=True)
                res = {}
            if tag == "seedE2":
                results_a[(coin, seed)] = res
            elif tag in ("plt08", "plt12"):
                results_b[(tag, coin)] = res
            else:
                e2_flag = tag.startswith("re2_")
                win = dfrom[:7]
                results_c[(win, coin, e2_flag)] = res

    elapsed = time.time() - t0
    print(f"\nAll runs done in {elapsed/60:.1f} min\n", flush=True)

    # ── Report ────────────────────────────────────────────────────────────────
    lines = []
    def w(line=""):
        print(line)
        lines.append(line)

    w("=" * 76)
    w("  E2 TIME EXIT — Full Validation on top of Option 3")
    w(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    w(f"  Variants tried total (across all E2 tests): ~{_total} runs")
    w("=" * 76)

    # ── Part A: 100-seed distribution ─────────────────────────────────────────
    w("\n── PART A: 100-Seed Sharpe Distribution (E2=True vs E2=False) ──")
    w(f"  {'Coin':<12} {'E2 Med':>8} {'E2 P25':>8} {'E2 P75':>8} "
      f"{'E2 s42':>8} {'Base s42':>9} {'Pctile':>8} {'N':>4}")
    w("  " + "─" * 68)

    a_pass_coins = 0
    for coin in COINS:
        sharpes = [sharpe(results_a.get((coin, s))) for s in SEEDS]
        sharpes = [x for x in sharpes if x is not None]
        if not sharpes:
            w(f"  {coin:<12} {'—':>8}")
            continue
        sharpes_s = sorted(sharpes)
        n = len(sharpes_s)
        med = statistics.median(sharpes_s)
        p25 = sharpes_s[max(0, int(n * 0.25))]
        p75 = sharpes_s[min(n-1, int(n * 0.75))]
        s42 = E2_SEED42_SHARPES.get(coin)
        base = E2_FALSE_SHARPES.get(coin)
        pct  = round(sum(1 for x in sharpes_s if x < s42) / n * 100, 1) if s42 is not None else None
        beat_base = s42 is not None and base is not None and s42 > base
        if med > 0: a_pass_coins += 1
        w("  %-12s %8.3f %8.3f %8.3f %8.3f %9.3f %8.1f%% %4d" % (
            coin, med, p25, p75,
            s42 if s42 is not None else 0,
            base if base is not None else 0,
            pct if pct is not None else 0, n,
        ))

    w(f"\n  E2 median Sharpe > 0 on {a_pass_coins}/10 coins")

    # ── Part B: Plateau test ───────────────────────────────────────────────────
    w("\n── PART B: Plateau Test (hold times ×0.8 and ×1.2) ──")
    w(f"  Thresholds  ×0.8: 12/17/24c   ×1.0: 15/22/30c (current)   ×1.2: 18/26/36c")
    w(f"  {'Coin':<12} {'×0.8 Sharpe':>12} {'×1.0 (s42)':>11} {'×1.2 Sharpe':>12} {'Stable?':>8}")
    w("  " + "─" * 56)

    plateau_ok = 0
    for coin in COINS:
        s08 = sharpe(results_b.get(("plt08", coin)))
        s12 = sharpe(results_b.get(("plt12", coin)))
        s10 = E2_SEED42_SHARPES.get(coin)
        vals = [x for x in (s08, s10, s12) if x is not None]
        # Stable if max deviation from ×1.0 is < 0.3
        stable = all(abs(v - s10) < 0.30 for v in [s08, s12] if v is not None and s10 is not None)
        if stable: plateau_ok += 1
        w("  %-12s %12s %11s %12s %8s" % (
            coin,
            f"{s08:.3f}" if s08 is not None else "—",
            f"{s10:.3f}" if s10 is not None else "—",
            f"{s12:.3f}" if s12 is not None else "—",
            "✓ stable" if stable else "⚠ varies",
        ))
    w(f"\n  Stable across thresholds: {plateau_ok}/10 coins")

    # ── Part C: Rolling windows ────────────────────────────────────────────────
    w("\n── PART C: Rolling Windows (6-month, E2 vs Baseline) ──")
    w(f"  {'Window':<14} {'E2 wins':>8} {'Base wins':>10} {'Ties':>6}")
    w("  " + "─" * 42)

    total_e2_wins = total_base_wins = 0
    for dfrom, dto in ROLLING:
        win = dfrom[:7]
        gw = bw = ties = 0
        for coin in COINS:
            se = sharpe(results_c.get((win, coin, True)))
            sb = sharpe(results_c.get((win, coin, False)))
            if se is None or sb is None: continue
            if se > sb: gw += 1
            elif sb > se: bw += 1
            else: ties += 1
        total_e2_wins += gw
        total_base_wins += bw
        w("  %-14s %8d %10d %6d" % (f"{dfrom[:7]}–{dto[:7]}", gw, bw, ties))

    e2_roll_pct = total_e2_wins / max(1, total_e2_wins + total_base_wins) * 100
    w(f"\n  Total: E2 {total_e2_wins}  vs  Base {total_base_wins}  ({e2_roll_pct:.0f}% E2 wins)")

    # ── Summary & Pass/Fail ────────────────────────────────────────────────────
    c1 = a_pass_coins >= 6          # E2 median > 0 on majority
    c2 = plateau_ok >= 6            # stable across ±20% thresholds
    c3 = e2_roll_pct >= 50.0        # E2 wins majority of rolling windows
    passed = c1 and c2 and c3

    w("\n" + "=" * 76)
    w("  PASS / FAIL")
    w("=" * 76)
    w(f"  1. E2 median Sharpe > 0 on ≥6/10 coins:    {a_pass_coins}/10  {'✓' if c1 else '✗'}")
    w(f"  2. Plateau stable (±20%) on ≥6/10 coins:   {plateau_ok}/10  {'✓' if c2 else '✗'}")
    w(f"  3. E2 wins majority of rolling windows:    {e2_roll_pct:.0f}%  {'✓' if c3 else '✗'}")
    w()
    w(f"  {'✅ E2 VALIDATED' if passed else '❌ E2 FAILS VALIDATION'}")
    if passed:
        w("  → Deploy to demo with ENABLE_E2=True. Monitor 7 days before real.")
    else:
        w("  → Keep ENABLE_E2=False. Review failing criteria before retry.")
    w()

    # Total variants tried across all filter tests
    w("── Total variants tried (all filter tests to date) ──")
    w("  Step 2 (score floors on Opt3):     4 variants × 10 coins × 2 windows = 80 runs")
    w("  Step 1 (regime gate on Opt3):      2 variants × 10 coins × 12 windows = 240 runs")
    w("  Step 8 (E2 on Opt3):               2 variants × 10 coins × 2 windows  = 40 runs")
    w(f"  E2 validation (this run):          {_total} runs")
    w(f"  100-seed Opt3 benchmark:           1000 runs")
    w(f"  Total: ~{80+240+40+_total+1000} runs")
    w()
    w(f"  Elapsed: {elapsed/60:.1f} min")
    w("=" * 76)

    with open("e2_validation_report.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport → e2_validation_report.txt")

    with open("e2_validation_results.json", "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "part_a_pass_coins": a_pass_coins,
            "plateau_ok": plateau_ok,
            "rolling_e2_pct": round(e2_roll_pct, 1),
        }, f, indent=2)
    print(f"Results → e2_validation_results.json")


if __name__ == "__main__":
    main()
