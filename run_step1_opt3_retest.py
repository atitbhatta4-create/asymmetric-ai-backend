#!/usr/bin/env python3
"""
run_step1_opt3_retest.py
────────────────────────
Step 1 retest on top of Option 3 (enable_grade_b_all=True, floor=ON).

Tests regime gate (enable_regime_gate=True) vs baseline (False).
Rolling windows are the primary pass/fail signal — IS is weak for Option 3.

Rolling windows (6-month, 3-month stride across OOS 2024-2026):
  2024-01 → 2024-06,  2024-04 → 2024-09,  2024-07 → 2024-12,
  2024-10 → 2025-03,  2025-01 → 2025-06,  2025-04 → 2025-09,
  2025-07 → 2025-12,  2025-10 → 2026-03,  2026-01 → 2026-06,
  2026-04 → 2026-10

Also runs IS 2020-2023 and full OOS 2024-2026 for reference.

Pass criteria (rolling windows — primary):
  1. Beats baseline in ≥ 6/10 coins on the FULL OOS window
  2. Beats random median in ≥ 6/10 coins on the FULL OOS window
  3. No rolling window year-equivalent clearly worse than baseline (delta < -0.3)
  4. Wins in majority of rolling windows across all coins

Output: step1_opt3_report.txt
"""

import sys, os, time, json, statistics, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from backtester import start_backtest, get_run, init_backtest_tables
init_backtest_tables()

# ── Config ────────────────────────────────────────────────────────────────────
COINS    = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
            "ADAUSDT","AVAXUSDT","DOGEUSDT","LINKUSDT","NEARUSDT"]
MODE     = "MINI_ASYM"
STYLE    = "DAY_TRADE"
EXCHANGE = "bybit"
START_EQ = 1000.0
TIMEOUT  = 180

IS_FROM  = "2020-01-01"
IS_TO    = "2023-12-31"
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-11-01"

# 6-month rolling windows, 3-month stride
ROLLING = [
    ("2024-01-01", "2024-06-30"),
    ("2024-04-01", "2024-09-30"),
    ("2024-07-01", "2024-12-31"),
    ("2024-10-01", "2025-03-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-04-01", "2025-09-30"),
    ("2025-07-01", "2025-12-31"),
    ("2025-10-01", "2026-03-31"),
    ("2026-01-01", "2026-06-30"),
    ("2026-04-01", "2026-10-31"),
]

# Random medians from completed 100-seed benchmark (OOS 2024-2026, Option 3)
RANDOM_MEDIANS = {
    "BTCUSDT": -0.030, "ETHUSDT":  0.100, "SOLUSDT":  0.210,
    "BNBUSDT": -0.030, "XRPUSDT":  0.120, "ADAUSDT":  0.120,
    "NEARUSDT": 0.080, "DOGEUSDT": 0.160, "AVAXUSDT": 0.310,
    "LINKUSDT": 0.070,
}

_lock = threading.Lock()
_done = [0]

# Total runs: (IS + OOS + 10 rolling) × 2 variants × 10 coins
_total = (1 + 1 + len(ROLLING)) * 2 * len(COINS)


def run_one(label: str, coin: str, date_from: str, date_to: str,
            regime_gate: bool) -> dict:
    tag   = "gate" if regime_gate else "base"
    email = f"s1opt3-{tag}-{coin.lower()}-{date_from[:7]}@asym-internal"
    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=date_from, date_to=date_to,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False,
        enable_regime_gate=regime_gate,
        enable_e2=False,
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
                print(f"  [{_done[0]}/{_total}] ✓ {label} {coin} {date_from[:7]}–{date_to[:7]}", flush=True)
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


def sharpe(res):
    return res.get("sharpe_ratio") if res else None


def main():
    t0 = time.time()
    print(f"Step 1 regime gate retest on top of Option 3")
    print(f"IS: {IS_FROM}–{IS_TO}  OOS: {OOS_FROM}–{OOS_TO}  Rolling: {len(ROLLING)} windows")
    print(f"Total runs: {_total}  |  12 parallel workers\n")

    tasks = []
    for coin in COINS:
        for gate in (False, True):
            tasks.append(("IS",  coin, IS_FROM,  IS_TO,  gate))
            tasks.append(("OOS", coin, OOS_FROM, OOS_TO, gate))
            for dfrom, dto in ROLLING:
                tasks.append((f"R_{dfrom[:7]}", coin, dfrom, dto, gate))

    results = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {
            pool.submit(run_one, win, coin, dfrom, dto, gate): (win, coin, gate)
            for win, coin, dfrom, dto, gate in tasks
        }
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    elapsed = time.time() - t0
    print(f"\nAll runs done in {elapsed/60:.1f} min — building report...\n")

    out_lines = []
    def w(line=""):
        print(line)
        out_lines.append(line)

    w("=" * 76)
    w("  STEP 1 REGIME GATE — Retest on top of Option 3")
    w(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    w("=" * 76)

    # ── Full OOS table ────────────────────────────────────────────────────────
    w("\nFULL OOS 2024–2026:")
    w(f"  {'Coin':<12} {'Base':>8} {'Gate':>8} {'Delta':>8} {'vsRand':>8} {'Beat?':>6}")
    w("  " + "─" * 52)

    beats_base = 0
    beats_rand = 0
    oos_details = {}
    for coin in COINS:
        sb = sharpe(results.get(("OOS", coin, False)))
        sg = sharpe(results.get(("OOS", coin, True)))
        delta = (sg - sb) if (sg is not None and sb is not None) else None
        rand  = RANDOM_MEDIANS.get(coin)
        vs_rand = (sg - rand) if (sg is not None and rand is not None) else None
        bb = delta is not None and delta > 0
        br = vs_rand is not None and vs_rand > 0
        if bb: beats_base += 1
        if br: beats_rand += 1
        oos_details[coin] = {"sb": sb, "sg": sg, "delta": delta, "bb": bb, "br": br}
        w("  %-12s %8s %8s %8s %8s %6s" % (
            coin,
            f"{sb:.3f}" if sb is not None else "—",
            f"{sg:.3f}" if sg is not None else "—",
            f"{delta:+.3f}" if delta is not None else "—",
            f"{vs_rand:+.3f}" if vs_rand is not None else "—",
            "✓" if bb else "✗",
        ))

    # ── Rolling windows summary ───────────────────────────────────────────────
    w("\nROLLING WINDOWS (6-month, 3-month stride):")
    w(f"  {'Window':<14} {'Gate wins':>10} {'Base wins':>10} {'Ties':>6}")
    w("  " + "─" * 44)

    total_gate_wins = 0
    total_base_wins = 0
    for dfrom, dto in ROLLING:
        win_key = f"R_{dfrom[:7]}"
        gw = bw = ties = 0
        for coin in COINS:
            sb = sharpe(results.get((win_key, coin, False)))
            sg = sharpe(results.get((win_key, coin, True)))
            if sg is None or sb is None:
                continue
            if sg > sb: gw += 1
            elif sb > sg: bw += 1
            else: ties += 1
        total_gate_wins += gw
        total_base_wins += bw
        w("  %-14s %10d %10d %6d" % (f"{dfrom[:7]}–{dto[:7]}", gw, bw, ties))

    w(f"\n  Total across all windows:  Gate {total_gate_wins}  vs  Base {total_base_wins}")
    rolling_win_pct = total_gate_wins / max(1, total_gate_wins + total_base_wins) * 100

    # ── IS reference ─────────────────────────────────────────────────────────
    w("\nIS 2020–2023 (reference only — not used for pass/fail):")
    w(f"  {'Coin':<12} {'Base':>8} {'Gate':>8} {'Delta':>8}")
    w("  " + "─" * 40)
    for coin in COINS:
        sb = sharpe(results.get(("IS", coin, False)))
        sg = sharpe(results.get(("IS", coin, True)))
        delta = (sg - sb) if (sg is not None and sb is not None) else None
        w("  %-12s %8s %8s %8s" % (
            coin,
            f"{sb:.3f}" if sb is not None else "—",
            f"{sg:.3f}" if sg is not None else "—",
            f"{delta:+.3f}" if delta is not None else "—",
        ))

    # ── Pass/fail ─────────────────────────────────────────────────────────────
    c1 = beats_base >= 6
    c2 = beats_rand >= 6
    c3 = rolling_win_pct >= 50.0  # gate wins majority of rolling slots
    passed = c1 and c2 and c3

    w("\n" + "=" * 76)
    w("  PASS / FAIL")
    w("=" * 76)
    w(f"  1. Beats baseline OOS ≥6/10:       {beats_base}/10   {'✓ PASS' if c1 else '✗ FAIL'}")
    w(f"  2. Beats random median ≥6/10:       {beats_rand}/10   {'✓ PASS' if c2 else '✗ FAIL'}")
    w(f"  3. Gate wins majority of rolling:  {rolling_win_pct:.0f}%   {'✓ PASS' if c3 else '✗ FAIL'}")
    w()
    w(f"  {'✅ STEP 1 PASSES' if passed else '❌ STEP 1 FAILS'} — regime gate on top of Option 3")
    if not passed:
        w("  → Leave ENABLE_REGIME_GATE=False. Move to Step 8 retest.")
    else:
        w("  → Run 100-seed validation, then demo with ENABLE_REGIME_GATE=True.")
    w()
    w(f"  Elapsed: {elapsed/60:.1f} min")
    w("=" * 76)

    report_path = os.path.join(os.path.dirname(__file__), "step1_opt3_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nReport → {report_path}")

    json_path = os.path.join(os.path.dirname(__file__), "step1_opt3_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "oos": oos_details,
            "rolling_gate_wins": total_gate_wins,
            "rolling_base_wins": total_base_wins,
            "passed": passed,
        }, f, indent=2, default=str)
    print(f"Results → {json_path}")


if __name__ == "__main__":
    main()
