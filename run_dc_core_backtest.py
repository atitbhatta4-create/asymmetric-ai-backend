#!/usr/bin/env python3
"""
run_dc_core_backtest.py
───────────────────────
Backtest the daily/4h Donchian core direction gate on top of Option 3.

Spec:
  - Uses 4h klines (higher_slice already fetched by backtester for DAY_TRADE)
  - 20-period Donchian channel on 4h
  - Block LONG  when 4h DC position < 0.35 (HTF downtrend)
  - Block SHORT when 4h DC position > 0.65 (HTF uptrend)
  - Mid-channel (0.35–0.65): both directions allowed
  - Not enough 4h data → allow (never block on missing data)

Variants (2 max):
  BASELINE — Option 3, DC core OFF  (enable_dc_core=False)
  DC_CORE  — Option 3, DC core ON   (enable_dc_core=True)

Same IS/OOS/rolling/100-seed protocol as all other steps.
Output: dc_core_backtest_report.txt  dc_core_backtest_results.json
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
IS_FROM  = "2020-01-01"
IS_TO    = "2023-12-31"
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-11-01"
TIMEOUT  = 300
N_SEEDS  = 100

ROLLING = [
    ("2024-01-01","2024-06-30"), ("2024-04-01","2024-09-30"),
    ("2024-07-01","2024-12-31"), ("2024-10-01","2025-03-31"),
    ("2025-01-01","2025-06-30"), ("2025-04-01","2025-09-30"),
    ("2025-07-01","2025-12-31"), ("2025-10-01","2026-03-31"),
    ("2026-01-01","2026-06-30"), ("2026-04-01","2026-10-31"),
]

_lock  = threading.Lock()
_done  = [0]

_total_is      = 2 * len(COINS)
_total_oos     = 2 * len(COINS)
_total_rolling = 2 * len(ROLLING) * len(COINS)
_total_seeds   = N_SEEDS * len(COINS)
_total         = _total_is + _total_oos + _total_rolling + _total_seeds

print(f"DC core backtest — {_total} total runs")
print(f"  IS {IS_FROM}→{IS_TO} · OOS {OOS_FROM}→{OOS_TO}")
print(f"  Variants: BASELINE vs DC_CORE · 10 coins · {len(ROLLING)} rolling windows · {N_SEEDS} seeds\n",
      flush=True)


def _run(tag, coin, dfrom, dto, dc_core=False, seed=None):
    email = f"dcbt-{tag}-{coin.lower()}{f'-s{seed}' if seed else ''}@asym-internal"
    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=dfrom, date_to=dto,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False, enable_regime_gate=False,
        enable_e2=False, enable_step2_score=False,
        enable_grade_b_all=True, enable_floor=True,
        random_seed=seed,
        enable_dc_core=dc_core,
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
                if _done[0] % 50 == 0 or _done[0] <= 4:
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


def main():
    t0 = time.time()

    tasks = []
    for coin in COINS:
        tasks.append(("is_base", coin, IS_FROM,  IS_TO,  False, None))
        tasks.append(("is_dc",   coin, IS_FROM,  IS_TO,  True,  None))
        tasks.append(("oos_base",coin, OOS_FROM, OOS_TO, False, None))
        tasks.append(("oos_dc",  coin, OOS_FROM, OOS_TO, True,  None))

    for dfrom, dto in ROLLING:
        for coin in COINS:
            tasks.append((f"rbase_{dfrom[:7]}", coin, dfrom, dto, False, None))
            tasks.append((f"rdc_{dfrom[:7]}",   coin, dfrom, dto, True,  None))

    for coin in COINS:
        for seed in range(1, N_SEEDS + 1):
            tasks.append(("seed_dc", coin, OOS_FROM, OOS_TO, True, seed))

    results = {}

    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = {
            pool.submit(_run, tag, coin, dfrom, dto, dc_core, seed): (tag, coin, dfrom)
            for tag, coin, dfrom, dto, dc_core, seed in tasks
        }
        for fut in as_completed(futs):
            tag, coin, dfrom = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  ✗ {tag} {coin} crashed: {e}", flush=True)
                res = {}
            results[(tag, coin)] = res

    elapsed = time.time() - t0

    lines = []
    def w(line=""):
        print(line)
        lines.append(line)

    w("=" * 72)
    w("  DC CORE (daily/4h Donchian gate) — Backtest Report on top of Option 3")
    w(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    w("=" * 72)

    # IS
    w("\n── In-Sample (2020–2023) ──────────────────────────────────────────────")
    w(f"  {'Coin':<12} {'Base Sharpe':>12} {'DC Sharpe':>12} {'Δ':>8} {'Better?':>8}")
    w("  " + "─" * 54)
    is_dc_wins = 0
    for coin in COINS:
        sb = sharpe(results.get(("is_base", coin)))
        sd = sharpe(results.get(("is_dc",   coin)))
        if sb is None or sd is None:
            w(f"  {coin:<12} {'—':>12} {'—':>12}")
            continue
        delta = sd - sb
        better = "✓" if sd > sb else "✗"
        if sd > sb: is_dc_wins += 1
        w(f"  {coin:<12} {sb:>12.3f} {sd:>12.3f} {delta:>+8.3f} {better:>8}")
    w(f"\n  IS: DC Core beats Baseline on {is_dc_wins}/10 coins")

    # OOS
    w("\n── Out-of-Sample (2024–2026) ──────────────────────────────────────────")
    w(f"  {'Coin':<12} {'Base Sharpe':>12} {'DC Sharpe':>12} {'Δ':>8} {'Better?':>8}")
    w("  " + "─" * 54)
    oos_dc_wins = 0
    for coin in COINS:
        sb = sharpe(results.get(("oos_base", coin)))
        sd = sharpe(results.get(("oos_dc",   coin)))
        if sb is None or sd is None:
            w(f"  {coin:<12} {'—':>12} {'—':>12}")
            continue
        delta = sd - sb
        better = "✓" if sd > sb else "✗"
        if sd > sb: oos_dc_wins += 1
        w(f"  {coin:<12} {sb:>12.3f} {sd:>12.3f} {delta:>+8.3f} {better:>8}")
    w(f"\n  OOS: DC Core beats Baseline on {oos_dc_wins}/10 coins")

    # Rolling
    w("\n── Rolling Windows (6-month, DC Core vs Baseline) ────────────────────")
    w(f"  {'Window':<14} {'DC wins':>10} {'Base wins':>10} {'Ties':>6}")
    w("  " + "─" * 44)
    total_dc_wins = total_base_wins = 0
    for dfrom, dto in ROLLING:
        win = dfrom[:7]
        dw = bw = ties = 0
        for coin in COINS:
            sd = sharpe(results.get((f"rdc_{win}",   coin)))
            sb = sharpe(results.get((f"rbase_{win}", coin)))
            if sd is None or sb is None: continue
            if sd > sb: dw += 1
            elif sb > sd: bw += 1
            else: ties += 1
        total_dc_wins += dw
        total_base_wins += bw
        w(f"  {dfrom[:7]}–{dto[:7]} {dw:>10} {bw:>10} {ties:>6}")

    roll_pct = total_dc_wins / max(1, total_dc_wins + total_base_wins) * 100
    w(f"\n  Total: DC {total_dc_wins}  vs  Base {total_base_wins}  ({roll_pct:.0f}% DC wins)")

    # Pass/Fail
    c1 = oos_dc_wins >= 6
    c2 = roll_pct >= 50.0
    passed = c1 and c2

    w("\n" + "=" * 72)
    w("  PASS / FAIL")
    w("=" * 72)
    w(f"  1. DC Core beats Baseline OOS on ≥6/10 coins:   {oos_dc_wins}/10  {'✓' if c1 else '✗'}")
    w(f"  2. DC Core wins majority of rolling windows:     {roll_pct:.0f}%  {'✓' if c2 else '✗'}")
    w()
    w(f"  {'✅ DC CORE PASSES — deploy to demo with ENABLE_DC_FILTER=True' if passed else '❌ DC CORE FAILS — keep ENABLE_DC_FILTER=False'}")
    w()
    w(f"  Elapsed: {elapsed/60:.1f} min  |  Total runs: {_total}")
    w("=" * 72)

    with open("dc_core_backtest_report.txt", "w") as f:
        f.write("\n".join(lines))
    with open("dc_core_backtest_results.json", "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "oos_dc_wins": oos_dc_wins,
            "rolling_dc_pct": round(roll_pct, 1),
        }, f, indent=2)

    print(f"\nReport → dc_core_backtest_report.txt")
    print(f"Results → dc_core_backtest_results.json")


if __name__ == "__main__":
    main()
