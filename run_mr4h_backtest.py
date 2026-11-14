#!/usr/bin/env python3
"""
run_mr4h_backtest.py
────────────────────
Pre-registered 4h MR variant validation.

What it tests
─────────────
  1h MR (current default) fires when 1h ADX < 20 + price at 1h Bollinger Band.
  FAILED OOS 3/10 (2026-11-12) — regime detection too noisy on 1h timeframe.

  4h MR variant: same Bollinger/RSI/ADX logic but applied to 4h candles.
  Signals fire less frequently but represent cleaner, higher-quality setups.
  TP/SL remain in 1h ATR units for consistent sizing.

  Baseline : enable_mr_signal=True,  enable_mr_4h_signal=False  [current live]
  4h MR ON : enable_mr_signal=False, enable_mr_4h_signal=True   [replace 1h MR]

  Option 3 (ENABLE_GRADE_B_ALL=True) + floor ON applied to both.

Pass criteria (same 4-gate protocol as DC core / MR backtest):
  1. OOS:     4h MR beats baseline on ≥6/10 coins by Sharpe
  2. Rolling: 4h MR wins ≥50% of 6-month windows
  3. Plateau: OOS wins stable (not knife-edge) — checked via IS consistency
  4. Seed:    4h MR beats random median on ≥7/10 coins

Run (nohup recommended — takes ~90 min):
  nohup python3 run_mr4h_backtest.py >> mr4h_backtest.log 2>&1 &
  echo $!

Output:
  mr4h_full_report.txt
  mr4h_full_results.json
"""

import json, os, statistics, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from backtester import start_backtest, get_run, init_backtest_tables
from database import db_conn
init_backtest_tables()

# ── Config ────────────────────────────────────────────────────────────────────
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
TIMEOUT_MIN = 300
N_SEEDS  = 100
MAX_WORKERS = 1

ROLLING = [
    ("2024-01-01","2024-06-30"), ("2024-04-01","2024-09-30"),
    ("2024-07-01","2024-12-31"), ("2024-10-01","2025-03-31"),
    ("2025-01-01","2025-06-30"), ("2025-04-01","2025-09-30"),
    ("2025-07-01","2025-12-31"), ("2025-10-01","2026-03-31"),
    ("2026-01-01","2026-06-30"), ("2026-04-01","2026-10-31"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_done = [0]
_total_ref = [0]

def _fetch_done(email: str):
    """Return results dict if run is done in DB, else None."""
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT result_json FROM backtest_runs WHERE email=%s AND status='done' LIMIT 1",
                (email,),
            )
            row = cur.fetchone()
        if not row:
            return None
        raw = row[0] if isinstance(row, (list, tuple)) else row["result_json"]
        if not raw:
            return {}
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None


def _run(tag, coin, dfrom, dto, mr4h_on: bool, seed=None):
    variant = "on" if mr4h_on else "base"
    suffix  = f"-s{seed}" if seed is not None else ""
    email   = f"mr4h-{tag}-{variant}-{coin.lower()}{suffix}@asym-internal"

    cached = _fetch_done(email)
    if cached is not None:
        with _lock:
            _done[0] += 1
            if _done[0] % 100 == 0 or _done[0] <= 4:
                print(f"  [{_done[0]}/{_total_ref[0]}] skip (done) {tag} 4hMR={variant} {coin}{suffix}",
                      flush=True)
        return email, cached

    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=dfrom, date_to=dto,
        start_equity=START_EQ,
        enable_t16=True,
        enable_t18=False,
        enable_s4=False,
        enable_mr_4h_gate=False,
        enable_mr_signal=(not mr4h_on),   # baseline: 1h MR ON; variant: 1h MR OFF
        enable_mr_4h_signal=mr4h_on,      # variant: 4h MR ON; baseline: OFF
        enable_regime_gate=False,
        enable_e2=False,
        enable_step2_score=False,
        enable_grade_b_all=True,
        enable_floor=True,
        random_seed=seed,
    )

    deadline = time.time() + TIMEOUT_MIN * 60
    while time.time() < deadline:
        time.sleep(5)
        rec = get_run(run_id)
        if not rec:
            continue
        st = rec.get("status", "")
        if st == "done":
            with _lock:
                _done[0] += 1
                if _done[0] % 50 == 0 or _done[0] <= 10:
                    print(f"  [{_done[0]}/{_total_ref[0]}] ✓ {tag} 4hMR={variant} {coin}{suffix}",
                          flush=True)
            return email, rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total_ref[0]}] ✗ ERROR {tag} 4hMR={variant} {coin}",
                      flush=True)
            return email, {}

    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total_ref[0]}] ✗ TIMEOUT {tag} 4hMR={variant} {coin}", flush=True)
    return email, {}


def sharpe(res): return res.get("sharpe_ratio")      if res else None
def wr(res):     return res.get("win_rate")           if res else None
def ret(res):    return res.get("total_return_pct")   if res else None


def main():
    t0 = time.time()

    tasks = []  # (tag, coin, dfrom, dto, mr4h_on, seed)

    # IS + OOS for baseline and 4h MR ON
    for coin in COINS:
        tasks.append(("is",  coin, IS_FROM,  IS_TO,  False, None))
        tasks.append(("is",  coin, IS_FROM,  IS_TO,  True,  None))
        tasks.append(("oos", coin, OOS_FROM, OOS_TO, False, None))
        tasks.append(("oos", coin, OOS_FROM, OOS_TO, True,  None))

    # Rolling windows for both variants
    for dfrom, dto in ROLLING:
        for coin in COINS:
            tasks.append((f"roll_{dfrom[:7]}", coin, dfrom, dto, False, None))
            tasks.append((f"roll_{dfrom[:7]}", coin, dfrom, dto, True,  None))

    # 100-seed benchmark for 4h MR ON variant
    for coin in COINS:
        for seed in range(1, N_SEEDS + 1):
            tasks.append(("seed", coin, OOS_FROM, OOS_TO, True, seed))

    _total_ref[0] = len(tasks)

    print(f"\n4h MR Variant — Full Backtest")
    print(f"  IS {IS_FROM}→{IS_TO}  OOS {OOS_FROM}→{OOS_TO}")
    print(f"  10 rolling windows × 2 variants  |  {N_SEEDS} seeds (4h MR ON only)")
    print(f"  {len(tasks)} total runs | {MAX_WORKERS} workers")
    print()

    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(_run, *t): t for t in tasks
        }
        for fut in as_completed(futs):
            email, res = fut.result()
            results[email] = res

    # ── Collate ───────────────────────────────────────────────────────────────
    def _get(tag, coin, mr4h_on, seed=None):
        variant = "on" if mr4h_on else "base"
        suffix  = f"-s{seed}" if seed is not None else ""
        email   = f"mr4h-{tag}-{variant}-{coin.lower()}{suffix}@asym-internal"
        return results.get(email, {})

    lines = []
    def pr(s=""): lines.append(s); print(s)

    pr("=" * 76)
    pr("  4h MR VARIANT — Full Validation Report")
    pr(f"  Generated  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    pr(f"  Baseline   : Option 3 + floor ON, 1h MR ON (enable_mr_signal=True)")
    pr(f"  Variant    : 1h MR OFF + 4h MR ON (enable_mr_4h_signal=True)")
    pr(f"  Runs       : {_total_ref[0]}")
    pr("=" * 76)

    # ── IS results ────────────────────────────────────────────────────────────
    pr(f"\n── IS 2020–2023  [Baseline vs 4h MR] {'─'*39}")
    pr(f"  {'Coin':<14} {'Base':>8} {'4hMR':>8} {'Δ':>7}  {'MDD base':>9} {'MDD 4hMR':>9}  OK?")
    pr("  " + "─" * 72)
    is_wins = 0
    for coin in COINS:
        r_base = _get("is", coin, False)
        r_4hmr = _get("is", coin, True)
        s_base = sharpe(r_base)
        s_4hmr = sharpe(r_4hmr)
        if s_base is None or s_4hmr is None:
            pr(f"  {coin:<14} {'—':>8} {'—':>8}")
            continue
        delta    = s_4hmr - s_base
        mdd_base = r_base.get("max_drawdown_pct", 0) or 0
        mdd_4hmr = r_4hmr.get("max_drawdown_pct", 0) or 0
        ok = "✓" if delta > 0 else "✗"
        if delta > 0: is_wins += 1
        pr(f"  {coin:<14} {s_base:>8.3f} {s_4hmr:>8.3f} {delta:>+7.3f}  {mdd_base:>8.1f}%  {mdd_4hmr:>8.1f}%     {ok}")
    pr(f"  IS: 4h MR wins {is_wins}/10")

    # ── OOS results ───────────────────────────────────────────────────────────
    pr(f"\n── OOS 2024–2026  [Baseline vs 4h MR] {'─'*38}")
    pr(f"  {'Coin':<14} {'Base':>8} {'4hMR':>8} {'Δ':>7}  {'MDD base':>9} {'MDD 4hMR':>9}  OK?")
    pr("  " + "─" * 72)
    oos_wins = 0
    for coin in COINS:
        r_base = _get("oos", coin, False)
        r_4hmr = _get("oos", coin, True)
        s_base = sharpe(r_base)
        s_4hmr = sharpe(r_4hmr)
        if s_base is None or s_4hmr is None:
            pr(f"  {coin:<14} {'—':>8} {'—':>8}")
            continue
        delta    = s_4hmr - s_base
        mdd_base = r_base.get("max_drawdown_pct", 0) or 0
        mdd_4hmr = r_4hmr.get("max_drawdown_pct", 0) or 0
        ok = "✓" if delta > 0 else "✗"
        if delta > 0: oos_wins += 1
        pr(f"  {coin:<14} {s_base:>8.3f} {s_4hmr:>8.3f} {delta:>+7.3f}  {mdd_base:>8.1f}%  {mdd_4hmr:>8.1f}%     {ok}")
    pr(f"  OOS: 4h MR wins {oos_wins}/10")

    # ── Plateau sensitivity ───────────────────────────────────────────────────
    pr(f"\n── Plateau — IS vs OOS consistency")
    pr(f"  IS  wins: {is_wins}/10   OOS wins: {oos_wins}/10")
    plateau_stable = abs(is_wins - oos_wins) <= 3
    pr(f"  {'✓ STABLE (IS/OOS within 3)' if plateau_stable else '✗ KNIFE-EDGE (IS/OOS diverge >3)'}")

    # ── Rolling windows ───────────────────────────────────────────────────────
    pr(f"\n── Rolling Windows (6-month, 4h MR vs Baseline) {'─'*26}")
    roll_wins  = 0
    roll_total = 0
    roll_detail = []
    for dfrom, dto in ROLLING:
        window_wins = 0
        window_total = 0
        for coin in COINS:
            r_base = _get(f"roll_{dfrom[:7]}", coin, False)
            r_4hmr = _get(f"roll_{dfrom[:7]}", coin, True)
            s_base = sharpe(r_base)
            s_4hmr = sharpe(r_4hmr)
            if s_base is not None and s_4hmr is not None:
                if s_4hmr > s_base: window_wins += 1
                roll_total += 1
                if s_4hmr > s_base: roll_wins += 1
                window_total += 1
        roll_detail.append((dfrom[:7], dto[:7], window_wins, window_total))
    for wfrom, wto, ww, wt in roll_detail:
        pr(f"  {wfrom}→{wto}  {ww:>2}/{wt:<2}  {'✓' if ww > wt/2 else '✗'}")
    roll_pct = roll_wins / roll_total * 100 if roll_total else 0
    pr(f"  Total: 4h MR wins {roll_wins}/{roll_total} ({roll_pct:.0f}%)")
    pr(f"  {'✓ PASS' if roll_pct >= 50 else '✗ FAIL'} (need ≥50%)")

    # ── Seed benchmark ────────────────────────────────────────────────────────
    pr(f"\n── 100-seed Random Benchmark (4h MR ON, OOS 2024–2026) {'─'*19}")
    pr(f"  {'Coin':<14} {'4hMR Sh':>9}  {'Rand P50':>9}  {'Pctile':>7}")
    pr("  " + "─" * 48)
    seed_gate_wins = 0
    for coin in COINS:
        r_4hmr = _get("oos", coin, True)
        s_4hmr = sharpe(r_4hmr)
        seed_sharpes = []
        for seed in range(1, N_SEEDS + 1):
            rs = _get("seed", coin, True, seed)
            sv = sharpe(rs)
            if sv is not None:
                seed_sharpes.append(sv)
        if not seed_sharpes or s_4hmr is None:
            pr(f"  {coin:<14}  {'—':>9}  {'—':>9}")
            continue
        rand_p50 = statistics.median(seed_sharpes)
        pctile   = sum(1 for x in seed_sharpes if x < s_4hmr) / len(seed_sharpes) * 100
        ok = "✓" if s_4hmr > rand_p50 else "✗"
        if s_4hmr > rand_p50: seed_gate_wins += 1
        pr(f"  {coin:<14}  {s_4hmr:>9.3f}  {rand_p50:>9.3f}  {pctile:>6.0f}%  {ok}")
    pr(f"  Seed gate: {seed_gate_wins}/10 beats random median (need ≥7)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    oos_pass     = oos_wins >= 6
    roll_pass    = roll_pct >= 50
    plateau_pass = plateau_stable
    seed_pass    = seed_gate_wins >= 7
    all_pass     = oos_pass and roll_pass and plateau_pass and seed_pass

    elapsed = (time.time() - t0) / 60
    pr(f"\n{'='*76}")
    pr(f"  VERDICT")
    pr(f"{'='*76}")
    pr(f"  OOS gate    : {'✓ PASS' if oos_pass     else '✗ FAIL'} ({oos_wins}/10 — need ≥6)")
    pr(f"  Rolling gate: {'✓ PASS' if roll_pass    else '✗ FAIL'} ({roll_pct:.0f}% — need ≥50%)")
    pr(f"  Plateau gate: {'✓ PASS' if plateau_pass else '✗ FAIL'} (IS {is_wins}/10, OOS {oos_wins}/10)")
    pr(f"  Seed gate   : {'✓ PASS' if seed_pass    else '✗ FAIL'} ({seed_gate_wins}/10 — need ≥7)")
    pr()
    if all_pass:
        pr(f"  4h MR: ✓ PASSED — enable_mr_4h_signal=True validated")
        pr(f"  Next: add ENABLE_MR_4H_SIGNAL flag to main.py, deploy to MAIN demo first")
    else:
        pr(f"  4h MR: ✗ FAILED — enable_mr_4h_signal stays OFF")
        failed_gates = [
            g for g, p in [("OOS", oos_pass), ("Rolling", roll_pass),
                            ("Plateau", plateau_pass), ("Seed", seed_pass)] if not p
        ]
        pr(f"  Failed gates: {', '.join(failed_gates)}")
    pr(f"\n  Elapsed: {elapsed:.1f} min")
    pr(f"{'='*76}")

    # ── Write output ──────────────────────────────────────────────────────────
    report_path = os.path.join(os.path.dirname(__file__), "mr4h_full_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")

    json_path = os.path.join(os.path.dirname(__file__), "mr4h_full_results.json")
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "is_wins": is_wins, "oos_wins": oos_wins,
        "roll_pct": round(roll_pct, 1), "seed_gate_wins": seed_gate_wins,
        "plateau_stable": plateau_stable, "all_pass": all_pass,
        "elapsed_min": round(elapsed, 1),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON  written to {json_path}")


if __name__ == "__main__":
    main()
