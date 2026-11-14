#!/usr/bin/env python3
"""
run_sentiment_backtest.py
─────────────────────────
Pre-registered Step 7: Fear & Greed sentiment filter validation.

What it tests
─────────────
  Fear & Greed index (alternative.me) as a macro entry gate.
  Extreme Greed (≥80) blocks LONG entries — avoids buying market tops.
  Extreme Fear  (≤20) blocks SHORT entries — avoids shorting capitulation.

  Baseline : Option3 + DC filter + 4h MR + floor ON  [current best config]
  Variant  : + sentiment filter ON (enable_sentiment_filter=True)

  Thresholds pre-registered: LONG blocked at F&G ≥ 80, SHORT at F&G ≤ 20.
  Do NOT change thresholds after seeing results.

Pass criteria (same 4-gate protocol):
  1. OOS:     Sentiment ON beats baseline on ≥6/10 coins by Sharpe
  2. Rolling: Sentiment ON wins ≥50% of 6-month windows
  3. Plateau: OOS wins stable (IS/OOS within 3)
  4. Seed:    Sentiment ON beats random median on ≥7/10 coins

Prerequisites:
  Run python3 fetch_sentiment_data.py FIRST to populate sentiment_fng table.

Run (~90 min on Render with 1 worker):
  nohup python3 run_sentiment_backtest.py >> sentiment_backtest.log 2>&1 &
  echo $!

Output:
  sentiment_full_report.txt
  sentiment_full_results.json
"""

import json, os, statistics, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from backtester import start_backtest, get_run, init_backtest_tables
from database import db_conn
from sentiment import load_fng_history
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

_lock = threading.Lock()
_done = [0]
_total_ref = [0]


def _fetch_done(email: str):
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


def _run(tag, coin, dfrom, dto, sent_on: bool, seed=None):
    variant = "on" if sent_on else "base"
    suffix  = f"-s{seed}" if seed is not None else ""
    email   = f"sent-{tag}-{variant}-{coin.lower()}{suffix}@asym-internal"

    cached = _fetch_done(email)
    if cached is not None:
        with _lock:
            _done[0] += 1
            if _done[0] % 100 == 0 or _done[0] <= 4:
                print(f"  [{_done[0]}/{_total_ref[0]}] skip (done) {tag} sent={variant} {coin}{suffix}",
                      flush=True)
        return email, cached

    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=dfrom, date_to=dto,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False,
        enable_mr_signal=False,       # 4h MR replaces 1h MR
        enable_mr_4h_signal=True,     # 4h MR ON (current best config)
        enable_regime_gate=False, enable_e2=False, enable_step2_score=False,
        enable_grade_b_all=True, enable_floor=True,
        enable_dc_core=True, dc_long_cut=0.35, dc_short_cut=0.65,
        enable_sentiment_filter=sent_on,
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
                    print(f"  [{_done[0]}/{_total_ref[0]}] ✓ {tag} sent={variant} {coin}{suffix}",
                          flush=True)
            return email, rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total_ref[0]}] ✗ ERROR {tag} sent={variant} {coin}", flush=True)
            return email, {}

    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total_ref[0]}] ✗ TIMEOUT {tag} sent={variant} {coin}", flush=True)
    return email, {}


def sharpe(res): return res.get("sharpe_ratio")    if res else None
def wr(res):     return res.get("win_rate")         if res else None
def ret(res):    return res.get("total_return_pct") if res else None


def main():
    # Verify F&G data is loaded
    fng = load_fng_history()
    if not fng:
        print("\n✗ sentiment_fng table is empty.")
        print("  Run: python3 fetch_sentiment_data.py FIRST")
        return

    print(f"\n  F&G history: {len(fng)} days ({sorted(fng)[0]} → {sorted(fng)[-1]})")

    t0 = time.time()
    tasks = []

    for coin in COINS:
        tasks.append(("is",  coin, IS_FROM,  IS_TO,  False, None))
        tasks.append(("is",  coin, IS_FROM,  IS_TO,  True,  None))
        tasks.append(("oos", coin, OOS_FROM, OOS_TO, False, None))
        tasks.append(("oos", coin, OOS_FROM, OOS_TO, True,  None))

    for dfrom, dto in ROLLING:
        for coin in COINS:
            tasks.append((f"roll_{dfrom[:7]}", coin, dfrom, dto, False, None))
            tasks.append((f"roll_{dfrom[:7]}", coin, dfrom, dto, True,  None))

    for coin in COINS:
        for seed in range(1, N_SEEDS + 1):
            tasks.append(("seed", coin, OOS_FROM, OOS_TO, True, seed))

    _total_ref[0] = len(tasks)

    print(f"\nSentiment Filter (Step 7) — Full Backtest")
    print(f"  Baseline : Option3 + DC filter + 4h MR + floor ON")
    print(f"  Variant  : + F&G extreme gate (≥80 blocks LONG, ≤20 blocks SHORT)")
    print(f"  IS {IS_FROM}→{IS_TO}  OOS {OOS_FROM}→{OOS_TO}")
    print(f"  {len(tasks)} total runs | {MAX_WORKERS} workers")
    print()

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run, *t): t for t in tasks}
        for fut in as_completed(futs):
            email, res = fut.result()
            results[email] = res

    def _get(tag, coin, sent_on, seed=None):
        variant = "on" if sent_on else "base"
        suffix  = f"-s{seed}" if seed is not None else ""
        return results.get(f"sent-{tag}-{variant}-{coin.lower()}{suffix}@asym-internal", {})

    lines = []
    def pr(s=""): lines.append(s); print(s)

    pr("=" * 76)
    pr("  SENTIMENT FILTER — Full Validation Report")
    pr(f"  Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    pr(f"  Baseline  : Option3 + DC filter + 4h MR + floor ON")
    pr(f"  Variant   : + F&G sentiment gate (LONG blocked ≥80, SHORT blocked ≤20)")
    pr(f"  Runs      : {_total_ref[0]}")
    pr("=" * 76)

    # ── IS results ────────────────────────────────────────────────────────────
    pr(f"\n── IS 2020–2023 {'─'*55}")
    pr(f"  {'Coin':<14} {'Base':>8} {'Sent':>8} {'Δ':>7}  OK?")
    pr("  " + "─" * 48)
    is_wins = 0
    for coin in COINS:
        b = sharpe(_get("is", coin, False))
        s = sharpe(_get("is", coin, True))
        if b is None or s is None:
            pr(f"  {coin:<14} {'—':>8} {'—':>8}"); continue
        d = s - b
        ok = "✓" if d > 0 else "✗"
        if d > 0: is_wins += 1
        pr(f"  {coin:<14} {b:>8.3f} {s:>8.3f} {d:>+7.3f}  {ok}")
    pr(f"  IS: Sentiment wins {is_wins}/10")

    # ── OOS results ───────────────────────────────────────────────────────────
    pr(f"\n── OOS 2024–2026 {'─'*54}")
    pr(f"  {'Coin':<14} {'Base':>8} {'Sent':>8} {'Δ':>7}  OK?")
    pr("  " + "─" * 48)
    oos_wins = 0
    for coin in COINS:
        b = sharpe(_get("oos", coin, False))
        s = sharpe(_get("oos", coin, True))
        if b is None or s is None:
            pr(f"  {coin:<14} {'—':>8} {'—':>8}"); continue
        d = s - b
        ok = "✓" if d > 0 else "✗"
        if d > 0: oos_wins += 1
        pr(f"  {coin:<14} {b:>8.3f} {s:>8.3f} {d:>+7.3f}  {ok}")
    pr(f"  OOS: Sentiment wins {oos_wins}/10")

    # ── Plateau ───────────────────────────────────────────────────────────────
    plateau_stable = abs(is_wins - oos_wins) <= 3
    pr(f"\n── Plateau — IS {is_wins}/10  OOS {oos_wins}/10  "
       f"{'✓ STABLE' if plateau_stable else '✗ KNIFE-EDGE'}")

    # ── Rolling windows ───────────────────────────────────────────────────────
    pr(f"\n── Rolling Windows {'─'*52}")
    roll_wins = roll_total = 0
    for dfrom, dto in ROLLING:
        for coin in COINS:
            b = sharpe(_get(f"roll_{dfrom[:7]}", coin, False))
            s = sharpe(_get(f"roll_{dfrom[:7]}", coin, True))
            if b is not None and s is not None:
                roll_total += 1
                if s > b: roll_wins += 1
    roll_pct = roll_wins / roll_total * 100 if roll_total else 0
    pr(f"  Sentiment wins {roll_wins}/{roll_total} ({roll_pct:.0f}%)")
    pr(f"  {'✓ PASS' if roll_pct >= 50 else '✗ FAIL'} (need ≥50%)")

    # ── Seed benchmark ────────────────────────────────────────────────────────
    pr(f"\n── 100-seed Benchmark {'─'*49}")
    pr(f"  {'Coin':<14} {'Sent Sh':>9}  {'Rand P50':>9}  {'Pctile':>7}")
    pr("  " + "─" * 48)
    seed_wins = 0
    for coin in COINS:
        s_oos = sharpe(_get("oos", coin, True))
        seed_sharpes = [sharpe(_get("seed", coin, True, sd)) for sd in range(1, N_SEEDS+1)]
        seed_sharpes = [x for x in seed_sharpes if x is not None]
        if not seed_sharpes or s_oos is None:
            pr(f"  {coin:<14}  {'—':>9}  {'—':>9}"); continue
        rand_p50 = statistics.median(seed_sharpes)
        pctile   = sum(1 for x in seed_sharpes if x < s_oos) / len(seed_sharpes) * 100
        ok = "✓" if s_oos > rand_p50 else "✗"
        if s_oos > rand_p50: seed_wins += 1
        pr(f"  {coin:<14}  {s_oos:>9.3f}  {rand_p50:>9.3f}  {pctile:>6.0f}%  {ok}")
    pr(f"  Seed gate: {seed_wins}/10 (need ≥7)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    oos_pass     = oos_wins >= 6
    roll_pass    = roll_pct >= 50
    plateau_pass = plateau_stable
    seed_pass    = seed_wins >= 7
    all_pass     = oos_pass and roll_pass and plateau_pass and seed_pass

    elapsed = (time.time() - t0) / 60
    pr(f"\n{'='*76}")
    pr(f"  VERDICT")
    pr(f"{'='*76}")
    pr(f"  OOS gate    : {'✓ PASS' if oos_pass     else '✗ FAIL'} ({oos_wins}/10 — need ≥6)")
    pr(f"  Rolling gate: {'✓ PASS' if roll_pass    else '✗ FAIL'} ({roll_pct:.0f}% — need ≥50%)")
    pr(f"  Plateau gate: {'✓ PASS' if plateau_pass else '✗ FAIL'} (IS {is_wins}/10, OOS {oos_wins}/10)")
    pr(f"  Seed gate   : {'✓ PASS' if seed_pass    else '✗ FAIL'} ({seed_wins}/10 — need ≥7)")
    pr()
    if all_pass:
        pr(f"  SENTIMENT FILTER: ✓ PASSED")
        pr(f"  Next: add ENABLE_SENTIMENT_FILTER flag to main.py, deploy to MAIN demo first")
    else:
        pr(f"  SENTIMENT FILTER: ✗ FAILED — ENABLE_SENTIMENT_FILTER stays OFF")
        failed_gates = [
            g for g, p in [("OOS", oos_pass), ("Rolling", roll_pass),
                            ("Plateau", plateau_pass), ("Seed", seed_pass)] if not p
        ]
        pr(f"  Failed gates: {', '.join(failed_gates)}")
    pr(f"\n  Elapsed: {elapsed:.1f} min")
    pr(f"{'='*76}")

    report_path = os.path.join(os.path.dirname(__file__), "sentiment_full_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport → {report_path}")

    json_path = os.path.join(os.path.dirname(__file__), "sentiment_full_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "is_wins": is_wins, "oos_wins": oos_wins,
            "roll_pct": round(roll_pct, 1),
            "seed_wins": seed_wins, "plateau_stable": plateau_stable,
            "all_pass": all_pass, "elapsed_min": round(elapsed, 1),
        }, f, indent=2)
    print(f"JSON  → {json_path}")


if __name__ == "__main__":
    main()
