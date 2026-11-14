#!/usr/bin/env python3
"""
run_pattern_backtest.py
───────────────────────
Step: Candle Pattern — entry confirmation gate + grade boost validation.

What it tests
─────────────
  Baseline : Option3 + DC filter + 4h MR + floor ON  [current best config]
  Gate     : + enable_pattern_gate=True  (require momentum/hammer/engulfing+ to fire)
  Boost    : + enable_pattern_boost=True (upgrade Grade B→A on _raw_pat ≥ 0.85)

  Gate blocks basic bullish/bearish closes (score < 0.65) — only high-quality
  candle patterns confirm the entry. Expects: fewer trades, higher win rate.

  Boost upgrades Grade B to A when pattern is very strong (hammer, engulfing,
  three_white_soldiers). Full position sizing on high-conviction setups.
  Expects: same trade count, better average return per trade.

Pass criteria (4-gate protocol):
  1. OOS:     variant beats baseline on ≥6/10 coins by Sharpe
  2. Rolling: variant wins ≥50% of 6-month windows
  3. Plateau: IS/OOS wins within 3 of each other
  4. Seed:    variant beats random median on ≥7/10 coins

Run (~4-5 hours on Render with 1 worker):
  nohup python3 run_pattern_backtest.py >> pattern_backtest.log 2>&1 &
  echo $!

Output:
  pattern_full_report.txt
  pattern_full_results.json
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

# Variants: (label, gate, boost)
VARIANTS = [
    ("base",  False, False),
    ("gate",  True,  False),
    ("boost", False, True),
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


def _run(tag, coin, dfrom, dto, variant, seed=None):
    suffix = f"-s{seed}" if seed is not None else ""
    email  = f"pat-{tag}-{variant}-{coin.lower()}{suffix}@asym-internal"

    cached = _fetch_done(email)
    if cached is not None:
        with _lock:
            _done[0] += 1
            if _done[0] % 200 == 0 or _done[0] <= 4:
                print(f"  [{_done[0]}/{_total_ref[0]}] skip {tag} {variant} {coin}{suffix}", flush=True)
        return email, cached

    gate  = variant == "gate"
    boost = variant == "boost"

    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=dfrom, date_to=dto,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False,
        enable_mr_signal=False,
        enable_mr_4h_signal=True,
        enable_regime_gate=False, enable_e2=False, enable_step2_score=False,
        enable_grade_b_all=True, enable_floor=True,
        enable_dc_core=True, dc_long_cut=0.35, dc_short_cut=0.65,
        enable_sentiment_filter=False,
        enable_pattern_gate=gate,
        enable_pattern_boost=boost,
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
                if _done[0] % 100 == 0 or _done[0] <= 10:
                    print(f"  [{_done[0]}/{_total_ref[0]}] ✓ {tag} {variant} {coin}{suffix}", flush=True)
            return email, rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total_ref[0]}] ✗ ERROR {tag} {variant} {coin}", flush=True)
            return email, {}

    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total_ref[0]}] ✗ TIMEOUT {tag} {variant} {coin}", flush=True)
    return email, {}


def sharpe(res): return res.get("sharpe_ratio")    if res else None
def trades(res): return res.get("total_trades", 0) if res else 0


def main():
    t0 = time.time()
    tasks = []

    for vlabel, _, _ in VARIANTS:
        for coin in COINS:
            tasks.append(("is",  coin, IS_FROM,  IS_TO,  vlabel, None))
            tasks.append(("oos", coin, OOS_FROM, OOS_TO, vlabel, None))
        for dfrom, dto in ROLLING:
            for coin in COINS:
                tasks.append((f"roll_{dfrom[:7]}", coin, dfrom, dto, vlabel, None))
        for coin in COINS:
            for seed in range(1, N_SEEDS + 1):
                tasks.append(("seed", coin, OOS_FROM, OOS_TO, vlabel, seed))

    _total_ref[0] = len(tasks)

    print(f"\nCandle Pattern — Full Backtest")
    print(f"  Baseline : Option3 + DC filter + 4h MR + floor ON")
    print(f"  Gate     : + pattern_gate (require momentum/hammer/engulfing+)")
    print(f"  Boost    : + pattern_boost (Grade B→A when raw_pat ≥ 0.85)")
    print(f"  IS {IS_FROM}→{IS_TO}  OOS {OOS_FROM}→{OOS_TO}")
    print(f"  {len(tasks)} total runs | {MAX_WORKERS} worker(s)")
    print()

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run, *t): t for t in tasks}
        for fut in as_completed(futs):
            email, res = fut.result()
            results[email] = res

    def _get(tag, coin, variant, seed=None):
        suffix = f"-s{seed}" if seed is not None else ""
        return results.get(f"pat-{tag}-{variant}-{coin.lower()}{suffix}@asym-internal", {})

    lines = []
    def pr(s=""): lines.append(s); print(s)

    all_verdicts = {}

    for vlabel, gate, boost in VARIANTS[1:]:  # skip baseline, compare gate and boost vs base
        pr("=" * 76)
        pr(f"  VARIANT: {vlabel.upper()}")
        if gate:
            pr(f"  Pattern gate ON — blocks basic bullish/bearish, requires momentum+")
        else:
            pr(f"  Pattern boost ON — Grade B→A when raw_pat ≥ 0.85")
        pr("=" * 76)

        # ── IS ────────────────────────────────────────────────────────────────
        pr(f"\n── IS 2020–2023 {'─'*55}")
        pr(f"  {'Coin':<14} {'Base Sh':>8} {'Var Sh':>8} {'Δ':>7}  {'Base Tr':>8} {'Var Tr':>8}  OK?")
        pr("  " + "─" * 60)
        is_wins = 0
        for coin in COINS:
            b = sharpe(_get("is", coin, "base"))
            v = sharpe(_get("is", coin, vlabel))
            if b is None or v is None:
                pr(f"  {coin:<14} {'—':>8} {'—':>8}"); continue
            d = v - b
            ok = "✓" if d > 0 else "✗"
            if d > 0: is_wins += 1
            bt = trades(_get("is", coin, "base"))
            vt = trades(_get("is", coin, vlabel))
            pr(f"  {coin:<14} {b:>8.3f} {v:>8.3f} {d:>+7.3f}  {bt:>8} {vt:>8}  {ok}")
        pr(f"  IS: {vlabel} wins {is_wins}/10")

        # ── OOS ───────────────────────────────────────────────────────────────
        pr(f"\n── OOS 2024–2026 {'─'*54}")
        pr(f"  {'Coin':<14} {'Base Sh':>8} {'Var Sh':>8} {'Δ':>7}  {'Base Tr':>8} {'Var Tr':>8}  OK?")
        pr("  " + "─" * 60)
        oos_wins = 0
        for coin in COINS:
            b = sharpe(_get("oos", coin, "base"))
            v = sharpe(_get("oos", coin, vlabel))
            if b is None or v is None:
                pr(f"  {coin:<14} {'—':>8} {'—':>8}"); continue
            d = v - b
            ok = "✓" if d > 0 else "✗"
            if d > 0: oos_wins += 1
            bt = trades(_get("oos", coin, "base"))
            vt = trades(_get("oos", coin, vlabel))
            pr(f"  {coin:<14} {b:>8.3f} {v:>8.3f} {d:>+7.3f}  {bt:>8} {vt:>8}  {ok}")
        pr(f"  OOS: {vlabel} wins {oos_wins}/10")

        # ── Plateau ───────────────────────────────────────────────────────────
        plateau_stable = abs(is_wins - oos_wins) <= 3
        pr(f"\n── Plateau — IS {is_wins}/10  OOS {oos_wins}/10  "
           f"{'✓ STABLE' if plateau_stable else '✗ KNIFE-EDGE'}")

        # ── Rolling ───────────────────────────────────────────────────────────
        pr(f"\n── Rolling Windows {'─'*52}")
        roll_wins = roll_total = 0
        for dfrom, dto in ROLLING:
            for coin in COINS:
                b = sharpe(_get(f"roll_{dfrom[:7]}", coin, "base"))
                v = sharpe(_get(f"roll_{dfrom[:7]}", coin, vlabel))
                if b is not None and v is not None:
                    roll_total += 1
                    if v > b: roll_wins += 1
        roll_pct = roll_wins / roll_total * 100 if roll_total else 0
        pr(f"  {vlabel} wins {roll_wins}/{roll_total} ({roll_pct:.0f}%)")
        pr(f"  {'✓ PASS' if roll_pct >= 50 else '✗ FAIL'} (need ≥50%)")

        # ── Seed ──────────────────────────────────────────────────────────────
        pr(f"\n── 100-seed Benchmark {'─'*49}")
        pr(f"  {'Coin':<14} {'Var Sh':>9}  {'Rand P50':>9}  {'Pctile':>7}")
        pr("  " + "─" * 48)
        seed_wins = 0
        for coin in COINS:
            v_oos = sharpe(_get("oos", coin, vlabel))
            seed_sharpes = [sharpe(_get("seed", coin, vlabel, sd)) for sd in range(1, N_SEEDS+1)]
            seed_sharpes = [x for x in seed_sharpes if x is not None]
            if not seed_sharpes or v_oos is None:
                pr(f"  {coin:<14}  {'—':>9}  {'—':>9}"); continue
            rand_p50 = statistics.median(seed_sharpes)
            pctile   = sum(1 for x in seed_sharpes if x < v_oos) / len(seed_sharpes) * 100
            ok = "✓" if v_oos > rand_p50 else "✗"
            if v_oos > rand_p50: seed_wins += 1
            pr(f"  {coin:<14}  {v_oos:>9.3f}  {rand_p50:>9.3f}  {pctile:>6.0f}%  {ok}")
        pr(f"  Seed gate: {seed_wins}/10 (need ≥7)")

        # ── Verdict ───────────────────────────────────────────────────────────
        oos_pass     = oos_wins >= 6
        roll_pass    = roll_pct >= 50
        plateau_pass = plateau_stable
        seed_pass    = seed_wins >= 7
        passed       = oos_pass and roll_pass and plateau_pass and seed_pass

        pr(f"\n{'='*76}")
        pr(f"  VERDICT — {vlabel.upper()}")
        pr(f"{'='*76}")
        pr(f"  OOS gate    : {'✓ PASS' if oos_pass     else '✗ FAIL'} ({oos_wins}/10 — need ≥6)")
        pr(f"  Rolling gate: {'✓ PASS' if roll_pass    else '✗ FAIL'} ({roll_pct:.0f}% — need ≥50%)")
        pr(f"  Plateau gate: {'✓ PASS' if plateau_pass else '✗ FAIL'} (IS {is_wins}/10, OOS {oos_wins}/10)")
        pr(f"  Seed gate   : {'✓ PASS' if seed_pass    else '✗ FAIL'} ({seed_wins}/10 — need ≥7)")
        pr()
        if passed:
            flag = "ENABLE_PATTERN_GATE" if gate else "ENABLE_PATTERN_BOOST"
            pr(f"  {vlabel.upper()}: ✓ PASSED — set {flag}=True in main.py, deploy to MAIN demo first")
        else:
            flag = "ENABLE_PATTERN_GATE" if gate else "ENABLE_PATTERN_BOOST"
            pr(f"  {vlabel.upper()}: ✗ FAILED — {flag} stays OFF")
            failed_gates = [
                g for g, p in [("OOS", oos_pass), ("Rolling", roll_pass),
                                ("Plateau", plateau_pass), ("Seed", seed_pass)] if not p
            ]
            pr(f"  Failed gates: {', '.join(failed_gates)}")
        pr()

        all_verdicts[vlabel] = {
            "passed": passed, "oos_wins": oos_wins, "roll_pct": round(roll_pct, 1),
            "is_wins": is_wins, "seed_wins": seed_wins,
        }

    elapsed = (time.time() - t0) / 60
    pr(f"\n{'='*76}")
    pr(f"  SUMMARY")
    pr(f"{'='*76}")
    for vlabel, v in all_verdicts.items():
        status = "✓ PASSED" if v["passed"] else "✗ FAILED"
        pr(f"  {vlabel:<8}: {status}  OOS {v['oos_wins']}/10  Rolling {v['roll_pct']:.0f}%  Seed {v['seed_wins']}/10")
    pr(f"\n  Elapsed: {elapsed:.1f} min")
    pr(f"{'='*76}")

    report_path = os.path.join(os.path.dirname(__file__), "pattern_full_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport → {report_path}")

    json_path = os.path.join(os.path.dirname(__file__), "pattern_full_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "variants": all_verdicts,
            "elapsed_min": round(elapsed, 1),
        }, f, indent=2)
    print(f"JSON  → {json_path}")


if __name__ == "__main__":
    main()
