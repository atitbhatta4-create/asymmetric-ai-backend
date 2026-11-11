#!/usr/bin/env python3
"""
run_grid_v2_backtest.py
───────────────────────
Full Grid v2 range-module validation.  Same protocol as DC core backtest.
Resumes from wherever it left off (DB IS the checkpoint).

Test plan
─────────
  Phase 1  IS + OOS: 3 ADX-threshold variants × baseline + grid-alone + grid+DC
             v12: adx_thresh=12  (strict — only quiet markets)
             v15: adx_thresh=15  (baseline production candidate)
             v18: adx_thresh=18  (loose — wider RANGING definition)

             Runs: baseline, grid-alone, grid+DC for each variant

  Phase 2  Rolling windows (10 × 6-month): same 3 variants, grid-alone only

  Phase 3  100-seed random benchmark for v15 (production candidate)

Ideas tried: 3 ADX variants (v12/v15/v18) × 2 combo variants (alone/+DC) = 6 ideas

Deployment gate
───────────────
  Pass: OOS ≥6/10 ✓, rolling ≥50% ✓, plateau stable ✓, seed median ≥7/10 ✓
  Fail: leave ENABLE_GRID_V2=False, record why

Run with:
    nohup python3 run_grid_v2_backtest.py >> gv2_backtest.log 2>&1 &
    echo $!   # note PID
"""

import json
import os
import sqlite3
import statistics
import sys
import threading
import time
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
IS_FROM  = "2020-01-01"
IS_TO    = "2023-12-31"
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-11-01"
TIMEOUT_MIN = 300
N_SEEDS     = 100
MAX_WORKERS = 8

ROLLING = [
    ("2024-01-01","2024-06-30"), ("2024-04-01","2024-09-30"),
    ("2024-07-01","2024-12-31"), ("2024-10-01","2025-03-31"),
    ("2025-01-01","2025-06-30"), ("2025-04-01","2025-09-30"),
    ("2025-07-01","2025-12-31"), ("2025-10-01","2026-03-31"),
    ("2026-01-01","2026-06-30"), ("2026-04-01","2026-10-31"),
]

# Plateau variants: (label, adx_thresh)
GV2_VARIANTS = [
    ("v12", 12.0),
    ("v15", 15.0),  # production candidate
    ("v18", 18.0),
]

# Fixed grid params (not being tuned — tune only adx_thresh)
GV2_N_CANDLES  = 4
GV2_MAX_LEVELS = 3
GV2_STEP_MULT  = 0.50
GV2_BUFFER_ATR = 0.75
GV2_COOLDOWN   = 4
GV2_EQUITY_PCT = 0.25

# DC filter params (v035 — pre-registered production value)
DC_LONG_CUT  = 0.35
DC_SHORT_CUT = 0.65

# ── Helpers ───────────────────────────────────────────────────────────────────
_lock     = threading.Lock()
_done     = [0]
_total_ref = [0]

DB = "asymmetric_demo.db"


def _already_done(email: str) -> bool:
    conn = sqlite3.connect(DB)
    cur  = conn.cursor()
    cur.execute(
        "SELECT 1 FROM backtest_runs WHERE email=? AND status='done' LIMIT 1",
        (email,)
    )
    found = cur.fetchone() is not None
    conn.close()
    return found


def _run(tag, coin, dfrom, dto,
         gv2=False, adx_thresh=15.0, dc_core=False,
         seed=None, variant=""):
    suffix = f"-s{seed}" if seed is not None else ""
    v_part = f"_{variant}" if variant else ""
    email  = f"gv2bt-{tag}{v_part}-{coin.lower()}{suffix}@asym-internal"

    if _already_done(email):
        with _lock:
            _done[0] += 1
            if _done[0] % 100 == 0 or _done[0] <= 4:
                print(f"  [{_done[0]}/{_total_ref[0]}] skip {tag}{v_part} {coin}{suffix}",
                      flush=True)
        return email, {}

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
        dc_long_cut=DC_LONG_CUT,
        dc_short_cut=DC_SHORT_CUT,
        enable_grid_v2=gv2,
        gv2_adx_thresh=adx_thresh,
        gv2_n_candles=GV2_N_CANDLES,
        gv2_max_levels=GV2_MAX_LEVELS,
        gv2_step_mult=GV2_STEP_MULT,
        gv2_buffer_atr=GV2_BUFFER_ATR,
        gv2_cooldown=GV2_COOLDOWN,
        gv2_equity_pct=GV2_EQUITY_PCT,
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
                    print(f"  [{_done[0]}/{_total_ref[0]}] ✓ {tag}{v_part} {coin}{suffix}",
                          flush=True)
            return email, rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total_ref[0]}] ✗ ERROR {tag}{v_part} {coin}",
                      flush=True)
            return email, {}

    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total_ref[0]}] ✗ TIMEOUT {tag}{v_part} {coin}",
              flush=True)
    return email, {}


def sharpe(res): return res.get("sharpe_ratio")      if res else None
def wr(res):     return res.get("win_rate")           if res else None
def ret(res):    return res.get("total_return_pct")   if res else None
def mdd(res):    return res.get("max_drawdown_pct")   if res else None


def main():
    t0 = time.time()
    tasks = []

    # ── Baseline (gv2=OFF, dc=OFF) — shared across all variants ──────────────
    for coin in COINS:
        tasks.append(("is_base",  coin, IS_FROM,  IS_TO,  False, 15.0, False, None, ""))
        tasks.append(("oos_base", coin, OOS_FROM, OOS_TO, False, 15.0, False, None, ""))
    for dfrom, dto in ROLLING:
        for coin in COINS:
            tasks.append((f"rbase_{dfrom[:7]}", coin, dfrom, dto, False, 15.0, False, None, ""))

    # ── Grid v2 alone — 3 ADX variants ───────────────────────────────────────
    for vlabel, adx_t in GV2_VARIANTS:
        for coin in COINS:
            tasks.append(("is_gv2",  coin, IS_FROM,  IS_TO,  True, adx_t, False, None, vlabel))
            tasks.append(("oos_gv2", coin, OOS_FROM, OOS_TO, True, adx_t, False, None, vlabel))
        for dfrom, dto in ROLLING:
            for coin in COINS:
                tasks.append((f"rgv2_{dfrom[:7]}", coin, dfrom, dto, True, adx_t, False, None, vlabel))

    # ── Grid v2 + DC (v035) — IS/OOS only for v15 (production candidate) ─────
    _, adx15 = GV2_VARIANTS[1]
    for coin in COINS:
        tasks.append(("is_gv2dc",  coin, IS_FROM,  IS_TO,  True, adx15, True, None, "v15dc"))
        tasks.append(("oos_gv2dc", coin, OOS_FROM, OOS_TO, True, adx15, True, None, "v15dc"))

    # ── 100-seed benchmark — v15 grid alone ───────────────────────────────────
    for coin in COINS:
        for seed in range(1, N_SEEDS + 1):
            tasks.append(("seed_gv2", coin, OOS_FROM, OOS_TO, True, adx15, False, seed, ""))

    _total_ref[0] = len(tasks)
    ideas = len(GV2_VARIANTS) * 2  # 3 ADX variants × (alone + dc)
    print(f"\nGrid v2 Full Backtest — {len(tasks)} runs, {ideas} ideas tried")
    print(f"  ADX variants: {', '.join(v for v,_ in GV2_VARIANTS)}")
    print(f"  IS {IS_FROM}→{IS_TO} · OOS {OOS_FROM}→{OOS_TO}")
    print(f"  10 rolling windows (grid-alone only) · {N_SEEDS} seeds (v15) · {MAX_WORKERS} workers\n",
          flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(_run, tag, coin, dfrom, dto, gv2, adx_t, dc_core, seed, variant):
            (tag, coin, variant)
            for tag, coin, dfrom, dto, gv2, adx_t, dc_core, seed, variant in tasks
        }
        for fut in as_completed(futs):
            tag, coin, variant = futs[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"  ✗ {tag} {variant} {coin} crashed: {e}", flush=True)

    elapsed = time.time() - t0

    # ── Load all results from DB ───────────────────────────────────────────────
    _db = sqlite3.connect(DB)
    _db.row_factory = sqlite3.Row
    _cur = _db.cursor()
    _cur.execute(
        "SELECT email, result_json FROM backtest_runs "
        "WHERE email LIKE 'gv2bt-%@asym-internal' AND status='done'"
    )
    _rows = {r["email"]: json.loads(r["result_json"] or "{}") for r in _cur.fetchall()}
    _db.close()

    def _get(tag, coin, variant="", seed=None):
        suffix = f"-s{seed}" if seed is not None else ""
        v_part = f"_{variant}" if variant else ""
        email  = f"gv2bt-{tag}{v_part}-{coin.lower()}{suffix}@asym-internal"
        return _rows.get(email, {})

    # ── Build report ──────────────────────────────────────────────────────────
    lines = []
    def w(line=""):
        print(line)
        lines.append(line)

    w("=" * 80)
    w("  GRID V2 RANGE MODULE — Full Validation Report")
    w(f"  Generated  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    w(f"  Baseline   : Option 3 (ENABLE_GRADE_B_ALL=True) + floor ON, no grid")
    w(f"  Ideas tried: {ideas} ({len(GV2_VARIANTS)} ADX variants × alone + with DC filter)")
    w(f"  Runs       : {len(tasks)}")
    w("=" * 80)

    # IS section
    for vlabel, adx_t in GV2_VARIANTS:
        w(f"\n── IS 2020–2023  [{vlabel}: adx_thresh={adx_t}] " + "─" * 36)
        w(f"  {'Coin':<10} {'Base':>8} {'GV2':>8} {'Δ':>7}  {'MDD base':>9} {'MDD gv2':>8}  {'OK?':>4}")
        w("  " + "─" * 58)
        is_wins = 0
        for coin in COINS:
            eb = _get("is_base", coin)
            eg = _get("is_gv2",  coin, vlabel)
            sb, sg = sharpe(eb), sharpe(eg)
            if sb is None or sg is None:
                w(f"  {coin:<10} {'—':>8} {'—':>8}")
                continue
            ok = "✓" if sg > sb else "✗"
            if sg > sb: is_wins += 1
            mb = mdd(eb) or 0.0
            mg = mdd(eg) or 0.0
            w(f"  {coin:<10} {sb:>8.3f} {sg:>8.3f} {sg-sb:>+7.3f}  "
              f"{mb:>8.1f}% {mg:>7.1f}%  {ok:>4}")
        w(f"  IS [{vlabel}]: GV2 wins {is_wins}/10")

    # OOS section
    w("\n" + "=" * 80)
    w("  OUT-OF-SAMPLE 2024–2026")
    w("=" * 80)
    oos_wins_all = {}
    for vlabel, adx_t in GV2_VARIANTS:
        w(f"\n── OOS 2024–2026  [{vlabel}: adx_thresh={adx_t}] " + "─" * 34)
        w(f"  {'Coin':<10} {'Base':>8} {'GV2':>8} {'Δ':>7}  {'MDD base':>9} {'MDD gv2':>8}  {'OK?':>4}")
        w("  " + "─" * 58)
        oos_wins = 0
        for coin in COINS:
            eb = _get("oos_base", coin)
            eg = _get("oos_gv2",  coin, vlabel)
            sb, sg = sharpe(eb), sharpe(eg)
            if sb is None or sg is None:
                w(f"  {coin:<10} {'—':>8} {'—':>8}")
                continue
            ok = "✓" if sg > sb else "✗"
            if sg > sb: oos_wins += 1
            mb = mdd(eb) or 0.0
            mg = mdd(eg) or 0.0
            w(f"  {coin:<10} {sb:>8.3f} {sg:>8.3f} {sg-sb:>+7.3f}  "
              f"{mb:>8.1f}% {mg:>7.1f}%  {ok:>4}")
        oos_wins_all[vlabel] = oos_wins
        w(f"  OOS [{vlabel}]: GV2 wins {oos_wins}/10")

    # Plateau summary
    w("\n── Plateau sensitivity (OOS wins) ──────────────────────────────────────")
    for vlabel, adx_t in GV2_VARIANTS:
        mark = " ← production candidate" if vlabel == "v15" else ""
        w(f"  {vlabel} (adx≤{adx_t}) : {oos_wins_all.get(vlabel, '?')}/10{mark}")
    plateau_ok = all(oos_wins_all.get(v, 0) >= 5 for v, _ in GV2_VARIANTS)
    w(f"  Plateau {'✓ stable' if plateau_ok else '✗ knife-edge — do NOT deploy'}")

    # Grid v2 + DC section
    w("\n── Grid v2 + DC filter (v15 only) ──────────────────────────────────────")
    w(f"  {'Coin':<10} {'Base':>8} {'GV2':>8} {'GV2+DC':>8}  {'OK?':>4}")
    w("  " + "─" * 48)
    dc_wins = 0
    for coin in COINS:
        eb  = _get("oos_base",  coin)
        eg  = _get("oos_gv2",   coin, "v15")
        edc = _get("oos_gv2dc", coin, "v15dc")
        sb, sg, sdc = sharpe(eb), sharpe(eg), sharpe(edc)
        if sb is None:
            w(f"  {coin:<10} {'—':>8}")
            continue
        ok = "✓" if (sdc or 0) > sb else "✗"
        if (sdc or 0) > sb: dc_wins += 1
        w(f"  {coin:<10} {sb:>8.3f} {sg or 0:>8.3f} {sdc or 0:>8.3f}  {ok:>4}")
    w(f"  OOS [v15+DC]: DC-combo wins {dc_wins}/10")

    # Rolling section
    w("\n── Rolling Windows (6-month, grid-alone) ────────────────────────────────")
    roll_pcts = {}
    for vlabel, adx_t in GV2_VARIANTS:
        total_gv2 = total_base = 0
        for dfrom, dto in ROLLING:
            win = dfrom[:7]
            for coin in COINS:
                sg = sharpe(_get(f"rgv2_{win}", coin, vlabel))
                sb = sharpe(_get(f"rbase_{win}", coin))
                if sg is None or sb is None: continue
                if sg > sb:   total_gv2  += 1
                elif sb > sg: total_base += 1
        pct = total_gv2 / max(1, total_gv2 + total_base) * 100
        roll_pcts[vlabel] = pct
        mark = " ← production candidate" if vlabel == "v15" else ""
        w(f"  {vlabel}: GV2 {total_gv2} vs Base {total_base} ({pct:.0f}% GV2 wins){mark}")

    # 100-seed benchmark
    w("\n── 100-seed Random Benchmark (v15, OOS 2024–2026) ──────────────────────")
    w(f"  {'Coin':<10} {'GV2 Sharpe':>11}  {'Rand P50':>9} {'Rand P90':>9}  {'Pctile':>7}")
    w("  " + "─" * 58)
    seed_pass = 0
    for coin in COINS:
        eg = _get("oos_gv2", coin, "v15")
        sg = sharpe(eg)
        seed_sharpes = [sharpe(_get("seed_gv2", coin, seed=s)) for s in range(1, N_SEEDS + 1)]
        seed_sharpes = [x for x in seed_sharpes if x is not None]
        if not seed_sharpes or sg is None:
            w(f"  {coin:<10}  {'—':>11}  {'—':>9} {'—':>9}  {'—':>7}")
            continue
        ss = sorted(seed_sharpes)
        p50 = statistics.median(ss)
        p90 = ss[int(len(ss) * 0.90)]
        pct = sum(1 for x in ss if x < sg) / len(ss) * 100
        ok  = "✓" if sg > p50 else "✗"
        if sg > p50: seed_pass += 1
        w(f"  {coin:<10}  {sg:>11.3f}  {p50:>9.3f} {p90:>9.3f}  {pct:>6.1f}%  {ok}")
    w(f"  Seed gate: {seed_pass}/10 coins beat random median (need ≥7)")

    # Final verdict
    w("\n" + "=" * 80)
    w("  VERDICT")
    w("=" * 80)
    best_v = max(oos_wins_all, key=lambda v: oos_wins_all[v]) if oos_wins_all else "v15"
    best_wins = oos_wins_all.get(best_v, 0)
    best_roll = roll_pcts.get(best_v, 0)
    w(f"  Best variant: {best_v} ({best_wins}/10 OOS wins, {best_roll:.0f}% rolling wins)")
    w(f"  OOS gate    : {'✓ PASS' if best_wins >= 6 else '✗ FAIL'} (need ≥6/10)")
    w(f"  Rolling gate: {'✓ PASS' if best_roll >= 50 else '✗ FAIL'} (need ≥50%)")
    w(f"  Plateau gate: {'✓ PASS' if plateau_ok else '✗ FAIL'} (all variants ≥5/10)")
    w(f"  Seed gate   : {'✓ PASS' if seed_pass >= 7 else '✗ FAIL'} (need ≥7/10)")

    all_pass = best_wins >= 6 and best_roll >= 50 and plateau_ok and seed_pass >= 7
    w(f"\n  GRID V2: {'✓ ACCEPTED — deploy behind ENABLE_GRID_V2=False, demo first' if all_pass else '✗ FAILED — ENABLE_GRID_V2 stays OFF'}")
    w(f"\n  Elapsed: {elapsed/60:.1f} min")
    w("=" * 80)

    report_txt  = "gv2_full_report.txt"
    report_json = "gv2_full_results.json"

    with open(report_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {report_txt}")

    # JSON export: all OOS Sharpes for external comparison
    _json_out = {}
    for vlabel, _ in GV2_VARIANTS:
        _json_out[vlabel] = {}
        for coin in COINS:
            eg = _get("oos_gv2", coin, vlabel)
            eb = _get("oos_base", coin)
            _json_out[vlabel][coin] = {"gv2": sharpe(eg), "base": sharpe(eb)}
    with open(report_json, "w") as f:
        json.dump(_json_out, f, indent=2)
    print(f"JSON  written to {report_json}")


if __name__ == "__main__":
    main()
