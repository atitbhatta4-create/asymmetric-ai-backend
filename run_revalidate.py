#!/usr/bin/env python3
"""
run_revalidate.py
─────────────────
Quick re-validation of Option 3 baseline + DC v035 with the
volume-bypass backtester fix applied 2026-11-12.

Runs 20 OOS jobs only (10 coins × baseline + DC v035).
~5-10 min on Render.  Confirms the fix does not flip any coin result.

Reference (original dc_core_full_backtest run 2026-11-11):
  DC v035 OOS: 7/10 coins beat baseline  (AVAX / DOGE / NEAR = 3 losers)
  Gate: ≥6/10 — PASSED

Usage:
    python3 run_revalidate.py
    nohup python3 run_revalidate.py >> revalidate.log 2>&1 &
"""

import os, sys, time, json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
TIMEOUT  = 25 * 60    # 25 min per run
# Original OOS results from dc_core_full_report.txt (pre-volume-bypass-fix)
REF_DC_OOS_WINNERS = {"ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
                      "ADAUSDT","LINKUSDT","BTCUSDT"}
REF_DC_OOS_LOSERS  = {"AVAXUSDT","DOGEUSDT","NEARUSDT"}

SEP  = "=" * 68
SEP2 = "-" * 68


def _run_one(tag: str, coin: str, dc: bool) -> dict:
    # No caching — 20 runs take ~10 min, not worth the complexity on Render PG
    email  = f"reval-{tag}-{coin.lower()}@asym-internal"
    run_id = start_backtest(
        email=email, symbol=coin, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=OOS_FROM, date_to=OOS_TO,
        start_equity=START_EQ,
        enable_t16=True, enable_t18=False, enable_s4=False,
        enable_mr_4h_gate=False, enable_regime_gate=False,
        enable_e2=False, enable_step2_score=False,
        enable_grade_b_all=True, enable_floor=True,
        enable_dc_core=dc,
        dc_long_cut=0.35, dc_short_cut=0.65,
    )

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        time.sleep(5)
        rec = get_run(run_id)
        if not rec:
            continue
        st = rec.get("status", "")
        if st == "done":
            print(f"  ✓ {tag} {coin}", flush=True)
            return rec.get("results") or {}
        if st == "error":
            print(f"  ✗ ERROR {tag} {coin} — {rec.get('error','?')}", flush=True)
            return {}
    print(f"  ✗ TIMEOUT {tag} {coin}", flush=True)
    return {}


def fmt(v):
    return f"{v:+.3f}" if v is not None else "   N/A"


def main():
    t0 = time.time()
    print(f"\n{SEP}")
    print("  REVALIDATE — Option 3 + DC v035 with volume-bypass fix")
    print(f"  OOS: {OOS_FROM} → {OOS_TO}   |   20 runs (10 coins × 2 variants)")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(SEP)
    print()

    base_res = {}
    dc_res   = {}
    total    = len(COINS) * 2
    done     = 0

    for coin in COINS:
        done += 1
        print(f"  [{done}/{total}] baseline {coin}", flush=True)
        base_res[coin] = _run_one("base", coin, dc=False)
        done += 1
        print(f"  [{done}/{total}] dc-v035  {coin}", flush=True)
        dc_res[coin]   = _run_one("dc035", coin, dc=True)

    elapsed = (time.time() - t0) / 60

    # ── OOS comparison table ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  OOS RESULTS  (Option 3 baseline  vs  DC v035, floor ON)")
    print(f"  {'Coin':<10}  {'Base Sh':>8}  {'DC Sh':>8}  {'DC>Base?':>9}  "
          f"{'Ref':>8}  {'Regressed?':>10}")
    print(SEP2)

    dc_wins     = 0
    regressions = []
    results_out = []

    for coin in COINS:
        b = base_res.get(coin, {})
        d = dc_res.get(coin, {})
        b_sh = b.get("sharpe_ratio")
        d_sh = d.get("sharpe_ratio")

        if b_sh is not None and d_sh is not None:
            dc_beat = d_sh > b_sh
        else:
            dc_beat = None

        if dc_beat:
            dc_wins += 1

        ref_win  = coin in REF_DC_OOS_WINNERS
        marker   = "✓" if dc_beat else ("✗" if dc_beat is False else "?")
        ref_str  = "WIN" if ref_win else "LOSS"

        regressed = dc_beat is not None and dc_beat != ref_win
        reg_str   = "⚠ YES" if regressed else ""
        if regressed:
            regressions.append(
                f"  {coin}: was {ref_str} → now {'WIN' if dc_beat else 'LOSS'}  "
                f"(base={fmt(b_sh)}, dc={fmt(d_sh)})"
            )

        print(f"  {marker} {coin:<10}  {fmt(b_sh):>8}  {fmt(d_sh):>8}  "
              f"{'YES' if dc_beat else 'NO':>9}  {ref_str:>8}  {reg_str:>10}")

        results_out.append({
            "coin": coin, "base_sharpe": b_sh, "dc_sharpe": d_sh,
            "dc_beats_base": dc_beat, "ref_was_win": ref_win,
            "regressed": regressed,
        })

    print(SEP2)

    # ── Gates ─────────────────────────────────────────────────────────────────
    gate_oos = dc_wins >= 6
    print(f"\n  Gate 1 (OOS ≥6/10):   {'PASS ✓' if gate_oos else 'FAIL ✗'}  "
          f"({dc_wins}/10 — orig: 7/10)")
    print(f"  Regressions:          {len(regressions)}")
    print()

    if regressions:
        print("  ⚠  Coin(s) changed result vs original run:")
        for r in regressions:
            print(r)
        print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(SEP)
    print("  VERDICT")
    print(SEP2)
    if gate_oos and not regressions:
        print("  ✓ Volume bypass fix confirmed safe — no regressions.")
        print("  ✓ DC v035 still passes OOS gate with corrected backtester.")
        print("  ✓ Parity gate (≥55% after Nov 18) is the only remaining blocker.")
    elif not gate_oos:
        print("  ✗ OOS gate FAILED — volume bypass fix changed results.")
        print("    Investigate before DC filter deploy.")
    else:
        print(f"  ⚠  OOS gate PASSES but {len(regressions)} regression(s) detected.")
        print("    Review regressions before DC filter deploy.")

    print(f"\n  Elapsed: {elapsed:.1f} min")
    print(SEP)

    # Save JSON
    out_path = "revalidate_results.json"
    with open(out_path, "w") as jf:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "oos_from": OOS_FROM, "oos_to": OOS_TO,
            "dc_wins": dc_wins, "gate_oos_pass": gate_oos,
            "regression_count": len(regressions),
            "coins": results_out,
        }, jf, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
