"""
run_batch_optimizer.py — Full parameter grid search across all modes, styles, and coins.

Usage (Render shell):
    python run_batch_optimizer.py

Scope : 5 modes × 2 styles × 10 coins = 100 optimizer runs
        Each run tests 300 parameter combos (ADX / score / SL / TP offsets)
        TP range expanded to 4.0× (was 2.5×) targeting 40-60% yearly return.
Period: 2020-01-01 → 2026-05-31
Output: Best params per combo printed to console + saved to optimizer_runs/optimizer_results DB tables
        At the end: copy the printed coin_params recommendations into coin_params.py,
        then run python run_comprehensive_report.py for the updated full PDF.

Estimated time: 8-12 hours on Render (300 combos × 9× speedup ≈ same as old 135). Safe to leave running overnight.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

from database import db_conn, USING_PG
from optimizer import _run_opt_worker, _parse_ms, OPT_GRID

# ── Config ─────────────────────────────────────────────────────────────────────
COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]
MODES  = ["ULTRA_SAFE", "SAFE", "NORMAL", "MINI_ASYM", "AGGRESSIVE"]
STYLES = ["DAY_TRADE", "SWING"]

DATE_FROM    = "2020-01-01"
DATE_TO      = "2026-05-31"
START_EQUITY = 1000.0
EXCHANGE     = "bybit"
ADMIN_EMAIL  = "mes29571@gmail.com"

START_MS = _parse_ms(DATE_FROM)
END_MS   = _parse_ms(DATE_TO, end_of_day=True)

TOTAL = len(MODES) * len(STYLES) * len(COINS)
COMBOS_PER_RUN = (
    len(OPT_GRID["adx_delta"]) *
    len(OPT_GRID["score_delta"]) *
    len(OPT_GRID["sl_mult"]) *
    len(OPT_GRID["tp_mult"])
)

TF_FROM_STYLE = {"SCALP": "15m", "DAY_TRADE": "1h", "SWING": "4h"}


def _create_run_record(symbol: str, mode: str, style: str) -> str:
    run_id = str(uuid.uuid4())
    tf = TF_FROM_STYLE.get(style, "1h")
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO optimizer_runs"
            "(run_id,email,symbol,tf,mode,style,exchange,date_from,date_to,"
            "start_equity,status,progress,total_combos,done_combos,created_at)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            if USING_PG else
            "INSERT INTO optimizer_runs"
            "(run_id,email,symbol,tf,mode,style,exchange,date_from,date_to,"
            "start_equity,status,progress,total_combos,done_combos,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, ADMIN_EMAIL, symbol, tf, mode, style, EXCHANGE,
             DATE_FROM, DATE_TO, START_EQUITY,
             "pending", 0, COMBOS_PER_RUN, 0,
             datetime.utcnow().isoformat()),
        )
        conn.commit()
    return run_id


def _get_best_result(run_id: str) -> dict | None:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT params_json, total_trades, win_rate, total_return, "
            "sharpe_ratio, max_drawdown FROM optimizer_results "
            "WHERE run_id=%s AND passed_filters=1 "
            "ORDER BY sharpe_ratio DESC LIMIT 1",
            (run_id,),
        )
        return cur.fetchone()


def _print_separator(char="─", width=72):
    print(char * width)


def main():
    grand_t0 = time.time()

    _print_separator("═")
    print("  ASYMMETRIC AI — FULL BATCH PARAMETER OPTIMIZER")
    _print_separator("═")
    print(f"  Period  : {DATE_FROM} → {DATE_TO}")
    print(f"  Modes   : {' | '.join(MODES)}")
    print(f"  Styles  : {' | '.join(STYLES)}")
    print(f"  Coins   : {', '.join(c.replace('USDT','') for c in COINS)}")
    print(f"  Runs    : {TOTAL} optimizer runs × {COMBOS_PER_RUN} combos = {TOTAL*COMBOS_PER_RUN:,} total simulations")
    print(f"  Started : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Est.    : 8-12 hours — safe to leave running")
    _print_separator("═")
    print()

    done = 0
    results_log: list[dict] = []

    for mode in MODES:
        for style in STYLES:
            _print_separator()
            print(f"  MODE: {mode}  |  STYLE: {style}")
            _print_separator()

            for sym in COINS:
                done += 1
                t0 = time.time()
                elapsed_total = round((time.time() - grand_t0) / 60, 1)
                print(f"\n[{done:>3}/{TOTAL}] {sym:<10} {mode:<12} {style}  "
                      f"(total elapsed: {elapsed_total} min)")

                try:
                    run_id = _create_run_record(sym, mode, style)
                    _run_opt_worker(
                        run_id=run_id,
                        email=ADMIN_EMAIL,
                        symbol=sym,
                        mode=mode,
                        style=style,
                        exchange=EXCHANGE,
                        start_ms=START_MS,
                        end_ms=END_MS,
                        start_equity=START_EQUITY,
                    )

                    elapsed = round(time.time() - t0, 1)
                    best = _get_best_result(run_id)

                    if best:
                        params = json.loads(best["params_json"])
                        print(f"        ✓ Done in {elapsed}s  |  "
                              f"Sharpe: {best['sharpe_ratio']:.3f}  |  "
                              f"Return: {best['total_return']*100:.1f}%  |  "
                              f"WR: {best['win_rate']*100:.1f}%  |  "
                              f"DD: {best['max_drawdown']*100:.1f}%  |  "
                              f"Trades: {best['total_trades']}")
                        print(f"        Best params: adx_delta={params.get('adx_delta'):+.0f}  "
                              f"score_delta={params.get('score_delta'):+.2f}  "
                              f"sl_mult={params.get('sl_mult'):.1f}  "
                              f"tp_mult={params.get('tp_mult'):.1f}")
                        results_log.append({
                            "symbol": sym, "mode": mode, "style": style,
                            "run_id": run_id,
                            "sharpe": round(best["sharpe_ratio"], 3),
                            "total_return": f"{best['total_return']*100:.1f}%",
                            "win_rate": f"{best['win_rate']*100:.1f}%",
                            "max_drawdown": f"{best['max_drawdown']*100:.1f}%",
                            "total_trades": best["total_trades"],
                            "best_params": params,
                        })
                    else:
                        print(f"        ⚠ Done in {elapsed}s — no results passed filters (too few trades)")
                        results_log.append({
                            "symbol": sym, "mode": mode, "style": style,
                            "run_id": run_id, "error": "no results passed filters",
                        })

                except Exception as exc:
                    elapsed = round(time.time() - t0, 1)
                    print(f"        ✗ FAILED in {elapsed}s: {exc}")
                    results_log.append({
                        "symbol": sym, "mode": mode, "style": style,
                        "error": str(exc),
                    })

    # ── Final summary ──────────────────────────────────────────────────────────
    grand_elapsed = round((time.time() - grand_t0) / 60, 1)
    print()
    _print_separator("═")
    print(f"  ALL DONE — {grand_elapsed} minutes total")
    _print_separator("═")
    print()
    print("  BEST RESULTS PER COMBINATION (sorted by Sharpe):")
    print()

    passed = [r for r in results_log if "error" not in r]
    passed.sort(key=lambda x: x["sharpe"], reverse=True)

    for r in passed[:30]:  # top 30
        p = r["best_params"]
        print(f"  {r['symbol']:<10} {r['mode']:<12} {r['style']:<10}  "
              f"Sharpe={r['sharpe']:.3f}  Return={r['total_return']}  "
              f"WR={r['win_rate']}  DD={r['max_drawdown']}  "
              f"tp={p.get('tp_mult'):.1f}  sl={p.get('sl_mult'):.1f}  "
              f"adx_d={p.get('adx_delta'):+.0f}  score_d={p.get('score_delta'):+.2f}")

    print()
    _print_separator()
    print("  MINI_ASYM + DAY_TRADE (live mode) — recommended coin_params.py updates:")
    _print_separator()

    mini_dt = [r for r in passed
               if r["mode"] == "MINI_ASYM" and r["style"] == "DAY_TRADE"]
    for r in sorted(mini_dt, key=lambda x: x["sharpe"], reverse=True):
        p = r["best_params"]
        print(f"  {r['symbol']:<10}  tp_multiplier={p.get('tp_mult'):.1f}  "
              f"sl_multiplier={p.get('sl_mult'):.1f}  "
              f"adx_delta={p.get('adx_delta'):+.0f}  "
              f"score_delta={p.get('score_delta'):+.2f}  "
              f"→ Sharpe {r['sharpe']:.3f}")

    print()
    print("  Next step: update coin_params.py with the above values,")
    print("  then run:  python run_comprehensive_report.py")
    print()
    _print_separator("═")
    print(f"  Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    _print_separator("═")


if __name__ == "__main__":
    main()
