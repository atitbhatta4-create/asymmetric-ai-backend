#!/usr/bin/env python3
"""
run_full_benchmark.py
─────────────────────
Full benchmark: all modes × styles × coins with the final deployed config.

Config: Option 3 (ENABLE_GRADE_B_ALL=True) + floor ON + DC filter ON + 4h MR ON
Period: OOS 2024-01-01 → 2026-11-01  (live-validated window)

Modes  : ULTRA_SAFE, SAFE, NORMAL, MINI_ASYM, AGGRESSIVE  (5)
Styles : SCALP, DAY_TRADE, SWING                           (3)
Coins  : 10 (candle data available; add more coins to DB to expand)

Total  : 5 × 3 × 10 = 150 runs

NOTE: Run AFTER ENABLE_DC_FILTER and ENABLE_MR_4H_SIGNAL are both live.
      These results are for the Help Center /help/best-settings page.

      To add more coins: populate backtest_candles table for each new symbol,
      then add the symbol to COINS list below.

Usage:
  nohup python3 run_full_benchmark.py >> full_benchmark.log 2>&1 &
  echo $!

Output:
  full_benchmark_report.txt
  full_benchmark_results.json
"""

import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from backtester import start_backtest, get_run, init_backtest_tables
from database import db_conn
init_backtest_tables()

# ── Config ────────────────────────────────────────────────────────────────────
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "NEARUSDT",
    # Add more coins here once candle data is populated in DB:
    # "DOTUSDT", "MATICUSDT", "LTCUSDT", "UNIUSDT", "TRXUSDT",
    # "ATOMUSDT", "OPUSDT", "APTUSDT", "SHIBUSDT", "SANDUSDT",
]

MODES  = ["ULTRA_SAFE", "SAFE", "NORMAL", "MINI_ASYM", "AGGRESSIVE"]
STYLES = ["SCALP", "DAY_TRADE", "SWING"]

EXCHANGE = "bybit"
START_EQ = 1000.0
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-11-01"
TIMEOUT_MIN = 300
MAX_WORKERS = 1   # 1 worker to stay under Render 512MB memory limit

# ── Helpers ───────────────────────────────────────────────────────────────────
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


def _run(mode, style, coin):
    email = f"bench-{mode.lower()}-{style.lower()}-{coin.lower()}@asym-internal"

    cached = _fetch_done(email)
    if cached is not None:
        with _lock:
            _done[0] += 1
            if _done[0] % 30 == 0 or _done[0] <= 3:
                print(f"  [{_done[0]}/{_total_ref[0]}] skip {mode} {style} {coin}", flush=True)
        return email, cached

    try:
        run_id = start_backtest(
            email=email, symbol=coin, mode=mode, style=style,
            exchange=EXCHANGE, date_from=OOS_FROM, date_to=OOS_TO,
            start_equity=START_EQ,
            enable_t16=True, enable_t18=False, enable_s4=False,
            enable_mr_4h_gate=False,
            enable_mr_signal=False,       # 4h MR replaces 1h MR
            enable_mr_4h_signal=True,     # 4h MR ON
            enable_regime_gate=False,
            enable_e2=False, enable_step2_score=False,
            enable_grade_b_all=True,
            enable_floor=True,
            enable_dc_core=True,          # DC filter ON
            dc_long_cut=0.35, dc_short_cut=0.65,
        )
    except ValueError as e:
        with _lock:
            _done[0] += 1
            print(f"  [{_done[0]}/{_total_ref[0]}] ✗ INVALID {mode} {style} {coin}: {e}", flush=True)
        return email, {}

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
                if _done[0] % 10 == 0 or _done[0] <= 5:
                    print(f"  [{_done[0]}/{_total_ref[0]}] ✓ {mode} {style} {coin}", flush=True)
            return email, rec.get("results") or {}
        if st == "error":
            with _lock:
                _done[0] += 1
                print(f"  [{_done[0]}/{_total_ref[0]}] ✗ ERROR {mode} {style} {coin}", flush=True)
            return email, {}

    with _lock:
        _done[0] += 1
        print(f"  [{_done[0]}/{_total_ref[0]}] ✗ TIMEOUT {mode} {style} {coin}", flush=True)
    return email, {}


def _get(results, mode, style, coin):
    email = f"bench-{mode.lower()}-{style.lower()}-{coin.lower()}@asym-internal"
    return results.get(email) or {}


def fmt_sh(v): return f"{v:+.3f}" if v is not None else "   N/A"
def fmt_ret(v): return f"{v:+.1f}%" if v is not None else "  N/A"


def main():
    t0 = time.time()

    tasks = [(mode, style, coin)
             for mode in MODES
             for style in STYLES
             for coin in COINS]

    _total_ref[0] = len(tasks)

    print(f"\n{'='*72}")
    print(f"  FULL BENCHMARK — All Modes × Styles × Coins")
    print(f"  Config : Option3 + DC filter + 4h MR + floor ON")
    print(f"  Period : {OOS_FROM} → {OOS_TO}")
    print(f"  Runs   : {len(tasks)} ({len(MODES)} modes × {len(STYLES)} styles × {len(COINS)} coins)")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*72}\n")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run, *t): t for t in tasks}
        for fut in as_completed(futs):
            email, res = fut.result()
            results[email] = res

    elapsed = (time.time() - t0) / 60

    lines = []
    def pr(s=""): lines.append(s); print(s)

    pr(f"{'='*72}")
    pr(f"  FULL BENCHMARK REPORT")
    pr(f"  Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    pr(f"  Config    : Option3 + DC filter + 4h MR + floor ON")
    pr(f"  Period    : {OOS_FROM} → {OOS_TO}")
    pr(f"  Elapsed   : {elapsed:.1f} min")
    pr(f"{'='*72}")

    # ── Per mode/style summary ────────────────────────────────────────────────
    summary_grid = {}
    for mode in MODES:
        for style in STYLES:
            sharpes = []
            returns = []
            wrs     = []
            for coin in COINS:
                r = _get(results, mode, style, coin)
                sh = r.get("sharpe_ratio")
                rt = r.get("total_return_pct")
                wr = r.get("win_rate")
                if sh is not None: sharpes.append(sh)
                if rt is not None: returns.append(rt)
                if wr is not None: wrs.append(wr)
            summary_grid[(mode, style)] = {
                "median_sharpe": sorted(sharpes)[len(sharpes)//2] if sharpes else None,
                "median_return": sorted(returns)[len(returns)//2] if returns else None,
                "median_wr":     sorted(wrs)[len(wrs)//2]         if wrs     else None,
                "coins_positive": sum(1 for s in sharpes if s > 0),
                "n": len(sharpes),
            }

    pr(f"\n── MODE × STYLE SUMMARY (median across {len(COINS)} coins) {'─'*22}")
    pr(f"  {'Mode':<13} {'Style':<12} {'Med Sharpe':>11} {'Med Return':>11} {'Med WR':>7} {'Coins>0':>8}")
    pr(f"  {'─'*68}")
    for mode in MODES:
        for style in STYLES:
            g = summary_grid[(mode, style)]
            sh = fmt_sh(g["median_sharpe"])
            rt = fmt_ret(g["median_return"])
            wr = f"{g['median_wr']:.1f}%" if g["median_wr"] else "  N/A"
            pos = f"{g['coins_positive']}/{g['n']}"
            pr(f"  {mode:<13} {style:<12} {sh:>11} {rt:>11} {wr:>7} {pos:>8}")

    # ── Per coin detail (MINI_ASYM DAY_TRADE only — the live config) ──────────
    pr(f"\n── MINI_ASYM DAY_TRADE — Per Coin Detail {'─'*30}")
    pr(f"  {'Coin':<12} {'Sharpe':>8} {'Return':>9} {'WR':>7} {'MDD':>8} {'Trades':>7}")
    pr(f"  {'─'*60}")
    for coin in COINS:
        r = _get(results, "MINI_ASYM", "DAY_TRADE", coin)
        sh  = r.get("sharpe_ratio")
        rt  = r.get("total_return_pct")
        wr  = r.get("win_rate")
        mdd = r.get("max_drawdown_pct")
        cnt = r.get("trade_count")
        pr(f"  {coin:<12} {fmt_sh(sh):>8} {fmt_ret(rt):>9} "
           f"{f'{wr:.1f}%' if wr else 'N/A':>7} "
           f"{f'{mdd:.1f}%' if mdd else 'N/A':>8} "
           f"{cnt if cnt else 'N/A':>7}")

    # ── Best combos ───────────────────────────────────────────────────────────
    pr(f"\n── TOP 5 MODE/STYLE COMBINATIONS (by median Sharpe) {'─'*20}")
    ranked = sorted(
        [(k, v) for k, v in summary_grid.items() if v["median_sharpe"] is not None],
        key=lambda x: x[1]["median_sharpe"], reverse=True
    )[:5]
    for (mode, style), g in ranked:
        pr(f"  {mode:<13} {style:<12}  Sharpe {fmt_sh(g['median_sharpe'])}  "
           f"Return {fmt_ret(g['median_return'])}  {g['coins_positive']}/{g['n']} positive")

    pr(f"\n{'='*72}")

    report_path = os.path.join(os.path.dirname(__file__), "full_benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport → {report_path}")

    json_path = os.path.join(os.path.dirname(__file__), "full_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "config": "Option3+DCFilter+4hMR+floor",
            "period": f"{OOS_FROM}→{OOS_TO}",
            "coins": COINS, "modes": MODES, "styles": STYLES,
            "summary_grid": {f"{k[0]}_{k[1]}": v for k, v in summary_grid.items()},
        }, f, indent=2)
    print(f"JSON  → {json_path}")


if __name__ == "__main__":
    main()
