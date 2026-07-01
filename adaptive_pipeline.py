"""
adaptive_pipeline.py — Rolling walk-forward validation system.

Phase 1: Optimise on TRAIN period, validate on TEST period, report degradation.
Monthly auto-run: re-runs on first day of each month with a rolling 2yr window.
Degradation alert: Telegram + AI log if live performance diverges >25% from backtest.

Coins:  BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, NEAR
Mode:   MINI_ASYM  |  Style: DAY_TRADE  (Phase 1 only)
"""
from __future__ import annotations

import itertools
import json
import os
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from backtester import (
    ATR_BASELINE,
    MODE_PRESETS,
    WARMUP_CANDLES,
    _calc_metrics,
    _ensure_candles,
    _get_aligned_slice,
    _session_quality_for_ts,
    _sim_check_exit,
)
from config import HIGHER_TF_MAP, TRADE_STYLE_PARAMS
from database import db_conn, serial_pk, USING_PG
from indicators import MODE_SIGNAL_PARAMS, _compute_signal_layers
from notifications import tg_alert

# ── Constants ──────────────────────────────────────────────────────────────────

PIPELINE_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                  "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT"]

DEFAULT_MODE  = "MINI_ASYM"
DEFAULT_STYLE = "DAY_TRADE"

# Walk-forward window sizes (years)
TRAIN_YEARS = 2
TEST_YEARS  = 2

# Degradation threshold: flag if live Sharpe is X% below backtested Sharpe
DEGRADE_THRESHOLD = 0.25   # 25%

# Minimum trades in a period to trust the metrics
MIN_TRADES_TRAIN = 15
MIN_TRADES_TEST  = 5

# Parameter grid (same as optimizer.py — reuse the same search space)
OPT_GRID = {
    "adx_delta":   [-4, -2, 0, 2, 4],
    "score_delta": [-0.04, 0.0, 0.04],
    "sl_mult":     [0.8, 1.0, 1.2],
    "tp_mult":     [1.5, 2.0, 2.5],
}

# Active pipeline runs in memory
_active: Dict[str, Dict] = {}
_active_lock = threading.Lock()

# Monthly scheduler state
_scheduler_thread: Optional[threading.Thread] = None
_scheduler_stop   = threading.Event()


# ── DB tables ──────────────────────────────────────────────────────────────────

def init_pipeline_tables() -> None:
    """Create pipeline tables on startup. Safe to call repeatedly."""
    pk = serial_pk()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id       TEXT PRIMARY KEY,
                triggered_by TEXT NOT NULL,
                mode         TEXT NOT NULL,
                style        TEXT NOT NULL,
                train_from   TEXT NOT NULL,
                train_to     TEXT NOT NULL,
                test_from    TEXT NOT NULL,
                test_to      TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'running',
                coins_total  INTEGER DEFAULT 0,
                coins_done   INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL,
                completed_at TEXT,
                error        TEXT
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pipeline_coin_results (
                id              {pk},
                run_id          TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                train_trades    INTEGER,
                train_win_rate  REAL,
                train_return    REAL,
                train_max_dd    REAL,
                train_sharpe    REAL,
                test_trades     INTEGER,
                test_win_rate   REAL,
                test_return     REAL,
                test_max_dd     REAL,
                test_sharpe     REAL,
                degrade_sharpe  REAL,
                degrade_wr      REAL,
                degrade_return  REAL,
                best_params     TEXT,
                verdict         TEXT,
                created_at      TEXT NOT NULL
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pipeline_alerts (
                id          {pk},
                symbol      TEXT NOT NULL,
                mode        TEXT NOT NULL,
                style       TEXT NOT NULL,
                metric      TEXT NOT NULL,
                expected    REAL,
                actual      REAL,
                degrade_pct REAL,
                fired_at    TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_coin_run"
            " ON pipeline_coin_results(run_id, symbol)"
        )
        # Mark any stuck runs from a previous server crash
        cur.execute(
            "UPDATE pipeline_runs SET status=%s, error=%s, completed_at=%s"
            " WHERE status='running'",
            ("error", "Server restarted", datetime.utcnow().isoformat()),
        )
        conn.commit()


# ── Core: optimise on train, backtest on test ──────────────────────────────────

def _run_one_coin(
    symbol: str,
    mode: str,
    style: str,
    train_start_ms: int,
    train_end_ms: int,
    test_start_ms: int,
    test_end_ms: int,
    exchange: str = "bybit",
    start_equity: float = 1000.0,
) -> Dict:
    """
    1. Fetch train + test candles.
    2. Grid-search params on train period — pick best Sharpe.
    3. Replay test period with those exact params (no refitting).
    4. Return both sets of metrics + degradation %.
    """
    tf        = TRADE_STYLE_PARAMS[style]["tf"]
    higher_tf = HIGHER_TF_MAP.get(tf, tf)
    st        = TRADE_STYLE_PARAMS[style]
    c         = MODE_PRESETS[mode]

    # Fetch candles for both periods
    train_main = _ensure_candles(symbol, tf,        train_start_ms, train_end_ms)
    train_high = _ensure_candles(symbol, higher_tf, train_start_ms, train_end_ms)
    test_main  = _ensure_candles(symbol, tf,        test_start_ms,  test_end_ms)
    test_high  = _ensure_candles(symbol, higher_tf, test_start_ms,  test_end_ms)

    if len(train_main) < WARMUP_CANDLES + MIN_TRADES_TRAIN * 20:
        return {"error": f"Not enough train candles ({len(train_main)}) for {symbol}"}
    if len(test_main) < WARMUP_CANDLES + MIN_TRADES_TEST * 20:
        return {"error": f"Not enough test candles ({len(test_main)}) for {symbol}"}

    # ── Step 1: Grid search on train period ───────────────────────────────────
    combos = list(itertools.product(
        OPT_GRID["adx_delta"], OPT_GRID["score_delta"],
        OPT_GRID["sl_mult"],   OPT_GRID["tp_mult"],
    ))
    base_params = MODE_SIGNAL_PARAMS.get(mode, MODE_SIGNAL_PARAMS["NORMAL"])

    best_sharpe = -999.0
    best_params = {}
    train_metrics_best = None

    for adx_d, score_d, sl_m, tp_m in combos:
        param_overrides = {
            "adx_min":   max(8.0,  base_params["adx_min"]   + adx_d),
            "min_score": max(0.40, min(0.92, base_params["min_score"] + score_d)),
        }
        m = _sim_period(
            train_main, train_high, mode, style, exchange,
            start_equity, param_overrides, sl_m, tp_m,
        )
        if m and m["total_trades"] >= MIN_TRADES_TRAIN:
            if m["sharpe_ratio"] > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_params = {
                    "adx_min":   param_overrides["adx_min"],
                    "min_score": param_overrides["min_score"],
                    "sl_mult":   sl_m,
                    "tp_mult":   tp_m,
                }
                train_metrics_best = m

    if train_metrics_best is None:
        return {"error": f"No valid parameter combo found for {symbol} on train period"}

    # ── Step 2: Replay test period with best params (no refitting) ────────────
    param_overrides_best = {
        "adx_min":   best_params["adx_min"],
        "min_score": best_params["min_score"],
    }
    test_metrics = _sim_period(
        test_main, test_high, mode, style, exchange,
        start_equity, param_overrides_best,
        best_params["sl_mult"], best_params["tp_mult"],
    )
    if test_metrics is None:
        test_metrics = {
            "total_trades": 0, "win_rate": 0.0, "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
        }

    # ── Step 3: Degradation ───────────────────────────────────────────────────
    def _degrade(train_val, test_val):
        if train_val == 0:
            return 0.0
        return round((train_val - test_val) / abs(train_val) * 100, 1)

    degrade_sharpe = _degrade(train_metrics_best["sharpe_ratio"], test_metrics["sharpe_ratio"])
    degrade_wr     = _degrade(train_metrics_best["win_rate"],     test_metrics["win_rate"])
    degrade_return = _degrade(train_metrics_best["total_return_pct"], test_metrics["total_return_pct"])

    # Verdict
    if test_metrics["total_trades"] < MIN_TRADES_TEST:
        verdict = "INSUFFICIENT_DATA"
    elif degrade_sharpe > DEGRADE_THRESHOLD * 100:
        verdict = "NEEDS_REVIEW"
    elif test_metrics["sharpe_ratio"] > 0.5 and test_metrics["win_rate"] >= 30:
        verdict = "PASS"
    else:
        verdict = "NEEDS_REVIEW"

    return {
        "symbol":          symbol,
        "train_trades":    train_metrics_best["total_trades"],
        "train_win_rate":  train_metrics_best["win_rate"],
        "train_return":    train_metrics_best["total_return_pct"],
        "train_max_dd":    train_metrics_best["max_drawdown_pct"],
        "train_sharpe":    train_metrics_best["sharpe_ratio"],
        "test_trades":     test_metrics["total_trades"],
        "test_win_rate":   test_metrics["win_rate"],
        "test_return":     test_metrics["total_return_pct"],
        "test_max_dd":     test_metrics["max_drawdown_pct"],
        "test_sharpe":     test_metrics["sharpe_ratio"],
        "degrade_sharpe":  degrade_sharpe,
        "degrade_wr":      degrade_wr,
        "degrade_return":  degrade_return,
        "best_params":     best_params,
        "verdict":         verdict,
    }


def _sim_period(
    main_candles: List[Dict],
    higher_candles: List[Dict],
    mode: str,
    style: str,
    exchange: str,
    start_equity: float,
    param_overrides: Dict,
    sl_mult: float,
    tp_mult: float,
) -> Optional[Dict]:
    """Run one simulation pass. Returns metrics dict or None if too few trades."""
    st = TRADE_STYLE_PARAMS[style]
    c  = MODE_PRESETS[mode]
    tf = TRADE_STYLE_PARAMS[style]["tf"]

    equity      = start_equity
    peak_equity = start_equity
    trades: List[Dict] = []
    equity_curve: List[Dict] = [{"ts": main_candles[WARMUP_CANDLES]["t"], "equity": equity}]
    open_trade: Optional[Dict] = None

    for i in range(WARMUP_CANDLES, len(main_candles)):
        candle = main_candles[i]

        if open_trade:
            result = _sim_check_exit(open_trade, candle, exchange)
            if result and result.get("closed"):
                equity += result["pnl"]
                peak_equity = max(peak_equity, equity)
                trades.append({
                    "pnl": result["pnl"], "outcome": result["outcome"],
                    "side": open_trade["side"], "grade": open_trade["grade"],
                    "signal": open_trade.get("signal", "-"),
                    "score": open_trade.get("score", 0.0),
                    "regime": open_trade.get("regime", "-"),
                    "open_ts": open_trade["open_ts"], "exit_ts": result["exit_ts"],
                    "entry": open_trade["entry"],
                })
                equity_curve.append({"ts": result["exit_ts"], "equity": round(equity, 4)})
                open_trade = None

        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if drawdown >= 0.15:
            break

        if open_trade is None:
            if _session_quality_for_ts(style, candle["t"]) == 0.0:
                continue
            klines       = main_candles[max(0, i - 299): i + 1]
            higher_slice = _get_aligned_slice(higher_candles, candle["t"], 100)
            try:
                sig = _compute_signal_layers(
                    klines, mode, 1.0, higher_slice, style, None,
                    param_overrides=param_overrides,
                )
            except Exception:
                continue
            if not sig.get("ok"):
                continue

            atr_pct  = sig.get("atr_pct", ATR_BASELINE.get(tf, 0.009))
            sl_pct   = min(atr_pct * st["sl_atr"] * sl_mult, st["sl_max"] / 100)
            tp_pct   = min(atr_pct * st["tp_atr"] * tp_mult, st["tp_max"] / 100)
            baseline  = ATR_BASELINE.get(tf, 0.009)
            vol_ratio = atr_pct / baseline if baseline > 0 else 1.0
            vol_mult  = max(0.4, 1.0 / vol_ratio) if vol_ratio > 1.5 else 1.0
            dd_pct   = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            dd_mult  = 0.25 if dd_pct >= 0.10 else 0.40 if dd_pct >= 0.07 else 0.65 if dd_pct >= 0.04 else 1.0
            eff_size = c["size"] * vol_mult * dd_mult

            open_trade = {
                "entry": candle["close"], "side": sig["side"],
                "grade": sig.get("grade", "B"),
                "sl_pct": sl_pct, "tp_pct": tp_pct,
                "effective_size": eff_size * equity,
                "leverage": c["leverage"],
                "equity_at_open": equity,
                "t1_size":           (eff_size * equity) * 0.60,
                "t2_size":           (eff_size * equity) * 0.40,
                "t1_done":           False,
                "t2_sl_is_breakeven": False,
                "accumulated_pnl":   0.0,
                "open_ts": candle["t"],
                "signal": sig.get("label", "-"),
                "score":  round(sig.get("score", 0.0), 3),
                "regime": sig.get("market_regime", "-"),
            }

    if not trades:
        return None
    return _calc_metrics(
        trades, equity_curve, start_equity,
        main_candles[WARMUP_CANDLES]["close"],
        main_candles[-1]["close"],
    )


# ── Pipeline orchestrator ──────────────────────────────────────────────────────

def _pipeline_worker(
    run_id: str,
    coins: List[str],
    mode: str,
    style: str,
    train_start_ms: int,
    train_end_ms: int,
    test_start_ms: int,
    test_end_ms: int,
    exchange: str = "bybit",
    start_equity: float = 1000.0,
) -> None:
    """Background thread: runs pipeline for all coins, stores results, generates report."""
    now_iso = datetime.utcnow().isoformat()

    def _set_status(status: str, done: int = 0, error: str = None):
        with _active_lock:
            _active[run_id] = {
                "status": status, "done": done, "total": len(coins), "error": error,
            }
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE pipeline_runs SET status=%s, coins_done=%s WHERE run_id=%s",
                    (status, done, run_id),
                )
                conn.commit()
        except Exception:
            pass

    _set_status("running", 0)
    results = []

    for idx, symbol in enumerate(coins):
        try:
            result = _run_one_coin(
                symbol, mode, style,
                train_start_ms, train_end_ms,
                test_start_ms, test_end_ms,
                exchange, start_equity,
            )
        except Exception as exc:
            result = {"symbol": symbol, "error": str(exc)}

        result["symbol"] = symbol
        results.append(result)
        _set_status("running", idx + 1)

        # Store in DB
        try:
            _store_coin_result(run_id, result)
        except Exception:
            pass

    # Finalise run
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE pipeline_runs SET status=%s, completed_at=%s, coins_done=%s"
                " WHERE run_id=%s",
                ("complete", datetime.utcnow().isoformat(), len(coins), run_id),
            )
            conn.commit()
    except Exception:
        pass

    with _active_lock:
        _active[run_id] = {"status": "complete", "done": len(coins), "total": len(coins)}

    # Send Telegram summary
    try:
        _send_pipeline_telegram(run_id, results, mode, style)
    except Exception:
        pass


def _store_coin_result(run_id: str, result: Dict) -> None:
    symbol = result.get("symbol", "?")
    err    = result.get("error")
    verdict = result.get("verdict", "ERROR" if err else "UNKNOWN")
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO pipeline_coin_results"
            "(run_id,symbol,train_trades,train_win_rate,train_return,train_max_dd,"
            "train_sharpe,test_trades,test_win_rate,test_return,test_max_dd,"
            "test_sharpe,degrade_sharpe,degrade_wr,degrade_return,"
            "best_params,verdict,created_at)"
            " VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                run_id, symbol,
                result.get("train_trades"),  result.get("train_win_rate"),
                result.get("train_return"),  result.get("train_max_dd"),
                result.get("train_sharpe"),
                result.get("test_trades"),   result.get("test_win_rate"),
                result.get("test_return"),   result.get("test_max_dd"),
                result.get("test_sharpe"),
                result.get("degrade_sharpe"), result.get("degrade_wr"),
                result.get("degrade_return"),
                json.dumps(result.get("best_params") or {}),
                verdict,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def _send_pipeline_telegram(run_id: str, results: List[Dict], mode: str, style: str) -> None:
    passes       = sum(1 for r in results if r.get("verdict") == "PASS")
    needs_review = sum(1 for r in results if r.get("verdict") == "NEEDS_REVIEW")
    errors       = sum(1 for r in results if r.get("error") or r.get("verdict") == "ERROR")

    lines = [
        f"<b>Pipeline complete</b> — {mode} {style}",
        f"Run ID: <code>{run_id[:8]}</code>",
        f"PASS: {passes} | NEEDS REVIEW: {needs_review} | ERROR: {errors}",
        "",
    ]
    for r in results:
        sym  = r.get("symbol", "?").replace("USDT", "")
        mark = {"PASS": "✅", "NEEDS_REVIEW": "⚠️", "INSUFFICIENT_DATA": "⚪"}.get(r.get("verdict", ""), "❌")
        if r.get("error"):
            lines.append(f"{mark} {sym}: ERROR — {str(r['error'])[:40]}")
        else:
            lines.append(
                f"{mark} {sym}: "
                f"train Sharpe={r.get('train_sharpe',0):.2f} | "
                f"test Sharpe={r.get('test_sharpe',0):.2f} | "
                f"degrade={r.get('degrade_sharpe',0):.1f}%"
            )

    tg_alert("\n".join(lines))


# ── Live degradation check ─────────────────────────────────────────────────────

def check_live_degradation(email: str = None, lookback_days: int = 30) -> List[Dict]:
    """
    Compare recent live trade_outcomes vs most recent pipeline test period.
    Fires Telegram alert if Sharpe or win-rate degrades more than DEGRADE_THRESHOLD.
    Called once per day from the monthly scheduler.
    """
    alerts_fired = []
    cutoff_iso = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")

    try:
        with db_conn() as conn:
            cur = conn.cursor()
            # Get latest pipeline results per symbol
            cur.execute(
                "SELECT DISTINCT symbol FROM pipeline_coin_results"
            )
            symbols = [r["symbol"] for r in cur.fetchall()]

        for symbol in symbols:
            with db_conn() as conn:
                cur = conn.cursor()
                # Best test_sharpe and test_win_rate from latest pipeline run
                cur.execute(
                    "SELECT pcr.test_sharpe, pcr.test_win_rate "
                    "FROM pipeline_coin_results pcr "
                    "JOIN pipeline_runs pr ON pr.run_id = pcr.run_id "
                    "WHERE pcr.symbol=%s AND pr.status='complete' "
                    "ORDER BY pr.completed_at DESC LIMIT 1",
                    (symbol,),
                )
                pipeline_row = cur.fetchone()
                if not pipeline_row:
                    continue

                expected_sharpe = float(pipeline_row["test_sharpe"] or 0)
                expected_wr     = float(pipeline_row["test_win_rate"] or 0)

                # Live trades for this symbol over last N days
                email_filter = f" AND email=%s" if email else ""
                params = [symbol]
                if email:
                    params.append(email)

                cur.execute(
                    "SELECT pnl_pct, outcome FROM trade_outcomes "
                    "WHERE symbol=%s AND closed_at >= %s" + email_filter,
                    [symbol, cutoff_iso] + ([email] if email else []),
                )
                live_rows = cur.fetchall()

            if len(live_rows) < 5:
                continue  # not enough live data to judge

            live_pnls   = [float(r["pnl_pct"] or 0) for r in live_rows]
            live_wins   = sum(1 for r in live_rows if r["outcome"] in ("TP_HIT","T1","T2","SCALE_TRAIL"))
            live_wr     = live_wins / len(live_rows) * 100 if live_rows else 0.0
            live_sharpe = _quick_sharpe(live_pnls)

            # Check degradation
            for metric, expected, actual in [
                ("sharpe", expected_sharpe, live_sharpe),
                ("win_rate", expected_wr, live_wr),
            ]:
                if expected <= 0:
                    continue
                degrade = (expected - actual) / abs(expected)
                if degrade >= DEGRADE_THRESHOLD:
                    _fire_degradation_alert(symbol, metric, expected, actual, degrade * 100)
                    alerts_fired.append({
                        "symbol": symbol, "metric": metric,
                        "expected": expected, "actual": actual,
                        "degrade_pct": round(degrade * 100, 1),
                    })

    except Exception as exc:
        print(f"[pipeline] live degradation check failed: {exc}")

    return alerts_fired


def _fire_degradation_alert(symbol: str, metric: str, expected: float, actual: float, degrade_pct: float) -> None:
    msg = (
        f"⚠️ <b>Performance degradation alert</b>\n"
        f"Symbol: {symbol} | Metric: {metric}\n"
        f"Expected: {expected:.2f} | Actual: {actual:.2f}\n"
        f"Degradation: <b>{degrade_pct:.1f}%</b> (threshold {DEGRADE_THRESHOLD*100:.0f}%)\n"
        f"Action: manual review recommended before next trade."
    )
    tg_alert(msg)
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO pipeline_alerts"
                "(symbol,mode,style,metric,expected,actual,degrade_pct,fired_at)"
                " VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                (symbol, DEFAULT_MODE, DEFAULT_STYLE, metric,
                 round(expected, 4), round(actual, 4),
                 round(degrade_pct, 1), datetime.utcnow().isoformat()),
            )
            conn.commit()
    except Exception:
        pass


def _quick_sharpe(pnl_pcts: List[float]) -> float:
    if len(pnl_pcts) < 2:
        return 0.0
    try:
        mean = statistics.mean(pnl_pcts)
        std  = statistics.stdev(pnl_pcts)
        return round(mean / std * (252 ** 0.5) if std > 0 else 0.0, 4)
    except Exception:
        return 0.0


# ── Public API ─────────────────────────────────────────────────────────────────

def start_pipeline(
    triggered_by: str = "manual",
    coins: List[str] = None,
    mode: str = DEFAULT_MODE,
    style: str = DEFAULT_STYLE,
    train_years: int = TRAIN_YEARS,
    test_years: int = TEST_YEARS,
    exchange: str = "bybit",
    start_equity: float = 1000.0,
) -> str:
    """Launch pipeline in background thread. Returns run_id."""
    if coins is None:
        coins = PIPELINE_COINS

    # Rolling window: test = most recent `test_years`, train = `train_years` before that
    now      = datetime.utcnow()
    test_end = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp() * 1000)  # start of this month
    test_start  = _ms_subtract_years(test_end, test_years)
    train_end   = test_start
    train_start = _ms_subtract_years(train_end, train_years)

    run_id = str(uuid.uuid4())[:8]
    now_iso = datetime.utcnow().isoformat()

    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO pipeline_runs"
                "(run_id,triggered_by,mode,style,train_from,train_to,"
                "test_from,test_to,coins_total,created_at)"
                " VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    run_id, triggered_by, mode, style,
                    _ms_to_date(train_start), _ms_to_date(train_end),
                    _ms_to_date(test_start),  _ms_to_date(test_end),
                    len(coins), now_iso,
                ),
            )
            conn.commit()
    except Exception as exc:
        print(f"[pipeline] DB write failed: {exc}")
        return run_id

    t = threading.Thread(
        target=_pipeline_worker,
        args=(run_id, coins, mode, style,
              train_start, train_end, test_start, test_end,
              exchange, start_equity),
        daemon=True,
        name=f"pipeline-{run_id}",
    )
    t.start()
    return run_id


def get_pipeline_status(run_id: str) -> Dict:
    with _active_lock:
        mem = _active.get(run_id, {})
    if mem:
        return mem
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT status, coins_done, coins_total, error, completed_at"
                " FROM pipeline_runs WHERE run_id=%s", (run_id,),
            )
            row = cur.fetchone()
        if row:
            return dict(row)
    except Exception:
        pass
    return {"status": "not_found"}


def get_pipeline_results(run_id: str) -> List[Dict]:
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM pipeline_coin_results WHERE run_id=%s ORDER BY id",
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def list_pipeline_runs(limit: int = 20) -> List[Dict]:
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT run_id, triggered_by, mode, style, train_from, train_to,"
                " test_from, test_to, status, coins_done, coins_total, created_at, completed_at"
                " FROM pipeline_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


# ── HTML report ────────────────────────────────────────────────────────────────

def generate_html_report(run_id: str) -> str:
    """Generate a printable HTML report for this pipeline run. Print to PDF from browser."""
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM pipeline_runs WHERE run_id=%s", (run_id,))
            run = cur.fetchone()
            cur.execute(
                "SELECT * FROM pipeline_coin_results WHERE run_id=%s ORDER BY id",
                (run_id,),
            )
            rows = cur.fetchall()
    except Exception as exc:
        return f"<p>Error loading results: {exc}</p>"

    if not run:
        return "<p>Run not found.</p>"

    def _fmt(v, decimals=2):
        if v is None:
            return "—"
        try:
            return f"{float(v):.{decimals}f}"
        except Exception:
            return str(v)

    def _verdict_badge(v):
        colors = {
            "PASS": ("#00c48c", "white"),
            "NEEDS_REVIEW": ("#f59e0b", "white"),
            "INSUFFICIENT_DATA": ("#6b7280", "white"),
            "ERROR": ("#ef4444", "white"),
        }
        bg, fg = colors.get(v or "", ("#999", "#fff"))
        return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-size:12px;font-weight:700;">{v or "?"}</span>'

    passes       = sum(1 for r in rows if r.get("verdict") == "PASS")
    needs_review = sum(1 for r in rows if r.get("verdict") == "NEEDS_REVIEW")
    errors       = sum(1 for r in rows if r.get("error") or r.get("verdict") in ("ERROR", None))

    table_rows = ""
    for r in rows:
        dd = float(r.get("degrade_sharpe") or 0)
        dd_color = "#ef4444" if dd > 25 else "#f59e0b" if dd > 10 else "#00c48c"
        table_rows += f"""
        <tr>
          <td><b>{r['symbol'].replace('USDT','')}</b></td>
          <td>{_fmt(r.get('train_trades'), 0)}</td>
          <td>{_fmt(r.get('train_win_rate'))}%</td>
          <td>{_fmt(r.get('train_return'))}%</td>
          <td>{_fmt(r.get('train_max_dd'))}%</td>
          <td>{_fmt(r.get('train_sharpe'))}</td>
          <td>{_fmt(r.get('test_trades'), 0)}</td>
          <td>{_fmt(r.get('test_win_rate'))}%</td>
          <td>{_fmt(r.get('test_return'))}%</td>
          <td>{_fmt(r.get('test_max_dd'))}%</td>
          <td>{_fmt(r.get('test_sharpe'))}</td>
          <td style="color:{dd_color};font-weight:700;">{_fmt(r.get('degrade_sharpe'))}%</td>
          <td>{_verdict_badge(r.get('verdict'))}</td>
        </tr>"""

    best_params_section = ""
    for r in rows:
        bp = r.get("best_params") or "{}"
        try:
            params = json.loads(bp) if isinstance(bp, str) else bp
            sym = r['symbol'].replace('USDT','')
            best_params_section += f"""
            <div style="margin:8px 0;padding:8px 12px;background:#f9fafb;border-radius:6px;font-size:13px;">
              <b>{sym}</b> — ADX≥{params.get('adx_min','?'):.0f}, min_score={params.get('min_score','?'):.3f},
              SL×{params.get('sl_mult','?'):.1f}, TP×{params.get('tp_mult','?'):.1f}
            </div>"""
        except Exception:
            pass

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Asymmetric AI — Pipeline Report {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 40px; color: #1f2937; }}
    h1   {{ font-size: 22px; margin-bottom: 4px; }}
    h2   {{ font-size: 16px; margin: 28px 0 8px; border-bottom: 2px solid #e5e7eb; padding-bottom: 4px; }}
    .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
    .summary {{ display: flex; gap: 24px; margin-bottom: 24px; }}
    .card {{ background: #f3f4f6; border-radius: 8px; padding: 14px 20px; min-width: 100px; text-align: center; }}
    .card .val {{ font-size: 28px; font-weight: 800; }}
    .card .lbl {{ font-size: 12px; color: #6b7280; margin-top: 2px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th    {{ background: #f3f4f6; padding: 8px 10px; text-align: left; border-bottom: 2px solid #e5e7eb; white-space: nowrap; }}
    td    {{ padding: 7px 10px; border-bottom: 1px solid #f3f4f6; }}
    tr:hover td {{ background: #fafafa; }}
    .note {{ background: #fffbeb; border: 1px solid #fde68a; padding: 10px 14px; border-radius: 6px; font-size: 13px; margin-top: 16px; }}
    @media print {{
      .note {{ page-break-inside: avoid; }}
      table  {{ font-size: 11px; }}
    }}
  </style>
</head>
<body>
  <h1>Asymmetric AI — Walk-Forward Validation Report</h1>
  <div class="meta">
    Run ID: <b>{run_id}</b> &nbsp;|&nbsp;
    Mode: <b>{run.get('mode')}</b> &nbsp;|&nbsp;
    Style: <b>{run.get('style')}</b> &nbsp;|&nbsp;
    Generated: <b>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</b><br>
    Train period: <b>{run.get('train_from')}</b> → <b>{run.get('train_to')}</b>
    &nbsp;(parameters optimised on this period)<br>
    Test period: <b>{run.get('test_from')}</b> → <b>{run.get('test_to')}</b>
    &nbsp;(out-of-sample validation, params unchanged)
  </div>

  <div class="summary">
    <div class="card"><div class="val" style="color:#00c48c;">{passes}</div><div class="lbl">PASS</div></div>
    <div class="card"><div class="val" style="color:#f59e0b;">{needs_review}</div><div class="lbl">NEEDS REVIEW</div></div>
    <div class="card"><div class="val" style="color:#ef4444;">{errors}</div><div class="lbl">ERROR / NO DATA</div></div>
    <div class="card"><div class="val">{len(rows)}</div><div class="lbl">COINS TOTAL</div></div>
  </div>

  <h2>Results Table</h2>
  <table>
    <thead>
      <tr>
        <th>Coin</th>
        <th colspan="5" style="text-align:center;background:#dbeafe;">— TRAIN (optimised) —</th>
        <th colspan="5" style="text-align:center;background:#dcfce7;">— TEST (out-of-sample) —</th>
        <th>Sharpe<br>Degrade</th>
        <th>Verdict</th>
      </tr>
      <tr>
        <th></th>
        <th style="background:#eff6ff;">Trades</th>
        <th style="background:#eff6ff;">WR%</th>
        <th style="background:#eff6ff;">Return%</th>
        <th style="background:#eff6ff;">MaxDD%</th>
        <th style="background:#eff6ff;">Sharpe</th>
        <th style="background:#f0fdf4;">Trades</th>
        <th style="background:#f0fdf4;">WR%</th>
        <th style="background:#f0fdf4;">Return%</th>
        <th style="background:#f0fdf4;">MaxDD%</th>
        <th style="background:#f0fdf4;">Sharpe</th>
        <th></th>
        <th></th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h2>Best Parameters per Coin</h2>
  {best_params_section or "<p style='color:#9ca3af'>No valid params found.</p>"}

  <div class="note">
    <b>How to read this report:</b><br>
    PASS = test Sharpe &gt; 0.5, WR ≥ 30%, Sharpe degradation &lt; 25%.<br>
    NEEDS REVIEW = Sharpe degraded &gt;25% between train and test periods — investigate before using these params live.<br>
    Sharpe Degrade = (train_sharpe − test_sharpe) / train_sharpe × 100. Positive = performance dropped in test period.<br>
    Parameters were optimised ONLY on the train period. Test period was never seen during optimisation.
  </div>
</body>
</html>"""
    return html


# ── Monthly scheduler ──────────────────────────────────────────────────────────

def _monthly_scheduler_loop() -> None:
    """
    Runs in a daemon thread. On the 1st of each month (UTC), triggers a new pipeline run.
    Also runs a daily degradation check every 24h.
    """
    last_pipeline_month = None  # track which month we last ran the pipeline
    last_degrade_check  = 0.0
    last_backup_day     = None

    while not _scheduler_stop.is_set():
        now = datetime.utcnow()

        # Monthly pipeline: trigger on day 1 of each month
        month_key = (now.year, now.month)
        if now.day == 1 and month_key != last_pipeline_month:
            print(f"[pipeline] Monthly auto-run triggered for {now.strftime('%Y-%m')}")
            try:
                run_id = start_pipeline(triggered_by="monthly_auto")
                last_pipeline_month = month_key
                print(f"[pipeline] Monthly run started: {run_id}")
            except Exception as exc:
                print(f"[pipeline] Monthly run failed to start: {exc}")

        # Daily degradation check
        if time.time() - last_degrade_check >= 86400:
            try:
                alerts = check_live_degradation(lookback_days=30)
                if alerts:
                    print(f"[pipeline] {len(alerts)} degradation alert(s) fired")
            except Exception as exc:
                print(f"[pipeline] Degradation check error: {exc}")
            last_degrade_check = time.time()

        # Daily R2 backup at 02:00 UTC
        today_key = (now.year, now.month, now.day)
        if now.hour == 2 and today_key != last_backup_day:
            try:
                from backup_r2 import run_backup_safe
                run_backup_safe()
            except Exception as exc:
                print(f"[pipeline] Backup error: {exc}")
            last_backup_day = today_key

        # Sleep in 1h chunks (check once per hour is plenty)
        _scheduler_stop.wait(timeout=3600)


def start_monthly_scheduler() -> None:
    """Call from main.py startup once. Starts the background scheduler thread."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(
        target=_monthly_scheduler_loop,
        daemon=True,
        name="pipeline-scheduler",
    )
    _scheduler_thread.start()
    print("[pipeline] Monthly scheduler started")


# ── PDF report ────────────────────────────────────────────────────────────────

def generate_pdf_report(run_id: str) -> bytes:
    """
    Generate a PDF report using fpdf2. Returns raw PDF bytes.
    Falls back to empty bytes if fpdf2 is unavailable.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM pipeline_runs WHERE run_id=%s", (run_id,))
            run = cur.fetchone()
            cur.execute(
                "SELECT * FROM pipeline_coin_results WHERE run_id=%s ORDER BY id",
                (run_id,),
            )
            rows = cur.fetchall()
    except Exception:
        return b""

    if not run:
        return b""

    def _fv(v, dec=2):
        if v is None:
            return "—"
        try:
            return f"{float(v):.{dec}f}"
        except Exception:
            return str(v)

    passes       = sum(1 for r in rows if r.get("verdict") == "PASS")
    needs_review = sum(1 for r in rows if r.get("verdict") == "NEEDS_REVIEW")

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_margins(12, 12, 12)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Asymmetric AI - Walk-Forward Validation Report", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Run: {run_id}  |  Mode: {run.get('mode')}  |  Style: {run.get('style')}  |  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ln=True)
    pdf.cell(0, 5, f"Train: {run.get('train_from')} to {run.get('train_to')}  (params optimised here)", ln=True)
    pdf.cell(0, 5, f"Test:  {run.get('test_from')} to {run.get('test_to')}  (out-of-sample, params unchanged)", ln=True)
    pdf.ln(3)

    # Summary bar
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 255, 240)
    pdf.cell(45, 8, f"PASS: {passes}", border=1, fill=True)
    pdf.set_fill_color(255, 248, 220)
    pdf.cell(55, 8, f"NEEDS REVIEW: {needs_review}", border=1, fill=True)
    pdf.cell(55, 8, f"TOTAL COINS: {len(rows)}", border=1)
    pdf.ln(10)

    # Table header — two rows
    COL = [16, 16, 14, 16, 14, 14,   16, 14, 16, 14, 14,   20, 22]
    HEADS1 = ["", "TRAIN (optimised)", "", "", "", "",
               "TEST (out-of-sample)", "", "", "", "", "", ""]
    HEADS2 = ["Coin",
              "Trades","WR%","Return%","MaxDD%","Sharpe",
              "Trades","WR%","Return%","MaxDD%","Sharpe",
              "Sharpe Degrade","Verdict"]

    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(220, 235, 255)
    # Row 1 — group headers (spans)
    pdf.cell(COL[0], 6, "", border=1, fill=True)
    # Train group
    train_w = sum(COL[1:6])
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(train_w, 6, "TRAIN (optimised)", border=1, align="C", fill=True)
    # Test group
    test_w = sum(COL[6:11])
    pdf.set_fill_color(210, 245, 220)
    pdf.cell(test_w, 6, "TEST (out-of-sample)", border=1, align="C", fill=True)
    pdf.cell(COL[11], 6, "", border=1, fill=False)
    pdf.cell(COL[12], 6, "", border=1, fill=False)
    pdf.ln()

    # Row 2 — column names
    pdf.set_fill_color(240, 244, 255)
    for i, (h, w) in enumerate(zip(HEADS2, COL)):
        if i in (1, 2, 3, 4, 5):
            pdf.set_fill_color(235, 242, 255)
        elif i in (6, 7, 8, 9, 10):
            pdf.set_fill_color(230, 248, 235)
        else:
            pdf.set_fill_color(245, 245, 245)
        pdf.cell(w, 6, h, border=1, align="C", fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 8)
    for r in rows:
        verdict = r.get("verdict") or "?"
        dd = float(r.get("degrade_sharpe") or 0)

        # Row background
        if verdict == "PASS":
            pdf.set_fill_color(245, 255, 248)
        elif verdict == "NEEDS_REVIEW":
            pdf.set_fill_color(255, 252, 235)
        else:
            pdf.set_fill_color(255, 245, 245)

        sym = (r.get("symbol") or "?").replace("USDT", "")
        vals = [
            sym,
            _fv(r.get("train_trades"), 0),
            _fv(r.get("train_win_rate")),
            _fv(r.get("train_return")),
            _fv(r.get("train_max_dd")),
            _fv(r.get("train_sharpe")),
            _fv(r.get("test_trades"), 0),
            _fv(r.get("test_win_rate")),
            _fv(r.get("test_return")),
            _fv(r.get("test_max_dd")),
            _fv(r.get("test_sharpe")),
            f"{dd:.1f}%" if dd else "—",
            verdict,
        ]

        for i, (v, w) in enumerate(zip(vals, COL)):
            if i == 11:  # degrade col — colour by severity
                if dd > 25:
                    pdf.set_text_color(200, 0, 0)
                elif dd > 10:
                    pdf.set_text_color(180, 100, 0)
                else:
                    pdf.set_text_color(0, 140, 60)
            elif i == 12:  # verdict
                if verdict == "PASS":
                    pdf.set_text_color(0, 140, 60)
                elif verdict == "NEEDS_REVIEW":
                    pdf.set_text_color(180, 100, 0)
                else:
                    pdf.set_text_color(200, 0, 0)
            else:
                pdf.set_text_color(0, 0, 0)
            pdf.cell(w, 6, str(v), border=1, align="C" if i > 0 else "L", fill=(i == 0))
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    # Best params section
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Best Parameters (from train period - used unchanged on test)", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    for r in rows:
        bp_raw = r.get("best_params") or "{}"
        try:
            bp = json.loads(bp_raw) if isinstance(bp_raw, str) else bp_raw
            sym = (r.get("symbol") or "?").replace("USDT", "")
            line = (f"{sym:6s}  ADX≥{bp.get('adx_min',0):.0f}  "
                    f"min_score={bp.get('min_score',0):.3f}  "
                    f"SL×{bp.get('sl_mult',0):.1f}  TP×{bp.get('tp_mult',0):.1f}")
            pdf.cell(0, 5, line, ln=True)
        except Exception:
            pass

    # Footer note
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4,
        "PASS = test Sharpe > 0.5, WR >= 30%, Sharpe degradation < 25%.  "
        "NEEDS REVIEW = Sharpe degraded >25% between train and test - investigate before live use.  "
        "Parameters were optimised ONLY on the train period. Test period was never seen during optimisation."
    )

    return bytes(pdf.output())


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms_subtract_years(ts_ms: int, years: int) -> int:
    dt = datetime.utcfromtimestamp(ts_ms / 1000)
    try:
        earlier = dt.replace(year=dt.year - years)
    except ValueError:
        earlier = dt.replace(year=dt.year - years, day=28)
    return int(earlier.timestamp() * 1000)


def _ms_to_date(ts_ms: int) -> str:
    return datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
