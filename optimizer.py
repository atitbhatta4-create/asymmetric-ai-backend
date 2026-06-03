"""
optimizer.py — Parameter grid search over cached backtest candles.

Loads candles from DB (must be cached via backtester first), then runs
the simulation loop 135 times with different signal/SL/TP parameter combos.
Ranks results by Sharpe ratio. Admin applies best params with one click.

Import chain: config → indicators → backtester → optimizer (no circular imports)
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from backtester import (
    ATR_BASELINE,
    MODE_PRESETS,
    WARMUP_CANDLES,
    _calc_metrics,
    _ensure_candles,
    _fee_cost,
    _get_aligned_slice,
    _session_quality_for_ts,
    _sim_check_exit,
)
from config import HIGHER_TF_MAP, TRADE_STYLE_PARAMS
from database import USING_PG, db_conn, serial_pk
from indicators import _compute_signal_layers

# ── Parameter search grid ───────────────────────────────────────────────────────
# Offsets are applied ON TOP of the mode's default values.
# 5 × 3 × 3 × 3 = 135 combinations per run.
OPT_GRID: Dict[str, List] = {
    "adx_delta":   [-4, -2, 0, 2, 4],      # added to mode's adx_min
    "score_delta": [-0.04, 0.0, 0.04],     # added to mode's min_score
    "sl_mult":     [0.8, 1.0, 1.2],        # multiplied on style's sl_atr
    "tp_mult":     [1.5, 2.0, 2.5],        # multiplied on style's tp_atr
}

TF_FROM_STYLE: Dict[str, str] = {
    "SCALP": "15m", "DAY_TRADE": "1h", "SWING": "4h",
}

MIN_TRADES     = 10   # reject combos with fewer trades (unreliable stats)
MIN_WIN_RATE   = 0.25 # flag combos below this win rate (lowered from 0.35)
MAX_DRAWDOWN   = 0.25 # flag combos with deeper drawdown
TOP_RESULTS    = 50   # store top N results per run (raised from 20)

# Active optimizer runs in memory: run_id → {status, progress, done, total}
_active: Dict[str, Dict] = {}
_active_lock = threading.Lock()


# ── DB setup ────────────────────────────────────────────────────────────────────
def init_optimizer_tables() -> None:
    with db_conn() as conn:
        cur = conn.cursor()
        pk = serial_pk()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS optimizer_runs (
                run_id       TEXT PRIMARY KEY,
                email        TEXT NOT NULL,
                symbol       TEXT NOT NULL,
                tf           TEXT NOT NULL,
                mode         TEXT NOT NULL,
                style        TEXT NOT NULL,
                exchange     TEXT NOT NULL DEFAULT 'bybit',
                date_from    TEXT NOT NULL,
                date_to      TEXT NOT NULL,
                start_equity REAL NOT NULL DEFAULT 1000,
                status       TEXT NOT NULL DEFAULT 'pending',
                progress     INTEGER DEFAULT 0,
                total_combos INTEGER DEFAULT 135,
                done_combos  INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL,
                completed_at TEXT,
                error        TEXT
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS optimizer_results (
                id             {pk},
                run_id         TEXT NOT NULL,
                params_json    TEXT NOT NULL,
                total_trades   INTEGER,
                win_rate       REAL,
                total_return   REAL,
                max_drawdown   REAL,
                sharpe_ratio   REAL,
                reward_risk    REAL,
                passed_filters INTEGER DEFAULT 1,
                is_applied     INTEGER DEFAULT 0,
                created_at     TEXT NOT NULL
            )
        """)
        # Migration: add passed_filters if upgrading from older schema
        try:
            cur.execute("ALTER TABLE optimizer_results ADD COLUMN passed_filters INTEGER DEFAULT 1")
            conn.commit()
        except Exception:
            pass
        cur.execute("""
            CREATE TABLE IF NOT EXISTS optimizer_applied (
                symbol      TEXT NOT NULL,
                mode        TEXT NOT NULL,
                style       TEXT NOT NULL,
                params_json TEXT NOT NULL,
                sharpe      REAL,
                applied_at  TEXT NOT NULL,
                applied_by  TEXT NOT NULL,
                PRIMARY KEY (symbol, mode, style)
            )
        """)
        # On restart mark stuck runs as error
        cur.execute(
            "UPDATE optimizer_runs SET status=%s, error=%s, completed_at=%s"
            " WHERE status IN ('pending','running')",
            ("error", "Server restarted — please retry",
             datetime.utcnow().isoformat()),
        )
        conn.commit()


# ── Core simulation (no candle fetch, no DB writes, pure Python) ───────────────
def _sim_one(
    main_candles: List[Dict],
    higher_candles: List[Dict],
    mode: str,
    style: str,
    exchange: str,
    start_equity: float,
    adx_delta: float,
    score_delta: float,
    sl_mult: float,
    tp_mult: float,
) -> Optional[Dict]:
    """
    Run one simulation pass with the given parameter offsets.
    Returns metrics dict or None if not enough trades to be meaningful.
    """
    st = TRADE_STYLE_PARAMS[style]
    c  = MODE_PRESETS[mode]
    tf = TF_FROM_STYLE[style]

    # Build signal param overrides — deltas on top of mode defaults
    signal_overrides = {
        "adx_min_delta":   adx_delta,    # handled inside _compute_signal_layers
        "score_min_delta": score_delta,  # handled inside _compute_signal_layers
    }
    # We pass explicit absolute values so _compute_signal_layers doesn't need
    # delta logic. Instead we compute absolute values here and pass them.
    # These keys match MODE_SIGNAL_PARAMS — they override BEFORE style adjustments.
    from indicators import MODE_SIGNAL_PARAMS
    base = MODE_SIGNAL_PARAMS.get(mode, MODE_SIGNAL_PARAMS["NORMAL"])
    param_overrides = {
        "adx_min":   max(8.0,  base["adx_min"]   + adx_delta),
        "min_score": max(0.40, min(0.92, base["min_score"] + score_delta)),
    }

    equity      = start_equity
    peak_equity = start_equity
    trades:       List[Dict] = []
    equity_curve: List[Dict] = [{"ts": main_candles[WARMUP_CANDLES]["t"], "equity": equity}]
    open_trade: Optional[Dict] = None
    total_candles = len(main_candles) - WARMUP_CANDLES

    for i in range(WARMUP_CANDLES, len(main_candles)):
        candle = main_candles[i]

        # Exit check
        if open_trade:
            result = _sim_check_exit(open_trade, candle, exchange)
            if result and result.get("closed"):
                equity += result["pnl"]
                peak_equity = max(peak_equity, equity)
                trades.append({
                    "pnl":     result["pnl"],
                    "outcome": result["outcome"],
                    "side":    open_trade["side"],
                    "grade":   open_trade["grade"],
                    "signal":  open_trade.get("signal", "-"),
                    "score":   open_trade.get("score", 0.0),
                    "regime":  open_trade.get("regime", "-"),
                    "open_ts": open_trade["open_ts"],
                    "exit_ts": result["exit_ts"],
                    "entry":   open_trade["entry"],
                })
                equity_curve.append({"ts": result["exit_ts"], "equity": round(equity, 4)})
                open_trade = None

        # Hard floor
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if drawdown >= 0.15:
            break

        # Entry check
        if open_trade is None:
            hist_sess = _session_quality_for_ts(style, candle["t"])
            if hist_sess == 0.0:
                continue

            klines       = main_candles[max(0, i - 299): i + 1]
            higher_slice = _get_aligned_slice(higher_candles, candle["t"], 100)

            sig = _compute_signal_layers(
                klines, mode, 1.0, higher_slice, style, None,
                param_overrides=param_overrides,
            )
            if not sig.get("ok"):
                continue

            side  = sig["side"]
            grade = sig.get("grade", "B")
            atr_pct       = sig.get("atr_pct", ATR_BASELINE.get(tf, 0.009))
            mtf_size_mult = sig.get("mtf_size_mult", 1.0)

            sl_pct = min(atr_pct * st["sl_atr"] * sl_mult, st["sl_max"] / 100)
            tp_pct = min(atr_pct * st["tp_atr"] * tp_mult, st["tp_max"] / 100)

            baseline  = ATR_BASELINE.get(tf, 0.009)
            vol_ratio = atr_pct / baseline if baseline > 0 else 1.0
            vol_mult  = max(0.4, 1.0 / vol_ratio) if vol_ratio > 1.5 else 1.0

            dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            dd_mult = (
                0.25 if dd_pct >= 0.10 else
                0.40 if dd_pct >= 0.07 else
                0.65 if dd_pct >= 0.04 else 1.0
            )

            effective_size = c["size"] * mtf_size_mult * vol_mult * dd_mult

            open_trade = {
                "entry":              candle["close"],
                "side":               side,
                "grade":              grade,
                "sl_pct":             sl_pct,
                "tp_pct":             tp_pct,
                "effective_size":     effective_size,
                "leverage":           c["leverage"],
                "equity_at_open":     equity,
                "t1_size":            effective_size * 0.60,
                "t2_size":            effective_size * 0.40,
                "t1_done":            False,
                "t2_sl_is_breakeven": False,
                "accumulated_pnl":    0.0,
                "open_ts":            candle["t"],
                "signal":             sig.get("signal", "-"),
                "score":              round(sig.get("score", 0.0), 3),
                "regime":             sig.get("market_regime", "-"),
            }

    if len(trades) < MIN_TRADES:
        return None

    buy_hold_entry = main_candles[WARMUP_CANDLES]["close"]
    buy_hold_exit  = main_candles[-1]["close"]
    return _calc_metrics(trades, equity_curve, start_equity, buy_hold_entry, buy_hold_exit)


# ── Background worker ───────────────────────────────────────────────────────────
def _run_opt_worker(
    run_id: str,
    email: str,
    symbol: str,
    mode: str,
    style: str,
    exchange: str,
    start_ms: int,
    end_ms: int,
    start_equity: float,
) -> None:
    tf        = TF_FROM_STYLE[style]
    higher_tf = HIGHER_TF_MAP.get(tf, tf)
    combos    = list(itertools.product(
        OPT_GRID["adx_delta"],
        OPT_GRID["score_delta"],
        OPT_GRID["sl_mult"],
        OPT_GRID["tp_mult"],
    ))
    total = len(combos)
    _last_db = [0.0]

    def _set(status: str, done: int = 0, error: str = None):
        pct = int(done / max(1, total) * 100)
        with _active_lock:
            _active[run_id] = {"status": status, "progress": pct,
                                "done": done, "total": total, "error": error}
        now = time.time()
        if now - _last_db[0] >= 5:
            _last_db[0] = now
            try:
                with db_conn() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE optimizer_runs SET status=%s, progress=%s,"
                        " done_combos=%s WHERE run_id=%s",
                        (status, pct, done, run_id),
                    )
                    conn.commit()
            except Exception:
                pass

    try:
        _set("loading", 0)
        main_candles   = _ensure_candles(symbol, tf, start_ms, end_ms)
        higher_candles = _ensure_candles(symbol, higher_tf, start_ms, end_ms)

        if len(main_candles) < WARMUP_CANDLES + 10:
            raise ValueError(
                f"Not enough {tf} candles for {symbol} — run the backtester first "
                f"for this symbol/date range to cache candles."
            )

        _set("running", 0)
        results: List[Dict] = []

        for i, (adx_d, score_d, sl_m, tp_m) in enumerate(combos):
            metrics = _sim_one(
                main_candles, higher_candles,
                mode, style, exchange, start_equity,
                adx_d, score_d, sl_m, tp_m,
            )
            if metrics:
                wr = metrics.get("win_rate_pct", 0) / 100
                dd = abs(metrics.get("max_drawdown_pct", 0)) / 100
                passed = (
                    metrics.get("total_trades", 0) >= MIN_TRADES
                    and wr >= MIN_WIN_RATE
                    and dd <= MAX_DRAWDOWN
                )
                results.append({
                    "adx_delta":     adx_d,
                    "score_delta":   score_d,
                    "sl_mult":       sl_m,
                    "tp_mult":       tp_m,
                    "total_trades":  metrics.get("total_trades", 0),
                    "win_rate":      round(wr, 4),
                    "total_return":  round(metrics.get("total_return_pct", 0), 2),
                    "max_drawdown":  round(metrics.get("max_drawdown_pct", 0), 2),
                    "sharpe_ratio":  round(metrics.get("sharpe_ratio", 0), 3),
                    "reward_risk":   round(metrics.get("reward_risk", 0), 3),
                    "passed_filters": 1 if passed else 0,
                })
            if i % 5 == 0:
                _set("running", i + 1)

        # Sort passing combos first, then by Sharpe descending within each group
        results.sort(key=lambda r: (r["passed_filters"], r["sharpe_ratio"]), reverse=True)
        top = results[:TOP_RESULTS]

        with db_conn() as conn:
            cur = conn.cursor()
            now_iso = datetime.utcnow().isoformat()
            for r in top:
                cur.execute(
                    "INSERT INTO optimizer_results"
                    "(run_id,params_json,total_trades,win_rate,total_return,"
                    "max_drawdown,sharpe_ratio,reward_risk,passed_filters,created_at)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    if USING_PG else
                    "INSERT INTO optimizer_results"
                    "(run_id,params_json,total_trades,win_rate,total_return,"
                    "max_drawdown,sharpe_ratio,reward_risk,passed_filters,created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (run_id, json.dumps({
                        "adx_delta":   r["adx_delta"],
                        "score_delta": r["score_delta"],
                        "sl_mult":     r["sl_mult"],
                        "tp_mult":     r["tp_mult"],
                    }),
                    r["total_trades"], r["win_rate"], r["total_return"],
                    r["max_drawdown"], r["sharpe_ratio"], r["reward_risk"],
                    r["passed_filters"], now_iso),
                )
            cur.execute(
                "UPDATE optimizer_runs SET status=%s, progress=100,"
                " done_combos=%s, completed_at=%s WHERE run_id=%s",
                ("done", total, now_iso, run_id),
            )
            conn.commit()

        with _active_lock:
            _active[run_id] = {"status": "done", "progress": 100,
                                "done": total, "total": total}

    except Exception as exc:
        err = str(exc)
        with _active_lock:
            _active[run_id] = {"status": "error", "progress": 0, "error": err}
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE optimizer_runs SET status=%s, error=%s,"
                    " completed_at=%s WHERE run_id=%s",
                    ("error", err, datetime.utcnow().isoformat(), run_id),
                )
                conn.commit()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────────
def _parse_ms(date_str: str, end_of_day: bool = False) -> int:
    from datetime import timezone as tz
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.replace(tzinfo=tz.utc).timestamp() * 1000)


def start_optimizer(
    email: str,
    symbol: str,
    mode: str,
    style: str,
    exchange: str,
    date_from: str,
    date_to: str,
    start_equity: float = 1000.0,
) -> str:
    start_ms = _parse_ms(date_from)
    end_ms   = _parse_ms(date_to, end_of_day=True)
    run_id   = str(uuid.uuid4())
    tf       = TF_FROM_STYLE.get(style, "1h")

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
            (run_id, email, symbol.upper(), tf, mode, style, exchange,
             date_from, date_to, start_equity, "pending", 0,
             len(OPT_GRID["adx_delta"]) * len(OPT_GRID["score_delta"]) *
             len(OPT_GRID["sl_mult"]) * len(OPT_GRID["tp_mult"]),
             0, datetime.utcnow().isoformat()),
        )
        conn.commit()

    t = threading.Thread(
        target=_run_opt_worker,
        args=(run_id, email, symbol.upper(), mode, style, exchange,
              start_ms, end_ms, start_equity),
        daemon=True,
    )
    t.start()
    return run_id


def get_opt_run(run_id: str) -> Optional[Dict]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id, email, symbol, tf, mode, style, exchange,"
            " date_from, date_to, start_equity, status, progress,"
            " total_combos, done_combos, created_at, completed_at, error"
            " FROM optimizer_runs WHERE run_id=%s" if USING_PG else
            "SELECT run_id, email, symbol, tf, mode, style, exchange,"
            " date_from, date_to, start_equity, status, progress,"
            " total_combos, done_combos, created_at, completed_at, error"
            " FROM optimizer_runs WHERE run_id=?",
            (run_id,),
        )
        run = cur.fetchone()
        if not run:
            return None
        run = dict(run)

        cur.execute(
            "SELECT id, params_json, total_trades, win_rate, total_return,"
            " max_drawdown, sharpe_ratio, reward_risk, passed_filters, is_applied"
            " FROM optimizer_results WHERE run_id=%s"
            " ORDER BY passed_filters DESC, sharpe_ratio DESC LIMIT 50" if USING_PG else
            "SELECT id, params_json, total_trades, win_rate, total_return,"
            " max_drawdown, sharpe_ratio, reward_risk, passed_filters, is_applied"
            " FROM optimizer_results WHERE run_id=?"
            " ORDER BY passed_filters DESC, sharpe_ratio DESC LIMIT 50",
            (run_id,),
        )
        rows = cur.fetchall()

    with _active_lock:
        mem = _active.get(run_id)
    if mem:
        run["status"]   = mem["status"]
        run["progress"] = mem["progress"]
        run["done_combos"] = mem.get("done", run.get("done_combos", 0))
        if mem.get("error"):
            run["error"] = mem["error"]

    result_rows = rows or []
    run["results"] = [
        {
            "id":             r["id"],
            "params":         json.loads(r["params_json"]),
            "total_trades":   r["total_trades"],
            "win_rate_pct":   round(r["win_rate"] * 100, 1),
            "total_return":   r["total_return"],
            "max_drawdown":   r["max_drawdown"],
            "sharpe_ratio":   r["sharpe_ratio"],
            "reward_risk":    r["reward_risk"],
            "passed_filters": bool(r.get("passed_filters", 1)),
            "is_applied":     bool(r["is_applied"]),
        }
        for r in result_rows
    ]
    run["passed_count"] = sum(1 for r in result_rows if r.get("passed_filters", 1))
    return run


def list_opt_runs(email: str, limit: int = 10) -> List[Dict]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id, symbol, tf, mode, style, exchange,"
            " date_from, date_to, status, progress, total_combos, done_combos,"
            " created_at, completed_at, error"
            " FROM optimizer_runs WHERE email=%s"
            " ORDER BY created_at DESC LIMIT %s" if USING_PG else
            "SELECT run_id, symbol, tf, mode, style, exchange,"
            " date_from, date_to, status, progress, total_combos, done_combos,"
            " created_at, completed_at, error"
            " FROM optimizer_runs WHERE email=?"
            " ORDER BY created_at DESC LIMIT ?",
            (email, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def apply_opt_params(result_id: int, email: str) -> Dict:
    """Mark a result as applied and save params to optimizer_applied table."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT r.params_json, r.sharpe_ratio, r.run_id,"
            " o.symbol, o.mode, o.style"
            " FROM optimizer_results r"
            " JOIN optimizer_runs o ON o.run_id = r.run_id"
            " WHERE r.id=%s" if USING_PG else
            "SELECT r.params_json, r.sharpe_ratio, r.run_id,"
            " o.symbol, o.mode, o.style"
            " FROM optimizer_results r"
            " JOIN optimizer_runs o ON o.run_id = r.run_id"
            " WHERE r.id=?",
            (result_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("Result not found")

        now_iso = datetime.utcnow().isoformat()
        if USING_PG:
            cur.execute(
                "INSERT INTO optimizer_applied"
                "(symbol,mode,style,params_json,sharpe,applied_at,applied_by)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (symbol,mode,style) DO UPDATE"
                " SET params_json=EXCLUDED.params_json,"
                "     sharpe=EXCLUDED.sharpe,"
                "     applied_at=EXCLUDED.applied_at,"
                "     applied_by=EXCLUDED.applied_by",
                (row["symbol"], row["mode"], row["style"],
                 row["params_json"], row["sharpe_ratio"], now_iso, email),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO optimizer_applied"
                "(symbol,mode,style,params_json,sharpe,applied_at,applied_by)"
                " VALUES (?,?,?,?,?,?,?)",
                (row["symbol"], row["mode"], row["style"],
                 row["params_json"], row["sharpe_ratio"], now_iso, email),
            )
        cur.execute(
            "UPDATE optimizer_results SET is_applied=1 WHERE id=%s" if USING_PG
            else "UPDATE optimizer_results SET is_applied=1 WHERE id=?",
            (result_id,),
        )
        conn.commit()

    return {
        "symbol":  row["symbol"],
        "mode":    row["mode"],
        "style":   row["style"],
        "params":  json.loads(row["params_json"]),
        "sharpe":  row["sharpe_ratio"],
        "message": f"Params applied for {row['symbol']} {row['mode']} {row['style']}",
    }


def list_applied_params() -> List[Dict]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, mode, style, params_json, sharpe, applied_at, applied_by"
            " FROM optimizer_applied ORDER BY applied_at DESC"
        )
        return [
            {**dict(r), "params": json.loads(r["params_json"])}
            for r in cur.fetchall()
        ]


def get_applied_params(symbol: str, mode: str, style: str) -> Optional[Dict]:
    """Called by the live engine to get optimized params for this symbol/mode/style."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT params_json FROM optimizer_applied"
            " WHERE symbol=%s AND mode=%s AND style=%s" if USING_PG else
            "SELECT params_json FROM optimizer_applied"
            " WHERE symbol=? AND mode=? AND style=?",
            (symbol.upper(), mode, style),
        )
        row = cur.fetchone()
    return json.loads(row["params_json"]) if row else None


def get_opt_results_csv(run_id: str) -> Optional[str]:
    """Return all optimizer results for a run as a CSV string."""
    import csv, io
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT r.symbol, r.mode, r.style, r.date_from, r.date_to,"
            " o.params_json, o.total_trades, o.win_rate, o.total_return,"
            " o.max_drawdown, o.sharpe_ratio, o.reward_risk, o.passed_filters, o.is_applied"
            " FROM optimizer_results o"
            " JOIN optimizer_runs r ON r.run_id = o.run_id"
            " WHERE o.run_id=%s ORDER BY o.passed_filters DESC, o.sharpe_ratio DESC" if USING_PG else
            "SELECT r.symbol, r.mode, r.style, r.date_from, r.date_to,"
            " o.params_json, o.total_trades, o.win_rate, o.total_return,"
            " o.max_drawdown, o.sharpe_ratio, o.reward_risk, o.passed_filters, o.is_applied"
            " FROM optimizer_results o"
            " JOIN optimizer_runs r ON r.run_id = o.run_id"
            " WHERE o.run_id=? ORDER BY o.passed_filters DESC, o.sharpe_ratio DESC",
            (run_id,),
        )
        rows = cur.fetchall()
    if not rows:
        return None
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "symbol", "mode", "style", "date_from", "date_to",
        "adx_delta", "score_delta", "sl_mult", "tp_mult",
        "total_trades", "win_rate_pct", "total_return_pct",
        "max_drawdown_pct", "sharpe_ratio", "reward_risk",
        "passed_filters", "is_applied",
    ])
    for r in rows:
        p = json.loads(r["params_json"])
        writer.writerow([
            r["symbol"], r["mode"], r["style"], r["date_from"], r["date_to"],
            p.get("adx_delta", 0), p.get("score_delta", 0),
            p.get("sl_mult", 1.0), p.get("tp_mult", 1.0),
            r["total_trades"],
            round(r["win_rate"] * 100, 1),
            r["total_return"], r["max_drawdown"],
            r["sharpe_ratio"], r["reward_risk"],
            1 if r.get("passed_filters", 1) else 0,
            1 if r.get("is_applied", 0) else 0,
        ])
    return buf.getvalue()
