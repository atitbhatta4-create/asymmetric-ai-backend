"""
backtester.py — Historical simulation engine for Asymmetric AI.

Fetches OHLCV candles from OKX (same source as the live engine; Bybit and
Binance are geo-blocked on Render US servers), runs the exact signal logic
and P&L rules on historical data, and stores results in the DB.

Wire up in main.py with two lines:
    from routes_backtest import backtest_router
    app.include_router(backtest_router)
"""
from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from config import DUBAI_TZ, HIGHER_TF_MAP, OKX_BASE, OKX_TF_MAP, TRADE_STYLE_PARAMS
from database import db_conn, serial_pk, USING_PG
from indicators import _atr, _compute_signal_layers

# ── Constants ──────────────────────────────────────────────────────────────────
WARMUP_CANDLES = 240  # extra buffer above the 220-candle minimum in indicators.py

TF_FROM_STYLE: Dict[str, str] = {
    "SCALP":     "15m",
    "DAY_TRADE": "1h",
    "SWING":     "4h",
}

MODE_PRESETS: Dict[str, Dict] = {
    "ULTRA_SAFE": {"size": 0.30, "leverage": 2},
    "SAFE":       {"size": 0.45, "leverage": 3},
    "NORMAL":     {"size": 0.60, "leverage": 5},
    "MINI_ASYM":  {"size": 0.65, "leverage": 6},
    "AGGRESSIVE": {"size": 0.85, "leverage": 8},
}

ATR_BASELINE: Dict[str, float] = {
    "15m": 0.0040,
    "1h":  0.0090,
    "4h":  0.0180,
    "1d":  0.0350,
}

# Fee model per exchange (maker entry + limit/taker exit + slippage)
EXCHANGE_FEES: Dict[str, Dict] = {
    "bybit":   {"maker": 0.00020, "taker": 0.00055, "slippage": 0.00025},
    "okx":     {"maker": 0.00020, "taker": 0.00050, "slippage": 0.00030},
    "binance": {"maker": 0.00020, "taker": 0.00040, "slippage": 0.00020},
}

OKX_HIST_CANDLES_LIMIT = 100  # OKX history-candles max per request

# Active runs tracked in memory: run_id → {status, progress, error}
_active: Dict[str, Dict] = {}
_active_lock = threading.Lock()


# ── Symbol helpers ─────────────────────────────────────────────────────────────
def _to_okx_inst(symbol: str) -> str:
    """BTCUSDT / BTCUSDC / ETHBTC → BTC-USDT / BTC-USDC / ETH-BTC  (OKX format)."""
    s = symbol.upper().strip()
    if "-" in s:
        return s
    for quote in ("USDT", "USDC", "BTC", "ETH", "BNB"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}-{quote}"
    return s


# ── Fee helper ─────────────────────────────────────────────────────────────────
def _fee_cost(size_dollar: float, outcome: str, exchange: str = "bybit") -> float:
    """Total round-trip fee + slippage for one trade."""
    f = EXCHANGE_FEES.get(exchange, EXCHANGE_FEES["bybit"])
    entry_fee = size_dollar * f["maker"]
    exit_fee  = size_dollar * (f["maker"] if outcome == "TP_HIT" else f["taker"])
    slippage  = size_dollar * f["slippage"]
    return round(entry_fee + exit_fee + slippage, 4)


# ── Session quality from historical timestamp ──────────────────────────────────
def _session_quality_for_ts(trade_style: str, ts_ms: int) -> float:
    """Return session quality score based on the candle's actual timestamp."""
    if trade_style == "SWING":
        return 1.0
    hour = datetime.fromtimestamp(ts_ms / 1000, tz=DUBAI_TZ).hour
    if 2 <= hour < 6:
        return 0.0 if trade_style == "SCALP" else 0.72
    if 6  <= hour < 12: return 0.85
    if 12 <= hour < 16: return 0.95
    if 16 <= hour < 21: return 1.00
    if 21 <= hour <= 23: return 0.92
    return 0.80  # 01:00-02:00 NY close


# ── DB helpers ─────────────────────────────────────────────────────────────────
def _insert_candle_sql() -> str:
    if USING_PG:
        return (
            "INSERT INTO backtest_candles(symbol,tf,ts,open,high,low,close,volume)"
            " VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING"
        )
    return (
        "INSERT OR IGNORE INTO backtest_candles(symbol,tf,ts,open,high,low,close,volume)"
        " VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
    )


def init_backtest_tables() -> None:
    """Create backtester tables if not already present. Called once at startup."""
    with db_conn() as conn:
        cur = conn.cursor()
        pk = serial_pk()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS backtest_candles (
                id     {pk},
                symbol TEXT    NOT NULL,
                tf     TEXT    NOT NULL,
                ts     BIGINT  NOT NULL,
                open   REAL    NOT NULL,
                high   REAL    NOT NULL,
                low    REAL    NOT NULL,
                close  REAL    NOT NULL,
                volume REAL    NOT NULL,
                UNIQUE(symbol, tf, ts)
            )
        """)
        # Migrate existing INTEGER column to BIGINT (ms timestamps overflow INT)
        if USING_PG:
            try:
                cur.execute(
                    "ALTER TABLE backtest_candles ALTER COLUMN ts TYPE BIGINT"
                )
            except Exception:
                pass  # already BIGINT or table doesn't exist yet
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_bt_candles_sym_tf_ts"
            " ON backtest_candles(symbol, tf, ts)"
        )
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id       TEXT PRIMARY KEY,
                email        TEXT NOT NULL,
                symbol       TEXT NOT NULL,
                tf           TEXT NOT NULL,
                mode         TEXT NOT NULL,
                style        TEXT NOT NULL,
                exchange     TEXT NOT NULL DEFAULT 'bybit',
                date_from    TEXT NOT NULL,
                date_to      TEXT NOT NULL,
                start_equity REAL NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                progress     INTEGER       DEFAULT 0,
                created_at   TEXT NOT NULL,
                completed_at TEXT,
                error        TEXT,
                result_json  TEXT
            )
        """)
        conn.commit()


# ── OKX candle fetching ────────────────────────────────────────────────────────
def _fetch_okx_history_page(
    inst: str, bar: str, after_ms: Optional[int] = None
) -> List[Dict]:
    """Fetch one page (≤100) of OKX history-candles. Returns chronological list."""
    params: Dict = {"instId": inst, "bar": bar, "limit": OKX_HIST_CANDLES_LIMIT}
    if after_ms is not None:
        params["after"] = str(after_ms)
    try:
        r = httpx.get(
            f"{OKX_BASE}/api/v5/market/history-candles",
            params=params,
            timeout=20,
            headers={"accept": "application/json"},
        )
        if r.status_code != 200:
            return []
        rows = (r.json() or {}).get("data") or []
        out: List[Dict] = []
        for k in rows:
            # OKX returns: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
            out.append({
                "t":      int(k[0]),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            })
        out.reverse()  # chronological (oldest first)
        return out
    except Exception:
        return []


def _ensure_candles(
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    on_progress: Optional[callable] = None,
) -> List[Dict]:
    """
    Return candles for symbol/tf in [start_ms, end_ms] from DB.
    Fetches from OKX and stores in DB any missing range.
    Candles are returned in chronological order (oldest first).
    """
    inst = _to_okx_inst(symbol)
    bar  = OKX_TF_MAP.get(tf)
    if not bar:
        return []

    # Check what's already in DB
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM backtest_candles"
            " WHERE symbol=%s AND tf=%s AND ts>=%s AND ts<=%s",
            (symbol.upper(), tf, start_ms, end_ms),
        )
        row = cur.fetchone()
        cached_count = row["cnt"] if row else 0

    # Estimate expected candle count from date range
    tf_ms = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    interval_ms = tf_ms.get(tf, 3_600_000)
    expected = max(1, (end_ms - start_ms) // interval_ms)

    if cached_count >= int(expected * 0.97):  # 97% threshold — small gaps OK
        pass  # already have enough
    else:
        # Fetch from OKX and store — paginate backwards from end_ms to start_ms
        insert_sql = _insert_candle_sql()
        current_after = end_ms
        pages_fetched = 0

        with db_conn() as conn:
            cur = conn.cursor()
            while True:
                page = _fetch_okx_history_page(inst, bar, after_ms=current_after)
                if not page:
                    break

                for c in page:
                    if c["t"] < start_ms:
                        continue
                    cur.execute(
                        insert_sql,
                        (symbol.upper(), tf, c["t"],
                         c["open"], c["high"], c["low"], c["close"], c["volume"]),
                    )

                conn.commit()
                pages_fetched += 1

                oldest_in_page = page[0]["t"]  # page is chronological, oldest = index 0
                if oldest_in_page <= start_ms:
                    break

                current_after = oldest_in_page
                if on_progress:
                    on_progress(pages_fetched)
                time.sleep(0.25)  # OKX rate limit

    # Read back from DB in chronological order
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT ts, open, high, low, close, volume FROM backtest_candles"
            " WHERE symbol=%s AND tf=%s AND ts>=%s AND ts<=%s"
            " ORDER BY ts ASC",
            (symbol.upper(), tf, start_ms, end_ms),
        )
        rows = cur.fetchall()

    return [
        {"t": r["ts"], "open": r["open"], "high": r["high"],
         "low": r["low"], "close": r["close"], "volume": r["volume"]}
        for r in rows
    ]


# ── Simulation helpers ─────────────────────────────────────────────────────────
def _get_aligned_slice(
    candles: List[Dict], before_ts: int, n: int
) -> List[Dict]:
    """
    Return the last n candles with ts ≤ before_ts (chronological order).
    Used to align higher-TF and MTF candles to the current simulation candle.
    """
    # Binary search for the rightmost candle with t ≤ before_ts
    lo, hi = 0, len(candles) - 1
    idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if candles[mid]["t"] <= before_ts:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if idx < 0:
        return []
    start = max(0, idx - n + 1)
    return candles[start : idx + 1]


def _sim_check_exit(trade: Dict, candle: Dict, exchange: str) -> Optional[Dict]:
    """
    Check whether the open trade exits on this candle.
    Mutates trade dict for Grade B T1/T2 state.
    Returns dict with {closed, outcome, pnl} or None if trade stays open.
    """
    side     = trade["side"]
    entry    = trade["entry"]
    sl_pct   = trade["sl_pct"]
    tp_pct   = trade["tp_pct"]
    grade    = trade["grade"]
    leverage = trade["leverage"]
    equity   = trade["equity_at_open"]

    high = candle["high"]
    low  = candle["low"]

    def sl_hit(price: float) -> bool:
        return low <= price if side == "LONG" else high >= price

    def tp_hit(price: float) -> bool:
        return high >= price if side == "LONG" else low <= price

    sl_price = entry * (1 - sl_pct) if side == "LONG" else entry * (1 + sl_pct)
    tp_price = entry * (1 + tp_pct) if side == "LONG" else entry * (1 - tp_pct)

    if grade == "B":
        t1_tp_pct   = tp_pct * 0.80
        t1_tp_price = entry * (1 + t1_tp_pct) if side == "LONG" else entry * (1 - t1_tp_pct)

        t1_just_closed = False

        if not trade.get("t1_done"):
            t1_dollar = equity * trade["t1_size"]

            if sl_hit(sl_price):
                # SL hit before T1 TP — both legs close at SL
                t1_pnl = t1_dollar * (-sl_pct * leverage) - _fee_cost(t1_dollar, "SL_HIT", exchange)
                t2_pnl = equity * trade["t2_size"] * (-sl_pct * leverage) - _fee_cost(equity * trade["t2_size"], "SL_HIT", exchange)
                return {
                    "closed": True,
                    "outcome": "SL_HIT",
                    "pnl": trade["accumulated_pnl"] + t1_pnl + t2_pnl,
                    "exit_ts": candle["t"],
                }

            if tp_hit(t1_tp_price):
                # T1 TP hit — partial close; T2 SL moves to breakeven next candle
                t1_pnl = t1_dollar * (t1_tp_pct * leverage) - _fee_cost(t1_dollar, "TP_HIT", exchange)
                trade["accumulated_pnl"] += t1_pnl
                trade["t1_done"] = True
                trade["t2_sl_is_breakeven"] = True
                t1_just_closed = True

        # Check T2 — only if T1 was already done before this candle
        if trade.get("t1_done") and not t1_just_closed:
            t2_dollar = equity * trade["t2_size"]

            if trade.get("t2_sl_is_breakeven"):
                t2_sl_price = entry  # breakeven = entry
            else:
                t2_sl_price = sl_price

            if tp_hit(tp_price):
                t2_pnl = t2_dollar * (tp_pct * leverage) - _fee_cost(t2_dollar, "TP_HIT", exchange)
                return {
                    "closed": True,
                    "outcome": "T1_TP_T2_TP",
                    "pnl": trade["accumulated_pnl"] + t2_pnl,
                    "exit_ts": candle["t"],
                }

            if sl_hit(t2_sl_price):
                if trade.get("t2_sl_is_breakeven"):
                    # Price retraced to entry — close T2 at 0% P&L (only fees)
                    t2_pnl = -_fee_cost(t2_dollar, "SL_HIT", exchange)
                    outcome = "T1_TP_T2_BE"
                else:
                    t2_pnl = t2_dollar * (-sl_pct * leverage) - _fee_cost(t2_dollar, "SL_HIT", exchange)
                    outcome = "T1_TP_T2_SL"
                return {
                    "closed": True,
                    "outcome": outcome,
                    "pnl": trade["accumulated_pnl"] + t2_pnl,
                    "exit_ts": candle["t"],
                }

        return None  # trade still open

    else:
        # Grade A — standard single exit
        size_dollar = equity * trade["effective_size"]

        if tp_hit(tp_price) and sl_hit(sl_price):
            # Both hit same candle → SL wins (worst-case, conservative)
            pnl = size_dollar * (-sl_pct * leverage) - _fee_cost(size_dollar, "SL_HIT", exchange)
            return {"closed": True, "outcome": "SL_HIT", "pnl": pnl, "exit_ts": candle["t"]}

        if tp_hit(tp_price):
            pnl = size_dollar * (tp_pct * leverage) - _fee_cost(size_dollar, "TP_HIT", exchange)
            return {"closed": True, "outcome": "TP_HIT", "pnl": pnl, "exit_ts": candle["t"]}

        if sl_hit(sl_price):
            pnl = size_dollar * (-sl_pct * leverage) - _fee_cost(size_dollar, "SL_HIT", exchange)
            return {"closed": True, "outcome": "SL_HIT", "pnl": pnl, "exit_ts": candle["t"]}

        return None


# ── Metrics calculation ────────────────────────────────────────────────────────
def _calc_metrics(
    trades: List[Dict],
    equity_curve: List[Dict],
    start_equity: float,
    buy_hold_entry: float,
    buy_hold_exit: float,
) -> Dict:
    """Compute all statistics from a completed simulation."""
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0, "win_count": 0, "loss_count": 0, "win_rate": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "reward_risk": 0.0,
            "total_return_pct": 0.0, "final_equity": start_equity,
            "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
            "buy_hold_return_pct": 0.0,
            "monthly_returns": [], "equity_curve": equity_curve,
        }

    final_equity = equity_curve[-1]["equity"] if equity_curve else start_equity

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    win_rate     = round(len(wins) / total * 100, 1)
    avg_win_pct  = round(statistics.mean(w["pnl"] / start_equity * 100 for w in wins),  2) if wins   else 0.0
    avg_loss_pct = round(statistics.mean(l["pnl"] / start_equity * 100 for l in losses), 2) if losses else 0.0
    reward_risk  = round(abs(avg_win_pct / avg_loss_pct), 2) if avg_loss_pct != 0 else 0.0

    total_return_pct = round((final_equity - start_equity) / start_equity * 100, 2)

    # Max drawdown
    peak = start_equity
    max_dd = 0.0
    for pt in equity_curve:
        eq = pt["equity"]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = round(max_dd * 100, 2)

    # Monthly returns
    monthly: Dict[str, Dict] = {}
    prev_eq = start_equity
    for t in trades:
        ts   = t["exit_ts"]
        dt   = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        key  = dt.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"start_equity": prev_eq, "pnl": 0.0, "trades": 0}
        monthly[key]["pnl"]    += t["pnl"]
        monthly[key]["trades"] += 1
        prev_eq += t["pnl"]

    monthly_returns = []
    for key in sorted(monthly.keys()):
        m = monthly[key]
        ret_pct = round(m["pnl"] / m["start_equity"] * 100, 2) if m["start_equity"] > 0 else 0.0
        monthly_returns.append({"month": key, "return_pct": ret_pct, "trades": m["trades"]})

    # Sharpe ratio (annualised from monthly returns)
    monthly_rets = [m["return_pct"] / 100 for m in monthly_returns]
    if len(monthly_rets) >= 2:
        mean_m = statistics.mean(monthly_rets)
        std_m  = statistics.stdev(monthly_rets)
        sharpe = round((mean_m / std_m) * math.sqrt(12), 2) if std_m > 0 else 0.0
    else:
        sharpe = 0.0

    buy_hold_ret = round((buy_hold_exit - buy_hold_entry) / buy_hold_entry * 100, 2) if buy_hold_entry > 0 else 0.0

    return {
        "total_trades": total,
        "win_count":    len(wins),
        "loss_count":   len(losses),
        "win_rate":     win_rate,
        "avg_win_pct":  avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "reward_risk":  reward_risk,
        "total_return_pct": total_return_pct,
        "final_equity": round(final_equity, 2),
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "buy_hold_return_pct": buy_hold_ret,
        "monthly_returns": monthly_returns,
        "equity_curve": equity_curve,
    }


# ── Main simulation worker ─────────────────────────────────────────────────────
def _run_worker(
    run_id: str,
    email: str,
    symbol: str,
    tf: str,
    mode: str,
    style: str,
    exchange: str,
    start_ms: int,
    end_ms: int,
    start_equity: float,
) -> None:
    def _set(status: str, progress: int = 0, error: str = None):
        with _active_lock:
            _active[run_id] = {"status": status, "progress": progress, "error": error}

    _set("fetching", 0)

    try:
        higher_tf = HIGHER_TF_MAP.get(tf, tf)
        mtf_tf    = "15m" if style in ("DAY_TRADE", "SWING") and tf != "15m" else None

        # ── 1. Fetch candles (cached in DB after first fetch) ──────────────
        _set("fetching", 5)
        main_candles = _ensure_candles(symbol, tf, start_ms, end_ms)
        if len(main_candles) < WARMUP_CANDLES + 10:
            raise ValueError(
                f"Not enough {tf} candles for {symbol} ({len(main_candles)} fetched, "
                f"need at least {WARMUP_CANDLES + 10}). "
                f"Try a wider date range or check the symbol name."
            )

        _set("fetching", 25)
        higher_candles = _ensure_candles(symbol, higher_tf, start_ms, end_ms)

        _set("fetching", 40)
        mtf_candles: List[Dict] = []
        if mtf_tf:
            mtf_candles = _ensure_candles(symbol, mtf_tf, start_ms, end_ms)

        _set("running", 50)

        # ── 2. Simulation loop ─────────────────────────────────────────────
        st = TRADE_STYLE_PARAMS[style]
        c  = MODE_PRESETS[mode]

        equity      = start_equity
        peak_equity = start_equity
        trades:       List[Dict] = []
        equity_curve: List[Dict] = [{"ts": main_candles[WARMUP_CANDLES]["t"], "equity": equity}]

        open_trade: Optional[Dict] = None
        total_candles = len(main_candles) - WARMUP_CANDLES

        for i in range(WARMUP_CANDLES, len(main_candles)):
            candle = main_candles[i]

            # Progress reporting (50-99%)
            done_pct = (i - WARMUP_CANDLES) / max(1, total_candles)
            _set("running", 50 + int(done_pct * 49))

            # ── Check exit for open trade ──────────────────────────────
            if open_trade:
                result = _sim_check_exit(open_trade, candle, exchange)
                if result and result.get("closed"):
                    equity += result["pnl"]
                    peak_equity = max(peak_equity, equity)
                    trades.append({
                        "pnl":      result["pnl"],
                        "outcome":  result["outcome"],
                        "side":     open_trade["side"],
                        "grade":    open_trade["grade"],
                        "signal":   open_trade["signal"],
                        "score":    open_trade["score"],
                        "regime":   open_trade["regime"],
                        "open_ts":  open_trade["open_ts"],
                        "exit_ts":  result["exit_ts"],
                        "entry":    open_trade["entry"],
                    })
                    equity_curve.append({"ts": result["exit_ts"], "equity": round(equity, 4)})
                    open_trade = None

            # ── Drawdown stop (same as live engine) ───────────────────
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if drawdown >= 0.15:
                break

            # ── Try to open new trade ─────────────────────────────────
            if open_trade is None:
                klines = main_candles[max(0, i - 299) : i + 1]  # last 300, chron. order

                higher_slice = _get_aligned_slice(higher_candles, candle["t"], 100)
                mtf_slice    = _get_aligned_slice(mtf_candles, candle["t"], 50) if mtf_candles else None

                # Session quality for this historical candle's timestamp
                hist_sess = _session_quality_for_ts(style, candle["t"])
                if hist_sess == 0.0:
                    continue  # SCALP dead-zone block

                sig = _compute_signal_layers(klines, mode, 1.0, higher_slice, style, mtf_slice)

                if sig.get("ok"):
                    side          = sig["side"]
                    grade         = sig.get("grade", "B")
                    atr_pct       = sig.get("atr_pct", ATR_BASELINE.get(tf, 0.009))
                    mtf_size_mult = sig.get("mtf_size_mult", 1.0)

                    sl_pct = min(atr_pct * st["sl_atr"], st["sl_max"] / 100)
                    tp_pct = min(atr_pct * st["tp_atr"], st["tp_max"] / 100)

                    # Volatility-adjusted size (same formula as live engine)
                    baseline  = ATR_BASELINE.get(tf, 0.009)
                    vol_ratio = atr_pct / baseline if baseline > 0 else 1.0
                    vol_mult  = max(0.4, 1.0 / vol_ratio) if vol_ratio > 1.5 else 1.0

                    # Drawdown tier size adjustment
                    dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                    dd_mult = (
                        0.25 if dd_pct >= 0.10 else
                        0.40 if dd_pct >= 0.07 else
                        0.65 if dd_pct >= 0.04 else
                        1.0
                    )

                    effective_size = c["size"] * mtf_size_mult * vol_mult * dd_mult

                    open_trade = {
                        "entry":          candle["close"],
                        "side":           side,
                        "grade":          grade,
                        "sl_pct":         sl_pct,
                        "tp_pct":         tp_pct,
                        "effective_size": effective_size,
                        "leverage":       c["leverage"],
                        "equity_at_open": equity,
                        # Grade B legs
                        "t1_size":           effective_size * 0.60,
                        "t2_size":           effective_size * 0.40,
                        "t1_done":           False,
                        "t2_sl_is_breakeven": False,
                        "accumulated_pnl":   0.0,
                        # Metadata for result record
                        "open_ts": candle["t"],
                        "signal":  sig.get("signal", "-"),
                        "score":   round(sig.get("score", 0.0), 3),
                        "regime":  sig.get("market_regime", "-"),
                    }

        # ── 3. Calculate metrics ───────────────────────────────────────────
        buy_hold_entry = main_candles[WARMUP_CANDLES]["close"]
        buy_hold_exit  = main_candles[-1]["close"]
        result_json    = _calc_metrics(trades, equity_curve, start_equity, buy_hold_entry, buy_hold_exit)
        result_json["candles_fetched"]   = len(main_candles)
        result_json["sim_candles_used"]  = len(main_candles) - WARMUP_CANDLES
        result_json["session_note"] = (
            "Session quality computed at backtest run time (not historical candle time). "
            "SCALP dead-zone blocks are applied correctly; DAY_TRADE/SWING scores may "
            "vary slightly by ≤5% depending on when this backtest was run."
        )

        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE backtest_runs SET status=%s, progress=%s, completed_at=%s,"
                " result_json=%s WHERE run_id=%s",
                ("done", 100, datetime.utcnow().isoformat(),
                 json.dumps(result_json), run_id),
            )
            conn.commit()

        _set("done", 100)

    except Exception as exc:
        err = str(exc)[:500]
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE backtest_runs SET status=%s, error=%s, completed_at=%s WHERE run_id=%s",
                ("error", err, datetime.utcnow().isoformat(), run_id),
            )
            conn.commit()
        _set("error", 0, err)


# ── Public API ─────────────────────────────────────────────────────────────────
def start_backtest(
    email: str,
    symbol: str,
    mode: str,
    style: str,
    exchange: str,
    date_from: str,
    date_to: str,
    start_equity: float,
) -> str:
    """Validate inputs, create DB record, start background thread, return run_id."""
    from datetime import datetime

    tf = TF_FROM_STYLE.get(style)
    if not tf:
        raise ValueError(f"Unknown style '{style}'. Use SCALP, DAY_TRADE, or SWING.")
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown mode '{mode}'.")
    exchange = exchange.lower().strip()
    if exchange not in EXCHANGE_FEES:
        raise ValueError(f"Unknown exchange '{exchange}'. Use bybit, okx, or binance.")

    try:
        dt_from = datetime.strptime(date_from, "%Y-%m-%d")
        dt_to   = datetime.strptime(date_to,   "%Y-%m-%d")
    except ValueError:
        raise ValueError("date_from / date_to must be in YYYY-MM-DD format.")

    start_ms = int(dt_from.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(dt_to.replace(tzinfo=timezone.utc).timestamp() * 1000)

    if end_ms <= start_ms:
        raise ValueError("date_to must be after date_from.")
    if (end_ms - start_ms) < 86_400_000 * 7:
        raise ValueError("Date range must be at least 7 days.")
    if start_equity < 100:
        raise ValueError("start_equity must be at least $100.")

    # Check for already-running backtest for this email
    with _active_lock:
        for rid, state in _active.items():
            if state.get("status") in ("fetching", "running"):
                pass  # allow multiple runs

    run_id = str(uuid.uuid4())

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO backtest_runs(run_id,email,symbol,tf,mode,style,exchange,"
            "date_from,date_to,start_equity,status,progress,created_at)"
            " VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (run_id, email, symbol.upper(), tf, mode, style, exchange,
             date_from, date_to, start_equity, "pending", 0,
             datetime.utcnow().isoformat()),
        )
        conn.commit()

    t = threading.Thread(
        target=_run_worker,
        args=(run_id, email, symbol.upper(), tf, mode, style, exchange,
              start_ms, end_ms, start_equity),
        daemon=True,
    )
    t.start()

    return run_id


def get_run(run_id: str) -> Optional[Dict]:
    """Return full run record from DB (or None if not found)."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id, email, symbol, tf, mode, style, exchange,"
            " date_from, date_to, start_equity, status, progress,"
            " created_at, completed_at, error, result_json"
            " FROM backtest_runs WHERE run_id=%s",
            (run_id,),
        )
        row = cur.fetchone()
    if not row:
        return None

    # Overlay in-memory progress (more up-to-date than DB during a run)
    with _active_lock:
        mem = _active.get(run_id)
    if mem:
        row = dict(row)
        row["status"]   = mem["status"]
        row["progress"] = mem["progress"]
        if mem.get("error"):
            row["error"] = mem["error"]

    if row.get("result_json"):
        try:
            row["results"] = json.loads(row["result_json"])
        except Exception:
            row["results"] = None
        del row["result_json"]
    else:
        row["results"] = None

    return dict(row)


def list_runs(email: str, limit: int = 20) -> List[Dict]:
    """Return latest runs for a user (most recent first)."""
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id, symbol, tf, mode, style, exchange,"
            " date_from, date_to, start_equity, status, progress,"
            " created_at, completed_at, error"
            " FROM backtest_runs WHERE email=%s"
            " ORDER BY created_at DESC LIMIT %s",
            (email, limit),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def compare_runs(run_id_a: str, run_id_b: str) -> Dict:
    """Return side-by-side key metrics for two completed runs."""
    a = get_run(run_id_a)
    b = get_run(run_id_b)
    if not a or not b:
        raise ValueError("One or both run IDs not found.")

    def _summary(r: Dict) -> Dict:
        res = r.get("results") or {}
        return {
            "run_id":      r["run_id"],
            "symbol":      r["symbol"],
            "mode":        r["mode"],
            "style":       r["style"],
            "date_from":   r["date_from"],
            "date_to":     r["date_to"],
            "total_trades":      res.get("total_trades", 0),
            "win_rate":          res.get("win_rate", 0),
            "total_return_pct":  res.get("total_return_pct", 0),
            "max_drawdown_pct":  res.get("max_drawdown_pct", 0),
            "sharpe_ratio":      res.get("sharpe_ratio", 0),
            "buy_hold_return_pct": res.get("buy_hold_return_pct", 0),
            "final_equity":      res.get("final_equity", 0),
        }

    return {"a": _summary(a), "b": _summary(b)}
