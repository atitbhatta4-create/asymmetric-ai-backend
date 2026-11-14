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
import random as _random_module
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

from config import DUBAI_TZ, HIGHER_TF_MAP, OKX_BASE, OKX_TF_MAP, TRADE_STYLE_PARAMS
from database import db_conn, serial_pk, USING_PG
from indicators import (
    _atr, _adx, _adx_series, _atr_series, _rsi_series, _ema,
    _compute_signal_layers,
    get_market_regime, check_weekly_trend,
    check_pullback_entry, check_volume_hour_confirmed,
    compute_grid_signal, donchian_htf_filter, compute_grid_v2_band,
)
from coin_params import get_coin_params
from mode_params import get_mode_config, get_style_config

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

# Mirror of main.py TRAIL_LEVELS — must stay in sync with live engine.
# (tp_fraction, lock_fraction, level_name)
TRAIL_LEVELS = [
    (0.35, 0.00, "L1-BE"),
    (0.40, 0.05, "L2"),
    (0.50, 0.25, "L3"),
    (0.60, 0.35, "L4"),
    (0.70, 0.50, "L5"),
    (0.80, 0.65, "L6"),
    (0.90, 0.80, "L7"),
]


def _sim_floor(peak: float, start: float) -> float:
    """R1 Hybrid floor — mirrors live _compute_floor() in user_state.py."""
    if peak >= 2.0 * start:
        return max(start * 0.85, peak * 0.90)
    return peak * 0.85


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
        # On restart, any run still "pending/fetching/running" was killed mid-flight
        cur.execute(
            "UPDATE backtest_runs SET status=%s, error=%s, completed_at=%s"
            " WHERE status IN ('pending','fetching','running')",
            ("error", "Server restarted — please retry",
             datetime.utcnow().isoformat()),
        )
        conn.commit()


# ── OKX candle fetching ────────────────────────────────────────────────────────
def _fetch_okx_history_page(
    inst: str, bar: str, after_ms: Optional[int] = None
) -> List[Dict]:
    """Fetch one page (≤100) of OKX history-candles with exponential-backoff retry."""
    params: Dict = {"instId": inst, "bar": bar, "limit": OKX_HIST_CANDLES_LIMIT}
    if after_ms is not None:
        params["after"] = str(after_ms)
    for attempt in range(4):  # 1 initial + 3 retries
        try:
            r = httpx.get(
                f"{OKX_BASE}/api/v5/market/history-candles",
                params=params,
                timeout=20,
                headers={"accept": "application/json"},
            )
            if r.status_code == 200:
                rows = (r.json() or {}).get("data") or []
                out: List[Dict] = []
                for k in rows:
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
            # 429 = rate-limited, 5xx = server error — worth retrying
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(2.0 * (2 ** attempt))  # 2 s, 4 s, 8 s
                continue
            # 4xx non-429 (bad request, not found): no point retrying
            return []
        except Exception:
            if attempt < 3:
                time.sleep(2.0 * (2 ** attempt))
                continue
            return []
    return []


def _ensure_candles(
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    on_progress: Optional[callable] = None,
    force_refetch: bool = False,
) -> List[Dict]:
    """
    Return candles for symbol/tf in [start_ms, end_ms] from DB.
    Fetches from OKX and stores in DB any missing range.
    force_refetch=True deletes cached rows first, forcing a clean re-fetch.
    Candles are returned in chronological order (oldest first).
    """
    inst = _to_okx_inst(symbol)
    bar  = OKX_TF_MAP.get(tf)
    if not bar:
        return []

    if force_refetch:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM backtest_candles WHERE symbol=%s AND tf=%s AND ts>=%s AND ts<=%s",
                (symbol.upper(), tf, start_ms, end_ms),
            )
            conn.commit()

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


def check_candle_coverage(
    symbols: List[str],
    tfs: List[str],
    start_ms: int,
    end_ms: int,
    threshold: float = 0.97,
) -> Tuple[bool, List[Dict]]:
    """
    Check candle coverage for all symbols × tfs.
    Returns (all_ok, rows) where all_ok is False if any entry is under threshold.
    Call this before running baseline to catch incomplete caches early.
    """
    _tf_ms = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1W": 604_800_000}
    rows: List[Dict] = []
    all_ok = True

    with db_conn() as conn:
        cur = conn.cursor()
        for sym in symbols:
            for tf in tfs:
                interval_ms = _tf_ms.get(tf, 3_600_000)
                expected    = max(1, (end_ms - start_ms) // interval_ms)
                cur.execute(
                    "SELECT COUNT(*) AS cnt, MIN(ts) AS first_ts, MAX(ts) AS last_ts"
                    " FROM backtest_candles WHERE symbol=%s AND tf=%s AND ts>=%s AND ts<=%s",
                    (sym.upper(), tf, start_ms, end_ms),
                )
                row     = cur.fetchone()
                cnt     = row["cnt"]     if row else 0
                first_v = row["first_ts"] if row else None
                last_v  = row["last_ts"]  if row else None
                pct     = cnt / expected
                ok      = pct >= threshold
                if not ok:
                    all_ok = False
                rows.append({
                    "symbol":   sym.upper(),
                    "tf":       tf,
                    "count":    cnt,
                    "expected": expected,
                    "pct":      round(pct * 100, 1),
                    "first":    datetime.utcfromtimestamp(first_v / 1000).strftime("%Y-%m-%d") if first_v else "—",
                    "last":     datetime.utcfromtimestamp(last_v  / 1000).strftime("%Y-%m-%d") if last_v  else "—",
                    "ok":       ok,
                })
    return all_ok, rows


def prefetch_all_candles(
    symbols: List[str],
    date_from: str,
    date_to: str,
    style: str = "DAY_TRADE",
    force_refetch: bool = False,
) -> None:
    """
    Pre-fetch and cache candles for all symbols before running simulations.
    Runs sequentially to avoid OKX rate limits. Checks 97% threshold per coin.
    Call this once at the start of a baseline run.
    """
    from datetime import datetime as _dt
    tf       = TF_FROM_STYLE.get(style, "1h")
    higher_tf = HIGHER_TF_MAP.get(tf, tf)
    start_ms = int(_dt.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(_dt.strptime(date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    daily_start_ms    = start_ms - 30 * 86_400_000
    weekly_start_ms   = start_ms - 210 * 7 * 24 * 3600 * 1000

    for sym in symbols:
        print(f"  prefetch {sym} … ", end="", flush=True)
        _ensure_candles(sym, tf,        start_ms,        end_ms,   force_refetch=force_refetch)
        _ensure_candles(sym, higher_tf, start_ms,        end_ms,   force_refetch=force_refetch)
        _ensure_candles(sym, "1d",      daily_start_ms,  end_ms,   force_refetch=force_refetch)
        _ensure_candles("BTCUSDT", "1W", weekly_start_ms, end_ms,  force_refetch=force_refetch)
        print("done")


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


def _sim_check_exit(
    trade: Dict, candle: Dict, exchange: str, regime: str = "TRENDING",
    force_grade_b: bool = False,
) -> Optional[Dict]:
    """
    Check whether the open trade exits on this candle.
    Mutates trade dict for Grade B T1/T2 state and trail ratchet state.
    Returns dict with {closed, outcome, pnl} or None if trade stays open.

    regime        — current market regime.
    force_grade_b — Option 3: use Grade B T1/T2 exit for all trades regardless
                    of trade['grade']. Grade label in trade record is preserved.
    """
    side     = trade["side"]
    entry    = trade["entry"]
    sl_pct   = trade["sl_pct"]
    tp_pct   = trade["tp_pct"]
    grade    = "B" if force_grade_b else trade["grade"]
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
        t1_tp_pct   = tp_pct * trade.get("t1_tp_mult", 1.0)
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

        # ── T2 phase — only evaluated from the candle AFTER T1 closed ──────
        if trade.get("t1_done") and not t1_just_closed:
            t2_dollar = equity * trade["t2_size"]
            _be = trade.get("t2_sl_is_breakeven", False)
            _atr_pct = trade.get("atr_pct", ATR_BASELINE.get("1h", 0.009))

            # Base SL — updated below if trail is active
            t2_sl_price = entry if _be else sl_price

            if _be:
                # ── Trail ratchet on T2 (mirrors live _close_one_trade
                #    after breakeven is set) ────────────────────────────
                candle_best = high if side == "LONG" else low
                _t2_bm = max(
                    float(trade.get("t2_trail_best_move", 0.0)),
                    (candle_best - entry) / entry if side == "LONG"
                    else (entry - candle_best) / entry,
                )
                trade["t2_trail_best_move"] = _t2_bm

                _t2_locked = 0.0
                for _tf, _lf, _ in reversed(TRAIL_LEVELS):
                    if _t2_bm >= tp_pct * _tf:
                        if _lf == 0.0 and _t2_bm < _atr_pct * 0.50:
                            continue
                        _t2_locked = _t2_bm * _lf
                        break

                _prev_t2 = float(trade.get("t2_trail_locked_pct", 0.0))
                _t2_locked = max(_t2_locked, _prev_t2)
                trade["t2_trail_locked_pct"] = _t2_locked

                if _t2_locked > 0.0:
                    t2_sl_price = (entry * (1.0 + _t2_locked) if side == "LONG"
                                   else entry * (1.0 - _t2_locked))

                # ── Beyond-TP activation and trailing for T2 ─────────────
                _choppy = regime in ("RANGING", "CHOPPY")
                _t2_act_pct   = tp_pct * 1.5 if _choppy else tp_pct
                _t2_act_price = (entry * (1.0 + _t2_act_pct) if side == "LONG"
                                 else entry * (1.0 - _t2_act_pct))
                _t2_act_hit   = (high >= _t2_act_price if side == "LONG"
                                 else low <= _t2_act_price)

                _t2_tp_hit = tp_hit(tp_price)

                if _t2_tp_hit and not trade.get("tp_trail_active_t2"):
                    if _t2_act_hit:
                        trade["tp_trail_active_t2"] = True
                        trade["tp_trail_floor_t2"]  = _t2_act_price
                        trade["tp_trail_peak_t2"]   = high if side == "LONG" else low
                        _t2_tp_hit = False  # trail takes over; suppress normal close
                elif _t2_tp_hit and trade.get("tp_trail_active_t2"):
                    _t2_tp_hit = False

                if trade.get("tp_trail_active_t2"):
                    _adx_open   = float(trade.get("adx_at_open", 25.0))
                    _trail_mult = 1.5 if _adx_open >= 35.0 else 1.0
                    if side == "LONG":
                        trade["tp_trail_peak_t2"] = max(
                            float(trade.get("tp_trail_peak_t2", tp_price)), high)
                    else:
                        trade["tp_trail_peak_t2"] = min(
                            float(trade.get("tp_trail_peak_t2", tp_price)), low)
                    _pk = float(trade["tp_trail_peak_t2"])
                    _fl = float(trade.get("tp_trail_floor_t2",
                        entry * (1.0 + tp_pct) if side == "LONG"
                        else entry * (1.0 - tp_pct)))
                    if side == "LONG":
                        _bsl = max(_pk * (1.0 - _trail_mult * _atr_pct), _fl)
                        _bhit = low <= _bsl
                        final_move = max((_bsl - entry) / entry, tp_pct) if _bhit else 0.0
                    else:
                        _bsl = min(_pk * (1.0 + _trail_mult * _atr_pct), _fl)
                        _bhit = high >= _bsl
                        final_move = max((entry - _bsl) / entry, tp_pct) if _bhit else 0.0
                    if _bhit:
                        t2_pnl = t2_dollar * (final_move * leverage) - _fee_cost(t2_dollar, "TP_HIT", exchange)
                        return {"closed": True, "outcome": "T1_TP_T2_TRAIL",
                                "pnl": trade["accumulated_pnl"] + t2_pnl, "exit_ts": candle["t"]}

                # ── Normal T2 TP / SL ─────────────────────────────────────
                _t2_sl_hit = sl_hit(t2_sl_price)

                if _t2_sl_hit and _t2_tp_hit:
                    _t2_tp_hit = _t2_locked > 0.0
                    _t2_sl_hit = not _t2_tp_hit

                if _t2_tp_hit:
                    t2_pnl = t2_dollar * (tp_pct * leverage) - _fee_cost(t2_dollar, "TP_HIT", exchange)
                    return {"closed": True, "outcome": "T1_TP_T2_TP",
                            "pnl": trade["accumulated_pnl"] + t2_pnl, "exit_ts": candle["t"]}

                if _t2_sl_hit:
                    if _t2_locked > 0.0:
                        t2_pnl = (t2_dollar * (_t2_locked * leverage)
                                  - _fee_cost(t2_dollar, "TP_HIT", exchange))
                        return {"closed": True, "outcome": "T1_TP_T2_TRAIL_SL",
                                "pnl": trade["accumulated_pnl"] + t2_pnl, "exit_ts": candle["t"]}
                    else:
                        # Breakeven hit — exit T2 at 0 P&L (only fees)
                        t2_pnl = -_fee_cost(t2_dollar, "SL_HIT", exchange)
                        return {"closed": True, "outcome": "T1_TP_T2_BE",
                                "pnl": trade["accumulated_pnl"] + t2_pnl, "exit_ts": candle["t"]}

            else:
                # T2 still on original SL (pre-breakeven) — no trail yet
                if tp_hit(tp_price):
                    t2_pnl = t2_dollar * (tp_pct * leverage) - _fee_cost(t2_dollar, "TP_HIT", exchange)
                    return {"closed": True, "outcome": "T1_TP_T2_TP",
                            "pnl": trade["accumulated_pnl"] + t2_pnl, "exit_ts": candle["t"]}
                if sl_hit(t2_sl_price):
                    t2_pnl = t2_dollar * (-sl_pct * leverage) - _fee_cost(t2_dollar, "SL_HIT", exchange)
                    return {"closed": True, "outcome": "T1_TP_T2_SL",
                            "pnl": trade["accumulated_pnl"] + t2_pnl, "exit_ts": candle["t"]}

        return None  # trade still open

    else:
        # Grade A — single exit with trail ratchet + beyond-TP trailing
        # (mirrors _close_one_trade in main.py exactly)
        size_dollar = equity * trade["effective_size"]

        # T18 scale-in exits (only active when enable_t18=True in _run_worker)
        profit_lock = trade.get("profit_lock", 0.0)
        if profit_lock > 0:
            lock_price = (entry * (1 + profit_lock) if side == "LONG"
                          else entry * (1 - profit_lock))
            if (side == "LONG" and low <= lock_price) or (side == "SHORT" and high >= lock_price):
                pnl = size_dollar * (profit_lock * leverage) - _fee_cost(size_dollar, "TP_HIT", exchange)
                return {"closed": True, "outcome": "SCALE_TRAIL", "pnl": pnl, "exit_ts": candle["t"]}

        if trade.get("be_active"):
            if (side == "LONG" and low <= entry) or (side == "SHORT" and high >= entry):
                pnl = -_fee_cost(size_dollar, "SL_HIT", exchange)
                return {"closed": True, "outcome": "SCALE_BE", "pnl": pnl, "exit_ts": candle["t"]}
            if tp_hit(tp_price):
                pnl = size_dollar * (tp_pct * leverage) - _fee_cost(size_dollar, "TP_HIT", exchange)
                return {"closed": True, "outcome": "TP_HIT", "pnl": pnl, "exit_ts": candle["t"]}
            return None

        # ── 7-level trail ratchet ─────────────────────────────────────────
        # Runs every candle while trade is open (mirrors TRAIL_LEVELS logic).
        # Only active when beyond-TP trail is not yet triggered.
        if not trade.get("tp_trail_active"):
            candle_best = high if side == "LONG" else low
            candle_best_move = ((candle_best - entry) / entry if side == "LONG"
                                else (entry - candle_best) / entry)
            lifetime_best_move = max(float(trade.get("trail_best_move", 0.0)), candle_best_move)
            trade["trail_best_move"] = lifetime_best_move

            trail_locked_pct = 0.0
            _atr_pct = trade.get("atr_pct", ATR_BASELINE.get("1h", 0.009))
            for _tl_tp_frac, _tl_lock_frac, _ in reversed(TRAIL_LEVELS):
                if lifetime_best_move >= tp_pct * _tl_tp_frac:
                    # Breakeven level: require ≥0.5× ATR move to avoid micro-noise trigger
                    if _tl_lock_frac == 0.0 and lifetime_best_move < _atr_pct * 0.50:
                        continue
                    trail_locked_pct = lifetime_best_move * _tl_lock_frac
                    break

            # Ratchet: locked profit can only increase
            prev_locked = float(trade.get("trail_locked_pct", 0.0))
            trail_locked_pct = max(trail_locked_pct, prev_locked)
            trade["trail_locked_pct"] = trail_locked_pct

            if trail_locked_pct > 0.0:
                if side == "LONG":
                    sl_price = entry * (1.0 + trail_locked_pct)
                else:
                    sl_price = entry * (1.0 - trail_locked_pct)
                sl_pct = trail_locked_pct

        # ── Beyond-TP activation (regime-aware) ──────────────────────────
        # RANGING/CHOPPY: trail fires at 1.5×TP. All others: 1.0×TP.
        _choppy = regime in ("RANGING", "CHOPPY")
        _trail_act_pct   = tp_pct * 1.5 if _choppy else tp_pct
        _trail_act_price = (entry * (1.0 + _trail_act_pct) if side == "LONG"
                            else entry * (1.0 - _trail_act_pct))
        _trail_act_hit   = (high >= _trail_act_price if side == "LONG"
                            else low <= _trail_act_price)

        intrabar_tp = tp_hit(tp_price)

        if intrabar_tp and not trade.get("tp_trail_active"):
            if _trail_act_hit:
                trade["tp_trail_active"] = True
                trade["tp_trail_floor"]  = _trail_act_price
                trade["tp_trail_peak"]   = high if side == "LONG" else low
                intrabar_tp = False  # suppress normal TP close; trail takes over
            # else: choppy + below 1.5×TP → normal TP close at 1×TP
        elif intrabar_tp and trade.get("tp_trail_active"):
            intrabar_tp = False  # already in trail mode

        # ── Beyond-TP trail stop ──────────────────────────────────────────
        _beyond_hit = False
        if trade.get("tp_trail_active"):
            _atr_val  = trade.get("atr_pct", ATR_BASELINE.get("1h", 0.009))
            _adx_open = float(trade.get("adx_at_open", 25.0))
            _trail_mult = 1.5 if _adx_open >= 35.0 else 1.0
            # Ratchet peak
            if side == "LONG":
                trade["tp_trail_peak"] = max(float(trade.get("tp_trail_peak", tp_price)), high)
            else:
                trade["tp_trail_peak"] = min(float(trade.get("tp_trail_peak", tp_price)), low)
            _tp_peak  = float(trade["tp_trail_peak"])
            _tp_floor = float(trade.get("tp_trail_floor",
                entry * (1.0 + tp_pct) if side == "LONG" else entry * (1.0 - tp_pct)))
            if side == "LONG":
                _beyond_sl = max(_tp_peak * (1.0 - _trail_mult * _atr_val), _tp_floor)
                _beyond_hit = low <= _beyond_sl
                if _beyond_hit:
                    final_move = max((_beyond_sl - entry) / entry, tp_pct)
            else:
                _beyond_sl = min(_tp_peak * (1.0 + _trail_mult * _atr_val), _tp_floor)
                _beyond_hit = high >= _beyond_sl
                if _beyond_hit:
                    final_move = max((entry - _beyond_sl) / entry, tp_pct)
            if _beyond_hit:
                pnl = size_dollar * (final_move * leverage) - _fee_cost(size_dollar, "TP_HIT", exchange)
                return {"closed": True, "outcome": "TRAIL_STOP", "pnl": pnl, "exit_ts": candle["t"]}

        # ── Standard SL/TP ───────────────────────────────────────────────
        intrabar_sl = sl_hit(sl_price)

        if intrabar_sl and intrabar_tp:
            # Both hit: TP wins if trail is active (trade moved to profit first)
            intrabar_tp = trade.get("trail_locked_pct", 0.0) > 0
            intrabar_sl = not intrabar_tp

        if intrabar_tp:
            pnl = size_dollar * (tp_pct * leverage) - _fee_cost(size_dollar, "TP_HIT", exchange)
            return {"closed": True, "outcome": "TP_HIT", "pnl": pnl, "exit_ts": candle["t"]}

        if intrabar_sl:
            exit_pct = sl_pct  # may be trail-adjusted (locked profit)
            pnl = size_dollar * (-exit_pct * leverage) - _fee_cost(size_dollar, "SL_HIT", exchange)
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

    # T16 reversal sub-stats
    rev_trades = [t for t in trades if t.get("signal") == "REVERSAL"]
    rev_wins   = [t for t in rev_trades if t["pnl"] > 0]
    rev_win_rate = round(len(rev_wins) / len(rev_trades) * 100, 1) if rev_trades else 0.0

    # T18 scaling sub-stats (scale-in events tracked per trade)
    scaled_trades = [t for t in trades if t.get("scaled")]
    scaled_wins   = [t for t in scaled_trades if t["pnl"] > 0]
    avg_scaled_win  = round(statistics.mean(t["pnl"] / start_equity * 100 for t in scaled_wins), 2)  if scaled_wins  else 0.0
    avg_normal_win  = round(statistics.mean(t["pnl"] / start_equity * 100 for t in wins if not t.get("scaled")), 2) if [w for w in wins if not w.get("scaled")] else 0.0

    # Outcome breakdown — counts per exit type to diagnose loss sources
    outcome_counts: Dict[str, int] = {}
    outcome_pnl:    Dict[str, float] = {}
    for t in trades:
        o = t.get("outcome", "unknown")
        outcome_counts[o] = outcome_counts.get(o, 0) + 1
        outcome_pnl[o]    = round(outcome_pnl.get(o, 0.0) + t["pnl"], 4)

    # BE-specific: trades that reached breakeven then closed at ~0
    be_outcomes = {"SCALE_BE", "T1_TP_T2_BE", "T1_TP_T2_TRAIL_SL"}
    be_count    = sum(outcome_counts.get(o, 0) for o in be_outcomes)
    sl_count    = outcome_counts.get("SL_HIT", 0)
    tp_count    = sum(outcome_counts.get(o, 0) for o in ("TP_HIT", "T1_TP_T2_TP", "TRAIL_STOP"))
    # Win rate excluding BE closures (shows "true" directional win rate)
    non_be_total  = total - be_count
    non_be_wins   = [t for t in wins if t.get("outcome") not in be_outcomes]
    win_rate_excl_be = round(len(non_be_wins) / non_be_total * 100, 1) if non_be_total > 0 else 0.0

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
        # T16 reversal stats
        "reversal_trades":   len(rev_trades),
        "reversal_win_rate": rev_win_rate,
        # T18 scaling stats
        "scaled_trades":    len(scaled_trades),
        "avg_scaled_win_pct": avg_scaled_win,
        "avg_normal_win_pct": avg_normal_win,
        # Outcome breakdown — key diagnostic for loss source
        "outcome_counts":     outcome_counts,
        "outcome_pnl":        outcome_pnl,
        "sl_hit_count":       sl_count,
        "tp_hit_count":       tp_count,
        "be_close_count":     be_count,
        "win_rate_excl_be":   win_rate_excl_be,
        # Raw trade log — used by run_parity_test.py for trade-by-trade comparison
        "trade_log": [
            {
                "open_ts":  t["open_ts"],
                "exit_ts":  t["exit_ts"],
                "side":     t["side"],
                "grade":    t["grade"],
                "entry":    t["entry"],
                "pnl":      round(t["pnl"], 4),
                "outcome":  t["outcome"],
                "sl_pct":   round(t.get("sl_pct", 0.0), 6),
                "tp_pct":   round(t.get("tp_pct", 0.0), 6),
                "regime":   t.get("regime", ""),
                "signal":   t.get("signal", ""),
                "score":    t.get("score", 0.0),
            }
            for t in trades
        ],
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
    enable_t16: bool = True,
    enable_t18: bool = True,
    enable_s4: bool = False,
    enable_mr_4h_gate: bool = False,
    enable_mr_signal: bool = True,
    enable_mr_4h_signal: bool = False,
    enable_regime_gate: bool = False,
    enable_e2: bool = False,
    enable_step2_score: bool = False,
    enable_grade_b_all: bool = False,
    enable_score_cap: bool = False,
    score_cap: float = 0.78,
    t1_tp_mult: float = 1.0,
    enable_floor: bool = True,
    random_seed: Optional[int] = None,
    score_floor: float = 0.0,
    e2_hold_mult: float = 1.0,
    enable_grid: bool = False,
    enable_dc_core: bool = False,
    dc_long_cut: float = 0.35,
    dc_short_cut: float = 0.65,
    enable_grid_v2: bool = False,
    gv2_adx_thresh: float = 15.0,
    gv2_n_candles:  int   = 4,
    gv2_max_levels: int   = 3,
    gv2_step_mult:  float = 0.50,
    gv2_buffer_atr: float = 0.75,
    gv2_cooldown:   int   = 4,
    gv2_equity_pct: float = 0.25,
    enable_sentiment_filter: bool = False,
) -> None:
    _last_db_write = [0.0]  # throttle DB progress writes

    def _set(status: str, progress: int = 0, error: str = None):
        with _active_lock:
            _active[run_id] = {"status": status, "progress": progress, "error": error}
        # Write to DB every 5s so multi-worker setups see live progress
        now = time.time()
        if now - _last_db_write[0] >= 5:
            _last_db_write[0] = now
            try:
                with db_conn() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE backtest_runs SET status=%s, progress=%s WHERE run_id=%s",
                        (status, progress, run_id),
                    )
                    conn.commit()
            except Exception:
                pass

    # Build a fetch-progress callback for _ensure_candles
    _tf_ms = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    def _fetch_cb(start_pct: int, end_pct: int, tf_name: str):
        interval_ms   = _tf_ms.get(tf_name, 3_600_000)
        expected_pages = max(1, (end_ms - start_ms) // interval_ms // 100 + 1)
        def cb(pages_done: int):
            frac = min(1.0, pages_done / expected_pages)
            _set("fetching", int(start_pct + frac * (end_pct - start_pct)))
        return cb

    _set("fetching", 0)

    try:
        higher_tf = HIGHER_TF_MAP.get(tf, tf)
        # MTF (15m) skipped in backtesting: 2yr of 15m = 700+ pages (3+ min fetch).
        # MTF only adjusts size ±15% and never blocks — omitting it is acceptable.

        # ── 1. Fetch candles (cached in DB after first fetch) ──────────────
        _set("fetching", 5)
        main_candles = _ensure_candles(symbol, tf, start_ms, end_ms,
                                       on_progress=_fetch_cb(5, 60, tf))
        if len(main_candles) < WARMUP_CANDLES + 10:
            raise ValueError(
                f"Not enough {tf} candles for {symbol} ({len(main_candles)} fetched, "
                f"need at least {WARMUP_CANDLES + 10}). "
                f"Try a wider date range or check the symbol name."
            )

        _set("fetching", 60)
        higher_candles = _ensure_candles(symbol, higher_tf, start_ms, end_ms,
                                         on_progress=_fetch_cb(60, 73, higher_tf))
        mtf_candles: List[Dict] = []

        # ── Fetch BTC weekly candles for regime + weekly trend detection ──
        # Extend range back by 210 weeks so the warmup period has enough history.
        btc_weekly_start_ms = start_ms - 210 * 7 * 24 * 3600 * 1000
        btc_weekly_candles: List[Dict] = []
        try:
            btc_weekly_candles = _ensure_candles(
                "BTCUSDT", "1W", btc_weekly_start_ms, end_ms,
                on_progress=_fetch_cb(73, 75, "1W"),
            )
        except Exception:
            btc_weekly_candles = []  # degrade to TRENDING if fetch fails

        # ── Fetch daily candles for daily bias (live engine uses 30-day window) ──
        # Extend range back 30 days so the warmup period has daily history.
        daily_start_ms = start_ms - 30 * 86_400_000
        daily_candles: List[Dict] = []
        try:
            daily_candles = _ensure_candles(
                symbol, "1d", daily_start_ms, end_ms,
                on_progress=_fetch_cb(75, 77, "1d"),
            )
        except Exception:
            daily_candles = []  # degrade to NEUTRAL daily bias if fetch fails

        _set("running", 77)

        _mode_cfg  = get_mode_config(mode)
        _style_cfg = get_style_config(style)

        # ── 2. Simulation loop ─────────────────────────────────────────────
        st = TRADE_STYLE_PARAMS[style]
        c  = MODE_PRESETS[mode]

        orig_start_equity = start_equity
        equity      = start_equity
        peak_equity = start_equity
        _floor_resets: List[Dict] = []
        _floor_cooldown_until: int = 0
        trades:       List[Dict] = []
        equity_curve: List[Dict] = [{"ts": main_candles[WARMUP_CANDLES]["t"], "equity": equity}]

        open_trade: Optional[Dict] = None
        total_candles = len(main_candles) - WARMUP_CANDLES

        # Regime + pullback state (mirrors AutoRunner instance vars)
        _is_parabolic:       bool = False
        _signal_pending:     bool = False
        _pending_side:       str  = ""
        _pending_score:      float = 0.0
        _candles_since_sig:  int  = 0

        # Post-reset size cap: 50% until +10% recovery; 2+ resets in 90d → cap stays
        _post_reset_active: bool  = False
        _post_reset_base:   float = 0.0
        _post_reset_cap:    float = 1.0

        # Grid v2 state machine
        _gv2_active:        bool       = False
        _gv2_band_high:     float      = 0.0
        _gv2_band_low:      float      = 0.0
        _gv2_midpoint:      float      = 0.0
        _gv2_step_pct:      float      = 0.0
        _gv2_levels:        List[Dict] = []   # [{side, entry_price, tp_price, filled, fill_idx, size_pct, equity_at_open}]
        _gv2_equity_locked: float      = 0.0
        _gv2_cooldown_until: int       = 0

        # Precompute indicator series — eliminates ~9× O(n) computation per candle
        _all_closes = [c["close"] for c in main_candles]
        _all_highs  = [c["high"]  for c in main_candles]
        _all_lows   = [c["low"]   for c in main_candles]
        _adx_vals   = _adx_series(_all_highs, _all_lows, _all_closes, 14)
        _rsi_raw    = _rsi_series(_all_closes, 14)
        _rsi_full   = [None] * (len(_all_closes) - len(_rsi_raw)) + _rsi_raw
        _precomp = {
            "ema9":  _ema(_all_closes, 9),
            "ema21": _ema(_all_closes, 21),
            "ema50": _ema(_all_closes, 50),
            "rsi":   _rsi_full,
            "atr14": _atr_series(_all_highs, _all_lows, _all_closes, 14),
            "adx":   _adx_vals,
            "atr50": _atr_series(_all_highs, _all_lows, _all_closes, 50),
        }

        # Load F&G history once per run (tiny — ~2000 rows, negligible memory)
        _fng_history: dict = {}
        if enable_sentiment_filter:
            try:
                from sentiment import load_fng_history, fng_at_ts
                _fng_history = load_fng_history()
            except Exception:
                pass

        # Random-entry benchmark: deterministic RNG, bypasses signal computation
        _rng = _random_module.Random(random_seed) if random_seed is not None else None

        for i in range(WARMUP_CANDLES, len(main_candles)):
            candle = main_candles[i]

            # Progress reporting every 50 candles (50-99%)
            if i % 50 == 0:
                done_pct = (i - WARMUP_CANDLES) / max(1, total_candles)
                _set("running", 75 + int(done_pct * 24))

            # ── Check exit for open trade ──────────────────────────────
            if open_trade:
                # T18: track best unrealized move before exit check, scale in if thresholds met
                if enable_t18 and open_trade.get("grade") == "A":
                    _entry = open_trade["entry"]
                    _side  = open_trade["side"]
                    _atr   = open_trade.get("atr_pct", ATR_BASELINE.get(tf, 0.009))
                    _bm_now = (candle["high"] - _entry) / _entry if _side == "LONG" else (_entry - candle["low"]) / _entry
                    open_trade["best_move"] = max(open_trade.get("best_move", 0.0), _bm_now)
                    _bm = open_trade["best_move"]
                    # Scale-in 1: +1.5× ATR → add 20%, SL → breakeven
                    if not open_trade.get("scale1_done") and _bm >= 1.5 * _atr and _atr > 0:
                        open_trade["effective_size"] *= 1.20
                        open_trade["scale1_done"]  = True
                        open_trade["be_active"]    = True   # SL → entry (breakeven)
                        open_trade["scaled"]       = True
                    # Scale-in 2: +3× ATR → add 10%, lock SL at +1.5× ATR profit
                    elif open_trade.get("scale1_done") and not open_trade.get("scale2_done") and _bm >= 3.0 * _atr and _atr > 0:
                        open_trade["effective_size"] *= (1.10 / 1.20)  # net relative to current size
                        open_trade["scale2_done"]   = True
                        open_trade["profit_lock"]   = 1.5 * _atr   # SL at +1.5× ATR from entry

                # ── E2: Score-scaled time exit ────────────────────────────────
                if enable_e2 and open_trade:
                    _e2_tf_ms  = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}.get(tf, 3_600_000)
                    _e2_held   = (candle["t"] - open_trade["open_ts"]) // _e2_tf_ms
                    _e2_score  = float(open_trade.get("score", 0.65))
                    _e2_max    = round((30 if _e2_score >= 0.80 else (22 if _e2_score >= 0.70 else 15)) * e2_hold_mult)
                    if _e2_held >= _e2_max:
                        _e2_entry = open_trade["entry"]
                        _e2_side  = open_trade["side"]
                        _e2_close = candle["close"]
                        _e2_lev   = open_trade["leverage"]
                        _e2_raw   = ((_e2_close - _e2_entry) / _e2_entry if _e2_side == "LONG"
                                     else (_e2_entry - _e2_close) / _e2_entry)
                        if open_trade["grade"] == "B" and open_trade.get("t1_done"):
                            _e2_size = open_trade["equity_at_open"] * open_trade["t2_size"]
                        else:
                            _e2_size = open_trade["equity_at_open"] * open_trade["effective_size"]
                        _e2_pnl = (open_trade.get("accumulated_pnl", 0.0)
                                   + _e2_size * _e2_lev * _e2_raw
                                   - _fee_cost(_e2_size, "SL_HIT", exchange))
                        equity += _e2_pnl
                        peak_equity = max(peak_equity, equity)
                        trades.append({
                            "pnl":     _e2_pnl,
                            "outcome": "E2_TIME",
                            "side":    open_trade["side"],
                            "grade":   open_trade["grade"],
                            "signal":  open_trade["signal"],
                            "score":   open_trade["score"],
                            "regime":  open_trade["regime"],
                            "open_ts": open_trade["open_ts"],
                            "exit_ts": candle["t"],
                            "entry":   open_trade["entry"],
                            "sl_pct":  open_trade["sl_pct"],
                            "tp_pct":  open_trade["tp_pct"],
                            "scaled":  open_trade.get("scaled", False),
                        })
                        equity_curve.append({"ts": candle["t"], "equity": round(equity, 4)})
                        open_trade = None

                if open_trade:
                    result = _sim_check_exit(open_trade, candle, exchange,
                                            regime=open_trade.get("regime", "TRENDING"),
                                            force_grade_b=enable_grade_b_all)
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
                            "sl_pct":   open_trade["sl_pct"],
                            "tp_pct":   open_trade["tp_pct"],
                            "scaled":   open_trade.get("scaled", False),
                        })
                        equity_curve.append({"ts": result["exit_ts"], "equity": round(equity, 4)})
                        open_trade = None

            # ── Grid v2 state machine ──────────────────────────────────
            if enable_grid_v2 and _gv2_active:
                _gv2_tf_ms  = _tf_ms.get(tf, 3_600_000)
                _gv2_fees   = EXCHANGE_FEES[exchange]["taker"] * 2 + EXCHANGE_FEES[exchange]["slippage"]
                _gv2_buffer = (_precomp["atr14"][i] or 0.0) * gv2_buffer_atr

                # Hard exit: candle CLOSE outside band + buffer
                if (candle["close"] < _gv2_band_low  - _gv2_buffer or
                        candle["close"] > _gv2_band_high + _gv2_buffer):
                    for _lv in _gv2_levels:
                        if not _lv["filled"]:
                            continue
                        _lv_move = (
                            (candle["close"] - _lv["entry_price"]) / _lv["entry_price"]
                            if _lv["side"] == "LONG"
                            else (_lv["entry_price"] - candle["close"]) / _lv["entry_price"]
                        )
                        _lv_pnl = _lv["equity_at_open"] * _lv["size_pct"] * (_lv_move - _gv2_fees)
                        equity += _lv_pnl
                        trades.append({
                            "pnl":     _lv_pnl,
                            "outcome": "gv2_hard_exit",
                            "side":    _lv["side"],
                            "grade":   "B",
                            "signal":  "GRID_V2",
                            "score":   0.60,
                            "regime":  "RANGING",
                            "open_ts": _lv["fill_ts"],
                            "exit_ts": candle["t"],
                            "entry":   _lv["entry_price"],
                            "sl_pct":  _gv2_buffer / _lv["entry_price"],
                            "tp_pct":  abs(_gv2_midpoint - _lv["entry_price"]) / _lv["entry_price"],
                            "scaled":  False,
                        })
                    equity_curve.append({"ts": candle["t"], "equity": round(equity, 4)})
                    _gv2_active        = False
                    _gv2_levels        = []
                    _gv2_cooldown_until = candle["t"] + _gv2_tf_ms * gv2_cooldown
                else:
                    # Fill unfilled levels (pessimistic: no exit on same candle as fill)
                    for _lv in _gv2_levels:
                        if _lv["filled"]:
                            continue
                        _touched = (
                            candle["low"]  <= _lv["entry_price"] if _lv["side"] == "LONG"
                            else candle["high"] >= _lv["entry_price"]
                        )
                        if _touched:
                            _lv["filled"]   = True
                            _lv["fill_ts"]  = candle["t"]
                            _lv["fill_idx"] = i

                    # Check TP for filled levels (only if not filled this same candle)
                    _any_open = False
                    for _lv in _gv2_levels:
                        if not _lv["filled"]:
                            _any_open = True
                            continue
                        if _lv.get("fill_idx") == i:
                            _any_open = True
                            continue  # pessimistic: can't exit same candle as fill
                        if _lv.get("tp_done"):
                            continue
                        _tp_hit = (
                            candle["high"] >= _lv["tp_price"] if _lv["side"] == "LONG"
                            else candle["low"] <= _lv["tp_price"]
                        )
                        if _tp_hit:
                            _lv_move = abs(_lv["tp_price"] - _lv["entry_price"]) / _lv["entry_price"]
                            _lv_pnl  = _lv["equity_at_open"] * _lv["size_pct"] * (_lv_move - _gv2_fees)
                            equity  += _lv_pnl
                            trades.append({
                                "pnl":     _lv_pnl,
                                "outcome": "gv2_tp",
                                "side":    _lv["side"],
                                "grade":   "B",
                                "signal":  "GRID_V2",
                                "score":   0.60,
                                "regime":  "RANGING",
                                "open_ts": _lv["fill_ts"],
                                "exit_ts": candle["t"],
                                "entry":   _lv["entry_price"],
                                "sl_pct":  _gv2_buffer / _lv["entry_price"],
                                "tp_pct":  _lv_move,
                                "scaled":  False,
                            })
                            equity_curve.append({"ts": candle["t"], "equity": round(equity, 4)})
                            _lv["tp_done"] = True
                        else:
                            _any_open = True

                    # All levels completed → deactivate
                    if not _any_open:
                        _gv2_active = False
                        _gv2_levels = []

            # ── Hard floor stop (R1 Hybrid — mirrors live _compute_floor) ─
            if candle["t"] < _floor_cooldown_until:
                continue  # inside cooldown window — skip candle

            if equity <= _sim_floor(peak_equity, start_equity):
                if not enable_floor:
                    pass  # floor OFF: ignore hit, keep trading
                else:
                    # Force-close any open trade at this candle's close price
                    if open_trade:
                        _close_price  = candle["close"]
                        _entry        = open_trade["entry"]
                        _side         = open_trade["side"]
                        _eff_size     = open_trade["effective_size"]
                        _lev          = open_trade["leverage"]
                        _raw_move     = (_close_price - _entry) / _entry if _side == "LONG" else (_entry - _close_price) / _entry
                        fees          = EXCHANGE_FEES[exchange]["taker"] * 2 + EXCHANGE_FEES[exchange]["slippage"]
                        _force_pnl    = equity * _eff_size * _lev * (_raw_move - fees)
                        equity       += _force_pnl
                        trades.append({
                            "pnl":     _force_pnl,
                            "outcome": "floor_close",
                            "side":    open_trade["side"],
                            "grade":   open_trade["grade"],
                            "signal":  open_trade["signal"],
                            "score":   open_trade["score"],
                            "regime":  open_trade["regime"],
                            "open_ts": open_trade["open_ts"],
                            "exit_ts": candle["t"],
                            "entry":   open_trade["entry"],
                            "sl_pct":  open_trade["sl_pct"],
                            "tp_pct":  open_trade["tp_pct"],
                            "scaled":  open_trade.get("scaled", False),
                        })
                        equity_curve.append({"ts": candle["t"], "equity": round(equity, 4)})
                        open_trade = None

                    # Determine cooldown: 30 days if second hit within 90 days
                    _last_hit_ts  = _floor_resets[-1]["ts"] if _floor_resets else 0
                    _days_since   = (candle["t"] - _last_hit_ts) / 86_400_000
                    _double_hit   = bool(_floor_resets) and _days_since <= 90
                    _cooldown_days = 30 if _double_hit else 7

                    _floor_resets.append({
                        "ts":           candle["t"],
                        "equity":       round(equity, 2),
                        "cooldown_days": _cooldown_days,
                        "double_hit":   _double_hit,
                    })
                    _floor_cooldown_until = candle["t"] + int(_cooldown_days * 86_400_000)
                    start_equity  = equity   # new floor anchors to current equity
                    peak_equity   = equity

                    # Clear any stale pending signal — it was based on pre-reset context
                    _signal_pending    = False
                    _pending_side      = ""
                    _pending_score     = 0.0
                    _candles_since_sig = 0

                    # Post-reset size cap: restart at 50%, full only after +10% recovery
                    _post_reset_active = True
                    _post_reset_base   = equity
                    _post_reset_cap    = 0.5
                    continue

            # ── Try to open new trade ─────────────────────────────────
            if open_trade is None:
                klines = main_candles[max(0, i - 299) : i + 1]  # last 300, chron. order

                higher_slice = _get_aligned_slice(higher_candles, candle["t"], 100)
                mtf_slice    = _get_aligned_slice(mtf_candles, candle["t"], 50) if mtf_candles else None

                # Session quality for this historical candle's timestamp
                hist_sess = _session_quality_for_ts(style, candle["t"])
                if hist_sess == 0.0:
                    continue  # SCALP dead-zone block

                # ── Change 4: Regime detection ────────────────────────
                btc_weekly_slice = _get_aligned_slice(btc_weekly_candles, candle["t"], 210) if btc_weekly_candles else []
                cur_adx = _adx_vals[i] or 0.0
                _ranging_thresh = _style_cfg.get("ranging_adx_threshold", 15)
                regime, _is_parabolic = get_market_regime(btc_weekly_slice, cur_adx, _is_parabolic, ranging_threshold=_ranging_thresh)

                # RANGING = no trades (unless grid module is on)
                if regime == "RANGING":
                    if enable_grid:
                        _slice_h = [k["high"]  for k in klines[-20:]]
                        _slice_l = [k["low"]   for k in klines[-20:]]
                        _slice_c = [k["close"] for k in klines[-20:]]
                        _cur_atr = (_precomp["atr14"][i] or 0.0)
                        _grid = compute_grid_signal(_slice_h, _slice_l, _slice_c, _cur_atr, cur_adx)
                        if not _grid["ok"]:
                            if _signal_pending:
                                _signal_pending = False
                                _pending_side   = ""
                                _candles_since_sig = 0
                            continue
                        # Grid entry: use grid tp/sl directly (override ATR-based values)
                        _g_side = _grid["side"]
                        _g_atr  = _cur_atr if _cur_atr > 0 else ATR_BASELINE[tf]
                        _g_sl_pct = abs(_grid["sl"] - candle["close"]) / candle["close"]
                        _g_tp_pct = abs(_grid["tp"] - candle["close"]) / candle["close"]
                        _g_sl_pct = max(_g_sl_pct, 0.001)
                        _g_tp_pct = max(_g_tp_pct, 0.001)
                        _esize = c["size"] * _mode_cfg.get("position_size_mult", 1.0) * _post_reset_cap
                        open_trade = {
                            "entry":          candle["close"],
                            "side":           _g_side,
                            "grade":          "B",
                            "sl_pct":         _g_sl_pct,
                            "tp_pct":         _g_tp_pct,
                            "t1_tp_mult":     t1_tp_mult,
                            "effective_size": _esize,
                            "leverage":       c["leverage"],
                            "equity_at_open": equity,
                            "atr_pct":        _g_atr / candle["close"],
                            "adx_at_open":    cur_adx,
                            "t1_size":        _esize * 0.60,
                            "t2_size":        _esize * 0.40,
                            "t1_done":        False,
                            "t2_sl_is_breakeven": False,
                            "accumulated_pnl": 0.0,
                            "open_ts":        candle["t"],
                            "signal":         "GRID",
                            "score":          0.65,
                            "regime":         regime,
                        }
                    elif enable_grid_v2 and not _gv2_active and candle["t"] >= _gv2_cooldown_until:
                        # Grid v2 activation: build multi-level mean-reversion grid
                        _gv2_win  = max(gv2_n_candles + 2, 20)
                        _gv2_h    = [k["high"]  for k in klines[-_gv2_win:]]
                        _gv2_l    = [k["low"]   for k in klines[-_gv2_win:]]
                        _gv2_c    = [k["close"] for k in klines[-_gv2_win:]]
                        _gv2_atr  = (_precomp["atr14"][i] or 0.0)
                        _gv2_band = compute_grid_v2_band(
                            _gv2_h, _gv2_l, _gv2_c, _gv2_atr, cur_adx,
                            n_candles=gv2_n_candles,
                            adx_threshold=gv2_adx_thresh,
                            step_atr_mult=gv2_step_mult,
                        )
                        if _gv2_band["ok"]:
                            _gv2_band_high  = _gv2_band["band_high"]
                            _gv2_band_low   = _gv2_band["band_low"]
                            _gv2_midpoint   = _gv2_band["midpoint"]
                            _gv2_step_pct   = _gv2_band["step_pct"]
                            _gv2_equity_locked = equity * gv2_equity_pct
                            _n_long  = gv2_max_levels // 2
                            _n_short = gv2_max_levels - _n_long
                            _n_total = _n_long + _n_short
                            _sz_each = gv2_equity_pct / _n_total if _n_total > 0 else gv2_equity_pct
                            _gv2_levels = []
                            for _k in range(_n_long):
                                _ep = _gv2_band_low * (1.0 + _k * _gv2_step_pct)
                                _gv2_levels.append({
                                    "side":          "LONG",
                                    "entry_price":   _ep,
                                    "tp_price":      _gv2_midpoint,
                                    "filled":        False,
                                    "fill_ts":       0,
                                    "fill_idx":      -1,
                                    "tp_done":       False,
                                    "size_pct":      _sz_each,
                                    "equity_at_open": equity,
                                })
                            for _k in range(_n_short):
                                _ep = _gv2_band_high * (1.0 - _k * _gv2_step_pct)
                                _gv2_levels.append({
                                    "side":          "SHORT",
                                    "entry_price":   _ep,
                                    "tp_price":      _gv2_midpoint,
                                    "filled":        False,
                                    "fill_ts":       0,
                                    "fill_idx":      -1,
                                    "tp_done":       False,
                                    "size_pct":      _sz_each,
                                    "equity_at_open": equity,
                                })
                            _gv2_active = True
                    else:
                        if _signal_pending:
                            _signal_pending = False
                            _pending_side   = ""
                            _candles_since_sig = 0
                    continue

                # ── Change 5: Mode regime guard ───────────────────────
                if regime not in _mode_cfg["allowed_regimes"]:
                    if _signal_pending:
                        _signal_pending = False
                        _pending_side   = ""
                        _candles_since_sig = 0
                    continue

                # ── Change 5: SWING weekly trend filter ───────────────
                _allowed_sides: Optional[List[str]] = None
                if _style_cfg.get("swing_weekly_filter") and btc_weekly_slice:
                    weekly_trend = check_weekly_trend(btc_weekly_slice)
                    if weekly_trend == "TRANSITIONING":
                        if _signal_pending:
                            _signal_pending = False
                            _pending_side   = ""
                            _candles_since_sig = 0
                        continue
                    _allowed_sides = (
                        ["LONG"]  if weekly_trend == "BULLISH"
                        else ["SHORT"] if weekly_trend == "BEARISH"
                        else None
                    )

                # PARABOLIC overrides: loosen score, LONG only
                _param_overrides = {}
                if regime == "PARABOLIC":
                    _cp_par = get_coin_params(symbol)
                    _param_overrides = {"min_score": max(0.35, _cp_par["score_threshold"] - 0.05)}
                    _allowed_sides = ["LONG"]

                daily_slice = _get_aligned_slice(daily_candles, candle["t"], 30) if daily_candles else []

                # Random-entry benchmark: bypass signal/pullback, enter randomly
                if _rng is not None:
                    if _rng.random() < (1.0 / 80.0):
                        _r_atr   = ATR_BASELINE[tf]
                        _r_sl    = min(_r_atr * st["sl_atr"], st["sl_max"] / 100)
                        _r_tp    = min(_r_atr * st["tp_atr"], st["tp_max"] / 100)
                        _r_esize = c["size"] * _mode_cfg.get("position_size_mult", 1.0) * _post_reset_cap
                        open_trade = {
                            "entry": candle["close"], "side": _rng.choice(["LONG", "SHORT"]),
                            "grade": "B", "sl_pct": _r_sl, "tp_pct": _r_tp,
                            "t1_tp_mult": t1_tp_mult, "effective_size": _r_esize,
                            "leverage": c["leverage"], "equity_at_open": equity,
                            "atr_pct": _r_atr, "adx_at_open": 0.0,
                            "t1_size": _r_esize * 0.60, "t2_size": _r_esize * 0.40,
                            "t1_done": False, "t2_sl_is_breakeven": False,
                            "accumulated_pnl": 0.0,
                            "open_ts": candle["t"], "signal": "RANDOM", "score": 0.5,
                            "regime": "TRENDING",
                        }
                    continue  # skip full signal chain regardless

                _fng_score = None
                if enable_sentiment_filter and _fng_history:
                    try:
                        from sentiment import fng_at_ts as _fng_at_ts
                        _fng_score = _fng_at_ts(_fng_history, candle["t"])
                    except Exception:
                        pass

                sig = _compute_signal_layers(
                    klines, mode, 1.0, higher_slice, style, mtf_slice,
                    enable_t16=enable_t16, enable_s4=enable_s4, symbol=symbol,
                    param_overrides=_param_overrides if _param_overrides else None,
                    daily_klines=daily_slice or None,
                    candle_ts_ms=candle["t"],
                    enable_mr_4h_gate=enable_mr_4h_gate,
                    enable_mr_signal=enable_mr_signal,
                    enable_mr_4h_signal=enable_mr_4h_signal,
                    enable_regime_gate=enable_regime_gate,
                    enable_step2_score=enable_step2_score,
                    enable_sentiment_filter=enable_sentiment_filter,
                    fng_score=_fng_score,
                    score_floor=score_floor,
                    _precomp={**_precomp, "idx": i},
                )

                # Enforce side restriction
                if _allowed_sides and sig.get("ok") and sig.get("side") not in _allowed_sides:
                    sig["ok"] = False

                # ── Change 3: Volume hour confirmation ────────────────
                # Mirror main.py: score >= 0.68 bypasses volume check (high
                # conviction — 6-7 layers already confirmed, volume redundant).
                if sig.get("ok") and sig.get("score", 0.0) < 0.68:
                    _cp_vol = get_coin_params(symbol)
                    _vol_ok, _, _ = check_volume_hour_confirmed(klines, _cp_vol)
                    if not _vol_ok:
                        sig["ok"] = False

                # Score cap: pre-registered variant — block entries above cap
                if enable_score_cap and sig.get("ok") and sig.get("score", 0.0) > score_cap:
                    sig["ok"] = False

                # ── Cancel pending signal if conditions no longer met ──
                if not sig.get("ok"):
                    if _signal_pending:
                        _signal_pending    = False
                        _pending_side      = ""
                        _pending_score     = 0.0
                        _candles_since_sig = 0
                    continue

                # ── Change 2: Pullback entry gate ─────────────────────
                desired_side = sig["side"]
                if not _signal_pending:
                    _signal_pending    = True
                    _pending_side      = desired_side
                    _pending_score     = sig.get("score", 0.0)
                    _candles_since_sig = 0
                elif _pending_side != desired_side:
                    _pending_side      = desired_side
                    _pending_score     = sig.get("score", 0.0)
                    _candles_since_sig = 0

                _candles_since_sig += 1

                if _style_cfg.get("pullback_required", True):
                    _cp_pb   = get_coin_params(symbol)
                    _pb_res  = check_pullback_entry(
                        klines, _pending_side, _cp_pb, regime,
                        _candles_since_sig, trade_style=style,
                    )
                    if _pb_res == "TIMEOUT":
                        _signal_pending    = False
                        _pending_side      = ""
                        _pending_score     = 0.0
                        _candles_since_sig = 0
                        continue
                    if _pb_res == "WAIT":
                        continue
                    # ENTER or IMMEDIATE → proceed

                # Confirmed entry
                desired_side   = _pending_side
                _signal_pending    = False
                _pending_side      = ""
                _pending_score     = 0.0
                _candles_since_sig = 0

                # DC core: block entries against 4h Donchian HTF trend
                if enable_dc_core and higher_slice:
                    _dc_ok, _ = donchian_htf_filter(
                        higher_slice, desired_side, period=20,
                        long_cut=dc_long_cut, short_cut=dc_short_cut,
                    )
                    if not _dc_ok:
                        continue

                grade         = sig.get("grade", "B")
                atr_pct       = sig.get("atr_pct", ATR_BASELINE.get(tf, 0.009))
                mtf_size_mult = sig.get("mtf_size_mult", 1.0)

                # T16 reversal uses wider SL/TP multipliers for choppy-market entries
                sl_atr_mult = sig.get("sl_atr_override", st["sl_atr"])
                _bt_tp_m    = _style_cfg.get("backtest_default_tp_mult", 1.0)
                tp_atr_mult = sig.get("tp_atr_override", st["tp_atr"] * _bt_tp_m)
                sl_pct = min(atr_pct * sl_atr_mult, st["sl_max"] / 100)
                tp_pct = min(atr_pct * tp_atr_mult, st["tp_max"] / 100)

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

                # ── Change 5: Mode position_size_mult ─────────────────
                mode_size_mult = _mode_cfg.get("position_size_mult", 1.0)

                # Post-reset cap: check recovery (+10% from reset base) before sizing
                if _post_reset_active and equity >= _post_reset_base * 1.10:
                    _post_reset_active = False
                    _recent_90d = sum(
                        1 for r in _floor_resets
                        if (candle["t"] - r["ts"]) <= 90 * 86_400_000
                    )
                    if _recent_90d < 2:
                        _post_reset_cap = 1.0  # recovered, lift cap

                effective_size = (
                    c["size"] * mtf_size_mult * vol_mult * dd_mult
                    * mode_size_mult * _post_reset_cap
                )

                # Degrade Grade B → A if T1 leg would be below $5 (mirrors live min-order check)
                if grade == "B" and equity * effective_size * 0.60 < 5.0:
                    grade = "A"

                side = desired_side

                # Grid v2 mutual exclusion: trend signal wins, close any grid levels first
                if enable_grid_v2 and _gv2_active:
                    _gv2_fees_me = EXCHANGE_FEES[exchange]["taker"] * 2 + EXCHANGE_FEES[exchange]["slippage"]
                    for _lv in _gv2_levels:
                        if not _lv["filled"] or _lv.get("tp_done"):
                            continue
                        _lv_move = (
                            (candle["close"] - _lv["entry_price"]) / _lv["entry_price"]
                            if _lv["side"] == "LONG"
                            else (_lv["entry_price"] - candle["close"]) / _lv["entry_price"]
                        )
                        _lv_pnl = _lv["equity_at_open"] * _lv["size_pct"] * (_lv_move - _gv2_fees_me)
                        equity += _lv_pnl
                        trades.append({
                            "pnl":     _lv_pnl,
                            "outcome": "gv2_trend_exit",
                            "side":    _lv["side"],
                            "grade":   "B",
                            "signal":  "GRID_V2",
                            "score":   0.60,
                            "regime":  "RANGING",
                            "open_ts": _lv["fill_ts"],
                            "exit_ts": candle["t"],
                            "entry":   _lv["entry_price"],
                            "sl_pct":  0.0,
                            "tp_pct":  abs(_gv2_midpoint - _lv["entry_price"]) / _lv["entry_price"],
                            "scaled":  False,
                        })
                    equity_curve.append({"ts": candle["t"], "equity": round(equity, 4)})
                    _gv2_active = False
                    _gv2_levels = []

                open_trade = {
                    "entry":          candle["close"],
                    "side":           side,
                    "grade":          grade,
                    "sl_pct":         sl_pct,
                    "tp_pct":         tp_pct,
                    "t1_tp_mult":     t1_tp_mult,
                    "effective_size": effective_size,
                    "leverage":       min(c["leverage"], 3) if style == "SWING" and mode == "AGGRESSIVE" else c["leverage"],
                    "equity_at_open": equity,
                    "atr_pct":        atr_pct,
                    "adx_at_open":    cur_adx,
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
                    "regime":  regime,
                }

        # ── 3. Calculate metrics ───────────────────────────────────────────
        buy_hold_entry = main_candles[WARMUP_CANDLES]["close"]
        buy_hold_exit  = main_candles[-1]["close"]
        result_json    = _calc_metrics(trades, equity_curve, orig_start_equity, buy_hold_entry, buy_hold_exit)
        result_json["candles_fetched"]   = len(main_candles)
        result_json["sim_candles_used"]  = len(main_candles) - WARMUP_CANDLES
        result_json["floor_resets"]      = _floor_resets
        result_json["floor_reset_count"] = len(_floor_resets)
        result_json["session_note"] = (
            "Session quality computed using historical candle timestamp (candle_ts_ms). "
            "Daily bias passed from 30-day daily slice aligned to each candle. "
            "enable_mr_4h_gate=False matches live engine flags."
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
    enable_t16: bool = True,
    enable_t18: bool = True,
    enable_s4: bool = False,
    enable_mr_4h_gate: bool = False,
    enable_mr_signal: bool = True,
    enable_mr_4h_signal: bool = False,
    enable_regime_gate: bool = False,
    enable_e2: bool = False,
    enable_step2_score: bool = False,
    enable_grade_b_all: bool = False,
    enable_score_cap: bool = False,
    score_cap: float = 0.78,
    t1_tp_mult: float = 1.0,
    enable_floor: bool = True,
    random_seed: Optional[int] = None,
    score_floor: float = 0.0,
    e2_hold_mult: float = 1.0,
    enable_grid: bool = False,
    enable_dc_core: bool = False,
    dc_long_cut: float = 0.35,
    dc_short_cut: float = 0.65,
    enable_grid_v2: bool = False,
    gv2_adx_thresh: float = 15.0,
    gv2_n_candles:  int   = 4,
    gv2_max_levels: int   = 3,
    gv2_step_mult:  float = 0.50,
    gv2_buffer_atr: float = 0.75,
    gv2_cooldown:   int   = 4,
    gv2_equity_pct: float = 0.25,
    enable_sentiment_filter: bool = False,
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
              start_ms, end_ms, start_equity, enable_t16, enable_t18, enable_s4,
              enable_mr_4h_gate, enable_mr_signal, enable_mr_4h_signal, enable_regime_gate, enable_e2,
              enable_step2_score, enable_grade_b_all, enable_score_cap, score_cap,
              t1_tp_mult, enable_floor, random_seed, score_floor, e2_hold_mult,
              enable_grid, enable_dc_core, dc_long_cut, dc_short_cut,
              enable_grid_v2, gv2_adx_thresh, gv2_n_candles, gv2_max_levels,
              gv2_step_mult, gv2_buffer_atr, gv2_cooldown, gv2_equity_pct,
              enable_sentiment_filter),
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
