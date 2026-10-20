"""
run_backtest_2020_2026.py — Full backtest BTCUSDT 1h, Jan 2020 → Oct 2026

Covers all engine upgrades:
  - 4-layer signal scoring (regime / direction / entry / momentum)
  - Market structure (swing HH/HL) + daily bias
  - T16 Reversal detection (extreme RSI + volume spike)
  - Mean Reversion (Bollinger Band touch in ranging market)
  - 4 issue fixes (threshold capping, HTF timeout, volume bypass, logging)
  - Grade A / B split sizing

Run:  python3 run_backtest_2020_2026.py

Baseline (2024-2025 BTC 1h MINI_ASYM): Sharpe 0.36 | WR 34.2% | 281 trades
"""
from __future__ import annotations
import json, math, os, statistics, sys, time
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from config import DUBAI_TZ, TRADE_STYLE_PARAMS
from indicators import (
    _compute_signal_layers, _adx as _adx_fn,
    get_market_regime, check_pullback_entry, check_volume_hour_confirmed,
)
from coin_params import get_coin_params
from mode_params import get_mode_config, get_style_config

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOL    = "BTCUSDT"
MODE      = "MINI_ASYM"
STYLE     = "DAY_TRADE"
TF        = "1h"
HIGHER_TF = "4h"
WEEKLY_TF = "1w"
WARMUP    = 240
START_MS  = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS    = int(datetime(2026, 10, 12, tzinfo=timezone.utc).timestamp() * 1000)

BASELINE_2024_2025 = {"sharpe": -0.16, "wr_pct": 20.8, "trades": 1337}  # corrected (proper candle timestamps)
ENABLE_TRAIL          = True   # beyond-TP trail (deployed)
ENABLE_FIX4           = False  # ADX position scaling — REJECTED (hurts at 19% WR)
ENABLE_FIX3           = False  # entry body gate — REJECTED (killed avg win)
ENABLE_CANDLE_PAT     = False  # Full candle pattern system — REJECTED (Sharpe -0.23 vs -0.16 baseline)
ENABLE_WEEKLY_GATE    = False  # Fix 1: weekly EMA9 slope gate — REJECTED (2022 got worse -$125)
ENABLE_FIX2           = False  # Fix 2: ADX floor 16→20 — DEPLOYED to indicators.py (adx_min=20 is now baseline)
ENABLE_MR_4H_GATE     = False  # Fix 3: MR 4h direction gate — DEPLOYED (enable_mr_4h_gate=True in main.py)

MODE_PRESETS = {"MINI_ASYM": {"size": 0.65, "leverage": 6}}
FEE  = {"maker": 0.00020, "taker": 0.00055, "slip": 0.00025}
ST   = TRADE_STYLE_PARAMS["DAY_TRADE"]
CACHE_FILE = os.path.join(os.path.dirname(__file__), "backtest_2020_2026_cache.json")


# ── Candle fetch (Binance 1000/page) ──────────────────────────────────────────
def fetch_binance(symbol: str, tf: str, start_ms: int, end_ms: int) -> List[Dict]:
    url = "https://api.binance.com/api/v3/klines"
    candles, current, pages = [], start_ms, 0
    print(f"  Fetching {symbol} {tf}...", end="", flush=True)
    while current < end_ms:
        try:
            r = httpx.get(url, params={
                "symbol": symbol, "interval": tf,
                "startTime": current, "endTime": end_ms, "limit": 1000,
            }, timeout=20)
            if r.status_code != 200:
                time.sleep(2)
                continue
            data = r.json()
            if not data:
                break
            for k in data:
                candles.append({
                    "t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                })
            current = int(data[-1][0]) + 1
            pages += 1
            if len(data) < 1000:
                break
            time.sleep(0.12)
            if pages % 5 == 0:
                print(".", end="", flush=True)
        except Exception as e:
            print(f" [retry:{e}]", end="", flush=True)
            time.sleep(3)
    print(f" {len(candles):,} candles")
    return candles


def load_candles() -> tuple:
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print("  Using cached candles (delete backtest_2020_2026_cache.json to re-fetch)")
        except Exception:
            cache = {}

    main_key   = f"{SYMBOL}_{TF}"
    higher_key = f"{SYMBOL}_{HIGHER_TF}"
    weekly_key = f"{SYMBOL}_{WEEKLY_TF}"
    changed    = False

    if main_key not in cache:
        cache[main_key] = fetch_binance(SYMBOL, TF, START_MS, END_MS)
        changed = True
    else:
        print(f"  {SYMBOL} {TF}: {len(cache[main_key]):,} candles (cached)")

    if higher_key not in cache:
        cache[higher_key] = fetch_binance(SYMBOL, HIGHER_TF, START_MS, END_MS)
        changed = True
    else:
        print(f"  {SYMBOL} {HIGHER_TF}: {len(cache[higher_key]):,} candles (cached)")

    if weekly_key not in cache:
        cache[weekly_key] = fetch_binance(SYMBOL, WEEKLY_TF, START_MS, END_MS)
        changed = True
    else:
        print(f"  {SYMBOL} {WEEKLY_TF}: {len(cache[weekly_key]):,} candles (cached)")

    if changed:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

    return cache[main_key], cache[higher_key], cache[weekly_key]


# ── Helpers ───────────────────────────────────────────────────────────────────
def aligned_slice(candles: List[Dict], before_ts: int, n: int) -> List[Dict]:
    lo, hi, idx = 0, len(candles) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if candles[mid]["t"] <= before_ts:
            idx = mid; lo = mid + 1
        else:
            hi = mid - 1
    if idx < 0:
        return []
    return candles[max(0, idx - n + 1): idx + 1]


def trade_fee(size_usd: float, is_tp: bool) -> float:
    return size_usd * (FEE["maker"] + (FEE["maker"] if is_tp else FEE["taker"]) + FEE["slip"])


def check_exit(trade: Dict, candle: Dict) -> Optional[Dict]:
    """
    Simulates T1/T2 as separate phases for both trail and no-trail modes.

    Phase 1 (t1_done not set): wait for T1 (1× TP) or SL.
      - SL hit → close full position at loss.
      - T1 hit → close 60% at 1× TP, keep T2 (40%) open.

    Phase 2 (t1_done=True): T2 running with breakeven SL.
      ENABLE_TRAIL=False → close T2 when 2× TP hit, or breakeven SL.
      ENABLE_TRAIL=True  → trail T2 from 2× TP floor, close on trail stop.
    """
    side, entry = trade["side"], trade["entry"]
    sl, tp  = trade["sl_pct"], trade["tp_pct"]
    lev, eq = trade["leverage"], trade["equity_at_open"]
    hi, lo  = candle["high"], candle["low"]
    sl_p  = entry * (1 - sl)       if side == "LONG" else entry * (1 + sl)
    tp_p  = entry * (1 + tp)       if side == "LONG" else entry * (1 - tp)
    tp_p2 = entry * (1 + tp * 2.0) if side == "LONG" else entry * (1 - tp * 2.0)
    sl_hit  = lo <= sl_p  if side == "LONG" else hi >= sl_p
    tp_hit  = hi >= tp_p  if side == "LONG" else lo <= tp_p
    size    = trade["effective_size"] * eq

    # ── Phase 2: T2 holding after T1 closed ──────────────────────────
    if trade.get("t1_done"):
        t2_size = float(trade["t2_size"])
        t1_pnl  = float(trade["t1_pnl"])
        be_hit  = (lo <= entry if side == "LONG" else hi >= entry)

        tp2_hit = (hi >= tp_p2 if side == "LONG" else lo <= tp_p2)

        if ENABLE_TRAIL:
            # Activate trail when 2× TP first touched
            if tp2_hit and not trade.get("tp_trail_active"):
                trade["tp_trail_active"] = True
                trade["tp_trail_peak"]   = hi if side == "LONG" else lo

            if trade.get("tp_trail_active"):
                # Ratchet peak
                if side == "LONG":
                    trade["tp_trail_peak"] = max(float(trade.get("tp_trail_peak", tp_p2)), hi)
                else:
                    trade["tp_trail_peak"] = min(float(trade.get("tp_trail_peak", tp_p2)), lo)
                _tp_peak    = float(trade["tp_trail_peak"])
                _atr        = float(trade.get("atr_pct", 0.009))
                _adx_open   = float(trade.get("adx_at_open", 25.0))
                _trail_mult = 1.5 if _adx_open >= 35.0 else 1.0
                if side == "LONG":
                    _beyond_sl = max(_tp_peak * (1.0 - _trail_mult * _atr), tp_p2)
                    _trail_hit = lo <= _beyond_sl
                    if _trail_hit:
                        _move = max((_beyond_sl - entry) / entry, tp * 2.0)
                        pnl2 = t2_size * _move * lev - trade_fee(t2_size, True)
                        return {"outcome": "TRAIL_STOP", "pnl": t1_pnl + pnl2}
                else:
                    _beyond_sl = min(_tp_peak * (1.0 + _trail_mult * _atr), tp_p2)
                    _trail_hit = hi >= _beyond_sl
                    if _trail_hit:
                        _move = max((entry - _beyond_sl) / entry, tp * 2.0)
                        pnl2 = t2_size * _move * lev - trade_fee(t2_size, True)
                        return {"outcome": "TRAIL_STOP", "pnl": t1_pnl + pnl2}
        else:
            # No trail: T2 closes at fixed 2× TP
            if tp2_hit:
                tp2  = min(tp * 2.0, ST["tp_max"] / 100)
                pnl2 = t2_size * tp2 * lev - trade_fee(t2_size, True)
                return {"outcome": "TP_HIT", "pnl": t1_pnl + pnl2}

        # Breakeven SL (T2 free ride protection)
        if be_hit:
            return {"outcome": "SL_HIT", "pnl": t1_pnl - trade_fee(t2_size, False)}

        return None  # T2 still open

    # ── Phase 1: Full position, waiting for T1 or SL ─────────────────
    if tp_hit and sl_hit:
        return {"outcome": "SL_HIT", "pnl": -size * sl * lev - trade_fee(size, False)}

    if tp_hit:
        t1   = size * 0.60
        pnl1 = t1 * tp * lev - trade_fee(t1, True)
        t2   = size * 0.40
        trade["t1_done"] = True
        trade["t2_size"] = t2
        trade["t1_pnl"]  = pnl1
        return None  # T2 still open — check again next candle

    if sl_hit:
        return {"outcome": "SL_HIT", "pnl": -size * sl * lev - trade_fee(size, False)}

    return None


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(main_c: List[Dict], higher_c: List[Dict], weekly_c: List[Dict] = None) -> Dict:
    c         = MODE_PRESETS[MODE]
    mode_cfg  = get_mode_config(MODE)
    style_cfg = get_style_config(STYLE)
    coin_p    = get_coin_params(SYMBOL)

    equity       = 1000.0
    peak_equity  = 1000.0
    floor_hit    = False
    trades: List[Dict] = []
    pnls:   List[float] = []
    open_trade: Optional[Dict] = None
    is_parabolic = False
    sig_pending   = False
    pending_side  = ""
    candles_since = 0

    # Year → stats tracking
    year_trades: Dict[int, List] = {}

    total = len(main_c) - WARMUP
    for i in range(WARMUP, len(main_c)):
        candle = main_c[i]
        pct = (i - WARMUP) / max(1, total) * 100
        if i % 500 == 0:
            print(f"\r  Simulating {pct:.0f}% | equity=${equity:.0f} | trades={len(trades)}...", end="", flush=True)

        # ── Exit check ────────────────────────────────────────────────
        if open_trade:
            result = check_exit(open_trade, candle)
            if result:
                pnl = result["pnl"]
                equity += pnl
                peak_equity = max(peak_equity, equity)
                pnls.append(pnl)
                yr = datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc).year
                trade_rec = {
                    "outcome": result["outcome"], "pnl": pnl,
                    "year": yr, "signal_type": open_trade.get("signal_type", "NORMAL"),
                    "adx_at_open": open_trade.get("adx_at_open", 0.0),
                }
                trades.append(trade_rec)
                year_trades.setdefault(yr, []).append(trade_rec)
                open_trade = None

        # Track floor hits for reporting (don't stop — floor is live trading protection,
        # not a backtest terminator; we want to see full 6-year performance)
        if peak_equity > 0 and (peak_equity - equity) / peak_equity >= 0.15:
            floor_hit = True

        if open_trade is not None:
            continue

        # ── Market regime pre-check ───────────────────────────────────
        klines        = main_c[max(0, i - 299): i + 1]
        higher_slice  = aligned_slice(higher_c, candle["t"], 100)
        weekly_slice  = aligned_slice(weekly_c, candle["t"], 15) if weekly_c else []

        cur_adx = 0.0
        if len(klines) >= 15:
            try:
                cur_adx = _adx_fn(
                    [k["high"]  for k in klines],
                    [k["low"]   for k in klines],
                    [k["close"] for k in klines], 14,
                ) or 0.0
            except Exception:
                cur_adx = 0.0

        _rng_thresh = style_cfg.get("ranging_adx_threshold", 15)
        regime, is_parabolic = get_market_regime([], cur_adx, is_parabolic, ranging_threshold=_rng_thresh)

        # RANGING: don't skip — let _compute_signal_layers handle it.
        # In RANGING, mean reversion can fire (ADX < 20 + BB touch).
        # Only skip regimes that are explicitly disallowed by mode config.
        # PARABOLIC is allowed (LONG only), RANGING + TRENDING are allowed.
        if regime not in mode_cfg["allowed_regimes"] and regime != "RANGING":
            if sig_pending:
                sig_pending = False; pending_side = ""; candles_since = 0
            continue

        _param_overrides = {}
        _allowed_sides: Optional[List[str]] = None
        if regime == "PARABOLIC":
            _param_overrides = {"min_score": max(0.35, coin_p["score_threshold"] - 0.05)}
            _allowed_sides = ["LONG"]
        if ENABLE_FIX2:
            _param_overrides["adx_min_delta"] = 4  # raises adx_min from 16 → 20

        # ── Signal ────────────────────────────────────────────────────
        sig = _compute_signal_layers(
            klines, MODE, 1.0, higher_slice, STYLE, None,
            param_overrides=_param_overrides if _param_overrides else None,
            symbol=SYMBOL,
            enable_candle_pat=ENABLE_CANDLE_PAT,
            candle_ts_ms=candle["t"],
            weekly_klines=weekly_slice,
            enable_weekly_gate=ENABLE_WEEKLY_GATE,
            enable_mr_4h_gate=ENABLE_MR_4H_GATE,
        )

        if _allowed_sides and sig.get("ok") and sig.get("side") not in _allowed_sides:
            sig["ok"] = False

        # Volume hour confirmation (skip for mean reversion / reversal — they have their own vol checks)
        if sig.get("ok") and sig.get("signal_type", "NORMAL") == "NORMAL":
            _vol_ok, _, _ = check_volume_hour_confirmed(klines, coin_p)
            if not _vol_ok:
                sig["ok"] = False

        if not sig.get("ok"):
            if sig_pending:
                sig_pending = False; pending_side = ""; candles_since = 0
            continue

        desired_side  = sig["side"]
        signal_type   = sig.get("signal_type", "NORMAL")

        # Mean reversion and T16 reversal enter immediately (no pullback gate)
        if signal_type in ("MEAN_REVERSION", "REVERSAL"):
            sig_pending = False; pending_side = ""; candles_since = 0
            grade    = sig.get("grade", "A")
            atr_pct  = sig.get("atr_pct", 0.009)
            _sl_atr  = sig.get("sl_atr_override") or ST["sl_atr"]
            _tp_atr  = sig.get("tp_atr_override") or ST["tp_atr"]
            sl_pct   = min(atr_pct * _sl_atr, ST["sl_max"] / 100)
            tp_pct   = min(max(sl_pct * 1.5, atr_pct * _tp_atr), ST["tp_max"] / 100)
            eff_size = c["size"] if grade == "A" else c["size"] * 0.70
            # Fix 4: only scale UP on strong-trend NORMAL signals — never cut MR trades
            if ENABLE_FIX4 and signal_type == "NORMAL" and cur_adx > 0:
                _adx_sm = (1.25 if cur_adx < 50 else 1.50) if cur_adx >= 35 else 1.0
                eff_size *= _adx_sm
            open_trade = {
                "side": desired_side, "entry": candle["close"],
                "sl_pct": sl_pct, "tp_pct": tp_pct,
                "leverage": c["leverage"], "effective_size": eff_size,
                "equity_at_open": equity, "signal_type": signal_type,
                "atr_pct": atr_pct, "adx_at_open": cur_adx,
            }
            continue

        # Normal trend signal: pullback gate
        if not sig_pending:
            sig_pending   = True
            pending_side  = desired_side
            candles_since = 0
        elif pending_side != desired_side:
            pending_side  = desired_side
            candles_since = 0
        candles_since += 1

        _pb_res = check_pullback_entry(
            klines, pending_side, coin_p, regime,
            candles_since, trade_style=STYLE,
            enable_fix3=ENABLE_FIX3,
            strong_trend_override=sig.get("strong_trend_override", False),
        )
        if _pb_res == "TIMEOUT":
            sig_pending = False; pending_side = ""; candles_since = 0
            continue
        if _pb_res == "WAIT":
            continue

        sig_pending = False; pending_side = ""; candles_since = 0

        grade   = sig.get("grade", "B")
        atr_pct = sig.get("atr_pct", 0.009)
        sl_pct  = min(atr_pct * ST["sl_atr"], ST["sl_max"] / 100)
        tp_pct  = min(max(sl_pct * 1.5, atr_pct * ST["tp_atr"]), ST["tp_max"] / 100)
        eff_size = c["size"] if grade == "A" else c["size"] * 0.70
        # Fix 4: only scale UP on strong-trend NORMAL signals
        if ENABLE_FIX4 and signal_type == "NORMAL" and cur_adx > 0:
            _adx_sm = (1.25 if cur_adx < 50 else 1.50) if cur_adx >= 35 else 1.0
            eff_size *= _adx_sm

        open_trade = {
            "side": desired_side, "entry": candle["close"],
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "leverage": c["leverage"], "effective_size": eff_size,
            "equity_at_open": equity, "signal_type": signal_type,
            "atr_pct": atr_pct, "adx_at_open": cur_adx,
        }

    print(f"\r  Simulation complete — {len(trades)} trades        ")
    return {
        "trades": trades, "pnls": pnls,
        "final_equity": equity, "peak": peak_equity,
        "year_trades": year_trades, "floor_hit": floor_hit,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_stats(result: Dict, start_equity: float = 1000.0) -> Dict:
    trades = result["trades"]
    pnls   = result["pnls"]
    n      = len(trades)
    if n == 0:
        return {"trades": 0, "wr": 0, "sharpe": 0, "return_pct": 0,
                "profit_factor": 0, "max_dd_pct": 0, "calmar": 0}

    wins    = sum(1 for t in trades if t["outcome"] in ("TP_HIT", "TRAIL_STOP"))
    wr      = wins / n * 100
    final   = result["final_equity"]
    ret_pct = (final - start_equity) / start_equity * 100

    # Max drawdown
    equity  = start_equity
    peak    = start_equity
    max_dd  = 0.0
    for pnl in pnls:
        equity += pnl
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak
        max_dd  = max(max_dd, dd)

    # Profit factor
    gross_win  = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Sharpe (annualized, assuming 1h candles = ~8760 trades/year max)
    sharpe = 0.0
    if len(pnls) >= 2:
        avg    = statistics.mean(pnls)
        stddev = statistics.stdev(pnls)
        sharpe = round((avg / stddev) * math.sqrt(252), 2) if stddev > 0 else 0.0

    calmar = round(ret_pct / (max_dd * 100), 2) if max_dd > 0 else 0.0

    avg_win  = round(statistics.mean([p for p in pnls if p > 0]), 2) if any(p > 0 for p in pnls) else 0
    avg_loss = round(statistics.mean([p for p in pnls if p < 0]), 2) if any(p < 0 for p in pnls) else 0

    # Signal type breakdown
    by_type: Dict[str, Dict] = {}
    for t in trades:
        st = t.get("signal_type", "NORMAL")
        if st not in by_type:
            by_type[st] = {"count": 0, "wins": 0}
        by_type[st]["count"] += 1
        if t["outcome"] in ("TP_HIT", "TRAIL_STOP"):
            by_type[st]["wins"] += 1

    return {
        "trades": n, "wr": round(wr, 1), "sharpe": sharpe,
        "return_pct": round(ret_pct, 1), "profit_factor": pf,
        "max_dd_pct": round(max_dd * 100, 1), "calmar": calmar,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "final_equity": round(final, 2), "by_signal_type": by_type,
    }


def year_summary(year_trades: Dict[int, List], start_equity: float = 1000.0) -> Dict[int, Dict]:
    out = {}
    running_eq = start_equity
    for yr in sorted(year_trades.keys()):
        tlist = year_trades[yr]
        n     = len(tlist)
        wins  = sum(1 for t in tlist if t["outcome"] in ("TP_HIT", "TRAIL_STOP"))
        pnls  = []
        for t in tlist:
            pnls.append(t["pnl"])
            running_eq += t["pnl"]
        yr_ret = sum(pnls)
        out[yr] = {
            "trades": n, "wins": wins, "wr": round(wins / n * 100, 1) if n else 0,
            "pnl": round(yr_ret, 2),
        }
    return out


def adx_bucket_stats(result: Dict) -> Dict:
    """Split baseline trades by ADX 16–19 vs 20+ to see if the low-ADX trades are the bad ones."""
    lo, hi = {"trades": 0, "wins": 0, "pnls": []}, {"trades": 0, "wins": 0, "pnls": []}
    for t in result["trades"]:
        adx = t.get("adx_at_open", 0.0)
        win = t["outcome"] in ("TP_HIT", "TRAIL_STOP")
        bucket = lo if adx < 20 else hi
        bucket["trades"] += 1
        bucket["pnls"].append(t["pnl"])
        if win:
            bucket["wins"] += 1
    for b in (lo, hi):
        n = b["trades"]
        b["wr"]      = round(b["wins"] / n * 100, 1) if n else 0.0
        b["avg_pnl"] = round(sum(b["pnls"]) / n, 2) if n else 0.0
    return {"adx_lo": lo, "adx_hi": hi}


# ── Helpers: print A/B comparison block ──────────────────────────────────────
def print_ab(label_a, label_b, stats_a, stats_b, t_a, t_b, yr_a, yr_b, result_b, adx_buckets=None):
    print("\n" + "=" * 68)
    print("  A/B COMPARISON — 2020–Oct 2026  (BTCUSDT 1h MINI_ASYM)")
    print(f"  A: {label_a}  |  B: {label_b}")
    print("=" * 68)
    print(f"  {'Metric':<22} {'A':>14} {'B':>14} {'Delta':>10}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*10}")
    for label, k in [
        ("Trades",        "trades"),
        ("Win Rate %",    "wr"),
        ("Sharpe",        "sharpe"),
        ("Profit Factor", "profit_factor"),
        ("Avg Win $",     "avg_win"),
        ("Total Return %","return_pct"),
        ("Max Drawdown %","max_dd_pct"),
    ]:
        va, vb = stats_a[k], stats_b[k]
        delta = vb - va if isinstance(vb, (int, float)) else "—"
        print(f"  {label:<22} {va:>14.2f} {vb:>14.2f} {delta:>+10.2f}")
    print(f"  {'Run time':<22} {t_a:>13.1f}s {t_b:>13.1f}s")

    print("\n" + "-" * 68)
    print("  SIGNAL TYPE BREAKDOWN (B)")
    print("-" * 68)
    for st, d in stats_b["by_signal_type"].items():
        wr = round(d["wins"] / d["count"] * 100, 1) if d["count"] else 0
        print(f"  {st:<24} {d['count']:>4} trades  |  WR {wr:>5.1f}%")

    print("\n" + "-" * 68)
    print("  YEAR-BY-YEAR  (A vs B)")
    print("-" * 68)
    print(f"  {'Year':<6} {'A Trades':>8} {'A WR':>6} {'A P&L':>9}   {'B Trades':>8} {'B WR':>6} {'B P&L':>9}")
    print(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*9}   {'-'*8} {'-'*6} {'-'*9}")
    all_yrs = sorted(set(list(yr_a.keys()) + list(yr_b.keys())))
    for yr in all_yrs:
        da = yr_a.get(yr, {"trades": 0, "wr": 0.0, "pnl": 0.0})
        db = yr_b.get(yr, {"trades": 0, "wr": 0.0, "pnl": 0.0})
        print(f"  {yr:<6} {da['trades']:>8} {da['wr']:>5.1f}% {da['pnl']:>+9.2f}   "
              f"{db['trades']:>8} {db['wr']:>5.1f}% {db['pnl']:>+9.2f}")

    if adx_buckets:
        lo, hi = adx_buckets["adx_lo"], adx_buckets["adx_hi"]
        total_t = lo["trades"] + hi["trades"]
        lo_pct  = lo["trades"] / total_t * 100 if total_t else 0
        print("\n" + "-" * 68)
        print("  ADX BUCKET ANALYSIS (Baseline A — adx_min=16)")
        print("  Shows whether ADX 16–19 trades were actually bad")
        print("-" * 68)
        print(f"  {'Range':<12} {'Trades':>8} {'% Total':>8} {'WR %':>7} {'Avg P&L':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")
        print(f"  {'ADX 16–19':<12} {lo['trades']:>8} {lo_pct:>7.1f}% {lo['wr']:>6.1f}% {lo['avg_pnl']:>+10.2f}")
        print(f"  {'ADX 20+':<12} {hi['trades']:>8} {100-lo_pct:>7.1f}% {hi['wr']:>6.1f}% {hi['avg_pnl']:>+10.2f}")
        verdict = "WORSE" if lo["wr"] < hi["wr"] else "SIMILAR/BETTER"
        print(f"\n  ADX 16–19 trades: WR {lo['wr']:.1f}% vs ADX 20+ WR {hi['wr']:.1f}%")
        print(f"  → Low-ADX trades are {verdict} than high-ADX trades.")

    print("\n" + "-" * 68)
    print("  VS BASELINE (corrected 6.8yr, Sharpe -0.16)")
    print("-" * 68)
    b = BASELINE_2024_2025
    print(f"  {'Metric':<18} {'Baseline':>10} {'B (new)':>10} {'Delta':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Sharpe':<18} {b['sharpe']:>10.2f} {stats_b['sharpe']:>10.2f} {stats_b['sharpe']-b['sharpe']:>+10.2f}")
    print(f"  {'Win Rate %':<18} {b['wr_pct']:>10.1f} {stats_b['wr']:>10.1f} {stats_b['wr']-b['wr_pct']:>+10.1f}")
    print(f"  {'Trade Count':<18} {b['trades']:>10} {stats_b['trades']:>10} {stats_b['trades']-b['trades']:>+10}")
    print("=" * 68)

    print()
    s = stats_b["sharpe"]
    pf = stats_b["profit_factor"]
    if s >= 0.50 and pf >= 1.10:
        print("  VERDICT: Engine is profitable and improving.")
    elif pf >= 1.0:
        print("  VERDICT: Engine is marginally profitable — more upgrades will help.")
    else:
        print("  VERDICT: Engine needs further tuning — profit factor below 1.0.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global ENABLE_TRAIL, ENABLE_CANDLE_PAT, ENABLE_WEEKLY_GATE, ENABLE_FIX2, ENABLE_MR_4H_GATE

    print("\n" + "=" * 68)
    print("  Asymmetric AI — A/B Backtest: BTCUSDT 1h  Jan 2020 → Oct 2026")
    print("  Mode: MINI_ASYM | Style: DAY_TRADE | Start equity: $1,000")
    print("=" * 68)

    print("\nStep 1 — Fetching candles:")
    main_c, higher_c, weekly_c = load_candles()
    total_candles = len(main_c) - WARMUP
    years = (END_MS - START_MS) / (1000 * 60 * 60 * 24 * 365.25)
    print(f"  Data range: {total_candles:,} active candles | {years:.1f} years")

    # ── Run A: current baseline (MR 4h gate OFF) ─────────────────────
    print(f"\nStep 2a — Baseline (adx_min=20, MR 4h gate OFF)...")
    ENABLE_MR_4H_GATE = False
    t0       = time.time()
    result_a = simulate(main_c, higher_c, weekly_c)
    t_a      = time.time() - t0
    stats_a  = compute_stats(result_a)
    yr_a     = year_summary(result_a["year_trades"])

    # ── Run B: Fix 3 ON (MR 4h gate) ─────────────────────────────────
    print(f"\nStep 2b — Fix 3 (MR 4h direction gate ON)...")
    ENABLE_MR_4H_GATE = True
    t0       = time.time()
    result_b = simulate(main_c, higher_c, weekly_c)
    t_b      = time.time() - t0
    stats_b  = compute_stats(result_b)
    yr_b     = year_summary(result_b["year_trades"])

    print_ab(
        "Baseline (MR gate OFF)",
        "Fix 3 (MR 4h gate ON)",
        stats_a, stats_b, t_a, t_b, yr_a, yr_b, result_b,
    )


if __name__ == "__main__":
    main()
