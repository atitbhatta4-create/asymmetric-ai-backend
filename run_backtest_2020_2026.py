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
    _candle_pattern, get_pattern_exec_params,
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
ENABLE_PAT_EXEC       = False  # Fix 4: pattern-adjusted execution (size + trail by entry candle pattern)
# Test 6 — T18 MR exemption (model T18 as 70% Grade A; test MR at 100%)
ENABLE_T18            = False  # when True: all Grade A enters at 70% (T18 model)
T18_MR_EXEMPT         = False  # when True: MR/REVERSAL Grade A bypasses T18 → 100% size
# Test 7 — Choppy market trail delay
ENABLE_CHOPPY_TRAIL   = False  # when True: trail activates at 3× TP (not 2×) in RANGING/CHOPPY regime
ENABLE_HTF_GATE_NORMAL = False  # Task 4: block NORMAL entries that go against the 4h trend direction
ENABLE_OE_GATE         = False  # Block overextended entries: LONG if RSI>58 & EMA50>+1%; SHORT if RSI<42 & EMA50<-1%
ENABLE_HARD_FLOOR      = True   # 15% hard floor: stop new entries when equity < peak * 0.85, reset at year start

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


def check_exit(trade: Dict, candle: Dict, candle_regime: str = "TRENDING") -> Optional[Dict]:
    """
    Simulates T1/T2 as separate phases for both trail and no-trail modes.

    Phase 1 (t1_done not set): wait for T1 (1× TP) or SL.
      - SL hit → close full position at loss.
      - T1 hit → close 60% at 1× TP, keep T2 (40%) open.

    Phase 2 (t1_done=True): T2 running with breakeven SL.
      ENABLE_TRAIL=False → close T2 when 2× TP hit, or breakeven SL.
      ENABLE_TRAIL=True  → trail T2 from activation floor, close on trail stop.
      ENABLE_CHOPPY_TRAIL=True → trail activates at 3× TP (not 2×) in RANGING/CHOPPY.
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
            # Determine trail activation threshold (Test 7: raise to 3× TP in choppy markets)
            if ENABLE_CHOPPY_TRAIL and candle_regime in ("RANGING", "CHOPPY"):
                _act_mult  = 3.0
            else:
                _act_mult  = 2.0
            _tp_act_price = entry * (1 + tp * _act_mult) if side == "LONG" else entry * (1 - tp * _act_mult)
            tp_act_hit    = (hi >= _tp_act_price if side == "LONG" else lo <= _tp_act_price)

            # Activate trail when activation threshold is first touched
            if tp_act_hit and not trade.get("tp_trail_active"):
                trade["tp_trail_active"] = True
                trade["tp_trail_peak"]   = hi if side == "LONG" else lo
                trade["tp_trail_floor"]  = _tp_act_price  # floor = activation level

            if trade.get("tp_trail_active"):
                # Ratchet peak
                if side == "LONG":
                    trade["tp_trail_peak"] = max(float(trade.get("tp_trail_peak", _tp_act_price)), hi)
                else:
                    trade["tp_trail_peak"] = min(float(trade.get("tp_trail_peak", _tp_act_price)), lo)
                _tp_peak    = float(trade["tp_trail_peak"])
                _tp_floor   = float(trade.get("tp_trail_floor", tp_p2))
                _atr        = float(trade.get("atr_pct", 0.009))
                _adx_open   = float(trade.get("adx_at_open", 25.0))
                _pat_ovr    = float(trade.get("pat_trail_mult", 0.0))
                _trail_mult = _pat_ovr if _pat_ovr > 0.0 else (1.5 if _adx_open >= 35.0 else 1.0)
                if side == "LONG":
                    _beyond_sl = max(_tp_peak * (1.0 - _trail_mult * _atr), _tp_floor)
                    _trail_hit = lo <= _beyond_sl
                    if _trail_hit:
                        _move = max((_beyond_sl - entry) / entry, tp * _act_mult)
                        pnl2 = t2_size * _move * lev - trade_fee(t2_size, True)
                        return {"outcome": "TRAIL_STOP", "pnl": t1_pnl + pnl2}
                else:
                    _beyond_sl = min(_tp_peak * (1.0 + _trail_mult * _atr), _tp_floor)
                    _trail_hit = hi >= _beyond_sl
                    if _trail_hit:
                        _move = max((entry - _beyond_sl) / entry, tp * _act_mult)
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


# ── +DI / -DI helper ──────────────────────────────────────────────────────────
def _adx_full_di(klines: List[Dict], period: int = 14):
    """Return (adx, plus_di, minus_di) for the last candle in klines."""
    highs  = [k["high"]  for k in klines]
    lows   = [k["low"]   for k in klines]
    closes = [k["close"] for k in klines]
    if len(closes) < period * 2 + 1:
        return 0.0, 0.0, 0.0
    pdms, ndms, trs = [], [], []
    for i in range(1, len(closes)):
        hd = highs[i] - highs[i - 1]; ld = lows[i - 1] - lows[i]
        pdms.append(hd if hd > ld and hd > 0 else 0.0)
        ndms.append(ld if ld > hd and ld > 0 else 0.0)
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
    def _w(s):
        v = sum(s[:period]); out = [v]
        for x in s[period:]: v = v - v/period + x; out.append(v)
        return out
    s_tr, s_p, s_n = _w(trs), _w(pdms), _w(ndms)
    dxs = []
    for i in range(len(s_tr)):
        pdi = 100 * s_p[i] / s_tr[i] if s_tr[i] else 0
        ndi = 100 * s_n[i] / s_tr[i] if s_tr[i] else 0
        dxs.append(100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else 0)
    if len(dxs) < period:
        return 0.0, 0.0, 0.0
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]: adx = (adx * (period - 1) + dx) / period
    pdi_last = 100 * s_p[-1] / s_tr[-1] if s_tr[-1] else 0.0
    ndi_last = 100 * s_n[-1] / s_tr[-1] if s_tr[-1] else 0.0
    return round(adx, 2), round(pdi_last, 2), round(ndi_last, 2)


# ── Entry-time analysis helpers ───────────────────────────────────────────────
def _rsi14(klines: List[Dict]) -> float:
    """Wilder smoothed 14-period RSI on the most recent candles."""
    closes = [k["close"] for k in klines]
    if len(closes) < 16:
        return 50.0
    period = 14
    gains  = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    avg_g  = sum(gains[:period]) / period
    avg_l  = sum(losses[:period]) / period
    for j in range(period, len(gains)):
        avg_g = (avg_g * 13 + gains[j]) / 14
        avg_l = (avg_l * 13 + losses[j]) / 14
    if avg_l == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_g / avg_l), 1)


def _ema_pct_dist(klines: List[Dict], period: int = 50) -> float:
    """% distance of last close from the {period}-period EMA."""
    closes = [k["close"] for k in klines]
    if len(closes) < period:
        return 0.0
    ema  = sum(closes[:period]) / period
    mult = 2.0 / (period + 1)
    for c_ in closes[period:]:
        ema = c_ * mult + ema * (1 - mult)
    return round((closes[-1] - ema) / ema * 100, 2) if ema else 0.0


def _daily_bias(klines: List[Dict]) -> bool:
    """True if current close > close 24 candles ago (1-day lookback on 1h data)."""
    closes = [k["close"] for k in klines]
    return closes[-1] > closes[-25] if len(closes) >= 25 else True


def _dd_mult(equity: float, peak: float) -> float:
    """Position-size multiplier from live drawdown tiers.
    Mirrors main.py: -4% → 0.65, -7% → 0.40, -10% → 0.25, <4% → 1.0."""
    if peak <= 0:
        return 1.0
    dd = (peak - equity) / peak
    if dd >= 0.10:
        return 0.25
    if dd >= 0.07:
        return 0.40
    if dd >= 0.04:
        return 0.65
    return 1.0


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(main_c: List[Dict], higher_c: List[Dict], weekly_c: List[Dict] = None) -> Dict:
    c         = MODE_PRESETS[MODE]
    mode_cfg  = get_mode_config(MODE)
    style_cfg = get_style_config(STYLE)
    coin_p    = get_coin_params(SYMBOL)

    equity       = 1000.0
    peak_equity  = 1000.0
    floor_hit    = False
    floor_stopped = False   # hard floor: stop new entries until year restart
    trades: List[Dict] = []
    pnls:   List[float] = []
    open_trade: Optional[Dict] = None
    is_parabolic = False
    sig_pending   = False
    pending_side  = ""
    candles_since = 0
    _last_regime  = "TRENDING"  # prior-candle regime passed to check_exit for trail decisions

    # Year → stats tracking
    year_trades: Dict[int, List] = {}
    year_start_equity: Dict[int, float] = {}
    _tracked_yr = -1

    total = len(main_c) - WARMUP
    for i in range(WARMUP, len(main_c)):
        candle = main_c[i]
        _cyr = datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc).year
        if _cyr != _tracked_yr:
            year_start_equity[_cyr] = equity
            _tracked_yr = _cyr
            if floor_stopped:
                peak_equity = equity  # restart after floor stop: 15% floor now measured from current equity
            floor_stopped = False  # year restart — re-enable entries
        pct = (i - WARMUP) / max(1, total) * 100
        if i % 500 == 0:
            print(f"\r  Simulating {pct:.0f}% | equity=${equity:.0f} | trades={len(trades)}...", end="", flush=True)

        # ── Exit check ────────────────────────────────────────────────
        if open_trade:
            result = check_exit(open_trade, candle, candle_regime=_last_regime)
            if result:
                pnl = result["pnl"]
                equity += pnl
                peak_equity = max(peak_equity, equity)
                pnls.append(pnl)
                _close_dt = datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc)
                yr = _close_dt.year
                trade_rec = {
                    "outcome": result["outcome"], "pnl": pnl,
                    "year": yr, "signal_type": open_trade.get("signal_type", "NORMAL"),
                    "adx_at_open": open_trade.get("adx_at_open", 0.0),
                    "side": open_trade.get("side", "LONG"),
                    "month": _close_dt.strftime("%Y-%m"),
                    "open_date": open_trade.get("open_date", ""),
                    "plus_di": open_trade.get("plus_di", 0.0),
                    "minus_di": open_trade.get("minus_di", 0.0),
                    "htf_bullish": open_trade.get("htf_bullish", True),
                    "equity_at_open_val": open_trade.get("equity_at_open", 0.0),
                    "rsi_at_open": open_trade.get("rsi_at_open", 50.0),
                    "ema50_dist": open_trade.get("ema50_dist", 0.0),
                    "daily_bullish": open_trade.get("daily_bullish", True),
                }
                trades.append(trade_rec)
                year_trades.setdefault(yr, []).append(trade_rec)
                open_trade = None

        # Hard floor: equity < peak * 0.85 → stop new entries until year restart
        if peak_equity > 0 and (peak_equity - equity) / peak_equity >= 0.15:
            floor_hit = True
            if ENABLE_HARD_FLOOR:
                floor_stopped = True

        if open_trade is not None:
            continue

        if floor_stopped:
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
        _last_regime = regime  # used by check_exit() on the next candle

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
            # T18 model: Grade A at 70%; MR exempt when T18_MR_EXEMPT=True
            if grade == "A":
                _mr_exempt = T18_MR_EXEMPT and signal_type in ("MEAN_REVERSION", "REVERSAL")
                eff_size = c["size"] if (not ENABLE_T18 or _mr_exempt) else c["size"] * 0.70
            else:
                eff_size = c["size"] * 0.70
            pat_trail = 0.0
            if ENABLE_PAT_EXEC:
                _psm, pat_trail, sl_pct, _ = get_pattern_exec_params(klines, desired_side, sl_pct, atr_pct)
                eff_size = eff_size * _psm
            eff_size  *= _dd_mult(equity, peak_equity)
            _pdi, _ndi = _adx_full_di(klines)[1], _adx_full_di(klines)[2]
            _htf_bull  = higher_slice[-2]["close"] > higher_slice[-2]["open"] if len(higher_slice) >= 2 else True
            _rsi_now   = _rsi14(klines)
            _ema50     = _ema_pct_dist(klines)
            _d_bull    = _daily_bias(klines)
            open_trade = {
                "side": desired_side, "entry": candle["close"],
                "sl_pct": sl_pct, "tp_pct": tp_pct,
                "leverage": c["leverage"], "effective_size": eff_size,
                "equity_at_open": equity, "signal_type": signal_type,
                "atr_pct": atr_pct, "adx_at_open": cur_adx,
                "pat_trail_mult": pat_trail,
                "open_date": datetime.fromtimestamp(candle["t"]/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "plus_di": _pdi, "minus_di": _ndi, "htf_bullish": _htf_bull,
                "rsi_at_open": _rsi_now, "ema50_dist": _ema50, "daily_bullish": _d_bull,
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
        # T18 model: NORMAL Grade A at 70% (T18 applies); Grade B unchanged
        if grade == "A":
            eff_size = c["size"] if not ENABLE_T18 else c["size"] * 0.70
        else:
            eff_size = c["size"] * 0.70
        pat_trail = 0.0
        if ENABLE_PAT_EXEC:
            _psm, pat_trail, sl_pct, _ = get_pattern_exec_params(klines, desired_side, sl_pct, atr_pct)
            eff_size = eff_size * _psm

        _pdi2, _ndi2 = _adx_full_di(klines)[1], _adx_full_di(klines)[2]
        _htf_bull2   = higher_slice[-2]["close"] > higher_slice[-2]["open"] if len(higher_slice) >= 2 else True
        _rsi_now2    = _rsi14(klines)
        _ema50_2     = _ema_pct_dist(klines)
        _d_bull2     = _daily_bias(klines)

        if ENABLE_HTF_GATE_NORMAL:
            if (desired_side == "LONG" and not _htf_bull2) or (desired_side == "SHORT" and _htf_bull2):
                sig_pending = False; pending_side = ""; candles_since = 0
                continue

        if ENABLE_OE_GATE:
            if desired_side == "LONG" and _rsi_now2 > 58 and _ema50_2 > 1.0:
                sig_pending = False; pending_side = ""; candles_since = 0
                continue
            if desired_side == "SHORT" and _rsi_now2 < 42 and _ema50_2 < -1.0:
                sig_pending = False; pending_side = ""; candles_since = 0
                continue

        eff_size  *= _dd_mult(equity, peak_equity)
        open_trade = {
            "side": desired_side, "entry": candle["close"],
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "leverage": c["leverage"], "effective_size": eff_size,
            "equity_at_open": equity, "signal_type": signal_type,
            "atr_pct": atr_pct, "adx_at_open": cur_adx,
            "pat_trail_mult": pat_trail,
            "open_date": datetime.fromtimestamp(candle["t"]/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "plus_di": _pdi2, "minus_di": _ndi2, "htf_bullish": _htf_bull2,
            "rsi_at_open": _rsi_now2, "ema50_dist": _ema50_2, "daily_bullish": _d_bull2,
        }

    print(f"\r  Simulation complete — {len(trades)} trades        ")
    return {
        "trades": trades, "pnls": pnls,
        "final_equity": equity, "peak": peak_equity,
        "year_trades": year_trades, "floor_hit": floor_hit,
        "year_start_equity": year_start_equity,
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


def compute_stats_years(result: Dict, years: List[int]) -> Dict:
    """Compute stats for a subset of calendar years (for in-sample / OOS split)."""
    subset = [t for t in result["trades"] if t["year"] in years]
    if not subset:
        return {"trades": 0, "wr": 0.0, "sharpe": 0.0, "return_pct": 0.0,
                "max_dd_pct": 0.0, "start_eq": 0.0, "end_eq": 0.0}
    pnls   = [t["pnl"] for t in subset]
    wins   = sum(1 for t in subset if t["outcome"] in ("TP_HIT", "TRAIL_STOP"))
    wr     = wins / len(subset) * 100
    start_eq = result["year_start_equity"].get(min(years), 1000.0)
    total_pnl = sum(pnls)
    end_eq    = start_eq + total_pnl
    ret_pct   = total_pnl / start_eq * 100 if start_eq else 0.0
    # Max DD within subset (walking equity from start_eq)
    eq, peak, max_dd = start_eq, start_eq, 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100)
    # Sharpe
    sharpe = 0.0
    if len(pnls) >= 2:
        avg    = statistics.mean(pnls)
        stddev = statistics.stdev(pnls)
        sharpe = round((avg / stddev) * math.sqrt(252), 2) if stddev > 0 else 0.0
    return {
        "trades": len(subset), "wr": round(wr, 1), "sharpe": sharpe,
        "return_pct": round(ret_pct, 1), "max_dd_pct": round(max_dd, 1),
        "start_eq": round(start_eq, 0), "end_eq": round(end_eq, 0),
    }


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


# ── SHORT Signal Diagnostic ───────────────────────────────────────────────────
def run_signal_diagnostic(main_c: List[Dict], higher_c: List[Dict],
                           weekly_c: List[Dict], diag_year: int = 2022) -> Dict:
    """
    Full simulation pass from warmup, capturing for diag_year:
      - ADX distribution across ALL candles in the year
      - For candles where no trade is open: desired direction (LONG/SHORT)
        and which signal layer blocked the entry (regime / direction / entry /
        momentum / score)
      - Separate block counts for SHORT-desired vs all
    Trade state is tracked accurately from 2020 so a Dec 2021 trade that
    spills into Jan 2022 is handled correctly.
    """
    c          = MODE_PRESETS["MINI_ASYM"]
    mode_cfg   = get_mode_config(MODE)
    style_cfg  = get_style_config(STYLE)
    coin_p     = get_coin_params(SYMBOL)

    # ── Counters ──────────────────────────────────────────────────────────────
    adx_buckets: Dict[str, int] = {
        "<14": 0, "14-20": 0,
        "20-25": 0, "25-30": 0, "30-35": 0, "35-40": 0, "40+": 0,
    }
    # per-month ADX candle counts (for diag_year only)
    from collections import defaultdict as _dd2
    month_adx: dict = _dd2(lambda: {"20-25": 0, "25-30": 0, "30-35": 0, "35-40": 0, "40+": 0, "total": 0})
    # desired_dir: direction the HTF said to trade on candles the engine checked
    desired_dir: Dict[str, int] = {"LONG": 0, "SHORT": 0}
    blocked_all:   Dict[str, int] = {}   # block layer → count (all directions)
    blocked_short: Dict[str, int] = {}   # block layer → count (SHORT only)
    fired_dir:     Dict[str, int] = {"LONG": 0, "SHORT": 0}   # signals that fired

    open_trade: Optional[Dict] = None
    is_parabolic = False
    sig_pending   = False
    pending_side  = ""
    candles_since = 0
    _last_regime  = "TRENDING"

    for i in range(WARMUP, len(main_c)):
        candle   = main_c[i]
        candle_yr = datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc).year
        in_diag   = (candle_yr == diag_year)

        # ── Always: compute ADX for regime detection ──────────────────────────
        klines = main_c[max(0, i - 299): i + 1]
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

        # ── ADX distribution — only for diag_year candles ─────────────────────
        if in_diag:
            _mo = datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc).strftime("%Y-%m")
            month_adx[_mo]["total"] += 1
            if cur_adx < 14:
                adx_buckets["<14"] += 1
            elif cur_adx < 20:
                adx_buckets["14-20"] += 1
            elif cur_adx < 25:
                adx_buckets["20-25"] += 1
                month_adx[_mo]["20-25"] += 1
            elif cur_adx < 30:
                adx_buckets["25-30"] += 1
                month_adx[_mo]["25-30"] += 1
            elif cur_adx < 35:
                adx_buckets["30-35"] += 1
                month_adx[_mo]["30-35"] += 1
            elif cur_adx < 40:
                adx_buckets["35-40"] += 1
                month_adx[_mo]["35-40"] += 1
            else:
                adx_buckets["40+"] += 1
                month_adx[_mo]["40+"] += 1

        # ── Exit check (keeps trade state accurate) ───────────────────────────
        _rng_thresh  = style_cfg.get("ranging_adx_threshold", 15)
        regime, is_parabolic = get_market_regime([], cur_adx, is_parabolic,
                                                  ranging_threshold=_rng_thresh)
        _last_regime = regime
        if open_trade:
            result = check_exit(open_trade, candle, candle_regime=_last_regime)
            if result:
                open_trade = None

        if open_trade is not None:
            continue   # trade still running — don't check for new entry

        # ── Past the target year — stop (ADX already counted above) ───────────
        if candle_yr > diag_year:
            break

        # ── Regime filter (same as simulate) ─────────────────────────────────
        if regime not in mode_cfg["allowed_regimes"] and regime != "RANGING":
            if sig_pending:
                sig_pending = False; pending_side = ""; candles_since = 0
            continue

        # ── Call signal engine ─────────────────────────────────────────────────
        higher_slice = aligned_slice(higher_c, candle["t"], 100)
        weekly_slice = aligned_slice(weekly_c, candle["t"], 15) if weekly_c else []

        _param_overrides: Dict = {}
        _allowed_sides: Optional[List[str]] = None
        if regime == "PARABOLIC":
            _param_overrides = {"min_score": max(0.35, coin_p["score_threshold"] - 0.05)}
            _allowed_sides = ["LONG"]

        sig = _compute_signal_layers(
            klines, MODE, 1.0, higher_slice, STYLE, None,
            param_overrides=_param_overrides if _param_overrides else None,
            symbol=SYMBOL, enable_candle_pat=False,
            candle_ts_ms=candle["t"],
            weekly_klines=weekly_slice,
            enable_weekly_gate=ENABLE_WEEKLY_GATE,
            enable_mr_4h_gate=ENABLE_MR_4H_GATE,
        )

        side = sig.get("side", "LONG")

        if not sig.get("ok"):
            if in_diag:
                desired_dir[side] = desired_dir.get(side, 0) + 1
                breakdown = sig.get("breakdown", {})
                failed = [k for k, v in breakdown.items()
                          if not v.get("ok") and k not in ("mtf_15m", "weights")]
                layer = failed[0] if failed else "score"
                blocked_all[layer]  = blocked_all.get(layer, 0) + 1
                if side == "SHORT":
                    blocked_short[layer] = blocked_short.get(layer, 0) + 1
            if sig_pending:
                sig_pending = False; pending_side = ""; candles_since = 0
            continue

        # Signal OK — apply pullback gate and volume check (mirrors simulate())
        if _allowed_sides and side not in _allowed_sides:
            if in_diag:
                desired_dir[side] = desired_dir.get(side, 0) + 1
            continue

        signal_type = sig.get("signal_type", "NORMAL")
        if signal_type == "NORMAL":
            _vol_ok, _, _ = check_volume_hour_confirmed(klines, coin_p)
            if not _vol_ok:
                if in_diag:
                    desired_dir[side] = desired_dir.get(side, 0) + 1
                    blocked_all["volume"] = blocked_all.get("volume", 0) + 1
                    if side == "SHORT":
                        blocked_short["volume"] = blocked_short.get("volume", 0) + 1
                continue

            if not sig_pending:
                sig_pending = True; pending_side = side; candles_since = 0
            elif pending_side != side:
                pending_side = side; candles_since = 0
            candles_since += 1

            _pb_res = check_pullback_entry(
                klines, pending_side, coin_p, regime,
                candles_since, trade_style=STYLE, enable_fix3=ENABLE_FIX3,
                strong_trend_override=sig.get("strong_trend_override", False),
            )
            if _pb_res == "TIMEOUT":
                sig_pending = False; pending_side = ""; candles_since = 0
                continue
            if _pb_res == "WAIT":
                continue
            sig_pending = False; pending_side = ""; candles_since = 0

        if in_diag:
            desired_dir[side] = desired_dir.get(side, 0) + 1
            fired_dir[side] = fired_dir.get(side, 0) + 1

        # Open a trade (simplified — just need to block future entries)
        atr_pct = sig.get("atr_pct", 0.009)
        sl_pct  = min(atr_pct * ST["sl_atr"], ST["sl_max"] / 100)
        tp_pct  = min(max(sl_pct * 1.5, atr_pct * ST["tp_atr"]), ST["tp_max"] / 100)
        eff_size = c["size"]
        open_trade = {
            "side": side, "entry": candle["close"],
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "leverage": c["leverage"], "effective_size": eff_size,
            "equity_at_open": 1000.0, "signal_type": signal_type,
        }

    return {
        "adx_buckets": adx_buckets,
        "month_adx": dict(month_adx),
        "desired_dir": desired_dir,
        "fired_dir": fired_dir,
        "blocked_all": blocked_all,
        "blocked_short": blocked_short,
    }


def print_diagnostic(diag: Dict, sim_year_trades: Dict, diag_year: int = 2022,
                     year_start_equity: Dict = None):
    """Print the SHORT signal diagnostic report for a specific year."""
    SEP  = "=" * 68
    SEP2 = "-" * 68

    print(f"\n{SEP}")
    print(f"  SHORT SIGNAL DIAGNOSTIC — {diag_year}")
    print(f"  Current baseline config (all fixes + T18 + choppy trail)")
    print(SEP)

    # ── Per-direction trade count and P&L from simulation ─────────────────────
    tlist = sim_year_trades.get(diag_year, [])
    long_trades  = [t for t in tlist if t.get("side") == "LONG"]
    short_trades = [t for t in tlist if t.get("side") == "SHORT"]

    def dir_pnl(tl):
        wins = sum(1 for t in tl if t["outcome"] in ("TP_HIT", "TRAIL_STOP"))
        wr   = wins / len(tl) * 100 if tl else 0
        pnl  = sum(t["pnl"] for t in tl)
        return len(tl), wr, pnl

    ln, lwr, lpnl = dir_pnl(long_trades)
    sn, swr, spnl = dir_pnl(short_trades)

    print(f"\n  TRADES FIRED IN {diag_year}")
    print(SEP2)
    print(f"  {'Direction':<12} {'Count':>6} {'Win Rate':>9} {'P&L':>10}")
    print(SEP2)
    print(f"  {'LONG':<12} {ln:>6} {lwr:>8.1f}% {lpnl:>+9.2f}")
    print(f"  {'SHORT':<12} {sn:>6} {swr:>8.1f}% {spnl:>+9.2f}")
    print(f"  {'TOTAL':<12} {ln+sn:>6} {'':>9} {lpnl+spnl:>+9.2f}")

    if sn == 0:
        print(f"\n  ⚠  ZERO SHORT trades fired in {diag_year}.")
        print(f"     ADX floor theory: CONFIRMED. Engine went silent in the bear market.")

    # ── Month by month P&L ────────────────────────────────────────────────────
    from collections import defaultdict as _dd
    m_pnl:   dict = _dd(float)
    m_count: dict = _dd(int)
    m_wins:  dict = _dd(int)
    for t in tlist:
        mo = t.get("month", "unknown")
        m_pnl[mo]   += t["pnl"]
        m_count[mo] += 1
        if t["outcome"] in ("TP_HIT", "TRAIL_STOP"):
            m_wins[mo] += 1
    print(f"\n  MONTH BY MONTH P&L — {diag_year}")
    print(SEP2)
    print(f"  {'Month':<10} {'Trades':>7} {'Win Rate':>9} {'P&L':>10}")
    print(SEP2)
    for mo in sorted(m_pnl.keys()):
        cnt = m_count[mo]
        wr_m = m_wins[mo] / cnt * 100 if cnt else 0
        print(f"  {mo:<10} {cnt:>7} {wr_m:>8.1f}% {m_pnl[mo]:>+9.2f}")
    print(SEP2)
    total_mo_pnl = sum(m_pnl.values())
    print(f"  {'TOTAL':<10} {ln+sn:>7} {'':>9} {total_mo_pnl:>+9.2f}")

    # ── WORST 10 TRADES ───────────────────────────────────────────────────────────
    worst10 = sorted(tlist, key=lambda t: t["pnl"])[:10]
    print(f"\n  WORST 10 TRADES — {diag_year}")
    print(SEP2)
    print(f"  {'Date':<12} {'Side':<6} {'SigType':<16} {'4h Bull':>7} {'ADX':>6} {'RSI':>5} {'EMA50%':>7} {'P&L':>9} {'Out'}")
    print(SEP2)
    for t in worst10:
        htf = "YES" if t.get("htf_bullish") else "NO "
        sign = "+" if t["pnl"] >= 0 else ""
        print(f"  {t.get('open_date','?'):<12} {t.get('side','?'):<6} {t.get('signal_type','?'):<16} "
              f"{htf:>7} {t.get('adx_at_open',0):>6.1f} {t.get('rsi_at_open',0):>5.1f} "
              f"{t.get('ema50_dist',0):>+7.2f}% {sign}{t['pnl']:>8.0f}  {t['outcome']}")

    # ── Q1: Year-by-year return as % of starting equity (compounding proof) ─────
    all_yr_trades = []
    for yr_t in sim_year_trades.values():
        all_yr_trades.extend(yr_t)
    yse = year_start_equity or {}
    print(f"\n  Q1 — YEAR RETURNS vs STARTING EQUITY (compounding confirmed?)")
    print(SEP2)
    print(f"  {'Year':<6} {'Eq @ Jan 1':>11} {'P&L':>10} {'% of start':>11} {'Eq @ Dec 31':>12}")
    print(SEP2)
    _running = 1000.0
    for yr in sorted(sim_year_trades.keys()):
        yr_tl = sim_year_trades[yr]
        yr_pnl = sum(t["pnl"] for t in yr_tl)
        eq_start = yse.get(yr, _running)
        pct = yr_pnl / eq_start * 100 if eq_start else 0
        _running = eq_start + yr_pnl
        sign = "+" if yr_pnl >= 0 else ""
        print(f"  {yr:<6} {eq_start:>11.0f} {sign}{yr_pnl:>9.0f} {sign}{pct:>10.1f}% {_running:>12.0f}")

    # ── Q3: Full ADX distribution of trades (including < 20) ──────────────────
    print(f"\n  Q3 — FULL ADX DISTRIBUTION OF ALL TRADES ({diag_year}, incl. below-floor entries)")
    print(SEP2)
    FULL_BUCKETS = [
        ("<20",  0, 20), ("20-25", 20, 25), ("25-30", 25, 30),
        ("30-35", 30, 35), ("35-40", 35, 40), ("40+", 40, 999),
    ]
    print(f"  {'ADX':<10} {'7yr Trades':>11} {'7yr WR':>8}  {'2026 Trades':>12} {'2026 WR':>9} {'2026 P&L':>10}")
    print(SEP2)
    def _bk2(tl, lo, hi):
        s = [t for t in tl if lo <= t.get("adx_at_open", 0) < hi]
        if not s: return 0, 0.0, 0.0
        w = sum(1 for t in s if t["outcome"] in ("TP_HIT","TRAIL_STOP"))
        return len(s), w/len(s)*100, sum(t["pnl"] for t in s)
    tlist = sim_year_trades.get(diag_year, [])
    for lbl, lo, hi in FULL_BUCKETS:
        n7, wr7, _ = _bk2(all_yr_trades, lo, hi)
        n26, wr26, pnl26 = _bk2(tlist, lo, hi)
        sign = "+" if pnl26 >= 0 else ""
        print(f"  {lbl:<10} {n7:>11} {wr7:>7.1f}%  {n26:>12} {wr26:>8.1f}% {sign}{pnl26:>9.0f}")
    unaccounted = len(tlist) - sum(_bk2(tlist, lo, hi)[0] for _, lo, hi in FULL_BUCKETS)
    if unaccounted:
        print(f"  {'???':<10} {'':>11} {'':>8}  {unaccounted:>12}  (adx_at_open not set)")

    # ── Q2: The 13 SHORTs at ADX 35-40 — exact trade log ─────────────────────
    shorts_3540 = [t for t in tlist if t.get("side") == "SHORT"
                   and 35 <= t.get("adx_at_open", 0) < 40]
    print(f"\n  Q2 — SHORT TRADES AT ADX 35-40 IN {diag_year} ({len(shorts_3540)} trades)")
    print(SEP2)
    print(f"  {'Date':<12} {'SigType':<16} {'4h Bull':>7} {'+DI':>6} {'-DI':>6} {'ADX':>6} {'P&L':>9} {'Out'}")
    print(SEP2)
    for t in sorted(shorts_3540, key=lambda x: x.get("open_date", "")):
        htf = "YES" if t.get("htf_bullish") else "NO "
        sign = "+" if t["pnl"] >= 0 else ""
        print(f"  {t.get('open_date','?'):<12} {t.get('signal_type','?'):<16} "
              f"{htf:>7} {t.get('plus_di',0):>6.1f} {t.get('minus_di',0):>6.1f} "
              f"{t.get('adx_at_open',0):>6.1f} {sign}{t['pnl']:>8.0f}  {t['outcome']}")
    if not shorts_3540:
        print(f"  None found — may be in a different ADX range")

    # ── Q4: ADX bucket WR across ALL 7 years (is pattern universal?) ─────────
    print(f"\n  Q4 — ADX BUCKET WIN RATE ACROSS ALL YEARS (SHORT only)")
    print(SEP2)
    FINE = [("20-25",20,25),("25-30",25,30),("30-35",30,35),("35-40",35,40),("40+",40,999)]
    hdr = f"  {'ADX':<10}"
    for yr in sorted(sim_year_trades.keys()): hdr += f" {yr:>8}"
    print(hdr)
    print(f"  {'-'*10}", end="")
    for _ in sim_year_trades: print(f" {'-'*8}", end="")
    print()
    for lbl, lo, hi in FINE:
        row = f"  {lbl:<10}"
        for yr in sorted(sim_year_trades.keys()):
            yr_shorts = [t for t in sim_year_trades[yr] if t.get("side") == "SHORT"]
            n, wr, _ = _bk2(yr_shorts, lo, hi)
            row += f"  {wr:>5.0f}%" if n >= 3 else f"  {'—':>6}"
        print(row)
    print(f"\n  (— = fewer than 3 trades in bucket that year)")

    # ── Desired direction (what the HTF engine wanted to trade) ───────────────
    des = diag["desired_dir"]
    fired = diag["fired_dir"]
    total_des = sum(des.values())
    print(f"\n  HTF DESIRED DIRECTION (candles where engine checked for entry)")
    print(SEP2)
    print(f"  {'Direction':<12} {'Desired':>8} {'Fired':>7} {'% Fired':>9}")
    print(SEP2)
    for d in ("LONG", "SHORT"):
        n   = des.get(d, 0)
        f   = fired.get(d, 0)
        pct = f / n * 100 if n else 0
        print(f"  {d:<12} {n:>8} {f:>7} {pct:>8.1f}%")

    # ── Block reason breakdown ─────────────────────────────────────────────────
    blocked = diag["blocked_all"]
    blocked_sh = diag["blocked_short"]
    total_blocked = sum(blocked.values())

    layer_labels = {
        "regime":    "Layer 1 — ADX/ATR floor (regime)",
        "direction": "Layer 2 — HTF direction score",
        "entry":     "Layer 3 — pullback / RSI / pattern",
        "momentum":  "Layer 4 — momentum score",
        "score":     "Final score below threshold",
        "volume":    "Volume hour not confirmed",
    }
    print(f"\n  BLOCK REASON BREAKDOWN (all directions vs SHORT-only)")
    print(SEP2)
    print(f"  {'Layer':<38} {'All':>6} {'%':>6}  {'SHORT':>6} {'%':>6}")
    print(SEP2)
    for layer in ("regime", "direction", "entry", "momentum", "score", "volume"):
        n_all = blocked.get(layer, 0)
        n_sh  = blocked_sh.get(layer, 0)
        if n_all == 0:
            continue
        pct_all = n_all / total_blocked * 100 if total_blocked else 0
        pct_sh  = n_sh  / n_all * 100 if n_all else 0
        lbl = layer_labels.get(layer, layer)
        print(f"  {lbl:<38} {n_all:>6} {pct_all:>5.1f}%  {n_sh:>6} {pct_sh:>5.1f}%")

    # ── ADX DISTRIBUTION (candles) ────────────────────────────────────────────
    adx = diag["adx_buckets"]
    total_candles = sum(adx.values())
    print(f"\n  ADX DISTRIBUTION — ALL {diag_year} CANDLES  ({total_candles:,} total)")
    print(SEP2)
    print(f"  {'ADX Range':<14} {'Candles':>8} {'%':>7}  {'Note'}")
    print(SEP2)
    notes = {
        "<14":   "RANGING/CHOPPY — regime blocks completely",
        "14-20": "below MINI_ASYM floor — no signals",
        "20-25": "lower moderate trend — signals fire",
        "25-30": "upper moderate trend — signals fire",
        "30-35": "strong trend",
        "35-40": "very strong trend",
        "40+":   "parabolic / extreme trend",
    }
    for bk in ("<14", "14-20", "20-25", "25-30", "30-35", "35-40", "40+"):
        n   = adx.get(bk, 0)
        pct = n / total_candles * 100 if total_candles else 0
        print(f"  {bk:<14} {n:>8} {pct:>6.1f}%  {notes[bk]}")

    # ── TABLE 1: ADX bucket trade analysis (all years + diag_year) ────────────
    all_trades: List[Dict] = []
    for yr_trades in sim_year_trades.values():
        all_trades.extend(yr_trades)

    FINE_BUCKETS = [
        ("20-25", 20, 25), ("25-30", 25, 30),
        ("30-35", 30, 35), ("35-40", 35, 40), ("40+", 40, 999),
    ]

    def _bk_stats(tlist_in, lo, hi):
        sub = [t for t in tlist_in if lo <= t.get("adx_at_open", 0) < hi]
        if not sub:
            return 0, 0.0, 0.0
        wins = sum(1 for t in sub if t["outcome"] in ("TP_HIT", "TRAIL_STOP"))
        return len(sub), wins / len(sub) * 100, sum(t["pnl"] for t in sub)

    print(f"\n  TABLE 1 — ADX BUCKET TRADE STATS (ALL 7 YEARS, BTCUSDT)")
    print(SEP2)
    print(f"  {'ADX Range':<10} {'Trades':>7} {'WR %':>7} {'Total P&L':>11}")
    print(SEP2)
    for label, lo, hi in FINE_BUCKETS:
        n, wr_b, pnl_b = _bk_stats(all_trades, lo, hi)
        sign = "+" if pnl_b >= 0 else ""
        print(f"  {label:<10} {n:>7} {wr_b:>6.1f}% {sign}{pnl_b:>10.2f}")

    print(f"\n  TABLE 1b — ADX BUCKET TRADE STATS ({diag_year} ONLY)")
    print(SEP2)
    print(f"  {'ADX Range':<10} {'Trades':>7} {'WR %':>7} {'P&L':>11}  {'% of yr trades':>15}")
    print(SEP2)
    total_yr_trades = len(tlist) or 1
    for label, lo, hi in FINE_BUCKETS:
        n, wr_b, pnl_b = _bk_stats(tlist, lo, hi)
        pct_of_yr = n / total_yr_trades * 100
        sign = "+" if pnl_b >= 0 else ""
        print(f"  {label:<10} {n:>7} {wr_b:>6.1f}% {sign}{pnl_b:>10.2f}  {pct_of_yr:>14.1f}%")
    sub_20_30 = sum(_bk_stats(tlist, lo, hi)[0] for _, lo, hi in FINE_BUCKETS[:2])
    pct_20_30 = sub_20_30 / total_yr_trades * 100
    print(f"\n  ADX 20-30 trades = {sub_20_30}/{total_yr_trades} ({pct_20_30:.1f}% of {diag_year} trades)")

    # ── TABLE 2: LONG vs SHORT per ADX bucket (diag_year) ────────────────────
    print(f"\n  TABLE 2 — DIRECTION SPLIT PER ADX BUCKET ({diag_year})")
    print(SEP2)
    print(f"  {'ADX':<10} {'LONG':>5} {'L-WR':>7} {'L-P&L':>9}  {'SHORT':>5} {'S-WR':>7} {'S-P&L':>9}")
    print(SEP2)
    for label, lo, hi in FINE_BUCKETS:
        ln2, lwr2, lpnl2 = _bk_stats([t for t in tlist if t.get("side") == "LONG"],  lo, hi)
        sn2, swr2, spnl2 = _bk_stats([t for t in tlist if t.get("side") == "SHORT"], lo, hi)
        lsign = "+" if lpnl2 >= 0 else ""
        ssign = "+" if spnl2 >= 0 else ""
        print(f"  {label:<10} {ln2:>5} {lwr2:>6.1f}% {lsign}{lpnl2:>8.0f}  "
              f"{sn2:>5} {swr2:>6.1f}% {ssign}{spnl2:>8.0f}")

    # ── TABLE 3: Month-by-month with ADX candle % ────────────────────────────
    month_adx = diag.get("month_adx", {})
    print(f"\n  TABLE 3 — MONTH × ADX CANDLE DISTRIBUTION ({diag_year})")
    print(SEP2)
    print(f"  {'Month':<9} {'Trd':>4} {'WR':>6}  {'P&L':>8}  {'20-25':>6} {'25-30':>6} {'30-35':>6} {'35+':>5}  {'Chop%':>6}")
    print(SEP2)
    from collections import defaultdict as _dd3
    m_side: dict = _dd3(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in tlist:
        mo = t.get("month", "unknown")
        m_side[mo]["trades"] += 1
        m_side[mo]["pnl"]    += t["pnl"]
        if t["outcome"] in ("TP_HIT", "TRAIL_STOP"):
            m_side[mo]["wins"] += 1
    for mo in sorted(month_adx.keys()):
        md   = month_adx[mo]
        tot  = md.get("total", 1) or 1
        t20  = md.get("20-25", 0); t25 = md.get("25-30", 0)
        t30  = md.get("30-35", 0); t35 = md.get("35-40", 0) + md.get("40+", 0)
        chop_pct = (t20 + t25) / tot * 100
        mtr = m_side[mo]
        cnt = mtr["trades"]; wr_m = mtr["wins"] / cnt * 100 if cnt else 0
        pnl_m = mtr["pnl"]
        sign = "+" if pnl_m >= 0 else ""
        flag = " <-- CHOP" if chop_pct >= 60 else (" <-- TREND" if (t30 + t35) / tot * 100 >= 50 else "")
        print(f"  {mo:<9} {cnt:>4} {wr_m:>5.1f}%  {sign}{pnl_m:>7.0f}  "
              f"{t20/tot*100:>5.1f}% {t25/tot*100:>5.1f}% {t30/tot*100:>5.1f}% {t35/tot*100:>4.1f}%  "
              f"{chop_pct:>5.1f}%{flag}")

    # ── Q5: LONG winners vs losers comparison ────────────────────────────────────
    long_wins  = [t for t in tlist if t.get("side") == "LONG"
                  and t["outcome"] in ("TP_HIT", "TRAIL_STOP")]
    long_loses = [t for t in tlist if t.get("side") == "LONG"
                  and t["outcome"] not in ("TP_HIT", "TRAIL_STOP")]
    if long_wins or long_loses:
        def _avg(lst, field):
            return sum(t.get(field, 0) for t in lst) / len(lst) if lst else 0.0
        def _pct_true(lst, field):
            return sum(1 for t in lst if t.get(field)) / len(lst) * 100 if lst else 0.0

        nw, nl = len(long_wins), len(long_loses)
        print(f"\n  Q5 — {diag_year} LONG WINNERS ({nw}) vs LOSERS ({nl})")
        print(SEP2)
        print(f"  {'Metric':<26} {'Winners':>12} {'Losers':>12}  {'Gap':>8}")
        print(SEP2)

        lw_4h = _pct_true(long_wins, "htf_bullish")
        ll_4h = _pct_true(long_loses, "htf_bullish")
        print(f"  {'4h Bullish':<26} {lw_4h:>11.1f}% {ll_4h:>11.1f}%  {lw_4h-ll_4h:>+7.1f}pp")

        lw_db = _pct_true(long_wins, "daily_bullish")
        ll_db = _pct_true(long_loses, "daily_bullish")
        print(f"  {'Daily Bias Bullish (1d)':<26} {lw_db:>11.1f}% {ll_db:>11.1f}%  {lw_db-ll_db:>+7.1f}pp")

        lw_ema = _avg(long_wins, "ema50_dist")
        ll_ema = _avg(long_loses, "ema50_dist")
        print(f"  {'EMA50 Dist avg':<26} {lw_ema:>+11.2f}% {ll_ema:>+11.2f}%  {lw_ema-ll_ema:>+7.2f}pp")

        lw_rsi = _avg(long_wins, "rsi_at_open")
        ll_rsi = _avg(long_loses, "rsi_at_open")
        print(f"  {'RSI at entry avg':<26} {lw_rsi:>12.1f} {ll_rsi:>12.1f}  {lw_rsi-ll_rsi:>+7.1f}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    adx_gap_pct = (adx.get("14-20", 0) + adx.get("<14", 0)) / total_candles * 100 if total_candles else 0
    regime_block_pct = blocked.get("regime", 0) / total_blocked * 100 if total_blocked else 0
    print(f"\n  CONCLUSION")
    print(SEP2)
    print(f"  ADX below 20 in {adx_gap_pct:.1f}% of {diag_year} candles")
    print(f"  Regime (ADX floor) was {regime_block_pct:.1f}% of all block reasons")
    short_regime_pct = blocked_sh.get("regime", 0) / sum(blocked_sh.values()) * 100 if blocked_sh else 0
    print(f"  For SHORT-desired candles: regime blocked {short_regime_pct:.1f}% of attempts")
    if sn == 0 and regime_block_pct > 50:
        print(f"\n  ✓ ADX floor theory CONFIRMED:")
        print(f"    Zero SHORT trades fired. ADX below 20 dominates {diag_year}.")
        print(f"    Fix A (direction-aware ADX floor: SHORT uses 16, LONG stays 20)")
        print(f"    is the correct next step. Backtest before deploy.")
    elif sn > 0 and swr < 10:
        print(f"\n  ✗ SHORT signals fired but quality was poor ({sn} trades, {swr:.1f}% WR).")
        print(f"    Problem is signal quality, not ADX floor. Different fix needed.")
    else:
        print(f"\n  More investigation needed. Check numbers above.")
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global ENABLE_TRAIL, ENABLE_CANDLE_PAT, ENABLE_WEEKLY_GATE, ENABLE_FIX2
    global ENABLE_MR_4H_GATE, ENABLE_PAT_EXEC, ENABLE_T18, T18_MR_EXEMPT
    global ENABLE_CHOPPY_TRAIL, ENABLE_HTF_GATE_NORMAL, ENABLE_OE_GATE, ENABLE_HARD_FLOOR

    SEP  = "=" * 68
    SEP2 = "-" * 68

    print("\n" + SEP)
    print("  Asymmetric AI — Backtest: BTCUSDT 1h  Jan 2020 → Oct 2026")
    print("  Mode: MINI_ASYM | Style: DAY_TRADE | Start equity: $1,000")
    print(SEP)

    print("\nStep 1 — Fetching candles:")
    main_c, higher_c, weekly_c = load_candles()
    total_candles = len(main_c) - WARMUP
    years = (END_MS - START_MS) / (1000 * 60 * 60 * 24 * 365.25)
    print(f"  Data range: {total_candles:,} active candles | {years:.1f} years")

    # ════════════════════════════════════════════════════════════════════
    # TASK 1 — LOOK-AHEAD AUDIT
    # ════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  TASK 1 — LOOK-AHEAD AUDIT (1h / 4h / daily / weekly)")
    print(SEP)
    print("  FIXED this session:")
    print("    backtester : higher_slice[-2] for _htf_bull (MR + NORMAL, completed candle)")
    print("    indicators : get_4h_trend_direction → candles_4h[:-1] (excludes open candle)")
    print("    indicators : _compute_signal_layers HTF EMA → higher_klines[:-1]")
    print("    indicators : _compute_signal_layers weekly EMA → weekly_klines[:-1][-15:]")
    print("    indicators : _market_structure → higher_klines[:-1] (excludes open candle)")
    print("  REMAINING (real-time data in live trading, not look-ahead):")
    print("    _daily_bias (live): uses current daily close = real-time price, not future close")
    print("      Backtester does NOT pass daily_klines → daily bias = NEUTRAL → no bias applied")
    print("    check_weekly_trend (SWING only): backtester style = DAY_TRADE → unaffected")
    print("  BACKTESTER ↔ LIVE DIVERGENCES:")
    print("    Daily bias: backtest=NEUTRAL | live=±0.05 score adj per trade")
    print("    Severity: LOW — daily EMA21 changes slowly, ±0.05 rarely flips a signal")

    # ════════════════════════════════════════════════════════════════════
    # TASK 3 — FEE + SLIPPAGE CONFIRMATION
    # ════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  TASK 3 — FEE + SLIPPAGE CONFIRMATION")
    print(SEP)
    rt_sl = FEE["taker"] + FEE["maker"] + FEE["slip"] * 2   # entry taker + exit taker + 2× slip
    rt_tp = FEE["taker"] + FEE["maker"] + FEE["slip"] * 2   # entry taker + exit maker + 2× slip
    print(f"  Entry  : taker {FEE['taker']*100:.4f}% + slip {FEE['slip']*100:.4f}%")
    print(f"  SL exit: taker {FEE['taker']*100:.4f}% + slip {FEE['slip']*100:.4f}%")
    print(f"  TP exit: maker {FEE['maker']*100:.4f}% + slip {FEE['slip']*100:.4f}%")
    print(f"  Full round-trip (SL): {(FEE['taker']*2+FEE['maker']+FEE['slip']*2)*100:.4f}%")
    print(f"  Full round-trip (TP): {(FEE['taker']+FEE['maker']*2+FEE['slip']*2)*100:.4f}%")
    print(f"  Funding rate: NOT included — Binance USDM perps charge ~0.01%/8h")
    print(f"    Estimated drag per trade (4-12h avg hold): ~0.005–0.015%")
    print(f"    Annual drag at 433 trades avg hold ~6h: ~$3–9 on $1,000 equity")

    # ════════════════════════════════════════════════════════════════════
    # TASK 2 — FLOOR ON vs OFF
    # Isolates the hard floor's effect from signal quality.
    # Config: all deployed fixes ON, floor toggled.
    # ════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  TASK 2 — HARD FLOOR: ON vs OFF")
    print("  Config both: all deployed fixes + corrected look-ahead")
    print(SEP)

    ENABLE_MR_4H_GATE      = True
    ENABLE_PAT_EXEC        = True
    ENABLE_TRAIL           = True
    ENABLE_T18             = True
    T18_MR_EXEMPT          = True
    ENABLE_CHOPPY_TRAIL    = True
    ENABLE_HTF_GATE_NORMAL = True
    ENABLE_OE_GATE         = False

    ENABLE_HARD_FLOOR = False
    print(f"\nFloor OFF — all deployed fixes, no floor stop...")
    t0 = time.time()
    r_nof = simulate(main_c, higher_c, weekly_c)
    t_nof = time.time() - t0
    s_nof = compute_stats(r_nof)
    yr_nof = year_summary(r_nof["year_trades"])

    ENABLE_HARD_FLOOR = True
    print(f"\nFloor ON  — same + 15% hard floor with year restart...")
    t0 = time.time()
    r_flr = simulate(main_c, higher_c, weekly_c)
    t_flr = time.time() - t0
    s_flr = compute_stats(r_flr)
    yr_flr = year_summary(r_flr["year_trades"])

    print_ab("All fixes, floor OFF", "All fixes, floor ON",
             s_nof, s_flr, t_nof, t_flr, yr_nof, yr_flr, r_flr)

    # ════════════════════════════════════════════════════════════════════
    # TASK 4 — FIX-BY-FIX RETESTING ON CORRECTED RAW BASELINE
    # Raw baseline: look-ahead fixed, DD tiers, HARD FLOOR ON, no other fixes.
    # Each fix tested INDEPENDENTLY (one at a time) vs raw.
    # ════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  TASK 4 — FIX-BY-FIX RETESTING (each fix alone vs raw corrected)")
    print("  Raw corrected: look-ahead fixed + DD tiers + floor ON, NO other fixes")
    print(SEP)

    # Set all signal fixes OFF, floor ON
    ENABLE_TRAIL           = False
    ENABLE_MR_4H_GATE      = False
    ENABLE_PAT_EXEC        = False
    ENABLE_T18             = False
    T18_MR_EXEMPT          = False
    ENABLE_CHOPPY_TRAIL    = False
    ENABLE_HTF_GATE_NORMAL = False
    ENABLE_OE_GATE         = False
    ENABLE_HARD_FLOOR      = True

    print(f"\nRaw corrected baseline (all fixes OFF, floor ON)...")
    t0      = time.time()
    r_raw   = simulate(main_c, higher_c, weekly_c)
    t_raw   = time.time() - t0
    s_raw   = compute_stats(r_raw)
    yr_raw  = year_summary(r_raw["year_trades"])
    print(f"\r  Raw baseline: {len(r_raw['trades'])} trades | Sharpe {s_raw['sharpe']:.2f} | "
          f"WR {s_raw['wr']:.1f}% | Return {s_raw['return_pct']:+.1f}%        ")

    FIXES = [
        ("Trailing stop        (ENABLE_TRAIL)",        {"ENABLE_TRAIL": True}),
        ("MR 4h gate           (ENABLE_MR_4H_GATE)",  {"ENABLE_MR_4H_GATE": True}),
        ("Pattern exec         (ENABLE_PAT_EXEC)",     {"ENABLE_PAT_EXEC": True}),
        ("T18 model A@70%      (ENABLE_T18)",          {"ENABLE_T18": True}),
        ("T18 MR exempt        (T18_MR_EXEMPT)",       {"ENABLE_T18": True, "T18_MR_EXEMPT": True}),
        ("Choppy trail delay   (ENABLE_CHOPPY_TRAIL)", {"ENABLE_TRAIL": True, "ENABLE_CHOPPY_TRAIL": True}),
        ("HTF NORMAL gate      (ENABLE_HTF_GATE)",     {"ENABLE_HTF_GATE_NORMAL": True}),
    ]

    fix_results = []
    for fix_name, fix_flags in FIXES:
        # Reset all to OFF
        ENABLE_TRAIL = ENABLE_MR_4H_GATE = ENABLE_PAT_EXEC = False
        ENABLE_T18 = T18_MR_EXEMPT = ENABLE_CHOPPY_TRAIL = False
        ENABLE_HTF_GATE_NORMAL = ENABLE_OE_GATE = False
        # Apply this fix's flags
        for k, v in fix_flags.items():
            if k == "ENABLE_TRAIL":           ENABLE_TRAIL = v
            elif k == "ENABLE_MR_4H_GATE":    ENABLE_MR_4H_GATE = v
            elif k == "ENABLE_PAT_EXEC":      ENABLE_PAT_EXEC = v
            elif k == "ENABLE_T18":           ENABLE_T18 = v
            elif k == "T18_MR_EXEMPT":        T18_MR_EXEMPT = v
            elif k == "ENABLE_CHOPPY_TRAIL":  ENABLE_CHOPPY_TRAIL = v
            elif k == "ENABLE_HTF_GATE_NORMAL": ENABLE_HTF_GATE_NORMAL = v
        print(f"\n  Testing: {fix_name}...")
        t0 = time.time()
        r_fix = simulate(main_c, higher_c, weekly_c)
        t_fx  = time.time() - t0
        s_fix = compute_stats(r_fix)
        yr_fix = year_summary(r_fix["year_trades"])
        delta_sharpe = s_fix["sharpe"] - s_raw["sharpe"]
        fix_results.append((fix_name, s_fix, yr_fix, delta_sharpe, r_fix))
        print(f"\r    {len(r_fix['trades'])} trades | Sharpe {s_fix['sharpe']:.2f} "
              f"({delta_sharpe:+.2f}) | WR {s_fix['wr']:.1f}% | Return {s_fix['return_pct']:+.1f}%   ")

    print(f"\n\n{'='*68}")
    print(f"  TASK 4 SUMMARY — FIX vs RAW CORRECTED BASELINE (Sharpe {s_raw['sharpe']:.2f})")
    print(f"{'='*68}")
    print(f"  {'Fix':<42} {'Sharpe':>7} {'Delta':>7} {'Trades':>7} {'WR%':>6} {'Ret%':>7}  {'Keep?'}")
    print(f"  {'-'*42} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7}  {'-'*5}")
    print(f"  {'Raw corrected (baseline)':<42} {s_raw['sharpe']:>7.2f} {'—':>7} "
          f"{s_raw['trades']:>7} {s_raw['wr']:>5.1f}% {s_raw['return_pct']:>+6.1f}%  base")
    kept_fixes = []
    for fix_name, s_fix, yr_fix, delta, _ in fix_results:
        keep = "YES" if delta > 0 else "NO "
        if delta > 0:
            kept_fixes.append(fix_name.strip())
        sign = "+" if delta >= 0 else ""
        print(f"  {fix_name:<42} {s_fix['sharpe']:>7.2f} {sign}{delta:>6.2f} "
              f"{s_fix['trades']:>7} {s_fix['wr']:>5.1f}% {s_fix['return_pct']:>+6.1f}%  {keep}")

    print(f"\n  Fixes that improve Sharpe: {len(kept_fixes)}")
    for f in kept_fixes:
        print(f"    + {f}")

    # Year-by-year detail for fixes that passed
    if kept_fixes:
        print(f"\n  YEAR-BY-YEAR for kept fixes (raw baseline reference):")
        print(f"  {'Year':<6}", end="")
        print(f"  {'RAW':>10}", end="")
        for fix_name, s_fix, yr_fix, delta, _ in fix_results:
            if delta > 0:
                lbl = fix_name.split("(")[0].strip()[:12]
                print(f"  {lbl:>12}", end="")
        print()
        all_years = sorted(set(yr_raw.keys()))
        for yr in all_years:
            raw_pnl = yr_raw.get(yr, {}).get("pnl", 0)
            print(f"  {yr:<6}", end="")
            sign = "+" if raw_pnl >= 0 else ""
            print(f"  {sign}{raw_pnl:>9.0f}", end="")
            for fix_name, s_fix, yr_fix, delta, _ in fix_results:
                if delta > 0:
                    fx_pnl = yr_fix.get(yr, {}).get("pnl", 0)
                    sign2 = "+" if fx_pnl >= 0 else ""
                    print(f"  {sign2}{fx_pnl:>11.0f}", end="")
            print()

    # ════════════════════════════════════════════════════════════════════
    # TASK 5 — IN-SAMPLE (2020-2023) vs OUT-OF-SAMPLE (2024-2026)
    # Uses the "all deployed fixes + floor ON" result (r_flr from Task 2)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  TASK 5 — IN-SAMPLE (2020-2023) vs OUT-OF-SAMPLE (2024-2026)")
    print("  Config: all deployed fixes + floor ON (r_flr from Task 2)")
    print(SEP)

    IS_YEARS  = [2020, 2021, 2022, 2023]
    OOS_YEARS = [2024, 2025, 2026]

    s_is  = compute_stats_years(r_flr, IS_YEARS)
    s_oos = compute_stats_years(r_flr, OOS_YEARS)

    print(f"\n  {'Metric':<22} {'In-Sample (2020-23)':>20} {'Out-of-Sample (2024-26)':>24}")
    print(f"  {'-'*22} {'-'*20} {'-'*24}")
    print(f"  {'Trades':<22} {s_is['trades']:>20} {s_oos['trades']:>24}")
    print(f"  {'Win Rate %':<22} {s_is['wr']:>20.1f} {s_oos['wr']:>24.1f}")
    print(f"  {'Sharpe':<22} {s_is['sharpe']:>20.2f} {s_oos['sharpe']:>24.2f}")
    print(f"  {'Return %':<22} {s_is['return_pct']:>+20.1f} {s_oos['return_pct']:>+24.1f}")
    print(f"  {'Max DD %':<22} {s_is['max_dd_pct']:>20.1f} {s_oos['max_dd_pct']:>24.1f}")
    print(f"  {'Start equity $':<22} {s_is['start_eq']:>20.0f} {s_oos['start_eq']:>24.0f}")
    print(f"  {'End equity $':<22} {s_is['end_eq']:>20.0f} {s_oos['end_eq']:>24.0f}")

    # Year-by-year for context
    print(f"\n  Year-by-year (all fixes + floor):")
    print(f"  {'Year':<6} {'Trades':>7} {'WR%':>6} {'P&L':>10}  {'Segment'}")
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*10}  {'-'*12}")
    for yr in sorted(r_flr["year_trades"].keys()):
        yt = year_summary(r_flr["year_trades"])[yr]
        seg = "in-sample" if yr in IS_YEARS else "OOS"
        sign = "+" if yt["pnl"] >= 0 else ""
        print(f"  {yr:<6} {yt['trades']:>7} {yt['wr']:>5.1f}% {sign}{yt['pnl']:>9.0f}  {seg}")

    oos_verdict = "PASS" if s_oos["sharpe"] >= s_is["sharpe"] * 0.5 else "FAIL"
    print(f"\n  OOS verdict: {oos_verdict}  "
          f"(OOS Sharpe {s_oos['sharpe']:.2f} vs IS {s_is['sharpe']:.2f}; "
          f"threshold = IS × 0.5 = {s_is['sharpe']*0.5:.2f})")

    # ════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC — 2026 + 2020 (uses all-fixes + floor config)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  DIAGNOSTIC — 2026 + 2020 (all fixes + floor)")
    print(SEP)

    ENABLE_MR_4H_GATE      = True
    ENABLE_PAT_EXEC        = True
    ENABLE_TRAIL           = True
    ENABLE_T18             = True
    T18_MR_EXEMPT          = True
    ENABLE_CHOPPY_TRAIL    = True
    ENABLE_HTF_GATE_NORMAL = True
    ENABLE_OE_GATE         = False
    ENABLE_HARD_FLOOR      = True

    print(f"\nSignal diagnostic pass — 2026...")
    t0      = time.time()
    diag_26 = run_signal_diagnostic(main_c, higher_c, weekly_c, diag_year=2026)
    t_d2    = time.time() - t0
    print(f"\r  2026 diagnostic done ({t_d2:.1f}s)                               ")

    print(f"Signal diagnostic pass — 2020...")
    t0      = time.time()
    diag_20 = run_signal_diagnostic(main_c, higher_c, weekly_c, diag_year=2020)
    t_d3    = time.time() - t0
    print(f"\r  2020 diagnostic done ({t_d3:.1f}s)                               ")

    print_diagnostic(diag_26, r_flr["year_trades"], diag_year=2026,
                     year_start_equity=r_flr.get("year_start_equity", {}))
    print_diagnostic(diag_20, r_flr["year_trades"], diag_year=2020,
                     year_start_equity=r_flr.get("year_start_equity", {}))


if __name__ == "__main__":
    main()
