"""
run_direction_fix_test.py — Direction fix validation (BTCUSDT, 1h, 2024-2025)

Compares current engine against baseline:
  Baseline: Sharpe 0.36 | WR 34.2% | 281 trades | MINI_ASYM DAY_TRADE

Run: python3 run_direction_fix_test.py
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

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL     = "BTCUSDT"
MODE       = "MINI_ASYM"
STYLE      = "DAY_TRADE"
TF         = "1h"
HIGHER_TF  = "4h"
WARMUP     = 240
START_MS   = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS     = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

BASELINE = {"sharpe": 0.36, "wr_pct": 34.2, "trades": 281}

MODE_PRESETS = {
    "MINI_ASYM": {"size": 0.65, "leverage": 6},
}
FEE = {"maker": 0.00020, "taker": 0.00055, "slip": 0.00025}
ST  = TRADE_STYLE_PARAMS["DAY_TRADE"]
CACHE_FILE = os.path.join(os.path.dirname(__file__), "direction_fix_cache.json")


# ── Candle fetch (Binance 1000/page) ─────────────────────────────────────────
BN_TF_MAP = {"1h": "1h", "4h": "4h"}


def fetch_binance(symbol: str, tf: str) -> List[Dict]:
    url     = "https://api.binance.com/api/v3/klines"
    candles, current, pages = [], START_MS, 0
    print(f"  Fetching {symbol} {tf}...", end="", flush=True)
    while current < END_MS:
        try:
            r = httpx.get(url, params={
                "symbol": symbol, "interval": BN_TF_MAP[tf],
                "startTime": current, "endTime": END_MS, "limit": 1000,
            }, timeout=20)
            if r.status_code != 200:
                time.sleep(2); continue
            data = r.json()
            if not data: break
            for k in data:
                candles.append({
                    "t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                })
            current = int(data[-1][0]) + 1
            pages += 1
            if len(data) < 1000: break
            time.sleep(0.12)
            if pages % 5 == 0: print(".", end="", flush=True)
        except Exception as e:
            print(f" [retry:{e}]", end="", flush=True); time.sleep(3)
    print(f" {len(candles):,} candles")
    return candles


def load_candles() -> tuple:
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print("  Using cached candles")
        except Exception:
            cache = {}

    main_key   = f"{SYMBOL}_{TF}"
    higher_key = f"{SYMBOL}_{HIGHER_TF}"
    changed    = False

    if main_key not in cache:
        cache[main_key] = fetch_binance(SYMBOL, TF)
        changed = True
    else:
        print(f"  {SYMBOL} {TF}: {len(cache[main_key]):,} candles (cached)")

    if higher_key not in cache:
        cache[higher_key] = fetch_binance(SYMBOL, HIGHER_TF)
        changed = True
    else:
        print(f"  {SYMBOL} {HIGHER_TF}: {len(cache[higher_key]):,} candles (cached)")

    if changed:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

    return cache[main_key], cache[higher_key]


# ── Helpers ───────────────────────────────────────────────────────────────────
def aligned_slice(candles: List[Dict], before_ts: int, n: int) -> List[Dict]:
    lo, hi, idx = 0, len(candles) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if candles[mid]["t"] <= before_ts: idx = mid; lo = mid + 1
        else: hi = mid - 1
    if idx < 0: return []
    return candles[max(0, idx - n + 1): idx + 1]


def trade_fee(size_usd: float, is_tp: bool) -> float:
    return size_usd * (FEE["maker"] + (FEE["maker"] if is_tp else FEE["taker"]) + FEE["slip"])


def check_exit(trade: Dict, candle: Dict) -> Optional[Dict]:
    side, entry = trade["side"], trade["entry"]
    sl, tp  = trade["sl_pct"], trade["tp_pct"]
    lev, eq = trade["leverage"], trade["equity_at_open"]
    hi, lo  = candle["high"], candle["low"]
    sl_p = entry * (1 - sl) if side == "LONG" else entry * (1 + sl)
    tp_p = entry * (1 + tp) if side == "LONG" else entry * (1 - tp)
    sl_hit = lo <= sl_p if side == "LONG" else hi >= sl_p
    tp_hit = hi >= tp_p if side == "LONG" else lo <= tp_p
    size   = trade["effective_size"] * eq

    if tp_hit and sl_hit:
        return {"outcome": "SL_HIT", "pnl": -size * sl * lev - trade_fee(size, False)}
    if tp_hit:
        t1   = size * 0.60
        pnl1 = t1 * tp * lev - trade_fee(t1, True)
        t2   = size * 0.40
        tp2  = min(tp * 2.0, ST["tp_max"] / 100)
        pnl2 = t2 * tp2 * lev - trade_fee(t2, True)
        return {"outcome": "TP_HIT", "pnl": pnl1 + pnl2}
    if sl_hit:
        return {"outcome": "SL_HIT", "pnl": -size * sl * lev - trade_fee(size, False)}
    return None


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(main_c: List[Dict], higher_c: List[Dict]) -> Dict:
    c         = MODE_PRESETS[MODE]
    mode_cfg  = get_mode_config(MODE)
    style_cfg = get_style_config(STYLE)
    coin_p    = get_coin_params(SYMBOL)

    equity       = 1000.0
    peak_equity  = 1000.0
    trades: List[Dict] = []
    pnls: List[float]  = []
    open_trade: Optional[Dict] = None
    is_parabolic  = False
    sig_pending   = False
    pending_side  = ""
    candles_since = 0

    total = len(main_c) - WARMUP
    for i in range(WARMUP, len(main_c)):
        candle = main_c[i]
        pct = (i - WARMUP) / max(1, total) * 100
        if i % 200 == 0:
            print(f"\r  Simulating {pct:.0f}% ({len(trades)} trades)...", end="", flush=True)

        # ── Exit ──────────────────────────────────────────────────────
        if open_trade:
            result = check_exit(open_trade, candle)
            if result:
                pnl = result["pnl"]
                equity += pnl
                peak_equity = max(peak_equity, equity)
                pnls.append(pnl)
                trades.append({"outcome": result["outcome"], "pnl": pnl})
                open_trade = None

        # Hard floor
        if peak_equity > 0 and (peak_equity - equity) / peak_equity >= 0.15:
            break

        if open_trade is not None:
            continue

        # ── Regime ────────────────────────────────────────────────────
        klines       = main_c[max(0, i - 299): i + 1]
        higher_slice = aligned_slice(higher_c, candle["t"], 100)

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
        if regime == "RANGING":
            if sig_pending: sig_pending = False; pending_side = ""; candles_since = 0
            continue

        if regime not in mode_cfg["allowed_regimes"]:
            if sig_pending: sig_pending = False; pending_side = ""; candles_since = 0
            continue

        _param_overrides = {}
        _allowed_sides: Optional[List[str]] = None
        if regime == "PARABOLIC":
            _param_overrides = {"min_score": max(0.35, coin_p["score_threshold"] - 0.05)}
            _allowed_sides = ["LONG"]

        # ── Signal ────────────────────────────────────────────────────
        sig = _compute_signal_layers(
            klines, MODE, 1.0, higher_slice, STYLE, None,
            param_overrides=_param_overrides if _param_overrides else None,
            symbol=SYMBOL,
        )

        if _allowed_sides and sig.get("ok") and sig.get("side") not in _allowed_sides:
            sig["ok"] = False

        # Volume hour confirmation
        if sig.get("ok"):
            _cp_vol = coin_p
            _vol_ok, _, _ = check_volume_hour_confirmed(klines, _cp_vol)
            if not _vol_ok:
                sig["ok"] = False

        if not sig.get("ok"):
            if sig_pending: sig_pending = False; pending_side = ""; candles_since = 0
            continue

        desired_side = sig["side"]
        if not sig_pending:
            sig_pending   = True
            pending_side  = desired_side
            candles_since = 0
        elif pending_side != desired_side:
            pending_side  = desired_side
            candles_since = 0
        candles_since += 1

        # Pullback gate
        _pb_res = check_pullback_entry(
            klines, pending_side, coin_p, regime,
            candles_since, trade_style=STYLE,
        )
        if _pb_res == "TIMEOUT":
            sig_pending = False; pending_side = ""; candles_since = 0
            continue
        if _pb_res == "WAIT":
            continue

        sig_pending = False; pending_side = ""; candles_since = 0

        grade    = sig.get("grade", "B")
        atr_pct  = sig.get("atr_pct", 0.009)
        st_cfg   = TRADE_STYLE_PARAMS[STYLE]
        sl_pct   = atr_pct * st_cfg["sl_atr"]
        tp_pct   = max(sl_pct * 1.5, atr_pct * st_cfg["tp_atr"])
        sl_pct   = min(sl_pct, st_cfg["sl_max"] / 100)
        tp_pct   = min(tp_pct, st_cfg["tp_max"] / 100)

        eff_size = c["size"]
        if grade == "B": eff_size *= 0.70

        open_trade = {
            "side": desired_side,
            "entry": candle["close"],
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "leverage": c["leverage"],
            "effective_size": eff_size,
            "equity_at_open": equity,
        }

    print(f"\r  Simulation complete — {len(trades)} trades        ")
    return {"trades": trades, "pnls": pnls, "final_equity": equity, "peak": peak_equity}


# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_stats(result: Dict) -> Dict:
    trades = result["trades"]
    pnls   = result["pnls"]
    n      = len(trades)
    if n == 0:
        return {"trades": 0, "wr": 0, "sharpe": 0, "return_pct": 0}

    wins   = sum(1 for t in trades if t["outcome"] == "TP_HIT")
    wr     = wins / n * 100
    ret    = (result["final_equity"] - 1000) / 1000 * 100

    if len(pnls) >= 2:
        avg    = statistics.mean(pnls)
        stddev = statistics.stdev(pnls)
        sharpe = (avg / stddev) * math.sqrt(252) if stddev > 0 else 0.0
    else:
        sharpe = 0.0

    return {"trades": n, "wr": round(wr, 1), "sharpe": round(sharpe, 2), "return_pct": round(ret, 1)}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  Direction Fix Validation — BTCUSDT 1h 2024-2025")
    print("  Mode: MINI_ASYM | Style: DAY_TRADE")
    print("=" * 60)

    print("\nFetching candles:")
    main_c, higher_c = load_candles()

    print(f"\nRunning simulation on {len(main_c) - WARMUP:,} candles...")
    result = simulate(main_c, higher_c)
    stats  = compute_stats(result)

    b = BASELINE
    print("\n" + "=" * 60)
    print(f"{'Metric':<18} {'Baseline':>10} {'New':>10} {'Change':>10}")
    print("-" * 60)
    print(f"{'Sharpe Ratio':<18} {b['sharpe']:>10.2f} {stats['sharpe']:>10.2f} {stats['sharpe']-b['sharpe']:>+10.2f}")
    print(f"{'Win Rate %':<18} {b['wr_pct']:>10.1f} {stats['wr']:>10.1f} {stats['wr']-b['wr_pct']:>+10.1f}")
    print(f"{'Trade Count':<18} {b['trades']:>10} {stats['trades']:>10} {stats['trades']-b['trades']:>+10}")
    print(f"{'Total Return %':<18} {'—':>10} {stats['return_pct']:>10.1f}%{'':>9}")
    print("=" * 60)

    improved = stats['sharpe'] > b['sharpe'] and stats['wr'] > b['wr_pct']
    if improved:
        print("\n  RESULT: Direction fix IMPROVED the engine.")
        print("  Safe to push to real branch after manual review.\n")
    elif stats['sharpe'] > b['sharpe']:
        print("\n  RESULT: Sharpe improved but WR unchanged/lower.")
        print("  Review trade count — may be over-filtering.\n")
    elif stats['trades'] < b['trades'] * 0.7:
        print("\n  RESULT: Too few trades. Direction fix is too aggressive —")
        print("  consider reducing the ±0.05 score adjustment.\n")
    else:
        print("\n  RESULT: No clear improvement. Reconsider parameters.\n")


if __name__ == "__main__":
    main()
