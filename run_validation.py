"""
run_validation.py — 7-Step Strategy Validation
===============================================
Run on Render shell:  python3 run_validation.py

Set STEP below to run each validation step in order.
Each step adds one more change on top of the previous.

Results printed as a table — copy/paste to share.
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
    get_market_regime, check_weekly_trend,
    check_pullback_entry, check_volume_hour_confirmed,
)
from coin_params import get_coin_params
from mode_params import get_mode_config, get_style_config

# ══════════════════════════════════════════════════════════════════════════════
#  ▶  CONFIGURE HERE — change STEP to run each validation step
# ══════════════════════════════════════════════════════════════════════════════
STEP = 7          # 1=coin params, 2=+pullback, 3=+volume_hour, 4=+regime, 5=+mode_config, 7=all

MODE  = "MINI_ASYM"
STYLE = "DAY_TRADE"

DATE_FROM  = "2020-01-01"
DATE_TO    = "2026-05-31"
START_MS   = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS     = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)  # inclusive of May

COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]
COIN_SHORT = {
    "BTCUSDT":"BTC", "ETHUSDT":"ETH", "BNBUSDT":"BNB", "XRPUSDT":"XRP",
    "SOLUSDT":"SOL", "ADAUSDT":"ADA", "DOGEUSDT":"DOGE", "AVAXUSDT":"AVAX",
    "LINKUSDT":"LINK", "DOTUSDT":"DOT",
}

# ── Feature flags — set automatically by STEP ─────────────────────────────
ENABLE_PULLBACK    = STEP >= 2   # Change 2
ENABLE_VOLUME_HOUR = STEP >= 3   # Change 3
ENABLE_REGIME      = STEP >= 4   # Change 4
ENABLE_MODE_CONFIG = STEP >= 5   # Change 5
# Changes 6 (floor), 7 (calendar), 8 (BTC corr) can't be simulated historically — always omitted

# ── Fixed constants ────────────────────────────────────────────────────────
TF         = "1h"
HIGHER_TF  = "4h"
WARMUP     = 240
ATR_BASE   = 0.0090  # 1h baseline

MODE_PRESETS = {
    "ULTRA_SAFE": {"size": 0.30, "leverage": 2},
    "SAFE":       {"size": 0.45, "leverage": 3},
    "NORMAL":     {"size": 0.60, "leverage": 5},
    "MINI_ASYM":  {"size": 0.65, "leverage": 6},
    "AGGRESSIVE": {"size": 0.85, "leverage": 8},
}
FEE     = {"maker": 0.00020, "taker": 0.00055, "slip": 0.00025}
ST      = TRADE_STYLE_PARAMS["DAY_TRADE"]

CACHE_FILE = os.path.join(os.path.dirname(__file__), "validation_cache.json")

# ══════════════════════════════════════════════════════════════════════════════
#  CANDLE FETCHING (Binance — 1000/page, 10× faster than OKX 100/page)
# ══════════════════════════════════════════════════════════════════════════════
BN_TF_MAP = {"1h": "1h", "4h": "4h", "1W": "1w"}


def fetch_binance(symbol: str, tf: str, start_ms: int, end_ms: int) -> List[Dict]:
    interval = BN_TF_MAP.get(tf, tf)
    url      = "https://api.binance.com/api/v3/klines"
    candles, current, pages = [], start_ms, 0
    print(f"    {symbol} {tf} from Binance...", end="", flush=True)
    while current < end_ms:
        try:
            r = httpx.get(url, params={
                "symbol": symbol, "interval": interval,
                "startTime": current, "endTime": end_ms, "limit": 1000,
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
            pages  += 1
            if len(data) < 1000: break
            time.sleep(0.12)
            if pages % 10 == 0: print(".", end="", flush=True)
        except Exception as e:
            print(f" [retry:{e}]", end="", flush=True); time.sleep(3)
    print(f" {len(candles):,} candles")
    return candles


def _load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f: return json.load(f)
        except Exception: pass
    return {}


def _save_cache(cache: Dict):
    with open(CACHE_FILE, "w") as f: json.dump(cache, f)


def get_candles(symbol: str, tf: str, cache: Dict) -> List[Dict]:
    key = f"{symbol}_{tf}"
    if key in cache:
        print(f"    {symbol} {tf}: cached ({len(cache[key]):,} candles)")
        return cache[key]
    data = fetch_binance(symbol, tf, START_MS, END_MS)
    cache[key] = data
    _save_cache(cache)
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def aligned_slice(candles: List[Dict], before_ts: int, n: int) -> List[Dict]:
    lo, hi, idx = 0, len(candles) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if candles[mid]["t"] <= before_ts: idx = mid; lo = mid + 1
        else: hi = mid - 1
    if idx < 0: return []
    return candles[max(0, idx - n + 1): idx + 1]


def session_q(ts_ms: int) -> float:
    hour = datetime.fromtimestamp(ts_ms / 1000, tz=DUBAI_TZ).hour
    if 2 <= hour < 6:  return 0.72
    if 6 <= hour < 12: return 0.85
    if 12 <= hour < 16: return 0.95
    if 16 <= hour < 21: return 1.00
    if 21 <= hour <= 23: return 0.92
    return 0.80


def trade_fee(size_usd: float, is_tp: bool) -> float:
    return size_usd * (FEE["maker"] + (FEE["maker"] if is_tp else FEE["taker"]) + FEE["slip"])


def check_exit(trade: Dict, candle: Dict) -> Optional[Dict]:
    side, entry = trade["side"], trade["entry"]
    sl, tp  = trade["sl_pct"], trade["tp_pct"]
    lev, eq = trade["leverage"], trade["equity_at_open"]
    hi, lo  = candle["high"], candle["low"]
    sl_p = entry * (1 - sl) if side == "LONG" else entry * (1 + sl)
    tp_p = entry * (1 + tp) if side == "LONG" else entry * (1 - tp)
    sl_hit = lambda: lo <= sl_p if side == "LONG" else hi >= sl_p
    tp_hit = lambda: hi >= tp_p if side == "LONG" else lo <= tp_p
    size   = trade["effective_size"] * eq

    if tp_hit() and sl_hit():
        return {"outcome": "SL_HIT", "pnl": -size * sl * lev - trade_fee(size, False)}
    if tp_hit():
        # Grade B: T1 (60%) at tp, T2 (40%) at 2× tp
        t1   = size * 0.60
        pnl1 = t1 * tp * lev - trade_fee(t1, True)
        t2   = size * 0.40
        tp2  = min(tp * 2.0, ST["tp_max"] / 100)
        pnl2 = t2 * tp2 * lev - trade_fee(t2, True)
        return {"outcome": "TP_HIT", "pnl": pnl1 + pnl2}
    if sl_hit():
        return {"outcome": "SL_HIT", "pnl": -size * sl * lev - trade_fee(size, False)}
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def simulate(symbol: str, main_c: List[Dict], higher_c: List[Dict],
             btc_weekly_c: List[Dict]) -> Dict:
    c          = MODE_PRESETS[MODE]
    mode_cfg   = get_mode_config(MODE)
    style_cfg  = get_style_config(STYLE)
    coin_p     = get_coin_params(symbol)

    equity       = 1000.0
    peak_equity  = 1000.0
    trades: List[Dict] = []
    equity_curve = [{"ts": main_c[WARMUP]["t"], "equity": equity}]
    yearly: Dict[int, List[float]] = {}

    open_trade: Optional[Dict] = None
    is_parabolic  = False
    sig_pending   = False
    pending_side  = ""
    pending_score = 0.0
    candles_since = 0

    total = len(main_c) - WARMUP
    for i in range(WARMUP, len(main_c)):
        candle = main_c[i]
        year   = datetime.fromtimestamp(candle["t"] / 1000, tz=timezone.utc).year

        if i % 500 == 0:
            pct = (i - WARMUP) / max(1, total) * 100
            print(f"\r    {COIN_SHORT[symbol]} {pct:.0f}%...", end="", flush=True)

        # ── Exit check ────────────────────────────────────────────────
        if open_trade:
            result = check_exit(open_trade, candle)
            if result:
                pnl = result["pnl"]
                equity += pnl
                peak_equity = max(peak_equity, equity)
                equity_curve.append({"ts": candle["t"], "equity": round(equity, 4)})
                yearly.setdefault(year, []).append(pnl)
                trades.append({"outcome": result["outcome"], "pnl": pnl, "year": year})
                open_trade = None

        # ── Hard floor (15% drawdown cap) ─────────────────────────────
        if peak_equity > 0 and (peak_equity - equity) / peak_equity >= 0.15:
            break

        if open_trade is not None:
            continue

        # ── Session quality (skip 02-06 Dubai) ────────────────────────
        if session_q(candle["t"]) < 0.75:
            continue

        klines       = main_c[max(0, i - 299): i + 1]
        higher_slice = aligned_slice(higher_c, candle["t"], 100)

        # ── Change 4: Regime detection ────────────────────────────────
        regime = "TRENDING"
        if ENABLE_REGIME:
            weekly_slice = aligned_slice(btc_weekly_c, candle["t"], 210) if btc_weekly_c else []
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
            regime, is_parabolic = get_market_regime(weekly_slice, cur_adx, is_parabolic)
            if regime == "RANGING":
                if sig_pending: sig_pending = False; pending_side = ""; candles_since = 0
                continue

        # ── Change 5: Mode regime guard ───────────────────────────────
        if ENABLE_MODE_CONFIG and regime not in mode_cfg["allowed_regimes"]:
            if sig_pending: sig_pending = False; pending_side = ""; candles_since = 0
            continue

        # PARABOLIC overrides
        _allowed_sides: Optional[List[str]] = None
        _param_overrides = {}
        if ENABLE_REGIME and regime == "PARABOLIC":
            _param_overrides = {"min_score": max(0.35, coin_p["score_threshold"] - 0.05)}
            _allowed_sides = ["LONG"]

        # ── Signal ───────────────────────────────────────────────────
        sig = _compute_signal_layers(
            klines, MODE, 1.0, higher_slice, STYLE, None,
            param_overrides=_param_overrides if _param_overrides else None,
            symbol=symbol,
        )

        if _allowed_sides and sig.get("ok") and sig.get("side") not in _allowed_sides:
            sig["ok"] = False

        # ── Change 3: Volume hour confirmation ────────────────────────
        if ENABLE_VOLUME_HOUR and sig.get("ok"):
            vol_ok, _, _ = check_volume_hour_confirmed(klines, coin_p)
            if not vol_ok:
                sig["ok"] = False

        # Cancel pending if conditions changed
        if not sig.get("ok"):
            if sig_pending: sig_pending = False; pending_side = ""; pending_score = 0.0; candles_since = 0
            continue

        desired_side = sig["side"]

        # ── Change 2: Pullback entry gate ─────────────────────────────
        if ENABLE_PULLBACK:
            if not sig_pending:
                sig_pending = True; pending_side = desired_side
                pending_score = sig.get("score", 0.0); candles_since = 0
            elif pending_side != desired_side:
                pending_side = desired_side; pending_score = sig.get("score", 0.0); candles_since = 0
            candles_since += 1

            if style_cfg.get("pullback_required", True):
                pb = check_pullback_entry(klines, pending_side, coin_p, regime, candles_since, STYLE)
                if pb == "TIMEOUT":
                    sig_pending = False; pending_side = ""; pending_score = 0.0; candles_since = 0
                    continue
                if pb == "WAIT":
                    continue
            # ENTER or IMMEDIATE — proceed

            desired_side = pending_side
            sig_pending = False; pending_side = ""; pending_score = 0.0; candles_since = 0
        else:
            # No pullback — clear any stale state
            sig_pending = False; pending_side = ""; candles_since = 0

        # ── Open trade ────────────────────────────────────────────────
        atr_pct  = sig.get("atr_pct", ATR_BASE)
        vol_mult = max(0.4, 1.0 / (atr_pct / ATR_BASE)) if atr_pct / ATR_BASE > 1.5 else 1.0
        dd_pct   = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        dd_mult  = 0.25 if dd_pct >= 0.10 else 0.40 if dd_pct >= 0.07 else 0.65 if dd_pct >= 0.04 else 1.0
        mode_sm  = mode_cfg.get("position_size_mult", 1.0) if ENABLE_MODE_CONFIG else 1.0

        sl_pct = min(atr_pct * ST["sl_atr"], ST["sl_max"] / 100)
        tp_pct = min(atr_pct * ST["tp_atr"], ST["tp_max"] / 100)

        open_trade = {
            "side":           desired_side,
            "entry":          candle["close"],
            "sl_pct":         sl_pct,
            "tp_pct":         tp_pct,
            "leverage":       c["leverage"],
            "equity_at_open": equity,
            "effective_size": c["size"] * vol_mult * dd_mult * mode_sm,
        }

    print(f"\r    {COIN_SHORT[symbol]} done — {len(trades)} trades          ")

    # ── Metrics ───────────────────────────────────────────────────────
    wins    = [t for t in trades if t["outcome"] == "TP_HIT"]
    wr      = len(wins) / len(trades) * 100 if trades else 0.0
    total_r = (equity - 1000.0) / 1000.0 * 100
    pnls    = [t["pnl"] for t in trades]
    sharpe  = 0.0
    if len(pnls) >= 2:
        avg = statistics.mean(pnls); std = statistics.stdev(pnls)
        sharpe = (avg / std * math.sqrt(252)) if std > 0 else 0.0
    max_dd  = 0.0
    peak    = 1000.0
    eq      = 1000.0
    for pt in equity_curve:
        eq   = pt["equity"]
        peak = max(peak, eq)
        dd   = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Year-by-year returns
    eq_by_year: Dict[int, float] = {}
    running = 1000.0
    for yr in sorted(yearly.keys()):
        y_pnl = sum(yearly[yr])
        pct   = y_pnl / running * 100
        eq_by_year[yr] = round(pct, 1)
        running += y_pnl

    return {
        "trades": len(trades), "wr": round(wr, 1), "return": round(total_r, 1),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd * 100, 1),
        "by_year": eq_by_year,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*70}")
    print(f"  ASYMMETRIC AI — VALIDATION STEP {STEP}  |  {MODE} {STYLE}  |  2020-2026 May")
    flags = []
    if ENABLE_PULLBACK:    flags.append("PULLBACK")
    if ENABLE_VOLUME_HOUR: flags.append("VOL_HOUR")
    if ENABLE_REGIME:      flags.append("REGIME")
    if ENABLE_MODE_CONFIG: flags.append("MODE_CFG")
    print(f"  Active changes: COIN_PARAMS" + ((" + " + " + ".join(flags)) if flags else " only"))
    print(f"{'='*70}\n")

    cache = _load_cache()

    # ── Fetch BTC weekly candles (shared for all coins) ──────────────
    btc_weekly: List[Dict] = []
    if ENABLE_REGIME:
        print("  Fetching BTC weekly candles for regime detection...")
        btc_weekly = get_candles("BTCUSDT", "1W", cache)
        if not btc_weekly:
            print("  WARNING: BTC weekly fetch failed — regime defaults to TRENDING")

    # ── Run each coin ─────────────────────────────────────────────────
    results: Dict[str, Dict] = {}
    for sym in COINS:
        print(f"\n  [{COIN_SHORT[sym]}]")
        main_c   = get_candles(sym, TF, cache)
        higher_c = get_candles(sym, HIGHER_TF, cache)
        if len(main_c) < WARMUP + 50:
            print(f"  SKIP: only {len(main_c)} candles"); continue
        results[sym] = simulate(sym, main_c, higher_c, btc_weekly)

    # ── Print results table ───────────────────────────────────────────
    years = list(range(2020, 2027))
    col   = 7  # year column width
    print(f"\n{'='*70}")
    print(f"  RESULTS — STEP {STEP}  |  {MODE} {STYLE}  |  2020–2026 May")
    print(f"{'='*70}")

    hdr = f"  {'Coin':<6} {'Trades':>6} {'WinRate':>7} {'Return':>7} {'Sharpe':>6} {'MaxDD':>6}  "
    hdr += "  ".join(f"{y}" for y in years)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_positive = True
    for sym in COINS:
        r = results.get(sym)
        if not r: continue
        row = f"  {COIN_SHORT[sym]:<6} {r['trades']:>6} {r['wr']:>6.1f}% {r['return']:>+6.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>5.1f}%  "
        for yr in years:
            val = r["by_year"].get(yr)
            if val is None:
                row += f"{'  N/A':>{col}}  "
            else:
                flag = "✓" if val > 0 else "✗"
                row += f"{val:>+5.1f}%{flag}  "
                if val <= 0: all_positive = False
        print(row)

    print("\n" + "  " + "-" * 65)
    verdict = "✅ ALL POSITIVE" if all_positive else "❌ SOME NEGATIVE — review before proceeding to next step"
    print(f"  {verdict}")
    print(f"{'='*70}\n")

    # Step guidance
    next_steps = {
        1: "If all 10 coins positive → run STEP=2 (add pullback entry)",
        2: "If WR 42%+ on all coins → run STEP=3 (add volume hour)",
        3: "If WR 45%+ on all coins → run STEP=4 (add regime detector)",
        4: "If 2021 shows 80%+ on all coins → run STEP=5 (add mode config)",
        5: "If all 60 combo positive → run STEP=7 (full final test)",
        7: "Final validation complete. If all green → safe to keep changes live.",
    }
    if STEP in next_steps:
        print(f"  NEXT: {next_steps[STEP]}")
        print()


if __name__ == "__main__":
    main()
