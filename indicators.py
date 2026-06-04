"""
indicators.py — pure technical analysis functions.

No DB, no exchange, no FastAPI, no engine state.
Safe to import anywhere with zero side effects.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Literal
from config import now_dubai

RiskMode = Literal["ULTRA_SAFE", "SAFE", "NORMAL", "MINI_ASYM", "AGGRESSIVE"]


def _ema(series: List[float], period: int) -> List[float]:
    if not series:
        return []
    k = 2.0 / (period + 1.0)
    out = [series[0]]
    for x in series[1:]:
        out.append((x - out[-1]) * k + out[-1])
    return out


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    return 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al))


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
           for i in range(1, len(closes))]
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period * 2 + 1:
        return None
    pdms, ndms, trs = [], [], []
    for i in range(1, len(closes)):
        hd = highs[i] - highs[i - 1]
        ld = lows[i - 1] - lows[i]
        pdms.append(hd if hd > ld and hd > 0 else 0.0)
        ndms.append(ld if ld > hd and ld > 0 else 0.0)
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))

    def _wilder(s: List[float]) -> List[float]:
        v = sum(s[:period])
        out = [v]
        for x in s[period:]:
            v = v - v / period + x
            out.append(v)
        return out

    s_tr, s_p, s_n = _wilder(trs), _wilder(pdms), _wilder(ndms)
    dxs = []
    for i in range(len(s_tr)):
        pdi = 100 * s_p[i] / s_tr[i] if s_tr[i] else 0
        ndi = 100 * s_n[i] / s_tr[i] if s_tr[i] else 0
        dxs.append(100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else 0)
    if len(dxs) < period:
        return None
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


# Higher timeframe for trend bias — always one step up
HIGHER_TF_MAP: Dict[str, str] = {"15m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}


def _rsi_series(closes: List[float], period: int = 14) -> List[float]:
    """RSI value for every candle after warmup — needed for divergence check."""
    if len(closes) < period + 2:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    out = []
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out.append(100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al)))
    return out


def _session_quality(trade_style: str) -> tuple:
    """
    Session quality multiplier — maps real market sessions to Dubai time (UTC+4).
    Higher score = more liquidity = signal threshold easier to meet.
    SCALP is hard-blocked 02:00-06:00 (dead zone, thin market).

    Dubai (UTC+4) session schedule:
      02:00–06:00 → Dead zone (Asia asleep, US closed)        → SCALP: 0.0, others: 0.72
      06:00–12:00 → Asia / Tokyo session                      → 0.85
      12:00–16:00 → London open (big moves start)             → 0.95
      16:00–21:00 → London + NY overlap (BEST — peak volume)  → 1.00
      21:00–01:00 → NY session solo                           → 0.92
      01:00–02:00 → NY close / wind-down                      → 0.80
    """
    if trade_style == "SWING":
        return 1.0, "swing"
    hour = now_dubai().hour

    # Dead zone — thin market, wide spreads, low volume
    if 2 <= hour < 6:
        if trade_style == "SCALP":
            return 0.0, f"scalp-blocked {hour:02d}:xx (dead zone 02-06 Dubai)"
        return 0.72, f"dead-zone {hour:02d}:xx"

    # Asia / Tokyo session — decent crypto liquidity
    if 6 <= hour < 12:
        return 0.85, f"Asia {hour:02d}:xx"

    # London open — institutions enter, volume spikes, big directional moves
    if 12 <= hour < 16:
        return 0.95, f"London {hour:02d}:xx"

    # London + NY overlap — peak volume, best setups, highest probability
    if 16 <= hour < 21:
        return 1.00, f"London+NY {hour:02d}:xx"

    # NY solo session — still active, good liquidity
    if 21 <= hour <= 23:
        return 0.92, f"NY {hour:02d}:xx"

    # NY close / wind-down (01:00-02:00)
    return 0.80, f"NY-close {hour:02d}:xx"


def _ema_spread_trend(ema_fast: List[float], ema_slow: List[float], n: int = 4) -> tuple:
    """
    Checks if EMA gap is WIDENING (trend accelerating) or NARROWING (stalling).
    Returns (widening: bool, score: float 0-1)
    A widening spread = conviction in trend direction → better entry.
    Narrowing = trend losing steam → lower quality entry.
    """
    if len(ema_fast) < n + 1 or len(ema_slow) < n + 1:
        return False, 0.5
    spread_now  = abs(ema_fast[-1] - ema_slow[-1])
    spread_prev = abs(ema_fast[-n] - ema_slow[-n])
    if spread_prev < 1e-10:
        return False, 0.5
    ratio = spread_now / spread_prev
    widening = ratio >= 1.03   # at least 3% wider than 4 candles ago
    score = min(1.0, 0.5 + (ratio - 1.0) * 2.0) if widening else max(0.1, ratio * 0.5)
    return widening, round(score, 3)


def _candle_pattern(klines: List[Dict], side: str) -> tuple:
    """
    Score the last candle's pattern quality for the given trade direction.
    Returns (pattern_name: str, score: float 0-1)
    pin_bar=1.0, engulfing=0.90, momentum=0.75, basic=0.50, against=0.10
    """
    if len(klines) < 2:
        return "none", 0.3
    curr = klines[-1]
    prev = klines[-2]
    body  = abs(curr["close"] - curr["open"])
    rng   = curr["high"] - curr["low"]
    if rng < 1e-10:
        return "doji", 0.25
    body_ratio = body / rng

    if side == "LONG":
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]
        # Pin bar: hammer shape — long lower wick, small body, closes bullish
        if body_ratio < 0.35 and lower_wick > rng * 0.55 and curr["close"] > curr["open"]:
            return "pin_bar", 1.0
        # Bullish engulfing: current body covers previous body completely
        if (curr["close"] > curr["open"] and
                curr["open"] <= prev["close"] and curr["close"] >= prev["open"]):
            return "engulfing", 0.90
        # Strong momentum candle: large bullish body
        if curr["close"] > curr["open"] and body_ratio > 0.65:
            return "momentum", 0.75
        # Basic bullish
        if curr["close"] > curr["open"]:
            return "bullish", 0.50
    else:
        upper_wick = curr["high"] - max(curr["open"], curr["close"])
        # Shooting star / pin bar bearish
        if body_ratio < 0.35 and upper_wick > rng * 0.55 and curr["close"] < curr["open"]:
            return "pin_bar", 1.0
        # Bearish engulfing
        if (curr["close"] < curr["open"] and
                curr["open"] >= prev["close"] and curr["close"] <= prev["open"]):
            return "engulfing", 0.90
        # Strong bearish momentum candle
        if curr["close"] < curr["open"] and body_ratio > 0.65:
            return "momentum", 0.75
        # Basic bearish
        if curr["close"] < curr["open"]:
            return "bearish", 0.50

    return "against_trend", 0.10


def _rsi_divergence(closes: List[float], rsi_vals: List[float], side: str, n: int = 12) -> tuple:
    """
    Detect RSI divergence — price makes new extreme but RSI disagrees.
    Bearish divergence (during LONG): price higher high, RSI lower high → weakening bull.
    Bullish divergence (during SHORT): price lower low, RSI higher low → weakening bear.
    Returns (divergence: bool, score_penalty: float 0.0-0.30)
    """
    if len(closes) < n + 1 or len(rsi_vals) < n:
        return False, 0.0
    price_window = closes[-n:]
    rsi_window   = rsi_vals[-n:]
    p_latest     = closes[-1]
    r_latest     = rsi_vals[-1]

    if side == "LONG":
        p_prev_high = max(price_window[:-3])
        r_prev_high = max(rsi_window[:-3])
        divergence  = p_latest > p_prev_high * 1.001 and r_latest < r_prev_high * 0.96
    else:
        p_prev_low  = min(price_window[:-3])
        r_prev_low  = min(rsi_window[:-3])
        divergence  = p_latest < p_prev_low * 0.999 and r_latest > r_prev_low * 1.04

    return divergence, (0.28 if divergence else 0.0)

def _reversal_signal(klines: List[Dict], rsi: Optional[float], adx: Optional[float], atr: Optional[float]) -> Dict:
    """
    T16 Reversal Detection — catches extreme RSI bounces in choppy/ranging markets.
    Fires when the 4-layer trend system is blocked (ADX 12-22 choppy zone).

    All 5 conditions required:
      1. RSI < 28 (LONG) or RSI > 72 (SHORT)
      2. Volume ≥ 3× 20-candle median
      3. Price ≥ 3% beyond EMA21 in reversal direction
      4. Last candle closes in reversal direction (green/LONG, red/SHORT)
      5. ADX 12-22 (choppy — where reversals happen, unlike trends)

    Returns: {ok, side, score, signal_type, grade, sl_atr_override, tp_atr_override, log}
    """
    if rsi is None or adx is None or atr is None or len(klines) < 25:
        return {"ok": False}

    closes  = [k["close"]  for k in klines]
    opens   = [k["open"]   for k in klines]
    volumes = [k["volume"] for k in klines]

    price = closes[-1]
    ema21 = _ema(closes, 21)[-1]

    # 1 — Extreme RSI
    is_long = rsi < 28
    is_short = rsi > 72
    if not (is_long or is_short):
        return {"ok": False}
    side = "LONG" if is_long else "SHORT"

    # 2 — Volume ≥ 3× 20-candle median (use [-2] = last completed candle)
    vol_window = sorted(volumes[-22:-2]) if len(volumes) >= 22 else sorted(volumes[:-2])
    avg_vol = vol_window[len(vol_window) // 2] if vol_window else 0.0
    vol_ratio = volumes[-2] / avg_vol if avg_vol > 0 else 0.0
    if vol_ratio < 3.0:
        return {"ok": False}

    # 3 — Price ≥ 3% beyond EMA21 in reversal direction
    if side == "LONG":
        ema_dist = (ema21 - price) / ema21  # positive = price below EMA21
    else:
        ema_dist = (price - ema21) / ema21  # positive = price above EMA21
    if ema_dist < 0.03:
        return {"ok": False}

    # 4 — Last candle closes in reversal direction
    last_green = closes[-1] > opens[-1]
    last_red   = closes[-1] < opens[-1]
    if side == "LONG" and not last_green:
        return {"ok": False}
    if side == "SHORT" and not last_red:
        return {"ok": False}

    # 5 — ADX 12-22 (choppy/ranging — NOT trending)
    if not (12 <= adx <= 22):
        return {"ok": False}

    # All conditions met — score based on extremity
    score = 0.72
    if side == "LONG":
        score += min(0.08, (28 - rsi) / 100)
    else:
        score += min(0.08, (rsi - 72) / 100)
    score += min(0.05, (vol_ratio - 3.0) / 20)

    candle_word = "green" if side == "LONG" else "red"
    log_msg = (
        f"REVERSAL SIGNAL: RSI {rsi:.1f} extreme + volume {vol_ratio:.1f}× spike + "
        f"EMA {ema_dist*100:.1f}% away + {candle_word} candle confirmed → {side} reversal"
    )
    return {
        "ok": True, "side": side, "score": round(min(0.85, score), 3),
        "signal_type": "REVERSAL", "grade": "B",
        "sl_atr_override": 1.5, "tp_atr_override": 3.0,
        "log": log_msg,
    }


def _classify_regime(
    highs: List[float], lows: List[float], closes: List[float],
    adx: Optional[float], atr_pct: float,
) -> Dict:
    """
    Phase 4 — Market regime classifier.

    Compares current ATR to its own 50-candle baseline so the engine knows
    whether it is in a calm trend, a choppy range, or a volatility spike.

    Returns:
        regime   : "TRENDING" | "MARGINAL" | "CHOPPY" | "VOLATILE"
        block    : True  = hard block (do not trade)
        min_score_penalty : added to min_score for MARGINAL (raises the bar)
        spike_ratio : current ATR / 50-candle ATR baseline
        reason   : human-readable explanation
    """
    # 50-candle ATR as the volatility baseline for this symbol
    atr_slow = _atr(highs, lows, closes, 50)
    price    = closes[-1] if closes else 1.0
    atr_slow_pct = (atr_slow / price) if (atr_slow and price) else 0.0

    spike_ratio = round(atr_pct / atr_slow_pct, 2) if atr_slow_pct > 0 else 1.0
    adx_val     = round(adx or 0.0, 1)

    # ── VOLATILE — ATR spiked 2.5× above its own 50-candle baseline ─────────
    # News event, liquidation cascade, or major wick. SL distances are based on
    # normal ATR — in a spike the market can move 3-4× SL in one candle.
    if spike_ratio >= 2.5:
        return {
            "regime": "VOLATILE",
            "block":  True,
            "min_score_penalty": 0.0,
            "spike_ratio": spike_ratio,
            "adx": adx_val,
            "reason": (
                f"ATR {spike_ratio:.1f}× above baseline — news/liquidation spike. "
                f"SL distance is unreliable in these conditions. Waiting for volatility to normalize."
            ),
        }

    # ── CHOPPY — low ADX + quiet ATR = price ranging between same two levels ─
    # False breakouts dominate ranging markets. EMA21 pullback entries fail
    # because price re-crosses EMA21 multiple times with no follow-through.
    if adx is not None and adx < 18 and spike_ratio <= 0.90:
        return {
            "regime": "CHOPPY",
            "block":  True,
            "min_score_penalty": 0.0,
            "spike_ratio": spike_ratio,
            "adx": adx_val,
            "reason": (
                f"ADX {adx_val} < 18 and ATR {spike_ratio:.2f}× baseline — "
                f"market is ranging with no trend. False signals dominate. Waiting for trend to develop."
            ),
        }

    # ── MARGINAL — weak trend or mildly elevated volatility ─────────────────
    # Still tradeable but requires a stronger signal than normal.
    # Raise min_score by 0.07 so only the clearest setups fire.
    if (adx is not None and adx < 22) or spike_ratio >= 1.80:
        label = (
            f"ADX {adx_val} — weak trend"
            if (adx is not None and adx < 22)
            else f"ATR {spike_ratio:.1f}× baseline — elevated volatility"
        )
        return {
            "regime": "MARGINAL",
            "block":  False,
            "min_score_penalty": 0.07,
            "spike_ratio": spike_ratio,
            "adx": adx_val,
            "reason": f"{label}. Requiring stronger signal before entry.",
        }

    # ── TRENDING — clear directional move with normal volatility ────────────
    return {
        "regime": "TRENDING",
        "block":  False,
        "min_score_penalty": 0.0,
        "spike_ratio": spike_ratio,
        "adx": adx_val,
        "reason": "",
    }


# Per-mode signal thresholds + minimum quality score to trade
MODE_SIGNAL_PARAMS: Dict[str, Dict] = {
    "ULTRA_SAFE": dict(adx_min=28, atr_min=0.003, atr_max=0.020, rsi_min=42, rsi_max=58, pullback_max=0.012, vol_factor=1.40, mom_n=3, min_score=0.75),
    "SAFE":       dict(adx_min=22, atr_min=0.002, atr_max=0.025, rsi_min=40, rsi_max=62, pullback_max=0.015, vol_factor=1.25, mom_n=2, min_score=0.68),
    "NORMAL":     dict(adx_min=18, atr_min=0.002, atr_max=0.030, rsi_min=38, rsi_max=65, pullback_max=0.020, vol_factor=1.15, mom_n=2, min_score=0.62),
    "MINI_ASYM":  dict(adx_min=16, atr_min=0.001, atr_max=0.035, rsi_min=35, rsi_max=68, pullback_max=0.025, vol_factor=1.05, mom_n=1, min_score=0.63),
    "AGGRESSIVE": dict(adx_min=13, atr_min=0.001, atr_max=0.040, rsi_min=32, rsi_max=70, pullback_max=0.030, vol_factor=0.95, mom_n=1, min_score=0.58),
}


def _compute_signal_layers(
    klines: List[Dict],
    mode: RiskMode,
    adaptive_strictness: float = 1.0,
    higher_klines: Optional[List[Dict]] = None,
    trade_style: str = "DAY_TRADE",
    mtf_klines: Optional[List[Dict]] = None,
    param_overrides: Optional[Dict] = None,
    enable_t16: bool = True,
) -> Dict:
    """
    4-layer scored signal analysis.
    Direction from higher TF trend. Entry on pullback+bounce, not on crossover.
    Each layer returns a 0-1 score. Total must exceed mode threshold.
    """
    # ── Session quality multiplier ───────────────────────────────────────
    # Low-liquidity hours reduce the final score so signals must be stronger.
    # SCALP 02:00-06:00 Dubai returns sess_score=0.0 = hard block (scalping
    # thin markets is bad practice and the math is unfavourable at <0.72×).
    sess_score, sess_label = _session_quality(trade_style)
    if sess_score == 0.0:
        return {
            "ok": False,
            "blocked": f"SCALP_LOW_LIQ: Scalping paused 02:00–06:00 Dubai (thin market) — {sess_label}",
            "signal": "BLOCKED", "score": 0.0, "breakdown": {},
            "side": None, "adaptive_strictness": adaptive_strictness,
        }

    if len(klines) < 220:
        return {"ok": False, "blocked": "NO_DATA", "signal": "NO_DATA", "score": 0.0, "breakdown": {}}

    closes  = [k["close"]  for k in klines]
    highs   = [k["high"]   for k in klines]
    lows    = [k["low"]    for k in klines]
    volumes = [k["volume"] for k in klines]

    ema9      = _ema(closes, 9)
    ema21     = _ema(closes, 21)
    ema50     = _ema(closes, 50)
    ema200    = _ema(closes, 200)
    rsi       = _rsi(closes, 14)
    rsi_vals  = _rsi_series(closes, 14)   # full series for divergence
    atr       = _atr(highs, lows, closes, 14)
    adx       = _adx(highs, lows, closes, 14)

    price      = closes[-1]
    atr_pct    = (atr / price) if atr else 0.0

    # ── T16 Reversal check — runs before regime block ────────────────────────
    # Reversals fire in choppy conditions (ADX 12-22) that the normal 4-layer
    # system would reject. If regime would block AND reversal conditions are met,
    # return the reversal signal instead of BLOCKED.
    _rev = _reversal_signal(klines, rsi, adx, atr) if enable_t16 else {"ok": False}

    # ── Phase 4: Regime classification (before any layer scoring) ────────────
    regime_class = _classify_regime(highs, lows, closes, adx, atr_pct)
    if regime_class["block"]:
        if _rev["ok"]:
            return {
                "ok": True, "signal": "REVERSAL", "side": _rev["side"],
                "score": _rev["score"], "grade": "B",
                "signal_type": "REVERSAL",
                "sl_atr_override": _rev["sl_atr_override"],
                "tp_atr_override": _rev["tp_atr_override"],
                "reversal_log": _rev["log"],
                "breakdown": {"reversal": _rev["log"]},
                "atr_pct": atr_pct,
                "market_regime": regime_class["regime"],
                "adaptive_strictness": adaptive_strictness,
            }
        return {
            "ok": False,
            "blocked": f"REGIME_{regime_class['regime']}: {regime_class['reason']}",
            "signal": "BLOCKED", "score": 0.0, "breakdown": {},
            "side": None, "adaptive_strictness": adaptive_strictness,
            "market_regime": regime_class["regime"],
            "regime_detail": regime_class,
        }

    # Use volumes[-2] = last COMPLETED candle (OKX returns the in-progress candle
    # as volumes[-1], which has only partial volume — e.g. 1 minute into a 15m
    # candle shows 0.07x, making the volume filter block valid setups).
    # Use median of 40 completed candles as baseline (robust to spike candles
    # from big moves like BTC 74→77 inflating the mean and blocking normal vol).
    _vol_window = sorted(volumes[-42:-2]) if len(volumes) >= 42 else sorted(volumes[:-2])
    avg_vol    = _vol_window[len(_vol_window) // 2] if _vol_window else 0.0
    vol_ratio  = volumes[-2] / avg_vol if avg_vol > 0 and len(volumes) >= 3 else 0.0

    # Apply adaptive strictness for MINI_ASYM
    p = MODE_SIGNAL_PARAMS.get(mode, MODE_SIGNAL_PARAMS["NORMAL"]).copy()
    if param_overrides:
        p.update({k: v for k, v in param_overrides.items() if k in p})
        # Delta keys: optimizer passes offsets, not absolute replacements
        if "adx_min_delta" in param_overrides:
            p["adx_min"] = max(5, p["adx_min"] + param_overrides["adx_min_delta"])
        if "score_min_delta" in param_overrides:
            p["min_score"] = max(0.20, min(0.95, p["min_score"] + param_overrides["score_min_delta"]))

    # Trade-style tightens/relaxes entry thresholds — SCALP must be strict
    # because 15m candles are noisy; SWING can tolerate wider pullbacks.
    # mom_n_add=0 for SCALP: MINI_ASYM already requires 1 candle — adding a 2nd
    # blocks too many valid entries in volatile/recovering markets.
    _style_adj = {
        "SCALP":     dict(pullback_mult=0.40, mom_n_add=0, rsi_tighten=4,  score_add=0.02),
        "DAY_TRADE": dict(pullback_mult=0.70, mom_n_add=0, rsi_tighten=2,  score_add=0.02),
        "SWING":     dict(pullback_mult=1.20, mom_n_add=0, rsi_tighten=-3, score_add=0.0),
    }
    adj = _style_adj.get(trade_style, _style_adj["DAY_TRADE"])
    p["pullback_max"] = p["pullback_max"] * adj["pullback_mult"]
    p["mom_n"]        = max(1, p["mom_n"] + adj["mom_n_add"])
    p["rsi_min"]      = min(50, p["rsi_min"] + adj["rsi_tighten"])
    p["rsi_max"]      = max(50, p["rsi_max"] - adj["rsi_tighten"])
    p["min_score"]    = min(0.92, p["min_score"] + adj["score_add"])

    if mode == "MINI_ASYM" and adaptive_strictness != 1.0:
        s = adaptive_strictness
        p["adx_min"]      = p["adx_min"] * s
        new_rsi_min = p["rsi_min"] + (s - 1) * 8
        new_rsi_max = p["rsi_max"] - (s - 1) * 8
        # Clamp so the RSI window never shrinks below 12 points — prevents the
        # engine from becoming permanently blocked after a losing streak with no message.
        mid = (new_rsi_min + new_rsi_max) / 2.0
        if new_rsi_max - new_rsi_min < 12:
            new_rsi_min = mid - 6
            new_rsi_max = mid + 6
        p["rsi_min"]      = min(50, new_rsi_min)
        p["rsi_max"]      = max(50, new_rsi_max)
        p["pullback_max"] = p["pullback_max"] / max(0.5, s)
        p["vol_factor"]   = p["vol_factor"] * s
        p["min_score"]    = min(0.90, p["min_score"] + (s - 1) * 0.12)

    # Apply MARGINAL regime penalty — raises min_score so only strong signals fire
    p["min_score"] = min(0.92, p["min_score"] + regime_class["min_score_penalty"])

    # ── Higher TF trend direction ─────────────────────────────────────────
    # Primary: 4h EMA21 vs EMA50 crossover sets macro bias.
    # Early-bear override: if EMA21 has fallen >0.3% over last 8 candles (~32h)
    # the trend is already turning even before the full crossover — flip to SHORT early.
    # Threshold lowered from 0.5% → 0.3% to catch slow grinding bear moves (e.g. BTC -5% over 2 days).
    _htf_bear_debug = None
    _htf_bear_triggered = False
    _htf_bear_slope_pct = None
    if higher_klines and len(higher_klines) >= 55:
        htf_closes = [k["close"] for k in higher_klines]
        htf_ema21  = _ema(htf_closes, 21)
        htf_ema50  = _ema(htf_closes, 50)
        htf_bull_cross = htf_ema21[-1] > htf_ema50[-1]
        _slope_n = min(8, len(htf_ema21) - 1)
        _htf_slope = (htf_ema21[-1] - htf_ema21[-1 - _slope_n]) / max(htf_ema50[-1], 1e-9)
        _htf_bear_slope_pct = _htf_slope * 100
        # EMA21 dropping >0.3% = bear momentum even without full crossover (was 0.5%)
        _htf_bear_early = _htf_slope < -0.003
        _htf_bear_debug = (
            f"Early bear check: EMA21 change over last {_slope_n} candles = "
            f"{_htf_bear_slope_pct:+.3f}% (triggers at -0.30%)"
        )
        _htf_bear_triggered = _htf_bear_early
        htf_bull = htf_bull_cross and not _htf_bear_early
        htf_ok   = True
    else:
        htf_bull = ema21[-1] > ema50[-1]   # fallback: local EMA21/50 (faster than ema50/200)
        htf_ok   = False

    desired_side: Side = "LONG" if htf_bull else "SHORT"

    # ── Direction-dependent RSI shift ─────────────────────────────────────
    # LONG setups enter near oversold, so shift window lower.
    # SHORT setups enter near overbought, so shift window higher.
    # SWING uses a smaller shift — 4h bull trends sustain RSI 65-72 normally,
    # so a full -8 shift would block valid continuation entries.
    # SCALP uses full ±8 — 15m candles are noisy, tighter RSI = better entries.
    _shift_magnitude = {
        "SCALP":     8,
        "DAY_TRADE": 6,
        "SWING":     3,
    }.get(trade_style, 6)
    _rsi_shift = _shift_magnitude if desired_side == "SHORT" else -_shift_magnitude
    p["rsi_min"] = max(25, min(65, p["rsi_min"] + _rsi_shift))
    p["rsi_max"] = max(40, min(85, p["rsi_max"] + _rsi_shift))

    local_bull = ema21[-1] > ema50[-1]   # EMA21/50 cross — 4× faster than ema50/ema200
    ema9_aligned = (ema9[-1] > ema21[-1]) if desired_side == "LONG" else (ema9[-1] < ema21[-1])
    # Signal label: HTF trend direction + local EMA9 momentum alignment
    htf_label = "4h-BULL" if htf_bull else "4h-BEAR"
    ema9_label = "EMA9↑" if ema9[-1] > ema21[-1] else "EMA9↓"
    sig = f"{htf_label} {ema9_label}"

    # ── Layer 1: Regime — ADX strength + ATR volatility ──────────────────
    adx_ok  = adx is not None and adx >= p["adx_min"]
    atr_ok  = atr_pct > 0 and p["atr_min"] <= atr_pct <= p["atr_max"]
    adx_score = min(1.0, (adx - p["adx_min"]) / max(1.0, p["adx_min"])) if adx_ok else 0.0
    atr_score = 1.0 if atr_ok else 0.0
    regime_score = round(adx_score * 0.70 + atr_score * 0.30, 3)
    reg_reason = (
        f"ADX {adx:.1f} < {p['adx_min']:.0f} — market too choppy" if not adx_ok and adx
        else f"ATR {atr_pct*100:.2f}% outside safe range {p['atr_min']*100:.1f}–{p['atr_max']*100:.0f}%" if not atr_ok
        else ""
    )
    breakdown_regime = {
        "adx": round(adx or 0, 1), "adx_min": round(p["adx_min"], 1),
        "atr_pct": round(atr_pct * 100, 3),
        "atr_range": f"{p['atr_min']*100:.1f}–{p['atr_max']*100:.0f}%",
        "score": regime_score, "ok": regime_score > 0.10, "reason": reg_reason,
    }

    # ── Layer 2: Direction — higher TF trend + local confirmation + spread ──
    #
    # BUG FIXED: htf_aligned was a tautology.
    #   desired_side = "LONG" if htf_bull else "SHORT"
    #   htf_aligned  = htf_bull if desired_side=="LONG" else not htf_bull
    #   → always True regardless of market — htf_score was always 1.0.
    #
    # Fix: replace the self-referential check with an actual quality measure.
    # htf_score now reflects HOW STRONG the trend is (EMA spread size),
    # not whether the trend agrees with itself (which is trivially always true).
    local_aligned = (local_bull and price > ema50[-1]) if desired_side == "LONG" else (not local_bull and price < ema50[-1])

    if htf_ok:
        # Real 4h data: score by EMA21-EMA50 separation as % of price.
        # Near-zero spread = EMAs just crossed or ranging = low conviction.
        # 1.0%+ spread = established trend = full conviction.
        # BTC 4h typical: 0.1% (just crossed) → 1-3% (strong trend).
        _htf_spread = abs(htf_ema21[-1] - htf_ema50[-1]) / max(htf_ema50[-1], 1e-9)
        htf_score = min(1.0, _htf_spread / 0.010)   # 0%→0.0, 1.0%+→1.0
    else:
        # Fallback to local EMAs (less reliable) — cap at 0.5
        htf_score = 0.4

    # local_score = 0.0 when not aligned (was 0.4 — gave free credit even against trend).
    local_score   = 1.0 if local_aligned else 0.0
    ema9_bonus    = 0.20 if ema9_aligned else 0.0

    # EMA spread widening = trend accelerating = better entry quality
    spread_widening, spread_score = _ema_spread_trend(ema9, ema21, n=4)

    direction_score = round(min(1.0,
        htf_score * 0.45 + local_score * 0.25 + ema9_bonus * 0.15 + spread_score * 0.15
    ), 3)
    # Threshold raised 0.55 → 0.65: requires meaningful local + HTF agreement.
    # At 0.55, HTF alone (0.45) + neutral spread (0.075) = 0.525 could barely pass.
    # At 0.65, you need at least HTF confirmed + local OR HTF + ema9 + good spread.
    _DIR_PASS = 0.65
    dir_reason = (
        "" if direction_score >= _DIR_PASS
        else "EMA spread narrowing — trend losing momentum, wait for re-acceleration" if not spread_widening and htf_score > 0.5 and local_aligned
        else "HTF trend weak — EMAs nearly crossed, wait for stronger separation" if htf_score < 0.30
        else f"Price {'below' if desired_side == 'LONG' else 'above'} EMA50 — wait for local trend to confirm"
    )
    breakdown_direction = {
        "htf_trend": "BULL" if htf_bull else "BEAR",
        "htf_confirmed": htf_ok,
        "local_trend": "BULL" if local_bull else "BEAR",
        "ema9_momentum": "aligned" if ema9_aligned else "not aligned",
        "ema_spread": "widening" if spread_widening else "narrowing",
        "signal": sig, "side": desired_side,
        "score": direction_score, "ok": direction_score >= _DIR_PASS, "reason": dir_reason,
    }

    # ── Layer 3: Entry — pullback + candle pattern + RSI + divergence check ──
    pb_pct = abs(price - ema21[-1]) / max(1e-9, ema21[-1])
    pullback_score = max(0.0, 1.0 - pb_pct / p["pullback_max"]) if pb_pct <= p["pullback_max"] else 0.0

    # Candle pattern at pullback zone (replaces simple bounce check)
    pattern_name, pattern_score = _candle_pattern(klines, desired_side)

    # RSI: closer to center of the range = better entry quality
    print(f"RSI check: value={rsi:.1f} zone={p['rsi_min']}-{p['rsi_max']} mode={mode} style={trade_style} direction={desired_side}", flush=True)
    rsi_in_range = rsi is not None and p["rsi_min"] <= rsi <= p["rsi_max"]
    if rsi is not None and rsi_in_range:
        mid  = (p["rsi_min"] + p["rsi_max"]) / 2.0
        half = (p["rsi_max"] - p["rsi_min"]) / 2.0
        rsi_score = max(0.25, 1.0 - abs(rsi - mid) / half)
    else:
        rsi_score = 0.0

    # RSI divergence — price extreme not confirmed by RSI = weakening setup
    div_detected, div_penalty = _rsi_divergence(closes, rsi_vals, desired_side, n=12)

    entry_score = round(
        max(0.0, pullback_score * 0.35 + pattern_score * 0.35 + rsi_score * 0.30 - div_penalty),
        3,
    )
    # against_trend: last candle closed against the trade direction at the entry zone.
    # This is a hard block — not a soft penalty. If price is pulling back INTO
    # the EMA21 zone but the last candle is still bearish on a LONG setup,
    # the entry is not ready yet. Wait for a bullish close to confirm the bounce.
    against_trend_candle = pattern_name == "against_trend"
    ent_reason = (
        f"RSI divergence — price at new extreme but RSI disagrees, trend may be weakening" if div_detected
        else f"Price {pb_pct*100:.1f}% from EMA21 — need <{p['pullback_max']*100:.1f}% to enter" if pullback_score == 0
        else f"RSI {rsi:.0f} outside entry zone {p['rsi_min']}–{p['rsi_max']}" if not rsi_in_range
        else f"Candle closed against trade direction — wait for {'bullish' if desired_side == 'LONG' else 'bearish'} close to confirm bounce" if against_trend_candle
        else f"Candle pattern weak ({pattern_name}) — wait for pin bar or engulfing at EMA21" if pattern_score < 0.40
        else ""
    )
    breakdown_entry = {
        "price_vs_ema21_pct": round(pb_pct * 100, 2),
        "pullback_max_pct":   round(p["pullback_max"] * 100, 1),
        "candle_pattern":     pattern_name,
        "pattern_score":      round(pattern_score, 2),
        "rsi":                round(rsi or 0, 1),
        "rsi_range":          f"{p['rsi_min']}–{p['rsi_max']}",
        "rsi_divergence":     div_detected,
        # against_trend is a hard fail regardless of RSI/pullback quality
        "score": entry_score, "ok": entry_score > 0.25 and not against_trend_candle, "reason": ent_reason,
    }

    # ── Layer 4: Momentum — candle alignment + volume confirmation ────────
    n = p["mom_n"]
    candles_ok = all(
        (klines[-(i + 1)]["close"] > klines[-(i + 1)]["open"]) == (desired_side == "LONG")
        for i in range(n)
    )
    candle_score = 1.0 if candles_ok else 0.0
    vol_score    = min(1.0, vol_ratio / (p["vol_factor"] * 1.5)) if vol_ratio >= p["vol_factor"] else vol_ratio / p["vol_factor"] * 0.4
    momentum_score = round(candle_score * 0.55 + vol_score * 0.45, 3)
    # Hard block: volume below 0.15x average = dead market, fake move — never trade
    # 0.30 was too strict — crypto regularly trades at 0.15-0.28x during Asian session
    vol_too_low = vol_ratio < 0.10
    mom_reason = (
        f"Volume {vol_ratio:.2f}x average — too low (min 0.10x), likely dead market" if vol_too_low
        else f"Last {n} candle(s) not {('bullish' if desired_side == 'LONG' else 'bearish')} — no momentum yet" if not candles_ok
        else f"Volume {vol_ratio:.1f}x average — need {p['vol_factor']:.1f}x minimum" if vol_ratio < p["vol_factor"]
        else ""
    )
    if vol_too_low:
        momentum_score = 0.0
    breakdown_momentum = {
        "candles_n": n, "candles_ok": candles_ok,
        "volume_ratio": round(vol_ratio, 2), "vol_factor": round(p["vol_factor"], 2),
        "score": momentum_score, "ok": momentum_score > 0.20, "reason": mom_reason,
    }

    breakdown = {
        "regime":       breakdown_regime,
        "direction":    breakdown_direction,
        "entry":        breakdown_entry,
        "momentum":     breakdown_momentum,
        "market_regime": {
            "regime":      regime_class["regime"],
            "spike_ratio": regime_class["spike_ratio"],
            "adx":         regime_class["adx"],
            "ok":          not regime_class["block"],
            "reason":      regime_class["reason"],
            "score":       1.0 if regime_class["regime"] == "TRENDING" else (0.7 if regime_class["regime"] == "MARGINAL" else 0.0),
        },
    }

    # ── Final weighted score ──────────────────────────────────────────────
    # sess_score scales the total: 1.0 = full, 0.60 = needs 67% stronger signal
    # This means a news spike (high ATR, high ADX, high volume) can still
    # fire at 3am — but a mediocre setup in thin hours gets filtered out.
    raw_score = (
        regime_score    * 0.25 +
        direction_score * 0.30 +
        entry_score     * 0.30 +
        momentum_score  * 0.15
    )
    total_score = round(raw_score * sess_score, 3)
    min_score = p.get("min_score", 0.62)
    breakdown["session"] = {"label": sess_label, "quality": round(sess_score, 2), "raw_score": round(raw_score, 3), "ok": True, "reason": ""}

    # Market grade for trades that FIRE (ok=True path only):
    # A = high conviction (≥0.78) → full position
    # B = anything that passes min_score but < 0.78 → T1+T2 split
    # "C" never appears in the ok=True return — only in the blocked path below.
    # Bug fixed: scores like 0.62 in AGGRESSIVE (min=0.58) were graded "C" here,
    # then fell to the else-branch in _run_loop and fired as accidental Grade A.
    grade = "A" if total_score >= 0.78 else "B"

    # ── 15m Multi-Timeframe Momentum Confirmation ─────────────────────────
    # DAY_TRADE and SWING: check 15m chart as a 3rd confirmation layer after
    # the 4h (regime/direction) and 1h (entry) both pass.
    # SCALP already uses 15m as its primary TF — no extra 15m layer needed.
    # Confirms  → score +0.03 bonus, full size.
    # Not confirmed → size reduced 15%, trade still fires (soft filter only).
    mtf_confirmed = True
    mtf_size_mult = 1.0
    if mtf_klines and len(mtf_klines) >= 20 and trade_style in ("DAY_TRADE", "SWING"):
        _mtf_closes  = [k["close"]  for k in mtf_klines]
        _mtf_volumes = [k["volume"] for k in mtf_klines]
        _mtf_ema21   = _ema(_mtf_closes, min(21, len(_mtf_closes) - 1))
        _mtf_rsi     = _rsi(_mtf_closes, min(14, len(_mtf_closes) - 1))
        _mtf_price   = _mtf_closes[-1]
        _last3 = mtf_klines[-3:] if len(mtf_klines) >= 3 else mtf_klines
        _bull3 = sum(1 for c in _last3 if c["close"] > c["open"])
        _bear3 = sum(1 for c in _last3 if c["close"] < c["open"])
        _vol_up = len(_mtf_volumes) >= 3 and _mtf_volumes[-2] > _mtf_volumes[-3]
        _mtf_ema21_last = _mtf_ema21[-1] if _mtf_ema21 else _mtf_price
        if desired_side == "LONG":
            _mtf_ok = _bull3 >= 2 and _mtf_rsi > 40 and _vol_up and _mtf_price > _mtf_ema21_last
        else:
            _mtf_ok = _bear3 >= 2 and _mtf_rsi < 60 and _vol_up and _mtf_price < _mtf_ema21_last
        if _mtf_ok:
            total_score = round(min(1.0, total_score + 0.03), 3)
            mtf_confirmed = True
        else:
            mtf_size_mult = 0.85
            mtf_confirmed = False
    breakdown["mtf_15m"] = {
        "ok": mtf_confirmed,
        "confirmed": mtf_confirmed,
        "size_mult": mtf_size_mult,
        "reason": "" if mtf_confirmed else "15m not confirmed — size reduced 15% for lower conviction",
    }

    failed = [k for k, v in breakdown.items() if not v.get("ok") and k != "mtf_15m"]
    if failed or total_score < min_score:
        reasons = " | ".join(breakdown[k]["reason"] for k in failed if breakdown[k].get("reason"))
        if total_score < min_score and not failed:
            reasons = f"Signal quality {total_score:.2f} below threshold {min_score:.2f} — setup not strong enough"
        return {
            "ok": False,
            "blocked": f"BLOCKED ({', '.join(failed) or 'score'}): {reasons}",
            "signal": sig, "side": desired_side,
            "score": total_score, "min_score": min_score, "grade": "C",
            "breakdown": breakdown, "atr_pct": atr_pct,
            "market_regime": regime_class["regime"],
            "htf_bear_debug": _htf_bear_debug,
            "htf_bear_triggered": _htf_bear_triggered,
            "htf_bear_slope_pct": _htf_bear_slope_pct,
            "mtf_confirmed": mtf_confirmed,
            "mtf_size_mult": mtf_size_mult,
        }

    return {
        "ok": True, "blocked": None,
        "signal": sig, "side": desired_side,
        "score": total_score, "min_score": min_score, "grade": grade,
        "breakdown": breakdown, "atr_pct": atr_pct,
        "market_regime": regime_class["regime"],
        "htf_bear_debug": _htf_bear_debug,
        "htf_bear_triggered": _htf_bear_triggered,
        "htf_bear_slope_pct": _htf_bear_slope_pct,
        "mtf_confirmed": mtf_confirmed,
        "mtf_size_mult": mtf_size_mult,
    }

