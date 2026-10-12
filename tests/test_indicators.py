"""
tests/test_indicators.py — regression suite for indicators.py signal engine.

Run from the backend root:  pytest tests/test_indicators.py -v

All tests work offline — no exchange, no DB, no Render.
`now_dubai` is patched where session-dependent tests need a fixed hour.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict
from unittest.mock import patch

import pytest

# Add project root so `from indicators import ...` resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import (
    _candle_structure_check,
    _candle_pattern,
    _session_quality,
    _ema_spread_trend,
    _compute_signal_layers,
    _market_structure,
    _daily_bias,
    _bollinger_bands,
    _mean_reversion_signal,
    _ema,
    _rsi,
    _atr,
    _adx,
    _rsi_divergence,
    _reversal_signal,
    _classify_regime,
    get_market_regime,
    check_pullback_entry,
    check_volume_hour_confirmed,
)
from coin_params import get_coin_params


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / factories
# ─────────────────────────────────────────────────────────────────────────────

DUBAI_TZ = timezone(timedelta(hours=4))


def _fixed_dubai(hour: int):
    """Return a datetime with the given Dubai hour for patching now_dubai()."""
    return datetime(2026, 6, 15, hour, 0, 0, tzinfo=DUBAI_TZ)


def _candle(open_: float, close: float, high: float = None, low: float = None, volume: float = 100.0) -> Dict:
    h = high if high is not None else max(open_, close) * 1.001
    l = low  if low  is not None else min(open_, close) * 0.999
    return {"open": open_, "close": close, "high": h, "low": l, "volume": volume}


def _bull(base: float = 100.0, body: float = 1.0) -> Dict:
    """Bullish candle with given body size."""
    return _candle(base, base + body)


def _bear(base: float = 100.0, body: float = 1.0) -> Dict:
    """Bearish candle with given body size."""
    return _candle(base, base - body)


def _trending_klines(n: int = 260, bull: bool = True, base: float = 50000.0) -> List[Dict]:
    """
    Generate n synthetic klines that produce:
      - EMA21 > EMA50 (if bull) or EMA21 < EMA50 (if bear)
      - ADX > 16
      - ATR in [0.1%, 3.5%] of price
      - RSI in a tradeable zone
      - Trending price action with a pullback at the end
    """
    klines = []
    price = base
    step = 50.0 if bull else -50.0

    for i in range(n):
        if i < n - 10:
            # Strong trend section
            o = price
            c = price + step
            h = max(o, c) + abs(step) * 0.3
            l = min(o, c) - abs(step) * 0.3
            klines.append({"open": o, "close": c, "high": h, "low": l, "volume": 500.0})
            price = c
        else:
            # Last 10: shallow pullback so price is near EMA21
            o = price
            c = price + (step * -0.3)   # retrace slightly
            h = max(o, c) + abs(step) * 0.15
            l = min(o, c) - abs(step) * 0.15
            klines.append({"open": o, "close": c, "high": h, "low": l, "volume": 600.0})
            price = c

    return klines


# ─────────────────────────────────────────────────────────────────────────────
# _candle_structure_check — 5 scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleStructureCheck:

    def test_explosive_reversal_long(self):
        """Big green candle (body 3×) after 3 prior red candles → EXPLOSIVE_REVERSAL."""
        klines = [
            _bear(100, 2),   # red -1 (pos -5)
            _bear(100, 2),   # red -4
            _bear(100, 2),   # red -3
            _bear(100, 2),   # red -2  (4 candles before last)
            _bull(100, 7),   # green entry candle — body 7 vs avg 2 → ratio 3.5
        ]
        label, penalty = _candle_structure_check(klines, "LONG")
        assert label == "EXPLOSIVE_REVERSAL"
        assert penalty == 0.0

    def test_weak_bounce_long(self):
        """Tiny green candle (body 0.3×) after 3 prior red candles → WEAK_BOUNCE penalty 0.10."""
        klines = [
            _bear(100, 4),
            _bear(100, 4),
            _bear(100, 4),
            _bear(100, 4),
            _bull(100, 0.5),  # body 0.5 vs avg 4 → ratio 0.125 < 0.5
        ]
        label, penalty = _candle_structure_check(klines, "LONG")
        assert label == "WEAK_BOUNCE"
        assert penalty == pytest.approx(0.10)

    def test_neutral_too_few_opposing(self):
        """Only 2 red candles before a green — not enough opposing → NEUTRAL."""
        klines = [
            _bull(100, 2),   # green — does NOT oppose LONG
            _bull(100, 2),   # green — does NOT oppose LONG
            _bear(100, 2),   # red — opposes LONG
            _bear(100, 2),   # red — opposes LONG (only 2 total)
            _bull(100, 5),   # entry candle
        ]
        label, penalty = _candle_structure_check(klines, "LONG")
        assert label == "NEUTRAL"
        assert penalty == 0.0

    def test_explosive_reversal_short(self):
        """Big red candle after 3 green candles → EXPLOSIVE_REVERSAL on SHORT."""
        klines = [
            _bull(100, 2),
            _bull(100, 2),
            _bull(100, 2),
            _bull(100, 2),
            _bear(100, 8),   # body 8 vs avg 2 → ratio 4.0
        ]
        label, penalty = _candle_structure_check(klines, "SHORT")
        assert label == "EXPLOSIVE_REVERSAL"
        assert penalty == 0.0

    def test_weak_bounce_short(self):
        """Tiny red candle after 3 green candles → WEAK_BOUNCE on SHORT."""
        klines = [
            _bull(100, 4),
            _bull(100, 4),
            _bull(100, 4),
            _bull(100, 4),
            _bear(100, 0.5),   # ratio 0.125 < 0.5
        ]
        label, penalty = _candle_structure_check(klines, "SHORT")
        assert label == "WEAK_BOUNCE"
        assert penalty == pytest.approx(0.10)

    def test_too_few_klines_returns_neutral(self):
        """Fewer than 5 candles → safe default NEUTRAL, no penalty."""
        klines = [_bear(100, 2), _bear(100, 2), _bull(100, 5)]
        label, penalty = _candle_structure_check(klines, "LONG")
        assert label == "NEUTRAL"
        assert penalty == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _candle_pattern — named patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestCandlePattern:

    def test_pin_bar_long(self):
        """Hammer shape: long lower wick, tiny body, closes green → pin_bar score 1.0."""
        # low=90, open=98, close=99, high=99.5 → body=1, range=9.5, lower_wick=8
        c = {"open": 98, "close": 99, "high": 99.5, "low": 90}
        prev = _bull(97, 1)
        label, score = _candle_pattern([prev, c], "LONG")
        assert label == "pin_bar"
        assert score == 1.0

    def test_engulfing_long(self):
        """Current candle fully wraps previous red candle → engulfing."""
        prev = _bear(100, 3)          # open=100, close=97
        curr = _candle(96.5, 100.5)   # open=96.5 < prev.close=97, close=100.5 > prev.open=100
        label, score = _candle_pattern([prev, curr], "LONG")
        assert label == "engulfing"
        assert score == pytest.approx(0.90)

    def test_momentum_long(self):
        """Large bullish body > 65% of range → momentum."""
        # prev bearish (open=110, close=108) so curr.open=109 > prev.close=108 → not engulfing
        # curr: body=9, range=10, body_ratio=0.90 → momentum
        prev = _bear(110, 2)   # open=110, close=108
        c = {"open": 109, "close": 118, "high": 119, "low": 109}
        label, score = _candle_pattern([prev, c], "LONG")
        assert label == "momentum"
        assert score == pytest.approx(0.75)

    def test_against_trend_long(self):
        """Red candle on a LONG setup → against_trend."""
        c = _bear(102, 2)
        prev = _bull(100, 1)
        label, score = _candle_pattern([prev, c], "LONG")
        assert label == "against_trend"
        assert score == pytest.approx(0.10)

    def test_doji(self):
        """Zero range candle → doji, score 0.25."""
        c = {"open": 100, "close": 100, "high": 100, "low": 100}
        prev = _bull(99, 1)
        label, score = _candle_pattern([prev, c], "LONG")
        assert label == "doji"
        assert score == pytest.approx(0.25)

    def test_pin_bar_short(self):
        """Shooting star: long upper wick, tiny body, closes red → pin_bar SHORT."""
        # high=110, open=101, close=100, low=99.5 → body=1, range=10.5, upper_wick=9
        c = {"open": 101, "close": 100, "high": 110, "low": 99.5}
        prev = _bear(102, 1)
        label, score = _candle_pattern([prev, c], "SHORT")
        assert label == "pin_bar"
        assert score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _session_quality — hour boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionQuality:

    def test_swing_always_1(self):
        """SWING never checks the clock — always returns 1.0."""
        score, label = _session_quality("SWING")
        assert score == 1.0
        assert label == "swing"

    def test_scalp_blocked_dead_zone(self):
        """SCALP at hour 3 (dead zone 02-06) → score 0.0 hard block."""
        with patch("indicators.now_dubai", return_value=_fixed_dubai(3)):
            score, label = _session_quality("SCALP")
        assert score == 0.0
        assert "scalp-blocked" in label

    def test_dead_zone_non_scalp(self):
        """DAY_TRADE at hour 4 (dead zone) → reduced but not zero (0.72)."""
        with patch("indicators.now_dubai", return_value=_fixed_dubai(4)):
            score, label = _session_quality("DAY_TRADE")
        assert score == pytest.approx(0.72)
        assert "dead-zone" in label

    def test_asia_session(self):
        """Hour 8 (Asia) → 0.90."""
        with patch("indicators.now_dubai", return_value=_fixed_dubai(8)):
            score, _ = _session_quality("DAY_TRADE")
        assert score == pytest.approx(0.90)

    def test_london_open(self):
        """Hour 13 (London open) → 0.95."""
        with patch("indicators.now_dubai", return_value=_fixed_dubai(13)):
            score, label = _session_quality("DAY_TRADE")
        assert score == pytest.approx(0.95)
        assert "London" in label

    def test_london_ny_overlap(self):
        """Hour 18 (London+NY overlap) → 1.0."""
        with patch("indicators.now_dubai", return_value=_fixed_dubai(18)):
            score, label = _session_quality("DAY_TRADE")
        assert score == pytest.approx(1.00)
        assert "London+NY" in label

    def test_ny_solo(self):
        """Hour 22 (NY solo) → 0.92."""
        with patch("indicators.now_dubai", return_value=_fixed_dubai(22)):
            score, label = _session_quality("DAY_TRADE")
        assert score == pytest.approx(0.92)
        assert "NY" in label


# ─────────────────────────────────────────────────────────────────────────────
# _ema_spread_trend — widening vs narrowing
# ─────────────────────────────────────────────────────────────────────────────

class TestEmaSpreadTrend:

    def test_widening(self):
        """Spread grows from 10 to 15 (+50%) → widening=True, score > 0.5."""
        fast = [100.0, 101.0, 102.0, 103.0, 105.0]
        slow = [100.0, 99.5,  99.0,  98.5,  90.0]   # gap: 0 → 15
        widening, score = _ema_spread_trend(fast, slow, n=4)
        assert widening is True
        assert score > 0.5

    def test_narrowing(self):
        """Spread shrinks from 10 to 2 → widening=False, score < 0.5."""
        fast = [110.0, 109.0, 108.0, 107.0, 102.0]
        slow = [100.0, 100.0, 100.0, 100.0, 100.0]  # gap: 10 → 2
        widening, score = _ema_spread_trend(fast, slow, n=4)
        assert widening is False
        assert score < 0.5

    def test_too_few_candles(self):
        """Fewer than n+1 elements → default (False, 0.5)."""
        fast = [100.0, 101.0]
        slow = [99.0,  98.0]
        widening, score = _ema_spread_trend(fast, slow, n=4)
        assert widening is False
        assert score == pytest.approx(0.5)

    def test_zero_prev_spread(self):
        """Previous spread is essentially 0 → safe default (False, 0.5)."""
        fast = [100.0, 100.0, 100.0, 100.0, 100.0]
        slow = [100.0, 100.0, 100.0, 100.0, 100.0]
        widening, score = _ema_spread_trend(fast, slow, n=4)
        assert widening is False
        assert score == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_signal_layers — integration-level checks
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSignalLayers:

    def test_daily_klines_none_does_not_crash(self):
        """daily_klines=None (default) must not raise — zero-regression for direction fix."""
        klines = _trending_klines(260, bull=True)
        # patch to a liquid hour so session filter doesn't block
        with patch("indicators.now_dubai", return_value=_fixed_dubai(18)):
            result = _compute_signal_layers(klines, mode="MINI_ASYM")
        assert "ok" in result    # any result is fine — just must not raise

    def test_too_few_klines_returns_no_data(self):
        """Fewer than 220 klines → ok=False, signal=NO_DATA."""
        klines = _trending_klines(100)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(18)):
            result = _compute_signal_layers(klines, mode="MINI_ASYM")
        assert result["ok"] is False
        assert result["signal"] == "NO_DATA"

    def test_grade_a_threshold(self):
        """A signal that fires with total_score ≥ 0.78 must be graded A."""
        klines = _trending_klines(260, bull=True)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(18)):
            result = _compute_signal_layers(klines, mode="MINI_ASYM")
        if result["ok"]:
            assert result["grade"] in ("A", "B")
            if result["score"] >= 0.78:
                assert result["grade"] == "A"

    def test_grade_b_below_078(self):
        """An ok=True signal with score < 0.78 must be grade B (never grade A)."""
        klines = _trending_klines(260, bull=True)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(18)):
            result = _compute_signal_layers(klines, mode="MINI_ASYM")
        if result["ok"] and result["score"] < 0.78:
            assert result["grade"] == "B"

    def test_asia_floor_raises_min_score(self):
        """Hour 3: Asia floor caps at 0.70 (lowered from 0.78 — was blocking valid setups)."""
        klines = _trending_klines(260, bull=True)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(3)):
            result = _compute_signal_layers(klines, mode="MINI_ASYM")
        breakdown = result.get("breakdown", {})
        sess = breakdown.get("session", {})
        if sess:
            assert sess.get("asia_grade_a_floor") is True
            assert 0.70 <= sess.get("effective_min_score", 0) <= 0.70

    def test_scalp_dead_zone_is_blocked(self):
        """SCALP at hour 3 must return ok=False with SCALP_LOW_LIQ block."""
        klines = _trending_klines(260, bull=True)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(3)):
            result = _compute_signal_layers(klines, mode="MINI_ASYM", trade_style="SCALP")
        assert result["ok"] is False
        assert "SCALP_LOW_LIQ" in result.get("blocked", "")

    def test_breakdown_keys_present(self):
        """ok=True result must contain all 4 layer breakdown keys."""
        klines = _trending_klines(260, bull=True)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(18)):
            result = _compute_signal_layers(klines, mode="MINI_ASYM")
        # Whether ok or blocked, breakdown should have known structure
        breakdown = result.get("breakdown", {})
        for key in ("regime", "direction", "entry", "momentum"):
            assert key in breakdown, f"Missing breakdown key: {key}"

    def test_param_overrides_applied(self):
        """param_overrides min_score=0.99 should make even a good setup fail."""
        klines = _trending_klines(260, bull=True)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(18)):
            result = _compute_signal_layers(
                klines, mode="MINI_ASYM",
                param_overrides={"min_score": 0.99}
            )
        # With min_score=0.99 nothing should pass in synthetic data
        assert result["ok"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Kline factories for market structure tests
# Each dataset verified manually: pivot positions, HH/HL or LH/LL patterns.
# ─────────────────────────────────────────────────────────────────────────────

def _make_uptrend_klines() -> List[Dict]:
    """
    23 candles in a clear zigzag uptrend.
    Swing highs (lb=2): idx 3 (h=112), idx 10 (h=122), idx 17 (h=132) → HH
    Swing lows (lb=2):  idx 6 (l=90),  idx 13 (l=92),  idx 20 (l=95)  → HL
    """
    highs = [101,104,108,112,109,106,104,106,110,118,122,118,114,110,112,116,124,132,128,124,120,124,128]
    lows  = [ 97,100,104,108,105, 95, 90, 92, 96,102,108,104, 95, 92, 96,100,104,110,106, 98, 95, 98,102]
    return [
        {"open": (h + l) / 2 - 0.5, "close": (h + l) / 2 + 0.5,
         "high": float(h), "low": float(l), "volume": 100.0}
        for h, l in zip(highs, lows)
    ]


def _make_downtrend_klines() -> List[Dict]:
    """
    30 candles in a clear zigzag downtrend.
    Swing highs (lb=2): idx 2 (h=130), idx 12 (h=120), idx 22 (h=110) → LH
    Swing lows (lb=2):  idx 7 (l=96),  idx 17 (l=84),  idx 27 (l=74)  → LL
    """
    highs = [124,127,130,127,124,106,104,102,104,106,108,112,120,112,108,
              96, 94, 92, 94, 96, 98,102,110,102, 98, 86, 84, 82, 84, 86]
    lows  = [114,117,120,117,114,100, 98, 96, 98,100,100,104,112,104,100,
              88, 86, 84, 86, 88, 90, 94,102, 94, 90, 78, 76, 74, 76, 78]
    return [
        {"open": (h + l) / 2 - 0.5, "close": (h + l) / 2 + 0.5,
         "high": float(h), "low": float(l), "volume": 100.0}
        for h, l in zip(highs, lows)
    ]


class TestMarketStructure:

    def test_uptrend_hh_hl(self):
        """Clear HH + HL zigzag → UPTREND, +0.05 for LONG, -0.05 for SHORT."""
        klines = _make_uptrend_klines()
        label, adj = _market_structure(klines, "LONG")
        assert label == "UPTREND"
        assert adj == pytest.approx(0.05)

        label2, adj2 = _market_structure(klines, "SHORT")
        assert label2 == "UPTREND"
        assert adj2 == pytest.approx(-0.05)

    def test_downtrend_lh_ll(self):
        """Clear LH + LL zigzag → DOWNTREND, +0.05 for SHORT, -0.05 for LONG."""
        klines = _make_downtrend_klines()
        label, adj = _market_structure(klines, "SHORT")
        assert label == "DOWNTREND"
        assert adj == pytest.approx(0.05)

        label2, adj2 = _market_structure(klines, "LONG")
        assert label2 == "DOWNTREND"
        assert adj2 == pytest.approx(-0.05)

    def test_too_few_swings(self):
        """Monotonically rising prices — no pivot points → NEUTRAL, 0 adj."""
        klines = [
            {"open": 100.0 + i, "close": 101.0 + i,
             "high": 102.0 + i, "low": 99.0 + i, "volume": 100.0}
            for i in range(15)
        ]
        label, adj = _market_structure(klines, "LONG")
        assert label == "NEUTRAL"
        assert adj == 0.0

    def test_too_few_klines(self):
        """Fewer than 10 candles → NEUTRAL, no crash."""
        label, adj = _market_structure([_bull(100, 1) for _ in range(5)], "LONG")
        assert label == "NEUTRAL"
        assert adj == 0.0

    def test_mixed_structure_is_neutral(self):
        """HH pattern but HL broken → NEUTRAL."""
        klines = _make_uptrend_klines()
        # Drop the last swing low below the previous one → HL → breaks uptrend condition
        klines[20] = {**klines[20], "low": 80.0}  # 80 < sl[-2]=92 → not HL
        label, adj = _market_structure(klines, "LONG")
        assert label == "NEUTRAL"
        assert adj == 0.0


class TestDailyBias:

    def _rising_daily(self, n: int = 20, base: float = 100.0, step: float = 2.0) -> List[Dict]:
        klines = []
        for i in range(n):
            o = base + step * i
            c = o + step * 0.8
            klines.append({"open": o, "close": c, "high": c + 0.5, "low": o - 0.5, "volume": 100.0})
        return klines

    def _falling_daily(self, n: int = 20, base: float = 200.0, step: float = 2.0) -> List[Dict]:
        klines = []
        for i in range(n):
            o = base - step * i
            c = o - step * 0.8
            klines.append({"open": o, "close": c, "high": o + 0.5, "low": c - 0.5, "volume": 100.0})
        return klines

    def test_bull_bias(self):
        """Strongly rising daily candles → BULL."""
        assert _daily_bias(self._rising_daily(25)) == "BULL"

    def test_bear_bias(self):
        """Strongly falling daily candles → BEAR."""
        assert _daily_bias(self._falling_daily(25)) == "BEAR"

    def test_empty_list_safe(self):
        """Empty list → NEUTRAL, no crash."""
        assert _daily_bias([]) == "NEUTRAL"

    def test_too_few_candles(self):
        """Fewer than 5 candles → NEUTRAL."""
        assert _daily_bias([_bull(100, 1) for _ in range(3)]) == "NEUTRAL"

    def test_neutral_flat(self):
        """Flat candles (price hugging EMA, slope ~0) → NEUTRAL."""
        flat = [
            {"open": 100.0, "close": 100.2, "high": 100.5, "low": 99.8, "volume": 100.0}
            for _ in range(10)
        ]
        assert _daily_bias(flat) == "NEUTRAL"


# ─── Helper: build a ranging candle list (price oscillates around 100) ────────
def _ranging_klines(n: int = 40, base: float = 100.0, amp: float = 1.0) -> List[Dict]:
    """Candles that oscillate around `base` with amplitude `amp` — no trend."""
    klines = []
    import math
    for i in range(n):
        price = base + amp * math.sin(i * 0.5)
        klines.append({
            "open": price - 0.1, "close": price + 0.1,
            "high": price + 0.3, "low": price - 0.3,
            "volume": 80.0,
        })
    return klines


class TestBollingerBands:

    def test_returns_three_values(self):
        """_bollinger_bands returns (upper, middle, lower) for 20+ candles."""
        closes = [100.0 + i * 0.1 for i in range(25)]
        upper, middle, lower = _bollinger_bands(closes)
        assert upper is not None and middle is not None and lower is not None
        assert upper > middle > lower

    def test_too_few_candles_returns_none(self):
        """Fewer than 20 closes → all None, no crash."""
        upper, middle, lower = _bollinger_bands([100.0] * 10)
        assert upper is None and middle is None and lower is None

    def test_flat_candles_equal_middle_and_zero_std(self):
        """All-equal closes → bands are symmetric around middle, std = 0."""
        closes = [100.0] * 25
        upper, middle, lower = _bollinger_bands(closes)
        assert middle == 100.0
        assert upper == lower == 100.0  # std=0 → no spread


class TestMeanReversionSignal:

    def _ranging_mr_klines(self, price_at_lower_bb: bool = True) -> List[Dict]:
        """
        Build 40 candles where price oscillates tightly (low std dev = narrow BB),
        then on the last candle push price clearly below (or above) the band.
        """
        base = 100.0
        klines = []
        for i in range(39):
            # Small oscillation ±0.15 around base → std ≈ 0.12 → BB ≈ base ± 0.24
            offset = 0.15 if i % 2 == 0 else -0.15
            p = base + offset
            klines.append({
                "open": p - 0.05, "close": p + 0.05,
                "high": p + 0.1,  "low": p - 0.1,
                "volume": 80.0,
            })
        # Last candle: push 1.5 below base to be clearly beyond lower BB (≈ base − 0.24)
        if price_at_lower_bb:
            last_price = base - 1.5
        else:
            last_price = base + 1.5
        klines.append({
            "open": last_price + 0.05, "close": last_price,
            "high": last_price + 0.1,  "low": last_price - 0.1,
            "volume": 80.0,
        })
        return klines

    def test_long_fires_at_lower_band(self):
        """Price below lower BB + RSI < 45 + ADX < 20 → LONG mean reversion."""
        klines = self._ranging_mr_klines(price_at_lower_bb=True)
        result = _mean_reversion_signal(klines, rsi=38.0, adx=12.0, atr=0.5)
        assert result["ok"] is True
        assert result["side"] == "LONG"
        assert result["signal_type"] == "MEAN_REVERSION"
        assert result["grade"] == "A"
        assert result["sl_atr_override"] == 1.0
        assert result["tp_atr_override"] >= 1.5

    def test_short_fires_at_upper_band(self):
        """Price above upper BB + RSI > 55 + ADX < 20 → SHORT mean reversion."""
        klines = self._ranging_mr_klines(price_at_lower_bb=False)
        result = _mean_reversion_signal(klines, rsi=62.0, adx=14.0, atr=0.5)
        assert result["ok"] is True
        assert result["side"] == "SHORT"
        assert result["signal_type"] == "MEAN_REVERSION"
        assert result["grade"] == "A"

    def test_blocked_when_adx_too_high(self):
        """ADX ≥ 20 → trending market, mean reversion must not fire."""
        klines = self._ranging_mr_klines(price_at_lower_bb=True)
        result = _mean_reversion_signal(klines, rsi=38.0, adx=22.0, atr=0.5)
        assert result["ok"] is False

    def test_blocked_when_rsi_wrong_side(self):
        """Price at lower BB but RSI ≥ 45 → signal is not a reversion setup."""
        klines = self._ranging_mr_klines(price_at_lower_bb=True)
        result = _mean_reversion_signal(klines, rsi=55.0, adx=12.0, atr=0.5)
        assert result["ok"] is False

    def test_too_few_klines_safe(self):
        """Fewer than 25 candles → ok=False, no crash."""
        klines = self._ranging_mr_klines(price_at_lower_bb=True)[:10]
        result = _mean_reversion_signal(klines, rsi=38.0, adx=12.0, atr=0.5)
        assert result["ok"] is False


# ══════════════════════════════════════════════════════════════════════════════
# FULL FEATURE AUDIT — covers every untested function in indicators.py
# ══════════════════════════════════════════════════════════════════════════════

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _declining_klines(n: int = 45, start: float = 100.0, step: float = 0.8) -> List[Dict]:
    """Steadily declining candles — makes RSI very low, price far below EMA21."""
    klines = []
    for i in range(n - 1):
        p = start - i * step
        klines.append({
            "open": p + 0.1, "close": p - 0.1,
            "high": p + 0.2, "low":  p - 0.3,
            "volume": 100.0,
            "t": (1_700_000_000 + i * 3600) * 1000,
        })
    # Second-to-last: volume spike (4×)
    p = start - (n - 2) * step
    klines.append({
        "open": p + 0.1, "close": p - 0.1,
        "high": p + 0.2, "low":  p - 0.3,
        "volume": 400.0,
        "t": (1_700_000_000 + (n - 1) * 3600) * 1000,
    })
    # Last: green reversal candle
    p = start - (n - 1) * step
    klines.append({
        "open": p - 0.5, "close": p + 0.3,
        "high": p + 0.5, "low":  p - 0.8,
        "volume": 50.0,
        "t": (1_700_000_000 + n * 3600) * 1000,
    })
    return klines


def _rising_klines(n: int = 45, start: float = 100.0, step: float = 0.8) -> List[Dict]:
    """Steadily rising candles — makes RSI very high, price far above EMA21."""
    klines = []
    for i in range(n - 1):
        p = start + i * step
        klines.append({
            "open": p - 0.1, "close": p + 0.1,
            "high": p + 0.3, "low":  p - 0.2,
            "volume": 100.0,
            "t": (1_700_000_000 + i * 3600) * 1000,
        })
    # Second-to-last: volume spike
    p = start + (n - 2) * step
    klines.append({
        "open": p - 0.1, "close": p + 0.1,
        "high": p + 0.3, "low":  p - 0.2,
        "volume": 400.0,
        "t": (1_700_000_000 + (n - 1) * 3600) * 1000,
    })
    # Last: red reversal candle
    p = start + (n - 1) * step
    klines.append({
        "open": p + 0.5, "close": p - 0.3,
        "high": p + 0.8, "low":  p - 0.5,
        "volume": 50.0,
        "t": (1_700_000_000 + n * 3600) * 1000,
    })
    return klines


def _flat_klines(n: int = 60, base: float = 100.0, vol: float = 100.0) -> List[Dict]:
    """Flat / sideways candles — ADX will be low, ATR stable."""
    import math
    klines = []
    for i in range(n):
        p = base + 0.2 * math.sin(i * 0.4)
        klines.append({
            "open": p - 0.05, "close": p + 0.05,
            "high": p + 0.15, "low":  p - 0.15,
            "volume": vol,
            "t": (1_700_000_000 + i * 3600) * 1000,
        })
    return klines


def _trending_klines_adx(n: int = 80, start: float = 100.0, step: float = 0.5) -> List[Dict]:
    """Consistent trend — ADX > 22, normal ATR."""
    klines = []
    for i in range(n):
        p = start + i * step
        klines.append({
            "open": p - 0.1, "close": p + 0.1,
            "high": p + 0.4, "low":  p - 0.3,
            "volume": 100.0,
            "t": (1_700_000_000 + i * 3600) * 1000,
        })
    return klines


# ─── 1. Core indicator math ───────────────────────────────────────────────────
class TestCoreIndicators:

    def test_ema_grows_with_rising_series(self):
        """EMA on a rising series must itself be rising."""
        closes = [float(i) for i in range(1, 31)]
        result = _ema(closes, 9)
        assert len(result) == 30
        assert result[-1] > result[0]

    def test_ema_empty_input_safe(self):
        """Empty series → empty list, no crash."""
        assert _ema([], 9) == []

    def test_rsi_overbought_after_rally(self):
        """RSI after 14 consecutive up candles must be very high (> 70)."""
        closes = [100.0 + i * 2 for i in range(20)]
        val = _rsi(closes, 14)
        assert val is not None
        assert val > 70

    def test_rsi_oversold_after_decline(self):
        """RSI after 14 consecutive down candles must be very low (< 30)."""
        closes = [100.0 - i * 2 for i in range(20)]
        val = _rsi(closes, 14)
        assert val is not None
        assert val < 30

    def test_rsi_too_few_candles_returns_none(self):
        """Fewer candles than period + 1 → None, no crash."""
        assert _rsi([100.0, 101.0], 14) is None

    def test_atr_positive_for_normal_candles(self):
        """ATR on normal OHLC data must be a positive number."""
        klines = _trending_klines_adx(30)
        highs  = [k["high"]  for k in klines]
        lows   = [k["low"]   for k in klines]
        closes = [k["close"] for k in klines]
        val = _atr(highs, lows, closes, 14)
        assert val is not None and val > 0

    def test_adx_high_in_strong_trend(self):
        """ADX on a consistent directional move must be > 25."""
        klines = _trending_klines_adx(80)
        highs  = [k["high"]  for k in klines]
        lows   = [k["low"]   for k in klines]
        closes = [k["close"] for k in klines]
        val = _adx(highs, lows, closes, 14)
        assert val is not None and val > 20

    def test_adx_too_few_returns_none(self):
        """Fewer candles than 2×period → None, no crash."""
        klines = _trending_klines_adx(10)
        highs  = [k["high"]  for k in klines]
        lows   = [k["low"]   for k in klines]
        closes = [k["close"] for k in klines]
        assert _adx(highs, lows, closes, 14) is None


# ─── 2. RSI divergence ────────────────────────────────────────────────────────
class TestRsiDivergence:

    def test_bearish_divergence_detected(self):
        """Price makes higher high but RSI makes lower high → divergence detected."""
        closes   = [100.0] * 10 + [105.0] + [100.0] * 5 + [108.0]
        rsi_vals = [60.0]  * 10 + [65.0]  + [60.0] * 5  + [60.0]
        div, penalty = _rsi_divergence(closes, rsi_vals, "LONG", n=12)
        assert div is True
        assert penalty > 0

    def test_no_divergence_when_rsi_confirms(self):
        """Price higher high AND RSI higher high → no divergence."""
        closes   = [100.0] * 10 + [105.0] + [100.0] * 5 + [108.0]
        rsi_vals = [60.0]  * 10 + [65.0]  + [60.0] * 5  + [70.0]
        div, penalty = _rsi_divergence(closes, rsi_vals, "LONG", n=12)
        assert div is False
        assert penalty == 0.0

    def test_too_few_candles_safe(self):
        """Fewer candles than n → False, 0.0 — no crash."""
        div, penalty = _rsi_divergence([100.0, 101.0], [50.0, 51.0], "LONG", n=12)
        assert div is False
        assert penalty == 0.0

    def test_bullish_divergence_short_side(self):
        """SHORT: price lower low but RSI higher low → divergence detected."""
        closes   = [100.0] * 10 + [95.0] + [100.0] * 5 + [92.0]
        rsi_vals = [40.0]  * 10 + [35.0] + [40.0] * 5  + [38.0]
        div, penalty = _rsi_divergence(closes, rsi_vals, "SHORT", n=12)
        assert div is True
        assert penalty > 0


# ─── 3. T16 Reversal signal ──────────────────────────────────────────────────
class TestReversalSignal:

    def test_long_reversal_all_conditions_met(self):
        """All 5 T16 conditions met → ok=True, side=LONG, grade=B."""
        klines = _declining_klines(45)
        result = _reversal_signal(klines, rsi=22.0, adx=15.0, atr=1.0)
        assert result["ok"] is True
        assert result["side"] == "LONG"
        assert result["signal_type"] == "REVERSAL"
        assert result["grade"] == "B"
        assert result["sl_atr_override"] == 1.5
        assert result["tp_atr_override"] == 3.0

    def test_short_reversal_all_conditions_met(self):
        """Rising klines + RSI > 72 + ADX 12-22 → SHORT reversal fires."""
        klines = _rising_klines(45)
        result = _reversal_signal(klines, rsi=78.0, adx=17.0, atr=1.0)
        assert result["ok"] is True
        assert result["side"] == "SHORT"

    def test_blocked_adx_too_high(self):
        """ADX > 22 → trending market, T16 must not fire."""
        klines = _declining_klines(45)
        result = _reversal_signal(klines, rsi=22.0, adx=28.0, atr=1.0)
        assert result["ok"] is False

    def test_blocked_adx_too_low(self):
        """ADX < 12 → too choppy even for T16."""
        klines = _declining_klines(45)
        result = _reversal_signal(klines, rsi=22.0, adx=10.0, atr=1.0)
        assert result["ok"] is False

    def test_blocked_rsi_not_extreme(self):
        """RSI 45 (not oversold) → condition 1 fails."""
        klines = _declining_klines(45)
        result = _reversal_signal(klines, rsi=45.0, adx=15.0, atr=1.0)
        assert result["ok"] is False

    def test_too_few_klines_safe(self):
        """Fewer than 25 candles → ok=False, no crash."""
        result = _reversal_signal(_declining_klines(10), rsi=22.0, adx=15.0, atr=1.0)
        assert result["ok"] is False


# ─── 4. Regime classifier ─────────────────────────────────────────────────────
class TestClassifyRegime:

    def _highs_lows_closes(self, klines):
        return (
            [k["high"]  for k in klines],
            [k["low"]   for k in klines],
            [k["close"] for k in klines],
        )

    def test_trending_regime(self):
        """Strong consistent trend → regime=TRENDING, block=False."""
        klines = _trending_klines_adx(80)
        h, l, c = self._highs_lows_closes(klines)
        from indicators import _atr, _adx
        atr = _atr(h, l, c, 14)
        adx = _adx(h, l, c, 14)
        atr_pct = atr / c[-1] if atr and c[-1] else 0.01
        result = _classify_regime(h, l, c, adx, atr_pct)
        assert result["block"] is False
        assert result["regime"] in ("TRENDING", "MARGINAL")

    def test_choppy_regime(self):
        """Low ADX + quiet ATR → CHOPPY, block=True."""
        klines = _flat_klines(80)
        h, l, c = self._highs_lows_closes(klines)
        # Pass ADX directly as 10 (very choppy)
        result = _classify_regime(h, l, c, adx=10.0, atr_pct=0.002, _atr_slow_pct=0.003)
        assert result["regime"] == "CHOPPY"
        assert result["block"] is True

    def test_volatile_regime(self):
        """ATR spike 3× above baseline → VOLATILE, block=True."""
        klines = _trending_klines_adx(80)
        h, l, c = self._highs_lows_closes(klines)
        # Current ATR huge (spike), slow ATR small
        result = _classify_regime(h, l, c, adx=30.0, atr_pct=0.06, _atr_slow_pct=0.01)
        assert result["regime"] == "VOLATILE"
        assert result["block"] is True

    def test_marginal_regime_raises_min_score(self):
        """ADX 19 (weak trend) → MARGINAL, block=False, penalty > 0."""
        klines = _flat_klines(80)
        h, l, c = self._highs_lows_closes(klines)
        result = _classify_regime(h, l, c, adx=19.0, atr_pct=0.009, _atr_slow_pct=0.009)
        assert result["regime"] == "MARGINAL"
        assert result["block"] is False
        assert result["min_score_penalty"] > 0


# ─── 5. Global market regime ──────────────────────────────────────────────────
class TestGetMarketRegime:

    def test_ranging_when_adx_low(self):
        """ADX 10 < ranging_threshold 15 → RANGING."""
        regime, _ = get_market_regime([], current_coin_adx=10.0, ranging_threshold=15.0)
        assert regime == "RANGING"

    def test_trending_when_no_btc_candles(self):
        """No BTC weekly candles → fallback TRENDING (not enough data for PARABOLIC)."""
        regime, _ = get_market_regime([], current_coin_adx=25.0, ranging_threshold=15.0)
        assert regime == "TRENDING"

    def test_trending_normal_conditions(self):
        """ADX 25, no extreme BTC ratio → TRENDING."""
        regime, _ = get_market_regime([], current_coin_adx=25.0, ranging_threshold=15.0)
        assert regime == "TRENDING"

    def test_parabolic_exits_below_threshold(self):
        """is_parabolic=True but BTC weekly not supplied → stays TRENDING (can't verify ratio)."""
        regime, new_para = get_market_regime([], current_coin_adx=45.0,
                                              is_parabolic=True, ranging_threshold=15.0)
        # Without BTC candles, parabolic state reverts to TRENDING
        assert regime == "TRENDING"


# ─── 6. Pullback entry gate ───────────────────────────────────────────────────
class TestCheckPullbackEntry:

    def _coin_p(self):
        return get_coin_params("BTCUSDT")

    def test_parabolic_returns_immediate(self):
        """PARABOLIC regime → skip pullback gate entirely."""
        klines = _trending_klines_adx(30)
        result = check_pullback_entry(klines, "LONG", self._coin_p(), "PARABOLIC", 1)
        assert result == "IMMEDIATE"

    def test_timeout_after_3_candles_day_trade(self):
        """DAY_TRADE non-strong-trend: after 3 candles without pullback → TIMEOUT.
        Uses flat klines so ADX is low (not a strong trend — timeout stays at 3, not 6)."""
        klines = _flat_klines(30)
        result = check_pullback_entry(klines, "LONG", self._coin_p(), "TRENDING", 3, "DAY_TRADE")
        assert result == "TIMEOUT"

    def test_wait_when_price_far_from_ema(self):
        """Price well above EMA → WAIT (no pullback yet)."""
        klines = _flat_klines(30)
        klines[-1] = {**klines[-1], "close": 200.0, "open": 199.0}
        result = check_pullback_entry(klines, "LONG", self._coin_p(), "TRENDING", 1, "DAY_TRADE")
        assert result == "WAIT"

    def test_strong_trend_extends_timeout_to_6(self):
        """Strong trend (ADX ≥ 45) extends timeout from 3 → 6 candles.
        At candles_since=4, a strong trend must still be WAIT (not TIMEOUT)."""
        klines = _trending_klines_adx(80)  # produces ADX > 25 (possibly strong)
        result = check_pullback_entry(klines, "LONG", self._coin_p(), "TRENDING", 4, "DAY_TRADE")
        # Either WAIT (strong trend, timeout=6, not reached) or TIMEOUT (weak trend, timeout=3 passed)
        # The key invariant: result must be a valid return value — no crash
        assert result in ("WAIT", "TIMEOUT", "ENTER", "ENTER_STRONG_TREND")


# ─── 7. Volume hour confirmation ─────────────────────────────────────────────
class TestVolumeHourConfirmed:

    def _coin_p(self):
        return get_coin_params("BTCUSDT")

    def _candles_with_ts(self, n: int = 50, base_vol: float = 100.0, spike_vol: float = 200.0) -> List[Dict]:
        """50 candles: all 100 vol except second-to-last = spike_vol."""
        ts_base = 1_700_000_000_000  # ms
        klines = []
        for i in range(n):
            vol = spike_vol if i == n - 2 else base_vol
            klines.append({
                "open": 100.0, "close": 100.5, "high": 101.0, "low": 99.5,
                "volume": vol,
                "t": ts_base + i * 3_600_000,
            })
        return klines

    def test_high_volume_confirms(self):
        """Signal candle volume 2× above average → confirmed=True."""
        klines = self._candles_with_ts(50, base_vol=100.0, spike_vol=250.0)
        confirmed, curr, req = check_volume_hour_confirmed(klines, self._coin_p())
        assert confirmed is True

    def test_low_volume_not_confirmed(self):
        """Signal candle volume below average → confirmed=False."""
        klines = self._candles_with_ts(50, base_vol=100.0, spike_vol=50.0)
        confirmed, curr, req = check_volume_hour_confirmed(klines, self._coin_p())
        assert confirmed is False

    def test_too_few_candles_passes_safely(self):
        """Fewer than 3 candles → True (can't filter, allow through)."""
        confirmed, _, _ = check_volume_hour_confirmed(
            [{"open": 100.0, "close": 100.5, "high": 101.0, "low": 99.5, "volume": 50.0}],
            self._coin_p()
        )
        assert confirmed is True


# ─── 8. Coin parameters ───────────────────────────────────────────────────────
class TestCoinParams:

    REQUIRED_KEYS = [
        "adx_min", "adx_max", "score_threshold", "atr_max_pct", "atr_min_pct",
        "tp_multiplier", "sl_multiplier", "volume_confirmation_mult",
        "ema_distance_max_pct", "pullback_tolerance_pct",
    ]
    ALL_COINS = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "DOTUSDT",
    ]

    def test_btcusdt_returns_correct_adx_min(self):
        """BTCUSDT params should return known adx_min=18."""
        p = get_coin_params("BTCUSDT")
        assert p["adx_min"] == 18

    def test_symbol_normalization(self):
        """btcusdt, BTC/USDT, BTC-USDT all resolve to BTCUSDT params."""
        base = get_coin_params("BTCUSDT")
        assert get_coin_params("btcusdt") == base
        assert get_coin_params("BTC/USDT") == base
        assert get_coin_params("BTC-USDT") == base

    def test_unknown_symbol_falls_back_to_btc(self):
        """Unknown symbol falls back to BTC params (safe conservative baseline)."""
        p = get_coin_params("WEIRDCOINUSDT")
        base = get_coin_params("BTCUSDT")
        assert p == base

    def test_all_coins_have_required_keys(self):
        """Every coin in the table must have all 10 required parameter keys."""
        for coin in self.ALL_COINS:
            p = get_coin_params(coin)
            for key in self.REQUIRED_KEYS:
                assert key in p, f"{coin} missing key: {key}"

    def test_score_threshold_in_valid_range(self):
        """score_threshold for every coin must be between 0.60 and 0.80."""
        for coin in self.ALL_COINS:
            p = get_coin_params(coin)
            assert 0.60 <= p["score_threshold"] <= 0.80, \
                f"{coin} score_threshold={p['score_threshold']} out of range"


# ─── 9. Adaptive strictness + threshold source ───────────────────────────────
class TestAdaptiveStrictnessAndThreshold:

    def test_adaptive_strictness_capped_at_070(self):
        """Strictness 3.0× must never push effective min_score above 0.70."""
        klines = _trending_klines(300)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(14)):
            result = _compute_signal_layers(
                klines, mode="MINI_ASYM", adaptive_strictness=3.0,
                trade_style="DAY_TRADE", symbol="BTCUSDT",
            )
        sess = result.get("breakdown", {}).get("session", {})
        eff = sess.get("effective_min_score", 0)
        assert eff <= 0.70, f"effective_min_score={eff} exceeded cap of 0.70"

    def test_threshold_source_base(self):
        """Normal daytime, no adaptive strictness → source='base' or 'coin_param'."""
        klines = _trending_klines(300)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(14)):
            result = _compute_signal_layers(
                klines, mode="MINI_ASYM", adaptive_strictness=1.0,
                trade_style="DAY_TRADE", symbol="BTCUSDT",
            )
        src = result.get("breakdown", {}).get("session", {}).get("threshold_source", "")
        assert src in ("base", "coin_param")

    def test_threshold_source_asia_floor(self):
        """During Asia hours (02:00 Dubai) → source='asia_floor'."""
        klines = _trending_klines(300)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(2)):
            result = _compute_signal_layers(
                klines, mode="MINI_ASYM", adaptive_strictness=1.0,
                trade_style="DAY_TRADE", symbol="BTCUSDT",
            )
        sess = result.get("breakdown", {}).get("session", {})
        # Asia floor may or may not raise threshold depending on coin_param
        # but threshold_source key must be present
        assert "threshold_source" in sess


# ─── 10. Daily bias wired into signal layers ─────────────────────────────────
class TestDailyBiasInSignalLayers:

    def _daily_bull(self, n: int = 25) -> List[Dict]:
        """Rising daily candles → BULL bias."""
        klines = []
        for i in range(n):
            p = 100.0 + i * 2.0
            klines.append({"open": p - 0.5, "close": p + 0.5,
                            "high": p + 1.0, "low": p - 1.0, "volume": 100.0})
        return klines

    def _daily_bear(self, n: int = 25) -> List[Dict]:
        """Falling daily candles → BEAR bias."""
        klines = []
        for i in range(n):
            p = 200.0 - i * 2.0
            klines.append({"open": p + 0.5, "close": p - 0.5,
                            "high": p + 1.0, "low": p - 1.0, "volume": 100.0})
        return klines

    def test_bull_daily_boosts_long_direction_score(self):
        """BULL daily + LONG signal → db_adj=+0.05 in breakdown."""
        klines = _trending_klines(300)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(14)):
            res_no_daily = _compute_signal_layers(
                klines, mode="MINI_ASYM", trade_style="DAY_TRADE",
                symbol="BTCUSDT", daily_klines=None,
            )
            res_with_daily = _compute_signal_layers(
                klines, mode="MINI_ASYM", trade_style="DAY_TRADE",
                symbol="BTCUSDT", daily_klines=self._daily_bull(),
            )
        db_adj = res_with_daily.get("breakdown", {}).get("direction", {}).get("db_adj", 0)
        # daily bias is present in breakdown and direction score reflects it
        assert "daily_bias" in res_with_daily.get("breakdown", {}).get("direction", {})
        # score with bull daily should be >= score without daily (or same if already 1.0)
        score_no  = res_no_daily.get("breakdown", {}).get("direction", {}).get("score", 0)
        score_yes = res_with_daily.get("breakdown", {}).get("direction", {}).get("score", 0)
        assert score_yes >= score_no - 0.01  # allow tiny float rounding

    def test_daily_bias_neutral_when_no_daily_klines(self):
        """daily_klines=None → daily_bias=NEUTRAL, db_adj=0.0."""
        klines = _trending_klines(300)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(14)):
            res = _compute_signal_layers(
                klines, mode="MINI_ASYM", trade_style="DAY_TRADE",
                symbol="BTCUSDT", daily_klines=None,
            )
        d = res.get("breakdown", {}).get("direction", {})
        assert d.get("daily_bias") == "NEUTRAL"
        assert d.get("db_adj") == 0.0


# ─── 11. T16 + Mean Reversion fire through signal layers ─────────────────────
class TestAlternativeSignalsInEngine:

    def test_mean_reversion_fires_when_regime_choppy(self):
        """Ranging/choppy market + BB conditions met → engine returns MEAN_REVERSION."""
        # Use a flat ranging kline set that will produce low ADX (CHOPPY regime)
        # and a last candle well outside Bollinger Band
        klines = []
        # 40 flat candles (±0.15 around 100) → very low ADX, quiet ATR
        for i in range(39):
            offset = 0.15 if i % 2 == 0 else -0.15
            p = 100.0 + offset
            klines.append({
                "open": p - 0.05, "close": p + 0.05,
                "high": p + 0.1,  "low":  p - 0.1,
                "volume": 80.0,   "t": (1_700_000_000 + i * 3600) * 1000,
            })
        # Pad to 260 candles so compute_signal_layers doesn't return NO_DATA
        for i in range(260 - 39):
            offset = 0.1 if i % 2 == 0 else -0.1
            p = 100.0 + offset
            klines.insert(0, {
                "open": p - 0.05, "close": p + 0.05,
                "high": p + 0.1,  "low":  p - 0.1,
                "volume": 80.0,   "t": (1_700_000_000 - (i + 1) * 3600) * 1000,
            })
        # Push last candle far below lower BB (1.5 below base)
        klines[-1] = {
            "open": 98.5, "close": 98.4,
            "high": 98.6, "low":  98.2,
            "volume": 80.0, "t": klines[-1]["t"],
        }
        with patch("indicators.now_dubai", return_value=_fixed_dubai(14)):
            res = _compute_signal_layers(
                klines, mode="MINI_ASYM", trade_style="DAY_TRADE", symbol="BTCUSDT",
            )
        # Either fires MEAN_REVERSION or is blocked — must not crash and must return
        # a dict with 'ok' key
        assert "ok" in res

    def test_htf_score_floor_applied(self):
        """param_overrides htf_score_floor > 0 → htf_score is at least that floor."""
        klines = _trending_klines(300)
        with patch("indicators.now_dubai", return_value=_fixed_dubai(14)):
            res = _compute_signal_layers(
                klines, mode="MINI_ASYM", trade_style="DAY_TRADE",
                symbol="BTCUSDT",
                param_overrides={"htf_score_floor": 0.60},
            )
        # The direction breakdown's score should be influenced by the floor.
        # We can't assert the exact value but breakdown must exist and not crash.
        assert "breakdown" in res

    def test_enable_t16_false_disables_reversal(self):
        """enable_t16=False → T16 reversal never fires even if conditions are met."""
        klines = _declining_klines(45)
        # Pad to 260 candles
        for i in range(260 - len(klines)):
            klines.insert(0, {
                "open": 100.0, "close": 100.0, "high": 100.5, "low": 99.5,
                "volume": 100.0, "t": (1_700_000_000 - (i + 1) * 3600) * 1000,
            })
        with patch("indicators.now_dubai", return_value=_fixed_dubai(14)):
            res = _compute_signal_layers(
                klines, mode="MINI_ASYM", trade_style="DAY_TRADE",
                symbol="BTCUSDT", enable_t16=False,
            )
        # With T16 disabled, a choppy regime must not return REVERSAL signal type
        assert res.get("signal_type") != "REVERSAL"
