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
)


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
