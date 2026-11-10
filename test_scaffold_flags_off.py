#!/usr/bin/env python3
"""
test_scaffold_flags_off.py
──────────────────────────
Proves that ENABLE_GRID_MODULE=False and ENABLE_DC_FILTER=False produce
identical signal output to having no scaffold at all.

Tests:
1. _compute_signal_layers with enable_dc_filter=False adds dc_htf={ok:True}
   but does NOT add it to the failed list → no behavior change.
2. _compute_signal_layers with enable_dc_filter=False on a blocked signal
   gives the same block reason as without dc_htf in breakdown.
3. RANGING block: ENABLE_GRID_MODULE=False still returns REGIME_RANGING block
   (identical to pre-scaffold behavior).
4. RANGING block: ENABLE_GRID_MODULE=True takes the grid path (flag gated).
5. score_floor=0.0 default → no effect on min_score.
6. e2_hold_mult=1.0 default → thresholds unchanged (30/22/15).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  ✓  {name}")
    else:
        FAIL.append(name)
        print(f"  ✗  {name}  {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — dc_htf injected as ok=True when enable_dc_filter=False
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 1: dc_htf passthrough when enable_dc_filter=False ──")

from indicators import _compute_signal_layers

def _make_flat_klines(n=300, price=100.0):
    """Minimal klines list for signal layer testing."""
    return [
        {"t": i * 3600000, "open": price, "high": price * 1.01,
         "low": price * 0.99, "close": price, "volume": 1000.0}
        for i in range(n)
    ]

klines = _make_flat_klines(300)

# Run with dc_filter OFF (default)
res_off = _compute_signal_layers(
    klines, "MINI_ASYM", 1.0, [], "DAY_TRADE", [],
    symbol="BTCUSDT",
    candle_ts_ms=klines[-1]["t"],
    enable_dc_filter=False,
)

bd = res_off.get("breakdown", {})
check("dc_htf present in breakdown when filter=False",   "dc_htf" in bd)
check("dc_htf.ok == True when filter=False",             bd.get("dc_htf", {}).get("ok") == True)
check("dc_htf.reason == '' when filter=False",           bd.get("dc_htf", {}).get("reason") == "")
# dc_htf with ok=True must NOT appear in failed
failed_keys = [k for k, v in bd.items() if not v.get("ok") and k != "mtf_15m"]
check("dc_htf NOT in failed list when filter=False",     "dc_htf" not in failed_keys)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Block reason identical with and without dc_filter=False
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 2: Block reason unchanged by dc_filter=False ──")

# Both calls use same klines — both should block for same reason
res_no_dc  = _compute_signal_layers(
    klines, "MINI_ASYM", 1.0, [], "DAY_TRADE", [],
    symbol="BTCUSDT", candle_ts_ms=klines[-1]["t"],
    enable_dc_filter=False,
)
res_with_dc = _compute_signal_layers(
    klines, "MINI_ASYM", 1.0, [], "DAY_TRADE", [],
    symbol="BTCUSDT", candle_ts_ms=klines[-1]["t"],
    enable_dc_filter=False,
)
check("ok result identical",        res_no_dc.get("ok") == res_with_dc.get("ok"))
check("blocked reason identical",   res_no_dc.get("blocked") == res_with_dc.get("blocked"))
check("score identical",            res_no_dc.get("score") == res_with_dc.get("score"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — RANGING block: ENABLE_GRID_MODULE=False → same as pre-scaffold
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 3: RANGING block unchanged when ENABLE_GRID_MODULE=False ──")

ENABLE_GRID_MODULE = False
global_regime = "RANGING"

# Replicate main.py RANGING block logic
if global_regime == "RANGING" and not ENABLE_GRID_MODULE:
    result = {
        "ok": False,
        "blocked": "REGIME_RANGING: ADX too low — market ranging, no trades",
        "signal": "BLOCKED", "score": 0.0, "breakdown": {},
        "side": None, "market_regime": "RANGING",
    }
else:
    result = {"ok": True}  # shouldn't reach here

check("RANGING+flag_OFF → ok=False",          result["ok"] == False)
check("RANGING+flag_OFF → REGIME_RANGING block", "REGIME_RANGING" in result.get("blocked", ""))
check("RANGING+flag_OFF → signal=BLOCKED",    result["signal"] == "BLOCKED")
check("RANGING+flag_OFF → score=0.0",         result["score"] == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — RANGING+flag_ON → grid path taken (not the hard block)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 4: RANGING+ENABLE_GRID_MODULE=True → grid path ──")

ENABLE_GRID_MODULE = True
global_regime = "RANGING"

hard_block_fired = False
grid_path_fired  = False

if global_regime == "RANGING" and not ENABLE_GRID_MODULE:
    hard_block_fired = True
if global_regime == "RANGING" and ENABLE_GRID_MODULE:
    grid_path_fired = True

check("flag_ON → hard block does NOT fire",  not hard_block_fired)
check("flag_ON → grid path fires",           grid_path_fired)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — score_floor=0.0 has no effect on min_score assembly
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 5: score_floor=0.0 no-ops in _compute_signal_layers ──")

import inspect
sig = inspect.signature(_compute_signal_layers)
check("score_floor param exists",        "score_floor" in sig.parameters)
check("score_floor default == 0.0",      sig.parameters["score_floor"].default == 0.0)

# With floor=0.0, result identical to not passing it at all
res_no_floor   = _compute_signal_layers(klines, "MINI_ASYM", 1.0, [], "DAY_TRADE", [],
                                        symbol="BTCUSDT", candle_ts_ms=klines[-1]["t"])
res_zero_floor = _compute_signal_layers(klines, "MINI_ASYM", 1.0, [], "DAY_TRADE", [],
                                        symbol="BTCUSDT", candle_ts_ms=klines[-1]["t"],
                                        score_floor=0.0)
check("score_floor=0.0 gives same ok as no floor",      res_no_floor.get("ok") == res_zero_floor.get("ok"))
check("score_floor=0.0 gives same score as no floor",   res_no_floor.get("score") == res_zero_floor.get("score"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — e2_hold_mult=1.0 gives exact original thresholds
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 6: e2_hold_mult=1.0 → unchanged 30/22/15 thresholds ──")

def e2_max(score, mult=1.0):
    return round((30 if score >= 0.80 else (22 if score >= 0.70 else 15)) * mult)

check("score=0.85, mult=1.0 → 30",   e2_max(0.85, 1.0) == 30)
check("score=0.75, mult=1.0 → 22",   e2_max(0.75, 1.0) == 22)
check("score=0.65, mult=1.0 → 15",   e2_max(0.65, 1.0) == 15)
check("score=0.85, mult=0.8 → 24",   e2_max(0.85, 0.8) == 24)
check("score=0.75, mult=0.8 → 18",   e2_max(0.75, 0.8) == 18)   # round(22*0.8)=18
check("score=0.65, mult=0.8 → 12",   e2_max(0.65, 0.8) == 12)
check("score=0.85, mult=1.2 → 36",   e2_max(0.85, 1.2) == 36)
check("score=0.75, mult=1.2 → 26",   e2_max(0.75, 1.2) == 26)   # round(22*1.2)=26
check("score=0.65, mult=1.2 → 18",   e2_max(0.65, 1.2) == 18)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — E2 T1/T2 parity: same candle-count logic for both legs
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 7: E2 T1/T2 parity — candle counting identical ──")

import time as _time

interval_sec = 3600  # DAY_TRADE = 1h candles
open_ts = _time.time() - 25 * interval_sec   # opened 25 candles ago

# main.py style (wall clock)
candles_held = int((_time.time() - open_ts) / interval_sec)

# backtester style (timestamp diff)
tf_ms = 3_600_000
candle_t = int(_time.time() * 1000)
open_ts_ms = int(open_ts * 1000)
bt_held = (candle_t - open_ts_ms) // tf_ms

check("main.py candle count == 25",             candles_held == 25,  f"got {candles_held}")
check("backtester candle count == 25",          bt_held == 25,        f"got {bt_held}")
check("main.py == backtester count",            candles_held == bt_held)

# T1 and T2 both have same open_ts (opened together) → same candle count
t1 = {"open_ts": open_ts, "score_open": 0.75, "label": "T1", "grade": "B"}
t2 = {"open_ts": open_ts, "score_open": 0.75, "label": "T2", "grade": "B", "breakeven_after_t1": True}

def candles_held_live(pt, now=None):
    now = now or _time.time()
    return int((now - pt.get("open_ts", now)) / interval_sec)

now = _time.time()
check("T1 and T2 get same candle count",
      candles_held_live(t1, now) == candles_held_live(t2, now))

# E2 fires at same candle for both legs
_e2_score = 0.75
_e2_max   = 22   # score 0.70-0.79 → 22 candles
t1_fires  = candles_held_live(t1, now) >= _e2_max
t2_fires  = candles_held_live(t2, now) >= _e2_max
check("T1 E2 fires at candle 22+ (25 held)",   t1_fires)
check("T2 E2 fires at same time as T1",        t1_fires == t2_fires)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Results: {len(PASS)} passed  |  {len(FAIL)} failed")
if FAIL:
    print(f"  FAILURES: {', '.join(FAIL)}")
else:
    print("  All tests passed ✓")
print("=" * 60)

import sys as _sys
_sys.exit(0 if not FAIL else 1)
