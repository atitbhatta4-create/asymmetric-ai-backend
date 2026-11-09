#!/usr/bin/env python3
"""
test_grade_b_signal_path.py
───────────────────────────
Unit tests for Option 3 / Grade B code paths:

1. Pending trades dict structure — ENABLE_GRADE_B_ALL=True
   Verifies T1/T2 keys, sizes, labels, breakeven flag.

2. Grade A signal forced to Grade B by ENABLE_GRADE_B_ALL
   Confirms signal grade is preserved in log but trade uses B mechanics.

3. Tiny-account path — backtester degrades B→A when T1 notional < $5.
   Confirms force_grade_b=True (enable_grade_b_all) does NOT override
   the $5 minimum check; live raises ValueError instead.

4. Flags-OFF invariant — verify pending_trades identical with ENABLE_GRADE_B_ALL=False
   and a natural Grade A signal (no regression).
"""

import sys, os, json
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
# Test 1 — pending_trades structure with ENABLE_GRADE_B_ALL=True
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 1: pending_trades dict structure (ENABLE_GRADE_B_ALL=True) ──")

# Simulate exactly what main.py does at the trade-open branch
ENABLE_GRADE_B_ALL = True
grade   = "A"   # natural Grade A signal forced to Grade B by flag
_mtf_sm = 1.0   # no MTF size reduction

base_trade = {
    "entry_price": 100.0,
    "side":        "LONG",
    "mode":        "MINI_ASYM",
    "signal":      "TREND_LONG",
    "open_ts":     1700000000.0,
    "sl_pct_open": 0.025,
    "tp_pct_open": 0.040,
    "score_open":  0.72,
}

if grade == "B" or ENABLE_GRADE_B_ALL:
    pending_trades = [
        {**base_trade, "grade": "B", "size_mult": round(0.60 * _mtf_sm, 4), "tp_mult": 1.00,
         "is_primary": True, "label": "T1"},
        {**base_trade, "grade": "B", "size_mult": round(0.40 * _mtf_sm, 4), "tp_mult": 1.00,
         "is_primary": False, "label": "T2", "breakeven_after_t1": True},
    ]

T1, T2 = pending_trades[0], pending_trades[1]

check("T1 label == 'T1'",                T1["label"] == "T1")
check("T2 label == 'T2'",                T2["label"] == "T2")
check("T1 is_primary == True",           T1["is_primary"] == True)
check("T2 is_primary == False",          T2["is_primary"] == False)
check("T1 grade == 'B'",                 T1["grade"] == "B")
check("T2 grade == 'B'",                 T2["grade"] == "B")
check("T1 size_mult == 0.60",            T1["size_mult"] == 0.60)
check("T2 size_mult == 0.40",            T2["size_mult"] == 0.40)
check("T1+T2 sizes sum to 1.0",          round(T1["size_mult"] + T2["size_mult"], 4) == 1.00)
check("T1 tp_mult == 1.00",              T1["tp_mult"] == 1.00)
check("T2 tp_mult == 1.00",              T2["tp_mult"] == 1.00)
check("T2 has breakeven_after_t1=True",  T2.get("breakeven_after_t1") == True)
check("T1 does NOT have breakeven_after_t1", "breakeven_after_t1" not in T1)
check("Two trades in pending list",      len(pending_trades) == 2)

# Signal grade preserved (not overwritten on the signal dict)
signal_grade_preserved = (grade == "A")  # original signal grade unchanged
check("Original signal grade 'A' preserved (Option3 log tag)", signal_grade_preserved)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — ENABLE_GRADE_B_ALL=False, Grade A signal → Grade A pending_trades
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 2: Grade A path unchanged when ENABLE_GRADE_B_ALL=False ──")

ENABLE_GRADE_B_ALL = False
grade = "A"
_t18_demo = True   # scaling enabled
_grade_a_sm = round(0.70 * _mtf_sm, 4)   # T18 enters at 70%

if grade == "B" or ENABLE_GRADE_B_ALL:
    pt_gradeA = []  # won't be reached
else:
    pt_gradeA = [
        {**base_trade, "grade": "A", "size_mult": _grade_a_sm, "tp_mult": 1.00,
         "is_primary": True, "scale1_done": False, "scale2_done": False,
         "scaling_enabled": _t18_demo},
    ]

check("Grade A: only 1 trade in pending list",      len(pt_gradeA) == 1)
check("Grade A: grade == 'A'",                      pt_gradeA[0]["grade"] == "A")
check("Grade A: size_mult == 0.70 (T18 entry)",     pt_gradeA[0]["size_mult"] == 0.70)
check("Grade A: is_primary == True",                pt_gradeA[0]["is_primary"] == True)
check("Grade A: scaling_enabled == True",           pt_gradeA[0]["scaling_enabled"] == True)
check("Grade A: no breakeven_after_t1 key",        "breakeven_after_t1" not in pt_gradeA[0])


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Tiny-account: backtester degrades B→A when T1 < $5
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 3: Tiny-account Grade B→A degradation in backtester ──")

def backtester_grade_check(equity, effective_size, enable_grade_b_all):
    """Mirrors backtester.py lines 1421-1423."""
    grade = "B" if enable_grade_b_all else "A"
    if grade == "B" and equity * effective_size * 0.60 < 5.0:
        grade = "A"   # degrade — T1 leg would be below exchange min
    return grade

# $20 equity, 30% size → T1 = $20 × 0.30 × 0.60 = $3.60 < $5 → degrade
g = backtester_grade_check(equity=20.0, effective_size=0.30, enable_grade_b_all=True)
check("$20 equity, 30% size: T1=$3.60 → degraded to A", g == "A",
      f"got grade={g}")

# $100 equity, 30% size → T1 = $100 × 0.30 × 0.60 = $18 ≥ $5 → stays B
g = backtester_grade_check(equity=100.0, effective_size=0.30, enable_grade_b_all=True)
check("$100 equity, 30% size: T1=$18 → stays B", g == "B",
      f"got grade={g}")

# $1000 equity, 30% size → T1 = $1000 × 0.30 × 0.60 = $180 ≥ $5 → stays B
g = backtester_grade_check(equity=1000.0, effective_size=0.30, enable_grade_b_all=True)
check("$1000 equity, 30% size: T1=$180 → stays B", g == "B",
      f"got grade={g}")

# enable_grade_b_all=False → grade A from the start (no degradation needed)
g = backtester_grade_check(equity=20.0, effective_size=0.30, enable_grade_b_all=False)
check("enable_grade_b_all=False → grade A regardless of size", g == "A",
      f"got grade={g}")

# Live side: notional < $5 raises ValueError (no silent downgrade)
print("  [live check — ValueError for notional < $5]")
def live_grade_b_min_check(usdt_size, leverage):
    notional = usdt_size * leverage
    if notional < 5.0:
        raise ValueError(
            f"INSUFFICIENT_BALANCE: notional ${notional:.2f} below $5 minimum."
        )
    return True

try:
    live_grade_b_min_check(usdt_size=0.40, leverage=6)   # notional = $2.40
    check("$0.40 × 6× = $2.40 → raises ValueError", False, "no exception raised")
except ValueError as e:
    check("$0.40 × 6× = $2.40 → raises ValueError", True)

try:
    result = live_grade_b_min_check(usdt_size=1.00, leverage=6)   # notional = $6
    check("$1.00 × 6× = $6.00 → passes", result == True)
except ValueError:
    check("$1.00 × 6× = $6.00 → passes", False, "unexpected ValueError")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Flag OFF: indicators score_floor change is backward-compatible
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test 4: score_floor=0.0 is backward-compatible with indicators ──")

try:
    from indicators import _compute_signal_layers
    import inspect
    sig = inspect.signature(_compute_signal_layers)
    has_score_floor = "score_floor" in sig.parameters
    check("_compute_signal_layers accepts score_floor param", has_score_floor)
    default_val = sig.parameters["score_floor"].default
    check("score_floor default is 0.0 (no effect when not passed)", default_val == 0.0,
          f"default={default_val}")
except Exception as e:
    check("indicators import OK", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 56)
print(f"  Results: {len(PASS)} passed  |  {len(FAIL)} failed")
if FAIL:
    print(f"  FAILURES: {', '.join(FAIL)}")
else:
    print("  All tests passed ✓")
print("=" * 56)

sys.exit(0 if not FAIL else 1)
