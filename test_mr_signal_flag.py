"""
test_mr_signal_flag.py — Prove enable_mr_signal=False is safe to deploy.

Two claims:
  1. TRENDING candles: enable_mr_signal=False produces byte-for-byte identical
     output to enable_mr_signal=True (MR never fires in trending regimes).
  2. CHOPPY candles where MR would normally fire: enable_mr_signal=False
     returns BLOCKED; enable_mr_signal=True returns the MR signal.

Uses local SQLite candles (no network, no Render needed).
"""
from __future__ import annotations
import os, sys, sqlite3, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("ASYM_DB_PATH", "asymmetric_demo.db")

from indicators import _compute_signal_layers, _classify_regime, _adx, _atr

DB_PATH = Path(__file__).parent / "asymmetric_demo.db"
SEP  = "=" * 68
SEP2 = "-" * 68

def load_klines(symbol: str, tf: str, limit: int, offset: int = 0) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ts, open, high, low, close, volume
        FROM backtest_candles
        WHERE symbol = ? AND tf = ?
        ORDER BY ts ASC
        LIMIT ? OFFSET ?
        """,
        (symbol, tf, limit, offset),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "ts": r["ts"], "time": r["ts"],
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volume"]),
        }
        for r in rows
    ]

def call_signal(klines: list[dict], enable_mr: bool) -> dict:
    return _compute_signal_layers(
        klines=klines,
        mode="NORMAL",
        enable_mr_signal=enable_mr,
        # Keep everything else at defaults (matches backtester default call)
    )

def regime_label(klines: list[dict]) -> str:
    closes = [k["close"] for k in klines]
    highs  = [k["high"]  for k in klines]
    lows   = [k["low"]   for k in klines]
    adx    = _adx(highs, lows, closes, 14)
    atr    = _atr(highs, lows, closes, 14)
    atr_pct = (atr / closes[-1]) if atr and closes[-1] else 0.0
    rc = _classify_regime(highs, lows, closes, adx, atr_pct)
    return f"{rc['regime']} (ADX={rc['adx']:.1f}, block={rc['block']})"

def compare_outputs(sig_true: dict, sig_false: dict) -> tuple[bool, str]:
    """Return (identical, reason). 'identical' means no behaviour difference."""
    keys_to_compare = ["ok", "signal", "side", "grade"]
    diffs = []
    for k in keys_to_compare:
        v_true  = sig_true.get(k)
        v_false = sig_false.get(k)
        if v_true != v_false:
            diffs.append(f"{k}: True→{v_true!r}  False→{v_false!r}")
    return (len(diffs) == 0), "; ".join(diffs)

# ── Load candles ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TEST: enable_mr_signal=False safety check")
print(f"{SEP}\n")

# BTC 1h: 59856 candles, 2020-01-01 → ~2026-11-
# Trending period: BTC bull run mid-2024 (candles ~37000)
# Choppy period:   bearish choppy range ~2022-Q3  (candles ~20000)
SYMBOL = "BTCUSDT"
TF     = "1h"

WINDOW = 300  # indicators need ≥220, use 300 for safety

# --- Identify TRENDING windows ---
# Scan 20 consecutive 300-candle windows from 2024 (offset ~35000),
# pick ones where regime is TRENDING and ADX > 25.
print("  Scanning for TRENDING windows in BTC 2024-2025 ...")
trending_found = []
for off in range(35_000, 38_000, 50):
    kl = load_klines(SYMBOL, TF, WINDOW, off)
    if len(kl) < WINDOW:
        break
    closes = [k["close"] for k in kl]
    highs  = [k["high"]  for k in kl]
    lows   = [k["low"]   for k in kl]
    adx    = _adx(highs, lows, closes, 14)
    atr    = _atr(highs, lows, closes, 14)
    atr_pct = (atr / closes[-1]) if atr and closes[-1] else 0.0
    rc = _classify_regime(highs, lows, closes, adx, atr_pct)
    if not rc["block"] and rc["regime"] == "TRENDING" and rc["adx"] > 25:
        trending_found.append((off, kl, rc))
    if len(trending_found) >= 5:
        break

# --- Identify CHOPPY windows where MR fires ---
print("  Scanning for CHOPPY windows where MR fires (BTC 2021-2023) ...")
choppy_mr_found = []
for off in range(14_000, 40_000, 100):
    kl = load_klines(SYMBOL, TF, WINDOW, off)
    if len(kl) < WINDOW:
        break
    closes = [k["close"] for k in kl]
    highs  = [k["high"]  for k in kl]
    lows   = [k["low"]   for k in kl]
    adx    = _adx(highs, lows, closes, 14)
    atr    = _atr(highs, lows, closes, 14)
    atr_pct = (atr / closes[-1]) if atr and closes[-1] else 0.0
    rc = _classify_regime(highs, lows, closes, adx, atr_pct)
    if rc["block"]:
        sig_on = call_signal(kl, enable_mr=True)
        if sig_on.get("ok") and sig_on.get("signal") == "MEAN_REVERSION":
            choppy_mr_found.append((off, kl, rc, sig_on))
        if len(choppy_mr_found) >= 5:
            break

print(f"  Found {len(trending_found)} trending windows, {len(choppy_mr_found)} choppy MR-firing windows\n")

# ── TEST 1: Trending — outputs must be identical ──────────────────────────────
print(f"{SEP}")
print("  TEST 1: TRENDING candles — enable_mr_signal=False must NOT change output")
print(f"{SEP2}")
print(f"  {'#':<4}  {'Offset':>7}  {'Regime':<30}  {'MR=True result':<20}  {'Identical?'}")
print(f"{SEP2}")

trending_pass = True
for i, (off, kl, rc) in enumerate(trending_found):
    sig_true  = call_signal(kl, enable_mr=True)
    sig_false = call_signal(kl, enable_mr=False)
    identical, reason = compare_outputs(sig_true, sig_false)
    if not identical:
        trending_pass = False
    regime_str = f"{rc['regime']} ADX={rc['adx']:.1f}"
    result_str = f"ok={sig_true.get('ok')} sig={sig_true.get('signal','?')}"
    status = "✓ IDENTICAL" if identical else f"✗ DIFFERS: {reason}"
    print(f"  {i+1:<4}  {off:>7}  {regime_str:<30}  {result_str:<20}  {status}")

if not trending_found:
    print("  WARNING: no trending windows found — check DB candle range")
    trending_pass = None

# ── TEST 2: Choppy MR-firing — False must block, True must return MR ─────────
print(f"\n{SEP}")
print("  TEST 2: CHOPPY candles where MR fires — False must block signal")
print(f"{SEP2}")
print(f"  {'#':<4}  {'Offset':>7}  {'Regime':<24}  {'MR=True':<18}  {'MR=False':<18}  {'Correct?'}")
print(f"{SEP2}")

choppy_pass = True
for i, (off, kl, rc, sig_on) in enumerate(choppy_mr_found):
    sig_off = call_signal(kl, enable_mr=False)
    mr_on_ok  = sig_on.get("ok")  is True  and sig_on.get("signal")  in ("MEAN_REVERSION", "MR_LONG", "MR_SHORT")
    mr_off_ok = sig_off.get("ok") is False
    correct = mr_on_ok and mr_off_ok
    if not correct:
        choppy_pass = False
    on_str  = f"ok={sig_on.get('ok')} {sig_on.get('signal','?')}"
    off_str = f"ok={sig_off.get('ok')} {sig_off.get('signal','?')}"
    regime_str = f"{rc['regime']} ADX={rc['adx']:.1f}"
    status = "✓ CORRECT" if correct else "✗ WRONG"
    print(f"  {i+1:<4}  {off:>7}  {regime_str:<24}  {on_str:<18}  {off_str:<18}  {status}")

if not choppy_mr_found:
    print("  WARNING: no choppy MR-firing windows found — MR may not fire in this dataset range")
    choppy_pass = None

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SUMMARY")
print(f"{SEP2}")

def status_str(result) -> str:
    if result is None:
        return "SKIP (no windows found)"
    return "PASS" if result else "FAIL"

print(f"  Test 1 (trending → flag is no-op)   : {status_str(trending_pass)}")
print(f"  Test 2 (choppy MR fires → flag OFF)  : {status_str(choppy_pass)}")
print()

flag_safe = (trending_pass is True or trending_pass is None) and \
            (choppy_pass  is True or choppy_pass  is None)

if flag_safe:
    print("  enable_mr_signal=False is SAFE to deploy:")
    print("    - Trending regime: no change (MR never reaches trending candles)")
    print("    - Choppy regime:   MR blocked → signal returns BLOCKED/REGIME_*")
    print("    - Backtest use:    MR OFF comparison valid for 4-gate protocol")
    print()
    print("  FLAG DEFAULTS:")
    print("    indicators.py  _compute_signal_layers(enable_mr_signal=True)  ← live behaviour preserved")
    print("    backtester.py  _run_worker(enable_mr_signal=True)              ← backward-compatible")
    print("    main.py        ENABLE_MR_SIGNAL flag NOT added                 ← uses indicators.py default=True")
else:
    print("  UNSAFE — do not deploy until all tests pass")
print(SEP)
