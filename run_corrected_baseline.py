#!/usr/bin/env python3
"""
run_corrected_baseline.py
─────────────────────────
Single fresh backtest report — live-identical flags only.

Run from repo root:
  python run_corrected_baseline.py

No new features, no parameter changes. Pure measurement.
Results saved to corrected_baseline_report.txt in addition to stdout.
"""

import sys
import os
import time
import io
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from backtester import start_backtest, get_run

# ── Live feature flags (verbatim from main.py lines 97-104) ─────────────────
# ENABLE_MR_4H_GATE = False  → execution-only, not modelled in backtester
# ENABLE_PAT_EXEC   = False  → execution-only, not modelled in backtester
# ENABLE_T18        = False  → scale-in; mapped to enable_t18=False below
# TRAIL + CHOPPY_TRAIL: always ON (built into backtester exit loop)
LIVE_FLAGS = {
    "enable_t16": True,   # T16 reversal: no flag in main.py, always active
    "enable_t18": False,  # ENABLE_T18 = False in main.py
    "enable_s4":  False,  # S4 dynamic weights: not deployed
}

START_EQUITY = 1000.0
EXCHANGE     = "bybit"
MODE         = "MINI_ASYM"
STYLE        = "DAY_TRADE"
TEST_EMAIL   = "baseline@asym-ai-internal"

IS_FROM  = "2020-01-01"
IS_TO    = "2023-12-31"
OOS_FROM = "2024-01-01"
OOS_TO   = "2026-10-31"

COINS_10 = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "DOTUSDT",
]

# ─────────────────────────────────────────────────────────────────────────────

def run_and_wait(symbol: str, date_from: str, date_to: str, timeout_min: int = 30) -> dict:
    """Start a backtest, block until done, return raw results dict."""
    run_id = start_backtest(
        email=TEST_EMAIL,
        symbol=symbol,
        mode=MODE,
        style=STYLE,
        exchange=EXCHANGE,
        date_from=date_from,
        date_to=date_to,
        start_equity=START_EQUITY,
        enable_t16=LIVE_FLAGS["enable_t16"],
        enable_t18=LIVE_FLAGS["enable_t18"],
        enable_s4=LIVE_FLAGS["enable_s4"],
    )
    label = f"{symbol} {date_from[:4]}-{date_to[:4]}"
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        rec = get_run(run_id)
        if not rec:
            time.sleep(5)
            continue
        status = rec.get("status", "")
        if status == "done":
            print(f"  ✓ {label}")
            return rec.get("results") or {}
        if status == "error":
            raise RuntimeError(f"{label} failed: {rec.get('error', 'unknown')}")
        prog = rec.get("progress", 0)
        print(f"  … {label}  {status} {prog}%  ", end="\r", flush=True)
        time.sleep(10)
    raise TimeoutError(f"{label} did not finish within {timeout_min} min")


def summarise(r: dict) -> dict:
    return {
        "sharpe":  r.get("sharpe_ratio", 0.0),
        "ret":     r.get("total_return_pct", 0.0),
        "trades":  r.get("total_trades", 0),
        "wr":      r.get("win_rate", 0.0),
        "max_dd":  r.get("max_drawdown_pct", 0.0),
        "bnh":     r.get("buy_hold_return_pct", 0.0),
    }


def annual_return(r: dict, year: int) -> float:
    """Compound monthly returns for a calendar year → annual %."""
    months = r.get("monthly_returns", [])
    factor = 1.0
    for m in months:
        if m["month"].startswith(str(year)):
            factor *= (1 + m["return_pct"] / 100)
    return round((factor - 1) * 100, 1)


def monthly_table(r: dict, year_from: int, year_to: int) -> list[str]:
    """Build a grid: rows = months (Jan-Dec), cols = years."""
    months_map: dict[str, float] = {}
    for m in r.get("monthly_returns", []):
        months_map[m["month"]] = m["return_pct"]
    years = list(range(year_from, year_to + 1))
    header = f"  {'Mo':>3} " + "".join(f" {y:>8}" for y in years)
    sep    = "  " + "-" * (5 + 9 * len(years))
    rows   = [header, sep]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m_idx, name in enumerate(month_names, start=1):
        row = f"  {name:>3} "
        for y in years:
            key = f"{y}-{m_idx:02d}"
            val = months_map.get(key)
            cell = f"{val:>+7.1f}%" if val is not None else "       -"
            row += " " + cell
        rows.append(row)
    # Annual totals row
    row = f"  {'YTD':>3} "
    for y in years:
        ann = annual_return(r, y)
        row += f" {ann:>+7.1f}%"
    rows.append(sep)
    rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Output tee: write to both stdout and buffer for file save
class Tee:
    def __init__(self):
        self.buf = io.StringIO()
    def write(self, text: str):
        sys.stdout.write(text)
        self.buf.write(text)
        sys.stdout.flush()
    def writeln(self, text: str = ""):
        self.write(text + "\n")
    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.buf.getvalue())

out = Tee()

out.writeln()
out.writeln("=" * 72)
out.writeln("  ASYMMETRIC AI — CORRECTED BASELINE REPORT")
out.writeln(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out.writeln(f"  Mode/Style: {MODE} / {STYLE}   Exchange: {EXCHANGE.upper()}")
out.writeln(f"  Flags     : MR_4H_GATE=OFF  PAT_EXEC=OFF  T18=OFF  TRAIL=ON  CHOPPY=ON")
out.writeln(f"  IS window : {IS_FROM} → {IS_TO}")
out.writeln(f"  OOS window: {OOS_FROM} → {OOS_TO}")
out.writeln("=" * 72)

# ── Run 1: BTC IS ────────────────────────────────────────────────────────────
out.writeln()
out.writeln("Running BTC in-sample 2020-2023 ...")
btc_is_raw = run_and_wait("BTCUSDT", IS_FROM, IS_TO)
btc_is = summarise(btc_is_raw)

# ── Run 2: ETH IS ────────────────────────────────────────────────────────────
out.writeln("Running ETH in-sample 2020-2023 ...")
eth_is_raw = run_and_wait("ETHUSDT", IS_FROM, IS_TO)
eth_is = summarise(eth_is_raw)

# ── Runs 3-12: All 10 coins OOS ──────────────────────────────────────────────
out.writeln("Running 10-coin OOS 2024-2026 ...")
coin_oos_raw: dict[str, dict] = {}
for coin in COINS_10:
    coin_oos_raw[coin] = run_and_wait(coin, OOS_FROM, OOS_TO)

btc_oos_raw = coin_oos_raw["BTCUSDT"]
eth_oos_raw = coin_oos_raw["ETHUSDT"]
btc_oos = summarise(btc_oos_raw)
eth_oos = summarise(eth_oos_raw)

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 1 — LIVE FEATURE FLAGS")
out.writeln("=" * 72)
flag_rows = [
    ("ENABLE_MR_4H_GATE", "OFF", "execution-only — 4h EMA check before MR entry",   "not in backtester"),
    ("ENABLE_PAT_EXEC",   "OFF", "execution-only — candle pattern confirmation",     "not in backtester"),
    ("ENABLE_T18",        "OFF", "scale-in 70%→100% on Grade A winners",             "enable_t18=False"),
    ("TRAIL (7-level)",   "ON",  "ratchet locks 0/5/25/35/50/65/80% at 35-90% TP",  "always in exit loop"),
    ("CHOPPY_TRAIL",      "ON",  "beyond-TP trail activates at 1.5× TP in CHOPPY",  "always in exit loop"),
    ("T16 REVERSAL",      "ON",  "wider SL/TP for mean-reversion entries",           "enable_t16=True"),
    ("S4 DYN WEIGHTS",    "OFF", "dynamic signal layer weights",                     "enable_s4=False"),
]
out.writeln(f"  {'Flag':<22} {'State':>5}  {'Description':<46}  Backtest mapping")
out.writeln("  " + "-" * 95)
for name, state, desc, mapping in flag_rows:
    out.writeln(f"  {name:<22} {state:>5}  {desc:<46}  {mapping}")

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 2 — BTC RESULTS  (MINI_ASYM DAY_TRADE Bybit)")
out.writeln("=" * 72)

out.writeln()
out.writeln("  IN-SAMPLE (2020-2023, 4 years)")
out.writeln(f"  Sharpe     : {btc_is['sharpe']:+.2f}")
out.writeln(f"  Total Ret  : {btc_is['ret']:+.1f}%")
out.writeln(f"  Trades     : {btc_is['trades']}")
out.writeln(f"  Win Rate   : {btc_is['wr']:.1f}%")
out.writeln(f"  Max DD     : {btc_is['max_dd']:.1f}%")
out.writeln(f"  Buy & Hold : {btc_is['bnh']:+.1f}%")
out.writeln()
out.writeln("  Monthly grid (IS):")
for ln in monthly_table(btc_is_raw, 2020, 2023):
    out.writeln(ln)

out.writeln()
out.writeln("  OUT-OF-SAMPLE (2024-2026-10, 2.8 years)")
out.writeln(f"  Sharpe     : {btc_oos['sharpe']:+.2f}")
out.writeln(f"  Total Ret  : {btc_oos['ret']:+.1f}%")
out.writeln(f"  Trades     : {btc_oos['trades']}")
out.writeln(f"  Win Rate   : {btc_oos['wr']:.1f}%")
out.writeln(f"  Max DD     : {btc_oos['max_dd']:.1f}%")
out.writeln(f"  Buy & Hold : {btc_oos['bnh']:+.1f}%")
out.writeln()
out.writeln("  Monthly grid (OOS):")
for ln in monthly_table(btc_oos_raw, 2024, 2026):
    out.writeln(ln)

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 3 — ETH RESULTS  (MINI_ASYM DAY_TRADE Bybit)")
out.writeln("=" * 72)

out.writeln()
out.writeln("  IN-SAMPLE (2020-2023)")
out.writeln(f"  Sharpe {eth_is['sharpe']:+.2f}  Ret {eth_is['ret']:+.1f}%  "
            f"Trades {eth_is['trades']}  WR {eth_is['wr']:.1f}%  MaxDD {eth_is['max_dd']:.1f}%  "
            f"BnH {eth_is['bnh']:+.1f}%")
out.writeln()
out.writeln("  Monthly grid (IS):")
for ln in monthly_table(eth_is_raw, 2020, 2023):
    out.writeln(ln)

out.writeln()
out.writeln("  OUT-OF-SAMPLE (2024-2026-10)")
out.writeln(f"  Sharpe {eth_oos['sharpe']:+.2f}  Ret {eth_oos['ret']:+.1f}%  "
            f"Trades {eth_oos['trades']}  WR {eth_oos['wr']:.1f}%  MaxDD {eth_oos['max_dd']:.1f}%  "
            f"BnH {eth_oos['bnh']:+.1f}%")
out.writeln()
out.writeln("  Monthly grid (OOS):")
for ln in monthly_table(eth_oos_raw, 2024, 2026):
    out.writeln(ln)

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 4 — 10-COIN OOS SUMMARY  (2024-2026-10)")
out.writeln("=" * 72)
out.writeln()
out.writeln(f"  {'Coin':<12} {'Sharpe':>7} {'Return':>9} {'Trades':>7} {'WR':>6} {'MaxDD':>7}  {'BnH':>8}")
out.writeln("  " + "-" * 65)

pass_count = 0
for coin in COINS_10:
    r = summarise(coin_oos_raw[coin])
    flag = "PASS" if r["sharpe"] > 0 else "FAIL"
    if r["sharpe"] > 0:
        pass_count += 1
    out.writeln(
        f"  {coin:<12} {r['sharpe']:>+7.2f} {r['ret']:>+8.1f}% "
        f"{r['trades']:>7} {r['wr']:>5.1f}% {r['max_dd']:>6.1f}%  "
        f"{r['bnh']:>+7.1f}%  {flag}"
    )

out.writeln()
out.writeln(f"  Sharpe ≥ 0: {pass_count}/10 coins  "
            f"{'(PASS — >5/10 required)' if pass_count > 5 else '(FAIL — need >5/10)'}")

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 5 — BTC IS vs OOS COMPARISON")
out.writeln("=" * 72)
out.writeln()
out.writeln(f"  {'Metric':<18}  {'IS 2020-23':>12}  {'OOS 2024-26':>12}  Verdict")
out.writeln("  " + "-" * 60)

rows_cmp = [
    ("Sharpe",    btc_is["sharpe"],  btc_oos["sharpe"],  "higher is better"),
    ("Return %",  btc_is["ret"],     btc_oos["ret"],     "positive both = good"),
    ("Win Rate",  btc_is["wr"],      btc_oos["wr"],      "consistent = good"),
    ("Max DD %",  btc_is["max_dd"],  btc_oos["max_dd"],  "lower is better"),
    ("Trades",    btc_is["trades"],  btc_oos["trades"],  "scaled for period"),
]
for name, is_val, oos_val, hint in rows_cmp:
    drift = "→ stable" if abs(is_val - oos_val) / max(abs(is_val), 1) < 0.30 else "→ drift"
    out.writeln(f"  {name:<18}  {is_val:>+12.2f}  {oos_val:>+12.2f}  {hint} {drift}")

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 6 — ALL PAST TESTS: VALIDITY ON CORRECTED BASELINE")
out.writeln("  Corrected = T18 OFF, MR4H OFF, PAT OFF  (this run)")
out.writeln("=" * 72)
out.writeln()

PAST_TESTS = [
    # (name, old_delta, deploy_state, validity, action_needed)
    (
        "Fix 1: Weekly trend gate",
        "unknown",
        "REJECTED",
        "STALE — measured on false baseline (T18 ON)",
        "Re-run on corrected base before next eval",
    ),
    (
        "Fix 2: ADX floor raise +2",
        "unknown",
        "DEPLOYED (live)",
        "STALE — delta not on corrected base",
        "Re-run to confirm it still helps",
    ),
    (
        "Fix 3: Multi-candle confirm",
        "unknown",
        "REJECTED",
        "STALE — measured on false baseline",
        "Re-run before reconsidering",
    ),
    (
        "Fix 3b: ADX dynamic scale",
        "unknown",
        "REJECTED",
        "STALE — measured on false baseline",
        "Re-run before reconsidering",
    ),
    (
        "Fix 4: 7-level trail ratchet",
        "+0.25 (old)",
        "DEPLOYED — always ON",
        "BAKED IN — this run already includes it",
        "No action — trail is in baseline",
    ),
    (
        "Full candle pattern exec",
        "unknown",
        "REJECTED",
        "STALE — measured on false baseline",
        "Re-run before reconsidering",
    ),
    (
        "Test 6: T18 MR exemption",
        "+0.03 (old, T18 ON base)",
        "SUPERSEDED — T18 is now OFF",
        "VOID — T18 OFF makes this irrelevant",
        "Drop from list; re-evaluate if T18 re-enabled",
    ),
    (
        "Test 7: Choppy trail delay",
        "unknown (old base)",
        "STATUS UNCLEAR — check main.py",
        "STALE — delta was on T18-ON baseline",
        "Re-run to get corrected delta",
    ),
    (
        "Test 8: MR 4H gate",
        "-0.29 (old)",
        "REJECTED",
        "LIKELY STILL NEGATIVE — large clear loss",
        "Re-run to confirm before final close",
    ),
    (
        "Test 9a: Corrected base (no OE gate)",
        "Sharpe -0.04 (old base)",
        "BASELINE RUN",
        "SUPERSEDED BY THIS REPORT",
        "This report is the new canonical baseline",
    ),
    (
        "Test 9b: OE gate v2 (RSI+EMA50)",
        "-1.24 vs old base",
        "REJECTED",
        "LIKELY STILL NEGATIVE — large clear loss",
        "Re-run if reconsidering; low priority",
    ),
]

out.writeln(f"  {'Test':<38} {'Old Delta':>10}  {'State':<22}  Notes")
out.writeln("  " + "-" * 100)
for name, old_delta, deploy, validity, action in PAST_TESTS:
    out.writeln(f"  {name:<38} {old_delta:>10}  {deploy:<22}")
    out.writeln(f"  {'':38} {'':10}  Validity: {validity}")
    out.writeln(f"  {'':38} {'':10}  Action  : {action}")
    out.writeln()

# ─────────────────────────────────────────────────────────────────────────────
out.writeln("=" * 72)
out.writeln("  CORRECTED BASELINE — FINAL NUMBERS")
out.writeln("  These are the ONLY valid numbers. All future tests compare here.")
out.writeln("=" * 72)
out.writeln()
out.writeln(f"  BTC IS  (2020-2023): Sharpe {btc_is['sharpe']:+.2f}  Ret {btc_is['ret']:+.1f}%  "
            f"Trades {btc_is['trades']}  WR {btc_is['wr']:.1f}%  MaxDD {btc_is['max_dd']:.1f}%")
out.writeln(f"  BTC OOS (2024-2026): Sharpe {btc_oos['sharpe']:+.2f}  Ret {btc_oos['ret']:+.1f}%  "
            f"Trades {btc_oos['trades']}  WR {btc_oos['wr']:.1f}%  MaxDD {btc_oos['max_dd']:.1f}%")
out.writeln(f"  ETH IS  (2020-2023): Sharpe {eth_is['sharpe']:+.2f}  Ret {eth_is['ret']:+.1f}%  "
            f"Trades {eth_is['trades']}  WR {eth_is['wr']:.1f}%  MaxDD {eth_is['max_dd']:.1f}%")
out.writeln(f"  ETH OOS (2024-2026): Sharpe {eth_oos['sharpe']:+.2f}  Ret {eth_oos['ret']:+.1f}%  "
            f"Trades {eth_oos['trades']}  WR {eth_oos['wr']:.1f}%  MaxDD {eth_oos['max_dd']:.1f}%")
out.writeln(f"  10-coin OOS        : {pass_count}/10 coins Sharpe ≥ 0")
out.writeln()
out.writeln("  Next permitted step: Re-run all stale tests against THIS baseline.")
out.writeln("  No parameter tuning until every test has a corrected delta.")
out.writeln("=" * 72)
out.writeln()

# ── Save to file ─────────────────────────────────────────────────────────────
report_path = os.path.join(os.path.dirname(__file__), "corrected_baseline_report.txt")
out.save(report_path)
print(f"\nReport saved → {report_path}")
