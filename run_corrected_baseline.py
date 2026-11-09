#!/usr/bin/env python3
"""
run_corrected_baseline.py
─────────────────────────
Sequential dual-mode baseline — runs one backtest at a time to avoid
Bybit rate-limiting that corrupts parallel candle fetches.

Each coin runs floor-OFF then floor-ON back-to-back. The second run reuses
cached candles from the first (same symbol/date range), so it's fast.

Coin list: BTC ETH SOL BNB XRP ADA NEAR DOGE AVAX LINK
  — NEARUSDT replaces DOTUSDT (DOT <8 OOS trades; NEAR actively traded live)
"""

import sys
import os
import time
import io
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from backtester import (
    start_backtest, get_run, init_backtest_tables,
    check_candle_coverage, prefetch_all_candles,
)
init_backtest_tables()

# ── Live feature flags ───────────────────────────────────────────────────────
LIVE_FLAGS = {
    "enable_t16": True,
    "enable_t18": False,
    "enable_s4":  False,
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
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "NEARUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
]

TIMEOUT_MIN = 150  # 2.5 h per run; BTC/ETH OOS need ~90 min on SQLite

# ─────────────────────────────────────────────────────────────────────────────

def run_one(symbol: str, date_from: str, date_to: str,
            enable_floor: bool, out) -> dict:
    """Run a single backtest, block until done, return results dict."""
    run_id = start_backtest(
        email=TEST_EMAIL, symbol=symbol, mode=MODE, style=STYLE,
        exchange=EXCHANGE, date_from=date_from, date_to=date_to,
        start_equity=START_EQUITY,
        enable_t16=LIVE_FLAGS["enable_t16"],
        enable_t18=LIVE_FLAGS["enable_t18"],
        enable_s4=LIVE_FLAGS["enable_s4"],
        enable_floor=enable_floor,
    )
    floor_tag = "floor-ON" if enable_floor else "floor-OFF"
    label     = f"{symbol} {date_from[:4]}-{date_to[:4]} [{floor_tag}]"
    deadline  = time.time() + TIMEOUT_MIN * 60

    while time.time() < deadline:
        rec = get_run(run_id)
        if not rec:
            time.sleep(5)
            continue
        status = rec.get("status", "")
        if status == "done":
            out.writeln(f"  ✓ {label}")
            return rec.get("results") or {}
        if status == "error":
            out.writeln(f"  ✗ {label}  ERROR: {rec.get('error','?')}")
            return {}
        prog = rec.get("progress", 0)
        print(f"  … {label}  {status} {prog}%  ", end="\r", flush=True)
        time.sleep(10)

    out.writeln(f"  TIMEOUT {label}")
    return {}


def summarise(r: dict) -> dict:
    return {
        "sharpe":       r.get("sharpe_ratio", 0.0),
        "ret":          r.get("total_return_pct", 0.0),
        "trades":       r.get("total_trades", 0),
        "wr":           r.get("win_rate", 0.0),
        "max_dd":       r.get("max_drawdown_pct", 0.0),
        "bnh":          r.get("buy_hold_return_pct", 0.0),
        "floor_resets": r.get("floor_reset_count", 0),
        "reset_log":    r.get("floor_resets", []),
    }


def annual_return(r: dict, year: int) -> float:
    months = r.get("monthly_returns", [])
    factor = 1.0
    for m in months:
        if m["month"].startswith(str(year)):
            factor *= (1 + m["return_pct"] / 100)
    return round((factor - 1) * 100, 1)


def monthly_table(r: dict, year_from: int, year_to: int) -> list:
    months_map: dict = {}
    for m in r.get("monthly_returns", []):
        months_map[m["month"]] = m["return_pct"]
    years  = list(range(year_from, year_to + 1))
    header = f"  {'Mo':>3} " + "".join(f" {y:>8}" for y in years)
    sep    = "  " + "-" * (5 + 9 * len(years))
    rows   = [header, sep]
    mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m_idx, name in enumerate(mnames, start=1):
        row = f"  {name:>3} "
        for y in years:
            val  = months_map.get(f"{y}-{m_idx:02d}")
            cell = f"{val:>+7.1f}%" if val is not None else "       -"
            row += " " + cell
        rows.append(row)
    row = f"  {'YTD':>3} "
    for y in years:
        row += f" {annual_return(r, y):>+7.1f}%"
    rows.append(sep)
    rows.append(row)
    return rows


def fmt_reset_log(reset_log: list) -> str:
    if not reset_log:
        return "none"
    parts = []
    for ev in reset_log:
        ts_s = datetime.utcfromtimestamp(ev["ts"] / 1000).strftime("%Y-%m-%d")
        dbl  = " ⚠double" if ev.get("double_hit") else ""
        parts.append(f"{ts_s} (equity=${ev['equity']:.0f}, cd={ev['cooldown_days']}d{dbl})")
    return "  |  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self):
        self.buf = io.StringIO()
    def write(self, t: str):
        sys.stdout.write(t); self.buf.write(t); sys.stdout.flush()
    def writeln(self, t: str = ""):
        self.write(t + "\n")
    def save(self, path: str):
        with open(path, "w") as f: f.write(self.buf.getvalue())

out = Tee()

out.writeln()
out.writeln("=" * 72)
out.writeln("  ASYMMETRIC AI — CORRECTED BASELINE REPORT (DUAL-MODE)")
out.writeln(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out.writeln(f"  Mode/Style: {MODE} / {STYLE}   Exchange: {EXCHANGE.upper()}")
out.writeln(f"  Flags     : MR_4H_GATE=OFF  PAT_EXEC=OFF  T18=OFF  TRAIL=ON  CHOPPY=ON")
out.writeln(f"  IS window : {IS_FROM} → {IS_TO}")
out.writeln(f"  OOS window: {OOS_FROM} → {OOS_TO}")
out.writeln(f"  Floor policy: 7-day cooldown (30-day on double hit ≤90 days)")
out.writeln(f"  Coin list : {', '.join(c.replace('USDT','') for c in COINS_10)}")
out.writeln("=" * 72)

# ── Run all sequentially — floor-OFF then floor-ON per coin ──────────────────
# (floor-OFF run fetches + caches candles; floor-ON reuses cache → fast)

# ── Pre-run data check: abort if any coin is under 97% coverage ──────────────
from datetime import timezone as _tz

def _to_ms(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=_tz.utc).timestamp() * 1000)

_oos_start = _to_ms(OOS_FROM)
_oos_end   = _to_ms(OOS_TO)
_is_start  = _to_ms(IS_FROM)
_is_end    = _to_ms(IS_TO)

out.writeln()
out.writeln("── Pre-run candle coverage check ───────────────────────────────────────")
for _window, _start, _end, _label in [
    (COINS_10, _oos_start, _oos_end, "OOS 1h"),
    (["BTCUSDT","ETHUSDT"], _is_start, _is_end, "IS 1h"),
]:
    _all_ok, _rows = check_candle_coverage(_window, ["1h"], _start, _end)
    for _r in _rows:
        _status = "✓" if _r["ok"] else "✗ INCOMPLETE"
        out.writeln(f"  {_r['symbol']:<12} {_label:<8} {_r['count']:>6}  {_r['pct']:>5.1f}%  "
                    f"{_r['first']} → {_r['last']}  {_status}")
    if not _all_ok:
        out.writeln()
        out.writeln("ABORT: Some coins have <97% candle coverage.")
        out.writeln("Run:   python3 fix_candle_cache.py  (then retry baseline)")
        out.save(os.path.join(os.path.dirname(__file__), "corrected_baseline_report.txt"))
        raise SystemExit(1)

out.writeln("  All coins ≥97% — proceeding.\n")

out.writeln()
out.writeln("Running IS (BTC + ETH) ...")
btc_is_off_raw = run_one("BTCUSDT", IS_FROM, IS_TO, enable_floor=False, out=out)
btc_is_on_raw  = run_one("BTCUSDT", IS_FROM, IS_TO, enable_floor=True,  out=out)
eth_is_off_raw = run_one("ETHUSDT", IS_FROM, IS_TO, enable_floor=False, out=out)
eth_is_on_raw  = run_one("ETHUSDT", IS_FROM, IS_TO, enable_floor=True,  out=out)

out.writeln()
out.writeln("Running OOS (10 coins × 2 modes) ...")
coin_oos_off: dict = {}
coin_oos_on:  dict = {}
for coin in COINS_10:
    coin_oos_off[coin] = run_one(coin, OOS_FROM, OOS_TO, enable_floor=False, out=out)
    coin_oos_on[coin]  = run_one(coin, OOS_FROM, OOS_TO, enable_floor=True,  out=out)

btc_is_off  = summarise(btc_is_off_raw)
btc_is_on   = summarise(btc_is_on_raw)
eth_is_off  = summarise(eth_is_off_raw)
eth_is_on   = summarise(eth_is_on_raw)
btc_oos_off = summarise(coin_oos_off["BTCUSDT"])
btc_oos_on  = summarise(coin_oos_on["BTCUSDT"])
eth_oos_off = summarise(coin_oos_off["ETHUSDT"])
eth_oos_on  = summarise(coin_oos_on["ETHUSDT"])

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
    ("FLOOR RESET",       "ON",  "7-day cooldown, 30-day on double hit ≤90 days",    "enable_floor=True"),
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

for label, off, on, raw_off, raw_on, yf, yt in [
    ("IN-SAMPLE (2020-2023)",        btc_is_off,  btc_is_on,  btc_is_off_raw,  btc_is_on_raw,  2020, 2023),
    ("OUT-OF-SAMPLE (2024-2026-10)", btc_oos_off, btc_oos_on, coin_oos_off["BTCUSDT"], coin_oos_on["BTCUSDT"], 2024, 2026),
]:
    out.writeln()
    out.writeln(f"  {label}")
    out.writeln(f"  {'Metric':<14} {'Floor OFF':>12}  {'Floor ON':>12}")
    out.writeln("  " + "-" * 42)
    out.writeln(f"  {'Sharpe':<14} {off['sharpe']:>+12.2f}  {on['sharpe']:>+12.2f}")
    out.writeln(f"  {'Total Ret':<14} {off['ret']:>+11.1f}%  {on['ret']:>+11.1f}%")
    out.writeln(f"  {'Trades':<14} {off['trades']:>12}  {on['trades']:>12}")
    out.writeln(f"  {'Win Rate':<14} {off['wr']:>11.1f}%  {on['wr']:>11.1f}%")
    out.writeln(f"  {'Max DD':<14} {off['max_dd']:>11.1f}%  {on['max_dd']:>11.1f}%")
    out.writeln(f"  {'Buy & Hold':<14} {off['bnh']:>+11.1f}%  {on['bnh']:>+11.1f}%")
    out.writeln(f"  {'Floor Resets':<14} {'n/a':>12}  {on['floor_resets']:>12}")
    if on["reset_log"]:
        out.writeln(f"  Reset events: {fmt_reset_log(on['reset_log'])}")
    out.writeln()
    out.writeln(f"  Monthly grid (floor OFF):")
    for ln in monthly_table(raw_off, yf, yt): out.writeln(ln)
    out.writeln()
    out.writeln(f"  Monthly grid (floor ON):")
    for ln in monthly_table(raw_on, yf, yt): out.writeln(ln)

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 3 — ETH RESULTS  (MINI_ASYM DAY_TRADE Bybit)")
out.writeln("=" * 72)

for label, off, on, raw_off, raw_on, yf, yt in [
    ("IN-SAMPLE (2020-2023)",        eth_is_off,  eth_is_on,  eth_is_off_raw,  eth_is_on_raw,  2020, 2023),
    ("OUT-OF-SAMPLE (2024-2026-10)", eth_oos_off, eth_oos_on, coin_oos_off["ETHUSDT"], coin_oos_on["ETHUSDT"], 2024, 2026),
]:
    out.writeln()
    out.writeln(f"  {label}")
    out.writeln(f"  {'Metric':<14} {'Floor OFF':>12}  {'Floor ON':>12}")
    out.writeln("  " + "-" * 42)
    out.writeln(f"  {'Sharpe':<14} {off['sharpe']:>+12.2f}  {on['sharpe']:>+12.2f}")
    out.writeln(f"  {'Total Ret':<14} {off['ret']:>+11.1f}%  {on['ret']:>+11.1f}%")
    out.writeln(f"  {'Trades':<14} {off['trades']:>12}  {on['trades']:>12}")
    out.writeln(f"  {'Win Rate':<14} {off['wr']:>11.1f}%  {on['wr']:>11.1f}%")
    out.writeln(f"  {'Max DD':<14} {off['max_dd']:>11.1f}%  {on['max_dd']:>11.1f}%")
    out.writeln(f"  {'Buy & Hold':<14} {off['bnh']:>+11.1f}%  {on['bnh']:>+11.1f}%")
    out.writeln(f"  {'Floor Resets':<14} {'n/a':>12}  {on['floor_resets']:>12}")
    if on["reset_log"]:
        out.writeln(f"  Reset events: {fmt_reset_log(on['reset_log'])}")
    out.writeln()
    out.writeln(f"  Monthly grid (floor ON):")
    for ln in monthly_table(raw_on, yf, yt): out.writeln(ln)

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 4 — 10-COIN OOS SUMMARY  (2024-2026-10)")
out.writeln("=" * 72)
out.writeln()
out.writeln("  FLOOR OFF (signal edge — no floor interruptions)")
out.writeln(f"  {'Coin':<12} {'Sharpe':>7} {'Return':>9} {'Trades':>7} {'WR':>6} {'MaxDD':>7}  {'BnH':>8}")
out.writeln("  " + "-" * 65)
pass_off = 0
for coin in COINS_10:
    r = summarise(coin_oos_off[coin])
    flag = "PASS" if r["sharpe"] > 0 else "FAIL"
    if r["sharpe"] > 0: pass_off += 1
    out.writeln(f"  {coin:<12} {r['sharpe']:>+7.2f} {r['ret']:>+8.1f}% "
                f"{r['trades']:>7} {r['wr']:>5.1f}% {r['max_dd']:>6.1f}%  "
                f"{r['bnh']:>+7.1f}%  {flag}")
out.writeln(f"\n  Sharpe ≥ 0: {pass_off}/10 coins")

out.writeln()
out.writeln("  FLOOR ON (with reset policy — mirrors live behaviour)")
out.writeln(f"  {'Coin':<12} {'Sharpe':>7} {'Return':>9} {'Trades':>7} {'WR':>6} {'MaxDD':>7}  {'BnH':>8}  {'Resets':>7}  Floor triggered")
out.writeln("  " + "-" * 95)
pass_on = 0
for coin in COINS_10:
    r = summarise(coin_oos_on[coin])
    flag = "PASS" if r["sharpe"] > 0 else "FAIL"
    if r["sharpe"] > 0: pass_on += 1
    trigger = fmt_reset_log(r["reset_log"]) if r["floor_resets"] else "—"
    out.writeln(f"  {coin:<12} {r['sharpe']:>+7.2f} {r['ret']:>+8.1f}% "
                f"{r['trades']:>7} {r['wr']:>5.1f}% {r['max_dd']:>6.1f}%  "
                f"{r['bnh']:>+7.1f}%  {r['floor_resets']:>7}  {flag}")
    if r["floor_resets"]:
        out.writeln(f"             Resets: {trigger}")
out.writeln(f"\n  Sharpe ≥ 0: {pass_on}/10 coins  "
            f"{'(PASS — >5/10 required)' if pass_on > 5 else '(FAIL — need >5/10)'}")

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  SECTION 5 — BTC IS vs OOS COMPARISON  (floor ON)")
out.writeln("=" * 72)
out.writeln()
out.writeln(f"  {'Metric':<18}  {'IS 2020-23':>12}  {'OOS 2024-26':>12}  Verdict")
out.writeln("  " + "-" * 60)
for name, is_v, oos_v, hint in [
    ("Sharpe",    btc_is_on["sharpe"],  btc_oos_on["sharpe"],  "higher is better"),
    ("Return %",  btc_is_on["ret"],     btc_oos_on["ret"],     "positive both = good"),
    ("Win Rate",  btc_is_on["wr"],      btc_oos_on["wr"],      "consistent = good"),
    ("Max DD %",  btc_is_on["max_dd"],  btc_oos_on["max_dd"],  "lower is better"),
    ("Trades",    btc_is_on["trades"],  btc_oos_on["trades"],  "scaled for period"),
]:
    drift = "→ stable" if abs(is_v - oos_v) / max(abs(is_v), 1) < 0.30 else "→ drift"
    out.writeln(f"  {name:<18}  {is_v:>+12.2f}  {oos_v:>+12.2f}  {hint} {drift}")

# ─────────────────────────────────────────────────────────────────────────────
out.writeln()
out.writeln("=" * 72)
out.writeln("  CORRECTED BASELINE — FINAL NUMBERS  (floor ON = live-matching)")
out.writeln("  These are the ONLY valid numbers. All future tests compare here.")
out.writeln("=" * 72)
out.writeln()
out.writeln(f"  BTC IS  (2020-2023): Sharpe {btc_is_on['sharpe']:+.2f}  Ret {btc_is_on['ret']:+.1f}%  "
            f"Trades {btc_is_on['trades']}  WR {btc_is_on['wr']:.1f}%  MaxDD {btc_is_on['max_dd']:.1f}%  "
            f"Resets {btc_is_on['floor_resets']}")
out.writeln(f"  BTC OOS (2024-2026): Sharpe {btc_oos_on['sharpe']:+.2f}  Ret {btc_oos_on['ret']:+.1f}%  "
            f"Trades {btc_oos_on['trades']}  WR {btc_oos_on['wr']:.1f}%  MaxDD {btc_oos_on['max_dd']:.1f}%  "
            f"Resets {btc_oos_on['floor_resets']}")
out.writeln(f"  ETH IS  (2020-2023): Sharpe {eth_is_on['sharpe']:+.2f}  Ret {eth_is_on['ret']:+.1f}%  "
            f"Trades {eth_is_on['trades']}  WR {eth_is_on['wr']:.1f}%  MaxDD {eth_is_on['max_dd']:.1f}%  "
            f"Resets {eth_is_on['floor_resets']}")
out.writeln(f"  ETH OOS (2024-2026): Sharpe {eth_oos_on['sharpe']:+.2f}  Ret {eth_oos_on['ret']:+.1f}%  "
            f"Trades {eth_oos_on['trades']}  WR {eth_oos_on['wr']:.1f}%  MaxDD {eth_oos_on['max_dd']:.1f}%  "
            f"Resets {eth_oos_on['floor_resets']}")
out.writeln(f"  10-coin OOS        : {pass_on}/10 coins Sharpe ≥ 0  (floor ON)")
out.writeln()
out.writeln("  Signal edge view   : floor OFF numbers above in Section 4.")
out.writeln("  Next step          : Step 1 — regime switching (once demo logs clean).")
out.writeln("=" * 72)
out.writeln()

report_path = os.path.join(os.path.dirname(__file__), "corrected_baseline_report.txt")
out.save(report_path)
print(f"\nReport saved → {report_path}")
