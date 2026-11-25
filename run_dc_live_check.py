#!/usr/bin/env python3
"""
run_dc_live_check.py
────────────────────
DC Filter live health check. Run any time on Render shell:

  python3 run_dc_live_check.py           # current DC state for all coins
  python3 run_dc_live_check.py --days 7  # also compare trade frequency last 7 days

Reports:
  1. ENABLE_DC_FILTER flag status (live from main.py)
  2. Current DC position for each coin → which sides are blocked right now
  3. Last N days trade count vs prior N days (did filter reduce noise?)
  4. Recent live trades with DC position at the time of entry (would they pass today?)
"""

import os, sys, argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from market_data import _fetch_klines_sync
from indicators import donchian_htf_filter
from database import db_conn

COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
    "ADAUSDT","AVAXUSDT","DOGEUSDT","LINKUSDT","NEARUSDT",
]

LONG_CUT  = 0.35   # v035 production thresholds
SHORT_CUT = 0.65

SEP  = "=" * 68
SEP2 = "-" * 68

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7,
                   help="Days to look back for trade frequency check (default 7)")
    p.add_argument("--exchange", default="bybit",
                   help="Exchange for klines (default bybit)")
    return p.parse_args()


def get_dc_state(symbol: str, exchange: str):
    """Fetch 4h klines and return current DC position + block status."""
    try:
        klines = _fetch_klines_sync(symbol, "4h", limit=25, exchange=exchange)
        if not klines or len(klines) < 20:
            return None, None, None, "INSUFFICIENT_DATA"

        highs  = [k["high"]  for k in klines]
        lows   = [k["low"]   for k in klines]
        closes = [k["close"] for k in klines]

        upper = max(highs[-20:])
        lower = min(lows[-20:])
        span  = upper - lower

        if span <= 0:
            return None, None, None, "ZERO_SPAN"

        pos = (closes[-1] - lower) / span

        long_ok,  _ = donchian_htf_filter(klines, "LONG",  long_cut=LONG_CUT, short_cut=SHORT_CUT)
        short_ok, _ = donchian_htf_filter(klines, "SHORT", long_cut=LONG_CUT, short_cut=SHORT_CUT)

        return pos, long_ok, short_ok, None

    except Exception as e:
        return None, None, None, str(e)[:60]


def zone_label(pos):
    if pos is None:
        return "UNKNOWN"
    if pos < LONG_CUT:
        return "HTF-DOWN  (LONG blocked)"
    elif pos > SHORT_CUT:
        return "HTF-UP    (SHORT blocked)"
    else:
        return "MID-CHANNEL (both sides open)"


def pos_bar(pos, width=20):
    if pos is None:
        return "[" + "?" * width + "]"
    filled = int(pos * width)
    bar = "─" * filled + "●" + "─" * (width - filled - 1)
    return f"[{bar}]  {pos:.2f}"


def get_trade_counts(days: int):
    """Trade count: last N days vs prior N days."""
    now   = datetime.now(timezone.utc)
    cut1  = now - timedelta(days=days)
    cut2  = now - timedelta(days=days * 2)

    with db_conn() as conn:
        cur = conn.cursor()

        # Last N days
        cur.execute("""
            SELECT COUNT(*) FROM trades
            WHERE created_at >= %s
              AND signal_type = 'MINI_ASYM'
              AND entry_type = 'T1'
        """, (cut1,))
        recent = cur.fetchone()[0]

        # Prior N days
        cur.execute("""
            SELECT COUNT(*) FROM trades
            WHERE created_at >= %s AND created_at < %s
              AND signal_type = 'MINI_ASYM'
              AND entry_type = 'T1'
        """, (cut2, cut1))
        prior = cur.fetchone()[0]

    return recent, prior


def get_recent_trades(days: int):
    """Recent trades with entry price — to spot-check DC alignment."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT t.symbol, t.side, t.entry_price, t.created_at,
                   t.outcome, t.pnl_pct
            FROM trades t
            WHERE t.created_at >= %s
              AND t.signal_type = 'MINI_ASYM'
              AND t.entry_type = 'T1'
            ORDER BY t.created_at DESC
            LIMIT 20
        """, (since,))
        return cur.fetchall()


def main():
    args = parse_args()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{SEP}")
    print(f"  DC FILTER LIVE HEALTH CHECK")
    print(f"  Run at : {now_str}")
    print(f"  Thresholds : LONG<{LONG_CUT}  SHORT>{SHORT_CUT}  (v035 production)")
    print(SEP)

    # ── 1. Flag status ──────────────────────────────────────────────────────
    print("\n── 1. FLAG STATUS ──────────────────────────────────────────────────")
    try:
        import main as svc
        flag = getattr(svc, "ENABLE_DC_FILTER", "NOT_FOUND")
        status = "✅ ON" if flag is True else "❌ OFF"
        print(f"  ENABLE_DC_FILTER = {flag}  →  {status}")
    except Exception as e:
        print(f"  Could not read main.py flag: {e}")
        print(f"  (Check manually — flag should be True)")

    # ── 2. Current DC state per coin ───────────────────────────────────────
    print(f"\n── 2. CURRENT DC STATE  (4h, 20-period Donchian) ──────────────────")
    print(f"  {'COIN':<12}  {'POSITION':<28}  {'ZONE'}")
    print(f"  {SEP2[:60]}")

    blocked_long  = []
    blocked_short = []
    both_open     = []
    errors        = []

    for coin in COINS:
        pos, long_ok, short_ok, err = get_dc_state(coin, args.exchange)
        if err:
            errors.append((coin, err))
            print(f"  {coin:<12}  {'ERROR':<28}  {err}")
            continue

        bar   = pos_bar(pos)
        label = zone_label(pos)
        print(f"  {coin:<12}  {bar:<28}  {label}")

        if not long_ok and short_ok:
            blocked_long.append(coin)
        elif long_ok and not short_ok:
            blocked_short.append(coin)
        else:
            both_open.append(coin)

    print(f"\n  Summary:")
    print(f"    LONG  blocked : {len(blocked_long):2d} coins  — {', '.join(blocked_long) or 'none'}")
    print(f"    SHORT blocked : {len(blocked_short):2d} coins  — {', '.join(blocked_short) or 'none'}")
    print(f"    Both open     : {len(both_open):2d} coins  — {', '.join(both_open) or 'none'}")
    if errors:
        print(f"    Errors        : {len(errors):2d} coins  — {', '.join(c for c,_ in errors)}")

    # ── 3. Trade frequency ─────────────────────────────────────────────────
    print(f"\n── 3. TRADE FREQUENCY  (last {args.days}d vs prior {args.days}d) ──────────────")
    try:
        recent, prior = get_trade_counts(args.days)
        delta = recent - prior
        pct   = ((recent - prior) / prior * 100) if prior > 0 else float("inf")
        arrow = "▼" if delta < 0 else "▲" if delta > 0 else "="
        print(f"  Prior  {args.days}d : {prior:3d} unique entries")
        print(f"  Last   {args.days}d : {recent:3d} unique entries  {arrow} {delta:+d} ({pct:+.0f}%)")
        if delta < 0:
            print(f"  → DC filter is reducing trade count ✓ (less noise)")
        elif recent == prior == 0:
            print(f"  → No trades yet — filter just deployed, check back in a few days")
        else:
            print(f"  → Trade count unchanged or rising — filter may not be active")
    except Exception as e:
        print(f"  DB error: {e}")

    # ── 4. Recent trades ───────────────────────────────────────────────────
    print(f"\n── 4. RECENT TRADES  (last {args.days} days) ──────────────────────────────")
    try:
        rows = get_recent_trades(args.days)
        if not rows:
            print(f"  No trades in last {args.days} days.")
        else:
            print(f"  {'DATE':<20} {'COIN':<12} {'SIDE':<6} {'OUTCOME':<18} {'PnL%':>6}")
            print(f"  {SEP2[:64]}")
            for sym, side, entry, ts, outcome, pnl in rows:
                ts_str  = ts.strftime("%Y-%m-%d %H:%M") if ts else "?"
                out_str = (outcome or "open")[:17]
                pnl_str = f"{pnl:+.2f}%" if pnl is not None else "open"
                print(f"  {ts_str:<20} {sym:<12} {side:<6} {out_str:<18} {pnl_str:>7}")
    except Exception as e:
        print(f"  DB error: {e}")

    # ── 5. Verdict ────────────────────────────────────────────────────────
    print(f"\n── 5. VERDICT ──────────────────────────────────────────────────────")
    blocked_total = len(blocked_long) + len(blocked_short)
    if blocked_total == 0 and not errors:
        print("  ⚠️  All coins open in both directions — market is mid-channel.")
        print("     DC filter is live but not currently blocking anything.")
        print("     This is normal in trending / mid-range markets.")
    elif blocked_total > 0:
        print(f"  ✅ DC filter is active — blocking {blocked_total}/10 coins in one direction.")
    elif errors:
        print(f"  ⚠️  {len(errors)} coin(s) returned errors — check exchange connectivity.")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
