#!/usr/bin/env python3
"""
fetch_new_coins.py
──────────────────
Populates backtest_candles for 10 additional coins so run_full_benchmark.py
can test 20 coins instead of 10.

Fetches from OKX (same source as existing candle data).
Runs sequentially — takes ~30-60 min on Render.

Usage:
  python3 fetch_new_coins.py
  nohup python3 fetch_new_coins.py >> fetch_coins.log 2>&1 &

After this completes, add the NEW_COINS list to run_full_benchmark.py COINS.
"""

import os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from backtester import prefetch_all_candles, check_candle_coverage, init_backtest_tables

init_backtest_tables()

NEW_COINS = [
    "DOTUSDT",   # Polkadot   — launched 2020
    "LTCUSDT",   # Litecoin   — 2013 (long history)
    "UNIUSDT",   # Uniswap    — 2020
    "TRXUSDT",   # Tron       — 2017
    "ATOMUSDT",  # Cosmos     — 2019
    "MATICUSDT", # Polygon    — 2019
    "ETCUSDT",   # Ethereum Classic — 2016
    "SANDUSDT",  # Sandbox    — 2020
    "FILUSDT",   # Filecoin   — 2020
    "ARBUSDT",   # Arbitrum   — 2022 (partial IS, full OOS)
]

DATE_FROM = "2020-01-01"
DATE_TO   = "2026-11-01"
STYLE     = "DAY_TRADE"

SEP = "=" * 60

def main():
    t0 = time.time()
    print(f"\n{SEP}")
    print(f"  FETCH NEW COINS — {len(NEW_COINS)} coins")
    print(f"  Range: {DATE_FROM} → {DATE_TO}")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(SEP)
    print()

    success = []
    failed  = []

    for i, coin in enumerate(NEW_COINS, 1):
        print(f"  [{i}/{len(NEW_COINS)}] {coin} ...", flush=True)
        try:
            prefetch_all_candles([coin], DATE_FROM, DATE_TO, style=STYLE)
            # Verify coverage
            ok, msg = check_candle_coverage(coin, DATE_FROM, DATE_TO, style=STYLE)
            if ok:
                print(f"    ✓ {coin} — {msg}")
                success.append(coin)
            else:
                print(f"    ⚠ {coin} — {msg}")
                success.append(coin)  # partial data, still usable
        except Exception as e:
            print(f"    ✗ {coin} — ERROR: {e}")
            failed.append((coin, str(e)))

    elapsed = (time.time() - t0) / 60
    print(f"\n{SEP}")
    print(f"  DONE — {elapsed:.1f} min")
    print(f"  Success: {len(success)}/{ len(NEW_COINS)}: {', '.join(success)}")
    if failed:
        print(f"  Failed:  {len(failed)}: {', '.join(c for c,_ in failed)}")
    print()
    print(f"  Next: add successful coins to COINS list in run_full_benchmark.py")
    print(SEP)


if __name__ == "__main__":
    main()
