#!/usr/bin/env python3
"""
fetch_sentiment_data.py
───────────────────────
One-time setup: downloads all historical Fear & Greed data from
alternative.me and stores it in the sentiment_fng DB table.

Run this ONCE on Render before starting the sentiment backtest.
Safe to re-run — uses upsert.

Usage:
  python3 fetch_sentiment_data.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from sentiment import fetch_and_store_fng_history, load_fng_history

print("\nStep 7 — Sentiment Data Setup")
print("=" * 50)

count = fetch_and_store_fng_history(verbose=True)

if count > 0:
    history = load_fng_history()
    dates   = sorted(history.keys())
    print(f"\n  Coverage: {dates[0]} → {dates[-1]}")
    print(f"  Records : {len(history)}")
    print(f"\n  Sample scores:")
    for d in ["2020-03-13", "2021-11-08", "2022-06-18", "2024-03-05", "2026-01-01"]:
        s = history.get(d)
        print(f"    {d}: {s if s is not None else '(no data)'}")
    print(f"\n  ✓ Ready to run run_sentiment_backtest.py")
else:
    print("\n  ✗ No data stored — check network access to alternative.me")

print("=" * 50)
