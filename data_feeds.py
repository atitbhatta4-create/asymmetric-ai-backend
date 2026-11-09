"""External data feed helpers — purely standalone, zero imports from main.py.
Contains: weekly BTC candle fetch (for regime detection) + economic calendar fetch.
These are separated here so main.py / AutoRunner stay clean and testable in isolation."""

import json as _json
import urllib.request
from typing import Dict, List, Any

import httpx

# ── Bybit ────────────────────────────────────────────────────────────────────
_BYBIT_BASE = "https://api.bybit.com"
# Bybit V5 linear interval codes
_BYBIT_TF = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1W": "W"}


def _bybit_weekly_candles(limit: int = 210) -> List[Dict[str, Any]]:
    interval = _BYBIT_TF["1W"]
    try:
        r = httpx.get(
            f"{_BYBIT_BASE}/v5/market/kline",
            params={"category": "linear", "symbol": "BTCUSDT",
                    "interval": interval, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code != 200:
            return []
        items = ((r.json() or {}).get("result") or {}).get("list") or []
        if not items:
            return []
        # Bybit returns newest-first: [startTime, open, high, low, close, volume, turnover]
        out = [
            {"t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
            for k in items
        ]
        out.reverse()
        return out
    except Exception:
        return []


# ── OKX fallback ─────────────────────────────────────────────────────────────
_OKX_BASE = "https://www.okx.com"
_OKX_TF   = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D", "1W": "1W"}


def _okx_weekly_candles(limit: int = 210) -> List[Dict[str, Any]]:
    bar = _OKX_TF["1W"]
    try:
        r = httpx.get(
            f"{_OKX_BASE}/api/v5/market/candles",
            params={"instId": "BTC-USDT-SWAP", "bar": bar, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code != 200:
            return []
        rows = (r.json() or {}).get("data") or []
        if not rows:
            return []
        # OKX: [ts, open, high, low, close, vol, ...], newest-first
        out = [
            {"t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
            for k in rows
        ]
        out.reverse()
        return out
    except Exception:
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def _fetch_btc_weekly_candles(limit: int = 210) -> List[Dict[str, Any]]:
    """Fetch BTC weekly candles for regime detection.
    Tries Bybit first, falls back to OKX. Returns [] on total failure — callers
    degrade gracefully to TRENDING regime."""
    candles = _bybit_weekly_candles(limit)
    if candles:
        return candles
    return _okx_weekly_candles(limit)


def _fetch_economic_calendar() -> List[Dict[str, Any]]:
    """Fetch high-impact economic events from ForexFactory free JSON feed.
    Returns [] on any error so the caller never blocks on a dead API."""
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        req = urllib.request.Request(url, headers={"User-Agent": "AsymmetricAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode())
        return data if isinstance(data, list) else []
    except Exception:
        return []
