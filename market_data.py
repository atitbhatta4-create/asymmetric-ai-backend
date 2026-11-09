"""
market_data.py — Price feeds, klines, tickers, orderbook, symbol lists.

All external data fetching lives here. No FastAPI routes — pure functions only.
main.py imports these and wires them to endpoints.
"""
from __future__ import annotations

import re as _re
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from config import OKX_BASE, OKX_TF_MAP

# ── Bybit constants ───────────────────────────────────────────────────────────
BYBIT_BASE   = "https://api.bybit.com"
BYBIT_TF_MAP = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}

# ── MEXC constants ────────────────────────────────────────────────────────────
_MEXC_TF_MAP = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}

# ── Klines cache — shared across all users on same symbol+tf ─────────────────
_KLINES_CACHE: Dict[tuple, tuple] = {}   # (symbol, tf, exchange) → (fetched_at, klines)
_KLINES_CACHE_LOCK = threading.Lock()
_KLINES_CACHE_TTL  = 55  # refresh just under 1 minute

# ── Symbol list cache ─────────────────────────────────────────────────────────
_symbols_cache: Dict[str, Any] = {}  # exchange → {"ts": float, "symbols": list}
_SYMBOLS_TTL = 300  # 5 minutes

# ── Input validation ──────────────────────────────────────────────────────────
_EXCHANGE_ALLOWLIST = {"bybit", "okx", "binance"}
_SYMBOL_RE = _re.compile(r'^[A-Z0-9]{2,20}$')


def _validate_exchange_param(ex: Optional[str]) -> str:
    v = (ex or "okx").lower().strip()
    if v not in _EXCHANGE_ALLOWLIST:
        raise HTTPException(status_code=400, detail=f"Unknown exchange '{v}'. Use: bybit, okx, binance")
    return v


def _validate_symbol(v: str) -> str:
    return v.upper().strip()


def _validate_path_symbol(symbol: str) -> str:
    sym = symbol.upper().strip()
    if not _SYMBOL_RE.match(sym):
        raise HTTPException(status_code=400, detail=f"Invalid symbol '{sym}' — use format like BTCUSDT")
    return sym


# ── Symbol format converters ──────────────────────────────────────────────────

def to_okx_inst(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if "-" in s:
        return s
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
    return s


def to_bybit_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if "-" in s:
        return s.replace("-", "")
    return s


# ── Price feeds ───────────────────────────────────────────────────────────────

async def _fetch_price_binance_us(symbol: str) -> float:
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": symbol.upper()},
            headers={"accept": "application/json"},
        )
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:120]}")
    price = r.json().get("price")
    if price is None:
        raise ValueError("no price field")
    return float(price)


async def _fetch_price_mexc(symbol: str) -> float:
    """MEXC global exchange — no geo-blocking, same format as Binance."""
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(
            "https://api.mexc.com/api/v3/ticker/price",
            params={"symbol": symbol.upper()},
            headers={"accept": "application/json"},
        )
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:120]}")
    price = r.json().get("price")
    if price is None:
        raise ValueError("no price field")
    return float(price)


async def _fetch_price_okx(symbol: str) -> float:
    inst = to_okx_inst(symbol)
    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(
            f"{OKX_BASE}/api/v5/market/ticker",
            params={"instId": inst},
            headers={"accept": "application/json"},
        )
    if r.status_code == 429:
        raise ValueError("OKX 429 rate-limit — backing off")
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:120]}")
    arr = (r.json() or {}).get("data") or []
    if not arr:
        raise ValueError(f"no ticker for {inst}")
    last = arr[0].get("last")
    if last is None:
        raise ValueError(f"missing last price for {inst}")
    return float(last)


async def okx_price(symbol: str) -> float:
    """Fetch spot price with fallback: Binance US → MEXC → OKX."""
    errors: list = []
    for name, fn in [
        ("BinanceUS", _fetch_price_binance_us),
        ("MEXC",      _fetch_price_mexc),
        ("OKX",       _fetch_price_okx),
    ]:
        try:
            return await fn(symbol)
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"[price] {name} failed for {symbol}: {e}")
    raise HTTPException(
        status_code=503,
        detail=f"All price sources failed for {symbol} — {'; '.join(errors)}"
    )


async def bybit_price(symbol: str) -> float:
    sym = to_bybit_symbol(symbol)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{BYBIT_BASE}/v5/market/tickers",
            params={"category": "linear", "symbol": sym},
            headers={"accept": "application/json"},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Bybit price error: {r.text[:200]}")
    data = r.json() or {}
    items = (data.get("result") or {}).get("list") or []
    if not items:
        raise HTTPException(status_code=400, detail=f"Bybit: no ticker for {sym}")
    last = items[0].get("lastPrice")
    if last is None:
        raise HTTPException(status_code=400, detail=f"Bybit: missing lastPrice for {sym}")
    return float(last)


# price alias — okx_price already has BinanceUS/MEXC/OKX fallback chain
binance_price = okx_price


async def _route_price(symbol: str, exchange: str) -> float:
    return await okx_price(symbol)


# ── Klines fetchers ───────────────────────────────────────────────────────────

def _fetch_klines_raw(symbol: str, tf: str, limit: int) -> List[Dict[str, Any]]:
    """Direct OKX candle fetch — no cache, no fallback."""
    inst = to_okx_inst(symbol)
    bar  = OKX_TF_MAP.get(str(tf))
    if not bar:
        print(f"[klines-okx] unknown tf={tf!r} for {symbol}")
        return []
    try:
        r = httpx.get(
            f"{OKX_BASE}/api/v5/market/candles",
            params={"instId": inst, "bar": bar, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code == 429:
            print(f"[klines-okx] 429 rate-limit for {inst}/{bar}")
            return []
        if r.status_code != 200:
            print(f"[klines-okx] HTTP {r.status_code} for {inst}/{bar}: {r.text[:80]}")
            return []
        rows = (r.json() or {}).get("data") or []
        if not rows:
            print(f"[klines-okx] empty data for {inst}/{bar}")
            return []
        out: List[Dict[str, Any]] = []
        for k in rows:
            out.append({"t": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])})
        out.reverse()
        return out
    except Exception as e:
        print(f"[klines-okx] exception for {inst}/{bar}: {e}")
        return []


def _fetch_klines_bybit_sync(symbol: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    sym      = to_bybit_symbol(symbol)
    interval = BYBIT_TF_MAP.get(str(tf))
    if not interval:
        return []
    try:
        r = httpx.get(
            f"{BYBIT_BASE}/v5/market/kline",
            params={"category": "linear", "symbol": sym, "interval": interval, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code != 200:
            print(f"[klines-bybit] HTTP {r.status_code} for {sym}/{interval}: {r.text[:120]}")
            return []
        items = ((r.json() or {}).get("result") or {}).get("list") or []
        if not items:
            return []
        out: List[Dict[str, Any]] = []
        # Bybit returns: [startTime, open, high, low, close, volume, turnover]
        for k in items:
            out.append({
                "t":      int(k[0]),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            })
        out.reverse()
        return out
    except Exception:
        return []


def _fetch_klines_mexc_sync(symbol: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    interval = _MEXC_TF_MAP.get(str(tf))
    if not interval:
        print(f"[klines-mexc] unknown tf={tf!r} for {symbol}")
        return []
    sym = to_bybit_symbol(symbol)  # MEXC uses BTCUSDT format (no slash)
    try:
        r = httpx.get(
            "https://api.mexc.com/api/v3/klines",
            params={"symbol": sym, "interval": interval, "limit": int(limit)},
            timeout=12,
            headers={"accept": "application/json"},
        )
        if r.status_code != 200:
            print(f"[klines-mexc] HTTP {r.status_code} for {sym}/{interval}: {r.text[:80]}")
            return []
        rows = r.json() or []
        if not rows:
            return []
        out: List[Dict[str, Any]] = []
        # MEXC/Binance: [openTime, open, high, low, close, vol, closeTime, ...]
        for k in rows:
            out.append({
                "t":      int(k[0]),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            })
        return out  # already ascending order
    except Exception as e:
        print(f"[klines-mexc] exception for {sym}/{interval}: {e}")
        return []


def _route_klines(symbol: str, tf: str, exchange: str, limit: int) -> List[Dict[str, Any]]:
    """Route candle fetch to user's exchange first, fall back to 2 others.
    Bybit user: Bybit → OKX → MEXC
    OKX user:   OKX → Bybit → MEXC
    Binance user: MEXC → Bybit → OKX
    """
    ex = (exchange or "bybit").lower()
    if ex == "bybit":
        sources = [
            ("Bybit", lambda: _fetch_klines_bybit_sync(symbol, tf, limit=limit)),
            ("OKX",   lambda: _fetch_klines_raw(symbol, tf, limit)),
            ("MEXC",  lambda: _fetch_klines_mexc_sync(symbol, tf, limit=limit)),
        ]
    elif ex == "okx":
        sources = [
            ("OKX",   lambda: _fetch_klines_raw(symbol, tf, limit)),
            ("Bybit", lambda: _fetch_klines_bybit_sync(symbol, tf, limit=limit)),
            ("MEXC",  lambda: _fetch_klines_mexc_sync(symbol, tf, limit=limit)),
        ]
    else:  # binance — MEXC is the closest public substitute (same API format, no geo-block)
        sources = [
            ("MEXC",  lambda: _fetch_klines_mexc_sync(symbol, tf, limit=limit)),
            ("Bybit", lambda: _fetch_klines_bybit_sync(symbol, tf, limit=limit)),
            ("OKX",   lambda: _fetch_klines_raw(symbol, tf, limit)),
        ]

    primary = sources[0][0]
    for name, fn in sources:
        try:
            data = fn()
            if data:
                if name != primary:
                    print(f"[klines] {name} fallback used for {symbol}/{tf} (primary exchange={ex})")
                return data
        except Exception as e:
            print(f"[klines] {name} failed for {symbol}/{tf}: {e}")

    print(f"[klines] All 3 sources failed for {symbol}/{tf} (exchange={ex})")
    return []


def _fetch_klines_sync(symbol: str, tf: str, limit: int = 200, exchange: str = "bybit") -> List[Dict[str, Any]]:
    """Cached candle fetch — per exchange, with 3-source fallback.
    Cache key includes exchange so Bybit/OKX/Binance users each get their own slot."""
    ex  = (exchange or "bybit").lower()
    key = (symbol.upper(), tf, ex)
    now = time.time()
    with _KLINES_CACHE_LOCK:
        if key in _KLINES_CACHE:
            fetched_at, cached = _KLINES_CACHE[key]
            if now - fetched_at < _KLINES_CACHE_TTL and len(cached) >= limit:
                return cached
    fresh = _route_klines(symbol, tf, ex, limit)
    if fresh:
        with _KLINES_CACHE_LOCK:
            _KLINES_CACHE[key] = (now, fresh)
    return fresh


# ── Ticker 24h ────────────────────────────────────────────────────────────────

def _parse_binance_style_24h(d: dict) -> dict:
    """Parse Binance-format 24h ticker (used by Binance US and MEXC)."""
    last   = float(d.get("lastPrice") or 0)
    open24 = float(d.get("openPrice") or last or 1)
    high   = float(d.get("highPrice") or 0)
    low    = float(d.get("lowPrice") or 0)
    vol    = float(d.get("quoteVolume") or 0)
    chg    = ((last - open24) / open24 * 100) if open24 else 0.0
    return {"price": last, "open24h": open24, "high24h": high, "low24h": low,
            "change24h": round(chg, 3), "volume24h": round(vol, 2)}


async def _fetch_ticker_24h(symbol: str) -> dict:
    """24h stats: Binance US → MEXC → OKX."""
    sym = symbol.upper().strip()
    for name, url in [
        ("BinanceUS", "https://api.binance.us/api/v3/ticker/24hr"),
        ("MEXC",      "https://api.mexc.com/api/v3/ticker/24hr"),
    ]:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(url, params={"symbol": sym},
                                     headers={"accept": "application/json"})
            if r.status_code == 200:
                return _parse_binance_style_24h(r.json())
        except Exception as e:
            print(f"[ticker24h] {name} failed for {sym}: {e}")
    # OKX fallback
    try:
        inst = to_okx_inst(sym)
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                f"{OKX_BASE}/api/v5/market/ticker",
                params={"instId": inst},
                headers={"accept": "application/json"},
            )
        if r.status_code == 200:
            d = ((r.json() or {}).get("data") or [{}])[0]
            last   = float(d.get("last") or 0)
            open24 = float(d.get("open24h") or last or 1)
            high   = float(d.get("high24h") or 0)
            low    = float(d.get("low24h") or 0)
            vol    = float(d.get("volCcy24h") or 0)
            chg    = ((last - open24) / open24 * 100) if open24 else 0.0
            return {"price": last, "open24h": open24, "high24h": high, "low24h": low,
                    "change24h": round(chg, 3), "volume24h": round(vol, 2)}
    except Exception as e:
        print(f"[ticker24h] OKX failed for {sym}: {e}")
    return {"price": 0, "open24h": 0, "high24h": 0, "low24h": 0, "change24h": 0.0, "volume24h": 0}


# ── Symbol lists ──────────────────────────────────────────────────────────────

async def _fetch_symbols_for_exchange(exchange: str) -> List[str]:
    """Return all active USDT perpetual trading pairs for the given exchange.

    Render deploys on US servers — Binance and Bybit geo-block some endpoints.
    OKX does not. We try the native exchange first, fall back to OKX's list.
    Coins are ~95% the same across all three exchanges.

    Returns a sorted list of symbols e.g. ["BTCUSDT", "ETHUSDT"]
    """
    ex  = exchange.lower()
    now = time.time()

    cached = _symbols_cache.get(ex)
    if cached and (now - cached["ts"]) < _SYMBOLS_TTL:
        return cached["symbols"]

    symbols: List[str] = []

    async with httpx.AsyncClient(timeout=12) as client:

        if ex == "binance":
            try:
                r = await client.get(
                    "https://fapi.binance.com/fapi/v1/exchangeInfo",
                    headers={"accept": "application/json"},
                )
                if r.status_code == 200:
                    data = r.json()
                    symbols = sorted([
                        s["symbol"] for s in (data.get("symbols") or [])
                        if s.get("quoteAsset") == "USDT"
                        and s.get("status") == "TRADING"
                        and s.get("contractType") == "PERPETUAL"
                    ])
            except Exception:
                pass

        elif ex == "bybit":
            try:
                r = await client.get(
                    f"{BYBIT_BASE}/v5/market/instruments-info",
                    params={"category": "linear", "limit": 1000},
                    headers={"accept": "application/json"},
                )
                if r.status_code == 200:
                    items = ((r.json() or {}).get("result") or {}).get("list") or []
                    symbols = sorted([
                        i["symbol"] for i in items
                        if i.get("quoteCoin") == "USDT"
                        and i.get("status") == "Trading"
                        and i.get("symbol", "").endswith("USDT")
                    ])
            except Exception:
                pass

        # OKX fallback (or primary if exchange == "okx")
        if not symbols:
            try:
                r = await client.get(
                    f"{OKX_BASE}/api/v5/public/instruments",
                    params={"instType": "SWAP"},
                    headers={"accept": "application/json"},
                )
                if r.status_code == 200:
                    items = (r.json() or {}).get("data") or []
                    symbols = sorted([
                        i["instId"].replace("-USDT-SWAP", "USDT")
                        for i in items
                        if i.get("instId", "").endswith("-USDT-SWAP")
                        and i.get("state") == "live"
                    ])
            except Exception:
                pass

    _symbols_cache[ex] = {"ts": now, "symbols": symbols}
    return symbols
