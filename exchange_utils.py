"""
exchange_utils.py — CCXT wrappers, per-exchange fee table, and direct
balance fetchers for Bybit, OKX, and Binance.

Imported by main.py and the engine. No FastAPI routes live here.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
from datetime import datetime, timezone as _tz
from typing import Dict, Optional

import httpx

from user_state import get_exchange


# ── Per-exchange fee table (futures/perpetuals, USD-margined) ─────────────────
# maker = limit order (you add liquidity)   taker = market order (you take liquidity)
# SL exits are always market (taker). TP exits can be limit (maker).
# Slippage estimate is conservative round-trip for <$20k position size.
EXCHANGE_FEES: Dict[str, Dict] = {
    "bybit": {
        "maker":     0.00020,   # 0.020%
        "taker":     0.00055,   # 0.055%
        "slippage":  0.00025,   # ~0.025% round-trip (BTC liquid, alts slightly more)
        "min_qty":   0.001,     # BTC minimum order size
    },
    "okx": {
        "maker":     0.00020,   # 0.020%
        "taker":     0.00050,   # 0.050%
        "slippage":  0.00030,   # ~0.030% round-trip
        "min_qty":   0.001,
    },
    "binance": {
        "maker":     0.00020,   # 0.020% (futures)
        "taker":     0.00040,   # 0.040% (futures, lower than spot)
        "slippage":  0.00020,   # ~0.020% round-trip (most liquid exchange)
        "min_qty":   0.001,
    },
}


def exchange_fee_cost(exchange_id: str, size_dollar: float, outcome: str) -> float:
    """
    Calculate total fee + slippage cost for one complete trade (entry + exit).
    outcome: "TP_HIT" (limit exit) | "SL_HIT" | "TRAIL_STOP" | other (market exit)
    Returns dollar cost to deduct from PnL.
    """
    fees = EXCHANGE_FEES.get(exchange_id.lower(), EXCHANGE_FEES["bybit"])
    entry_fee = size_dollar * fees["maker"]           # entry always limit
    exit_fee  = size_dollar * (fees["maker"] if outcome == "TP_HIT" else fees["taker"])
    slippage  = size_dollar * fees["slippage"]
    return round(entry_fee + exit_fee + slippage, 4)


# ── CCXT helpers ──────────────────────────────────────────────────────────────

def _make_ccxt_exchange(row: dict):
    """Build a ccxt exchange instance from stored (decrypted) keys."""
    import ccxt
    exchange_id = (row.get("exchange") or "bybit").lower()
    config = {
        "apiKey":   row["api_key"],
        "secret":   row["api_secret"],
        "options":  {"defaultType": "linear"},
        "enableRateLimit": True,
        "timeout":  10000,   # 10-second timeout on every ccxt network call
    }
    if row.get("passphrase"):
        config["password"] = row["passphrase"]  # required for OKX
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    return cls(config)


def _ccxt_call(fn, *args, label: str = "exchange", retries: int = 1, **kwargs):
    """
    Wrap any ccxt call with timeout detection, 1 retry, and clear error logging.
    Raises a RuntimeError with a specific EXCHANGE_TIMEOUT prefix on timeout.
    Never crashes the engine — caller catches RuntimeError and continues.
    """
    import ccxt
    last_exc = None
    for attempt in range(1, retries + 2):  # attempts = retries + 1
        try:
            return fn(*args, **kwargs)
        except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
            last_exc = e
            is_timeout = isinstance(e, ccxt.RequestTimeout) or "timeout" in str(e).lower()
            tag = "EXCHANGE_TIMEOUT" if is_timeout else "EXCHANGE_NETWORK_ERROR"
            print(f"[{tag}] {label} attempt {attempt}: {e}", flush=True)
            if attempt <= retries:
                import time as _t
                _t.sleep(2)
        except ccxt.AuthenticationError as e:
            raise RuntimeError(f"EXCHANGE_AUTH_ERROR: API key rejected by exchange — {e}")
        except ccxt.ExchangeError as e:
            raise RuntimeError(f"EXCHANGE_ERROR ({label}): {e}")
        except Exception as e:
            raise RuntimeError(f"EXCHANGE_UNEXPECTED ({label}): {type(e).__name__}: {e}")
    raise RuntimeError(f"EXCHANGE_TIMEOUT: {label} did not respond within 10 seconds after {retries + 1} attempts")


# ── Bybit direct balance (V5 signed) ─────────────────────────────────────────

BYBIT_HOSTS = ["https://api.bybit.com", "https://api.bytick.com"]


def _bybit_signed_get(api_key: str, api_secret: str, path: str, query: str) -> tuple[Optional[dict], str]:
    """Make a signed Bybit GET request, trying both hosts. Returns (parsed JSON or None, last_error)."""
    ts = str(int(time.time() * 1000))
    recv_window = "5000"
    sign_str = ts + api_key + recv_window + query
    sig = hmac.new(api_secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY":     api_key,
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-SIGN":        sig,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type":       "application/json",
    }
    last_error = "unknown"
    for host in BYBIT_HOSTS:
        try:
            resp = httpx.get(f"{host}{path}?{query}", headers=headers, timeout=10)
            print(f"[bybit] {host}{path} → HTTP {resp.status_code}")
            if resp.status_code == 200:
                return resp.json(), ""
            last_error = f"HTTP {resp.status_code} from {host}: {resp.text[:200]}"
            print(f"[bybit] non-200: {last_error}")
        except Exception as e:
            last_error = f"{type(e).__name__} from {host}: {e}"
            print(f"[bybit] exception: {last_error}")
    return None, last_error


def _bybit_direct_balance(api_key: str, api_secret: str) -> tuple:
    """
    Direct Bybit V5 signed request for USDT balance.
    Checks Unified Trading wallet first, then Funding wallet.
    Returns (balance_float, None) on success, (None, error_str) on failure.
    """
    try:
        # 1. Try Unified Trading wallet (required for trading)
        data, net_err = _bybit_signed_get(api_key, api_secret, "/v5/account/wallet-balance", "accountType=UNIFIED")
        if data is None:
            return (None, f"Could not reach Bybit API{' (' + net_err + ')' if net_err else ''}")
        ret_code = data.get("retCode")
        ret_msg  = data.get("retMsg", "")
        print(f"[bybit-balance] UNIFIED retCode={ret_code} retMsg={ret_msg}")
        if ret_code in (10003, 10004):
            return (None, f"Invalid API key or secret (retCode={ret_code}: {ret_msg})")
        if ret_code != 0:
            return (None, f"Bybit API error retCode={ret_code}: {ret_msg}")
        for acc in data.get("result", {}).get("list", []):
            for coin in acc.get("coin", []):
                if coin.get("coin") == "USDT":
                    val = float(coin.get("walletBalance") or coin.get("equity") or 0)
                    print(f"[bybit-balance] USDT in Unified Trading: {val}")
                    return (val, None)

        # 2. USDT not in Unified — check Funding wallet
        print("[bybit-balance] No USDT in Unified, checking Funding wallet...")
        fund_data, _ = _bybit_signed_get(api_key, api_secret, "/v5/asset/transfer/query-account-coins-balance", "accountType=FUND&coin=USDT")
        if fund_data and fund_data.get("retCode") == 0:
            for item in fund_data.get("result", {}).get("balance", []):
                if item.get("coin") == "USDT":
                    val = float(item.get("walletBalance") or item.get("transferBalance") or 0)
                    print(f"[bybit-balance] USDT in Funding wallet: {val}")
                    if val > 0:
                        return (None, f"Your {val:.2f} USDT is in the Funding wallet — transfer it to Unified Trading in Bybit before the engine can trade.")
        return (None, "No USDT found in Unified Trading or Funding wallet. Please deposit USDT to your Bybit account.")
    except Exception as ex:
        return (None, f"Exception: {ex}")


# ── OKX direct balance ────────────────────────────────────────────────────────

def _okx_direct_balance(api_key: str, api_secret: str, passphrase: str) -> Optional[float]:
    """Direct OKX V5 signed request for USDT balance."""
    ts = datetime.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%S.') + \
         str(datetime.now(_tz.utc).microsecond // 1000).zfill(3) + 'Z'
    path = "/api/v5/account/balance"
    sign_str = ts + "GET" + path + ""
    sig = base64.b64encode(
        hmac.new(api_secret.encode(), sign_str.encode(), hashlib.sha256).digest()
    ).decode()
    headers = {
        "OK-ACCESS-KEY":        api_key,
        "OK-ACCESS-SIGN":       sig,
        "OK-ACCESS-TIMESTAMP":  ts,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type":         "application/json",
    }
    try:
        resp = httpx.get(f"https://www.okx.com{path}", headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            for detail in data.get("data", [{}])[0].get("details", []):
                if detail.get("ccy") == "USDT":
                    return float(detail.get("cashBal") or detail.get("availBal") or 0)
    except Exception:
        pass
    return None


# ── Binance direct balance ────────────────────────────────────────────────────

def _binance_direct_balance(api_key: str, api_secret: str) -> Optional[float]:
    """Direct Binance Futures V2 signed request for USDT balance."""
    ts = str(int(time.time() * 1000))
    params = f"timestamp={ts}"
    sig = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    try:
        resp = httpx.get(
            f"https://fapi.binance.com/fapi/v2/balance?{params}&signature={sig}",
            headers=headers, timeout=10,
        )
        data = resp.json()
        if isinstance(data, list):
            for item in data:
                if item.get("asset") == "USDT":
                    return float(item.get("balance") or item.get("availableBalance") or 0)
    except Exception:
        pass
    return None


# ── Cached real balance with background refresh ───────────────────────────────

_REAL_BAL_CACHE: Dict[str, tuple] = {}      # email → (balance, fetched_at_ts)
_REAL_BAL_TTL = 30                           # seconds before background refresh fires
_REAL_BAL_REFRESHING: set = set()            # emails with a refresh already in flight


def _do_balance_refresh(email: str) -> None:
    """Background thread: fetch live balance and silently update cache."""
    try:
        row = get_exchange(email)
        if not row:
            return
        ex_id = (row.get("exchange") or "bybit").lower()
        if ex_id == "bybit":
            bal, _ = _bybit_direct_balance(row["api_key"], row["api_secret"])
        elif ex_id == "okx":
            bal = _okx_direct_balance(row["api_key"], row["api_secret"], row.get("passphrase") or "")
        elif ex_id == "binance":
            bal = _binance_direct_balance(row["api_key"], row["api_secret"])
        else:
            bal = None
        if bal is not None:
            _REAL_BAL_CACHE[email] = (bal, time.time())
    finally:
        _REAL_BAL_REFRESHING.discard(email)


def get_real_usdt_balance(email: str, force: bool = False) -> Optional[float]:
    """Return cached balance instantly; refresh from exchange in background when stale.
    force=True blocks for a fresh fetch (used before runner init)."""
    cached = _REAL_BAL_CACHE.get(email)

    if not force and cached:
        bal, fetched_at = cached
        if time.time() - fetched_at >= _REAL_BAL_TTL and email not in _REAL_BAL_REFRESHING:
            _REAL_BAL_REFRESHING.add(email)
            threading.Thread(target=_do_balance_refresh, args=(email,), daemon=True).start()
        return bal  # always instant when any cached value exists

    # No cache yet or force=True — block for fresh data
    row = get_exchange(email)
    if not row:
        return None
    ex_id = (row.get("exchange") or "bybit").lower()
    if ex_id == "bybit":
        bal, _ = _bybit_direct_balance(row["api_key"], row["api_secret"])
    elif ex_id == "okx":
        bal = _okx_direct_balance(row["api_key"], row["api_secret"], row.get("passphrase") or "")
    elif ex_id == "binance":
        bal = _binance_direct_balance(row["api_key"], row["api_secret"])
    else:
        bal = None
    if bal is not None:
        _REAL_BAL_CACHE[email] = (bal, time.time())
    return bal
