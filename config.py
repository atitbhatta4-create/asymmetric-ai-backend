"""
config.py — app-wide constants and env vars.

Imported by notifications.py, indicators.py, and eventually engine.py.
Nothing here imports from main.py — this is the bottom of the import chain.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Literal

# ── Environment ──────────────────────────────────────────────────────────────
ENV         = os.getenv("ENV", "dev").lower().strip()   # dev | prod
IS_PROD     = ENV == "prod"
REAL_TRADING = os.getenv("REAL_TRADING", "").lower() == "true"

FRONTEND_ORIGINS_RAW = os.getenv("FRONTEND_ORIGINS", "")
if FRONTEND_ORIGINS_RAW.strip():
    FRONTEND_ORIGINS = [x.strip() for x in FRONTEND_ORIGINS_RAW.split(",") if x.strip()]
else:
    FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# ── Email (Gmail SMTP) ────────────────────────────────────────────────────────
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# ── Telegram ──────────────────────────────────────────────────────────────────
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID",  "").strip()

# ── Exchange / OKX ───────────────────────────────────────────────────────────
OKX_BASE   = "https://www.okx.com"
OKX_TF_MAP = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
TF_MIN_INTERVAL: Dict[str, int] = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# ── Starting equity (paper/demo) ──────────────────────────────────────────────
START_EQUITY = float(os.environ.get("SIMULATED_EQUITY", "1000"))

# ── Type aliases ─────────────────────────────────────────────────────────────
RiskMode   = Literal["ULTRA_SAFE", "SAFE", "NORMAL", "MINI_ASYM", "AGGRESSIVE"]
Side       = Literal["LONG", "SHORT"]
TF         = Literal["15m", "1h", "4h", "1d"]
TradeStyle = Literal["SCALP", "DAY_TRADE", "SWING"]
TF_MAP: Dict[str, str] = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
HIGHER_TF_MAP: Dict[str, str] = {"15m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}

# ── Trading style params ──────────────────────────────────────────────────────
# Each style sets the timeframe, candle interval, and ATR multiples for SL/TP.
TRADE_STYLE_PARAMS: Dict[str, Dict] = {
    "SCALP":     {"tf": "15m", "interval": 900,   "sl_atr": 0.8, "tp_atr": 1.6, "sl_max": 1.5, "tp_max": 3.0},
    "DAY_TRADE": {"tf": "1h",  "interval": 3600,  "sl_atr": 1.0, "tp_atr": 2.0, "sl_max": 2.5, "tp_max": 5.0},
    "SWING":     {"tf": "4h",  "interval": 14400, "sl_atr": 1.5, "tp_atr": 3.0, "sl_max": 4.0, "tp_max": 8.0},
}

# ── Mid-candle monitor thresholds ─────────────────────────────────────────────
_MID_CANDLE_THRESHOLDS: Dict[str, float] = {
    "SCALP":     0.010,
    "DAY_TRADE": 0.015,
    "SWING":     0.015,
}
_MID_CANDLE_MIN_SCORE = 0.75
_MID_CANDLE_INTERVAL  = 600   # seconds between mid-candle checks

# ── Dubai timezone ────────────────────────────────────────────────────────────
DUBAI_TZ = timezone(timedelta(hours=4))


def now_dubai() -> datetime:
    return datetime.now(tz=DUBAI_TZ)


def dubai_day_key(dt=None) -> str:
    d = dt or now_dubai()
    return d.strftime("%Y-%m-%d")
