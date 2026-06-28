# Per-coin signal parameters.
# Values calibrated from comprehensive backtest (2020-2026) and optimizer results.
# Each coin has its own ADX range, score threshold, ATR limits, and entry geometry.
# Use get_coin_params(symbol) everywhere — do not hardcode thresholds per coin.

from typing import Dict, Any

COIN_PARAMS: Dict[str, Dict[str, Any]] = {
    "BTCUSDT": {
        "adx_min":                  18,
        "adx_max":                  55,
        "score_threshold":          0.63,
        "atr_max_pct":              4.5,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.0,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.2,
        "ema_distance_max_pct":     1.7,
        "pullback_tolerance_pct":   0.5,
    },
    "ETHUSDT": {
        "adx_min":                  17,
        "adx_max":                  58,
        "score_threshold":          0.64,
        "atr_max_pct":              5.0,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.2,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.25,
        "ema_distance_max_pct":     2.0,
        "pullback_tolerance_pct":   0.6,
    },
    "ADAUSDT": {
        "adx_min":                  20,
        "adx_max":                  60,
        "score_threshold":          0.65,
        "atr_max_pct":              5.5,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.3,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.3,
        "ema_distance_max_pct":     2.0,
        "pullback_tolerance_pct":   0.6,
    },
    "SOLUSDT": {
        "adx_min":                  22,
        "adx_max":                  62,
        "score_threshold":          0.68,
        "atr_max_pct":              6.0,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.5,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.3,
        "ema_distance_max_pct":     2.2,
        "pullback_tolerance_pct":   0.7,
    },
    "BNBUSDT": {
        "adx_min":                  20,
        "adx_max":                  58,
        "score_threshold":          0.66,
        "atr_max_pct":              5.0,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.2,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.25,
        "ema_distance_max_pct":     2.0,
        "pullback_tolerance_pct":   0.6,
    },
    "XRPUSDT": {
        # Optimizer: AGGRESSIVE SWING showed +119% in 2020, best with tight SL.
        # Needs stricter filters — news-sensitive, spike-prone.
        "adx_min":                  24,
        "adx_max":                  65,
        "score_threshold":          0.70,
        "atr_max_pct":              5.5,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.5,
        "sl_multiplier":            0.9,
        "volume_confirmation_mult": 1.35,
        "ema_distance_max_pct":     2.2,
        "pullback_tolerance_pct":   0.7,
    },
    "AVAXUSDT": {
        # Consistently negative in backtest except SAFE SWING optimized.
        # Very high thresholds to enforce selectivity.
        "adx_min":                  25,
        "adx_max":                  65,
        "score_threshold":          0.71,
        "atr_max_pct":              6.0,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.8,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.4,
        "ema_distance_max_pct":     2.5,
        "pullback_tolerance_pct":   0.8,
    },
    "LINKUSDT": {
        # Most consistently negative coin in backtest. Highest filters of all coins.
        # Only trades when conditions are extremely clean.
        "adx_min":                  26,
        "adx_max":                  68,
        "score_threshold":          0.72,
        "atr_max_pct":              6.5,
        "atr_min_pct":              0.1,
        "tp_multiplier":            3.0,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.4,
        "ema_distance_max_pct":     2.5,
        "pullback_tolerance_pct":   0.8,
    },
    "DOGEUSDT": {
        # Meme coin — high volatility, large ATR tolerance, needs volume confirmation.
        "adx_min":                  26,
        "adx_max":                  70,
        "score_threshold":          0.73,
        "atr_max_pct":              8.0,
        "atr_min_pct":              0.1,
        "tp_multiplier":            3.0,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.5,
        "ema_distance_max_pct":     2.8,
        "pullback_tolerance_pct":   1.0,
    },
    "DOTUSDT": {
        "adx_min":                  25,
        "adx_max":                  65,
        "score_threshold":          0.71,
        "atr_max_pct":              6.0,
        "atr_min_pct":              0.1,
        "tp_multiplier":            2.8,
        "sl_multiplier":            1.0,
        "volume_confirmation_mult": 1.4,
        "ema_distance_max_pct":     2.5,
        "pullback_tolerance_pct":   0.8,
    },
}

# Default fallback — used for any symbol not in the table.
# Uses BTC params as a safe conservative baseline.
_DEFAULT_PARAMS = COIN_PARAMS["BTCUSDT"]


def get_coin_params(symbol: str) -> Dict[str, Any]:
    """Return per-coin signal parameters for the given symbol.

    Strips exchange suffixes and normalises to the COIN_PARAMS keys.
    e.g. "BTC/USDT", "btcusdt", "BTC-USDT" → "BTCUSDT"
    Falls back to BTC params for unknown symbols.
    """
    clean = symbol.replace("/", "").replace("-", "").upper()
    return COIN_PARAMS.get(clean, _DEFAULT_PARAMS)
