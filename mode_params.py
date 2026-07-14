# Mode and style configuration tables.
# These define how each risk mode and trade style adjusts the base coin params.
# Applied in _signal_and_filters() after coin_params and regime are resolved.

from typing import Dict, Any, Optional

MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "ULTRA_SAFE": {
        "score_add":            +0.12,
        "adx_min_add":          +8,
        "volume_mult_add":      +0.30,
        "max_trades_per_day":   1,
        # ULTRA_SAFE trades in PARABOLIC at reduced size — strong bull markets are safe for LONG
        "allowed_regimes":      ["TRENDING", "PARABOLIC"],
        "tp_multiplier_override": 1.5,
        "position_size_mult":   0.50,
        "description": "Minimum risk. Very few trades. Absolute beginners.",
    },
    "SAFE": {
        "score_add":            +0.08,
        "adx_min_add":          +5,
        "volume_mult_add":      +0.20,
        "max_trades_per_day":   1,
        # SAFE allowed in PARABOLIC but at reduced size (handled by position_size_mult)
        "allowed_regimes":      ["TRENDING", "PARABOLIC"],
        "tp_multiplier_override": None,
        "position_size_mult":   0.70,
        "description": "Capital protection. Strictest filters. 1 trade per day max.",
    },
    "NORMAL": {
        "score_add":            +0.02,
        "adx_min_add":          +2,
        "volume_mult_add":      +0.05,
        "max_trades_per_day":   2,
        "allowed_regimes":      ["TRENDING", "PARABOLIC"],
        "tp_multiplier_override": None,
        "position_size_mult":   0.85,
        "description": "Balanced. Slightly conservative. Good for most users.",
    },
    "MINI_ASYM": {
        "score_add":            0.0,
        "adx_min_add":          0,
        "volume_mult_add":      0.0,
        "max_trades_per_day":   3,
        "allowed_regimes":      ["TRENDING", "PARABOLIC"],
        "tp_multiplier_override": None,
        "position_size_mult":   1.0,
        "description": "Flagship mode. Best balance of returns and safety.",
    },
    "AGGRESSIVE": {
        "score_add":            -0.04,
        "adx_min_add":          -3,
        "volume_mult_add":      -0.10,
        "max_trades_per_day":   5,
        "allowed_regimes":      ["TRENDING", "PARABOLIC"],
        "tp_multiplier_override": None,
        "position_size_mult":   1.0,
        "description": "Maximum returns. More trades. Experienced traders only.",
    },
}

STYLE_CONFIG: Dict[str, Dict[str, Any]] = {
    "DAY_TRADE": {
        "candle_timeframe":     "1h",
        "pullback_required":    True,
        "pullback_timeout":     3,       # candles to wait for pullback before cancelling
        "swing_weekly_filter":  False,
        "scalp_mode":           False,
        "ranging_adx_threshold": 12,    # 1h candles consolidate often; only block truly flat ADX
        "backtest_default_tp_mult": 2.0, # DT WR ~30% → needs TP 2.5× to break even; 2.0× is conservative fix
        "description": "1h candles. Trades last hours to 1 day.",
    },
    "SWING": {
        "candle_timeframe":     "4h",
        "pullback_required":    True,
        "pullback_timeout":     5,       # 4h × 5 = 20h window
        "swing_weekly_filter":  True,
        "scalp_mode":           False,
        "ranging_adx_threshold": 15,    # 4h candles: ADX <15 is genuinely choppy
        "description": "4h candles. Trades last 1-5 days.",
    },
    "SCALP": {
        "candle_timeframe":     "15m",
        "pullback_required":    False,   # too noisy to wait for pullback on 15m
        "pullback_timeout":     0,
        "swing_weekly_filter":  False,
        "scalp_mode":           True,
        "ranging_adx_threshold": 18,    # 15m is noisiest; stricter ranging filter
        "experimental":         True,
        "description": "EXPERIMENTAL. 15m candles. High frequency. Experienced only.",
    },
}


def get_mode_config(mode: str) -> Dict[str, Any]:
    return MODE_CONFIG.get(mode, MODE_CONFIG["MINI_ASYM"])


def get_style_config(style: str) -> Dict[str, Any]:
    return STYLE_CONFIG.get(style, STYLE_CONFIG["DAY_TRADE"])
