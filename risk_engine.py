"""
risk_engine.py — Trade dataclass, risk presets, and the Mini-Asym risk engine.

Pure functions only — no FastAPI, no DB, no exchange calls.
Imported by main.py and engine/route modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import RiskMode, Side, START_EQUITY


@dataclass
class Trade:
    time: str
    side: Side
    symbol: str
    mode: RiskMode
    size: float
    sl: float
    tp: float
    leverage: float
    entry_price: float
    current_price: float
    unreal_pnl_percent: float
    unreal_pnl_value: float
    equity_after: float
    reason: Optional[str] = None
    is_primary: int = 1


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def presets_for_mode(mode: RiskMode) -> Dict[str, float]:
    return {
        "ULTRA_SAFE": dict(size=0.30, sl=0.35, tp=0.60, leverage=2),
        "SAFE": dict(size=0.45, sl=0.45, tp=0.80, leverage=3),
        "NORMAL": dict(size=0.60, sl=0.55, tp=1.00, leverage=5),
        "MINI_ASYM": dict(size=0.65, sl=0.55, tp=1.10, leverage=6),
        "AGGRESSIVE": dict(size=0.85, sl=0.70, tp=1.50, leverage=8),
    }[mode]


def default_max_trades_per_day(mode: RiskMode) -> int:
    return {"ULTRA_SAFE": 1, "SAFE": 2, "NORMAL": 3, "MINI_ASYM": 3, "AGGRESSIVE": 5}[mode]


def build_reason(
    mode: RiskMode,
    equity: float,
    computed: Dict[str, float],
    extra: Optional[str] = None,
    start_capital: Optional[float] = None,
) -> str:
    ref = start_capital if (start_capital and start_capital > 0) else START_EQUITY
    from_start = equity - ref
    dir_word = "up" if from_start >= 0 else "down"
    pct = abs(from_start / ref * 100) if ref > 0 else 0.0
    size_dollar = equity * computed["size"]
    effective = size_dollar * computed["leverage"]
    # Reducer only fires when actually UP 10%+ (not on any 10%+ move including losses)
    reduced = from_start >= 0 and pct >= 10.0

    lines = [
        f"{mode}  •  Manual trade",
        "",
        "Account",
        f"  Equity:      ${equity:,.2f}  ({dir_word} {pct:.1f}% from ${ref:,.2f} start)",
        "",
        "Position",
        f"  Size:        {computed['size'] * 100:.1f}% of equity  →  ${size_dollar:.2f} at risk",
        f"  Leverage:    {computed['leverage']:.0f}×  →  ${effective:.2f} total exposure",
        f"  Stop loss:   {computed['sl']:.2f}% from entry",
        f"  Take profit: {computed['tp']:.2f}% from entry",
    ]
    if reduced:
        lines.append("")
        lines.append("Risk note: Equity up 10%+ from start — size & leverage reduced ~10% to protect profits.")

    if extra:
        lines.append("")
        lines.append(extra)

    return "\n".join(lines)


def mini_asym_risk_engine(
    mode: RiskMode,
    equity: float,
    start_capital: Optional[float] = None,
) -> Dict[str, Any]:
    p = presets_for_mode(mode).copy()
    ref = start_capital if (start_capital and start_capital > 0) else START_EQUITY
    growth = (equity - ref) / ref if ref > 0 else 0.0
    if growth >= 0.10:
        p["size"] *= 0.9
        p["leverage"] *= 0.9
    computed = {
        "size": clamp(p["size"], 0.10, 1.50),
        "sl": clamp(p["sl"], 0.10, 2.50),
        "tp": clamp(p["tp"], 0.20, 5.00),
        "leverage": clamp(p["leverage"], 1.0, 10.0),
    }
    return {"allowed": True, "computed": computed, "reason": build_reason(mode, equity, computed, start_capital=ref)}
