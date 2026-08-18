"""
routes_portfolio.py — Portfolio, balance, trades, and risk-preview routes.

Includes: /risk/preview, /balance, /trades, /portfolio/stats,
          /portfolio/pnl, /portfolio/pnl/month/{year_month}, /portfolio/pnl/csv/{year}
"""
from __future__ import annotations

import io
import csv as _csv
import statistics as _st
from typing import Any, List, Optional
from datetime import timezone as _tz, timedelta as _td, datetime as _dt

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from auth_helpers import require_user
from config import REAL_TRADING, START_EQUITY, RiskMode, Side
from database import db_conn
from exchange_utils import get_real_usdt_balance
from market_data import _validate_symbol
from risk_engine import mini_asym_risk_engine
from user_state import get_equity, set_equity, get_session_id, _compute_floor

portfolio_router = APIRouter()


class RiskPreviewIn(BaseModel):
    symbol: str
    side: Side
    mode: RiskMode
    size: float = Field(..., gt=0, le=1_000_000)
    sl: float = Field(..., gt=0)
    tp: float = Field(..., gt=0)
    leverage: float = Field(..., ge=1, le=125)

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


@portfolio_router.post("/risk/preview")
def risk_preview(payload: RiskPreviewIn, user=Depends(require_user)):
    email = user["email"]
    equity = get_equity(email)

    # Read user's real starting capital so the preview shows accurate
    # growth/reduction relative to their actual first deposit, not the
    # paper trading $1000 default.
    start_capital: float = 0.0
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT starting_capital FROM user_state WHERE email = %s", (email,))
            row = cur.fetchone()
            if row:
                start_capital = float(row["starting_capital"] or 0)
    except Exception:
        pass

    out = mini_asym_risk_engine(payload.mode, equity, start_capital=start_capital or None)
    c = out["computed"]
    return {
        "allowed": True,
        "reason": out.get("reason"),
        "computed": {
            "symbol": payload.symbol.upper().strip(),
            "side": payload.side,
            "mode": payload.mode,
            "size": float(c["size"]),
            "sl": float(c["sl"]),
            "tp": float(c["tp"]),
            "leverage": float(c["leverage"]),
        },
    }


@portfolio_router.get("/balance")
def balance(user=Depends(require_user)):
    from main import AUTO_RUNNERS, AUTO_LOCK

    email = user["email"]

    real_bal = get_real_usdt_balance(email)
    if real_bal is not None:
        equity = real_bal
        db_equity = get_equity(email)
        if abs(db_equity - real_bal) > 0.01:
            set_equity(email, real_bal)
    elif REAL_TRADING:
        equity = 0.0
    else:
        equity = get_equity(email)

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT time FROM trades WHERE email = %s ORDER BY id DESC LIMIT 1", (email,))
        last_row = cur.fetchone()
        cur.execute(
            "SELECT peak_equity, all_time_high, floor_equity, starting_capital FROM user_state WHERE email = %s",
            (email,),
        )
        state_row = cur.fetchone()
        _db_peak = float(state_row["peak_equity"] or 0) if state_row else 0.0
        if _db_peak == 0:
            cur.execute(
                "SELECT MAX(equity_after) AS max_eq FROM trades WHERE email = %s",
                (email,),
            )
            _tp = cur.fetchone()
            _trade_peak = float((_tp or {}).get("max_eq") or 0)
            if _trade_peak > 0:
                _db_peak = _trade_peak
                _sc_fb = float((state_row or {}).get("starting_capital") or 0)
                _new_floor = _compute_floor(_db_peak, _sc_fb)
                cur.execute(
                    "UPDATE user_state SET peak_equity=%s, floor_equity=%s WHERE email=%s",
                    (_db_peak, _new_floor, email),
                )
                conn.commit()

    _sc = float(state_row["starting_capital"] or 0) if state_row else 0.0
    if real_bal is not None:
        start_eq = _sc if _sc > 0 else equity
    elif REAL_TRADING:
        start_eq = 0.0
    else:
        start_eq = START_EQUITY
    peak = _db_peak
    stored_floor = float(state_row["floor_equity"] or 0) if state_row else 0.0
    with AUTO_LOCK:
        _r = AUTO_RUNNERS.get(email)
    if _r and _r.is_running():
        peak = max(peak, _r.peak_equity)
        stored_floor = max(stored_floor, _r.floor_equity)
    if peak < equity:
        peak = equity
    hard_floor = round(max(peak * 0.85, stored_floor), 2)
    locked_profit = round(hard_floor - start_eq, 2)
    distance_to_floor = round(equity - hard_floor, 2)
    distance_pct = round(distance_to_floor / hard_floor * 100, 2) if hard_floor > 0 else 0.0
    ath = float(state_row["all_time_high"] or 0) if state_row else 0.0
    if ath < equity:
        ath = equity
    return {
        "total": equity,
        "equity_after_last_trade": equity,
        "last_trade": last_row["time"] if last_row else None,
        "start_equity": start_eq,
        "peak_equity": round(peak, 2),
        "hard_floor": hard_floor,
        "locked_profit": locked_profit,
        "distance_to_floor": distance_to_floor,
        "distance_pct": distance_pct,
        "all_time_high": round(ath, 2),
    }


@portfolio_router.get("/trades")
def trades(
    user=Depends(require_user),
    symbol: Optional[str] = Query(default=None),
    current_session: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=2000),
):
    email = user["email"]
    where = ["email = %s"]
    params: List[Any] = [email]

    if current_session:
        sid = get_session_id(email)
        where.append("session_id = %s")
        params.append(sid)

    if symbol:
        where.append("UPPER(symbol) = %s")
        params.append(symbol.strip().upper())

    q = f"SELECT * FROM trades WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT %s"
    params.append(int(limit))

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(q, tuple(params))
        rows = cur.fetchall()

    return {"trades": [dict(r) for r in rows]}


@portfolio_router.get("/portfolio/stats")
def portfolio_stats(user=Depends(require_user)):
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM trades WHERE email = %s ORDER BY time ASC", (email,))
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return {"empty": True}

    def _parse_dt(s):
        s = str(s).strip()
        iso = s if "T" in s else s.replace(" ", "T")
        if not iso.endswith("Z"):
            iso += "Z"
        try:
            return _dt.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            return None

    def _dubai_hour(s):
        dt = _parse_dt(s)
        if not dt:
            return -1
        return (dt + _td(hours=4)).hour

    def _session(h):
        if 2 <= h < 6:    return "Dead"
        if 6 <= h < 12:   return "Asia"
        if 12 <= h < 16:  return "London"
        if 16 <= h < 21:  return "London+NY"
        if 21 <= h <= 23: return "NY"
        return "NY-close"

    def _style(reason):
        r = str(reason or "")
        if "Scalp" in r or "SCALP" in r: return "SCALP"
        if "Swing" in r or "SWING" in r: return "SWING"
        return "DAY_TRADE"

    def _grade(trade: dict):
        stored = trade.get("grade")
        if stored in ("A", "B"):
            return stored
        return "B" if "Grade B" in str(trade.get("reason") or "") else "A"

    pnls     = [float(t["unreal_pnl_value"])   for t in rows]
    pnl_pcts = [float(t["unreal_pnl_percent"]) for t in rows]
    total    = len(rows)
    wins     = [t for t in rows if float(t["unreal_pnl_value"]) >= 0]
    losses   = [t for t in rows if float(t["unreal_pnl_value"]) <  0]
    win_rate = len(wins) / total if total else 0
    total_pnl = sum(pnls)

    first_eq = float(rows[0]["equity_after"]) - pnls[0]
    eq_curve = [{"time": str(rows[0]["time"]), "equity": round(first_eq, 2)}]
    for t in rows:
        eq_curve.append({"time": str(t["time"]), "equity": round(float(t["equity_after"]), 2)})

    peak = eq_curve[0]["equity"]
    max_dd = 0.0
    dd_curve = [{"time": eq_curve[0]["time"], "dd": 0.0}]
    for pt in eq_curve[1:]:
        if pt["equity"] > peak:
            peak = pt["equity"]
        dd = (peak - pt["equity"]) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        dd_curve.append({"time": pt["time"], "dd": round(-dd * 100, 3)})

    win_p  = [abs(p) for p in pnl_pcts if p >= 0]
    loss_p = [abs(p) for p in pnl_pcts if p <  0]
    avg_rr = ((_st.mean(win_p) if win_p else 0) / (_st.mean(loss_p) if loss_p else 1))

    monthly: dict = {}
    for t in rows:
        dt = _parse_dt(t["time"])
        if not dt:
            continue
        key = dt.strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"month": key, "pnl": 0.0, "trades": 0, "wins": 0}
        monthly[key]["pnl"]    += float(t["unreal_pnl_value"])
        monthly[key]["trades"] += 1
        if float(t["unreal_pnl_value"]) >= 0:
            monthly[key]["wins"] += 1
    monthly_list = sorted(monthly.values(), key=lambda x: x["month"])
    for m in monthly_list:
        m["pnl"] = round(m["pnl"], 2)
    best_month  = max(monthly_list, key=lambda x: x["pnl"])  if monthly_list else None
    worst_month = min(monthly_list, key=lambda x: x["pnl"]) if monthly_list else None

    streak_type  = "W" if pnls[-1] >= 0 else "L"
    streak_count = 0
    for p in reversed(pnls):
        if (p >= 0) == (streak_type == "W"):
            streak_count += 1
        else:
            break

    sessions: dict = {}
    for t in rows:
        h = _dubai_hour(t["time"])
        if h < 0:
            continue
        s = _session(h)
        if s not in sessions:
            sessions[s] = {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_history": []}
        sessions[s]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        sessions[s]["pnl"]  += pv
        sessions[s]["pnl_history"].append(round(pv, 2))
        if pv >= 0:
            sessions[s]["wins"] += 1
    for s in sessions.values():
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)

    grades: dict = {"A": 0, "B": 0}
    for t in rows:
        g = _grade(t)
        grades[g] = grades.get(g, 0) + 1

    syms: dict = {}
    for t in rows:
        sym = t["symbol"]
        if sym not in syms:
            syms[sym] = {"symbol": sym, "trades": 0, "wins": 0, "pnl": 0.0}
        syms[sym]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        syms[sym]["pnl"] += pv
        if pv >= 0:
            syms[sym]["wins"] += 1
    sym_list = sorted(syms.values(), key=lambda x: -x["trades"])
    for s in sym_list:
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)

    modes: dict = {}
    for t in rows:
        m = t["mode"]
        if m not in modes:
            modes[m] = {"mode": m, "trades": 0, "wins": 0, "pnl": 0.0}
        modes[m]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        modes[m]["pnl"] += pv
        if pv >= 0:
            modes[m]["wins"] += 1
    mode_list = sorted(modes.values(), key=lambda x: -x["trades"])
    for m in mode_list:
        m["win_rate"] = round(m["wins"] / m["trades"] * 100, 1) if m["trades"] else 0
        m["pnl"] = round(m["pnl"], 2)

    stls: dict = {}
    hold_map = {"SCALP": "~15m", "DAY_TRADE": "~1–2h", "SWING": "~4–12h"}
    for t in rows:
        st = _style(t.get("reason"))
        if st not in stls:
            stls[st] = {"style": st, "trades": 0, "wins": 0, "pnl": 0.0}
        stls[st]["trades"] += 1
        pv = float(t["unreal_pnl_value"])
        stls[st]["pnl"] += pv
        if pv >= 0:
            stls[st]["wins"] += 1
    style_list = sorted(stls.values(), key=lambda x: -x["trades"])
    for s in style_list:
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)
        s["avg_hold"] = hold_map.get(s["style"], "~1h")

    heatmap: dict = {h: {"wins": 0, "losses": 0} for h in range(24)}
    for t in rows:
        h = _dubai_hour(t["time"])
        if h < 0:
            continue
        if float(t["unreal_pnl_value"]) >= 0:
            heatmap[h]["wins"]   += 1
        else:
            heatmap[h]["losses"] += 1

    best_trade  = max(rows, key=lambda t: float(t["unreal_pnl_value"]))
    worst_trade = min(rows, key=lambda t: float(t["unreal_pnl_value"]))

    if len(pnls) > 1:
        mean_abs = _st.mean(abs(p) for p in pnls)
        std_pnl  = _st.stdev(pnls)
        cv       = std_pnl / mean_abs if mean_abs > 0 else 2.0
        consistency_score = max(0, min(100, int(100 - cv * 25)))
    else:
        consistency_score = 50

    wr_pts   = min(40, int(win_rate * 80))
    dd_pts   = max(0, int(30 * (1 - max_dd / 0.20)))
    con_pts  = int(consistency_score * 0.30)
    health_score = min(100, wr_pts + dd_pts + con_pts)

    daily: dict = {}
    for t in rows:
        dt = _parse_dt(t["time"])
        if not dt:
            continue
        key = dt.strftime("%Y-%m-%d")
        daily[key] = daily.get(key, 0.0) + float(t["unreal_pnl_percent"])
    daily_rets = list(daily.values())

    sharpe = sortino = calmar = 0.0
    if len(daily_rets) >= 2:
        mean_dr  = _st.mean(daily_rets)
        std_dr   = _st.stdev(daily_rets)
        neg_rets = [r for r in daily_rets if r < 0]
        sor_std  = _st.stdev(neg_rets) if len(neg_rets) > 1 else std_dr
        if std_dr:
            sharpe  = round(mean_dr * (365 ** 0.5) / std_dr, 2)
        if sor_std:
            sortino = round(mean_dr * (365 ** 0.5) / sor_std, 2)
        eq_start = eq_curve[0]["equity"]
        eq_end   = eq_curve[-1]["equity"]
        days     = max(1, len(daily_rets))
        ann_ret  = ((eq_end / eq_start) ** (365 / days) - 1) * 100 if eq_start > 0 else 0
        if max_dd > 0.001:
            calmar = round(ann_ret / (max_dd * 100), 2)

    mistakes = []
    eligible_sess = [(k, v) for k, v in sessions.items() if v["trades"] >= 2]
    if eligible_sess:
        worst_s = min(eligible_sess, key=lambda x: x[1]["win_rate"])
        if worst_s[1]["win_rate"] < 40:
            mistakes.append(
                f"Most losses during {worst_s[0]} session "
                f"({worst_s[1]['win_rate']:.0f}% win rate)"
            )
    eligible_modes = [m for m in mode_list if m["trades"] >= 2]
    if eligible_modes:
        worst_m = min(eligible_modes, key=lambda x: x["win_rate"])
        if worst_m["win_rate"] < win_rate * 100 * 0.8:
            mistakes.append(
                f"{worst_m['mode']} underperformed "
                f"({worst_m['win_rate']:.0f}% vs {win_rate*100:.0f}% avg)"
            )
    if not mistakes:
        mistakes.append("No significant loss pattern detected yet — keep trading!")

    streak_str = f"{streak_count} consecutive {'win' if streak_type=='W' else 'loss'}{'s' if streak_count!=1 else ''}"
    best_s_name = max(sessions.items(), key=lambda x: x[1]["win_rate"])[0] if sessions else None
    summary_parts = [
        f"{streak_str}",
        f"{win_rate*100:.0f}% win rate over {total} trades",
        f"{'Up' if total_pnl >= 0 else 'Down'} ${abs(total_pnl):.2f} all time",
    ]
    if best_s_name:
        summary_parts.append(f"Best during {best_s_name} session")
    ai_summary = " · ".join(summary_parts)

    return {
        "empty": False,
        "summary": {
            "total_pnl":         round(total_pnl, 2),
            "total_trades":      total,
            "win_rate":          round(win_rate * 100, 1),
            "avg_rr":            round(avg_rr, 2),
            "best_month":        best_month,
            "worst_month":       worst_month,
            "current_streak":    {"type": streak_type, "count": streak_count},
            "max_drawdown":      round(max_dd * 100, 2),
            "health_score":      health_score,
            "consistency_score": consistency_score,
        },
        "equity_curve":         eq_curve,
        "drawdown_curve":       dd_curve,
        "monthly_pnl":          monthly_list,
        "session_distribution": sessions,
        "grade_distribution":   grades,
        "symbol_breakdown":     sym_list[:10],
        "mode_breakdown":       mode_list,
        "style_breakdown":      style_list,
        "time_heatmap":         heatmap,
        "best_trade":           dict(best_trade),
        "worst_trade":          dict(worst_trade),
        "ai_summary":           ai_summary,
        "mistake_detection":    mistakes,
        "ratios":               {"sharpe": sharpe, "sortino": sortino, "calmar": calmar},
        "most_traded_symbol":   sym_list[0]["symbol"] if sym_list else "-",
    }


@portfolio_router.get("/portfolio/pnl")
def portfolio_pnl(user=Depends(require_user)):
    """Today / week / month PnL cards + full monthly history for the PnL section."""
    email = user["email"]
    now_utc = _dt.now(_tz.utc)
    dubai_now = now_utc + _td(hours=4)

    today_dubai = dubai_now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start  = today_dubai - _td(days=today_dubai.weekday())
    month_start = today_dubai.replace(day=1)

    def _to_utc(dt_dubai):
        return (dt_dubai - _td(hours=4)).strftime("%Y-%m-%dT%H:%M:%S")

    today_utc = _to_utc(today_dubai)
    week_utc  = _to_utc(week_start)
    month_utc = _to_utc(month_start)

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT time, unreal_pnl_value, equity_after FROM trades "
            "WHERE email = %s ORDER BY time ASC",
            (email,),
        )
        rows = cur.fetchall()

    if not rows:
        return {"empty": True}

    def _agg(subset):
        if not subset:
            return {"pnl": 0.0, "pnl_pct": 0.0, "trades": 0, "wins": 0}
        pnls = [float(r["unreal_pnl_value"]) for r in subset]
        wins = sum(1 for p in pnls if p >= 0)
        first_eq_after = float(subset[0]["equity_after"])
        first_pnl      = float(subset[0]["unreal_pnl_value"])
        base_eq        = first_eq_after - first_pnl
        total_pnl      = sum(pnls)
        pnl_pct        = round(total_pnl / base_eq * 100, 2) if base_eq > 0 else 0.0
        return {"pnl": round(total_pnl, 2), "pnl_pct": pnl_pct,
                "trades": len(subset), "wins": wins}

    today_rows = [r for r in rows if str(r["time"]) >= today_utc]
    week_rows  = [r for r in rows if str(r["time"]) >= week_utc]
    month_rows = [r for r in rows if str(r["time"]) >= month_utc]

    monthly: dict = {}
    for r in rows:
        ts  = str(r["time"])
        key = ts[:7]
        if key not in monthly:
            monthly[key] = {"period": key, "pnl": 0.0, "trades": 0, "wins": 0,
                            "_first_pnl": float(r["unreal_pnl_value"]),
                            "_first_eq_after": float(r["equity_after"])}
        monthly[key]["pnl"]    += float(r["unreal_pnl_value"])
        monthly[key]["trades"] += 1
        if float(r["unreal_pnl_value"]) >= 0:
            monthly[key]["wins"] += 1

    history = []
    for m in sorted(monthly.values(), key=lambda x: x["period"]):
        base = m["_first_eq_after"] - m["_first_pnl"]
        pnl  = round(m["pnl"], 2)
        pct  = round(pnl / base * 100, 2) if base > 0 else 0.0
        history.append({
            "period": m["period"],
            "pnl":    pnl,
            "pnl_pct": pct,
            "trades": m["trades"],
            "wins":   m["wins"],
            "losses": m["trades"] - m["wins"],
        })

    best_month  = max(history, key=lambda x: x["pnl"])  if history else None
    worst_month = min(history, key=lambda x: x["pnl"]) if history else None

    years_dict: dict = {}
    for m in history:
        yr = m["period"][:4]
        if yr not in years_dict:
            years_dict[yr] = {"year": yr, "pnl": 0.0, "trades": 0, "wins": 0, "months": []}
        years_dict[yr]["pnl"]    += m["pnl"]
        years_dict[yr]["trades"] += m["trades"]
        years_dict[yr]["wins"]   += m["wins"]
        years_dict[yr]["months"].append(m)
    for y in years_dict.values():
        y["pnl"]    = round(y["pnl"], 2)
        y["losses"] = y["trades"] - y["wins"]

    yearly = sorted(years_dict.values(), key=lambda x: x["year"])

    return {
        "today":              _agg(today_rows),
        "week":               _agg(week_rows),
        "month":              _agg(month_rows),
        "history":            history,
        "yearly":             yearly,
        "best_month":         best_month,
        "worst_month":        worst_month,
        "has_multiple_years": len(years_dict) > 1,
    }


@portfolio_router.get("/portfolio/pnl/month/{year_month}")
def portfolio_pnl_month(year_month: str, user=Depends(require_user)):
    """Return individual trades for a specific month (YYYY-MM) — drill-down view."""
    if len(year_month) != 7 or year_month[4] != "-":
        raise HTTPException(status_code=400, detail="Format must be YYYY-MM")
    email = user["email"]
    yr, mo = int(year_month[:4]), int(year_month[5:])
    next_yr, next_mo = (yr + 1, 1) if mo == 12 else (yr, mo + 1)
    end_str = f"{next_yr}-{next_mo:02d}-01"
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT time, side, symbol, mode, unreal_pnl_value, unreal_pnl_percent, "
            "entry_price, current_price, equity_after, leverage, reason "
            "FROM trades WHERE email = %s AND time >= %s AND time < %s "
            "ORDER BY time ASC",
            (email, year_month + "-01", end_str),
        )
        trades_list = [dict(r) for r in cur.fetchall()]
    return {"month": year_month, "trades": trades_list, "count": len(trades_list)}


@portfolio_router.get("/portfolio/pnl/csv/{year}")
def portfolio_pnl_csv(year: str, user=Depends(require_user)):
    """Download all trades for a given year as CSV (tax export)."""
    if not year.isdigit() or len(year) != 4:
        raise HTTPException(status_code=400, detail="Year must be 4 digits")

    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT time, side, symbol, mode, size, leverage, "
            "entry_price, current_price, unreal_pnl_value, unreal_pnl_percent, "
            "equity_after, reason "
            "FROM trades WHERE email = %s AND time >= %s AND time < %s "
            "ORDER BY time ASC",
            (email, f"{year}-01-01", f"{int(year)+1}-01-01"),
        )
        rows = cur.fetchall()

    buf = io.StringIO()
    w   = _csv.writer(buf)
    w.writerow(["Date (UTC)", "Side", "Symbol", "Mode", "Size %", "Leverage",
                "Entry Price", "Exit Price", "PnL ($)", "PnL (%)", "Equity After"])
    for r in rows:
        w.writerow([
            str(r["time"])[:19],
            r["side"], r["symbol"], r["mode"],
            r["size"], r["leverage"],
            r["entry_price"], r["current_price"],
            round(float(r["unreal_pnl_value"]), 4),
            round(float(r["unreal_pnl_percent"]), 4),
            round(float(r["equity_after"]), 4),
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{year}.csv"},
    )
