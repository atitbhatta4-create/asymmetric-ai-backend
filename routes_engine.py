"""
routes_engine.py — Manual trade, sandbox reset, and all /auto/* routes.

All heavy engine state (AUTO_RUNNERS, AUTO_LOCK, AutoRunner, persistence
helpers) stays in main.py. Routes access it via lazy function-body imports
to avoid any circular import at module load time — same pattern as
routes_admin.py and routes_portfolio.py.

Special case: /auto/start uses `import main as _main` so it can write back
to `_main._last_engine_start_ts` (a float that cannot be mutated via a
simple `from main import` name binding).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from auth_helpers import require_user
from config import (
    REAL_TRADING, START_EQUITY, RiskMode, Side, TradeStyle,
    TRADE_STYLE_PARAMS, now_dubai,
)
from database import db_conn, USING_PG
from exchange_utils import get_real_usdt_balance
from market_data import _validate_symbol
from notifications import tg_alert as _tg_alert, email_ai_started
from risk_engine import default_max_trades_per_day
from user_state import (
    get_equity, set_equity, get_session_id, set_session_id,
    update_peak_ath,
)

engine_router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class TradeIn(BaseModel):
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


class AutoStartIn(BaseModel):
    symbol: str
    trade_style: TradeStyle = "DAY_TRADE"
    mode: RiskMode = "MINI_ASYM"
    max_trades_per_day: Optional[int] = Field(default=None, ge=1, le=50)
    stop_after_bad_trades: int = Field(default=2, ge=1, le=20)
    duration_days: int = Field(default=7, ge=1, le=30)
    trend_filter: bool = True
    chop_min_sep_pct: float = Field(default=0.005, ge=0.001, le=0.05)
    floor_override: bool = False  # user confirmed restart after hard floor hit

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        return _validate_symbol(v)


class FloorResetIn(BaseModel):
    # confirm=False → dry-run (returns situation/status without applying)
    # confirm=True  → apply the reset
    confirm: bool = False
    # Must be exactly "CONFIRM" (case-insensitive) for Situation 2
    typed_confirm: str = ""


class CorrectTradeIn(BaseModel):
    symbol: str = Field(..., min_length=3, max_length=30)
    real_pnl: float = Field(..., ge=-100_000, le=100_000)
    num_trades: int = Field(default=2, ge=1, le=10)
    new_outcome: Literal["TP_HIT", "SL_HIT", "TRAIL_STOP", "NATURAL_CLOSE"] = "TP_HIT"


# ── Log type classifier (used only by /auto/sessions) ─────────────────────────

def _classify_log_type(msg: str) -> str:
    m = msg.upper()
    if any(x in m for x in ["TRADE OPENED", "TRADE CLOSED", "MID_CANDLE TRADE", "REAL ORDER"]):
        return "TRADE"
    if m.startswith("BLOCKED") or m.startswith("EARLY BEAR"):
        return "BLOCKED"
    if "MID_CANDLE" in m:
        return "MID_CANDLE"
    if m.startswith("HOLDING"):
        return "HOLDING"
    if "ERROR" in m or "FAILED" in m:
        return "ERROR"
    if any(x in m for x in ["RESET", "MIDNIGHT", "DAILY COUNTERS", "STRICTNESS"]):
        return "RESET"
    return "SYSTEM"


# ── Routes ────────────────────────────────────────────────────────────────────

@engine_router.post("/trade")
async def place_trade(payload: TradeIn, user=Depends(require_user)):
    from main import AUTO_RUNNERS, AUTO_LOCK, _place_trade_internal
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r and r.is_running():
            raise HTTPException(status_code=400, detail="Manual trading disabled while AI is running.")
    return await _place_trade_internal(
        email, payload.symbol, payload.side, payload.mode,
        extra_reason="Manual trade request.",
    )


@engine_router.post("/reset")
def reset_sandbox(user=Depends(require_user)):
    from main import set_ai_restart_lock
    email = user["email"]
    sid = get_session_id(email)
    new_sid = sid + 1
    set_session_id(email, new_sid)
    if not REAL_TRADING:
        # Demo only: reset equity and floor back to starting baseline.
        set_equity(email, START_EQUITY)
        _reset_floor = round(float(START_EQUITY) * 0.85, 2)
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE user_state SET peak_equity=%s, floor_equity=%s WHERE email=%s",
                (START_EQUITY, _reset_floor, email),
            )
            conn.commit()
        equity_after = START_EQUITY
    else:
        # Real trading: preserve equity, peak, and floor; only increment session counter.
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT equity FROM user_state WHERE email=%s", (email,))
            row = cur.fetchone()
        equity_after = float((row or {}).get("equity") or START_EQUITY)
    set_ai_restart_lock(email, 0)
    return {"ok": True, "equity": equity_after, "new_session_id": new_sid}


@engine_router.get("/auto/status")
def auto_status(user=Depends(require_user)):
    from main import AUTO_RUNNERS, AUTO_LOCK, get_ai_restart_lock
    email = user["email"]
    now_ts = int(time.time())
    lock_until = get_ai_restart_lock(email)
    lock_sec = max(0, lock_until - now_ts)
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if not r:
            _last_stop = None
            try:
                with db_conn() as _sc:
                    _sc_cur = _sc.cursor()
                    _sc_cur.execute(
                        "SELECT last_stop_reason FROM user_state WHERE email=%s", (email,)
                    )
                    _sc_row = _sc_cur.fetchone()
                    _last_stop = (_sc_row or {}).get("last_stop_reason")
            except Exception:
                pass
            return {
                "ok": True, "running": False,
                "blocked_reason": _last_stop,
                "restart_locked": lock_sec > 0,
                "restart_lock_sec": lock_sec,
            }
        st = r.status()
        return {
            "ok": True, **asdict(st),
            "restart_locked": lock_sec > 0,
            "restart_lock_sec": lock_sec,
        }


@engine_router.get("/auto/history")
def auto_history(user=Depends(require_user), limit: int = Query(default=40, ge=1, le=200)):
    from main import AUTO_RUNNERS, AUTO_LOCK
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            return {
                "ok": True,
                "session_id": r.ai_session_id,
                "events": list(r.history)[: int(limit)],
            }
    # No runner in memory (after redeploy) — read latest open session from DB
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM ai_sessions WHERE email=%s AND ended_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (email,),
        )
        sess = cur.fetchone()
        session_id = sess["id"] if sess else None
        if session_id:
            cur.execute(
                "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                "ORDER BY id DESC LIMIT %s",
                (email, session_id, int(limit)),
            )
        else:
            # No open session — find the latest (even if ended) to avoid bleeding logs
            cur.execute(
                "SELECT id FROM ai_sessions WHERE email=%s ORDER BY id DESC LIMIT 1",
                (email,),
            )
            latest = cur.fetchone()
            if latest:
                session_id = latest["id"]
                cur.execute(
                    "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                    "ORDER BY id DESC LIMIT %s",
                    (email, session_id, int(limit)),
                )
            else:
                cur.execute(
                    "SELECT t, msg FROM ai_logs WHERE email=%s ORDER BY id DESC LIMIT %s",
                    (email, int(limit)),
                )
        rows = cur.fetchall()
    return {
        "ok": True,
        "session_id": session_id,
        "events": [{"t": r["t"], "msg": r["msg"]} for r in rows],
    }


@engine_router.get("/auto/sessions")
def auto_sessions(
    user=Depends(require_user),
    limit: int = Query(default=20, ge=1, le=100),
    log_limit: int = Query(default=500, ge=1, le=2000),
    symbol: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None, max_length=200),
    log_type: Optional[str] = Query(default=None, max_length=30),
):
    """Return AI sessions newest-first with classified logs.
    Supports optional symbol filter, keyword search, and log_type filter."""
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        if symbol:
            cur.execute(
                "SELECT id, symbol, mode, trade_style, started_at, ended_at, stop_reason "
                "FROM ai_sessions WHERE email=%s AND UPPER(symbol)=UPPER(%s) ORDER BY id DESC LIMIT %s",
                (email, symbol.upper().strip(), int(limit)),
            )
        else:
            cur.execute(
                "SELECT id, symbol, mode, trade_style, started_at, ended_at, stop_reason "
                "FROM ai_sessions WHERE email=%s ORDER BY id DESC LIMIT %s",
                (email, int(limit)),
            )
        sessions = [dict(r) for r in cur.fetchall()]
        for sess in sessions:
            if search:
                cur.execute(
                    "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                    "AND LOWER(msg) LIKE LOWER(%s) ORDER BY id ASC LIMIT %s",
                    (email, sess["id"], f"%{search.strip()}%", int(log_limit)),
                )
            else:
                cur.execute(
                    "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                    "ORDER BY id ASC LIMIT %s",
                    (email, sess["id"], int(log_limit)),
                )
            events = [
                {"t": r["t"], "msg": r["msg"], "log_type": _classify_log_type(r["msg"])}
                for r in cur.fetchall()
            ]
            _VALID_LOG_TYPES = {"ALL", "TRADE", "SIGNAL", "RISK", "RESET", "SYSTEM", "ERROR"}
            if log_type and log_type.upper() != "ALL":
                lt = log_type.upper()
                if lt not in _VALID_LOG_TYPES:
                    lt = "SYSTEM"
                events = [e for e in events if e["log_type"] == lt]
            sess["events"] = events
            sess["total_events"] = len(events)
    return {"ok": True, "sessions": sessions}


@engine_router.get("/auto/signal")
def auto_signal(user=Depends(require_user)):
    """Live 4-layer signal breakdown for the running AutoRunner."""
    from main import AUTO_RUNNERS, AUTO_LOCK
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if not r:
            return {"ok": False, "running": False}
        return {
            "ok": True,
            "running": r.is_running(),
            "symbol": r.symbol,
            "mode": r.mode,
            "signal": r.last_signal,
            "side": r.last_side,
            "blocked": r.blocked_reason,
            "adaptive_strictness": r.adaptive_strictness,
            "signal_score": r.last_score,
            "trade_style": r.trade_style,
            "market_grade": r.market_grade,
            "pending_trades": r.pending_trades,
            "breakdown": r.last_breakdown,
        }


@engine_router.post("/auto/start")
def auto_start(payload: AutoStartIn, user=Depends(require_user)):
    # import main as _main gives us write-back access to module-level floats
    # (_last_engine_start_ts is reassigned, not mutated — a plain `from main import`
    # would only update the local binding, not the module attribute).
    import main as _main
    from main import AUTO_RUNNERS, AUTO_LOCK, AutoRunner, _save_runner_state, \
        get_ai_restart_lock, set_ai_restart_lock

    email = user["email"]
    symbol = payload.symbol.upper().strip()
    if not (symbol.endswith("USDT") or symbol.endswith("-USDT")):
        raise HTTPException(status_code=400, detail="Use symbol like BTCUSDT / ETHUSDT / SOLUSDT")

    _mode_max = default_max_trades_per_day(payload.mode)
    if payload.max_trades_per_day is not None:
        max_trades = min(int(payload.max_trades_per_day), _mode_max)
    else:
        max_trades = _mode_max

    # ── Check 1: Duration-session restart lock ────────────────────────────────
    now_ts = int(time.time())
    lock_until = get_ai_restart_lock(email)
    if lock_until > now_ts:
        remaining = lock_until - now_ts
        h, m = remaining // 3600, (remaining % 3600) // 60
        raise HTTPException(
            status_code=400,
            detail=(
                f"AI restart locked for {h}h {m}m — you stopped a duration session early. "
                f"This protects your account from impulsive restarts. "
                f"Use Reset Sandbox (demo) to clear, or wait until Dubai midnight."
            ),
        )

    # ── Check 2: Hard floor — require explicit user confirmation to restart ───
    try:
        with db_conn() as _hf_conn:
            _hf_cur = _hf_conn.cursor()
            _hf_cur.execute(
                "SELECT last_stop_reason, equity, peak_equity FROM user_state WHERE email=%s",
                (email,),
            )
            _hf_row = _hf_cur.fetchone()
        _last_reason = (_hf_row or {}).get("last_stop_reason") or ""
        if _last_reason == "HARD_FLOOR":
            _cur_eq  = float((_hf_row or {}).get("equity") or 0)
            _new_floor = round(_cur_eq * 0.85, 2)
            if not payload.floor_override:
                # Return 409 so the frontend knows to show the confirmation modal
                raise HTTPException(
                    status_code=409,
                    detail=json.dumps({
                        "code": "HARD_FLOOR_CONFIRM",
                        "current_equity": round(_cur_eq, 2),
                        "new_floor": _new_floor,
                        "message": (
                            f"Your hard floor was hit and the AI stopped to protect your account. "
                            f"To restart, your floor will be reset to ${_new_floor:.2f} "
                            f"(85% of current equity ${_cur_eq:.2f}). "
                            f"Confirm to proceed."
                        ),
                    }),
                )
            # floor_override=True — reset peak/floor to current equity and clear stop reason
            with db_conn() as _hf_reset:
                _hf_reset.cursor().execute(
                    "UPDATE user_state SET peak_equity=%s, floor_equity=%s, "
                    "last_stop_reason=NULL WHERE email=%s",
                    (_cur_eq, _new_floor, email),
                )
                _hf_reset.commit()
    except HTTPException:
        raise
    except Exception:
        pass  # DB error — allow start

    # ── Check 3: Bad-trade daily limit — enforce mode minimum ─────────────────
    mode_min_stop_after = {"ULTRA_SAFE": 1, "SAFE": 1, "NORMAL": 2, "MINI_ASYM": 2, "AGGRESSIVE": 3}
    effective_stop_after = max(int(payload.stop_after_bad_trades), mode_min_stop_after.get(payload.mode, 2))
    try:
        dubai_midnight = now_dubai().replace(hour=0, minute=0, second=0, microsecond=0)
        utc_midnight_str = dubai_midnight.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        sid = get_session_id(email)
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT SUM(CASE WHEN unreal_pnl_percent < 0 THEN 1 ELSE 0 END) as bad "
                "FROM trades WHERE email = %s AND time >= %s AND session_id = %s",
                (email, utc_midnight_str, sid),
            )
            row = cur.fetchone()
        bad_today = int((row or {}).get("bad") or 0)
        if bad_today >= effective_stop_after:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Daily risk limit reached — {bad_today} bad trades today "
                    f"(limit: {effective_stop_after} for {payload.mode}). "
                    f"AI is locked until Dubai midnight to protect your account."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        pass  # DB error — allow start, runner will recount

    # ── Sync real exchange balance before runner init ─────────────────────────
    if REAL_TRADING:
        try:
            _pre_bal = get_real_usdt_balance(email, force=True)
            if _pre_bal is not None:
                set_equity(email, _pre_bal)
                # Guard: stale demo peak ($1,000) on a real $25 account would cause
                # instant hard floor. If stored peak is 5× above real balance, reset it.
                try:
                    with db_conn() as conn:
                        _gpk_cur = conn.cursor()
                        _gpk_cur.execute(
                            "SELECT peak_equity, floor_equity FROM user_state WHERE email=%s",
                            (email,),
                        )
                        _gpk_row = _gpk_cur.fetchone()
                        _gpk_peak = float((_gpk_row or {}).get("peak_equity") or 0)
                        _is_stale_demo_peak = _gpk_peak > _pre_bal * 5 and _gpk_peak > 50
                        if _is_stale_demo_peak:
                            _corrected_floor = round(_pre_bal * 0.85, 2)
                            _gpk_cur.execute(
                                "UPDATE user_state SET peak_equity=%s, floor_equity=%s WHERE email=%s",
                                (_pre_bal, _corrected_floor, email),
                            )
                            conn.commit()
                            print(
                                f"[auto-start] REAL peak corrected: demo peak ${_gpk_peak:.2f} "
                                f"→ real balance ${_pre_bal:.2f} (was causing instant hard floor)"
                            )
                except Exception as _gpke:
                    print(f"[auto-start] real peak guard failed: {_gpke}")
        except Exception as _bal_err:
            print(f"[auto-start] real balance pre-sync failed (non-fatal): {_bal_err}")

    # ── Record starting_capital on very first AI start (never overwritten) ────
    try:
        with db_conn() as conn:
            _sc_cur = conn.cursor()
            _sc_cur.execute("SELECT starting_capital FROM user_state WHERE email=%s", (email,))
            _sc_row = _sc_cur.fetchone()
            if _sc_row and float(_sc_row.get("starting_capital") or 0) == 0:
                _sc_equity = get_equity(email)
                _sc_cur.execute(
                    "UPDATE user_state SET starting_capital=%s WHERE email=%s",
                    (_sc_equity, email),
                )
                conn.commit()
    except Exception as _sce:
        print(f"[auto-start] starting_capital set failed: {_sce}")

    # ── Engine start rate limiter ─────────────────────────────────────────────
    # _main._last_engine_start_ts is a float — must assign via module attribute,
    # not a re-bound local name.
    with _main._ENGINE_START_LOCK:
        _now = time.time()
        _main._engine_start_times[:] = [t for t in _main._engine_start_times if _now - t < 60]
        _count = len(_main._engine_start_times)
        _gap   = _now - _main._last_engine_start_ts

        if _count >= _main._MAX_STARTS_PER_MIN:
            _oldest   = min(_main._engine_start_times)
            _retry_in = int(60 - (_now - _oldest)) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Engine start queued — {_main._MAX_STARTS_PER_MIN} engines are starting this minute. "
                    f"You are position {_count - _main._MAX_STARTS_PER_MIN + 1} in queue. "
                    f"Retry in {_retry_in}s."
                ),
            )
        if _gap < _main._ENGINE_START_GAP:
            _retry_in = int(_main._ENGINE_START_GAP - _gap) + 1
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Engine start queued — system staggers starts to protect stability. "
                    f"Retry in {_retry_in}s."
                ),
            )
        _main._engine_start_times.append(_now)
        _main._last_engine_start_ts = _now

    # ── Start the runner ──────────────────────────────────────────────────────
    try:
        with AUTO_LOCK:
            old = AUTO_RUNNERS.get(email)
            if old:
                old.stop("Stopped: restarted with new settings.")
                del AUTO_RUNNERS[email]

            runner = AutoRunner(
                email=email, symbol=symbol, trade_style=payload.trade_style,
                mode=payload.mode,
                max_trades_per_day=int(max_trades),
                stop_after_bad_trades=int(payload.stop_after_bad_trades),
                duration_days=int(payload.duration_days),
                trend_filter=bool(payload.trend_filter),
                chop_min_sep_pct=float(payload.chop_min_sep_pct),
            )
            AUTO_RUNNERS[email] = runner
            runner.start()
            _save_runner_state(runner)
    except Exception as _start_err:
        import traceback as _tb
        print(f"[auto-start] CRITICAL: Engine start failed for {email} ({symbol}): {_start_err}\n{_tb.format_exc()}")
        _tg_alert(
            f"🚨 *Engine start FAILED* ({symbol})\n"
            f"User: `{email}`\n"
            f"Error: `{_start_err}`\n"
            f"This caused a 500 on /auto/start — engine NOT running."
        )
        raise HTTPException(status_code=500, detail="Engine failed to start. Our team has been notified.")

    # Clear any previous stop reason so the UI shows clean state
    try:
        with db_conn() as _clr_conn:
            _clr_conn.cursor().execute(
                "UPDATE user_state SET last_stop_reason=NULL WHERE email=%s", (email,)
            )
            _clr_conn.commit()
    except Exception:
        pass

    email_ai_started(
        to=email, symbol=symbol, mode=payload.mode, trade_style=payload.trade_style,
        duration_days=int(payload.duration_days),
        max_trades=int(max_trades), stop_after_bad=int(payload.stop_after_bad_trades),
    )

    sp = TRADE_STYLE_PARAMS.get(payload.trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
    return {
        "ok": True, "running": True, "symbol": symbol,
        "trade_style": payload.trade_style, "tf": sp["tf"],
        "interval_sec": sp["interval"], "mode": payload.mode,
        "max_trades_per_day": int(max_trades),
    }


@engine_router.post("/auto/stop")
def auto_stop(user=Depends(require_user)):
    from main import AUTO_RUNNERS, AUTO_LOCK, set_ai_restart_lock, _clear_runner_state
    email = user["email"]
    restart_locked = False
    lock_sec = 0
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            was_duration = r.duration_days > 0
            r.stop("Stopped by user.")
            del AUTO_RUNNERS[email]
            # Duration sessions: lock restarts until Dubai midnight to prevent bypass
            if was_duration:
                now_dt = now_dubai()
                next_midnight = (now_dt + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                lock_until = int(next_midnight.timestamp())
                set_ai_restart_lock(email, lock_until)
                lock_sec = max(0, lock_until - int(time.time()))
                restart_locked = True
    # Clear persisted state — user intentionally stopped, do NOT resume on redeploy
    _clear_runner_state(email)
    return {"ok": True, "running": False, "restart_locked": restart_locked, "restart_lock_sec": lock_sec}


@engine_router.post("/auto/reset-strictness")
def auto_reset_strictness(user=Depends(require_user)):
    """Reset adaptive_strictness to 1.0x — clears stuck values in DB and live runner."""
    from main import AUTO_RUNNERS, AUTO_LOCK, _save_runner_state
    email = user["email"]
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            r.adaptive_strictness = 1.0
            r.consecutive_wins = 0
            r.log("Strictness manually reset to 1.0x by user.")
            _save_runner_state(r)
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            if USING_PG:
                cur.execute(
                    "UPDATE ai_runner_state SET adaptive_strictness = 1.0, consecutive_wins = 0 WHERE email = %s",
                    (email,),
                )
            else:
                cur.execute(
                    "UPDATE ai_runner_state SET adaptive_strictness = 1.0, consecutive_wins = 0 WHERE email = ?",
                    (email,),
                )
            conn.commit()
    except Exception as e:
        print(f"[reset-strictness] DB patch failed: {e}")
    return {"ok": True, "adaptive_strictness": 1.0, "message": "Strictness reset to 1.0x"}


@engine_router.post("/auto/correct-real-trade")
def auto_correct_real_trade(payload: CorrectTradeIn, user=Depends(require_user)):
    """
    Correct trade records when paper simulation diverges from real exchange P&L.

    Finds the N most recent trades for the symbol in the current session,
    splits real_pnl proportionally by position size, and patches DB + live runner.
    Also resets bad_trades_today=0 and adaptive_strictness=1.0 when real_pnl > 0.
    """
    from main import AUTO_RUNNERS, AUTO_LOCK, _save_runner_state
    email = user["email"]
    symbol = payload.symbol.strip()
    real_pnl = payload.real_pnl
    num_trades = max(1, min(payload.num_trades, 5))
    new_outcome = payload.new_outcome.upper()
    if new_outcome not in ("TP_HIT", "SL_HIT", "TRAIL_STOP"):
        raise HTTPException(status_code=400, detail="new_outcome must be TP_HIT, SL_HIT, or TRAIL_STOP")

    sid = get_session_id(email)
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, size, unreal_pnl_value, unreal_pnl_percent, reason "
            "FROM trades WHERE email=%s AND symbol=%s AND session_id=%s "
            "ORDER BY id DESC LIMIT %s",
            (email, symbol, int(sid), num_trades),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No recent trades found for {symbol} in current session (session_id={sid}).",
        )

    # Split real_pnl proportionally by position size across all legs
    total_size = sum(float(r["size"]) for r in rows)
    corrections = []
    for r in rows:
        frac    = float(r["size"]) / total_size if total_size > 0 else 1.0 / len(rows)
        leg_pnl = round(real_pnl * frac, 4)
        old_pct = float(r["unreal_pnl_percent"])
        new_pct = round(leg_pnl / float(r["size"]) * 100, 4) if float(r["size"]) > 0 else old_pct
        corrections.append({
            "id": r["id"], "leg_pnl": leg_pnl, "new_pct": new_pct,
            "old_pnl": float(r["unreal_pnl_value"]), "reason": r["reason"],
        })

    now_str = now_dubai().strftime("%Y-%m-%d %H:%M:%S")
    with db_conn() as conn:
        cur = conn.cursor()
        for c_item in corrections:
            correction_note = (
                f"\n\n[MANUAL CORRECTION {now_str}] "
                f"Paper: {c_item['old_pnl']:+.4f} → Real: {c_item['leg_pnl']:+.4f} | "
                f"Outcome corrected to {new_outcome}"
            )
            cur.execute(
                "UPDATE trades SET unreal_pnl_value=%s, unreal_pnl_percent=%s, reason=%s "
                "WHERE id=%s",
                (c_item["leg_pnl"], c_item["new_pct"],
                 (c_item["reason"] or "") + correction_note, c_item["id"]),
            )
        conn.commit()

    runner_updated = False
    with AUTO_LOCK:
        r = AUTO_RUNNERS.get(email)
        if r:
            if real_pnl > 0:
                r.bad_trades_today   = 0
                r.adaptive_strictness = 1.0
                r.consecutive_wins   = 0
                r._last_trade_bad    = False
            r.log(
                f"[MANUAL CORRECTION] {symbol} trade P&L corrected: "
                f"${real_pnl:+.4f} total ({len(corrections)} leg(s)). "
                f"bad_trades=0, strictness=1.0"
            )
            _save_runner_state(r)
            runner_updated = True

    if real_pnl > 0:
        try:
            with db_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE ai_runner_state SET adaptive_strictness=1.0, "
                    "consecutive_wins=0, last_trade_bad=0 WHERE email=%s",
                    (email,),
                )
                conn.commit()
        except Exception as _sde:
            print(f"[correct-trade] DB strictness patch failed: {_sde}")

    current_equity = get_equity(email)
    update_peak_ath(email, current_equity, current_equity)

    return {
        "ok": True,
        "trades_corrected": len(corrections),
        "symbol": symbol,
        "real_pnl_total": real_pnl,
        "legs": [{"trade_id": c["id"], "pnl": c["leg_pnl"], "pnl_pct": c["new_pct"]} for c in corrections],
        "bad_trades_reset": real_pnl > 0,
        "strictness_reset": real_pnl > 0,
        "runner_updated": runner_updated,
        "peak_equity_updated": True,
    }


@engine_router.post("/auto/reset-floor")
def auto_reset_floor(payload: FloorResetIn, user=Depends(require_user)):
    """
    Hard-floor reset endpoint.

    Call with confirm=False first to get dry-run status (situation, cooldown, equity).
    Call with confirm=True to apply the reset.

    Situation 1 — equity >= starting_capital: one-click reset.
    Situation 2 — equity in [50%, 100%) of starting_capital: must type 'CONFIRM'.
    Situation 3 — equity < 50% of starting_capital: blocked, contact support.

    Cooldown: 3+ resets within 30 days triggers a 7-day lock from the most recent reset.
    """
    from main import AUTO_RUNNERS, AUTO_LOCK, _save_runner_state
    email = user["email"]
    now_ts  = time.time()
    now_str = now_dubai().strftime("%Y-%m-%d %H:%M:%S")

    with db_conn() as conn:
        _rc = conn.cursor()
        _rc.execute(
            "SELECT peak_equity, all_time_high, starting_capital, "
            "floor_reset_count, floor_reset_at FROM user_state WHERE email=%s",
            (email,),
        )
        _row = _rc.fetchone()

    if not _row:
        raise HTTPException(status_code=400, detail="No user state found. Start the AI first.")

    current_equity   = get_equity(email)
    starting_capital = float(_row.get("starting_capital") or 0)
    current_ath      = float(_row.get("all_time_high") or 0)

    try:
        reset_timestamps: List[float] = json.loads(_row.get("floor_reset_at") or "[]")
    except Exception:
        reset_timestamps = []

    thirty_days_ago = now_ts - (30 * 24 * 3600)
    recent_resets   = [t for t in reset_timestamps if t > thirty_days_ago]

    cooldown_remaining_sec = 0
    if len(recent_resets) >= 3:
        most_recent    = max(recent_resets)
        cooldown_until = most_recent + (7 * 24 * 3600)
        if now_ts < cooldown_until:
            cooldown_remaining_sec = int(cooldown_until - now_ts)

    if starting_capital <= 0 or current_equity >= starting_capital:
        situation = 1
    elif current_equity >= starting_capital * 0.50:
        situation = 2
    else:
        situation = 3

    status = {
        "ok": True,
        "situation": situation,
        "current_equity":     round(current_equity, 2),
        "starting_capital":   round(starting_capital, 2),
        "new_peak_if_reset":  round(current_equity, 2),
        "new_floor_if_reset": round(current_equity * 0.85, 2),
        "resets_in_30d":          len(recent_resets),
        "cooldown_remaining_sec": cooldown_remaining_sec,
        "reset_applied": False,
        "message": "",
    }

    # Dry-run: return status without applying
    if not payload.confirm:
        if situation == 1:
            status["message"] = (
                f"Situation 1 — your equity (${current_equity:.2f}) is at or above your "
                f"starting capital (${starting_capital:.2f}). One-click reset available."
            )
        elif situation == 2:
            status["message"] = (
                f"Situation 2 — your equity (${current_equity:.2f}) is below your starting "
                f"capital (${starting_capital:.2f}). You must type 'CONFIRM' to proceed."
            )
        else:
            status["message"] = (
                f"Situation 3 — your equity (${current_equity:.2f}) is below 50% of your "
                f"starting capital (${starting_capital:.2f}). Reset is blocked. "
                f"Please contact support."
            )
        return status

    # Validation before applying
    if cooldown_remaining_sec > 0:
        h = cooldown_remaining_sec // 3600
        m = (cooldown_remaining_sec % 3600) // 60
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reset cooldown — you have reset the floor 3 times in 30 days. "
                f"Cooldown expires in {h}h {m}m. "
                f"This protects your account from repeatedly overriding risk protection."
            ),
        )

    if situation == 3:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Floor reset blocked — your equity (${current_equity:.2f}) is below 50% of "
                f"your starting capital (${starting_capital:.2f}). "
                f"Please contact support to review your account."
            ),
        )

    if situation == 2 and payload.typed_confirm.strip().upper() != "CONFIRM":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Your equity (${current_equity:.2f}) is below your starting capital "
                f"(${starting_capital:.2f}). Please type 'CONFIRM' to accept this loss "
                f"and reset the safety floor."
            ),
        )

    # Apply the reset
    new_peak = current_equity
    new_floor = round(current_equity * 0.85, 2)
    new_ath   = max(current_ath, new_peak)
    updated_resets = recent_resets + [now_ts]

    with db_conn() as conn:
        _ac = conn.cursor()
        # floor_equity set explicitly (not via max) — this is an intentional user reset
        _ac.execute(
            "UPDATE user_state SET "
            "peak_equity=%s, all_time_high=%s, floor_equity=%s, "
            "floor_reset_count=%s, floor_reset_at=%s "
            "WHERE email=%s",
            (new_peak, new_ath, new_floor, len(updated_resets), json.dumps(updated_resets), email),
        )
        sid = get_session_id(email)
        _ac.execute(
            "INSERT INTO ai_logs(email, session_id, t, msg) VALUES(%s, %s, %s, %s)",
            (
                email, sid, now_str,
                (
                    f"FLOOR RESET (Situation {situation}) — "
                    f"new peak=${new_peak:.2f}, new floor=${new_floor:.2f}, "
                    f"starting_capital=${starting_capital:.2f}, "
                    f"resets_in_30d={len(updated_resets)}"
                ),
            ),
        )
        conn.commit()

    # Patch live runner if still present (defensive — runner normally stopped on hard floor)
    with AUTO_LOCK:
        _r = AUTO_RUNNERS.get(email)
    if _r:
        _r.peak_equity  = new_peak
        _r.floor_equity = new_floor
        _r.blocked_reason = None
        _r.log(
            f"FLOOR RESET applied by user — new peak=${new_peak:.2f}, "
            f"new floor=${new_floor:.2f} (Situation {situation})"
        )
        _save_runner_state(_r)

    status["reset_applied"] = True
    status["resets_in_30d"] = len(updated_resets)
    status["message"] = (
        f"Floor reset applied (Situation {situation}). "
        f"New peak: ${new_peak:.2f}, new floor: ${new_floor:.2f}. "
        f"Click Start to resume trading with the new floor."
    )
    return status
