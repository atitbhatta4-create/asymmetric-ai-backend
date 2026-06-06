"""
routes_admin.py — Admin management and analytics endpoints.

Wire up in main.py:
    from routes_admin import admin_router
    app.include_router(admin_router)
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query
from pydantic import BaseModel

from config import START_EQUITY, now_dubai
from database import db_conn, USING_PG

admin_router = APIRouter(tags=["admin"])

# ── Local helpers ──────────────────────────────────────────────────────────────
def _now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ── Auth session cache ─────────────────────────────────────────────────────────
_SESSION_CACHE: Dict[str, tuple] = {}
_SESSION_TTL = 300

# ── Admin config ───────────────────────────────────────────────────────────────
_ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.getenv("ADMIN_EMAILS", "admin@demo.com").split(",")
    if e.strip()
)


# ── Auth dependencies ──────────────────────────────────────────────────────────
def _require_user(session: Optional[str] = Cookie(default=None)) -> Dict:
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")
    now = time.time()
    cached = _SESSION_CACHE.get(session)
    if cached:
        email, ts = cached
        if now - ts < _SESSION_TTL:
            return {"email": email}
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email FROM sessions WHERE token = %s", (session,))
        row = cur.fetchone()
    if not row:
        _SESSION_CACHE.pop(session, None)
        raise HTTPException(status_code=401, detail="Unauthorized")
    _SESSION_CACHE[session] = (row["email"], now)
    return {"email": row["email"]}


def _require_admin(user=Depends(_require_user)) -> str:
    email = user["email"].strip().lower()
    if email not in _ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access only")
    return email


# ── Models ─────────────────────────────────────────────────────────────────────
class AdminSettingsIn(BaseModel):
    signup_enabled: bool
    seat_capacity: int


class AdminNotesIn(BaseModel):
    notes: str


# ── Analytics helper ───────────────────────────────────────────────────────────
def _analytics_win_rate(rows: List[Dict]) -> Dict:
    wins   = [float(r.get("unreal_pnl_value") or 0) for r in rows if float(r.get("unreal_pnl_value") or 0) > 0]
    losses = [float(r.get("unreal_pnl_value") or 0) for r in rows if float(r.get("unreal_pnl_value") or 0) <= 0]
    total  = len(rows)
    return {
        "total":     total,
        "wins":      len(wins),
        "losses":    len(losses),
        "win_rate":  round(len(wins) / total * 100, 1) if total > 0 else 0.0,
        "avg_win":   round(sum(wins)   / len(wins),   2) if wins   else 0.0,
        "avg_loss":  round(sum(losses) / len(losses), 2) if losses else 0.0,
        "total_pnl": round(sum(wins) + sum(losses),   2),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@admin_router.get("/admin/status")
def admin_status(admin=Depends(_require_admin)):
    from main import AUTO_RUNNERS, signup_is_enabled, seat_capacity, seats_used
    return {
        "ok": True, "admin": admin,
        "signup_enabled": signup_is_enabled(),
        "seat_capacity": seat_capacity(),
        "seats_used": seats_used(),
        "seats_remaining": max(0, seat_capacity() - seats_used()),
        "auto_runners": list(AUTO_RUNNERS.keys()),
    }


@admin_router.get("/admin/settings")
def admin_settings(admin=Depends(_require_admin)):
    from main import signup_is_enabled, seat_capacity, seats_used
    return {
        "signup_enabled": signup_is_enabled(),
        "seat_capacity": seat_capacity(),
        "seats_used": seats_used(),
        "seats_remaining": max(0, seat_capacity() - seats_used()),
    }


@admin_router.post("/admin/settings")
def admin_update_settings(payload: AdminSettingsIn, admin=Depends(_require_admin)):
    from main import admin_set_setting, signup_is_enabled, seat_capacity, seats_used
    sc = int(max(1, min(100000, payload.seat_capacity)))
    admin_set_setting("signup_enabled", "true" if payload.signup_enabled else "false")
    admin_set_setting("seat_capacity", str(sc))
    return {
        "ok": True,
        "signup_enabled": signup_is_enabled(),
        "seat_capacity": seat_capacity(),
        "seats_used": seats_used(),
        "seats_remaining": max(0, seat_capacity() - seats_used()),
    }


@admin_router.post("/admin/stop-all-ai")
def admin_stop_all_ai(admin=Depends(_require_admin)):
    from main import AUTO_RUNNERS, AUTO_LOCK
    stopped = []
    with AUTO_LOCK:
        for email, runner in list(AUTO_RUNNERS.items()):
            runner.stop("Stopped by admin.")
            stopped.append(email)
            del AUTO_RUNNERS[email]
    return {"ok": True, "stopped_ai_for": stopped}


@admin_router.post("/admin/reset-user")
def admin_reset_user(email: str, admin=Depends(_require_admin)):
    from main import ensure_user_state, set_equity, get_session_id, set_session_id
    email = (email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")
    ensure_user_state(email)
    set_equity(email, START_EQUITY)
    set_session_id(email, get_session_id(email) + 1)
    return {"ok": True, "reset_user": email, "equity": START_EQUITY}


@admin_router.get("/admin/users")
def admin_users(
    admin=Depends(_require_admin),
    q: Optional[str] = Query(default=None, max_length=200),
    limit: int = Query(default=100, ge=1, le=2000),
):
    from main import AUTO_RUNNERS, AUTO_LOCK
    qn = (q or "").strip().lower()
    with db_conn() as conn:
        cur = conn.cursor()

        if qn:
            cur.execute(
                """
                SELECT u.email, u.created_at,
                       COALESCE(s.equity, 0) AS equity,
                       COALESCE(s.session_id, 0) AS session_id
                FROM users u
                LEFT JOIN user_state s ON s.email = u.email
                WHERE LOWER(u.email) LIKE %s
                ORDER BY u.created_at DESC
                LIMIT %s
                """,
                (f"%{qn}%", int(limit)),
            )
        else:
            cur.execute(
                """
                SELECT u.email, u.created_at,
                       COALESCE(s.equity, 0) AS equity,
                       COALESCE(s.session_id, 0) AS session_id
                FROM users u
                LEFT JOIN user_state s ON s.email = u.email
                ORDER BY u.created_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )

        rows = cur.fetchall()
        out = []
        for r in rows:
            email = r["email"]
            cur.execute("SELECT 1 FROM exchange_keys WHERE email=%s LIMIT 1", (email,))
            ex = bool(cur.fetchone())
            cur.execute("SELECT COUNT(*) AS n FROM trades WHERE email=%s", (email,))
            tcount = int(cur.fetchone()["n"])
            with AUTO_LOCK:
                rr = AUTO_RUNNERS.get(email)
                running = bool(rr and rr.is_running())
            dubai_today = (now_dubai()).strftime("%Y-%m-%d")
            cur.execute(
                "SELECT COALESCE(SUM(unreal_pnl_value),0) AS pnl FROM trades WHERE email=%s AND time::text LIKE %s",
                (email, f"{dubai_today}%"),
            )
            today_pnl = float(cur.fetchone()["pnl"])

            cur.execute("SELECT COUNT(*) AS w FROM trades WHERE email=%s AND unreal_pnl_value>=0", (email,))
            wins = int(cur.fetchone()["w"])
            win_rate = round(wins / tcount * 100, 1) if tcount else 0.0

            cur.execute("SELECT time FROM trades WHERE email=%s ORDER BY id DESC LIMIT 1", (email,))
            la_row = cur.fetchone()
            last_active = str(la_row["time"]) if la_row else None

            open_positions = 0
            with AUTO_LOCK:
                rr2 = AUTO_RUNNERS.get(email)
                if rr2 and rr2.is_running():
                    open_positions = len(rr2.pending_trades)

            out.append({
                "email": email,
                "created_at": r["created_at"],
                "equity": float(r["equity"]),
                "session_id": int(r["session_id"]),
                "exchange_connected": ex,
                "trades_count": tcount,
                "ai_running": running,
                "today_pnl": round(today_pnl, 2),
                "win_rate": win_rate,
                "last_active": last_active,
                "open_positions": open_positions,
            })

    return {"ok": True, "users": out}


@admin_router.get("/admin/user/{email}")
def admin_user_details(
    email: str,
    admin=Depends(_require_admin),
    trades_limit: int = Query(default=50, ge=1, le=500),
):
    from main import AUTO_RUNNERS, AUTO_LOCK
    email = (email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email required")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT email, created_at FROM users WHERE email=%s", (email,))
        u = cur.fetchone()
        if not u:
            raise HTTPException(status_code=404, detail="User not found")

        cur.execute("SELECT equity, session_id FROM user_state WHERE email=%s", (email,))
        st = cur.fetchone()
        equity = float(st["equity"]) if st else START_EQUITY
        session_id = int(st["session_id"]) if st else 0

        cur.execute("SELECT exchange, created_at FROM exchange_keys WHERE email=%s", (email,))
        ex = cur.fetchone()

        cur.execute("SELECT COUNT(*) AS n FROM trades WHERE email=%s", (email,))
        tcount = int(cur.fetchone()["n"])

        cur.execute("SELECT * FROM trades WHERE email=%s ORDER BY id DESC LIMIT %s", (email, int(trades_limit)))
        trows = [dict(r) for r in cur.fetchall()]

    with AUTO_LOCK:
        rr = AUTO_RUNNERS.get(email)
        running = bool(rr and rr.is_running())
        auto_status_obj = asdict(rr.status()) if running and rr else None

    return {
        "ok": True,
        "user": {"email": u["email"], "created_at": u["created_at"]},
        "state": {"equity": equity, "session_id": session_id},
        "exchange": {
            "connected": bool(ex),
            "exchange": ex["exchange"] if ex else None,
            "connected_at": ex["created_at"] if ex else None,
        },
        "trades": {"count": tcount, "recent": trows},
        "ai": {"running": running, "status": auto_status_obj},
    }


@admin_router.post("/admin/log-access")
def admin_log_access(
    user_email: str,
    tab: str = "overview",
    admin=Depends(_require_admin),
):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO admin_access_log(admin_email, user_email, timestamp, tab_viewed) VALUES(%s,%s,%s,%s)",
            (admin, user_email, _now_utc_str(), tab),
        )
        conn.commit()
    return {"ok": True}


@admin_router.get("/admin/access-log")
def admin_access_log_list(
    admin=Depends(_require_admin),
    limit: int = Query(default=200, ge=1, le=2000),
):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT admin_email, user_email, timestamp, tab_viewed FROM admin_access_log ORDER BY id DESC LIMIT %s",
            (int(limit),),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"ok": True, "log": rows}


@admin_router.get("/admin/user/{email}/log")
def admin_user_log(
    email: str,
    admin=Depends(_require_admin),
    limit: int = Query(default=50, ge=1, le=200),
):
    from main import AUTO_RUNNERS, AUTO_LOCK
    email = (email or "").strip().lower()
    with AUTO_LOCK:
        rr = AUTO_RUNNERS.get(email)
        running = bool(rr and rr.is_running())

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, symbol, mode, trade_style, started_at, ended_at, stop_reason "
            "FROM ai_sessions WHERE email=%s ORDER BY id DESC LIMIT 20",
            (email,),
        )
        sessions = [dict(r) for r in cur.fetchall()]
        for sess in sessions:
            cur.execute(
                "SELECT t, msg FROM ai_logs WHERE email=%s AND session_id=%s "
                "ORDER BY id ASC LIMIT %s",
                (email, sess["id"], int(limit)),
            )
            sess["events"] = [{"t": r["t"], "msg": r["msg"]} for r in cur.fetchall()]

    return {"ok": True, "running": running, "sessions": sessions}


@admin_router.get("/admin/user/{email}/portfolio")
def admin_user_portfolio(email: str, admin=Depends(_require_admin)):
    import statistics as _st
    from datetime import timezone as _tz, timedelta as _td

    email = (email or "").strip().lower()
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
            from datetime import datetime as _dt
            return _dt.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            return None

    def _dubai_month(s):
        dt = _parse_dt(s)
        if not dt:
            return "Unknown"
        dubai = dt + _td(hours=4)
        return dubai.strftime("%Y-%m")

    pnls = [float(t["unreal_pnl_value"]) for t in rows]
    total = len(rows)
    wins = [t for t in rows if float(t["unreal_pnl_value"]) >= 0]
    losses = [t for t in rows if float(t["unreal_pnl_value"]) < 0]
    win_rate = len(wins) / total if total else 0
    total_pnl = sum(pnls)

    first_eq = float(rows[0]["equity_after"]) - pnls[0]
    eq_curve = [{"time": str(rows[0]["time"]), "equity": round(first_eq, 2)}]
    for t in rows:
        eq_curve.append({"time": str(t["time"]), "equity": round(float(t["equity_after"]), 2)})

    peak = eq_curve[0]["equity"]
    max_dd = 0.0
    for pt in eq_curve[1:]:
        if pt["equity"] > peak:
            peak = pt["equity"]
        dd = (peak - pt["equity"]) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    monthly: dict = {}
    for t in rows:
        m = _dubai_month(str(t["time"]))
        monthly.setdefault(m, 0.0)
        monthly[m] += float(t["unreal_pnl_value"])

    best = max(rows, key=lambda t: float(t["unreal_pnl_value"]))
    worst = min(rows, key=lambda t: float(t["unreal_pnl_value"]))

    return {
        "empty": False,
        "total_trades": total,
        "win_rate": round(win_rate * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "eq_curve": eq_curve,
        "monthly_pnl": [{"month": k, "pnl": round(v, 2)} for k, v in sorted(monthly.items())],
        "best_trade": {"symbol": best["symbol"], "pnl": round(float(best["unreal_pnl_value"]), 2), "time": str(best["time"])},
        "worst_trade": {"symbol": worst["symbol"], "pnl": round(float(worst["unreal_pnl_value"]), 2), "time": str(worst["time"])},
    }


@admin_router.get("/admin/user/{email}/notes")
def admin_get_notes(email: str, admin=Depends(_require_admin)):
    email = (email or "").strip().lower()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT notes, updated_at, updated_by FROM admin_notes WHERE email=%s", (email,))
        row = cur.fetchone()
    if row:
        return {"ok": True, "notes": row["notes"], "updated_at": row["updated_at"], "updated_by": row["updated_by"]}
    return {"ok": True, "notes": "", "updated_at": None, "updated_by": None}


@admin_router.post("/admin/user/{email}/notes")
def admin_save_notes(email: str, payload: AdminNotesIn, admin=Depends(_require_admin)):
    email = (email or "").strip().lower()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO admin_notes(email, notes, updated_at, updated_by)
               VALUES(%s,%s,%s,%s)
               ON CONFLICT(email) DO UPDATE SET notes=EXCLUDED.notes,
                   updated_at=EXCLUDED.updated_at, updated_by=EXCLUDED.updated_by""",
            (email, payload.notes, _now_utc_str(), admin),
        )
        conn.commit()
    return {"ok": True}


# ── Analytics ──────────────────────────────────────────────────────────────────

@admin_router.get("/analytics/overview")
def analytics_overview(admin=Depends(_require_admin)):
    from main import AUTO_RUNNERS
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT email, unreal_pnl_value, time, equity_after FROM trades ORDER BY time DESC")
            all_trades = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT COUNT(*) as cnt FROM user_state")
            user_count = int((cur.fetchone() or {}).get("cnt", 0))

        stats = _analytics_win_rate(all_trades)

        acct_pnl: Dict[str, float] = {}
        for t in all_trades:
            em = t.get("email") or "unknown"
            acct_pnl[em] = round(acct_pnl.get(em, 0.0) + float(t.get("unreal_pnl_value") or 0), 2)

        best_acct  = max(acct_pnl, key=acct_pnl.get) if acct_pnl else None
        worst_acct = min(acct_pnl, key=acct_pnl.get) if acct_pnl else None

        return {
            "ok": True,
            "total_trades":      stats["total"],
            "combined_win_rate": stats["win_rate"],
            "avg_win":           stats["avg_win"],
            "avg_loss":          stats["avg_loss"],
            "total_pnl":         stats["total_pnl"],
            "total_accounts":    user_count,
            "live_running":      len(AUTO_RUNNERS),
            "best_account":      best_acct,
            "worst_account":     worst_acct,
            "best_account_pnl":  round(acct_pnl.get(best_acct,  0), 2) if best_acct  else 0,
            "worst_account_pnl": round(acct_pnl.get(worst_acct, 0), 2) if worst_acct else 0,
        }
    except Exception as _e:
        print(f"[analytics/overview] error: {_e}")
        return {"ok": False, "error": str(_e), "total_trades": 0, "combined_win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "total_pnl": 0, "total_accounts": 0,
                "live_running": len(AUTO_RUNNERS), "best_account": None, "worst_account": None,
                "best_account_pnl": 0, "worst_account_pnl": 0}


@admin_router.get("/analytics/win-rates")
def analytics_win_rates(admin=Depends(_require_admin)):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT unreal_pnl_value, mode, side FROM trades")
            rows = [dict(r) for r in cur.fetchall()]

        def _band(rows_in: List[Dict], key: str, val: str) -> Dict:
            return _analytics_win_rate([r for r in rows_in if (r.get(key) or "") == val])

        def _lev_band(rows_in: List[Dict], lo: float, hi: float) -> Dict:
            return _analytics_win_rate([
                r for r in rows_in
                if lo <= float(r.get("leverage") or 0) <= hi
            ])

        return {
            "ok": True,
            "by_grade": {
                "LONG":  _band(rows, "side", "LONG"),
                "SHORT": _band(rows, "side", "SHORT"),
            },
            "by_score": {
                "ULTRA_SAFE": _band(rows, "mode", "ULTRA_SAFE"),
                "SAFE":       _band(rows, "mode", "SAFE"),
                "NORMAL":     _band(rows, "mode", "NORMAL"),
                "MINI_ASYM":  _band(rows, "mode", "MINI_ASYM"),
                "AGGRESSIVE": _band(rows, "mode", "AGGRESSIVE"),
            },
            "by_leverage": {
                "Low (1-5×)":  _lev_band(rows, 1, 5),
                "Mid (6-10×)": _lev_band(rows, 6, 10),
                "High (11+×)": _lev_band(rows, 11, 999),
            },
            "by_session": {},
        }
    except Exception as _e:
        print(f"[analytics/win-rates] error: {_e}")
        return {"ok": False, "error": str(_e), "by_grade": {}, "by_score": {}, "by_regime": {}, "by_session": {}}


@admin_router.get("/analytics/by-coin")
def analytics_by_coin(admin=Depends(_require_admin)):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT symbol, unreal_pnl_value FROM trades")
            rows = [dict(r) for r in cur.fetchall()]

        coins: Dict[str, List[Dict]] = {}
        for r in rows:
            sym = (r.get("symbol") or "UNKNOWN").upper()
            coins.setdefault(sym, []).append(r)

        result = {sym: _analytics_win_rate(coin_rows) for sym, coin_rows in sorted(coins.items())}
        return {"ok": True, "by_coin": result}
    except Exception as _e:
        print(f"[analytics/by-coin] error: {_e}")
        return {"ok": False, "error": str(_e), "by_coin": {}}


@admin_router.get("/analytics/by-regime")
def analytics_by_regime(admin=Depends(_require_admin)):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            if USING_PG:
                cur.execute(
                    "SELECT SUBSTRING(time, 1, 10) AS day, COUNT(*) AS cnt, "
                    "AVG(CASE WHEN unreal_pnl_value > 0 THEN 1.0 ELSE 0.0 END) AS win_rate "
                    "FROM trades "
                    "WHERE LENGTH(time) >= 10 AND SUBSTRING(time, 1, 10) >= "
                    "TO_CHAR(NOW() - INTERVAL '30 days', 'YYYY-MM-DD') "
                    "GROUP BY SUBSTRING(time, 1, 10) ORDER BY day ASC"
                )
            else:
                cur.execute(
                    "SELECT SUBSTR(time, 1, 10) AS day, COUNT(*) AS cnt, "
                    "AVG(CASE WHEN unreal_pnl_value > 0 THEN 1.0 ELSE 0.0 END) AS win_rate "
                    "FROM trades "
                    "WHERE LENGTH(time) >= 10 AND SUBSTR(time, 1, 10) >= "
                    "strftime('%Y-%m-%d', datetime('now', '-30 days')) "
                    "GROUP BY SUBSTR(time, 1, 10) ORDER BY day ASC"
                )
            rows = [
                {"day": str(r["day"]), "avg_score": round(float(r["win_rate"] or 0), 3), "count": int(r["cnt"])}
                for r in cur.fetchall()
            ]
        return {"ok": True, "signal_trend": rows}
    except Exception as _e:
        print(f"[analytics/by-regime] error: {_e}")
        return {"ok": False, "error": str(_e), "signal_trend": []}


@admin_router.get("/analytics/by-session")
def analytics_by_session(admin=Depends(_require_admin)):
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT unreal_pnl_value, time FROM trades")
            rows = [dict(r) for r in cur.fetchall()]

        monthly: Dict[str, float] = {}
        for r in rows:
            t_str = str(r.get("time") or "")
            if len(t_str) >= 7:
                month = t_str[:7]
                monthly[month] = round(monthly.get(month, 0.0) + float(r.get("unreal_pnl_value") or 0), 2)
        monthly_sorted = [{"month": k, "pnl": v} for k, v in sorted(monthly.items())[-6:]]

        return {"ok": True, "by_style": {}, "monthly_returns": monthly_sorted}
    except Exception as _e:
        print(f"[analytics/by-session] error: {_e}")
        return {"ok": False, "error": str(_e), "by_style": {}, "monthly_returns": []}
