"""
routes_support.py — FastAPI router for /support/* and /admin/support/* endpoints.

User endpoints:
    POST /support/ticket     — open a new ticket (or add to existing open one)
    GET  /support/ticket     — get latest ticket + all messages
    POST /support/message    — send a message on the current open ticket

Admin endpoints:
    GET  /admin/support/tickets                         — all tickets sorted by wait time
    GET  /admin/support/ticket/{id}/messages            — all messages for a ticket
    POST /admin/support/ticket/{id}/reply               — admin reply (sends email to user)
    POST /admin/support/ticket/{id}/resolve             — mark ticket resolved

Wire up in main.py:
    from routes_support import support_router
    app.include_router(support_router)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException
from pydantic import BaseModel, Field

from database import db_conn
from notifications import tg_alert, email_support_reply

# ── Bot FAQ knowledge base ─────────────────────────────────────────────────────

_FAQ: list[tuple[list[str], str]] = [
    (
        ["connect", "exchange", "api", "bybit", "okx", "binance", "key"],
        (
            "To connect your exchange, go to <b>Exchange</b> in the left menu and enter your API Key and Secret.\n\n"
            "When creating the API key on your exchange, make sure to enable <b>Read + Trade permissions</b> — do NOT enable withdrawal.\n\n"
            "📌 We recommend <b>Bybit</b> — best execution speed and lowest fees. OKX works great too. Binance US is geo-restricted on our servers, so use Bybit or OKX if you're in the US.\n\n"
            "Once connected, your exchange balance will appear on the Dashboard within a few seconds."
        ),
    ),
    (
        ["not starting", "won't start", "not start", "ai start", "start ai", "why isn't", "not working", "blocked"],
        (
            "Here are the most common reasons the AI won't start:\n\n"
            "1️⃣ <b>Exchange not connected</b> — go to the Exchange page and add your API keys first\n"
            "2️⃣ <b>Balance too low</b> — you need at least ~$50 to open a minimum-size position\n"
            "3️⃣ <b>Bad trade limit hit</b> — the AI pauses after too many losses in one day and auto-resets at midnight Dubai time (UTC+4)\n"
            "4️⃣ <b>Another session already running</b> — stop it first, then start a new one\n\n"
            "Check the status card on your Dashboard — it shows the exact reason the AI is currently blocked."
        ),
    ),
    (
        ["mini_asym", "mini asym", "miniasym", "flagship", "how does mini", "what is mini"],
        (
            "<b>MINI_ASYM</b> is our flagship mode and what we recommend for most users. Here's how it works:\n\n"
            "• <b>2× leverage</b> — much safer than AGGRESSIVE (8×)\n"
            "• <b>Dynamic position sizing</b> — automatically scales down when volatility spikes or your equity drops\n"
            "• <b>4-layer signal filter</b> — only enters trades when trend, momentum, market regime, and candle patterns all align at the same time\n"
            "• <b>Capital protection</b> — hard floor at 85% of starting equity; AI stops completely if it's ever hit\n\n"
            "Best pairing: <b>MINI_ASYM + DAY_TRADE</b> — this combination has the strongest historical Sharpe ratio in our backtests."
        ),
    ),
    (
        ["withdraw", "reset", "balance", "restart balance", "reset balance", "reset sandbox"],
        (
            "<b>Demo mode:</b> Go to Dashboard → scroll down → tap <b>Reset Sandbox</b>. This resets your balance back to the starting amount. Your trade history is kept.\n\n"
            "<b>Real trading mode:</b> Asymmetric AI never holds your funds — they stay on your exchange at all times. To withdraw, log directly into Bybit / OKX / Binance and withdraw from there.\n\n"
            "If your balance shows $0 and you haven't reset it, please message us and we'll fix it manually."
        ),
    ),
    (
        ["which mode", "best mode", "what mode", "recommend", "best for me", "which is best", "mode should"],
        (
            "Our recommendation for most users is <b>MINI_ASYM + DAY_TRADE</b>.\n\n"
            "Here's a quick comparison:\n\n"
            "🟢 <b>MINI_ASYM</b> (2× leverage) — Flagship. Best risk/reward. Recommended for 95% of users\n"
            "🔵 <b>SAFE</b> (1× leverage) — Most conservative. Slow but very protected\n"
            "🟡 <b>NORMAL</b> (3× leverage) — Good middle ground for experienced traders\n"
            "🔴 <b>AGGRESSIVE</b> (8× leverage) — Highest potential returns but significant drawdown risk. Not recommended for beginners\n\n"
            "For trade style: <b>DAY_TRADE</b> performs best on most coins in our backtests, followed by SWING. SCALP with AGGRESSIVE mode showed negative returns — avoid that combination."
        ),
    ),
    (
        ["leverage", "how much leverage", "what leverage"],
        (
            "Each mode uses different leverage:\n\n"
            "• ULTRA_SAFE: 0.5× (no real leverage)\n"
            "• SAFE: 1×\n"
            "• NORMAL: 3×\n"
            "• MINI_ASYM: 2× (with smart dynamic sizing)\n"
            "• AGGRESSIVE: 8×\n\n"
            "Remember: leverage multiplies both gains AND losses. MINI_ASYM's dynamic sizing means it uses less than 2× when conditions are risky — making it behave more conservatively than the number suggests."
        ),
    ),
    (
        ["fee", "fees", "cost", "pricing", "free", "subscription", "pay"],
        (
            "Asymmetric AI is currently <b>free to use</b> during our beta phase.\n\n"
            "Paid plans with additional features will launch later this year. All current users will be grandfathered in at a special rate.\n\n"
            "Your exchange charges its own trading fees directly — typically 0.02-0.1% per trade depending on the exchange."
        ),
    ),
    (
        ["safe", "is it safe", "trust", "funds", "money", "secure"],
        (
            "Your funds are always safe on your exchange — Asymmetric AI <b>never has access to your money directly</b>.\n\n"
            "We connect using an API key with Trade-only permissions (no withdrawal access). Even if our servers were compromised, your funds cannot be moved.\n\n"
            "The AI can only open and close positions. It cannot deposit, withdraw, or transfer funds."
        ),
    ),
]


def _find_bot_answer(message: str, user_mode: str | None = None) -> str | None:
    """Return a bot answer if the message matches a known FAQ, else None."""
    msg = message.lower()
    for keywords, answer in _FAQ:
        if any(kw in msg for kw in keywords):
            # Personalise with current mode if we know it
            if user_mode and "which mode" in msg or "best mode" in msg or "recommend" in msg:
                answer = (
                    f"You're currently on <b>{user_mode}</b> mode. "
                    + answer.replace("Our recommendation for most users is", "For context,")
                )
            return answer
    return None

support_router = APIRouter(tags=["support"])


# ── Auth (standalone — no import from main.py to avoid circular imports) ───────

_SESSION_CACHE: dict = {}
_SESSION_TTL = 300


def _require_user(session: Optional[str] = Cookie(default=None)) -> dict:
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


_ADMIN_EMAILS: set[str] | None = None


def _get_admin_emails() -> set[str]:
    global _ADMIN_EMAILS
    if _ADMIN_EMAILS is None:
        default = "admin@demo.com"
        _ADMIN_EMAILS = {
            e.strip().lower()
            for e in os.getenv("ADMIN_EMAILS", default).split(",")
            if e.strip()
        }
    return _ADMIN_EMAILS


def _require_admin(user=Depends(_require_user)) -> str:
    email = user["email"].strip().lower()
    if email not in _get_admin_emails():
        raise HTTPException(status_code=403, detail="Admin access only")
    return email


# ── Pydantic models ────────────────────────────────────────────────────────────

class TicketIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    subject: str = Field(default="", max_length=200)


class MessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ReplyIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


# ── User endpoints ─────────────────────────────────────────────────────────────

@support_router.post("/support/ticket")
def open_ticket(body: TicketIn, user=Depends(_require_user)):
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM support_tickets WHERE user_email=%s AND status='open' "
            "ORDER BY created_at DESC LIMIT 1",
            (email,),
        )
        row = cur.fetchone()
        if row:
            ticket_id = row["id"]
            cur.execute(
                "INSERT INTO support_messages(ticket_id, sender, message) VALUES(%s,'user',%s)",
                (ticket_id, body.message.strip()),
            )
            cur.execute(
                "UPDATE support_tickets SET last_user_msg_at=NOW() WHERE id=%s",
                (ticket_id,),
            )
        else:
            subject = body.subject.strip() or "Support Request"
            cur.execute(
                "INSERT INTO support_tickets(user_email, status, subject) VALUES(%s,'open',%s) RETURNING id",
                (email, subject),
            )
            ticket_id = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO support_messages(ticket_id, sender, message) VALUES(%s,'user',%s)",
                (ticket_id, body.message.strip()),
            )
        conn.commit()

    tg_alert(
        f"💬 <b>New support message</b>\n"
        f"From: {email}\n"
        f"Message: <code>{body.message.strip()[:200]}</code>"
    )
    return {"ok": True, "ticket_id": ticket_id}


@support_router.get("/support/ticket")
def get_ticket(user=Depends(_require_user)):
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, status, subject, created_at, last_user_msg_at, last_admin_reply_at "
            "FROM support_tickets WHERE user_email=%s ORDER BY created_at DESC LIMIT 1",
            (email,),
        )
        ticket = cur.fetchone()
        if not ticket:
            return {"ok": True, "ticket": None, "messages": []}
        ticket_id = ticket["id"]
        cur.execute(
            "SELECT sender, message, sent_at FROM support_messages "
            "WHERE ticket_id=%s ORDER BY sent_at ASC",
            (ticket_id,),
        )
        messages = [dict(m) for m in cur.fetchall()]

    return {
        "ok": True,
        "ticket": {
            "id": ticket["id"],
            "status": ticket["status"],
            "subject": ticket["subject"],
            "created_at": str(ticket["created_at"]),
            "last_admin_reply_at": (
                str(ticket["last_admin_reply_at"]) if ticket["last_admin_reply_at"] else None
            ),
        },
        "messages": [
            {"sender": m["sender"], "message": m["message"], "sent_at": str(m["sent_at"])}
            for m in messages
        ],
    }


@support_router.post("/support/message")
def send_message(body: MessageIn, user=Depends(_require_user)):
    email = user["email"]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM support_tickets WHERE user_email=%s AND status='open' "
            "ORDER BY created_at DESC LIMIT 1",
            (email,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No open ticket")
        ticket_id = row["id"]
        cur.execute(
            "INSERT INTO support_messages(ticket_id, sender, message) VALUES(%s,'user',%s)",
            (ticket_id, body.message.strip()),
        )
        cur.execute(
            "UPDATE support_tickets SET last_user_msg_at=NOW() WHERE id=%s",
            (ticket_id,),
        )
        conn.commit()

    tg_alert(
        f"💬 <b>Support message</b>\n"
        f"From: {email}\n"
        f"Message: <code>{body.message.strip()[:200]}</code>"
    )
    return {"ok": True}


@support_router.post("/support/bot-reply")
def bot_reply(body: MessageIn, user=Depends(_require_user)):
    """Check if message matches a known FAQ and save an automatic reply."""
    email = user["email"]

    # Look up user's current mode for personalised answers
    user_mode: str | None = None
    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT mode FROM ai_runner_state WHERE email=%s",
                (email,)
            )
            row = cur.fetchone()
            if row:
                user_mode = row["mode"]
    except Exception:
        pass

    answer = _find_bot_answer(body.message, user_mode)
    if not answer:
        return {"ok": True, "replied": False}

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM support_tickets WHERE user_email=%s AND status='open' "
            "ORDER BY created_at DESC LIMIT 1",
            (email,),
        )
        row = cur.fetchone()
        if not row:
            return {"ok": True, "replied": False}
        cur.execute(
            "INSERT INTO support_messages(ticket_id, sender, message) VALUES(%s,'admin',%s)",
            (row["id"], answer),
        )
        cur.execute(
            "UPDATE support_tickets SET last_admin_reply_at=NOW() WHERE id=%s",
            (row["id"],),
        )
        conn.commit()

    return {"ok": True, "replied": True}


# ── Admin endpoints ────────────────────────────────────────────────────────────

@support_router.get("/admin/support/tickets")
def admin_list_tickets(admin=Depends(_require_admin)):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_email, status, subject, created_at,
                   last_user_msg_at, last_admin_reply_at
            FROM support_tickets
            ORDER BY
                CASE WHEN status = 'open' THEN 0 ELSE 1 END,
                last_user_msg_at ASC
        """)
        tickets = [dict(r) for r in cur.fetchall()]

    now_ts = datetime.now(timezone.utc)
    result = []
    for t in tickets:
        last_msg = t["last_user_msg_at"]
        wait_hours = 0.0
        if last_msg:
            lm = last_msg if (hasattr(last_msg, "tzinfo") and last_msg.tzinfo) else last_msg.replace(tzinfo=timezone.utc)
            wait_hours = (now_ts - lm).total_seconds() / 3600
        result.append({
            "id": t["id"],
            "user_email": t["user_email"],
            "status": t["status"],
            "subject": t["subject"],
            "created_at": str(t["created_at"]),
            "last_user_msg_at": str(t["last_user_msg_at"]) if t["last_user_msg_at"] else None,
            "last_admin_reply_at": str(t["last_admin_reply_at"]) if t["last_admin_reply_at"] else None,
            "wait_hours": round(wait_hours, 1),
        })
    return {"ok": True, "tickets": result}


@support_router.get("/admin/support/ticket/{ticket_id}/messages")
def admin_get_messages(ticket_id: int, admin=Depends(_require_admin)):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_email, status, subject FROM support_tickets WHERE id=%s",
            (ticket_id,),
        )
        ticket = cur.fetchone()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        cur.execute(
            "SELECT sender, message, sent_at FROM support_messages "
            "WHERE ticket_id=%s ORDER BY sent_at ASC",
            (ticket_id,),
        )
        messages = [dict(m) for m in cur.fetchall()]

    return {
        "ok": True,
        "ticket": dict(ticket),
        "messages": [
            {"sender": m["sender"], "message": m["message"], "sent_at": str(m["sent_at"])}
            for m in messages
        ],
    }


@support_router.post("/admin/support/ticket/{ticket_id}/reply")
def admin_reply(ticket_id: int, body: ReplyIn, admin=Depends(_require_admin)):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_email FROM support_tickets WHERE id=%s",
            (ticket_id,),
        )
        ticket = cur.fetchone()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        cur.execute(
            "INSERT INTO support_messages(ticket_id, sender, message) VALUES(%s,'admin',%s)",
            (ticket_id, body.message.strip()),
        )
        cur.execute(
            "UPDATE support_tickets SET last_admin_reply_at=NOW() WHERE id=%s",
            (ticket_id,),
        )
        conn.commit()

    email_support_reply(ticket["user_email"], body.message.strip())
    return {"ok": True}


@support_router.post("/admin/support/ticket/{ticket_id}/resolve")
def admin_resolve(ticket_id: int, admin=Depends(_require_admin)):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM support_tickets WHERE id=%s", (ticket_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Ticket not found")
        cur.execute("UPDATE support_tickets SET status='resolved' WHERE id=%s", (ticket_id,))
        conn.commit()
    return {"ok": True}
