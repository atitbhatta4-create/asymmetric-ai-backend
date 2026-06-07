"""
notifications.py — Telegram alerts and all email functions.

Depends only on: config.py, stdlib.
Nothing here touches the DB, engine, or FastAPI.
"""
from __future__ import annotations

import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import (
    SMTP_USER, SMTP_PASS, SMTP_HOST, SMTP_PORT,
    TG_TOKEN, TG_CHAT,
    TRADE_STYLE_PARAMS,
)


# ── Telegram ──────────────────────────────────────────────────────────────────

def tg_alert(text: str) -> None:
    """Fire-and-forget Telegram message. Never raises, never blocks the engine."""
    if not (TG_TOKEN and TG_CHAT):
        return
    def _send():
        try:
            import urllib.request, urllib.parse
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            payload = urllib.parse.urlencode({
                "chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"
            }).encode()
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=8):
                pass
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


# ── Email helpers ─────────────────────────────────────────────────────────────

def _email_base(content: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#050814;font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb;">
  <div style="max-width:520px;margin:32px auto;padding:0 16px;">
    <div style="background:#0a0f1e;border:1px solid rgba(255,255,255,0.08);border-radius:18px;overflow:hidden;">
      <div style="padding:18px 24px;border-bottom:1px solid rgba(255,255,255,0.06);">
        <span style="font-size:17px;font-weight:900;color:#00ffe0;">Asymmetric AI</span>
      </div>
      <div style="padding:24px;">{content}</div>
      <div style="padding:14px 24px;border-top:1px solid rgba(255,255,255,0.06);font-size:12px;color:#4b5563;">
        Demo mode &nbsp;·&nbsp; No real funds &nbsp;·&nbsp; Educational only
      </div>
    </div>
  </div>
</body></html>"""


def _send_email_sync(to: str, subject: str, html: str) -> None:
    if not SMTP_USER or not SMTP_PASS:
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"Asymmetric AI <{SMTP_USER}>"
        msg["To"]      = to
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, to, msg.as_string())
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


def send_email(to: str, subject: str, html: str) -> None:
    """Fire-and-forget — never blocks the main thread."""
    threading.Thread(target=_send_email_sync, args=(to, subject, html), daemon=True).start()


# ── Email functions ───────────────────────────────────────────────────────────

def email_password_changed(to: str) -> None:
    content = f"""
    <h2 style="margin:0 0 14px;font-size:20px;font-weight:900;color:#f1f5f9;">Password Changed</h2>
    <p style="margin:0 0 14px;opacity:0.85;line-height:1.6;">
      Your password for <b>{to}</b> was changed successfully.
    </p>
    <div style="background:rgba(220,38,38,0.12);border:1px solid rgba(248,113,113,0.3);
                border-radius:12px;padding:12px 16px;font-size:13px;color:#fecaca;">
      If this wasn't you, change your password immediately and secure your account.
    </div>"""
    send_email(to, "Your Asymmetric AI password was changed", _email_base(content))


def email_ai_started(to: str, symbol: str, mode: str, trade_style: str,
                     duration_days: int, max_trades: int, stop_after_bad: int) -> None:
    sp = TRADE_STYLE_PARAMS.get(trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
    interval_str = f"{sp['interval'] // 60}m"
    duration_str = f"{duration_days} day{'s' if duration_days != 1 else ''}" if duration_days > 0 else "Unlimited"
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">AI Trading Started</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">
      Your AI trader is live and monitoring the market.
    </p>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:16px;margin-bottom:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        {''.join(f'<tr><td style="padding:7px 0;color:#6b7280;width:140px;">{k}</td><td style="padding:7px 0;font-weight:900;color:#f1f5f9;">{v}</td></tr>' for k,v in [
            ("Coin", symbol), ("Mode", mode), ("Style", trade_style),
            ("Timeframe", sp["tf"]), ("Check every", interval_str), ("Duration", duration_str),
            ("Max trades / day", str(max_trades)), ("Bad trade limit", f"{stop_after_bad} per day"),
        ])}
      </table>
    </div>
    <p style="margin:0;font-size:12px;color:#4b5563;">
      The AI only trades when all 4 signal layers pass.
      You will receive an email for each completed trade.
    </p>"""
    send_email(to, f"AI started — {symbol} {mode}", _email_base(content))


def email_trade_opened(to: str, symbol: str, side: str, mode: str,
                       grade: str, entry: float, sl: float, tp: float,
                       score: float, equity: float) -> None:
    side_color = "#00ff9d" if side == "LONG" else "#ff5078"
    grade_color = "#f1f5f9" if grade == "A" else "#f59e0b"
    risk_pct   = abs(entry - sl) / entry * 100
    reward_pct = abs(tp - entry) / entry * 100
    content = f"""
    <h2 style="margin:0 0 4px;font-size:20px;font-weight:900;color:#f1f5f9;">Trade Opened</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">{symbol} &nbsp;·&nbsp; {mode}</p>

    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:20px;margin-bottom:14px;text-align:center;">
      <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">Direction</div>
      <div style="font-size:32px;font-weight:900;color:{side_color};">{side}</div>
      <div style="font-size:14px;color:{grade_color};margin-top:6px;font-weight:700;">
        Grade {grade} &nbsp;·&nbsp; Score {score:.2f}
      </div>
    </div>

    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        {''.join(f'<tr><td style="padding:6px 0;color:#6b7280;width:140px;">{k}</td><td style="padding:6px 0;font-weight:900;color:{c};">{v}</td></tr>' for k,v,c in [
            ("Entry price",   f"${entry:,.4f}",                    "#f1f5f9"),
            ("Stop loss",     f"${sl:,.4f}  (−{risk_pct:.2f}%)",  "#ff5078"),
            ("Take profit",   f"${tp:,.4f}  (+{reward_pct:.2f}%)", "#00ff9d"),
            ("Equity",        f"${equity:,.2f} USDT",              "#f1f5f9"),
        ])}
      </table>
    </div>"""
    subject = f"Trade opened — {side} {symbol} @ ${entry:,.4f}"
    send_email(to, subject, _email_base(content))


def email_trade_closed(to: str, symbol: str, side: str, mode: str,
                       entry: float, exit_price: float, outcome: str,
                       pnl_pct: float, pnl_value: float, equity_after: float,
                       label: str = "",
                       session_trades: int = 0, session_wins: int = 0,
                       session_losses: int = 0, session_pnl: float = 0.0) -> None:
    outcome_label = (
        "Take profit hit"    if outcome == "TP_HIT"
        else "Stop loss hit" if outcome == "SL_HIT"
        else "Trailing stop" if outcome == "TRAIL_STOP"
        else "Natural close"
    )
    win        = pnl_value >= 0
    pnl_color  = "#00ff9d" if win else "#ff5078"
    side_color = "#00ff9d" if side == "LONG" else "#ff5078"
    sign       = "+" if win else ""
    outcome_icon = "✓" if win else "✗"

    # Trade label for subject: "T1", "T2", or "Grade A"
    trade_label = label if label else "Trade"
    sess_sign   = "+" if session_pnl >= 0 else ""
    sess_color  = "#00ff9d" if session_pnl >= 0 else "#ff5078"

    # Session summary block — only shown when session data is provided
    session_block = ""
    if session_trades > 0:
        session_block = f"""
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:16px;margin-top:14px;">
      <div style="font-size:12px;color:#6b7280;margin-bottom:10px;text-transform:uppercase;
                  letter-spacing:0.05em;">Today's Session</div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        {''.join(f'<tr><td style="padding:5px 0;color:#6b7280;width:130px;">{k}</td><td style="padding:5px 0;font-weight:900;color:{c};">{v}</td></tr>' for k,v,c in [
            ("Trades",    str(session_trades),                        "#f1f5f9"),
            ("Wins",      str(session_wins),                          "#00ff9d"),
            ("Losses",    str(session_losses),                        "#ff5078"),
            ("Net P&L",   f"{sess_sign}${session_pnl:.2f}",          sess_color),
        ])}
      </table>
    </div>"""

    content = f"""
    <h2 style="margin:0 0 4px;font-size:20px;font-weight:900;color:#f1f5f9;">{trade_label} Closed</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">{symbol} &nbsp;·&nbsp; {mode}</p>

    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:20px;margin-bottom:14px;text-align:center;">
      <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">
        {outcome_label} {outcome_icon}
      </div>
      <div style="font-size:38px;font-weight:900;color:{pnl_color};">{sign}{pnl_pct:.2f}%</div>
      <div style="font-size:16px;font-weight:900;color:{pnl_color};margin-top:4px;">{sign}${pnl_value:.2f}</div>
    </div>

    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        {''.join(f'<tr><td style="padding:6px 0;color:#6b7280;width:130px;">{k}</td><td style="padding:6px 0;font-weight:900;color:{c};">{v}</td></tr>' for k,v,c in [
            ("Direction", side, side_color),
            ("Entry price", f"${entry:,.4f}", "#f1f5f9"),
            ("Exit price", f"${exit_price:,.4f}", "#f1f5f9"),
            ("Equity after", f"${equity_after:,.2f}", "#f1f5f9"),
        ])}
      </table>
    </div>{session_block}"""

    # Subject: "{trade_label} {+$pnl} | Today: {+$session_pnl}"
    sess_summary = f" | Today: {sess_sign}${session_pnl:.2f}" if session_trades > 0 else ""
    subject = f"{trade_label} {sign}${pnl_value:.2f} ({sign}{pnl_pct:.2f}%) — {symbol}{sess_summary}"
    send_email(to, subject, _email_base(content))


def email_ai_stopped(to: str, symbol: str, reason: str, equity: float) -> None:
    reason_map = {
        "MAX_DRAWDOWN":   ("Drawdown Limit Reached",   "#ff5078", "The AI hit the maximum drawdown limit and stopped to protect your remaining capital."),
        "HARD_FLOOR":     ("Safety Floor Triggered",   "#ff5078", "Equity fell below the 85% safety floor. AI stopped completely to protect your funds."),
        "MAX_BAD_TRADES": ("Bad Trade Limit Hit",      "#f59e0b", "Too many consecutive losing trades. AI paused for today — resets at midnight Dubai time."),
        "DURATION_END":   ("Session Ended",            "#00ffe0", "The AI completed its scheduled trading duration."),
    }
    title, color, detail = reason_map.get(reason, ("AI Stopped", "#94a3b8", f"Reason: {reason}"))
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">AI Trading Stopped</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">{symbol}</p>
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                border-radius:14px;padding:18px;margin-bottom:16px;">
      <div style="font-size:16px;font-weight:900;color:{color};margin-bottom:8px;">{title}</div>
      <div style="font-size:14px;opacity:0.85;line-height:1.6;">{detail}</div>
    </div>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:12px;
                padding:14px;font-size:14px;">
      <div style="opacity:0.6;margin-bottom:4px;">Current equity</div>
      <div style="font-size:22px;font-weight:900;color:#f1f5f9;">${equity:,.2f}</div>
    </div>
    <p style="margin-top:16px;font-size:12px;color:#4b5563;">
      Log in to Asymmetric AI to review your trades and restart when ready.
    </p>"""
    send_email(to, f"AI stopped — {symbol} ({title})", _email_base(content))


def email_otp_reset(to: str, code: str) -> None:
    content = f"""
    <h2 style="margin:0 0 14px;font-size:20px;font-weight:900;color:#f1f5f9;">Reset Your Password</h2>
    <p style="margin:0 0 20px;opacity:0.85;line-height:1.6;">
      We received a request to reset the password for <b>{to}</b>.<br>
      Enter this code in the app to continue:
    </p>
    <div style="background:#0f172a;border:1px solid rgba(0,255,224,0.25);border-radius:14px;
                padding:28px;text-align:center;margin-bottom:20px;">
      <div style="font-size:42px;font-weight:900;letter-spacing:12px;color:#00ffe0;
                  font-family:monospace;">{code}</div>
      <div style="margin-top:10px;font-size:12px;color:#6b7280;">Expires in 15 minutes</div>
    </div>
    <div style="background:rgba(220,38,38,0.1);border:1px solid rgba(248,113,113,0.25);
                border-radius:10px;padding:12px 16px;font-size:13px;color:#fecaca;">
      If you did not request this, ignore this email. Your password has not been changed.
    </div>"""
    send_email(to, "Asymmetric AI — Password Reset Code", _email_base(content))


def email_support_reply(to: str, admin_message: str) -> None:
    content = f"""
    <h2 style="margin:0 0 14px;font-size:20px;font-weight:900;color:#f1f5f9;">Support Reply</h2>
    <p style="margin:0 0 18px;opacity:0.85;line-height:1.6;">
      Our team has replied to your support ticket:
    </p>
    <div style="background:#0f172a;border:1px solid rgba(0,255,224,0.20);border-radius:14px;
                padding:18px;margin-bottom:18px;font-size:14px;line-height:1.7;color:#f1f5f9;">
      {admin_message}
    </div>
    <p style="margin:0;font-size:12px;color:#4b5563;">
      Log in to Asymmetric AI to continue the conversation.
    </p>"""
    send_email(to, "Asymmetric AI — Support Reply", _email_base(content))


def email_2fa_enabled(to: str) -> None:
    content = f"""
    <h2 style="margin:0 0 14px;font-size:20px;font-weight:900;color:#f1f5f9;">Two-Factor Authentication Enabled</h2>
    <p style="margin:0 0 14px;opacity:0.85;line-height:1.6;">
      2FA has been successfully enabled on your account <b>{to}</b>.<br>
      You will now need your authenticator app every time you log in.
    </p>
    <div style="background:rgba(0,255,157,0.08);border:1px solid rgba(0,255,157,0.25);
                border-radius:12px;padding:12px 16px;font-size:13px;color:#a7f3d0;">
      If you did not enable this, contact support immediately and change your password.
    </div>"""
    send_email(to, "2FA enabled on your Asymmetric AI account", _email_base(content))
