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

def _letter_base(content: str, footer: str = "Automated trading involves risk. Never trade more than you can afford to lose.") -> str:
    """Light-background letter layout for welcome and guide emails."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Asymmetric AI</title></head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
  <div style="max-width:580px;margin:0 auto;padding:40px 16px 48px;">

    <!-- Logo -->
    <div style="padding:0 8px 28px;">
      <span style="font-size:16px;font-weight:900;color:#0a0f1e;letter-spacing:-0.01em;">Asymmetric</span><span style="font-size:16px;font-weight:900;color:#00b89c;letter-spacing:-0.01em;"> AI</span>
    </div>

    <!-- White card -->
    <div style="background:#ffffff;border-radius:12px;padding:36px 36px 32px;box-shadow:0 1px 4px rgba(0,0,0,0.08);">

      {content}

    </div>

    <!-- Footer -->
    <div style="padding:20px 8px 0;font-size:11px;color:#9ca3af;line-height:1.7;">
      {footer}<br>
      &copy; Asymmetric AI &nbsp;&middot;&nbsp; You received this because you have an account with us.
    </div>
  </div>
</body></html>"""


def _email_base(content: str, footer: str = "Automated trading involves risk. Never trade more than you can afford to lose.") -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Asymmetric AI</title></head>
<body style="margin:0;padding:0;background:#060a18;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;color:#e2e8f0;">
  <div style="max-width:540px;margin:36px auto;padding:0 16px 32px;">

    <!-- Header -->
    <div style="margin-bottom:6px;">
      <span style="font-size:15px;font-weight:900;color:#00ffe0;letter-spacing:-0.02em;">Asymmetric AI</span>
    </div>

    <!-- Card -->
    <div style="background:#0d1426;border:1px solid rgba(255,255,255,0.09);border-radius:16px;overflow:hidden;
                box-shadow:0 4px 24px rgba(0,0,0,0.4);">

      <!-- Accent bar -->
      <div style="height:3px;background:linear-gradient(90deg,#00ffe0,#00ff9d);"></div>

      <!-- Body -->
      <div style="padding:28px 28px 24px;">{content}</div>

      <!-- Footer -->
      <div style="padding:14px 28px 18px;border-top:1px solid rgba(255,255,255,0.06);">
        <div style="font-size:11px;color:#374151;line-height:1.6;">{footer}</div>
        <div style="margin-top:8px;font-size:11px;color:#1f2937;">
          &copy; Asymmetric AI &nbsp;&middot;&nbsp; You are receiving this because you have an account with us.
        </div>
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
    <div style="font-size:19px;font-weight:800;color:#f1f5f9;margin-bottom:6px;">Password Changed</div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:20px;">Security notification for {to}</div>
    <div style="font-size:14px;color:#94a3b8;line-height:1.7;margin-bottom:18px;">
      Your password was changed successfully. You can continue using Asymmetric AI with your new password.
    </div>
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
                border-radius:10px;padding:13px 16px;font-size:13px;color:#fca5a5;line-height:1.6;">
      If you did not make this change, reset your password immediately and contact our support team.
    </div>"""
    send_email(to, "Your Asymmetric AI password was changed", _email_base(content))


def email_ai_started(to: str, symbol: str, mode: str, trade_style: str,
                     duration_days: int, max_trades: int, stop_after_bad: int) -> None:
    sp = TRADE_STYLE_PARAMS.get(trade_style, TRADE_STYLE_PARAMS["DAY_TRADE"])
    interval_str = f"{sp['interval'] // 60}m"
    duration_str = f"{duration_days} day{'s' if duration_days != 1 else ''}" if duration_days > 0 else "Unlimited"
    content = f"""
    <div style="font-size:19px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">AI Trading Started</div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:22px;">Your AI is live and monitoring the market.</div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;overflow:hidden;margin-bottom:18px;">
      {''.join(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.05);"><span style="font-size:12px;color:#6b7280;">{k}</span><span style="font-size:13px;font-weight:700;color:#f1f5f9;">{v}</span></div>' for k,v in [
          ("Coin", symbol), ("Mode", mode), ("Style", trade_style),
          ("Timeframe", sp["tf"]), ("Checks every", interval_str), ("Duration", duration_str),
          ("Max trades / day", str(max_trades)), ("Bad trade limit", f"{stop_after_bad} per day"),
      ])}
      <div style="padding:10px 14px;font-size:11px;color:#374151;">Session configuration</div>
    </div>

    <div style="font-size:12px;color:#4b5563;line-height:1.7;">
      The AI only trades when all 4 signal layers align. Expect fewer trades than you might anticipate — this is by design. You will receive a notification for every trade that opens and closes.
    </div>"""
    send_email(to, f"AI started — {symbol} · {mode}", _email_base(content))


def email_trade_opened(to: str, symbol: str, side: str, mode: str,
                       grade: str, entry: float, sl: float, tp: float,
                       score: float, equity: float) -> None:
    side_color = "#00ff9d" if side == "LONG" else "#ff5078"
    grade_color = "#f1f5f9" if grade == "A" else "#f59e0b"
    risk_pct   = abs(entry - sl) / entry * 100
    reward_pct = abs(tp - entry) / entry * 100
    content = f"""
    <div style="font-size:19px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">Trade Opened</div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:22px;">{symbol} &nbsp;&middot;&nbsp; {mode}</div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;
                padding:20px;margin-bottom:14px;text-align:center;">
      <div style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;">Direction</div>
      <div style="font-size:34px;font-weight:900;color:{side_color};letter-spacing:0.04em;">{side}</div>
      <div style="margin-top:8px;display:inline-block;padding:3px 10px;border-radius:6px;
                  background:rgba(255,255,255,0.05);font-size:12px;font-weight:700;color:{grade_color};">
        Grade {grade} &nbsp;&middot;&nbsp; Score {score:.2f}
      </div>
    </div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;overflow:hidden;">
      {''.join(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:12px;color:#6b7280;">{k}</span><span style="font-size:13px;font-weight:700;color:{c};">{v}</span></div>' for k,v,c in [
          ("Entry price",   f"${entry:,.4f}",                    "#f1f5f9"),
          ("Stop loss",     f"${sl:,.4f}  (−{risk_pct:.2f}%)",  "#ff5078"),
          ("Take profit",   f"${tp:,.4f}  (+{reward_pct:.2f}%)", "#00ff9d"),
          ("Account equity", f"${equity:,.2f} USDT",             "#f1f5f9"),
      ])}
      <div style="padding:8px 14px;font-size:11px;color:#374151;">Trade details</div>
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
    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;overflow:hidden;margin-top:12px;">
      <div style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.05);">
        <span style="font-size:11px;font-weight:700;color:#4b5563;text-transform:uppercase;letter-spacing:0.06em;">Today's Session</span>
      </div>
      {''.join(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 14px;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:12px;color:#6b7280;">{k}</span><span style="font-size:13px;font-weight:700;color:{c};">{v}</span></div>' for k,v,c in [
          ("Trades today",  str(session_trades),                        "#f1f5f9"),
          ("Wins",          str(session_wins),                          "#00ff9d"),
          ("Losses",        str(session_losses),                        "#ff5078"),
          ("Net P&L",       f"{sess_sign}${session_pnl:.2f}",          sess_color),
      ])}
    </div>"""

    content = f"""
    <div style="font-size:19px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">{trade_label} Closed</div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:22px;">{symbol} &nbsp;&middot;&nbsp; {mode}</div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;
                padding:22px;margin-bottom:14px;text-align:center;">
      <div style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;">
        {outcome_label} &nbsp;{outcome_icon}
      </div>
      <div style="font-size:40px;font-weight:900;color:{pnl_color};letter-spacing:-0.02em;">{sign}{pnl_pct:.2f}%</div>
      <div style="font-size:16px;font-weight:700;color:{pnl_color};margin-top:4px;opacity:0.85;">{sign}${pnl_value:.2f}</div>
    </div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;overflow:hidden;">
      {''.join(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:12px;color:#6b7280;">{k}</span><span style="font-size:13px;font-weight:700;color:{c};">{v}</span></div>' for k,v,c in [
          ("Direction",     side,                          side_color),
          ("Entry price",   f"${entry:,.4f}",              "#f1f5f9"),
          ("Exit price",    f"${exit_price:,.4f}",         "#f1f5f9"),
          ("Equity after",  f"${equity_after:,.2f} USDT",  "#f1f5f9"),
      ])}
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
    renewal_note = ""
    if reason == "DURATION_END":
        renewal_note = """
    <div style="margin-top:18px;background:rgba(0,255,224,0.05);border:1px solid rgba(0,255,224,0.15);
                border-radius:10px;padding:13px 16px;font-size:13px;color:#a7f3d0;line-height:1.6;">
      Ready to continue? Log in to Asymmetric AI, set a new duration, and start a new session whenever you're ready.
    </div>"""

    content = f"""
    <div style="font-size:19px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">AI Trading Stopped</div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:22px;">{symbol}</div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;margin-bottom:16px;">
      <div style="font-size:14px;font-weight:800;color:{color};margin-bottom:8px;">{title}</div>
      <div style="font-size:13px;color:#94a3b8;line-height:1.65;">{detail}</div>
    </div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 16px;">
      <div style="font-size:11px;color:#4b5563;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;">Current equity</div>
      <div style="font-size:24px;font-weight:900;color:#f1f5f9;">${equity:,.2f} <span style="font-size:13px;color:#6b7280;font-weight:500;">USDT</span></div>
    </div>{renewal_note}

    <div style="margin-top:18px;font-size:12px;color:#4b5563;line-height:1.6;">
      Log in to Asymmetric AI to review your trade history and performance report.
    </div>"""
    send_email(to, f"AI stopped — {symbol} · {title}", _email_base(content))


def email_otp_reset(to: str, code: str) -> None:
    content = f"""
    <div style="font-size:19px;font-weight:800;color:#f1f5f9;margin-bottom:6px;">Reset Your Password</div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:22px;">
      We received a request to reset the password for {to}.
    </div>

    <div style="background:#0b1120;border:1px solid rgba(0,255,224,0.2);border-radius:12px;
                padding:30px 20px;text-align:center;margin-bottom:18px;">
      <div style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:14px;">Your reset code</div>
      <div style="font-size:38px;font-weight:900;letter-spacing:10px;color:#00ffe0;
                  font-family:'Courier New',monospace;">{code}</div>
      <div style="margin-top:12px;font-size:12px;color:#4b5563;">Expires in 15 minutes</div>
    </div>

    <div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.18);
                border-radius:10px;padding:12px 16px;font-size:13px;color:#fca5a5;line-height:1.6;">
      If you did not request a password reset, you can safely ignore this email. Your password has not been changed.
    </div>"""
    send_email(to, "Asymmetric AI — Password reset code", _email_base(content))


def email_support_reply(to: str, admin_message: str) -> None:
    content = f"""
    <div style="font-size:15px;color:#94a3b8;margin-bottom:18px;">Hi,</div>

    <div style="font-size:14px;color:#94a3b8;line-height:1.8;margin-bottom:20px;">
      You have a new message from our support team:
    </div>

    <div style="background:#0b1120;border-left:3px solid #00ffe0;border-radius:0 10px 10px 0;
                padding:16px 18px;margin-bottom:24px;font-size:14px;line-height:1.85;color:#e2e8f0;">
      {admin_message}
    </div>

    <div style="font-size:13px;color:#94a3b8;line-height:1.75;margin-bottom:28px;">
      You can reply directly from the <b style="color:#f1f5f9;">Support</b> section inside the Asymmetric AI app.
      We typically respond within a few hours.
    </div>

    <div style="font-size:13px;color:#64748b;line-height:1.7;">
      Warm regards,<br>
      <b style="color:#94a3b8;">The Asymmetric AI Team</b>
    </div>"""
    send_email(to, "Message from the Asymmetric AI Team", _email_base(content))


def email_2fa_enabled(to: str) -> None:
    content = f"""
    <div style="font-size:19px;font-weight:800;color:#f1f5f9;margin-bottom:6px;">Two-Factor Authentication Enabled</div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:20px;">Security update for {to}</div>

    <div style="font-size:14px;color:#94a3b8;line-height:1.7;margin-bottom:18px;">
      Two-factor authentication has been successfully enabled on your account. From now on, you will need your authenticator app every time you sign in.
    </div>

    <div style="background:rgba(0,255,157,0.06);border:1px solid rgba(0,255,157,0.2);
                border-radius:10px;padding:13px 16px;font-size:13px;color:#a7f3d0;line-height:1.6;">
      If you did not make this change, contact our support team immediately and change your password.
    </div>"""
    send_email(to, "2FA enabled on your Asymmetric AI account", _email_base(content))


# ── Onboarding emails ─────────────────────────────────────────────────────────

def email_welcome(to: str) -> None:
    content = """
      <p style="font-size:15px;color:#374151;margin:0 0 22px;line-height:1;">Hey!</p>

      <h1 style="font-size:24px;font-weight:800;color:#111827;margin:0 0 18px;line-height:1.25;letter-spacing:-0.02em;">
        Welcome to Asymmetric AI.
      </h1>

      <p style="font-size:15px;color:#4b5563;line-height:1.75;margin:0 0 16px;">
        We're really glad to have you here. Your account is active and ready to use.
      </p>

      <p style="font-size:15px;color:#4b5563;line-height:1.75;margin:0 0 16px;">
        Asymmetric AI is an AI-powered trading system that monitors the market 24/7 and places trades on your behalf using a 4-layer signal system — designed to find high-quality entries and protect your capital at every step. You stay in full control. The AI only uses your exchange API keys to trade, and can never access or withdraw your funds.
      </p>

      <p style="font-size:15px;color:#4b5563;line-height:1.75;margin:0 0 28px;">
        Before you start, open the app and complete the short setup guide — it walks you through connecting your exchange, choosing your mode and style, and placing your first session. It takes about 5 minutes, and once you're done we'll send you a full reference guide to keep in your inbox.
      </p>

      <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;margin-bottom:28px;">
        <p style="font-size:13px;font-weight:700;color:#111827;margin:0 0 10px;">What the setup guide covers</p>
        <p style="font-size:14px;color:#6b7280;margin:0;line-height:1.9;">
          &rarr; &nbsp;Connecting your exchange API (Bybit, OKX, or Binance)<br>
          &rarr; &nbsp;Where to keep your funds on the exchange<br>
          &rarr; &nbsp;Choosing the right Mode and Style for your goals<br>
          &rarr; &nbsp;How the 4-layer signal system works<br>
          &rarr; &nbsp;Capital protection: hard floor, drawdown tiers, per-trade caps<br>
          &rarr; &nbsp;Starting your first session
        </p>
      </div>

      <p style="font-size:15px;color:#4b5563;line-height:1.75;margin:0 0 28px;">
        If you have any questions at any point, you can reach us through the support chat inside the app. We're here to help.
      </p>

      <p style="font-size:14px;color:#6b7280;margin:0;line-height:1.6;">
        Welcome aboard,<br>
        <span style="font-weight:700;color:#374151;">The Asymmetric AI Team</span>
      </p>"""
    send_email(
        to,
        "Welcome to Asymmetric AI",
        _letter_base(content, footer="You are in control. The AI only trades with your API keys on your exchange account."),
    )


def email_onboarding_complete(to: str) -> None:
    def _heading(text: str) -> str:
        return (
            f'<p style="font-size:11px;font-weight:800;color:#00957a;text-transform:uppercase;'
            f'letter-spacing:0.1em;margin:32px 0 14px;padding-bottom:10px;'
            f'border-bottom:1px solid #e5e7eb;">{text}</p>'
        )

    def _exchange_block(name: str, steps: str, funds: str) -> str:
        return (
            f'<p style="font-size:13px;font-weight:700;color:#111827;margin:16px 0 4px;">{name}</p>'
            f'<p style="font-size:14px;color:#4b5563;line-height:1.7;margin:0 0 4px;">{steps}</p>'
            f'<p style="font-size:12px;color:#9ca3af;margin:0 0 12px;">Funds &rarr; {funds}</p>'
        )

    def _layer(num: str, title: str, desc: str) -> str:
        return (
            f'<tr>'
            f'<td style="padding:10px 14px 10px 0;vertical-align:top;width:28px;">'
            f'<span style="display:inline-block;width:24px;height:24px;line-height:24px;text-align:center;'
            f'background:#f0fdf9;border:1px solid #6ee7d4;border-radius:50%;'
            f'font-size:11px;font-weight:800;color:#00957a;">{num}</span>'
            f'</td>'
            f'<td style="padding:10px 0;border-bottom:1px solid #f3f4f6;">'
            f'<div style="font-size:14px;font-weight:600;color:#111827;margin-bottom:3px;">{title}</div>'
            f'<div style="font-size:13px;color:#6b7280;line-height:1.6;">{desc}</div>'
            f'</td></tr>'
        )

    def _shield(label: str, value: str) -> str:
        return (
            f'<tr><td style="padding:9px 14px 9px 0;vertical-align:top;width:16px;">'
            f'<span style="color:#00957a;font-size:14px;font-weight:700;">&#x2714;</span></td>'
            f'<td style="padding:9px 0;border-bottom:1px solid #f3f4f6;">'
            f'<span style="font-size:14px;font-weight:600;color:#111827;">{label}</span>'
            f'<span style="font-size:13px;color:#6b7280;"> &mdash; {value}</span>'
            f'</td></tr>'
        )

    def _mode(name: str, size: str, lev: str, best: str, flagship: bool = False) -> str:
        name_cell = (
            f'<span style="font-size:13px;font-weight:700;color:#00957a;">{name}</span>'
            + (
                ' &nbsp;<span style="font-size:9px;background:#f0fdf9;color:#00957a;border:1px solid #6ee7d4;'
                'border-radius:3px;padding:1px 6px;font-weight:800;letter-spacing:0.05em;">FLAGSHIP</span>'
                if flagship else ""
            )
        )
        row_bg = 'background:#f9fffe;' if flagship else ''
        return (
            f'<tr style="{row_bg}">'
            f'<td style="padding:9px 10px 9px 0;border-bottom:1px solid #f3f4f6;">{name_cell}</td>'
            f'<td style="padding:9px 8px;border-bottom:1px solid #f3f4f6;font-size:13px;color:#6b7280;text-align:center;">{size}</td>'
            f'<td style="padding:9px 8px;border-bottom:1px solid #f3f4f6;font-size:13px;color:#6b7280;text-align:center;">{lev}</td>'
            f'<td style="padding:9px 0 9px 8px;border-bottom:1px solid #f3f4f6;font-size:13px;color:#6b7280;">{best}</td>'
            f'</tr>'
        )

    content = f"""
      <p style="font-size:15px;color:#374151;margin:0 0 22px;line-height:1;">Hey!</p>

      <h1 style="font-size:24px;font-weight:800;color:#111827;margin:0 0 16px;line-height:1.25;letter-spacing:-0.02em;">
        Welcome to Asymmetric AI — here&rsquo;s everything you need to know.
      </h1>

      <p style="font-size:15px;color:#4b5563;line-height:1.75;margin:0 0 28px;">
        Your account is set up and ready. This email is your permanent reference guide — save it for whenever you need a quick reminder of how the software works. It covers everything from connecting your exchange to understanding how the AI protects your capital.
      </p>

    {_heading("Minimum Capital")}

    <p style="font-size:15px;color:#4b5563;line-height:1.75;margin:0 0 8px;">
      <span style="color:#111827;font-weight:700;">Minimum: $50 USDT.</span> &nbsp;$100–$500 is the ideal starting range for meaningful results with controlled risk.
    </p>
    <p style="font-size:14px;color:#6b7280;line-height:1.7;margin:0;">
      Keep your funds in your exchange <b style="color:#374151;">trading account</b>, not in a bank or funding wallet. The AI can only access and trade funds in the trading account.
    </p>

    {_heading("Connecting Your Exchange")}

    <p style="font-size:14px;color:#4b5563;line-height:1.7;margin:0 0 4px;">
      Create a <b style="color:#111827;">trade-only API key</b> on your exchange. Always enable <b style="color:#111827;">Read</b> and <b style="color:#111827;">Trade</b> permissions. <b style="color:#dc2626;">Never enable Withdrawal</b> — Asymmetric AI cannot and should not be able to move your funds.
    </p>

    {_exchange_block(
        "Bybit",
        "Profile &rarr; API Management &rarr; Create New Key &rarr; Enable Read + Trade &rarr; Disable Withdrawal",
        "Unified Trading Account &middot; USDT"
    )}
    {_exchange_block(
        "OKX",
        "Profile &rarr; API &rarr; Create V5 API Key &rarr; Enable Read + Trade, set a Passphrase &rarr; Disable Withdrawal",
        "Trading Account &middot; USDT"
    )}
    {_exchange_block(
        "Binance",
        "Profile &rarr; API Management &rarr; Create API &rarr; Enable Reading + Futures Trading &rarr; Disable Withdrawal",
        "Futures Account &middot; USDT"
    )}

    {_heading("How the 4-Layer Signal Works")}

    <p style="font-size:14px;color:#4b5563;line-height:1.75;margin:0 0 14px;">
      Every trade must pass all four layers before it is placed. If any layer fails, the AI waits. This is what keeps entry quality high — and why you'll see fewer trades than you might expect. Quality over quantity.
    </p>

    <table style="width:100%;border-collapse:collapse;margin-bottom:8px;">
      {_layer("1", "Regime", "ADX + ATR confirm that a real, tradeable trend is active — not chop or sideways noise.")}
      {_layer("2", "Direction", "4h EMA21 vs EMA50 determines the trend bias: long or short.")}
      {_layer("3", "Entry", "Price is in a pullback zone with a qualifying candle pattern and confirmed RSI.")}
      {_layer("4", "Momentum", "Volume is real and recent candles confirm the move is continuing.")}
    </table>

    {_heading("Capital Protection")}

    <p style="font-size:14px;color:#4b5563;line-height:1.75;margin:0 0 14px;">
      The risk engine runs continuously in the background — independent of the signal. These protections are always active and cannot be turned off.
    </p>

    <table style="width:100%;border-collapse:collapse;margin-bottom:8px;">
      {_shield("Hard floor", "Engine stops entirely if equity drops 15% from its peak.")}
      {_shield("Drawdown tiers", "Position size shrinks automatically at &minus;4% &rarr; 65%, &minus;7% &rarr; 40%, &minus;10% &rarr; 25%.")}
      {_shield("Per-trade loss cap", "Max 1.5% (Scalp) / 2% (Day Trade) / 3% (Swing) of equity per trade.")}
      {_shield("Non-custodial", "We hold no funds and cannot withdraw. API keys grant trade permission only.")}
    </table>

    {_heading("Modes &amp; Styles")}

    <p style="font-size:14px;color:#4b5563;line-height:1.75;margin:0 0 14px;">
      Choose a <b style="color:#111827;">Mode</b> (how aggressively the AI sizes positions) and a <b style="color:#111827;">Style</b> (which timeframe it trades on). If you're unsure, start with <b style="color:#00957a;">MINI_ASYM</b> mode and <b style="color:#00957a;">DAY_TRADE</b> style — the most popular combination.
    </p>

    <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
      <thead>
        <tr style="border-bottom:2px solid #e5e7eb;">
          <th style="padding:0 10px 10px 0;font-size:11px;color:#9ca3af;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Mode</th>
          <th style="padding:0 8px 10px;font-size:11px;color:#9ca3af;text-align:center;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Size</th>
          <th style="padding:0 8px 10px;font-size:11px;color:#9ca3af;text-align:center;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Lev</th>
          <th style="padding:0 0 10px 8px;font-size:11px;color:#9ca3af;text-align:left;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Best for</th>
        </tr>
      </thead>
      <tbody>
        {_mode("ULTRA_SAFE", "30%", "2×", "First week, getting familiar")}
        {_mode("SAFE", "45%", "3×", "Conservative, steady growth")}
        {_mode("MINI_ASYM", "65%", "6×", "Most popular, balanced performance", flagship=True)}
        {_mode("NORMAL", "60%", "5×", "Balanced exposure")}
        {_mode("AGGRESSIVE", "85%", "8×", "Experienced traders only")}
      </tbody>
    </table>

    <p style="font-size:14px;color:#4b5563;line-height:1.9;margin:0 0 4px;">
      <b style="color:#374151;">SCALP</b> &mdash; 15-minute chart, checks every 15 minutes.<br>
      <b style="color:#00957a;">DAY_TRADE</b> &mdash; 1-hour chart, checks every hour. <span style="color:#00957a;font-weight:600;">Recommended.</span><br>
      <b style="color:#374151;">SWING</b> &mdash; 4-hour chart, checks every 4 hours.
    </p>

    {_heading("Starting the AI")}

    <p style="font-size:15px;color:#4b5563;line-height:1.75;margin:0 0 10px;">
      From the Dashboard: add your API keys in Settings, select a coin, choose your Mode and Style, set a duration (1–30 days), and press <b style="color:#111827;">Start AI</b>. That's it.
    </p>
    <p style="font-size:14px;color:#6b7280;line-height:1.7;margin:0;">
      You'll receive an email every time a trade opens or closes. When your session ends, you can start a new one anytime.
    </p>

    <p style="font-size:15px;color:#4b5563;margin:36px 0 0;line-height:1.6;">
      If you have any questions, open the support chat inside the app and our team will get back to you. We're here to help.
    </p>

    <p style="font-size:14px;color:#6b7280;margin:24px 0 0;line-height:1.6;">
      Welcome aboard,<br>
      <span style="font-weight:700;color:#374151;">The Asymmetric AI Team</span>
    </p>"""

    send_email(
        to,
        "Welcome to Asymmetric AI — your setup guide",
        _letter_base(content, footer="You are in control. The AI only trades with your API keys on your exchange account. We can never withdraw your funds."),
    )


def email_api_key_guide(to: str, exchange: str = "Bybit") -> None:
    bybit_steps = [
        ("Log in to Bybit", "Go to bybit.com and sign in to your account."),
        ("Open API Management", "Click your profile icon (top right) then 'API Management'."),
        ("Create new API key", "Click 'Create New Key' and choose 'System-generated API Keys'."),
        ("Set permissions", "Enable: Read, Trade. Leave EVERYTHING else OFF especially Withdrawal."),
        ("Set IP restriction", "Leave IP restriction blank (or add your Render server IP for extra security)."),
        ("Copy your keys", "Copy the API Key and Secret — the Secret is only shown once. Paste both into Asymmetric AI Settings."),
    ]
    okx_steps = [
        ("Log in to OKX", "Go to okx.com and sign in to your account."),
        ("Open API Management", "Click profile icon then 'API' from the dropdown menu."),
        ("Create new API key", "Click 'Create V5 API Key'."),
        ("Set permissions", "Enable: Read, Trade. Do NOT enable Withdrawal or Transfer."),
        ("Set passphrase", "Create a passphrase — you will need this along with your API Key and Secret."),
        ("Copy your keys", "Copy the API Key, Secret, and Passphrase. Paste all 3 into Asymmetric AI Settings."),
    ]
    steps = bybit_steps if exchange.lower() == "bybit" else okx_steps
    rows = "".join(
        f"""<div style="display:flex;align-items:flex-start;margin-bottom:12px;">
          <div style="min-width:22px;height:22px;background:rgba(0,255,224,0.15);border:1px solid rgba(0,255,224,0.3);
                      border-radius:50%;display:flex;align-items:center;justify-content:center;
                      font-weight:900;font-size:11px;color:#00ffe0;margin-right:10px;flex-shrink:0;">{i+1}</div>
          <div><div style="font-weight:700;color:#f1f5f9;font-size:13px;margin-bottom:2px;">{title}</div>
          <div style="font-size:12px;color:#6b7280;">{desc}</div></div></div>"""
        for i, (title, desc) in enumerate(steps)
    )
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">How to connect your {exchange} API</h2>
    <p style="margin:0 0 18px;font-size:13px;color:#6b7280;">Follow these steps to create a trade-only API key and connect it to Asymmetric AI.</p>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;margin-bottom:16px;">
      {rows}
    </div>
    <div style="background:rgba(220,38,38,0.08);border:1px solid rgba(248,113,113,0.25);
                border-radius:12px;padding:12px 16px;font-size:12px;color:#fecaca;line-height:1.6;">
      <b>Security reminder:</b> Never enable Withdrawal permission on your API key.
      Asymmetric AI only needs Read + Trade access. We never ask for withdrawal permissions.
    </div>"""
    send_email(
        to,
        f"Asymmetric AI — How to connect your {exchange} API key",
        _email_base(content, footer="Your API keys are AES-128 encrypted and stored securely. We never have access to your funds directly."),
    )


def email_first_trade(to: str, symbol: str, side: str, grade: str, equity: float) -> None:
    side_color = "#00ff9d" if side == "LONG" else "#ff5078"
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">Your first trade just fired</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">
      The AI found a high-quality setup and entered your first position.
    </p>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;margin-bottom:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <tr><td style="padding:7px 0;color:#6b7280;width:120px;">Coin</td>
            <td style="padding:7px 0;font-weight:900;color:#f1f5f9;">{symbol}</td></tr>
        <tr><td style="padding:7px 0;color:#6b7280;">Direction</td>
            <td style="padding:7px 0;font-weight:900;color:{side_color};">{side}</td></tr>
        <tr><td style="padding:7px 0;color:#6b7280;">Signal grade</td>
            <td style="padding:7px 0;font-weight:900;color:#00ffe0;">Grade {grade}</td></tr>
        <tr><td style="padding:7px 0;color:#6b7280;">Account equity</td>
            <td style="padding:7px 0;font-weight:900;color:#f1f5f9;">${equity:,.2f}</td></tr>
      </table>
    </div>
    <div style="background:rgba(0,255,224,0.06);border:1px solid rgba(0,255,224,0.18);
                border-radius:12px;padding:14px 16px;font-size:13px;color:#a7f3d0;line-height:1.6;">
      The AI is managing this trade automatically. It will send you another email when the trade closes
      with the full result. You can monitor progress on your dashboard anytime.
    </div>"""
    send_email(
        to,
        f"Your first Asymmetric AI trade is live — {symbol} {side}",
        _email_base(content, footer="The AI manages risk automatically. Your hard floor protects your capital."),
    )


def email_api_key_expired(to: str, symbol: str) -> None:
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">Your API Key Has Expired</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">{symbol}</p>
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(248,113,113,0.3);
                border-radius:14px;padding:18px;margin-bottom:16px;">
      <div style="font-size:15px;font-weight:900;color:#f87171;margin-bottom:8px;">⚠️ AI Stopped — Action Required</div>
      <div style="font-size:13px;color:#fca5a5;line-height:1.7;">
        Bybit rejected your API key because it has expired. The AI has stopped automatically to prevent any issues.
      </div>
    </div>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px;margin-bottom:16px;">
      <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:12px;">What to do next:</div>
      <div style="font-size:12px;color:#9ca3af;line-height:1.8;">
        1. Log in to <b style="color:#f1f5f9;">bybit.com</b><br>
        2. Go to <b style="color:#f1f5f9;">Account → API Management</b><br>
        3. Create a new API key (Read + Trade permissions only)<br>
        4. Open <b style="color:#f1f5f9;">Asymmetric AI → Settings → Exchange</b><br>
        5. Disconnect the old key and connect your new key<br>
        6. Restart the AI
      </div>
    </div>
    <p style="font-size:12px;color:#4b5563;margin-top:12px;">
      No trades were affected — the AI stopped before attempting any new positions.
    </p>"""
    send_email(to, f"Action required — API key expired ({symbol})", _email_base(content))


def email_low_margin(to: str, symbol: str, exchange_id: str, available: float, mode: str) -> None:
    """Sent when the engine skips a trade because available margin is $0 or too low."""
    exchange_id = (exchange_id or "bybit").lower()
    wallet_map = {
        "bybit":   ("Bybit",   "Unified Trading Account",
                    "bybit.com &rarr; Assets &rarr; Unified Trading Account &rarr; transfer USDT from your Spot or Funding wallet"),
        "binance": ("Binance", "Futures Wallet",
                    "binance.com &rarr; Wallet &rarr; Futures &rarr; Transfer USDT from your Spot wallet into your Futures wallet"),
        "okx":     ("OKX",    "Trading Account",
                    "okx.com &rarr; Assets &rarr; Transfer &rarr; move USDT from your Funding Account into your Trading Account"),
    }
    ex_name, wallet_name, transfer_steps = wallet_map.get(exchange_id, wallet_map["bybit"])
    min_rec = {"ULTRA_SAFE": 50, "SAFE": 60, "NORMAL": 80, "MINI_ASYM": 80, "AGGRESSIVE": 100}.get(mode, 80)
    bal_str = "($0.00)" if available == 0 else f"(${available:.2f} USDT)"

    content = f"""
    <div style="font-size:15px;color:#94a3b8;margin-bottom:18px;">Hi,</div>

    <div style="font-size:14px;color:#94a3b8;line-height:1.85;margin-bottom:22px;">
      We wanted to let you know that your AI identified a valid trading signal on <b style="color:#f1f5f9;">{symbol}</b> today,
      but was unable to place the trade. Your available balance in your
      <b style="color:#f1f5f9;">{ex_name} {wallet_name}</b> was too low {bal_str} at the time the signal fired,
      so the trade was skipped to prevent any errors on the exchange.
    </div>

    <div style="font-size:14px;color:#94a3b8;line-height:1.85;margin-bottom:22px;">
      Your AI is still running and will continue scanning for signals — no action is needed on that side.
      However, to make sure the next opportunity isn't missed, you'll want to top up your trading wallet.
    </div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:18px;margin-bottom:18px;">
      <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:12px;">What to do</div>
      <div style="font-size:13px;color:#94a3b8;line-height:1.95;">
        Your funds need to be sitting inside your <b style="color:#f1f5f9;">{wallet_name}</b> on {ex_name} —
        not in your Spot, Funding, or any other wallet. To move them:<br><br>
        <span style="color:#f1f5f9;">{transfer_steps}</span><br><br>
        For <b style="color:#f1f5f9;">{mode}</b> mode, we recommend keeping at least
        <b style="color:#f1f5f9;">${min_rec} USDT</b> available in that wallet at all times.
      </div>
    </div>

    <div style="font-size:13px;color:#94a3b8;line-height:1.85;margin-bottom:28px;">
      Just to clarify — Asymmetric AI never holds your funds and has no ability to move them.
      Everything stays on your exchange at all times. We only place trades on your behalf using your API key.
      <br><br>
      If you have any questions, feel free to reach out through the <b style="color:#f1f5f9;">Support</b> section inside the app.
    </div>

    <div style="font-size:13px;color:#64748b;line-height:1.7;">
      Warm regards,<br>
      <b style="color:#94a3b8;">The Asymmetric AI Team</b>
    </div>"""
    send_email(
        to,
        f"Action needed — your {symbol} trade was skipped due to low balance",
        _email_base(content),
    )


def email_exchange_not_connected_reminder(to: str) -> None:
    """Sent every 3-4 days to users who have never connected an exchange."""
    content = """
    <div style="font-size:15px;color:#94a3b8;margin-bottom:18px;">Hi,</div>

    <div style="font-size:14px;color:#94a3b8;line-height:1.85;margin-bottom:22px;">
      We noticed your Asymmetric AI account is fully set up, but your exchange hasn't been connected yet.
      Until you link your exchange, the AI has nowhere to trade and will stay inactive — so you won't be
      getting any signals or positions placed on your behalf.
    </div>

    <div style="font-size:14px;color:#94a3b8;line-height:1.85;margin-bottom:22px;">
      The good news is it only takes a couple of minutes to get sorted.
    </div>

    <div style="background:#0b1120;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:18px;margin-bottom:18px;">
      <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:12px;">How to connect your exchange</div>
      <div style="font-size:13px;color:#94a3b8;line-height:2.0;">
        1. Log in to <b style="color:#f1f5f9;">Asymmetric AI</b><br>
        2. Head to <b style="color:#f1f5f9;">Settings &rarr; Exchange</b><br>
        3. Choose your exchange — we support <b style="color:#f1f5f9;">Bybit, Binance, and OKX</b>
           (Bybit is our top recommendation for speed and fees)<br>
        4. Enter your API Key and Secret — enable <b style="color:#f1f5f9;">Read and Trade</b> permissions,
           but do <b style="color:#f87171;">not</b> enable withdrawal<br>
        5. Hit <b style="color:#f1f5f9;">Save</b> — your balance will appear on the Dashboard within a few seconds
      </div>
    </div>

    <div style="font-size:14px;color:#94a3b8;line-height:1.85;margin-bottom:28px;">
      Once connected, you're ready to start your first AI session whenever you like.
      <br><br>
      Not sure how to create an API key? No problem — open the <b style="color:#f1f5f9;">Support</b> section
      inside the app and our team will walk you through it step by step.
    </div>

    <div style="font-size:13px;color:#64748b;line-height:1.7;">
      Warm regards,<br>
      <b style="color:#94a3b8;">The Asymmetric AI Team</b>
    </div>"""
    send_email(
        to,
        "Your Asymmetric AI account is ready — one step left",
        _email_base(content),
    )


def email_exchange_disconnected_open_trade(to: str, symbol: str, trade_count: int) -> None:
    content = f"""
    <h2 style="margin:0 0 6px;font-size:20px;font-weight:900;color:#f1f5f9;">Exchange Disconnected</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">{symbol}</p>
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(252,211,77,0.3);
                border-radius:14px;padding:18px;margin-bottom:16px;">
      <div style="font-size:15px;font-weight:900;color:#fbbf24;margin-bottom:8px;">
        ⚠️ You have {trade_count} open trade{'s' if trade_count != 1 else ''} — reconnect immediately
      </div>
      <div style="font-size:13px;color:#fde68a;line-height:1.7;">
        Your exchange was disconnected while a trade is still active on Bybit.
        The AI is paused and cannot manage your position until you reconnect.
      </div>
    </div>
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px;margin-bottom:16px;">
      <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:12px;">What to do:</div>
      <div style="font-size:12px;color:#9ca3af;line-height:1.8;">
        1. Open <b style="color:#f1f5f9;">Asymmetric AI → Settings → Exchange</b><br>
        2. Reconnect your exchange<br>
        3. The AI will automatically resume managing your position<br><br>
        <span style="color:#f87171;">If you do not reconnect, your open trade will not have a stop-loss managed by the AI.
        You may need to close it manually on Bybit.</span>
      </div>
    </div>"""
    send_email(to, f"Urgent — Exchange disconnected with open trade ({symbol})", _email_base(content))
