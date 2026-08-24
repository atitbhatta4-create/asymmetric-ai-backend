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

def _email_base(content: str, footer: str = "Automated trading involves risk. Never trade more than you can afford to lose.") -> str:
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
        {footer}
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
    renewal_cta = ""
    if reason == "DURATION_END":
        renewal_cta = """
    <a href="https://asymmetric-ai.vercel.app"
       style="display:block;margin-top:16px;padding:13px 16px;background:linear-gradient(90deg,#00ff9d,#00ffe0);
              color:#021018;font-weight:900;font-size:14px;text-align:center;border-radius:14px;
              text-decoration:none;">
      Start a new session &rarr;
    </a>"""

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
    </p>{renewal_cta}"""
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


# ── Onboarding emails ─────────────────────────────────────────────────────────

def email_welcome(to: str) -> None:
    def _step(n: str, title: str, body: str) -> str:
        return f"""
        <div style="display:flex;align-items:flex-start;margin-bottom:16px;">
          <div style="min-width:28px;height:28px;background:#00ffe0;border-radius:50%;display:flex;align-items:center;
                      justify-content:center;font-weight:900;font-size:13px;color:#050814;margin-right:12px;flex-shrink:0;">{n}</div>
          <div>
            <div style="font-weight:700;color:#f1f5f9;margin-bottom:4px;">{title}</div>
            <div style="font-size:13px;color:#94a3b8;line-height:1.55;">{body}</div>
          </div>
        </div>"""

    content = f"""
    <h2 style="margin:0 0 4px;font-size:22px;font-weight:900;color:#f1f5f9;">Welcome to Asymmetric AI</h2>
    <p style="margin:0 0 20px;font-size:13px;color:#6b7280;">Your account is ready. Read this guide before you start — it will save you time.</p>

    <!-- MINIMUM CAPITAL -->
    <div style="background:rgba(0,255,224,0.07);border:1px solid rgba(0,255,224,0.22);
                border-radius:12px;padding:13px 16px;font-size:13px;color:#a7f3d0;line-height:1.6;margin-bottom:18px;">
      <b>Minimum recommended capital: $50 USDT.</b><br>
      Smaller amounts work but fees will eat a larger percentage of each trade. $100–$500 is ideal to start.
    </div>

    <!-- SETUP STEPS -->
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:18px 16px;margin-bottom:18px;">
      {_step("1", "Connect your Bybit or OKX API keys",
        "Go to <b>Exchange</b> in the sidebar. Create a trade-only API key on Bybit or OKX. "
        "Enable: <b>Read ✓</b> and <b>Trade ✓</b>. <b>Never enable Withdrawal</b> — we do not need it and it is a security risk.")}
      {_step("2", "Keep your funds in the right place",
        "On Bybit: funds must be in your <b>Unified Trading Account</b> as USDT. "
        "On OKX: funds must be in your <b>Trading Account</b> as USDT. "
        "The AI reads your USDT balance and sizes positions from there.")}
      {_step("3", "Choose a coin, mode, and style",
        "Start with <b>BTCUSDT</b> or <b>ETHUSDT</b> (most liquid, cleaner signals). "
        "Recommended mode: <b>SAFE</b> or <b>MINI_ASYM</b>. Recommended style: <b>DAY_TRADE</b> (1h candles, checks every hour). "
        "Avoid AGGRESSIVE until you are comfortable.")}
      {_step("4", "Set a duration and start",
        "Choose how many days you want the AI to run (1–30 days). It stops automatically at the end. "
        "You can start a new session anytime. Press <b>Start AI</b> on the dashboard.")}
    </div>

    <!-- HOW IT WORKS -->
    <div style="margin-bottom:14px;">
      <div style="font-size:14px;font-weight:800;color:#f1f5f9;margin-bottom:10px;">Why does it trade less than expected?</div>
      <div style="font-size:13px;color:#94a3b8;line-height:1.65;">
        The engine only enters a trade when <b>all 4 signal layers align</b> at the same time: trend strength (ADX),
        direction (4h EMA), entry timing (pullback zone + candle pattern), and momentum (volume + candle direction).
        If any layer is missing, it waits. <b>Quality over quantity</b> — fewer but higher-probability trades protect your capital better than trading every hour.
      </div>
    </div>

    <!-- RISK PROTECTION -->
    <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;margin-bottom:18px;">
      <div style="font-size:14px;font-weight:800;color:#f1f5f9;margin-bottom:10px;">How it protects your money</div>
      <div style="font-size:13px;color:#94a3b8;line-height:1.7;">
        • <b>Hard floor:</b> if your account drops 15% from its peak, the engine stops completely.<br>
        • <b>Drawdown tiers:</b> position size automatically shrinks at −4%, −7%, and −10% drawdown.<br>
        • <b>Per-trade cap:</b> max loss per trade is 1.5–3% of your equity depending on your style.<br>
        • <b>Non-custodial:</b> the AI only has trade permission — it can never withdraw your funds. You can disconnect or withdraw from your exchange at any time.
      </div>
    </div>

    <!-- MODES + STYLES -->
    <div style="margin-bottom:18px;">
      <div style="font-size:14px;font-weight:800;color:#f1f5f9;margin-bottom:10px;">Modes and Styles — quick reference</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;color:#94a3b8;">
        <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
          <td style="padding:7px 4px;font-weight:700;color:#f1f5f9;">Mode</td>
          <td style="padding:7px 4px;">Size</td><td style="padding:7px 4px;">Leverage</td><td style="padding:7px 4px;">Best for</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
          <td style="padding:7px 4px;color:#00ffe0;">ULTRA_SAFE</td><td style="padding:7px 4px;">30%</td><td style="padding:7px 4px;">2×</td><td style="padding:7px 4px;">First week, testing</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
          <td style="padding:7px 4px;color:#00ffe0;">SAFE</td><td style="padding:7px 4px;">45%</td><td style="padding:7px 4px;">3×</td><td style="padding:7px 4px;">Conservative, steady</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
          <td style="padding:7px 4px;color:#00ffe0;">MINI_ASYM</td><td style="padding:7px 4px;">65%</td><td style="padding:7px 4px;">6×</td><td style="padding:7px 4px;">Flagship, balanced</td>
        </tr>
        <tr>
          <td style="padding:7px 4px;color:#00ffe0;">AGGRESSIVE</td><td style="padding:7px 4px;">85%</td><td style="padding:7px 4px;">8×</td><td style="padding:7px 4px;">Experienced only</td>
        </tr>
      </table>
      <div style="margin-top:10px;font-size:12px;color:#94a3b8;">
        <b>Styles:</b> SCALP (15m charts, checks every 15 min) · DAY_TRADE (1h, every hour) · SWING (4h, every 4 hours)
      </div>
    </div>"""

    send_email(
        to,
        "Welcome to Asymmetric AI — Complete setup guide",
        _email_base(content, footer="You are in control. The AI only trades with your API keys on your exchange account. We can never withdraw your funds."),
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
