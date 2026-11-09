"""
verify_deployment.py — Run on Render shell (! python verify_deployment.py)
Checks all 4 deployment validations after the 2026-06-19 push.

Requires DATABASE_URL env var (automatically set on Render).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("ASYM_DB_PATH", "asymmetric_demo.db")

from database import db_conn

EMAIL = os.getenv("VERIFY_EMAIL", "mes29571@gmail.com")
PASS_MARK  = "  [PASS]"
FAIL_MARK  = "  [FAIL]"
INFO_MARK  = "  [INFO]"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── Check 1 — trade_outcomes rows ─────────────────────────────────────────────
section("CHECK 1 — trade_outcomes table population")

with db_conn() as conn:
    cur = conn.cursor()

    # Confirm table exists
    try:
        cur.execute("SELECT COUNT(*) AS n FROM trade_outcomes WHERE email = %s", (EMAIL,))
        total = cur.fetchone()["n"]
        print(f"{INFO_MARK} trade_outcomes rows for {EMAIL}: {total}")
    except Exception as e:
        print(f"{FAIL_MARK} Table missing or query failed: {e}")
        total = 0

    if total == 0:
        print(f"{INFO_MARK} No closed trades yet since deployment. Re-run after first trade closes.")
    else:
        # Print last 5 rows, all 31 fields
        cur.execute(
            "SELECT * FROM trade_outcomes WHERE email = %s ORDER BY id DESC LIMIT 5",
            (EMAIL,),
        )
        rows = cur.fetchall()
        REQUIRED_FIELDS = [
            "id","email","symbol","side","mode","trade_style","grade","label","signal",
            "score","regime","outcome","entry_price","exit_price","sl_pct","tp_pct",
            "atr_pct","effective_size","leverage","pnl_pct","pnl_value","equity_before",
            "equity_after","peak_equity","t18_scale1","t18_scale2","trailing_used",
            "vol_adj","dd_adj","candles_open","opened_at","closed_at",
            "consecutive_wins_before",
        ]
        for i, row in enumerate(rows):
            print(f"\n  Trade #{i+1} (id={row.get('id')}):")
            missing = []
            null_important = []
            for field in REQUIRED_FIELDS:
                val = row.get(field)
                if field not in row:
                    missing.append(field)
                elif val is None and field in ("outcome","entry_price","exit_price","sl_pct","tp_pct","pnl_pct","equity_before","equity_after"):
                    null_important.append(field)
                else:
                    print(f"    {field:30s} = {val}")
            if missing:
                print(f"  {FAIL_MARK} Missing fields: {missing}")
            elif null_important:
                print(f"  {FAIL_MARK} Null on critical fields: {null_important}")
            else:
                print(f"  {PASS_MARK} All 31 fields present")

# ── Check 2 — Grade B T1 R:R verification ─────────────────────────────────────
section("CHECK 2 — Grade B T1 R:R = 2.0 exactly")

with db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT id, entry_price, exit_price, sl_pct, tp_pct, outcome, side "
        "FROM trade_outcomes "
        "WHERE email=%s AND grade='B' AND label='T1' "
        "ORDER BY id DESC LIMIT 10",
        (EMAIL,),
    )
    t1_trades = cur.fetchall()

if not t1_trades:
    print(f"{INFO_MARK} No Grade B T1 trades found yet. Re-run after first Grade B T1 closes.")
else:
    all_pass = True
    for t in t1_trades:
        entry    = float(t["entry_price"] or 0)
        exit_p   = float(t["exit_price"]  or 0)
        sl_pct   = float(t["sl_pct"]      or 0)
        tp_pct   = float(t["tp_pct"]      or 0)
        side     = t["side"]
        outcome  = t["outcome"]

        # R:R from stored SL/TP percentages (most reliable — actual params used)
        if sl_pct > 0 and tp_pct > 0:
            rr = round(tp_pct / sl_pct, 4)
            status = PASS_MARK if abs(rr - 2.0) < 0.05 else FAIL_MARK
            if abs(rr - 2.0) >= 0.05:
                all_pass = False
            print(f"  Trade id={t['id']} | outcome={outcome} | TP={tp_pct*100:.3f}% / SL={sl_pct*100:.3f}% | R:R={rr:.3f} {status}")
        else:
            print(f"  Trade id={t['id']} | sl_pct or tp_pct is 0 — cannot compute R:R {FAIL_MARK}")
            all_pass = False

    # Also verify via price if available
    for t in t1_trades:
        entry  = float(t["entry_price"] or 0)
        exit_p = float(t["exit_price"]  or 0)
        sl_pct = float(t["sl_pct"]      or 0)
        side   = t["side"]
        if entry > 0 and exit_p > 0 and sl_pct > 0:
            if side == "LONG":
                reward = (exit_p - entry) / entry
                risk   = sl_pct
            else:
                reward = (entry - exit_p) / entry
                risk   = sl_pct
            if risk > 0:
                price_rr = round(reward / risk, 4)
                print(f"  Trade id={t['id']} | price-based R:R={price_rr:.3f} (entry=${entry:.4f} exit=${exit_p:.4f})")

    if all_pass:
        print(f"\n{PASS_MARK} All Grade B T1 trades show R:R ≈ 2.0 — tp_mult=1.00 fix confirmed.")

# ── Check 3 — Hard floor formula ──────────────────────────────────────────────
section("CHECK 3 — Hard floor = peak × 0.85 (below 2× threshold)")

with db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT peak_equity, floor_equity, starting_capital "
        "FROM user_state WHERE email = %s",
        (EMAIL,),
    )
    row = cur.fetchone()

if not row:
    print(f"{FAIL_MARK} No user_state row for {EMAIL}")
else:
    peak    = float(row["peak_equity"]      or 0)
    floor   = float(row["floor_equity"]     or 0)
    start   = float(row["starting_capital"] or 0)
    expected_flat   = round(peak * 0.85, 2)
    threshold_2x    = start * 2.0 if start > 0 else None
    ratchet_active  = threshold_2x is not None and peak >= threshold_2x

    print(f"{INFO_MARK} starting_capital = ${start:.2f}")
    print(f"{INFO_MARK} peak_equity      = ${peak:.2f}")
    print(f"{INFO_MARK} floor_equity(DB) = ${floor:.2f}")
    print(f"{INFO_MARK} expected flat floor (peak×0.85) = ${expected_flat:.2f}")
    if threshold_2x:
        print(f"{INFO_MARK} 2× threshold = ${threshold_2x:.2f}  |  ratchet active = {ratchet_active}")

    if ratchet_active:
        expected_ratchet = round(max(start * 0.85, peak * 0.90), 2)
        delta = abs(floor - expected_ratchet)
        marker = PASS_MARK if delta < 0.02 else FAIL_MARK
        print(f"  Ratchet formula expected: ${expected_ratchet:.2f}  |  DB has: ${floor:.2f}  {marker}")
        if delta >= 0.02:
            print(f"  {FAIL_MARK} Floor mismatch by ${delta:.2f} — check _compute_floor implementation")
    else:
        delta = abs(floor - expected_flat)
        marker = PASS_MARK if delta < 0.02 else FAIL_MARK
        print(f"  Flat formula expected: ${expected_flat:.2f}  |  DB has: ${floor:.2f}  {marker}")
        if delta >= 0.02:
            print(f"  {FAIL_MARK} Floor mismatch by ${delta:.2f} — may be stale from before deployment")
        else:
            print(f"  Ratchet NOT active (peak ${peak:.2f} < 2× start ${threshold_2x:.2f}) — correct, using flat 15%")

# ── Check 4 — Sentry ──────────────────────────────────────────────────────────
section("CHECK 4 — Sentry (manual step required)")

print(f"""
  Sentry cannot be queried from this script — check manually:

  1. Open Sentry dashboard → Issues → Sort by "First Seen"
  2. Filter: First Seen = today (2026-06-19) or since deployment
  3. Look for any NEW error type not seen before, especially:
       - KeyError on 'start_capital' (runner init)
       - AttributeError on _compute_floor
       - TypeError in log_trade_outcome (wrong field types)
       - IntegrityError on trade_outcomes INSERT
       - Any import error (log_trade_outcome, init_outcome_log_table)

  If none of the above appear: CHECK 4 PASS
  If any appear: stop and fix before adding changes.
""")

# ── Summary ────────────────────────────────────────────────────────────────────
section("SUMMARY")
print("""
  Check 1 (trade_outcomes rows)   — see output above
  Check 2 (Grade B T1 R:R = 2.0) — see output above
  Check 3 (floor = peak × 0.85)  — see output above
  Check 4 (Sentry new errors)     — MANUAL: check Sentry dashboard

  Re-run this script after each new closed trade to accumulate more data.
  Run with:  VERIFY_EMAIL=your@email.com python verify_deployment.py
""")
