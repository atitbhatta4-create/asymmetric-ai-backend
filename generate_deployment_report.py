"""
generate_deployment_report.py
Generates a PDF deployment verification report for the 2026-06-19 push.
Run locally: python3 generate_deployment_report.py
"""
from fpdf import FPDF
from datetime import datetime

OUT = "deployment_verification_2026-06-19.pdf"

PASS_COLOR  = (0, 153, 76)
FAIL_COLOR  = (204, 0, 0)
WARN_COLOR  = (204, 122, 0)
INFO_COLOR  = (0, 102, 204)
DARK        = (30, 30, 30)
LIGHT_GRAY  = (245, 245, 245)
MID_GRAY    = (200, 200, 200)
WHITE       = (255, 255, 255)
BG          = (18, 18, 28)
GOLD        = (212, 175, 55)


class PDF(FPDF):
    def header(self):
        self.set_fill_color(*BG)
        self.rect(0, 0, 297, 22, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*GOLD)
        self.set_xy(10, 5)
        self.cell(0, 12, "ASYMMETRIC AI  |  Deployment Verification Report  |  2026-06-19", ln=True)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Page {self.page_no()}", align="C")

    def section_title(self, title, color=BG):
        self.ln(4)
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 9, f"  {title}", ln=True, fill=True)
        self.set_text_color(*DARK)
        self.ln(2)

    def status_row(self, label, value, status="INFO"):
        colors = {"PASS": PASS_COLOR, "FAIL": FAIL_COLOR, "WARN": WARN_COLOR, "INFO": INFO_COLOR, "PENDING": WARN_COLOR}
        badges = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "INFO": "INFO", "PENDING": "PENDING"}
        c = colors.get(status, INFO_COLOR)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        self.set_fill_color(*LIGHT_GRAY)
        self.cell(70, 7, f"  {label}", border="B", fill=True)
        self.cell(160, 7, f"  {value}", border="B", fill=False)
        self.set_fill_color(*c)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        self.cell(27, 7, badges[status], border=0, fill=True, align="C", ln=True)
        self.set_text_color(*DARK)

    def info_line(self, text, indent=4):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, " " * indent + text, ln=True)
        self.set_text_color(*DARK)

    def highlight_box(self, lines, color=LIGHT_GRAY, border_color=MID_GRAY):
        self.set_fill_color(*color)
        self.set_draw_color(*border_color)
        x, y = self.get_x(), self.get_y()
        h = len(lines) * 6 + 4
        self.rect(x + 5, y, 247, h, "FD")
        for line in lines:
            self.set_xy(x + 8, y + 2)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(40, 40, 40)
            self.cell(0, 6, line, ln=True)
            y += 6
        self.set_xy(x, y + 4)
        self.set_text_color(*DARK)


pdf = PDF(orientation="L", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_left_margin(10)
pdf.set_right_margin(10)

#  Hero block 
pdf.ln(4)
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(*BG)
pdf.cell(0, 12, "Deployment Verification  -  June 19, 2026", ln=True, align="C")
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 6, "R1 Hybrid Floor  +  Trade Outcome Logging  +  Grade B tp_mult fix  +  Adaptive Pipeline", ln=True, align="C")
pdf.ln(4)

#  What was deployed 
pdf.section_title("WHAT WAS DEPLOYED IN THIS PUSH", color=(30, 30, 80))
changes = [
    ("R1 Hybrid Floor  (Scenario C)", "Flat 15% below peak until equity reaches 2x start capital, then ratchet tightens to 10%"),
    ("Grade B T1  tp_mult fix",       "Changed 4 locations: tp_mult 0.80 -> 1.00  (restores correct 1:2 R:R for Grade B T1 exits)"),
    ("trade_outcomes table",          "31-field analytics table - logs every closed trade, never blocks engine"),
    ("RSI debug print removed",        "Cleaned production logs (indicators.py)"),
    ("adaptive_pipeline.py",          "Walk-forward validation engine - 10 coins, MINI_ASYM DAY_TRADE, monthly auto-run"),
    ("routes_pipeline.py",            "Admin-only API: /pipeline/run, /status, /results, /report, /report/pdf, /history"),
    ("Monthly scheduler",             "Auto-runs pipeline on 1st of each month; daily degradation check vs live trades"),
    ("fpdf2 added to requirements",   "Enables PDF generation on Render server"),
]
for name, desc in changes:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*DARK)
    pdf.cell(70, 6, f"  {name}", ln=False)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, desc, ln=True)
pdf.ln(3)

#  Check 1 
pdf.section_title("CHECK 1 - trade_outcomes Table Population", color=(0, 80, 0))
pdf.status_row("Table created on Render", "Confirmed - init_outcome_log_table() runs at server startup", "PASS")
pdf.status_row("31-field row validation", "Pending - no closed trades yet since deployment", "PENDING")
pdf.status_row("Field match vs dashboard", "Pending - re-run verify_deployment.py after next trade closes", "PENDING")
pdf.ln(2)
pdf.info_line("Re-run command (Render shell):  VERIFY_EMAIL=mes29571@gmail.com python verify_deployment.py")
pdf.info_line("Check 1 will auto-PASS once first trade closes and all 31 fields populate correctly.")

#  Check 2 
pdf.section_title("CHECK 2 - Grade B T1  R:R = 2.0 Exactly", color=(0, 80, 0))
pdf.status_row("tp_mult fix deployed", "4 locations changed from 0.80 to 1.00 in main.py", "PASS")
pdf.status_row("Live R:R confirmation", "Pending - no Grade B T1 trade has closed since deployment", "PENDING")
pdf.status_row("Formula used by script", "R:R = tp_pct / sl_pct  (expects 2.00 +/- 0.05 tolerance)", "INFO")
pdf.ln(2)
pdf.info_line("Grade B T1 uses TP = 2 x SL distance. Before fix: tp_mult=0.80 compressed TP to 1.6x SL (broken).")
pdf.info_line("After fix: tp_mult=1.00 -> TP = 2.0x SL (correct). Will auto-confirm on next Grade B T1 close.")

#  Check 3 
pdf.section_title("CHECK 3 - Hard Floor Formula Confirmed  [VERIFIED LIVE]", color=(0, 100, 0))

pdf.set_fill_color(230, 255, 230)
pdf.set_draw_color(*PASS_COLOR)
pdf.rect(10, pdf.get_y(), 257, 28, "FD")
pdf.ln(2)

vals = [
    ("starting_capital",         "$20.49"),
    ("peak_equity",              "$45.72  (2.23x starting capital)"),
    ("2x threshold",             "$40.98  ->  RATCHET IS ACTIVE"),
    ("Expected floor (ratchet)", "max($20.49 x 0.85, $45.72 x 0.90) = max($17.42, $41.15) = $41.15"),
    ("DB floor_equity",          "$41.15  ->  EXACT MATCH"),
]
for k, v in vals:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*DARK)
    pdf.set_x(14)
    pdf.cell(58, 5, k, ln=False)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, v, ln=True)
pdf.ln(6)

pdf.status_row("Ratchet formula", "_compute_floor(peak, start_capital) -> max(start*0.85, peak*0.90) when peak >= 2x start", "PASS")
pdf.status_row("Ratchet activation", "Real account crossed 2x threshold - ratchet is LIVE on actual trading account", "PASS")
pdf.status_row("Floor value in DB", "$41.15 vs expected $41.15 - zero delta", "PASS")
pdf.status_row("Balance endpoint", "Uses max(peak*0.85, stored_floor) - correct because stored_floor is already $41.15", "PASS")
pdf.ln(2)
pdf.info_line("Scenario C (Hybrid 2x): protects profits aggressively once compounded gains are proven.")
pdf.info_line("If equity pulls back to $41.15, system stops. Current equity $45.72 has $4.57 breathing room.")

#  Check 4 
pdf.section_title("CHECK 4 - Sentry New Errors Since Deployment  [MANUAL STEP]", color=(100, 60, 0))
pdf.status_row("Sentry check", "Cannot be automated - requires dashboard login. Instructions below.", "PENDING")
pdf.ln(2)

steps = [
    "1. Go to sentry.io and log in",
    "2. Left sidebar: Projects -> select your Asymmetric AI / FastAPI project",
    "3. Click 'Issues' in left menu",
    "4. Set filter: First Seen = 2026-06-19 (today) | Sort by: First Seen",
    "5. Scan for ANY of these specific new errors:",
    "     - KeyError: 'start_capital'  (runner init broke)",
    "     - AttributeError on _compute_floor  (floor function crashed)",
    "     - TypeError in log_trade_outcome  (wrong field types on trade close)",
    "     - IntegrityError on trade_outcomes INSERT  (duplicate or schema mismatch)",
    "     - ImportError mentioning log_trade_outcome or init_outcome_log_table",
    "     - Any fpdf2 or adaptive_pipeline import error",
    "6. If list is EMPTY or contains only pre-existing errors -> CHECK 4 PASS",
    "7. If new error found -> paste title + traceback to Claude before continuing",
]
for s in steps:
    pdf.info_line(s, indent=2)

#  Overall summary 
pdf.add_page()
pdf.ln(4)
pdf.section_title("DEPLOYMENT VERIFICATION SUMMARY", color=BG)

summary = [
    ("Check 1", "trade_outcomes rows",          "PENDING",  "Re-run after next closed trade"),
    ("Check 2", "Grade B T1 R:R = 2.0",        "PENDING",  "Re-run after next Grade B T1 close"),
    ("Check 3", "Hard floor = Hybrid formula",  "PASS",     "VERIFIED LIVE: floor $41.15, ratchet active"),
    ("Check 4", "Sentry - no new errors",       "PENDING",  "Manual: check sentry.io -> Issues, First Seen today"),
]

pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(*BG)
pdf.set_text_color(*WHITE)
for col, w in [("Check", 20), ("Name", 65), ("Status", 30), ("Notes", 140)]:
    pdf.cell(w, 8, f"  {col}", fill=True, border=0)
pdf.ln()

for i, (check, name, status, note) in enumerate(summary):
    fill = LIGHT_GRAY if i % 2 == 0 else WHITE
    sc   = PASS_COLOR if status == "PASS" else WARN_COLOR
    pdf.set_fill_color(*fill)
    pdf.set_text_color(*DARK)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(20, 7, f"  {check}", fill=True, border="B")
    pdf.cell(65, 7, f"  {name}", fill=True, border="B")
    pdf.set_fill_color(*sc)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(30, 7, f"  {status}", fill=True, border="B")
    pdf.set_fill_color(*fill)
    pdf.set_text_color(*DARK)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(140, 7, f"  {note}", fill=True, border="B", ln=True)

pdf.ln(6)
pdf.info_line("NEXT STEPS:")
pdf.info_line("  1. Check Sentry now (Check 4) - if clean, deployment is safe to trust")
pdf.info_line("  2. Checks 1 and 2 resolve automatically after next trade closes")
pdf.info_line("  3. After all 4 checks confirmed: run adaptive pipeline Phase 1")
pdf.info_line("     Render shell: python3 -c \"from adaptive_pipeline import start_pipeline; print(start_pipeline('manual'))\"")
pdf.info_line("  4. Pipeline sends Telegram notification when complete (~15-30 min for 10 coins)")
pdf.info_line("  5. Download Phase 1 PDF from: /pipeline/report/{run_id}/pdf")
pdf.ln(4)

#  R1 floor formula reference 
pdf.section_title("REFERENCE - R1 Hybrid Floor Formula (Scenario C)", color=(30, 30, 80))
formula_lines = [
    "def _compute_floor(peak, start_capital):",
    "    if start_capital > 0 and peak >= 2.0 * start_capital:",
    "        return round(max(start_capital * 0.85, peak * 0.90), 2)   # Ratchet: tighten to 10% from peak",
    "    return round(peak * 0.85, 2)                                   # Flat: 15% from peak (proving phase)",
]
pdf.set_font("Courier", "", 9)
pdf.set_fill_color(240, 240, 255)
pdf.set_draw_color(180, 180, 220)
pdf.rect(10, pdf.get_y(), 257, len(formula_lines)*6+4, "FD")
y0 = pdf.get_y() + 2
for j, line in enumerate(formula_lines):
    pdf.set_xy(14, y0 + j*6)
    pdf.set_text_color(20, 20, 100)
    pdf.cell(0, 6, line, ln=False)
pdf.set_xy(10, y0 + len(formula_lines)*6 + 2)
pdf.ln(3)

pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(60, 60, 60)
scenarios = [
    "Scenario A  (Flat 15%):     floor = peak x 0.85 always - simple, never tightens",
    "Scenario B  (Ratchet only): floor = max(start x 0.85, peak x 0.90) always - aggressive from day 1",
    "Scenario C  (Hybrid 2x):    Scenario A until 2x growth, then Scenario B - CHOSEN (deployed)",
    "",
    "Why Scenario C: flat floor while proving edge on small account; ratchet kicks in only after",
    "doubling capital, locking in compounded gains without over-tightening early volatile runs.",
]
for s in scenarios:
    pdf.info_line(s, indent=2)

pdf.output(OUT)
print(f"PDF saved: {OUT}")
