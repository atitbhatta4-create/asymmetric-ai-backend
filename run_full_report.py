"""
run_full_report.py — Backtest + Optimizer + PDF for top 10 coins.
Uses Binance public API (1,000 candles/page) instead of OKX (100/page) — 10x faster.
"""
from __future__ import annotations
import itertools, os, sys, time, statistics, math, json
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from config import HIGHER_TF_MAP, TRADE_STYLE_PARAMS, DUBAI_TZ
from indicators import _compute_signal_layers, MODE_SIGNAL_PARAMS
import indicators as _ind_mod   # needed for monkey-patching _session_quality
from optimizer import OPT_GRID, MIN_TRADES, _sim_one, WARMUP_CANDLES as OPT_WARMUP

# ── Historical session quality patch ───────────────────────────────────────────
# _compute_signal_layers internally calls _session_quality(trade_style) which
# uses now_dubai().hour (current real time). During backtesting we must use the
# historical candle's timestamp instead, otherwise every signal gets penalised
# by whatever session the script happens to run in.
_BACKTEST_TS_MS: List[int] = [0]   # mutable cell; set per-candle before signal call

def _session_quality_historical(trade_style: str) -> tuple:
    if trade_style == "SWING":
        return 1.0, "swing"
    ts = _BACKTEST_TS_MS[0]
    hour = datetime.fromtimestamp(ts / 1000, tz=DUBAI_TZ).hour if ts else 14  # default noon
    if 2 <= hour < 6:
        if trade_style == "SCALP":
            return 0.0, f"scalp-blocked {hour:02d}:xx"
        return 0.72, f"dead-zone {hour:02d}:xx"
    if 6  <= hour < 12: return 0.85, f"Asia {hour:02d}:xx"
    if 12 <= hour < 16: return 0.95, f"London {hour:02d}:xx"
    if 16 <= hour < 21: return 1.00, f"London+NY {hour:02d}:xx"
    if 21 <= hour <= 23: return 0.92, f"NY {hour:02d}:xx"
    return 0.80, f"NY-close {hour:02d}:xx"

# ── Config ─────────────────────────────────────────────────────────────────────
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","DOTUSDT",
]
COIN_LABELS = {
    "BTCUSDT":"Bitcoin (BTC)","ETHUSDT":"Ethereum (ETH)","BNBUSDT":"BNB (BNB)",
    "XRPUSDT":"XRP (XRP)","SOLUSDT":"Solana (SOL)","ADAUSDT":"Cardano (ADA)",
    "DOGEUSDT":"Dogecoin (DOGE)","AVAXUSDT":"Avalanche (AVAX)",
    "LINKUSDT":"Chainlink (LINK)","DOTUSDT":"Polkadot (DOT)",
}
MODE         = "MINI_ASYM"
STYLE        = "DAY_TRADE"
TF           = "1h"
HIGHER_TF    = "4h"
EXCHANGE     = "bybit"
START_EQUITY = 1000.0
DATE_FROM    = "2020-01-01"
DATE_TO      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
START_MS     = int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp()*1000)
END_MS       = int(datetime.now(timezone.utc).timestamp()*1000)
PDF_OUT      = os.path.join(os.path.dirname(__file__),"asymmetric_ai_backtest_report.pdf")
CACHE_FILE   = os.path.join(os.path.dirname(__file__),"candle_cache.json")

ATR_BASELINE = {"15m":0.0040,"1h":0.0090,"4h":0.0180,"1d":0.0350}
MODE_PRESETS = {
    "ULTRA_SAFE":{"size":0.30,"leverage":2},
    "SAFE":{"size":0.45,"leverage":3},
    "NORMAL":{"size":0.60,"leverage":5},
    "MINI_ASYM":{"size":0.65,"leverage":6},
    "AGGRESSIVE":{"size":0.85,"leverage":8},
}
WARMUP = 240
TF_BN  = {"1h":"1h","4h":"4h","15m":"15m"}   # Binance interval strings
FEE    = {"maker":0.00020,"taker":0.00055,"slip":0.00025}   # Bybit model

# ── Binance candle fetch (1000/page, much faster than OKX 100/page) ────────────
def fetch_binance(symbol:str, interval:str, start_ms:int, end_ms:int) -> List[Dict]:
    url = "https://api.binance.com/api/v3/klines"
    candles = []
    current = start_ms
    pages = 0
    print(f"    Fetching {symbol} {interval} from Binance...", end="", flush=True)
    while current < end_ms:
        try:
            r = httpx.get(url, params={
                "symbol":symbol,"interval":interval,
                "startTime":current,"endTime":end_ms,"limit":1000
            }, timeout=20)
            if r.status_code != 200:
                time.sleep(2); continue
            data = r.json()
            if not data: break
            for k in data:
                candles.append({"t":int(k[0]),"open":float(k[1]),"high":float(k[2]),
                                 "low":float(k[3]),"close":float(k[4]),"volume":float(k[5])})
            current = int(data[-1][0]) + 1
            pages += 1
            if len(data) < 1000: break
            time.sleep(0.12)   # ~8 req/s — well within Binance 1200/min limit
            if pages % 10 == 0: print(".", end="", flush=True)
        except Exception as e:
            print(f" [retry: {e}]", end="", flush=True)
            time.sleep(3)
    print(f" {len(candles):,} candles ({pages} pages)")
    return candles

# ── Cache helpers (JSON file per symbol/tf) ────────────────────────────────────
def cache_key(symbol:str, tf:str) -> str:
    return f"{symbol}_{tf}"

def load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_cache(cache:Dict):
    with open(CACHE_FILE,"w") as f: json.dump(cache,f)

def get_candles(symbol:str, tf:str, cache:Dict) -> List[Dict]:
    key = cache_key(symbol, tf)
    if key in cache:
        print(f"    {symbol} {tf}: cached ({len(cache[key]):,} candles)")
        return cache[key]
    data = fetch_binance(symbol, TF_BN[tf], START_MS, END_MS)
    cache[key] = data
    save_cache(cache)
    return data

# ── Session quality ────────────────────────────────────────────────────────────
def session_q(ts_ms:int) -> float:
    hour = datetime.fromtimestamp(ts_ms/1000, tz=DUBAI_TZ).hour
    if 2<=hour<6:  return 0.72
    if 6<=hour<12: return 0.85
    if 12<=hour<16: return 0.95
    if 16<=hour<21: return 1.00
    if 21<=hour<=23: return 0.92
    return 0.80

def aligned_slice(candles:List[Dict], before_ts:int, n:int) -> List[Dict]:
    lo,hi,idx = 0,len(candles)-1,-1
    while lo<=hi:
        mid=(lo+hi)//2
        if candles[mid]["t"]<=before_ts: idx=mid; lo=mid+1
        else: hi=mid-1
    if idx<0: return []
    return candles[max(0,idx-n+1):idx+1]

# ── Fee helper ─────────────────────────────────────────────────────────────────
def fee(size_usd:float, is_tp:bool) -> float:
    ex = FEE["maker"] + (FEE["maker"] if is_tp else FEE["taker"]) + FEE["slip"]
    return size_usd * ex

# ── Exit checker ──────────────────────────────────────────────────────────────
def check_exit(trade:Dict, candle:Dict) -> Optional[Dict]:
    side,entry = trade["side"],trade["entry"]
    sl,tp      = trade["sl_pct"],trade["tp_pct"]
    lev        = trade["leverage"]
    eq         = trade["equity_at_open"]
    hi,lo      = candle["high"],candle["low"]
    sl_p = entry*(1-sl) if side=="LONG" else entry*(1+sl)
    tp_p = entry*(1+tp) if side=="LONG" else entry*(1-tp)
    def sl_hit(): return lo<=sl_p if side=="LONG" else hi>=sl_p
    def tp_hit(): return hi>=tp_p if side=="LONG" else lo<=tp_p

    if trade["grade"]=="B":
        t1_tp_p = entry*(1+tp) if side=="LONG" else entry*(1-tp)
        if not trade.get("t1_done"):
            siz = eq*trade["t1_size"]
            if sl_hit():
                p = siz*(-sl*lev)-fee(siz,False) + eq*trade["t2_size"]*(-sl*lev)-fee(eq*trade["t2_size"],False)
                return {"closed":True,"outcome":"SL_HIT","pnl":trade["acc"]+p,"exit_ts":candle["t"]}
            if (side=="LONG" and hi>=t1_tp_p) or (side=="SHORT" and lo<=t1_tp_p):
                p = siz*(tp*lev)-fee(siz,True)
                trade["acc"]+=p; trade["t1_done"]=True; trade["be_next"]=True
                return None
        if trade.get("t1_done"):
            if trade.get("be_next"): trade["be"]=True; trade["be_next"]=False
            t2s = eq*trade["t2_size"]
            t2sl = entry if trade.get("be") else sl_p
            if tp_hit():
                p=t2s*(tp*lev)-fee(t2s,True)
                return {"closed":True,"outcome":"T1_TP_T2_TP","pnl":trade["acc"]+p,"exit_ts":candle["t"]}
            if (side=="LONG" and lo<=t2sl) or (side=="SHORT" and hi>=t2sl):
                if trade.get("be"):
                    p=-fee(t2s,False)
                    return {"closed":True,"outcome":"T1_TP_T2_BE","pnl":trade["acc"]+p,"exit_ts":candle["t"]}
                p=t2s*(-sl*lev)-fee(t2s,False)
                return {"closed":True,"outcome":"T1_TP_T2_SL","pnl":trade["acc"]+p,"exit_ts":candle["t"]}
        return None
    else:
        sz = eq*trade["eff_size"]
        if tp_hit() and sl_hit():
            p=sz*(-sl*lev)-fee(sz,False)
            return {"closed":True,"outcome":"SL_HIT","pnl":p,"exit_ts":candle["t"]}
        if tp_hit():
            p=sz*(tp*lev)-fee(sz,True)
            return {"closed":True,"outcome":"TP_HIT","pnl":p,"exit_ts":candle["t"]}
        if sl_hit():
            p=sz*(-sl*lev)-fee(sz,False)
            return {"closed":True,"outcome":"SL_HIT","pnl":p,"exit_ts":candle["t"]}
        return None

# ── Metrics ────────────────────────────────────────────────────────────────────
def calc_metrics(trades:List[Dict], curve:List[Dict], bh_in:float, bh_out:float) -> Dict:
    total = len(trades)
    if total==0:
        return {"total_trades":0,"win_rate":0,"total_return_pct":0,
                "final_equity":START_EQUITY,"max_drawdown_pct":0,
                "sharpe_ratio":0,"reward_risk":0,"buy_hold_return_pct":0,
                "monthly_returns":[],"equity_curve":curve,"win_count":0,"loss_count":0}
    final = curve[-1]["equity"] if curve else START_EQUITY
    wins  = [t for t in trades if t["pnl"]>0]
    loss  = [t for t in trades if t["pnl"]<=0]
    wr    = len(wins)/total*100
    avg_w = statistics.mean(w["pnl"]/START_EQUITY*100 for w in wins)  if wins else 0
    avg_l = statistics.mean(l["pnl"]/START_EQUITY*100 for l in loss)  if loss else 0
    rr    = round(abs(avg_w/avg_l),2) if avg_l!=0 else 0
    ret   = (final-START_EQUITY)/START_EQUITY*100
    peak,max_dd=START_EQUITY,0.0
    for pt in curve:
        eq=pt["equity"]
        if eq>peak: peak=eq
        dd=(peak-eq)/peak if peak>0 else 0
        if dd>max_dd: max_dd=dd
    # Monthly
    monthly:Dict[str,Dict]={}; prev=START_EQUITY
    for t in sorted(trades,key=lambda x:x["exit_ts"]):
        key=datetime.fromtimestamp(t["exit_ts"]/1000,tz=timezone.utc).strftime("%Y-%m")
        if key not in monthly: monthly[key]={"start":prev,"pnl":0,"trades":0}
        monthly[key]["pnl"]+=t["pnl"]; monthly[key]["trades"]+=1; prev+=t["pnl"]
    mr=[{"month":k,"return_pct":round(v["pnl"]/v["start"]*100,2),"trades":v["trades"]}
        for k,v in sorted(monthly.items())]
    mr_vals=[m["return_pct"]/100 for m in mr]
    sharpe=round((statistics.mean(mr_vals)/statistics.stdev(mr_vals))*math.sqrt(12),2) if len(mr_vals)>=2 and statistics.stdev(mr_vals)>0 else 0
    bh = round((bh_out-bh_in)/bh_in*100,2) if bh_in>0 else 0
    return {"total_trades":total,"win_count":len(wins),"loss_count":len(loss),
            "win_rate":round(wr,1),"avg_win_pct":round(avg_w,2),"avg_loss_pct":round(avg_l,2),
            "reward_risk":rr,"total_return_pct":round(ret,2),"final_equity":round(final,2),
            "max_drawdown_pct":round(max_dd*100,2),"sharpe_ratio":sharpe,
            "buy_hold_return_pct":bh,"monthly_returns":mr,"equity_curve":curve}

# ── Run backtest ───────────────────────────────────────────────────────────────
def run_backtest(symbol:str, label:str, cache:Dict) -> Optional[Dict]:
    print(f"\n[BACKTEST] {label}")
    main_c   = get_candles(symbol, TF, cache)
    higher_c = get_candles(symbol, HIGHER_TF, cache)
    if len(main_c)<WARMUP+10:
        print(f"  SKIP: only {len(main_c)} candles"); return None

    st=TRADE_STYLE_PARAMS[STYLE]; c=MODE_PRESETS[MODE]
    equity=START_EQUITY; peak=START_EQUITY
    trades:List[Dict]=[]; curve:List[Dict]=[{"ts":main_c[WARMUP]["t"],"equity":equity}]
    open_t=None; total=len(main_c)-WARMUP; t0=time.time()

    # Patch _session_quality to use historical timestamps for this backtest run
    _orig_sq = _ind_mod._session_quality
    _ind_mod._session_quality = _session_quality_historical

    for i in range(WARMUP,len(main_c)):
        candle=main_c[i]
        if i%8000==0:
            pct=(i-WARMUP)/max(1,total)*100
            print(f"  {pct:.0f}% ({i-WARMUP:,}/{total:,} candles)...", end="\r", flush=True)

        if open_t:
            res=check_exit(open_t,candle)
            if res and res.get("closed"):
                equity+=res["pnl"]; peak=max(peak,equity)
                trades.append({"pnl":res["pnl"],"outcome":res["outcome"],
                    "side":open_t["side"],"grade":open_t["grade"],
                    "signal":open_t.get("signal","-"),"score":open_t.get("score",0),
                    "regime":open_t.get("regime","-"),
                    "open_ts":open_t["open_ts"],"exit_ts":res["exit_ts"],"entry":open_t["entry"]})
                curve.append({"ts":res["exit_ts"],"equity":round(equity,4)})
                open_t=None

        dd=(peak-equity)/peak if peak>0 else 0
        if dd>=0.15: break

        if open_t is None:
            _BACKTEST_TS_MS[0] = candle["t"]   # historical session quality
            klines=main_c[max(0,i-299):i+1]
            h_slice=aligned_slice(higher_c,candle["t"],100)
            sig=_compute_signal_layers(klines,MODE,1.0,h_slice,STYLE,None)
            if not sig.get("ok"): continue

            grade=sig.get("grade","B")
            atr=sig.get("atr_pct",ATR_BASELINE.get(TF,0.009))
            sl_pct=min(atr*st["sl_atr"],st["sl_max"]/100)
            tp_pct=min(atr*st["tp_atr"],st["tp_max"]/100)
            baseline=ATR_BASELINE.get(TF,0.009)
            vol_r=atr/baseline if baseline>0 else 1
            vol_m=max(0.4,1/vol_r) if vol_r>1.5 else 1.0
            dd_pct=(peak-equity)/peak if peak>0 else 0
            dd_m=(0.25 if dd_pct>=0.10 else 0.40 if dd_pct>=0.07 else 0.65 if dd_pct>=0.04 else 1.0)
            eff=c["size"]*vol_m*dd_m
            open_t={"entry":candle["close"],"side":sig["side"],"grade":grade,
                    "sl_pct":sl_pct,"tp_pct":tp_pct,"eff_size":eff,
                    "leverage":c["leverage"],"equity_at_open":equity,
                    "t1_size":eff*0.60,"t2_size":eff*0.40,
                    "t1_done":False,"be":False,"be_next":False,"acc":0.0,
                    "open_ts":candle["t"],"signal":sig.get("signal","-"),
                    "score":round(sig.get("score",0),3),"regime":sig.get("market_regime","-")}

    _ind_mod._session_quality = _orig_sq   # restore
    print(f"\n  Done in {time.time()-t0:.0f}s — {len(trades)} trades")
    m=calc_metrics(trades,curve,main_c[WARMUP]["close"],main_c[-1]["close"])
    m["symbol"]=symbol; m["label"]=label; m["equity_curve"]=curve
    print(f"  Return:{m['total_return_pct']:+.1f}%  WR:{m['win_rate']:.1f}%  Sharpe:{m['sharpe_ratio']:.2f}  MaxDD:{m['max_drawdown_pct']:.1f}%")
    return m

# ── Run optimizer ──────────────────────────────────────────────────────────────
def run_optimizer(symbol:str, label:str, cache:Dict) -> Optional[Dict]:
    print(f"  [OPT] {label} — 135 combos...", end="", flush=True)
    main_c   = cache.get(cache_key(symbol,TF),[])
    higher_c = cache.get(cache_key(symbol,HIGHER_TF),[])
    if len(main_c)<WARMUP+10: print(" SKIP"); return None

    combos=list(itertools.product(OPT_GRID["adx_delta"],OPT_GRID["score_delta"],
                                   OPT_GRID["sl_mult"],OPT_GRID["tp_mult"]))
    results=[]; t0=time.time()
    # Neutralize session quality inside _compute_signal_layers for optimizer:
    # all 135 combos run on the same data so equal treatment = fair comparison.
    _orig_sq2 = _ind_mod._session_quality
    _ind_mod._session_quality = lambda _ts: (1.0, "opt-backtest")
    for i,(adx_d,sc_d,sl_m,tp_m) in enumerate(combos):
        if i%27==0: print(".",end="",flush=True)
        m=_sim_one(main_c,higher_c,MODE,STYLE,EXCHANGE,START_EQUITY,adx_d,sc_d,sl_m,tp_m)
        if m:
            wr=m.get("win_rate",0)/100; dd=abs(m.get("max_drawdown_pct",0))/100
            passed=(m.get("total_trades",0)>=MIN_TRADES and wr>=0.25 and dd<=0.25)
            results.append({"adx_delta":adx_d,"score_delta":sc_d,"sl_mult":sl_m,"tp_mult":tp_m,
                "total_trades":m.get("total_trades",0),"win_rate":round(wr*100,1),
                "total_return":round(m.get("total_return_pct",0),2),
                "max_drawdown":round(m.get("max_drawdown_pct",0),2),
                "sharpe_ratio":round(m.get("sharpe_ratio",0),3),
                "reward_risk":round(m.get("reward_risk",0),3),
                "final_equity":round(m.get("final_equity",START_EQUITY),2),"passed":passed})
    _ind_mod._session_quality = _orig_sq2   # restore
    results.sort(key=lambda r:(r["passed"],r["sharpe_ratio"]),reverse=True)
    print(f" done ({time.time()-t0:.0f}s) — best Sharpe: {results[0]['sharpe_ratio'] if results else 'N/A'}")
    return {"top":results[:5],"best":results[0] if results else None}

# ── PDF ────────────────────────────────────────────────────────────────────────
def build_pdf(all_bt:List[Dict], all_opt:Dict[str,Dict]):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate,Table,TableStyle,Paragraph,
                                     Spacer,Image,PageBreak,HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_CENTER,TA_LEFT

    print("\n[PDF] Building report...")
    doc=SimpleDocTemplate(PDF_OUT,pagesize=A4,rightMargin=1.5*cm,leftMargin=1.5*cm,
                           topMargin=1.8*cm,bottomMargin=1.8*cm,
                           title="Asymmetric AI — Top 10 Coin Backtest Report")

    def rgb(r,g,b): return colors.Color(r/255,g/255,b/255)
    CTEAL=rgb(0,212,170); CRED=rgb(255,77,109); CGOLD=rgb(255,215,0)
    CBLUE=rgb(0,168,255); CDIM=rgb(13,18,38); CDIM2=rgb(17,24,42)
    CGRID=rgb(30,42,65);  CTEXT=colors.Color(0.75,0.80,0.90)

    styles=getSampleStyleSheet()
    def sty(name,**kw): return ParagraphStyle(name,**kw)
    T1 =sty("T1",fontName="Helvetica-Bold",fontSize=26,textColor=CTEAL,alignment=TA_CENTER,spaceAfter=4,leading=32)
    T2 =sty("T2",fontName="Helvetica",fontSize=11,textColor=colors.Color(0.65,0.70,0.82),alignment=TA_CENTER,spaceAfter=2)
    H2 =sty("H2",fontName="Helvetica-Bold",fontSize=16,textColor=CTEAL,spaceBefore=10,spaceAfter=6)
    H3 =sty("H3",fontName="Helvetica-Bold",fontSize=12,textColor=colors.Color(0.80,0.85,0.95),spaceBefore=7,spaceAfter=4)
    BD =sty("BD",fontName="Helvetica",fontSize=9,textColor=CTEXT,spaceAfter=4,leading=14)
    NT =sty("NT",fontName="Helvetica-Oblique",fontSize=7.5,textColor=colors.Color(0.45,0.50,0.60),spaceAfter=4)
    FT =sty("FT",fontName="Helvetica",fontSize=7.5,textColor=colors.Color(0.38,0.43,0.53),alignment=TA_CENTER)
    META=sty("META",fontName="Helvetica",fontSize=8.5,textColor=colors.Color(0.48,0.53,0.65),alignment=TA_CENTER)

    story=[]

    def fig2img(fig,wc=17,hc=5):
        buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor()); buf.seek(0); plt.close(fig)
        return Image(buf,width=wc*cm,height=hc*cm)

    def equity_chart(bt):
        curve=bt.get("equity_curve",[])
        if len(curve)<2: return None
        xs=[datetime.fromtimestamp(p["ts"]/1000,tz=timezone.utc) for p in curve]
        ys=[p["equity"] for p in curve]
        fig,ax=plt.subplots(figsize=(13,4.2)); fig.patch.set_facecolor("#080c18"); ax.set_facecolor("#0d1226")
        ax.plot(xs,ys,color="#00d4aa",linewidth=1.5,zorder=3)
        ax.fill_between(xs,ys,min(ys),alpha=0.13,color="#00d4aa",zorder=2)
        ax.axhline(y=START_EQUITY,color="#6b7a99",linewidth=0.7,linestyle="--",alpha=0.5)
        ret=(ys[-1]-START_EQUITY)/START_EQUITY*100; col="#00d4aa" if ret>=0 else "#ff4d6d"
        ax.set_title(f"{bt['label']} — Equity Curve | Return: {ret:+.1f}% | Final: ${ys[-1]:,.0f}",
                     color=col,fontsize=10,fontweight="bold",pad=7)
        ax.tick_params(colors="#6b7a99",labelsize=8)
        [s.set_edgecolor("#1e2a40") for s in ax.spines.values()]
        ax.grid(True,alpha=0.1,color="#6b7a99",linewidth=0.5); plt.tight_layout()
        return fig2img(fig,17,4.5)

    def monthly_chart(bt):
        mr=bt.get("monthly_returns",[])
        if not mr: return None
        rets=[m["return_pct"] for m in mr]
        cols=["#00d4aa" if r>=0 else "#ff4d6d" for r in rets]
        fig,ax=plt.subplots(figsize=(13,3.2)); fig.patch.set_facecolor("#080c18"); ax.set_facecolor("#0d1226")
        ax.bar(range(len(mr)),rets,color=cols,alpha=0.85,width=0.8)
        ax.axhline(y=0,color="#6b7a99",linewidth=0.6,alpha=0.5)
        yr=[(i,m["month"][:4]) for i,m in enumerate(mr) if m["month"].endswith("-01")]
        if yr: ax.set_xticks([t[0] for t in yr]); ax.set_xticklabels([t[1] for t in yr],color="#6b7a99",fontsize=8)
        ax.tick_params(colors="#6b7a99",labelsize=8); ax.set_ylabel("Monthly %",color="#6b7a99",fontsize=8)
        [s.set_edgecolor("#1e2a40") for s in ax.spines.values()]
        ax.grid(True,alpha=0.1,axis="y",color="#6b7a99"); ax.set_title("Monthly Returns",color="#6b7a99",fontsize=9,pad=5)
        plt.tight_layout(); return fig2img(fig,17,3.2)

    def tbl(data,col_w,style_extra=[]):
        t=Table(data,colWidths=col_w)
        base=[("BACKGROUND",(0,0),(-1,0),rgb(0,28,48)),("TEXTCOLOR",(0,0),(-1,0),CTEAL),
              ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8),
              ("TOPPADDING",(0,0),(-1,0),7),("BOTTOMPADDING",(0,0),(-1,0),7),
              ("FONTSIZE",(0,1),(-1,-1),8),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
              ("TEXTCOLOR",(0,1),(-1,-1),CTEXT),
              ("ROWBACKGROUNDS",(0,1),(-1,-1),[CDIM,CDIM2]),
              ("TOPPADDING",(0,1),(-1,-1),6),("BOTTOMPADDING",(0,1),(-1,-1),6),
              ("GRID",(0,0),(-1,-1),0.3,CGRID),("LINEBELOW",(0,0),(-1,0),0.8,CTEAL),
              ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),
              ("LEFTPADDING",(0,0),(0,-1),6)]+style_extra
        t.setStyle(TableStyle(base)); return t

    valid=[b for b in all_bt if b]

    # ── COVER ──────────────────────────────────────────────────────────────────
    story+=[Spacer(1,1.8*cm),Paragraph("ASYMMETRIC AI",T1),
            Paragraph("Top 10 Coin Backtest & Parameter Optimization Report",T2),Spacer(1,0.3*cm),
            Paragraph(f"Period: {DATE_FROM} → {DATE_TO}  |  Mode: {MODE} + {STYLE} (1h candles)  |  Start: ${START_EQUITY:,.0f}  |  Exchange: Bybit fee model",META),
            HRFlowable(width="100%",thickness=1,color=CTEAL,spaceAfter=16),Spacer(1,0.4*cm)]

    if valid:
        avg_r=statistics.mean(b["total_return_pct"] for b in valid)
        avg_w=statistics.mean(b["win_rate"] for b in valid)
        avg_s=statistics.mean(b["sharpe_ratio"] for b in valid)
        avg_d=statistics.mean(b["max_drawdown_pct"] for b in valid)
        best=max(valid,key=lambda b:b["total_return_pct"])
        worst=min(valid,key=lambda b:b["total_return_pct"])
        cv_d=[["Metric","Value"],
              ["Coins Tested",str(len(valid))],
              ["Average Total Return",f"{avg_r:+.1f}%"],
              ["Average Win Rate",f"{avg_w:.1f}%"],
              ["Average Sharpe Ratio",f"{avg_s:.2f}"],
              ["Average Max Drawdown",f"{avg_d:.1f}%"],
              ["Best Performer",f"{best['label']} ({best['total_return_pct']:+.1f}%)"],
              ["Worst Performer",f"{worst['label']} ({worst['total_return_pct']:+.1f}%)"],
              ["Optimizer Combos / Coin","135 (5 ADX × 3 Score × 3 SL × 3 TP)"]]
        cv_t=Table(cv_d,colWidths=[6*cm,9*cm])
        cv_t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),rgb(0,28,48)),("TEXTCOLOR",(0,0),(-1,0),CTEAL),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[CDIM,CDIM2]),
            ("TEXTCOLOR",(0,1),(-1,-1),CTEXT),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTNAME",(1,1),(-1,-1),"Helvetica-Bold"),
            ("TEXTCOLOR",(1,7),(1,7),CTEAL),("TEXTCOLOR",(1,8),(1,8),CRED),
            ("GRID",(0,0),(-1,-1),0.3,CGRID),("LINEBELOW",(0,0),(-1,0),0.8,CTEAL),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(0,-1),10)]))
        story.append(cv_t)

    story+=[Spacer(1,0.6*cm),
            Paragraph("All results use real historical Binance candles with Bybit fee model "
                      "(0.020%/0.055% + 0.025% slippage). Parameter optimizer tested 135 combinations "
                      "per coin. Past performance does not guarantee future results.",NT),
            PageBreak()]

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    story+=[Paragraph("Portfolio Summary — All 10 Coins",H2),
            Paragraph(f"MINI_ASYM + DAY_TRADE (1h) | {DATE_FROM} → {DATE_TO} | $1,000 starting capital",BD),
            Spacer(1,0.3*cm)]

    hdr=["Coin","Trades","Win Rate","Return","Max DD","Sharpe","R:R","Buy&Hold","Final $"]
    rows=[hdr]+[[b["label"],str(b["total_trades"]),f"{b['win_rate']:.1f}%",
                  f"{b['total_return_pct']:+.1f}%",f"{b['max_drawdown_pct']:.1f}%",
                  f"{b['sharpe_ratio']:.2f}",f"{b['reward_risk']:.2f}",
                  f"{b['buy_hold_return_pct']:+.0f}%",f"${b['final_equity']:,.0f}"]
                 for b in valid]
    cw=[3.8*cm,1.4*cm,1.7*cm,1.7*cm,1.6*cm,1.4*cm,1.4*cm,2.0*cm,2.2*cm]
    extra=[]
    for ri,row in enumerate(rows[1:],1):
        c=CTEAL if "+" in row[3] else (CRED if row[3].startswith("-") else CTEXT)
        extra.append(("TEXTCOLOR",(3,ri),(3,ri),c))
    story+=[tbl(rows,cw,extra),Spacer(1,0.5*cm)]

    # Combined equity chart
    story.append(Paragraph("All Coins — Equity Curves Overlay",H3))
    palette=["#00d4aa","#00a8ff","#ffd700","#ff4d6d","#ff8c42","#8b5cf6","#06b6d4","#a3e635","#f43f5e","#fb923c"]
    fig,ax=plt.subplots(figsize=(13,4.5)); fig.patch.set_facecolor("#080c18"); ax.set_facecolor("#0d1226")
    for idx,bt in enumerate(valid):
        cv=bt.get("equity_curve",[])
        if len(cv)<2: continue
        xs=[datetime.fromtimestamp(p["ts"]/1000,tz=timezone.utc) for p in cv]
        ys=[p["equity"] for p in cv]
        ax.plot(xs,ys,color=palette[idx%len(palette)],linewidth=1.2,alpha=0.85,label=bt["label"].split(" (")[0])
    ax.axhline(y=START_EQUITY,color="#6b7a99",linewidth=0.7,linestyle="--",alpha=0.5)
    ax.legend(loc="upper left",fontsize=7,ncol=2,facecolor="#0d1226",edgecolor="#1e2a40",labelcolor="#c0c8d8")
    ax.tick_params(colors="#6b7a99",labelsize=8); [s.set_edgecolor("#1e2a40") for s in ax.spines.values()]
    ax.grid(True,alpha=0.1,color="#6b7a99"); ax.set_ylabel("Equity ($)",color="#6b7a99",fontsize=9)
    plt.tight_layout(); story+=[fig2img(fig,17,5),PageBreak()]

    # ── PER-COIN PAGES ─────────────────────────────────────────────────────────
    for bt in valid:
        sym=bt["symbol"]; opt=all_opt.get(sym)
        story+=[Paragraph(bt["label"],H2),HRFlowable(width="100%",thickness=0.5,color=CTEAL,spaceAfter=8)]
        ret=bt.get("total_return_pct",0); rc=CTEAL if ret>=0 else CRED
        sd=[["Trades","Win Rate","Return","Max DD","Sharpe","R:R","Buy & Hold"],
            [str(bt["total_trades"]),f"{bt['win_rate']:.1f}%",f"{ret:+.1f}%",
             f"{bt['max_drawdown_pct']:.1f}%",f"{bt['sharpe_ratio']:.2f}",
             f"{bt['reward_risk']:.2f}",f"{bt['buy_hold_return_pct']:+.0f}%"]]
        st2=Table(sd,colWidths=[2.5*cm]*7)
        st2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),rgb(0,28,48)),("TEXTCOLOR",(0,0),(-1,0),colors.Color(0.48,0.58,0.68)),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),7.5),
            ("TOPPADDING",(0,0),(-1,0),6),("BOTTOMPADDING",(0,0),(-1,0),6),
            ("BACKGROUND",(0,1),(-1,1),CDIM),("TEXTCOLOR",(0,1),(-1,1),colors.white),
            ("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),("FONTSIZE",(0,1),(-1,1),13),
            ("TOPPADDING",(0,1),(-1,1),9),("BOTTOMPADDING",(0,1),(-1,1),9),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("GRID",(0,0),(-1,-1),0.3,CGRID),("LINEBELOW",(0,0),(-1,0),0.8,CTEAL),
            ("TEXTCOLOR",(2,1),(2,1),rc)]))
        story+=[st2,Spacer(1,0.4*cm)]
        eq=equity_chart(bt)
        if eq: story.append(eq)
        mo=monthly_chart(bt)
        if mo: story+=[Spacer(1,0.25*cm),mo]
        story.append(Spacer(1,0.4*cm))

        if opt and opt.get("top"):
            story+=[Paragraph("Parameter Optimizer — Top 5 Results (135 combos tested)",H3),
                    Paragraph("Sorted by Sharpe ratio. ADX Δ / Score Δ = offsets from MINI_ASYM defaults. SL × / TP × = ATR multipliers.",NT)]
            oh=["#","ADX Δ","Score Δ","SL ×","TP ×","Trades","WR","Return","MaxDD","Sharpe","R:R"]
            or2=[oh]+[[f"#{r+1}",f"{x['adx_delta']:+d}",f"{x['score_delta']:+.2f}",
                       f"{x['sl_mult']:.1f}×",f"{x['tp_mult']:.1f}×",str(x["total_trades"]),
                       f"{x['win_rate']:.1f}%",f"{x['total_return']:+.1f}%",
                       f"{x['max_drawdown']:.1f}%",f"{x['sharpe_ratio']:.3f}",f"{x['reward_risk']:.2f}"]
                      for r,x in enumerate(opt["top"])]
            ow=[0.8*cm,1.2*cm,1.5*cm,1.2*cm,1.2*cm,1.4*cm,1.4*cm,1.6*cm,1.5*cm,1.5*cm,1.2*cm]
            extra2=[("BACKGROUND",(0,1),(-1,1),rgb(0,48,34)),("TEXTCOLOR",(0,1),(-1,1),CTEAL),
                    ("FONTNAME",(0,1),(-1,1),"Helvetica-Bold")]
            story+=[tbl(or2,ow,extra2),Spacer(1,0.3*cm)]

            bp=opt["best"]
            if bp:
                base=MODE_SIGNAL_PARAMS.get(MODE,MODE_SIGNAL_PARAMS.get("NORMAL",{}))
                opt_adx=int(base.get("adx_min",16)+bp["adx_delta"])
                opt_sc=round(base.get("min_score",0.65)+bp["score_delta"],3)
                rd=[["★  OPTIMAL PARAMETERS FOR "+sym.replace("USDT",""),"",""],
                    ["ADX minimum",f"{opt_adx}",f"default: {base.get('adx_min',16)}"],
                    ["Min signal score",f"{opt_sc}",f"default: {base.get('min_score',0.65):.2f}"],
                    ["SL multiplier",f"{bp['sl_mult']:.1f}×","× ATR stop distance"],
                    ["TP multiplier",f"{bp['tp_mult']:.1f}×","× ATR take-profit distance"],
                    ["Optimized return",f"{bp['total_return']:+.1f}%","backtest result"],
                    ["Optimized Sharpe",f"{bp['sharpe_ratio']:.3f}","higher = better risk-adjusted"]]
                rt=Table(rd,colWidths=[4.5*cm,3*cm,5.5*cm])
                rt.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),rgb(0,55,38)),("TEXTCOLOR",(0,0),(-1,0),CTEAL),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),9),
                    ("SPAN",(0,0),(-1,0)),("ALIGN",(0,0),(-1,0),"CENTER"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[CDIM,CDIM2]),
                    ("TEXTCOLOR",(0,1),(-1,-1),CTEXT),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
                    ("FONTNAME",(1,1),(1,-1),"Helvetica-Bold"),("TEXTCOLOR",(1,1),(1,-1),colors.white),
                    ("TEXTCOLOR",(2,1),(2,-1),colors.Color(0.45,0.50,0.60)),
                    ("GRID",(0,0),(-1,-1),0.3,CGRID),("LINEBELOW",(0,0),(-1,0),1,CTEAL),
                    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                    ("LEFTPADDING",(0,0),(0,-1),8)]))
                story.append(rt)
        story.append(PageBreak())

    # ── OPTIMIZER CROSS-COIN SUMMARY ───────────────────────────────────────────
    story+=[Paragraph("Optimizer Summary — Best Parameters Across All 10 Coins",H2),
            Paragraph("Optimal parameters found by 135-combo grid search for each coin. "
                      "Apply these to the Asymmetric AI optimizer settings for maximum Sharpe.",BD),
            Spacer(1,0.3*cm)]
    sh=["Coin","ADX Δ","Score Δ","SL ×","TP ×","Trades","Win Rate","Return","Sharpe"]
    sr=[sh]
    for bt in valid:
        opt=all_opt.get(bt["symbol"]); bp=opt["best"] if opt and opt.get("best") else None
        sr.append([bt["label"],
                   f"{bp['adx_delta']:+d}" if bp else "N/A",
                   f"{bp['score_delta']:+.2f}" if bp else "N/A",
                   f"{bp['sl_mult']:.1f}×" if bp else "N/A",
                   f"{bp['tp_mult']:.1f}×" if bp else "N/A",
                   str(bp["total_trades"]) if bp else "N/A",
                   f"{bp['win_rate']:.1f}%" if bp else "N/A",
                   f"{bp['total_return']:+.1f}%" if bp else "N/A",
                   f"{bp['sharpe_ratio']:.3f}" if bp else "N/A"])
    sw=[3.8*cm,1.4*cm,1.6*cm,1.3*cm,1.3*cm,1.5*cm,1.7*cm,1.7*cm,1.7*cm]
    story+=[tbl(sr,sw),Spacer(1,1*cm),
            HRFlowable(width="100%",thickness=0.5,color=CTEAL,spaceAfter=8),
            Paragraph(f"Asymmetric AI — Mini-Asym Engine  |  Data: Binance OHLCV  |  "
                      f"Report: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC  |  mes29571@gmail.com",FT)]

    doc.build(story)
    print(f"[PDF] Saved → {PDF_OUT}")

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  ASYMMETRIC AI — TOP 10 COIN BACKTEST REPORT")
    print("="*60)
    print(f"  Period : {DATE_FROM} → {DATE_TO}")
    print(f"  Mode   : {MODE} + {STYLE} (1h)")
    print(f"  Coins  : {', '.join(c.replace('USDT','') for c in COINS)}")
    print(f"  Data   : Binance public API (1,000 candles/page)")
    print("="*60)

    cache=load_cache()
    all_bt:List[Dict]=[]; all_opt:Dict[str,Dict]={}
    t0=time.time()

    for i,sym in enumerate(COINS,1):
        label=COIN_LABELS.get(sym,sym)
        print(f"\n{'─'*50}  [{i}/10] {label}  {'─'*5}")
        bt=run_backtest(sym,label,cache)
        if bt: all_bt.append(bt)
        opt=run_optimizer(sym,label,cache)
        if opt: all_opt[sym]=opt

    print(f"\n{'='*60}")
    print(f"  All done in {(time.time()-t0)/60:.1f} minutes")
    print(f"{'='*60}")
    build_pdf(all_bt,all_opt)
    print(f"\n✓ PDF ready → {PDF_OUT}")
    os.system(f"open '{PDF_OUT}'")

if __name__=="__main__":
    main()
