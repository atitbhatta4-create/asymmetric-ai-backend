"""
run_comprehensive_report.py  — Asymmetric AI Pre-Launch Strategy Validation
=============================================================================
Scope  : 5 modes × 2 styles (DAY_TRADE + SWING) × 10 coins = 100 backtests
         135 optimizer combos per combination = 13,500 simulation runs
Period : 2020-01-01 → today
Equity : $1,000 starting, compounding (NOT reset each year)

PDF sections:
  1.  Cover + global mode×style verdict table
  2.  Year-by-year best strategy summary (2020–2026)
  3.  Per-year deep-dive (monthly chart + coin×mode table)
  4.  Per-style mode comparison (DAY_TRADE / SWING)
  5.  MASTER RECORD — per coin:
        • Complete yearly table (WR, Return, Sharpe, MaxDD)
        • Full monthly heat-map (Jan–Dec × 2020–2026)
        • Best optimizer parameters highlighted
  6.  Optimizer master table (all 100 combos, best params)
"""
from __future__ import annotations
import gc, itertools, os, sys, time, statistics, math, json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from config import TRADE_STYLE_PARAMS, DUBAI_TZ
from indicators import (
    _compute_signal_layers, _adx as _adx_fn, MODE_SIGNAL_PARAMS,
    get_market_regime, check_weekly_trend,
    check_pullback_entry, check_volume_hour_confirmed,
)
import indicators as _ind_mod
from coin_params import get_coin_params
from mode_params import get_mode_config, get_style_config
from optimizer import OPT_GRID, MIN_TRADES, _sim_one

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
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
COIN_SHORT = {s: COIN_LABELS[s].split(" (")[0] for s in COINS}
MODES  = ["ULTRA_SAFE","SAFE","NORMAL","MINI_ASYM","AGGRESSIVE"]
STYLES = ["DAY_TRADE","SWING"]
STYLE_CFG = {
    "DAY_TRADE": {"main_tf":"1h", "higher_tf":"4h"},
    "SWING":     {"main_tf":"4h", "higher_tf":"1d"},
}
YEAR_CONTEXT = {
    "2020":"COVID crash Mar→explosive recovery. Highly volatile.",
    "2021":"Mega bull run. BTC+300%, alts+1000%. Trend-following thrives.",
    "2022":"Brutal bear. LUNA collapse + FTX crash. Capital protection critical.",
    "2023":"Slow recovery. BTC+ETH lead, alts lag. Selective entries key.",
    "2024":"BTC ETF + halving. New ATHs. Strong bull market resumes.",
    "2025":"Post-halving consolidation. Mixed signals. Quality setups matter.",
    "2026":"Partial year (Jan–Jun). Early data only.",
}
MONTHS_SHORT = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
EXCHANGE     = "bybit"
START_EQUITY = 1000.0
DATE_FROM    = "2020-01-01"
DATE_TO      = "2026-05-31"
START_MS     = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS       = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
ALL_YEARS    = [str(y) for y in range(2020, datetime.now().year+1)]
PDF_OUT        = os.path.join(os.path.dirname(__file__),"asymmetric_ai_comprehensive_report.pdf")
RESULTS_JSON   = os.path.join(os.path.dirname(__file__),"comprehensive_results.json")
CACHE_FILE     = os.path.join(os.path.dirname(__file__),"candle_cache.json")
COIN_CACHE_DIR = os.path.join(os.path.dirname(__file__),"coin_caches")
ATR_BASELINE = {"15m":0.0040,"1h":0.0090,"4h":0.0180,"1d":0.0350}
MODE_PRESETS = {
    "ULTRA_SAFE":{"size":0.30,"leverage":2},
    "SAFE":      {"size":0.45,"leverage":3},
    "NORMAL":    {"size":0.60,"leverage":5},
    "MINI_ASYM": {"size":0.65,"leverage":6},
    "AGGRESSIVE":{"size":0.85,"leverage":8},
}
WARMUP = 240
TF_BN  = {"1h":"1h","4h":"4h","1d":"1d","1W":"1w"}
FEE    = {"maker":0.00020,"taker":0.00055,"slip":0.00025}

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION QUALITY PATCH
# ═══════════════════════════════════════════════════════════════════════════════
_BT_TS: List[int] = [0]

def _sess_hist(trade_style: str) -> tuple:
    if trade_style == "SWING": return 1.0, "swing"
    ts = _BT_TS[0]
    hour = datetime.fromtimestamp(ts/1000, tz=DUBAI_TZ).hour if ts else 14
    if 2<=hour<6:
        if trade_style=="SCALP": return 0.0, f"scalp-blocked {hour:02d}:xx"
        return 0.72, f"dead-zone {hour:02d}:xx"
    if 6<=hour<12:  return 0.85, f"Asia {hour:02d}:xx"
    if 12<=hour<16: return 0.95, f"London {hour:02d}:xx"
    if 16<=hour<21: return 1.00, f"London+NY {hour:02d}:xx"
    if 21<=hour<=23: return 0.92, f"NY {hour:02d}:xx"
    return 0.80, f"NY-close {hour:02d}:xx"

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════
def _fetch(symbol:str, interval:str, start_ms:int, end_ms:int) -> List[Dict]:
    url="https://api.binance.com/api/v3/klines"
    candles=[]; cur=start_ms; pages=0
    print(f"    Fetching {symbol} {interval}...", end="", flush=True)
    while cur < end_ms:
        try:
            r=httpx.get(url,params={"symbol":symbol,"interval":interval,
                "startTime":cur,"endTime":end_ms,"limit":1000},timeout=20)
            if r.status_code!=200: time.sleep(2); continue
            data=r.json()
            if not data: break
            for k in data:
                candles.append({"t":int(k[0]),"open":float(k[1]),"high":float(k[2]),
                                 "low":float(k[3]),"close":float(k[4]),"volume":float(k[5])})
            cur=int(data[-1][0])+1; pages+=1
            if len(data)<1000: break
            time.sleep(0.12)
            if pages%10==0: print(".",end="",flush=True)
        except Exception as e:
            print(f" [retry:{e}]",end="",flush=True); time.sleep(3)
    print(f" {len(candles):,} candles"); return candles

def load_cache()->Dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_cache(cache:Dict):
    with open(CACHE_FILE,"w") as f: json.dump(cache,f)

def _coin_cache_path(sym:str)->str:
    os.makedirs(COIN_CACHE_DIR, exist_ok=True)
    return os.path.join(COIN_CACHE_DIR, f"{sym}.json")

def load_coin_data(sym:str)->Dict:
    """Load candles for one coin only — keeps memory low."""
    path=_coin_cache_path(sym)
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except: pass
    # Fall back to monolithic cache (backward compat)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f: full=json.load(f)
            coin_data={k:v for k,v in full.items() if k.startswith(sym+"_")}
            if coin_data:
                with open(path,"w") as f: json.dump(coin_data,f)
                return coin_data
        except: pass
    return {}

def save_coin_data(sym:str, data:Dict):
    with open(_coin_cache_path(sym),"w") as f: json.dump(data,f)

def get_candles(sym:str, tf:str, cache:Dict)->List[Dict]:
    key=f"{sym}_{tf}"
    if key in cache:
        print(f"    {sym} {tf}: cached ({len(cache[key]):,})"); return cache[key]
    data=_fetch(sym,TF_BN[tf],START_MS,END_MS); cache[key]=data; save_cache(cache); return data

def fetch_and_cache_coin(sym:str)->None:
    """Fetch all TFs for one coin, save to per-coin file, keep nothing in RAM."""
    coin_data=load_coin_data(sym)
    changed=False
    for tf in ["1h","4h","1d"]:
        key=f"{sym}_{tf}"
        if key not in coin_data:
            coin_data[key]=_fetch(sym,TF_BN[tf],START_MS,END_MS)
            changed=True
        else:
            print(f"    {sym} {tf}: cached ({len(coin_data[key]):,})")
    if changed: save_coin_data(sym,coin_data)
    total=sum(len(v) for v in coin_data.values())
    print(f"    {sym}: {total:,} candles ready")

def aligned_slice(candles:List[Dict], before_ts:int, n:int)->List[Dict]:
    lo,hi,idx=0,len(candles)-1,-1
    while lo<=hi:
        mid=(lo+hi)//2
        if candles[mid]["t"]<=before_ts: idx=mid; lo=mid+1
        else: hi=mid-1
    if idx<0: return []
    return candles[max(0,idx-n+1):idx+1]

def _fee(sz:float, tp:bool)->float:
    return sz*(FEE["maker"]+(FEE["maker"] if tp else FEE["taker"])+FEE["slip"])

# ═══════════════════════════════════════════════════════════════════════════════
#  EXIT CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
def _exit(trade:Dict, candle:Dict)->Optional[Dict]:
    side,entry=trade["side"],trade["entry"]
    sl,tp,lev=trade["sl_pct"],trade["tp_pct"],trade["leverage"]
    eq=trade["equity_at_open"]; hi,lo=candle["high"],candle["low"]
    slp=entry*(1-sl) if side=="LONG" else entry*(1+sl)
    tpp=entry*(1+tp) if side=="LONG" else entry*(1-tp)
    sl_hit=(lo<=slp) if side=="LONG" else (hi>=slp)
    tp_hit=(hi>=tpp) if side=="LONG" else (lo<=tpp)
    if trade["grade"]=="B":
        if not trade.get("t1_done"):
            siz=eq*trade["t1_size"]
            if sl_hit:
                p=siz*(-sl*lev)-_fee(siz,False)+eq*trade["t2_size"]*(-sl*lev)-_fee(eq*trade["t2_size"],False)
                return {"closed":True,"outcome":"SL_HIT","pnl":trade["acc"]+p,"exit_ts":candle["t"]}
            if tp_hit:
                trade["acc"]+=siz*(tp*lev)-_fee(siz,True); trade["t1_done"]=True; trade["be_next"]=True; return None
        else:
            if trade.get("be_next"): trade["be"]=True; trade["be_next"]=False
            t2s=eq*trade["t2_size"]; t2sl=entry if trade.get("be") else slp
            if tp_hit:
                return {"closed":True,"outcome":"T1_TP_T2_TP","pnl":trade["acc"]+t2s*(tp*lev)-_fee(t2s,True),"exit_ts":candle["t"]}
            stop=((lo<=t2sl) if side=="LONG" else (hi>=t2sl))
            if stop:
                p=-_fee(t2s,False) if trade.get("be") else t2s*(-sl*lev)-_fee(t2s,False)
                return {"closed":True,"outcome":"T1_TP_T2_BE" if trade.get("be") else "T1_TP_T2_SL","pnl":trade["acc"]+p,"exit_ts":candle["t"]}
        return None
    else:
        sz=eq*trade["eff_size"]
        if tp_hit and sl_hit: return {"closed":True,"outcome":"SL_HIT","pnl":sz*(-sl*lev)-_fee(sz,False),"exit_ts":candle["t"]}
        if tp_hit: return {"closed":True,"outcome":"TP_HIT","pnl":sz*(tp*lev)-_fee(sz,True),"exit_ts":candle["t"]}
        if sl_hit: return {"closed":True,"outcome":"SL_HIT","pnl":sz*(-sl*lev)-_fee(sz,False),"exit_ts":candle["t"]}
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS  (full period + yearly + monthly)
# ═══════════════════════════════════════════════════════════════════════════════
def _sharpe(monthly_rets:List[float])->float:
    if len(monthly_rets)<2: return 0.0
    std=statistics.stdev(monthly_rets)
    return round(statistics.mean(monthly_rets)/std*math.sqrt(12),2) if std>0 else 0.0

def calc_metrics(trades:List[Dict], curve:List[Dict], bh_in:float, bh_out:float)->Dict:
    if not trades:
        return {"total_trades":0,"win_rate":0,"total_return_pct":0,"final_equity":START_EQUITY,
                "max_drawdown_pct":0,"sharpe_ratio":0,"reward_risk":0,"buy_hold_return_pct":0,
                "monthly_returns":[],"yearly":{},"equity_curve":curve,"win_count":0,"loss_count":0,
                "avg_win_pct":0,"avg_loss_pct":0}
    total=len(trades); wins=[t for t in trades if t["pnl"]>0]; loss=[t for t in trades if t["pnl"]<=0]
    final=curve[-1]["equity"] if curve else START_EQUITY
    avg_w=statistics.mean(w["pnl"]/START_EQUITY*100 for w in wins) if wins else 0
    avg_l=statistics.mean(l["pnl"]/START_EQUITY*100 for l in loss) if loss else 0
    rr=round(abs(avg_w/avg_l),2) if avg_l!=0 else 0
    ret=(final-START_EQUITY)/START_EQUITY*100
    peak=START_EQUITY; max_dd=0.0
    for pt in curve:
        eq=pt["equity"]
        if eq>peak: peak=eq
        dd=(peak-eq)/peak if peak>0 else 0
        if dd>max_dd: max_dd=dd

    # Full-period monthly
    monthly:Dict[str,Dict]={}; prev=START_EQUITY
    for t in sorted(trades,key=lambda x:x["exit_ts"]):
        mk=datetime.fromtimestamp(t["exit_ts"]/1000,tz=timezone.utc).strftime("%Y-%m")
        if mk not in monthly: monthly[mk]={"start":prev,"pnl":0,"trades":0,"wins":0}
        monthly[mk]["pnl"]+=t["pnl"]; monthly[mk]["trades"]+=1
        if t["pnl"]>0: monthly[mk]["wins"]+=1
        prev+=t["pnl"]
    mr=[{"month":k,"return_pct":round(v["pnl"]/v["start"]*100,2),"trades":v["trades"],"wins":v["wins"]}
        for k,v in sorted(monthly.items())]
    sharpe=_sharpe([m["return_pct"]/100 for m in mr])

    # Yearly breakdown (chained equity)
    yrs_found=sorted({datetime.fromtimestamp(t["exit_ts"]/1000,tz=timezone.utc).strftime("%Y") for t in trades})
    yearly:Dict[str,Dict]={}; prev_eq=START_EQUITY
    for yr in yrs_found:
        yr_t=[t for t in trades if datetime.fromtimestamp(t["exit_ts"]/1000,tz=timezone.utc).strftime("%Y")==yr]
        yr_wins=[t for t in yr_t if t["pnl"]>0]
        yr_pnl=sum(t["pnl"] for t in yr_t)
        yr_wr=len(yr_wins)/len(yr_t)*100 if yr_t else 0
        yr_ret=yr_pnl/prev_eq*100 if prev_eq>0 else 0
        # Monthly within year
        yr_mo:Dict[str,Dict]={}; mo_prev=prev_eq
        for t in sorted(yr_t,key=lambda x:x["exit_ts"]):
            mk=datetime.fromtimestamp(t["exit_ts"]/1000,tz=timezone.utc).strftime("%Y-%m")
            if mk not in yr_mo: yr_mo[mk]={"start":mo_prev,"pnl":0,"trades":0,"wins":0}
            yr_mo[mk]["pnl"]+=t["pnl"]; yr_mo[mk]["trades"]+=1
            if t["pnl"]>0: yr_mo[mk]["wins"]+=1
            mo_prev+=t["pnl"]
        yr_mr=[{"month":k,"return_pct":round(v["pnl"]/v["start"]*100,2),"trades":v["trades"],"wins":v["wins"]}
               for k,v in sorted(yr_mo.items())]
        # MaxDD within year
        yr_peak=prev_eq; yr_dd=0.0; yr_eq=prev_eq
        for t in sorted(yr_t,key=lambda x:x["exit_ts"]):
            yr_eq+=t["pnl"]
            if yr_eq>yr_peak: yr_peak=yr_eq
            d=(yr_peak-yr_eq)/yr_peak if yr_peak>0 else 0
            if d>yr_dd: yr_dd=d
        end_eq=prev_eq+yr_pnl
        yearly[yr]={"trades":len(yr_t),"win_count":len(yr_wins),"win_rate":round(yr_wr,1),
                    "return_pct":round(yr_ret,2),"pnl":round(yr_pnl,4),
                    "max_dd_pct":round(yr_dd*100,2),"sharpe":_sharpe([m["return_pct"]/100 for m in yr_mr]),
                    "start_equity":round(prev_eq,2),"end_equity":round(end_eq,2),"monthly":yr_mr}
        prev_eq=end_eq

    bh=round((bh_out-bh_in)/bh_in*100,2) if bh_in>0 else 0
    return {"total_trades":total,"win_count":len(wins),"loss_count":len(loss),
            "win_rate":round(len(wins)/total*100,1),"avg_win_pct":round(avg_w,2),
            "avg_loss_pct":round(avg_l,2),"reward_risk":rr,
            "total_return_pct":round(ret,2),"final_equity":round(final,2),
            "max_drawdown_pct":round(max_dd*100,2),"sharpe_ratio":sharpe,
            "buy_hold_return_pct":bh,"monthly_returns":mr,"yearly":yearly,"equity_curve":curve}

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(sym:str, label:str, mode:str, style:str,
                 main_c:List[Dict], higher_c:List[Dict],
                 btc_weekly_c:List[Dict]=None)->Optional[Dict]:
    if len(main_c)<WARMUP+10: return None
    st=TRADE_STYLE_PARAMS[style]; c=MODE_PRESETS[mode]
    tf=STYLE_CFG[style]["main_tf"]
    mode_cfg=get_mode_config(mode); style_cfg=get_style_config(style)
    coin_p=get_coin_params(sym)
    equity=START_EQUITY; peak=START_EQUITY
    trades:List[Dict]=[]; curve=[{"ts":main_c[WARMUP]["t"],"equity":equity}]
    open_t=None; t0=time.time()
    btc_wk=btc_weekly_c or []
    # Regime + pullback state
    is_parabolic=False; sig_pending=False; pending_side=""; candles_since=0
    _orig=_ind_mod._session_quality; _ind_mod._session_quality=_sess_hist
    try:
        for i in range(WARMUP,len(main_c)):
            candle=main_c[i]
            if open_t:
                res=_exit(open_t,candle)
                if res and res.get("closed"):
                    equity+=res["pnl"]; peak=max(peak,equity)
                    trades.append({"pnl":res["pnl"],"outcome":res["outcome"],
                        "side":open_t["side"],"grade":open_t["grade"],
                        "open_ts":open_t["open_ts"],"exit_ts":res["exit_ts"],"entry":open_t["entry"]})
                    curve.append({"ts":res["exit_ts"],"equity":round(equity,4)}); open_t=None
            dd=(peak-equity)/peak if peak>0 else 0
            if dd>=0.15: break
            if open_t is not None: continue
            _BT_TS[0]=candle["t"]
            klines=main_c[max(0,i-299):i+1]
            h_sl=aligned_slice(higher_c,candle["t"],100)

            # ── Regime detection ──────────────────────────────────────────
            regime="TRENDING"
            if btc_wk:
                wk_sl=aligned_slice(btc_wk,candle["t"],210)
                cur_adx=0.0
                if len(klines)>=15:
                    try:
                        cur_adx=_adx_fn([k["high"] for k in klines],[k["low"] for k in klines],
                                        [k["close"] for k in klines],14) or 0.0
                    except Exception: cur_adx=0.0
                _rng_thresh=style_cfg.get("ranging_adx_threshold",15)
                regime,is_parabolic=get_market_regime(wk_sl,cur_adx,is_parabolic,ranging_threshold=_rng_thresh)
            if regime=="RANGING":
                if sig_pending: sig_pending=False; pending_side=""; candles_since=0
                continue
            if regime not in mode_cfg["allowed_regimes"]:
                if sig_pending: sig_pending=False; pending_side=""; candles_since=0
                continue

            # ── SWING weekly trend filter ─────────────────────────────────
            _allowed_sides=None
            if style_cfg.get("swing_weekly_filter") and btc_wk:
                wk_sl=aligned_slice(btc_wk,candle["t"],210)
                wt=check_weekly_trend(wk_sl)
                if wt=="TRANSITIONING":
                    if sig_pending: sig_pending=False; pending_side=""; candles_since=0
                    continue
                _allowed_sides=["LONG"] if wt=="BULLISH" else ["SHORT"] if wt=="BEARISH" else None

            # ── PARABOLIC overrides ───────────────────────────────────────
            _param_ov={}
            if regime=="PARABOLIC":
                _param_ov={"min_score":max(0.35,coin_p["score_threshold"]-0.05)}
                _allowed_sides=["LONG"]

            sig=_compute_signal_layers(klines,mode,1.0,h_sl,style,None,
                                       param_overrides=_param_ov if _param_ov else None,
                                       symbol=sym)
            if _allowed_sides and sig.get("ok") and sig.get("side") not in _allowed_sides:
                sig["ok"]=False

            # ── Volume hour confirmation ──────────────────────────────────
            if sig.get("ok"):
                vol_ok,_,_=check_volume_hour_confirmed(klines,coin_p)
                if not vol_ok: sig["ok"]=False

            if not sig.get("ok"):
                if sig_pending: sig_pending=False; pending_side=""; candles_since=0
                continue

            desired_side=sig["side"]
            # ── Pullback entry gate ───────────────────────────────────────
            if not sig_pending:
                sig_pending=True; pending_side=desired_side; candles_since=0
            elif pending_side!=desired_side:
                pending_side=desired_side; candles_since=0
            candles_since+=1
            if style_cfg.get("pullback_required",True):
                pb=check_pullback_entry(klines,pending_side,coin_p,regime,candles_since,style)
                if pb=="TIMEOUT":
                    sig_pending=False; pending_side=""; candles_since=0; continue
                if pb=="WAIT": continue
            desired_side=pending_side
            sig_pending=False; pending_side=""; candles_since=0

            grade=sig.get("grade","B"); atr=sig.get("atr_pct",ATR_BASELINE.get(tf,0.009))
            sl_pct=min(atr*st["sl_atr"],st["sl_max"]/100)
            tp_pct=min(atr*st["tp_atr"],st["tp_max"]/100)
            bl=ATR_BASELINE.get(tf,0.009); vol_m=max(0.4,1/(atr/bl)) if atr/bl>1.5 else 1.0
            dd_pct=(peak-equity)/peak if peak>0 else 0
            dd_m=0.25 if dd_pct>=0.10 else 0.40 if dd_pct>=0.07 else 0.65 if dd_pct>=0.04 else 1.0
            mode_sm=mode_cfg.get("position_size_mult",1.0)
            eff=c["size"]*vol_m*dd_m*mode_sm
            open_t={"entry":candle["close"],"side":desired_side,"grade":grade,
                    "sl_pct":sl_pct,"tp_pct":tp_pct,"eff_size":eff,"leverage":c["leverage"],
                    "equity_at_open":equity,"t1_size":eff*0.60,"t2_size":eff*0.40,
                    "t1_done":False,"be":False,"be_next":False,"acc":0.0,"open_ts":candle["t"]}
    finally:
        _ind_mod._session_quality=_orig
    m=calc_metrics(trades,curve,main_c[WARMUP]["close"],main_c[-1]["close"])
    m.update({"symbol":sym,"label":label,"mode":mode,"style":style,"elapsed":round(time.time()-t0,1)})
    return m

# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
def run_optimizer(sym:str, mode:str, style:str,
                  main_c:List[Dict], higher_c:List[Dict],
                  btc_weekly_c:List[Dict]=None)->Optional[Dict]:
    if len(main_c)<WARMUP+10: return None
    combos=list(itertools.product(OPT_GRID["adx_delta"],OPT_GRID["score_delta"],
                                   OPT_GRID["sl_mult"],OPT_GRID["tp_mult"]))
    results=[]; _orig=_ind_mod._session_quality
    _ind_mod._session_quality=lambda _:(1.0,"opt-bt")
    try:
        for adx_d,sc_d,sl_m,tp_m in combos:
            m=_sim_one(main_c,higher_c,mode,style,EXCHANGE,START_EQUITY,adx_d,sc_d,sl_m,tp_m,
                       symbol=sym,btc_weekly_candles=btc_weekly_c or [])
            if m:
                wr=m.get("win_rate",0)/100; dd=abs(m.get("max_drawdown_pct",0))/100
                passed=(m.get("total_trades",0)>=MIN_TRADES and wr>=0.25 and dd<=0.25)
                results.append({"adx_delta":adx_d,"score_delta":sc_d,"sl_mult":sl_m,"tp_mult":tp_m,
                    "total_trades":m.get("total_trades",0),"win_rate":round(wr*100,1),
                    "total_return":round(m.get("total_return_pct",0),2),
                    "max_drawdown":round(m.get("max_drawdown_pct",0),2),
                    "sharpe_ratio":round(m.get("sharpe_ratio",0),3),
                    "reward_risk":round(m.get("reward_risk",0),3),"passed":passed})
    finally:
        _ind_mod._session_quality=_orig
    results.sort(key=lambda r:(r["passed"],r["sharpe_ratio"]),reverse=True)
    return {"top":results[:10],"best":results[0] if results else None}

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS FOR PDF
# ═══════════════════════════════════════════════════════════════════════════════
def _best_for_coin_year(results:Dict, sym:str, yr:str)->Optional[Dict]:
    """Return best (mode, style, yearly_data) for a given coin+year by Sharpe."""
    best=None; best_sh=-999
    for style in STYLES:
        for mode in MODES:
            bt=results.get(style,{}).get(mode,{}).get(sym,{}).get("bt")
            if bt and yr in bt.get("yearly",{}):
                yd=bt["yearly"][yr]
                if yd["trades"]>=2 and yd["sharpe"]>best_sh:
                    best_sh=yd["sharpe"]; best={"mode":mode,"style":style,"yr_data":yd,"sharpe":yd["sharpe"]}
    return best

def _month_ret(monthly:List[Dict], mo_num:int)->Optional[float]:
    """Return return_pct for month mo_num (1=Jan) from a monthly list, or None."""
    suffix=f"-{mo_num:02d}"
    for m in monthly:
        if m["month"].endswith(suffix): return m["return_pct"]
    return None

def _best_opt(results:Dict, sym:str)->Optional[Dict]:
    """Return best optimizer result for a coin across all mode+style combos."""
    best=None; best_sh=-999
    for style in STYLES:
        for mode in MODES:
            opt=results.get(style,{}).get(mode,{}).get(sym,{}).get("opt")
            bp=opt["best"] if opt and opt.get("best") else None
            if bp and bp["sharpe_ratio"]>best_sh and bp.get("passed",False):
                best_sh=bp["sharpe_ratio"]
                best={"mode":mode,"style":style,"params":bp}
    if best is None:
        for style in STYLES:
            for mode in MODES:
                opt=results.get(style,{}).get(mode,{}).get(sym,{}).get("opt")
                bp=opt["best"] if opt and opt.get("best") else None
                if bp and bp["sharpe_ratio"]>best_sh:
                    best_sh=bp["sharpe_ratio"]; best={"mode":mode,"style":style,"params":bp}
    return best

# ═══════════════════════════════════════════════════════════════════════════════
#  PDF BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_pdf(results:Dict, output_path:str=None) -> bytes:
    """Build PDF, return raw bytes. Optionally also save to output_path."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate,Table,TableStyle,
                                    Paragraph,Spacer,Image,PageBreak,HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER,TA_LEFT

    print("\n[PDF] Building comprehensive report...")
    story = []  # will be built below; doc created after story is ready

    def rgb(r,g,b): return colors.Color(r/255,g/255,b/255)
    CTEAL=rgb(0,212,170); CRED=rgb(255,77,109); CGOLD=rgb(255,215,0)
    CDIM=rgb(13,18,38); CDIM2=rgb(17,24,42); CGRID=rgb(30,42,65)
    CTEXT=colors.Color(0.75,0.80,0.90); CBLUE=rgb(0,168,255); CORANGE=rgb(255,140,66)
    MODE_COL={"ULTRA_SAFE":CTEAL,"SAFE":CBLUE,"NORMAL":CGOLD,"MINI_ASYM":CORANGE,"AGGRESSIVE":CRED}
    PALETTE=["#00d4aa","#00a8ff","#ffd700","#ff8c42","#ff4d6d","#8b5cf6","#06b6d4","#a3e635","#f43f5e","#fb923c"]

    def P(name,**kw): return ParagraphStyle(name,**kw)
    T1  =P("T1",fontName="Helvetica-Bold",fontSize=24,textColor=CTEAL,alignment=TA_CENTER,spaceAfter=4,leading=30)
    T2  =P("T2",fontName="Helvetica",fontSize=10,textColor=colors.Color(0.65,0.70,0.82),alignment=TA_CENTER,spaceAfter=2)
    H2  =P("H2",fontName="Helvetica-Bold",fontSize=15,textColor=CTEAL,spaceBefore=8,spaceAfter=5)
    H3  =P("H3",fontName="Helvetica-Bold",fontSize=12,textColor=colors.Color(0.80,0.85,0.95),spaceBefore=6,spaceAfter=4)
    H4  =P("H4",fontName="Helvetica-Bold",fontSize=10,textColor=CTEAL,spaceBefore=4,spaceAfter=3)
    BD  =P("BD",fontName="Helvetica",fontSize=9,textColor=CTEXT,spaceAfter=4,leading=14)
    NT  =P("NT",fontName="Helvetica-Oblique",fontSize=7.5,textColor=colors.Color(0.45,0.50,0.60),spaceAfter=3)
    FT  =P("FT",fontName="Helvetica",fontSize=7.5,textColor=colors.Color(0.38,0.43,0.53),alignment=TA_CENTER)
    META=P("META",fontName="Helvetica",fontSize=8.5,textColor=colors.Color(0.48,0.53,0.65),alignment=TA_CENTER)

    story=[]

    def fig2img(fig,wc=17,hc=5):
        buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor=fig.get_facecolor()); buf.seek(0)
        plt.close(fig); plt.close("all"); gc.collect()
        return Image(buf,width=wc*cm,height=hc*cm)

    def T(data,cw,ex=[]):
        t=Table(data,colWidths=cw)
        base=[("BACKGROUND",(0,0),(-1,0),rgb(0,28,48)),("TEXTCOLOR",(0,0),(-1,0),CTEAL),
              ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),7.5),
              ("TOPPADDING",(0,0),(-1,0),5),("BOTTOMPADDING",(0,0),(-1,0),5),
              ("FONTSIZE",(0,1),(-1,-1),7.5),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
              ("TEXTCOLOR",(0,1),(-1,-1),CTEXT),("ROWBACKGROUNDS",(0,1),(-1,-1),[CDIM,CDIM2]),
              ("TOPPADDING",(0,1),(-1,-1),4),("BOTTOMPADDING",(0,1),(-1,-1),4),
              ("GRID",(0,0),(-1,-1),0.3,CGRID),("LINEBELOW",(0,0),(-1,0),0.8,CTEAL),
              ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),
              ("LEFTPADDING",(0,0),(0,-1),5)]+ex
        t.setStyle(TableStyle(base)); return t

    def rc(v): return CTEAL if "+" in str(v) else (CRED if str(v).startswith("-") else CTEXT)

    # ─── gather global stats ──────────────────────────────────────────────────
    def valid_bts(style,mode):
        return [results[style][mode][s]["bt"] for s in COINS
                if results.get(style,{}).get(mode,{}).get(s,{}).get("bt")
                and results[style][mode][s]["bt"].get("total_trades",0)>0]

    # ══════════════════════════════════════════════════════════════════════════
    #  1. COVER
    # ══════════════════════════════════════════════════════════════════════════
    story+=[Spacer(1,1.2*cm),Paragraph("ASYMMETRIC AI",T1),
            Paragraph("Comprehensive Pre-Launch Strategy Validation Report",T2),Spacer(1,0.3*cm),
            Paragraph(f"Period: {DATE_FROM} → {DATE_TO}  |  Starting Equity: $1,000  |  Bybit Fee Model",META),
            Paragraph("5 Modes × 2 Styles (DAY_TRADE + SWING) × 10 Coins = 100 Backtests + 13,500 Optimizer Runs",META),
            HRFlowable(width="100%",thickness=1,color=CTEAL,spaceAfter=14),Spacer(1,0.2*cm)]

    hdr=["Mode + Style","Coins","Avg Return","Avg WR","Avg Sharpe","Avg MaxDD","Verdict"]
    rows=[hdr]
    for style in STYLES:
        for mode in MODES:
            bts=valid_bts(style,mode)
            if not bts: rows.append([f"{mode} + {style}","0","—","—","—","—","No data"]); continue
            avg_r=statistics.mean(b["total_return_pct"] for b in bts)
            avg_w=statistics.mean(b["win_rate"] for b in bts)
            avg_s=statistics.mean(b["sharpe_ratio"] for b in bts)
            avg_d=statistics.mean(b["max_drawdown_pct"] for b in bts)
            v=("★ Excellent" if avg_s>=1.5 and avg_r>20 else "✓ Good" if avg_s>=0.5 and avg_r>0 else "△ Mixed" if avg_r>-10 else "✗ Poor")
            rows.append([f"{mode} + {style}",str(len(bts)),f"{avg_r:+.1f}%",f"{avg_w:.1f}%",f"{avg_s:.2f}",f"{avg_d:.1f}%",v])
    ex=[]
    for ri,row in enumerate(rows[1:],1):
        if row[2]!="—": ex.append(("TEXTCOLOR",(2,ri),(2,ri),rc(row[2])))
        if "Excellent" in row[-1]: ex+=[("TEXTCOLOR",(-1,ri),(-1,ri),CTEAL),("FONTNAME",(-1,ri),(-1,ri),"Helvetica-Bold")]
        elif "Poor" in row[-1]: ex.append(("TEXTCOLOR",(-1,ri),(-1,ri),CRED))
        mn=row[0].split(" +")[0]
        if mn in MODE_COL: ex.append(("TEXTCOLOR",(0,ri),(0,ri),MODE_COL[mn]))
    story+=[T(rows,[3.8*cm,1.2*cm,1.8*cm,1.8*cm,1.8*cm,1.6*cm,2.0*cm],ex),Spacer(1,0.4*cm),
            Paragraph("Sharpe ≥1.5 + Return >20% = Excellent. Past results do not guarantee future performance.",NT),PageBreak()]

    # ══════════════════════════════════════════════════════════════════════════
    #  2. YEAR-BY-YEAR BEST STRATEGY
    # ══════════════════════════════════════════════════════════════════════════
    story+=[Paragraph("Year-by-Year Best Strategy (2020–2026)",H2),
            Paragraph("For each year, the mode+style combination with the highest average Sharpe across coins that traded.",BD),
            Spacer(1,0.2*cm)]

    yr_winners:Dict[str,Optional[Dict]]={}
    for yr in ALL_YEARS:
        combos_yr=[]
        for style in STYLES:
            for mode in MODES:
                shs=[]; rets=[]
                for sym in COINS:
                    bt=results.get(style,{}).get(mode,{}).get(sym,{}).get("bt")
                    if bt and yr in bt.get("yearly",{}) and bt["yearly"][yr]["trades"]>=2:
                        shs.append(bt["yearly"][yr]["sharpe"]); rets.append(bt["yearly"][yr]["return_pct"])
                if shs: combos_yr.append({"mode":mode,"style":style,"avg_sh":round(statistics.mean(shs),2),"avg_ret":round(statistics.mean(rets),2),"coins":len(shs)})
        combos_yr.sort(key=lambda x:x["avg_sh"],reverse=True)
        yr_winners[yr]=combos_yr[0] if combos_yr else None

    yr_hdr=["Year","Market Context","Best Mode","Style","Avg Sharpe","Avg Return","Coins w/ Trades"]
    yr_rows=[yr_hdr]
    for yr in ALL_YEARS:
        w=yr_winners.get(yr)
        ctx=YEAR_CONTEXT.get(yr,"")[:50]
        if w: yr_rows.append([yr,ctx,w["mode"],w["style"],f"{w['avg_sh']:.2f}",f"{w['avg_ret']:+.1f}%",str(w["coins"])])
        else: yr_rows.append([yr,ctx,"—","—","—","—","0"])
    ex2=[]
    for ri,row in enumerate(yr_rows[1:],1):
        if row[5]!="—": ex2.append(("TEXTCOLOR",(5,ri),(5,ri),rc(row[5])))
        if row[2] in MODE_COL: ex2.append(("TEXTCOLOR",(2,ri),(2,ri),MODE_COL[row[2]]))
    story+=[T(yr_rows,[1.2*cm,5.0*cm,2.4*cm,2.0*cm,1.8*cm,1.8*cm,2.0*cm],ex2),Spacer(1,0.4*cm),PageBreak()]

    # ══════════════════════════════════════════════════════════════════════════
    #  3. PER-YEAR PAGES
    # ══════════════════════════════════════════════════════════════════════════
    for yr in ALL_YEARS:
        story+=[Paragraph(f"{yr} — Year in Detail",H2),
                Paragraph(YEAR_CONTEXT.get(yr,""),P("YC",fontName="Helvetica-Oblique",fontSize=8,textColor=colors.Color(0.60,0.65,0.75),spaceAfter=4)),
                HRFlowable(width="100%",thickness=0.5,color=CTEAL,spaceAfter=6)]
        w=yr_winners.get(yr)
        if w: story.append(Paragraph(f"Best Strategy: {w['mode']} + {w['style']}  —  avg Sharpe {w['avg_sh']:.2f}, avg return {w['avg_ret']:+.1f}%, {w['coins']} coins",BD))

        # Coin × mode return table for this year (DAY_TRADE columns)
        ythdr=["Coin"]+[f"{m[:3]}\nDT" for m in MODES]+[f"{m[:3]}\nSW" for m in MODES]
        ytrows=[ythdr]
        for sym in COINS:
            row=[COIN_SHORT[sym]]
            for style in STYLES:
                for mode in MODES:
                    bt=results.get(style,{}).get(mode,{}).get(sym,{}).get("bt")
                    yd=bt["yearly"].get(yr) if bt else None
                    row.append(f"{yd['return_pct']:+.0f}%\n{yd['win_rate']:.0f}%WR" if yd and yd["trades"]>0 else "—")
            ytrows.append(row)
        cw_yt=[2.0*cm]+[1.55*cm]*10
        ex_yt=[]
        for ri,row in enumerate(ytrows[1:],1):
            for ci,cell in enumerate(row[1:],1):
                if cell!="—" and cell!="": ex_yt.append(("TEXTCOLOR",(ci,ri),(ci,ri),rc(cell)))
        story+=[T(ytrows,cw_yt,ex_yt),Spacer(1,0.3*cm)]

        # Monthly bar chart — best mode + best coin for this year
        best_mn=w["mode"] if w else "MINI_ASYM"; best_st=w["style"] if w else "DAY_TRADE"
        best_coin_yd=None; best_sh2=-999; best_coin_name=""
        for sym in COINS:
            bt=results.get(best_st,{}).get(best_mn,{}).get(sym,{}).get("bt")
            if bt and yr in bt.get("yearly",{}):
                yd=bt["yearly"][yr]
                if yd["trades"]>=2 and yd["sharpe"]>best_sh2:
                    best_sh2=yd["sharpe"]; best_coin_yd=yd; best_coin_name=COIN_SHORT[sym]
        if best_coin_yd and best_coin_yd.get("monthly"):
            mr=best_coin_yd["monthly"]; rets=[m["return_pct"] for m in mr]
            labs=[MONTHS_SHORT[int(m["month"][5:])-1] for m in mr]
            fig,ax=plt.subplots(figsize=(13,2.6)); fig.patch.set_facecolor("#080c18"); ax.set_facecolor("#0d1226")
            ax.bar(range(len(mr)),rets,color=["#00d4aa" if r>=0 else "#ff4d6d" for r in rets],alpha=0.85,width=0.75)
            ax.axhline(y=0,color="#6b7a99",linewidth=0.5,alpha=0.5)
            ax.set_xticks(range(len(mr))); ax.set_xticklabels(labs,color="#6b7a99",fontsize=8)
            ax.tick_params(colors="#6b7a99",labelsize=8); ax.set_ylabel("Monthly %",color="#6b7a99",fontsize=8)
            [s.set_edgecolor("#1e2a40") for s in ax.spines.values()]; ax.grid(True,alpha=0.1,axis="y",color="#6b7a99")
            ax.set_title(f"{yr} Monthly — {best_mn}+{best_st} | {best_coin_name} | Year: {best_coin_yd['return_pct']:+.1f}%  WR: {best_coin_yd['win_rate']:.0f}%  Trades: {best_coin_yd['trades']}",
                         color="#c0c8d8",fontsize=8,pad=4); plt.tight_layout()
            story.append(fig2img(fig,17,2.8))
        story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  4. PER-STYLE SUMMARY PAGES
    # ══════════════════════════════════════════════════════════════════════════
    for style in STYLES:
        tf_lbl="1h candles" if style=="DAY_TRADE" else "4h candles"
        story+=[Paragraph(f"{style}  ({tf_lbl}) — All Modes × All Coins",H2),
                HRFlowable(width="100%",thickness=0.5,color=CTEAL,spaceAfter=6)]
        shdr=["Coin","Mode","Trades","WR","Return","MaxDD","Sharpe","R:R","Final $"]
        srows=[shdr]
        for sym in COINS:
            for mode in MODES:
                bt=results.get(style,{}).get(mode,{}).get(sym,{}).get("bt")
                if not bt: continue
                srows.append([COIN_SHORT[sym],mode,str(bt["total_trades"]),f"{bt['win_rate']:.1f}%",
                    f"{bt['total_return_pct']:+.1f}%",f"{bt['max_drawdown_pct']:.1f}%",
                    f"{bt['sharpe_ratio']:.2f}",f"{bt['reward_risk']:.2f}",f"${bt['final_equity']:,.0f}"])
        cw_s=[2.2*cm,2.2*cm,1.3*cm,1.3*cm,1.6*cm,1.3*cm,1.4*cm,1.2*cm,2.0*cm]
        ex_s=[]
        for ri,row in enumerate(srows[1:],1):
            if row[2] in MODE_COL: ex_s.append(("TEXTCOLOR",(1,ri),(1,ri),MODE_COL[row[2]]))
            if row[4]!="—": ex_s.append(("TEXTCOLOR",(4,ri),(4,ri),rc(row[4])))
        story+=[T(srows,cw_s,ex_s),Spacer(1,0.4*cm)]
        # BTC mode overlay equity chart
        fig,ax=plt.subplots(figsize=(13,3.8)); fig.patch.set_facecolor("#080c18"); ax.set_facecolor("#0d1226")
        any_p=False
        for mi,mode in enumerate(MODES):
            bt=results.get(style,{}).get(mode,{}).get("BTCUSDT",{}).get("bt")
            if bt and bt.get("equity_curve") and bt["total_trades"]>0:
                cv=bt["equity_curve"]; xs=[datetime.fromtimestamp(p["ts"]/1000,tz=timezone.utc) for p in cv]; ys=[p["equity"] for p in cv]
                ax.plot(xs,ys,color=PALETTE[mi],linewidth=1.3,label=f"{mode} ({bt['total_return_pct']:+.1f}%)",alpha=0.9); any_p=True
        if any_p:
            ax.axhline(y=START_EQUITY,color="#6b7a99",linewidth=0.6,linestyle="--",alpha=0.5)
            ax.legend(loc="upper left",fontsize=7,ncol=3,facecolor="#0d1226",edgecolor="#1e2a40",labelcolor="#c0c8d8")
            ax.tick_params(colors="#6b7a99",labelsize=8); [s.set_edgecolor("#1e2a40") for s in ax.spines.values()]
            ax.grid(True,alpha=0.1,color="#6b7a99"); ax.set_ylabel("Equity ($)",color="#6b7a99",fontsize=9)
            ax.set_title(f"{style} — BTC: All 5 Modes Equity Comparison",color="#6b7a99",fontsize=9,pad=5)
            plt.tight_layout(); story.append(fig2img(fig,17,4))
        else: plt.close(fig)
        story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  5. MASTER RECORD — PER COIN (one page each)
    # ══════════════════════════════════════════════════════════════════════════
    story+=[Paragraph("MASTER RECORD — Complete Yearly + Monthly Breakdown per Coin",H2),
            Paragraph("For each coin: best mode highlighted in green, full year-by-year + month-by-month return table, and optimal parameters from the parameter optimizer.",BD),
            HRFlowable(width="100%",thickness=1,color=CTEAL,spaceAfter=10)]

    for sym in COINS:
        coin_label=COIN_LABELS[sym]
        best_opt_info=_best_opt(results,sym)
        opt_line=""
        if best_opt_info:
            bp=best_opt_info["params"]
            opt_line=(f"Optimizer Best: {best_opt_info['mode']} + {best_opt_info['style']}  |  "
                      f"ADX {bp['adx_delta']:+d}  Score {bp['score_delta']:+.2f}  "
                      f"SL {bp['sl_mult']:.1f}×  TP {bp['tp_mult']:.1f}×  |  "
                      f"Return {bp['total_return']:+.1f}%  WR {bp['win_rate']:.1f}%  Sharpe {bp['sharpe_ratio']:.3f}")

        story+=[Paragraph(f"■  {coin_label}",H3),
                Paragraph(opt_line,P("OPT",fontName="Helvetica-Bold",fontSize=8,textColor=CTEAL,spaceAfter=4)) if opt_line else Spacer(1,0.1*cm),
                HRFlowable(width="100%",thickness=0.4,color=CTEAL,spaceAfter=5)]

        # ── A. Yearly summary: for each year, best mode+style ────────────────
        story.append(Paragraph("Yearly Performance (best mode per year)",H4))
        yr_hdr2=["Year","Best Mode","Style","Trades","Win Rate","Return","Sharpe","MaxDD","Start $","End $"]
        yr_rows2=[yr_hdr2]
        for yr in ALL_YEARS:
            byr=_best_for_coin_year(results,sym,yr)
            if byr:
                yd=byr["yr_data"]
                yr_rows2.append([yr,byr["mode"],byr["style"],str(yd["trades"]),
                    f"{yd['win_rate']:.1f}%",f"{yd['return_pct']:+.2f}%",
                    f"{yd['sharpe']:.2f}",f"{yd['max_dd_pct']:.1f}%",
                    f"${yd['start_equity']:,.0f}",f"${yd['end_equity']:,.0f}"])
            else:
                yr_rows2.append([yr,"—","—","0","—","—","—","—","—","—"])
        ex_y2=[]
        for ri,row in enumerate(yr_rows2[1:],1):
            if row[5]!="—": ex_y2.append(("TEXTCOLOR",(5,ri),(5,ri),rc(row[5])))
            if row[1] in MODE_COL: ex_y2.append(("TEXTCOLOR",(1,ri),(1,ri),MODE_COL[row[1]]))
        story+=[T(yr_rows2,[1.1*cm,2.4*cm,1.8*cm,1.2*cm,1.5*cm,1.6*cm,1.3*cm,1.3*cm,1.6*cm,1.6*cm],ex_y2),
                Spacer(1,0.3*cm)]

        # ── B. Monthly heat-map: Month × Year ───────────────────────────────
        story.append(Paragraph("Monthly Return Heat-Map (all years, best mode per year)",H4))
        mo_hdr=["Month"]+ALL_YEARS
        mo_rows=[mo_hdr]
        for mo_num in range(1,13):
            row=[MONTHS_SHORT[mo_num-1]]
            for yr in ALL_YEARS:
                byr=_best_for_coin_year(results,sym,yr)
                if byr:
                    ret=_month_ret(byr["yr_data"].get("monthly",[]),mo_num)
                    row.append(f"{ret:+.1f}%" if ret is not None else "—")
                else:
                    row.append("—")
            mo_rows.append(row)
        # Add a "Total" row per year
        tot_row=["YEAR TOTAL"]
        for yr in ALL_YEARS:
            byr=_best_for_coin_year(results,sym,yr)
            tot_row.append(f"{byr['yr_data']['return_pct']:+.1f}%\n{byr['yr_data']['win_rate']:.0f}%WR" if byr else "—")
        mo_rows.append(tot_row)
        n_yrs=len(ALL_YEARS); cw_mo=[1.3*cm]+[2.0*cm]*n_yrs
        ex_mo=[]
        for ri,row in enumerate(mo_rows[1:],1):
            for ci,cell in enumerate(row[1:],1):
                if cell!="—" and cell!="": ex_mo.append(("TEXTCOLOR",(ci,ri),(ci,ri),rc(cell)))
        # Bold + colour the TOTAL row
        last_ri=len(mo_rows)-1
        ex_mo+=[("FONTNAME",(0,last_ri),(-1,last_ri),"Helvetica-Bold"),
                ("BACKGROUND",(0,last_ri),(-1,last_ri),rgb(0,28,48)),
                ("TEXTCOLOR",(0,last_ri),(0,last_ri),CTEAL)]
        story+=[T(mo_rows,cw_mo,ex_mo),Spacer(1,0.3*cm)]

        # ── C. All-mode comparison table (current period, best style per mode)
        story.append(Paragraph("All Mode × Style Returns (full period 2020→today)",H4))
        am_hdr=["Mode","Style","Trades","Win Rate","Return","MaxDD","Sharpe","R:R","Final $","Opt Return","Opt Sharpe"]
        am_rows=[am_hdr]
        for mode in MODES:
            for style in STYLES:
                bt=results.get(style,{}).get(mode,{}).get(sym,{}).get("bt")
                opt=results.get(style,{}).get(mode,{}).get(sym,{}).get("opt")
                bp=opt["best"] if opt and opt.get("best") else None
                if not bt: continue
                am_rows.append([mode,style,str(bt["total_trades"]),f"{bt['win_rate']:.1f}%",
                    f"{bt['total_return_pct']:+.1f}%",f"{bt['max_drawdown_pct']:.1f}%",
                    f"{bt['sharpe_ratio']:.2f}",f"{bt['reward_risk']:.2f}",
                    f"${bt['final_equity']:,.0f}",
                    f"{bp['total_return']:+.1f}%" if bp else "—",
                    f"{bp['sharpe_ratio']:.3f}" if bp else "—"])
        cw_am=[2.2*cm,1.8*cm,1.2*cm,1.3*cm,1.5*cm,1.3*cm,1.3*cm,1.1*cm,1.5*cm,1.7*cm,1.5*cm]
        ex_am=[]
        for ri,row in enumerate(am_rows[1:],1):
            if row[0] in MODE_COL: ex_am.append(("TEXTCOLOR",(0,ri),(0,ri),MODE_COL[row[0]]))
            if row[4]!="—": ex_am.append(("TEXTCOLOR",(4,ri),(4,ri),rc(row[4])))
            if row[9]!="—": ex_am.append(("TEXTCOLOR",(9,ri),(9,ri),rc(row[9])))
        # Highlight best row (highest Sharpe)
        best_sh_r=-999; best_ri=1
        for ri,row in enumerate(am_rows[1:],1):
            try:
                sh=float(row[6])
                if sh>best_sh_r: best_sh_r=sh; best_ri=ri
            except: pass
        ex_am.append(("BACKGROUND",(0,best_ri),(-1,best_ri),rgb(0,45,30)))
        story+=[T(am_rows,cw_am,ex_am),Spacer(1,0.3*cm)]

        # ── D. Equity curve + monthly bars (best overall combo) ──────────────
        best_overall=None; best_sh3=-999
        for mode in MODES:
            for style in STYLES:
                bt=results.get(style,{}).get(mode,{}).get(sym,{}).get("bt")
                if bt and bt["total_trades"]>=5 and bt["sharpe_ratio"]>best_sh3:
                    best_sh3=bt["sharpe_ratio"]; best_overall=bt
        if best_overall:
            cv=best_overall.get("equity_curve",[]); mr=best_overall.get("monthly_returns",[])
            if len(cv)>=2 and mr:
                fig,axes=plt.subplots(2,1,figsize=(13,5.5),gridspec_kw={"height_ratios":[3,2],"hspace":0.45})
                fig.patch.set_facecolor("#080c18")
                ax1,ax2=axes
                for ax in axes: ax.set_facecolor("#0d1226"); [s.set_edgecolor("#1e2a40") for s in ax.spines.values()]
                xs=[datetime.fromtimestamp(p["ts"]/1000,tz=timezone.utc) for p in cv]; ys=[p["equity"] for p in cv]
                ax1.plot(xs,ys,color="#00d4aa",linewidth=1.4,zorder=3)
                ax1.fill_between(xs,ys,min(ys),alpha=0.10,color="#00d4aa")
                ax1.axhline(y=START_EQUITY,color="#6b7a99",linewidth=0.6,linestyle="--",alpha=0.5)
                ret_v=(ys[-1]-START_EQUITY)/START_EQUITY*100
                ax1.set_title(f"Best: {best_overall['mode']} + {best_overall['style']}  |  "
                              f"Return: {ret_v:+.1f}%  Trades: {best_overall['total_trades']}  "
                              f"WR: {best_overall['win_rate']:.1f}%  Sharpe: {best_overall['sharpe_ratio']:.2f}",
                              color="#00d4aa" if ret_v>=0 else "#ff4d6d",fontsize=8,fontweight="bold",pad=4)
                ax1.tick_params(colors="#6b7a99",labelsize=7); ax1.grid(True,alpha=0.1,color="#6b7a99")
                rets_m=[m["return_pct"] for m in mr]
                ax2.bar(range(len(mr)),rets_m,color=["#00d4aa" if r>=0 else "#ff4d6d" for r in rets_m],alpha=0.85,width=0.85)
                ax2.axhline(y=0,color="#6b7a99",linewidth=0.5,alpha=0.5)
                yr_tks=[(i,m["month"][:4]) for i,m in enumerate(mr) if m["month"].endswith("-01")]
                if yr_tks: ax2.set_xticks([t[0] for t in yr_tks]); ax2.set_xticklabels([t[1] for t in yr_tks],color="#6b7a99",fontsize=7)
                ax2.tick_params(colors="#6b7a99",labelsize=7); ax2.grid(True,alpha=0.1,axis="y",color="#6b7a99")
                ax2.set_ylabel("Monthly %",color="#6b7a99",fontsize=7)
                ax2.set_title("Monthly Returns — Full Period",color="#6b7a99",fontsize=7,pad=3)
                story.append(fig2img(fig,17,5.6))
        story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  6. OPTIMIZER MASTER TABLE
    # ══════════════════════════════════════════════════════════════════════════
    story+=[Paragraph("Parameter Optimizer — Best Settings per Coin × Mode × Style",H2),
            Paragraph("Top result from 135-combo grid search (ADX Δ / Score Δ / SL× / TP×). "
                      "Apply these in the app optimizer panel for maximum Sharpe.",BD),Spacer(1,0.2*cm)]
    oh=["Style","Mode","Coin","ADX Δ","Score Δ","SL ×","TP ×","Trades","WR","Return","Sharpe","Passed"]
    opt_rows=[oh]
    for style in STYLES:
        for mode in MODES:
            for sym in COINS:
                opt=results.get(style,{}).get(mode,{}).get(sym,{}).get("opt")
                bp=opt["best"] if opt and opt.get("best") else None
                if not bp: continue
                opt_rows.append([style,mode,COIN_SHORT[sym],
                    f"{bp['adx_delta']:+d}",f"{bp['score_delta']:+.2f}",
                    f"{bp['sl_mult']:.1f}×",f"{bp['tp_mult']:.1f}×",
                    str(bp["total_trades"]),f"{bp['win_rate']:.1f}%",
                    f"{bp['total_return']:+.1f}%",f"{bp['sharpe_ratio']:.3f}",
                    "✓" if bp.get("passed") else "✗"])
    ow=[1.8*cm,2.1*cm,1.8*cm,1.1*cm,1.4*cm,1.1*cm,1.1*cm,1.2*cm,1.2*cm,1.6*cm,1.5*cm,1.0*cm]
    ex_opt=[]
    for ri,row in enumerate(opt_rows[1:],1):
        if row[9]!="—": ex_opt.append(("TEXTCOLOR",(9,ri),(9,ri),rc(row[9])))
        if row[1] in MODE_COL: ex_opt.append(("TEXTCOLOR",(1,ri),(1,ri),MODE_COL[row[1]]))
        if row[-1]=="✓": ex_opt.append(("TEXTCOLOR",(-1,ri),(-1,ri),CTEAL))
        else: ex_opt.append(("TEXTCOLOR",(-1,ri),(-1,ri),CRED))
    story+=[T(opt_rows,ow,ex_opt),Spacer(1,0.8*cm),
            HRFlowable(width="100%",thickness=0.5,color=CTEAL,spaceAfter=8),
            Paragraph(f"Asymmetric AI — Pre-Launch Comprehensive Validation  |  Binance OHLCV  |  "
                      f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC  |  mes29571@gmail.com",FT)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=1.4*cm, leftMargin=1.4*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm,
                            title="Asymmetric AI — Comprehensive Strategy Report")
    doc.build(story)
    pdf_bytes = buf.getvalue()
    save_path = output_path or PDF_OUT
    try:
        with open(save_path, "wb") as f: f.write(pdf_bytes)
        print(f"[PDF] Saved → {save_path}  ({len(pdf_bytes)//1024} KB)")
    except Exception as e:
        print(f"[PDF] Could not save locally: {e}")
    return pdf_bytes

# ═══════════════════════════════════════════════════════════════════════════════
#  CALLABLE ENTRY POINT (used by routes_report.py on Render)
# ═══════════════════════════════════════════════════════════════════════════════
def run_full(progress_cb=None) -> bytes:
    """
    Run all backtests + optimizers, build PDF, return raw PDF bytes.
    progress_cb(msg: str, done: int) is called after each coin if provided.
    Processes one coin at a time to keep memory usage low.
    """
    print("[run_full] Phase 1: caching candle data per coin...")
    for sym in COINS:
        fetch_and_cache_coin(sym)
        gc.collect()

    # Fetch BTC weekly candles once — shared for regime + weekly trend detection
    print("[run_full] Fetching BTC weekly candles for regime detection...")
    _btc_wk_cache: Dict = {}
    btc_weekly_c: List[Dict] = []
    try:
        btc_weekly_c = get_candles("BTCUSDT", "1W", _btc_wk_cache)
        print(f"[run_full] BTC weekly: {len(btc_weekly_c)} candles ready")
    except Exception as _e:
        print(f"[run_full] BTC weekly fetch failed ({_e}) — regime defaults to TRENDING")

    results:Dict[str,Dict[str,Dict[str,Dict]]]={s:{m:{} for m in MODES} for s in STYLES}
    done=0; total=len(STYLES)*len(MODES)*len(COINS)

    def _serial(obj):
        if isinstance(obj,dict): return {k:_serial(v) for k,v in obj.items()}
        if isinstance(obj,list): return [_serial(i) for i in obj]
        if isinstance(obj,(int,float,str,bool)) or obj is None: return obj
        return str(obj)

    print("[run_full] Phase 2: backtesting (coin-by-coin)...")
    for sym in COINS:
        label=COIN_LABELS[sym]
        coin_data=load_coin_data(sym)

        for style,scfg in STYLE_CFG.items():
            main_tf=scfg["main_tf"]; higher_tf=scfg["higher_tf"]
            mc=coin_data.get(f"{sym}_{main_tf}",[]); hc=coin_data.get(f"{sym}_{higher_tf}",[])

            for mode in MODES:
                done+=1
                print(f"  [{done}/{total}] {label} | {mode} | {style}")
                bt=None
                try:
                    bt=run_backtest(sym,label,mode,style,mc,hc,btc_weekly_c)
                except Exception as e:
                    print(f"    BT ERROR: {e}")
                opt=None
                bt_trades=bt["total_trades"] if bt else 0
                if bt_trades>=5:
                    try:
                        opt=run_optimizer(sym,mode,style,mc,hc,btc_weekly_c)
                    except Exception as e:
                        print(f"    OPT ERROR: {e}")
                else:
                    print(f"    OPT SKIP ({bt_trades} trades)")
                results[style][mode][sym]={"bt":bt,"opt":opt}
                if progress_cb:
                    try: progress_cb(f"{done}/{total} — {label} {mode} {style}", done)
                    except Exception: pass
                try:
                    with open(RESULTS_JSON,"w") as f: json.dump(_serial(results),f)
                except Exception: pass

        del coin_data, mc, hc
        gc.collect()

    return build_pdf(results)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*72)
    print("  ASYMMETRIC AI — COMPREHENSIVE PRE-LAUNCH STRATEGY VALIDATION")
    print("="*72)
    n=len(MODES)*len(STYLES)*len(COINS)
    print(f"  Period : {DATE_FROM} → {DATE_TO}  |  Equity: ${START_EQUITY:,.0f} (compounding)")
    print(f"  Modes  : {' | '.join(MODES)}")
    print(f"  Styles : {' | '.join(STYLES)}")
    print(f"  Coins  : {', '.join(c.replace('USDT','') for c in COINS)}")
    print(f"  Total  : {n} backtests + {n*135:,} optimizer runs")
    print("="*72)

    # ── Phase 1: fetch & cache all candles to disk, one coin at a time ──────────
    print("\n[DATA] Verifying candle data...")
    for sym in COINS:
        fetch_and_cache_coin(sym)
        gc.collect()
    print("[DATA] All candles cached to disk.\n")

    print("[DATA] Fetching BTC weekly candles for regime detection...")
    _btc_wk_cache2: Dict = {}
    btc_weekly_c: List[Dict] = []
    try:
        btc_weekly_c = get_candles("BTCUSDT", "1W", _btc_wk_cache2)
        print(f"[DATA] BTC weekly: {len(btc_weekly_c)} candles ready\n")
    except Exception as _e:
        print(f"[DATA] BTC weekly fetch failed ({_e}) — regime defaults to TRENDING\n")

    results:Dict[str,Dict[str,Dict[str,Dict]]]={s:{m:{} for m in MODES} for s in STYLES}
    grand_t0=time.time(); done=0; total=len(STYLES)*len(MODES)*len(COINS)

    def _serial(obj):
        if isinstance(obj,dict): return {k:_serial(v) for k,v in obj.items()}
        if isinstance(obj,list): return [_serial(i) for i in obj]
        if isinstance(obj,(int,float,str,bool)) or obj is None: return obj
        return str(obj)

    # ── Phase 2: process one coin at a time (coin-outer loop saves ~10× memory) ─
    for sym in COINS:
        label=COIN_LABELS[sym]
        print(f"\n{'='*72}\n  COIN: {label}\n{'='*72}")

        coin_data=load_coin_data(sym)

        for style,scfg in STYLE_CFG.items():
            main_tf=scfg["main_tf"]; higher_tf=scfg["higher_tf"]
            mc=coin_data.get(f"{sym}_{main_tf}",[]); hc=coin_data.get(f"{sym}_{higher_tf}",[])
            print(f"\n  ── STYLE: {style} ──")

            for mode in MODES:
                done+=1
                print(f"\n  [{done}/{total}] {label} | {mode} | {style}")

                bt=None
                try:
                    bt=run_backtest(sym,label,mode,style,mc,hc,btc_weekly_c)
                except Exception as e:
                    print(f"    BT: ERROR — {e}")
                if bt:
                    print(f"    BT: {bt['total_trades']:3d} trades | {bt['total_return_pct']:+6.1f}% | WR {bt['win_rate']:.1f}% | Sharpe {bt['sharpe_ratio']:5.2f} | MaxDD {bt['max_drawdown_pct']:.1f}%  ({bt['elapsed']:.0f}s)")
                    for yr,yd in sorted(bt.get("yearly",{}).items()):
                        mo_str=" | ".join(f"{m['month'][5:]}:{m['return_pct']:+.1f}%" for m in yd.get("monthly",[]))
                        print(f"         {yr}: {yd['return_pct']:+.1f}%  WR:{yd['win_rate']:.0f}%  T:{yd['trades']}  Sh:{yd['sharpe']:.2f}  [{mo_str}]")
                else:
                    print("    BT: SKIP")

                bt_trades=bt["total_trades"] if bt else 0
                opt=None
                if bt_trades<5:
                    print(f"    OPT SKIP (only {bt_trades} trades)")
                else:
                    print(f"    OPT ...",end="",flush=True)
                    opt_t0=time.time()
                    try:
                        opt=run_optimizer(sym,mode,style,mc,hc,btc_weekly_c)
                    except Exception as e:
                        print(f" ERROR — {e}")
                    bp=opt["best"] if opt and opt.get("best") else None
                    if bp:
                        print(f" Sh={bp['sharpe_ratio']:.3f}  Ret={bp['total_return']:+.1f}%  ({time.time()-opt_t0:.0f}s)"
                              f"  ADX:{bp['adx_delta']:+d} Sc:{bp['score_delta']:+.2f} SL:{bp['sl_mult']:.1f}× TP:{bp['tp_mult']:.1f}×")
                    else:
                        print(f" no valid combo ({time.time()-opt_t0:.0f}s)")

                results[style][mode][sym]={"bt":bt,"opt":opt}
                try:
                    with open(RESULTS_JSON,"w") as f: json.dump(_serial(results),f)
                except Exception: pass

        # Free this coin's candles before loading the next one
        del coin_data, mc, hc
        gc.collect()
        print(f"[MEM] {label} candles freed")

    # Save results to JSON for recovery
    print("\n[SAVE] Writing final results to JSON...")
    try:
        def _serial(obj):
            if isinstance(obj,dict): return {k:_serial(v) for k,v in obj.items()}
            if isinstance(obj,list): return [_serial(i) for i in obj]
            if isinstance(obj,(int,float,str,bool)) or obj is None: return obj
            return str(obj)
        with open(RESULTS_JSON,"w") as f: json.dump(_serial(results),f)
        print(f"[SAVE] Results saved → {RESULTS_JSON}")
    except Exception as e:
        print(f"[SAVE] Warning: could not save JSON: {e}")

    elapsed=(time.time()-grand_t0)/60
    print(f"\n{'='*72}\n  Completed in {elapsed:.1f} minutes\n{'='*72}")

    pdf_bytes = build_pdf(results)

    # Upload to R2 if credentials are available (cloud runs)
    r2_key = None
    r2_access = os.getenv("R2_ACCESS_KEY_ID","")
    r2_secret = os.getenv("R2_SECRET_ACCESS_KEY","")
    r2_endpoint = os.getenv("R2_ENDPOINT","")
    r2_bucket = os.getenv("R2_BUCKET","asymmetric-ai-backups")
    if all([r2_access, r2_secret, r2_endpoint]):
        try:
            import boto3, httpx
            from botocore.config import Config as _Cfg
            client = boto3.client("s3", endpoint_url=r2_endpoint,
                                  aws_access_key_id=r2_access,
                                  aws_secret_access_key=r2_secret,
                                  config=_Cfg(signature_version="s3v4"), region_name="auto")
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            r2_key = f"reports/comprehensive_{ts}.pdf"
            presigned = client.generate_presigned_url("put_object",
                Params={"Bucket":r2_bucket,"Key":r2_key,"ContentType":"application/pdf"},
                ExpiresIn=600)
            httpx.put(presigned, content=pdf_bytes,
                      headers={"Content-Type":"application/pdf"}, timeout=300, verify=False).raise_for_status()
            print(f"[R2] PDF uploaded → {r2_key}  ({len(pdf_bytes)//1024} KB)")
        except Exception as e:
            print(f"[R2] Upload failed: {e}")

    # Telegram alert when done
    try:
        from notifications import tg_alert
        msg = (f"✅ Comprehensive Report DONE in {elapsed:.0f} min\n"
               f"100 backtests + optimizers complete.\n"
               + (f"PDF in R2 → {r2_key}" if r2_key else f"PDF saved locally → {PDF_OUT}"))
        tg_alert(msg)
    except Exception:
        pass

    if not r2_key:
        os.system(f"open '{PDF_OUT}'")

if __name__=="__main__":
    main()
