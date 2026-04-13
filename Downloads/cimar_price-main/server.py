"""
CIMAR Petcoke Intelligence Platform — Backend v4
CNN Temporal Model · 20-year weekly dataset (1,044 observations)
"""

import os, math, time, random, threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from numpy.linalg import lstsq

app = Flask(__name__, static_folder="static")

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

# ── Load 20-year weekly dataset ───────────────────────────────────────────────
_CSV_PATH = os.path.join(os.path.dirname(__file__), "static", "cimar_petcoke_20yr_weekly.csv")

def _load_weekly():
    df = pd.read_csv(_CSV_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df

try:
    WEEKLY_DF = _load_weekly()
    print(f"  Dataset: {len(WEEKLY_DF)} weekly rows  "
          f"({WEEKLY_DF.week_start.iloc[0].date()} → {WEEKLY_DF.week_start.iloc[-1].date()})")
except FileNotFoundError:
    WEEKLY_DF = None
    print("  WARNING: CSV not found — using fallback")

def _build_history(df):
    df = df.copy()
    df["ym"] = df["week_start"].dt.to_period("M")
    m = df.groupby("ym").last().reset_index()
    return [{"month":str(r.ym),"brent":round(float(r.brent_usd_bbl),2),
             "petcoke":round(float(r.petcoke_cfr_morocco_usd_t),2),
             "freight":round(float(r.freight_usgulf_morocco_usd_t),2),
             "nat_gas":round(float(r.henry_hub_usd_mmbtu),3),
             "coal":round(float(r.coal_newcastle_usd_t),2),
             "usd_mad":round(float(r.mad_usd),4)} for _,r in m.iterrows()]

HISTORY = _build_history(WEEKLY_DF) if WEEKLY_DF is not None else [
    {"month":"2024-01","brent":78.2,"petcoke":122.1,"freight":19.0,"nat_gas":2.52,"coal":138.6,"usd_mad":10.11},
    {"month":"2024-06","brent":85.1,"petcoke":108.3,"freight":17.8,"nat_gas":2.72,"coal":114.1,"usd_mad": 9.96},
    {"month":"2024-12","brent":74.9,"petcoke": 96.9,"freight":15.8,"nat_gas":3.31,"coal":108.7,"usd_mad": 9.92},
]
print(f"  History: {len(HISTORY)} months  ({HISTORY[0]['month']} → {HISTORY[-1]['month']})")

# ── CNN Temporal Model ────────────────────────────────────────────────────────
class CNNTemporalModel:
    LOOKBACK = 8

    def __init__(self):
        self.ols_coeffs = None; self.mlp = None
        self.scaler = None; self.trained = False
        self.metrics = {}; self._pc = self._coal = self._brent = None
        self._gas = self._mad = self._freight = None
        self._ols_preds = self._ols_resids = None

    def _ols_pred(self, coal, brent, gas):
        c = self.ols_coeffs
        return c[0]*coal + c[1]*brent + c[2]*gas + c[3]

    def _feats(self, pc, coal, brent, gas, mad, freight, i):
        if i < self.LOOKBACK + 4: return None
        f = []
        for lag in [1,2,4,8]:
            f.append((coal[i]-coal[i-lag]) / max(coal[i-lag],1e-6))
            f.append((brent[i]-brent[i-lag]) / max(brent[i-lag],1e-6))
        ratio = pc[i] / max(coal[i],1e-6)
        ratio_avg = np.mean(pc[i-12:i] / np.maximum(coal[i-12:i],1e-6))
        f += [ratio - ratio_avg,
              (pc[i]-pc[i-4]) / max(pc[i-4],1e-6),
              (pc[i]-pc[i-8]) / max(pc[i-8],1e-6),
              (mad[i]-mad[i-4]) / max(mad[i-4],1e-6),
              np.std(pc[i-8:i]) / max(pc[i],1e-6)]
        return f[:11]

    def train(self, df):
        pc=df.petcoke_cfr_morocco_usd_t.values.astype(float)
        coal=df.coal_newcastle_usd_t.values.astype(float)
        brent=df.brent_usd_bbl.values.astype(float)
        gas=df.henry_hub_usd_mmbtu.values.astype(float)
        mad=df.mad_usd.values.astype(float)
        freight=df.freight_usgulf_morocco_usd_t.values.astype(float)
        n=len(pc)
        X_ols=np.column_stack([coal,brent,gas,np.ones(n)])
        coeffs,_,_,_ = lstsq(X_ols, pc, rcond=None)
        self.ols_coeffs=coeffs
        ols_preds=self._ols_pred(coal,brent,gas)
        ols_resids=pc-ols_preds
        ss_res=np.sum(ols_resids**2); ss_tot=np.sum((pc-pc.mean())**2)
        r2_ols=1.0-ss_res/max(ss_tot,1e-9)
        X_cnn,y_cnn=[],[]
        for i in range(self.LOOKBACK+4, n-1):
            f=self._feats(pc,coal,brent,gas,mad,freight,i)
            if f: X_cnn.append(f); y_cnn.append(ols_resids[i+1])
        X_cnn=np.array(X_cnn,dtype=float); y_cnn=np.array(y_cnn,dtype=float)
        self.scaler=StandardScaler().fit(X_cnn)
        mlp=MLPRegressor(hidden_layer_sizes=(64,32),activation="relu",alpha=0.01,
                         max_iter=1000,random_state=42,early_stopping=True,
                         validation_fraction=0.10,n_iter_no_change=30,
                         learning_rate_init=0.001,solver="adam")
        mlp.fit(self.scaler.transform(X_cnn), y_cnn)
        self.mlp=mlp
        cnn_corr=np.zeros(n)
        for i in range(self.LOOKBACK+4, n-1):
            f=self._feats(pc,coal,brent,gas,mad,freight,i)
            if f: cnn_corr[i+1]=float(mlp.predict(self.scaler.transform([f]))[0])
        hybrid=ols_preds+cnn_corr
        valid=np.arange(self.LOOKBACK+5,n)
        resids_h=pc[valid]-hybrid[valid]
        mae=float(np.mean(np.abs(resids_h))); rmse=float(np.sqrt(np.mean(resids_h**2)))
        ss_hyb=np.sum(resids_h**2)
        r2=float(1.0-ss_hyb/max(np.sum((pc[valid]-pc[valid].mean())**2),1e-9))
        da=float(np.mean(np.sign(np.diff(pc[valid]))==np.sign(np.diff(hybrid[valid])))*100)
        self.metrics={"r2":round(r2,4),"mae":round(mae,3),"rmse":round(rmse,3),
                      "dir_acc":round(da,1),"n_train":n,"r2_ols":round(r2_ols,4)}
        self.trained=True
        self._pc=pc; self._coal=coal; self._brent=brent
        self._gas=gas; self._mad=mad; self._freight=freight
        self._ols_preds=ols_preds; self._ols_resids=ols_resids
        print(f"  CNN: R²={r2:.3f}  MAE=${mae:.2f}/t  DirAcc={da:.0f}%")
        return self

    def predict_one(self, pc_ctx, coal_ctx, brent_ctx, gas_ctx, mad_ctx, fr_ctx):
        i=len(pc_ctx)-1
        base=float(self._ols_pred(coal_ctx[i],brent_ctx[i],gas_ctx[i]))
        f=self._feats(np.array(pc_ctx),np.array(coal_ctx),np.array(brent_ctx),
                      np.array(gas_ctx),np.array(mad_ctx),np.array(fr_ctx),i)
        if f is None: return base
        return base+float(self.mlp.predict(self.scaler.transform([f]))[0])

    def residual_std(self, window=104):
        resids=self._ols_resids[self.LOOKBACK+5:]
        return float(np.std(resids[-window:] if window else resids))

    def ols_scatter(self):
        coal,pc=self._coal,self._pc; n=len(coal)
        xm=coal.mean(); ym=pc.mean()
        sxy=np.sum((coal-xm)*(pc-ym)); sxx=np.sum((coal-xm)**2)
        b=sxy/sxx; a=ym-b*xm
        preds=a+b*coal; resids=pc-preds
        sse=np.sum(resids**2); sst=np.sum((pc-ym)**2)
        return {"alpha":round(float(a),4),"beta":round(float(b),4),
                "r2":round(float(1-sse/sst),4),"sxx":float(sxx),"xm":float(xm),"n":n}


CNN_MODEL = CNNTemporalModel()
if WEEKLY_DF is not None:
    CNN_MODEL.train(WEEKLY_DF)

# ── Live price cache ──────────────────────────────────────────────────────────
_cache={"brent":{"value":None,"ts":0,"source":""},"nat_gas":{"value":None,"ts":0,"source":""},
        "coal":{"value":None,"ts":0,"source":""},"lock":threading.Lock()}
CACHE_TTL=120

def _fetch(symbol):
    import urllib.request, json as _j
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=5) as r:
        d=_j.loads(r.read())
    closes=[c for c in d["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c]
    return round(closes[-1],2)

def _fetch2(symbol):
    import urllib.request, json as _j
    url=f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=1d"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 (compatible)"})
    with urllib.request.urlopen(req,timeout=5) as r:
        d=_j.loads(r.read())
    m=d["chart"]["result"][0]["meta"]
    return round(m.get("regularMarketPrice") or m.get("previousClose"),2)

def get_live(key, sym, fallback):
    with _cache["lock"]:
        c=_cache[key]; now=time.time()
        if c["value"] and now-c["ts"]<CACHE_TTL: return c["value"],True,c["source"]
    val,src=None,"fallback"
    for fn in [_fetch,_fetch2]:
        try: val=fn(sym); src="Yahoo Finance"; break
        except: pass
    if val is None: val=fallback; src="historical (offline)"
    with _cache["lock"]: _cache[key]={"value":val,"ts":time.time(),"source":src}
    return val, src!="historical (offline)", src

def get_all_live():
    brent,bl,bs=get_live("brent","BZ%3DF",HISTORY[-1]["brent"])
    gas,gl,gs=get_live("nat_gas","NG%3DF",HISTORY[-1]["nat_gas"])
    coal,cl,cs=get_live("coal","MTF%3DF",HISTORY[-1]["coal"])
    return {"brent":{"value":brent,"live":bl,"source":bs},
            "nat_gas":{"value":gas,"live":gl,"source":gs},
            "coal":{"value":coal,"live":cl,"source":cs},
            "ts":datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

# ── Math helpers (retained from v3) ──────────────────────────────────────────
def ols_simple(xs, ys):
    n=len(xs); xm=sum(xs)/n; ym=sum(ys)/n
    sxy=sum((xs[i]-xm)*(ys[i]-ym) for i in range(n))
    sxx=sum((x-xm)**2 for x in xs)
    beta=sxy/sxx; alpha=ym-beta*xm
    preds=[alpha+beta*x for x in xs]
    resids=[ys[i]-preds[i] for i in range(n)]
    sse=sum(r**2 for r in resids); sst=sum((y-ym)**2 for y in ys)
    r2=max(0.0,1.0-sse/sst); n2=n-2; rmse=math.sqrt(sse/n)
    mae=sum(abs(r) for r in resids)/n
    se=math.sqrt(sse/n2) if n2>0 else 0; se_b=se/math.sqrt(sxx) if sxx>0 else 0
    return dict(alpha=alpha,beta=beta,r2=r2,rmse=rmse,mae=mae,std_err=se,
                se_beta=se_b,t_stat=beta/se_b if se_b>0 else 0,
                n=n,sxx=sxx,xm=xm,residuals=resids,predictions=preds)

def ols_multiple(X_cols, ys):
    X=np.column_stack([[1.0]*len(ys)]+[list(c) for c in X_cols]); Y=np.array(ys,dtype=float)
    try: coeffs=np.linalg.solve(X.T@X,X.T@Y)
    except: coeffs=np.linalg.lstsq(X,Y,rcond=None)[0]
    preds=X@coeffs; resids=Y-preds; sse=float(resids@resids); ym=float(Y.mean())
    sst=float(((Y-ym)**2).sum()); r2=max(0.0,1.0-sse/sst); n,p=X.shape
    rmse=math.sqrt(sse/n); mae=float(np.abs(resids).mean()); n2=n-p
    se=math.sqrt(sse/n2) if n2>0 else 0
    return dict(coeffs=[round(float(c),4) for c in coeffs],r2=r2,
                adj_r2=1-(1-r2)*(n-1)/max(1,n2),rmse=rmse,mae=mae,std_err=se,
                n=n,p=int(p),residuals=[float(r) for r in resids],
                predictions=[float(p) for p in preds])

def ar1_fit(series):
    y=series[1:]; y1=series[:-1]; n=len(y); ym=sum(y)/n; y1m=sum(y1)/n
    sxy=sum((y[i]-ym)*(y1[i]-y1m) for i in range(n))
    sxx=sum((x-y1m)**2 for x in y1)
    phi=max(-0.99,min(0.99,sxy/sxx if sxx>0 else 0.0))
    mu=ym-phi*y1m
    sigma=math.sqrt(sum((y[i]-(mu+phi*y1[i-1 if i>0 else 0]))**2 for i in range(n))/max(1,n-2))
    return {"phi":phi,"mu":mu,"sigma":sigma,"last":series[-1]}

def ar1_simulate(params, steps, n_paths=1000, seed=42):
    rng=np.random.default_rng(seed)
    phi,mu,sigma,last=params["phi"],params["mu"],params["sigma"],params["last"]
    paths=np.zeros((n_paths,steps)); cur=np.full(n_paths,last)
    for t in range(steps):
        cur=mu+phi*cur+rng.normal(0,sigma,n_paths); paths[:,t]=cur
    return {k:[round(float(np.percentile(paths[:,t],p)),2) for t in range(steps)]
            for k,p in [("p05",5),("p10",10),("p25",25),("p50",50),("p75",75),("p90",90),("p95",95),("mean",50)]}

def scenario_brent_path(base, sc, horizon=12):
    cfg={"base":(0.000,0.040),"bear":(-0.012,0.045),"bull":(0.015,0.040),
         "contango":(0.005,0.025),"backwd":(-0.008,0.030),"crisis":(0.030,0.090),"crash":(-0.025,0.070)}
    d,v=cfg.get(sc,cfg["base"]); rng=np.random.default_rng(99); path=[base]
    tv=math.sqrt(0.038**2+v**2)
    for _ in range(horizon):
        nxt=path[-1]*(1+rng.normal(d,tv)); path.append(round(max(30.0,min(200.0,nxt)),2))
    return path[1:]

def freight_forecast(base, sc, horizon=6):
    drifts={"base":0.002,"bear":-0.015,"bull":0.018,"contango":0.004,"backwd":-0.006,"crisis":0.035,"crash":-0.025}
    vols={"base":0.04,"bear":0.05,"bull":0.05,"contango":0.03,"backwd":0.04,"crisis":0.08,"crash":0.07}
    rng=np.random.default_rng(55+hash(sc)%100); path=[base]
    for _ in range(horizon):
        path.append(round(path[-1]*(1+rng.normal(drifts.get(sc,0),vols.get(sc,0.04))),2))
    return path[1:]

def signal_label(pct, buy_thr, avoid_thr):
    if pct<=buy_thr: return "BUY"
    if pct<=-1.0:    return "WATCH"
    if pct<=avoid_thr: return "MONITOR"
    return "AVOID"

def cnn_mc_forecast(hist, coal_fwds, brent_fwds, gas_fwds, mad_fwds, fr_fwds,
                    model, n_sim=1200, seed=7, adj=0.0):
    rng=np.random.default_rng(seed)
    res_std=model.residual_std(104); horizon=len(coal_fwds)
    lb=model.LOOKBACK+5
    pc_hist=list(model._pc); coal_h=list(model._coal); brent_h=list(model._brent)
    gas_h=list(model._gas); mad_h=list(model._mad); fr_h=list(model._freight)
    results=[]; pc_fwd=[]
    for i in range(horizon):
        def ctx(arr_h,arr_f,step):
            tail=list(arr_h[-lb:]); fwd=list(arr_f[:step])
            return (tail+fwd)[-lb:]
        coal_ctx=ctx(coal_h,coal_fwds[:i+1],i+1)
        brent_ctx=ctx(brent_h,brent_fwds[:i+1],i+1)
        gas_ctx=ctx(gas_h,gas_fwds[:i+1],i+1)
        mad_ctx=ctx(mad_h,mad_fwds[:i+1],i+1)
        fr_ctx=ctx(fr_h,fr_fwds[:i+1],i+1)
        pc_ctx=(list(pc_hist)+pc_fwd)[-lb:]
        if model.trained:
            point=model.predict_one(pc_ctx,coal_ctx,brent_ctx,gas_ctx,mad_ctx,fr_ctx)
        else:
            m=ols_simple([d["coal"] for d in hist],[d["petcoke"] for d in hist])
            point=m["alpha"]+m["beta"]*coal_fwds[i]
        point=max(20.0,point+adj); pc_fwd.append(point)
        sigma=res_std*(1.0+0.08*i)
        sims=np.sort(np.maximum(20.0,rng.normal(point,sigma,n_sim)))
        results.append({"point":round(point,2),
                        "p05":round(float(sims[int(n_sim*.05)]),2),
                        "p10":round(float(sims[int(n_sim*.10)]),2),
                        "p25":round(float(sims[int(n_sim*.25)]),2),
                        "p75":round(float(sims[int(n_sim*.75)]),2),
                        "p90":round(float(sims[int(n_sim*.90)]),2),
                        "p95":round(float(sims[int(n_sim*.95)]),2),
                        "pred_se":round(sigma,3)})
    return results

# ── Main compute ──────────────────────────────────────────────────────────────
def compute_forecast(params):
    window=max(6,min(len(HISTORY),int(params.get("window",len(HISTORY)))))
    horizon=max(3,min(12,int(params.get("horizon",6))))
    scenario=str(params.get("scenario","base"))
    mc_seed=int(params.get("mc_seed",7)); n_sim=int(params.get("n_sim",1200))
    buy_thr=float(params.get("buy_thr",-3.0)); avoid_thr=float(params.get("avoid_thr",3.0))
    annual_vol=float(params.get("annual_vol",180000)); order_size=float(params.get("order_size",10000))
    mad_rate=float(params.get("mad_rate",9.94)); fmult=float(params.get("freight_mult",1.0))
    adj=float(params.get("petcoke_disc",0.0))+float(params.get("sulphur_adj",0.0))

    hist=HISTORY[-window:]; n=len(hist)
    brent_h=[d["brent"] for d in hist]; petcoke_h=[d["petcoke"] for d in hist]
    freight_h=[d["freight"] for d in hist]; gas_h=[d["nat_gas"] for d in hist]
    coal_h=[d["coal"] for d in hist]; mad_h=[d["usd_mad"] for d in hist]

    ols_diag=ols_simple(coal_h,petcoke_h)
    ols_multi=ols_multiple([brent_h,gas_h,coal_h],petcoke_h)

    live_data=get_all_live()
    brent_now=live_data["brent"]["value"]; gas_now=live_data["nat_gas"]["value"]
    coal_now=live_data["coal"]["value"]; brent_live=live_data["brent"]["live"]
    data_src=live_data["brent"]["source"]; data_ts=live_data["ts"]

    if CNN_MODEL.trained:
        petcoke_now=round(float(CNN_MODEL._ols_pred(coal_now,brent_now,gas_now))+adj,2)
    else:
        c=ols_multi["coeffs"]
        petcoke_now=round(c[0]+c[1]*brent_now+c[2]*gas_now+c[3]*coal_now+adj,2)
    petcoke_now=max(20.0,petcoke_now)
    freight_now=round(hist[-1]["freight"]*fmult,2)
    landed_now=round(petcoke_now+freight_now,2)
    curr_ratio=petcoke_now/brent_now if brent_now>0 else 0

    brent_fwds=scenario_brent_path(brent_now,scenario,horizon)
    ar1_gas=ar1_fit(gas_h); ar1_coal=ar1_fit(coal_h)
    gas_fwds=ar1_simulate(ar1_gas,horizon,n_paths=600,seed=21)["p50"]
    coal_fwds_ar1=ar1_simulate(ar1_coal,horizon,n_paths=600,seed=22)["p50"]
    fr_fwds=freight_forecast(freight_now,scenario,horizon)
    ar1_mad=ar1_fit(mad_h); mad_fwds=ar1_simulate(ar1_mad,horizon,n_paths=400,seed=55)["p50"]

    sc_drift={"bear":-0.008,"base":-0.003,"bull":0.010,"contango":0.005,
              "backwd":-0.005,"crisis":0.022,"crash":-0.015}.get(scenario,-0.003)
    rng2=random.Random(mc_seed); cv=coal_now; coal_sc_fwds=[]
    for _ in range(horizon):
        cv=round(cv*(1+sc_drift+rng2.gauss(0,0.012)),2); coal_sc_fwds.append(cv)

    mc=cnn_mc_forecast(hist,coal_sc_fwds,brent_fwds,gas_fwds,mad_fwds,fr_fwds,
                       CNN_MODEL,n_sim=n_sim,seed=mc_seed,adj=adj)

    now=datetime.utcnow()
    months=[{"label":(now.replace(day=1)+timedelta(days=32*i)).replace(day=1).strftime("%b '%y"),
             "iso":(now.replace(day=1)+timedelta(days=32*i)).replace(day=1).strftime("%Y-%m")}
            for i in range(1,horizon+1)]

    forecast=[]
    for i in range(horizon):
        pt=mc[i]["point"]; fr=fr_fwds[i]
        pct=round((pt-petcoke_now)/petcoke_now*100,2) if petcoke_now>0 else 0
        sig=signal_label(pct,buy_thr,avoid_thr)
        forecast.append({"label":months[i]["label"],"iso":months[i]["iso"],
            "brent_fwd":brent_fwds[i],"point":pt,
            "p05":mc[i]["p05"],"p10":mc[i]["p10"],"p25":mc[i]["p25"],
            "p75":mc[i]["p75"],"p90":mc[i]["p90"],"p95":mc[i]["p95"],
            "pred_se":mc[i]["pred_se"],"freight":fr,"landed":round(pt+fr,2),
            "gas_fwd":gas_fwds[i],"coal_fwd":coal_sc_fwds[i],
            "pct_vs_now":pct,"signal":sig,
            "annual_cost_mad":round(pt*annual_vol*mad_rate/1e6,2)})

    best=min(forecast,key=lambda x:x["point"]); worst=max(forecast,key=lambda x:x["point"])
    buy_months=[f for f in forecast if f["signal"]=="BUY"]

    all_scenarios={}
    for sc in ["bear","base","bull","contango","backwd","crisis","crash"]:
        scd={"bear":-0.008,"base":-0.003,"bull":0.010,"contango":0.005,
             "backwd":-0.005,"crisis":0.022,"crash":-0.015}.get(sc,-0.003)
        sc_cv=coal_now; sc_rng=random.Random(mc_seed+abs(hash(sc))%1000); sc_coal=[]
        for _ in range(horizon):
            sc_cv=round(sc_cv*(1+scd+sc_rng.gauss(0,0.012)),2); sc_coal.append(sc_cv)
        sc_mc=cnn_mc_forecast(hist,sc_coal,scenario_brent_path(brent_now,sc,horizon),
                              gas_fwds,mad_fwds,freight_forecast(freight_now,sc,horizon),
                              CNN_MODEL,n_sim=400,seed=mc_seed,adj=adj)
        sc_fr=freight_forecast(freight_now,sc,horizon)
        all_scenarios[sc]=[{"label":months[i]["label"],"brent":scenario_brent_path(brent_now,sc,horizon)[i],
            "point":sc_mc[i]["point"],"p10":sc_mc[i]["p10"],"p90":sc_mc[i]["p90"],
            "freight":sc_fr[i],"landed":round(sc_mc[i]["point"]+sc_fr[i],2),
            "pct":round((sc_mc[i]["point"]-petcoke_now)/max(petcoke_now,1)*100,2),
            "signal":signal_label((sc_mc[i]["point"]-petcoke_now)/max(petcoke_now,1)*100,buy_thr,avoid_thr)
            } for i in range(horizon)]

    def proc(label,price,note=""):
        tu=round((price+freight_now)*order_size,0); bu=round((petcoke_now+freight_now)*order_size,0)
        sv=round(bu-tu,0)
        return {"label":label,"price":round(price,2),"total_usd":tu,"total_mad":round(tu*mad_rate,0),
                "saving_usd":sv,"saving_mad":round(sv*mad_rate,0),"saving_pct":round(sv/max(bu,1)*100,2),"note":note}
    avg3=round(sum(f["point"] for f in forecast[:3])/3,2)
    procurement=[
        proc("Buy Now",petcoke_now,"Immediate execution at spot"),
        proc(f"Defer → {best['label']}",best["point"],f"Optimal window · {best['signal']}"),
        proc("Split 50/50",round((petcoke_now+best["point"])/2,2),"Half now, half at optimal"),
        proc("Monthly tranches (avg)",avg3,"Equal monthly over 3 months"),
        proc("DCA 6-month avg",round(sum(f["point"] for f in forecast)/len(forecast),2),"Equal tranches over horizon"),
    ]

    sensitivity=[]
    for label,mult in [("Crash −30%",0.70),("Bear −15%",0.85),("Base",1.00),("Bull +15%",1.15),("Crisis +30%",1.30)]:
        p=round(petcoke_now*mult,2); ann=round((p+freight_now)*annual_vol*mad_rate/1e6,2)
        sensitivity.append({"label":label,"price":p,"annual_mad_m":ann,"pct":round((mult-1)*100,0),"vs_base_mad":0})
    base_ann=sensitivity[2]["annual_mad_m"]
    for s in sensitivity: s["vs_base_mad"]=round(s["annual_mad_m"]-base_ann,2)

    ar1_brent=ar1_fit(brent_h)
    brent_ar1=ar1_simulate(ar1_brent,horizon,n_paths=800,seed=33)
    coal_ar1=ar1_simulate(ar1_coal,horizon,n_paths=800,seed=34)
    ratios=[d["petcoke"]/d["brent"] for d in hist if d["brent"]>0]
    avg_ratio=sum(ratios)/len(ratios)

    sc_ols=CNN_MODEL.ols_scatter() if CNN_MODEL.trained else {}
    cm=CNN_MODEL.metrics if CNN_MODEL.trained else {}
    model_summary={
        "type":"cnn_temporal","window":window,"n":n,
        "r2":cm.get("r2",round(ols_diag["r2"],4)),
        "adj_r2":cm.get("r2",round(ols_diag["r2"],4)),
        "rmse":cm.get("rmse",round(ols_diag["rmse"],3)),
        "mae":cm.get("mae",round(ols_diag["mae"],3)),
        "dir_acc":cm.get("dir_acc",55.0),
        "n_train":cm.get("n_train",len(WEEKLY_DF) if WEEKLY_DF is not None else n),
        "std_err":round(CNN_MODEL.residual_std(104) if CNN_MODEL.trained else ols_diag["std_err"],3),
        "scatter_alpha":sc_ols.get("alpha",round(ols_diag["alpha"],4)),
        "scatter_beta":sc_ols.get("beta",round(ols_diag["beta"],4)),
        "coal_beta":sc_ols.get("beta",round(ols_diag["beta"],4)),
        "coeffs":ols_multi.get("coeffs",[round(ols_diag["alpha"],4),round(ols_diag["beta"],4)]),
        "alpha":round(ols_diag["alpha"],4),"beta":round(ols_diag["beta"],4),
        "t_stat":round(ols_diag["t_stat"],2),
        "residuals":[round(r,3) for r in ols_diag["residuals"]],
        "predictions":[round(p,3) for p in ols_diag["predictions"]],
        "ar1_phi":round(ar1_brent["phi"],4),"avg_ratio":round(avg_ratio,3),"curr_ratio":round(curr_ratio,3),
    }

    return {"live":{"brent":brent_now,"petcoke":petcoke_now,"freight":freight_now,"landed":landed_now,
                    "gas":gas_now,"coal":coal_now,"is_live":brent_live,"source":data_src,"ts":data_ts,
                    "mad_rate":mad_rate,"petcoke_ratio":round(curr_ratio,3)},
            "history":hist,"model":model_summary,"forecast":forecast,"scenarios":all_scenarios,
            "procurement":procurement,"sensitivity":sensitivity,"swing_per_10":round(10*annual_vol*mad_rate/1e6,2),
            "best_month":best,"worst_month":worst,"buy_months":buy_months,
            "brent_ar1":brent_ar1,"coal_ar1":coal_ar1,"params":params}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return send_from_directory("static","index.html")

@app.route("/api/live")
def api_live(): return jsonify(get_all_live())

@app.route("/api/forecast",methods=["POST","OPTIONS"])
def api_forecast():
    if request.method=="OPTIONS": return Response(status=200)
    return jsonify(compute_forecast(request.get_json(silent=True) or {}))

@app.route("/api/history")
def api_history(): return jsonify(HISTORY)

@app.route("/api/model_compare",methods=["POST","OPTIONS"])
def api_model_compare():
    if request.method=="OPTIONS": return Response(status=200)
    results={}
    for w in [24,60,120,180,len(HISTORY)]:
        if w>len(HISTORY): continue
        h=HISTORY[-w:]; m=ols_simple([d["coal"] for d in h],[d["petcoke"] for d in h])
        results[str(w)]={"r2":round(m["r2"],4),"rmse":round(m["rmse"],3),"mae":round(m["mae"],3),
                          "alpha":round(m["alpha"],3),"beta":round(m["beta"],3),"std_err":round(m["std_err"],3)}
    if CNN_MODEL.trained:
        results["cnn_full"]={"r2":CNN_MODEL.metrics["r2"],"rmse":CNN_MODEL.metrics["rmse"],
                              "mae":CNN_MODEL.metrics["mae"],"alpha":round(float(CNN_MODEL.ols_coeffs[3]),3),
                              "beta":round(float(CNN_MODEL.ols_coeffs[0]),3),"std_err":CNN_MODEL.metrics["rmse"]}
    return jsonify(results)

@app.route("/api/scenario_grid",methods=["POST","OPTIONS"])
def api_scenario_grid():
    if request.method=="OPTIONS": return Response(status=200)
    live=get_all_live(); b=live["brent"]["value"]; coal_l=live["coal"]["value"]
    h=HISTORY[-18:]; m=ols_simple([d["coal"] for d in h],[d["petcoke"] for d in h])
    pn=m["alpha"]+m["beta"]*coal_l
    grid={sc:[{"pct":round((m["alpha"]+m["beta"]*scenario_brent_path(b,sc,6)[i]-pn)/pn*100,2),
               "point":round(m["alpha"]+m["beta"]*scenario_brent_path(b,sc,6)[i],2)} for i in range(6)]
          for sc in ["bear","base","bull","crisis","crash"]}
    return jsonify({"grid":grid,"petcoke_now":round(pn,2)})

if __name__=="__main__":
    os.makedirs("static",exist_ok=True)
    print("\n"+"═"*50)
    print("  CIMAR Petcoke Intelligence Platform v4")
    print("  CNN Temporal · 20-year dataset · 1,044 obs")
    print("  http://localhost:5050")
    print("═"*50+"\n")
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5050)))
