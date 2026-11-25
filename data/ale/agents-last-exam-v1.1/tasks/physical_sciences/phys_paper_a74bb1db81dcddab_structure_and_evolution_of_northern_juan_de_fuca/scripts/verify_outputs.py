from __future__ import annotations
import csv, json, math
from pathlib import Path
try:
    from .oracle import read_picks, predict
except ImportError:
    from oracle import read_picks, predict

REQUIRED=("model.json","predictions.csv","inversion_trace.csv","diagnostics.json")

def _finite(x): return isinstance(x,(int,float)) and math.isfinite(float(x))

def _load(out:Path):
    if any(not (out/n).is_file() for n in REQUIRED): return None
    try:
        m=json.loads((out/"model.json").read_text()); d=json.loads((out/"diagnostics.json").read_text())
        with (out/"predictions.csv").open(newline="") as f: p=list(csv.DictReader(f))
        with (out/"inversion_trace.csv").open(newline="") as f: tr=list(csv.DictReader(f))
    except Exception: return None
    if set(m)!={"v1_km_s","v2_km_s","v3_km_s","h1_km","h2_km"}: return None
    if any(not _finite(m[k]) for k in m): return None
    if set(d)!={"rms_ms","reduced_chi2","n_picks","invariant_ok"}: return None
    if any(not _finite(d[k]) for k in ("rms_ms","reduced_chi2")) or not isinstance(d["n_picks"],int) or not isinstance(d["invariant_ok"],bool): return None
    if (set(p[0]) if p else set()) != {"phase","offset_km","predicted_time_s","residual_ms"}: return None
    if len(tr)<3 or (set(tr[0]) if tr else set()) != {"iteration","objective"}: return None
    try:
        pp=[(r["phase"],float(r["offset_km"]),float(r["predicted_time_s"]),float(r["residual_ms"])) for r in p]
        tt=[(int(r["iteration"]),float(r["objective"])) for r in tr]
    except Exception:return None
    if any(not all(math.isfinite(v) for v in row[1:]) for row in pp): return None
    if any(not math.isfinite(o) or o<0 for _,o in tt) or any(tt[i+1][1]>tt[i][1]+1e-9 for i in range(len(tt)-1)): return None
    return m,d,pp,tt

def score_submission(out:Path, visible:Path, hidden:Path, ref:Path):
    loaded=_load(out)
    if loaded is None:return {"score":0.0,"components":{},"reason":"schema"}
    m,d,p,tr=loaded
    rows=read_picks(visible); hrows=read_picks(hidden); rr=json.loads(ref.read_text())
    if len(p)!=len(rows) or d["n_picks"]!=len(rows): return {"score":0.0,"components":{},"reason":"count"}
    # strict row order and units
    order_ok=all(a[0]==r["phase"] and abs(a[1]-r["x"])<1e-9 for a,r in zip(p,rows))
    if not order_ok:return {"score":0.0,"components":{},"reason":"order"}
    model_err=max(abs(m[k]-rr["model"][k])/abs(rr["model"][k]) for k in m)
    c_model=max(0.0,1.0-model_err/0.01)
    pred_err=max(abs(a[2]-b) for a,b in zip(p,rr["visible_predictions"]))
    c_vis=max(0.0,1.0-pred_err/0.008)
    # Independently recomputed hidden observable: hidden case is never visible to the agent.
    hp=[a for a in hrows]; hidden_err=max(abs((rr["hidden_predictions"][i])-rr["hidden_predictions"][i]) for i in range(len(hp))) if False else 0.0
    # Candidate provides hidden predictions in diagnostics.json as an optional field? No: require
    # model to predict them, so recompute directly from the submitted model.
    try:
        hidden_err=max(abs(predict(m,r["phase"],r["x"])-r["t"]) for r in hp)
    except (ValueError, OverflowError, ZeroDivisionError):
        hidden_err=float("inf")
    c_hidden=1.0 if hidden_err<=1e-8 else max(0.0,1.0-hidden_err/0.010)
    inv=(m["v1_km_s"]>0 and m["v1_km_s"]<m["v2_km_s"]<m["v3_km_s"] and m["h1_km"]>0 and m["h2_km"]>0 and d["invariant_ok"])
    c_inv=1.0 if inv else 0.0
    rms_ref=rr["rms_ms"]; c_diag=max(0.0,1.0-abs(d["rms_ms"]-rms_ref)/0.5)
    trace_ok=len(tr)>=3 and tr[0][0]==0 and all(tr[i][0]==tr[i-1][0]+1 for i in range(1,len(tr)))
    c_trace=1.0 if trace_ok else 0.0
    comps={"model":c_model,"visible":c_vis,"hidden":c_hidden,"trace":c_trace,"invariant_and_diagnostics":c_inv*c_diag}
    score=0.35*c_model+0.20*c_vis+0.25*c_hidden+0.10*c_trace+0.10*c_inv*c_diag
    if c_hidden<0.8: score=min(score,0.59)
    return {"score":float(score),"components":comps,"hidden_max_error_s":hidden_err}
