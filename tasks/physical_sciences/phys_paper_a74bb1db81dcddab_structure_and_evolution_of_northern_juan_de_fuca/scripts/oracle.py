from __future__ import annotations
import csv, json, math
from pathlib import Path

PHASES = ("direct", "refracted_2", "refracted_3")

def read_picks(path: Path):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows or set(rows[0]) != {"phase","offset_km","time_s","sigma_ms"}:
        raise ValueError("bad pick schema")
    return [{"phase":r["phase"],"x":float(r["offset_km"]),"t":float(r["time_s"]),"sigma":float(r["sigma_ms"])} for r in rows]

def _line(rows):
    # Weighted normal equations for t=a+b*x; this is intentionally independent
    # of any optimizer used by a candidate.
    sw=sx=sy=sxx=sxy=0.0
    for r in rows:
        w=1.0/(r["sigma"]**2); x=r["x"]; y=r["t"]
        sw+=w; sx+=w*x; sy+=w*y; sxx+=w*x*x; sxy+=w*x*y
    den=sw*sxx-sx*sx
    b=(sw*sxy-sx*sy)/den; a=(sy-b*sx)/sw
    return a,b

def fit(rows):
    lines={p:_line([r for r in rows if r["phase"]==p]) for p in PHASES}
    a1,b1=lines["direct"]; a2,b2=lines["refracted_2"]; a3,b3=lines["refracted_3"]
    v1,v2,v3=1/b1,1/b2,1/b3
    h1=a2/(2*math.sqrt(1/v1**2-1/v2**2))
    h2=(a3/2-h1*math.sqrt(1/v1**2-1/v3**2))/math.sqrt(1/v2**2-1/v3**2)
    return {"v1_km_s":v1,"v2_km_s":v2,"v3_km_s":v3,"h1_km":h1,"h2_km":h2}

def predict(model, phase, x):
    v1,v2,v3=[model[k] for k in ("v1_km_s","v2_km_s","v3_km_s")]; h1,h2=model["h1_km"],model["h2_km"]
    if phase=="direct": return x/v1
    if phase=="refracted_2": return 2*h1*math.sqrt(1/v1**2-1/v2**2)+x/v2
    if phase=="refracted_3": return 2*(h1*math.sqrt(1/v1**2-1/v3**2)+h2*math.sqrt(1/v2**2-1/v3**2))+x/v3
    raise ValueError(phase)

def make_reference(visible: Path, hidden: Path, out: Path):
    rows=read_picks(visible); hrows=read_picks(hidden); model=fit(rows)
    pv=[predict(model,r["phase"],r["x"]) for r in rows]
    ph=[predict(model,r["phase"],r["x"]) for r in hrows]
    resid=[(p-r["t"])*1000 for p,r in zip(pv,rows)]
    obj=sum((x/r["sigma"])**2 for x,r in zip(resid,rows))
    ref={"model":model,"visible_predictions":pv,"hidden_predictions":ph,
         "rms_ms":math.sqrt(sum(x*x for x in resid)/len(resid)),
         "reduced_chi2":obj/(len(rows)-5),"objective":obj}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(ref,sort_keys=True,indent=2)+"\n")
    return ref

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("visible",type=Path); ap.add_argument("hidden",type=Path); ap.add_argument("out",type=Path)
    a=ap.parse_args(); make_reference(a.visible,a.hidden,a.out)
