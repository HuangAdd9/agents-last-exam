#!/usr/bin/env python3
"""Independent deterministic oracle for the benchmark-authored two-band model."""
import csv, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np

def periodic_delta(x, y):
    d = abs(float(x) - float(y))
    return min(d, 1.0 - d)

def compute(case):
    g = case["grid"]; n = int(g["n"])
    if n < 2 or g.get("includes_endpoint") is not False: raise ValueError("grid contract")
    v, c, gw = case["valence"], case["conduction"], case["gw_self_energy"]
    valley = np.asarray(c["valley"], dtype=float)
    rows=[]
    for ix in range(n):
      for iy in range(n):
       for iz in range(n):
        k=np.array([ix,iy,iz],dtype=float)/n
        s=np.sin(np.pi*k); q=float(np.dot(s,s))
        ev=float(v["e0"]-v["curvature"]*q+v["warping"]*np.prod(np.sin(2*np.pi*k)))
        d=np.array([periodic_delta(k[j],valley[j]) for j in range(3)])
        ec=float(c["e0"]+np.dot(np.asarray(c["curvature"],dtype=float),d*d)+c["warping"]*np.prod(np.cos(2*np.pi*k)))
        r=float(np.linalg.norm(d)); f=math.exp(-r/float(gw["screening_length"]))
        dv=float(gw["valence_shift"]-gw["nonlocal_strength"]*(1-f))
        dc=float(gw["conduction_shift"]+gw["nonlocal_strength"]*(1-f))
        rows.append((k,ev,ec,ev+dv,ec+dc))
    iv=max(range(len(rows)), key=lambda i: rows[i][1]); ic=min(range(len(rows)), key=lambda i: rows[i][2])
    iqv=max(range(len(rows)), key=lambda i: rows[i][3]); iqc=min(range(len(rows)), key=lambda i: rows[i][4])
    def extrema(i, band):
      return {"k": [float(x) for x in rows[i][0]], "energy_eV": float(rows[i][band])}
    dft_gap=float(rows[ic][2]-rows[iv][1]); qp_gap=float(rows[iqc][4]-rows[iqv][3])
    # path with shared endpoints emitted once
    p=case["path"]; pts=np.asarray(p["points"],dtype=float); seg=int(p["segments"])
    path=[]; dist=0.0; idx=0
    for si in range(len(pts)-1):
      a,b=pts[si],pts[si+1]
      for j in range(seg+1):
        if si and j==0: continue
        t=j/seg; k=(1-t)*a+t*b
        s=np.sin(np.pi*k); q=float(np.dot(s,s))
        ev=float(v["e0"]-v["curvature"]*q+v["warping"]*np.prod(np.sin(2*np.pi*k)))
        d=np.array([periodic_delta(k[z],valley[z]) for z in range(3)])
        ec=float(c["e0"]+np.dot(np.asarray(c["curvature"],dtype=float),d*d)+c["warping"]*np.prod(np.cos(2*np.pi*k)))
        r=float(np.linalg.norm(d)); f=math.exp(-r/float(gw["screening_length"]))
        evq=ev+float(gw["valence_shift"]-gw["nonlocal_strength"]*(1-f)); ecq=ec+float(gw["conduction_shift"]+gw["nonlocal_strength"]*(1-f))
        if path: dist += float(np.linalg.norm(k-path[-1]["k"]))
        path.append({"segment":si,"index":j,"distance":dist,"ev_dft":ev,"ec_dft":ec,"ev_qp":evq,"ec_qp":ecq,"k":k.tolist()})
    direct_gamma=float(path[0]["ec_qp"]-path[0]["ev_qp"])
    grid_blob="\n".join("%.12f %.12f %.12f %.12f %.12f %.12f"%(r[0][0],r[0][1],r[0][2],r[1],r[2],r[4]) for r in sorted(rows,key=lambda r: tuple(r[0])))
    mf={"units":"eV","grid_shape":[n,n,n],"vbm":extrema(iv,1),"cbm":extrema(ic,2),"indirect_gap_eV":dft_gap,"grid_sha256":hashlib.sha256(grid_blob.encode()).hexdigest()}
    qp={"units":"eV","vbm":extrema(iqv,3),"cbm":extrema(iqc,4),"indirect_gap_eV":qp_gap,"gap_opening_eV":qp_gap-dft_gap,"direct_gamma_gap_eV":direct_gamma,"valley_declared":[float(x) for x in valley]}
    diag={"units":"eV","grid_shape":[n,n,n],"structure_atom_count":2,"vbm_at_gamma":float(np.linalg.norm(rows[iv][0])) <= 1/n+1e-12,"cbm_near_declared_valley":float(np.linalg.norm(np.array([periodic_delta(rows[ic][0][j],valley[j]) for j in range(3)]))) <= math.sqrt(3)/n+1e-12,"qp_gap_opening_positive":qp_gap>dft_gap,"direct_gamma_gap_eV":direct_gamma,"path_rows":len(path)}
    return {"mean_field":mf,"quasiparticle":qp,"diagnostics":diag,"path":path}

def write(case, outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); r=compute(case)
    (out/"mean_field.json").write_text(json.dumps(r["mean_field"],sort_keys=True,indent=2)+"\n")
    (out/"quasiparticle.json").write_text(json.dumps(r["quasiparticle"],sort_keys=True,indent=2)+"\n")
    (out/"diagnostics.json").write_text(json.dumps(r["diagnostics"],sort_keys=True,indent=2)+"\n")
    with (out/"band_path.csv").open("w",newline="") as f:
      w=csv.writer(f); w.writerow(["segment","index","distance","ev_dft","ec_dft","ev_qp","ec_qp"])
      for x in r["path"]: w.writerow([x[k] for k in ["segment","index","distance","ev_dft","ec_dft","ev_qp","ec_qp"]])
    return r
if __name__=="__main__":
    if len(sys.argv)!=3: raise SystemExit("usage: oracle.py CASE.json OUTDIR")
    write(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])
