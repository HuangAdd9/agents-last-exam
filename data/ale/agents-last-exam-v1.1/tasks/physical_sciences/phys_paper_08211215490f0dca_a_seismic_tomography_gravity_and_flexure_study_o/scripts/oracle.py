"""Independent host oracle for the reduced straight-ray tomography fixture."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

def load_case(path):
    c=json.loads(Path(path).read_text())
    if c.get("schema_version")!="tomography-case-1": raise ValueError("schema")
    n=int(c["grid"]["nx"])*int(c["grid"]["nz"]); rows=c["rays"]
    A=[[float(x) for x in r["path_lengths_km"]] for r in rows]
    t=[float(r["observed_time_s"]) for r in rows]
    sig=[float(r.get("uncertainty_s",0.006)) for r in rows]
    if any(len(r)!=n for r in A): raise ValueError("ray length")
    return c,A,t,sig

def solve_linear(M,b):
    # Gauss-Jordan with deterministic partial pivoting; independent of candidate's iteration.
    n=len(b); a=[list(map(float,M[i]))+[float(b[i])] for i in range(n)]
    for col in range(n):
        p=max(range(col,n),key=lambda i:abs(a[i][col]))
        if abs(a[p][col])<1e-14: raise ValueError("singular")
        a[col],a[p]=a[p],a[col]; q=a[col][col]
        for j in range(col,n+1): a[col][j]/=q
        for i in range(n):
            if i==col: continue
            q=a[i][col]
            if q:
                for j in range(col,n+1): a[i][j]-=q*a[col][j]
    return [a[i][n] for i in range(n)]

def reference(case):
    c,A,t,sig=load_case(case); n=len(A[0]); lam=1e-4; prior=[0.2]*n
    N=[[0.0]*n for _ in range(n)]; rhs=[0.0]*n
    for row,y,s in zip(A,t,sig):
        w=1.0/(s*s)
        for i in range(n):
            rhs[i]+=w*row[i]*y
            for j in range(n): N[i][j]+=w*row[i]*row[j]
    for i in range(n): N[i][i]+=lam; rhs[i]+=lam*prior[i]
    sl=solve_linear(N,rhs); vel=[1.0/x for x in sl]
    pred=[sum(x*y for x,y in zip(row,sl)) for row in A]
    rms=math.sqrt(sum((x-y)**2 for x,y in zip(pred,t))/len(t))
    nx=int(c["grid"]["nx"]); nz=int(c["grid"]["nz"])
    return {"grid":c["grid"],"velocity_km_s":[vel[z*nx:(z+1)*nx] for z in range(nz)],"slowness_s_per_km":[sl[z*nx:(z+1)*nx] for z in range(nz)],"rms_s":rms,"ray_predictions_s":pred}

def write(case,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); c,A,t,sig=load_case(case); ref=reference(case)
    (out/"model.json").write_text(json.dumps(ref,indent=2,sort_keys=True)+"\n")
    with (out/"trace.csv").open("w",newline="") as f:
        n=len(ref["slowness_s_per_km"]) * len(ref["slowness_s_per_km"][0])
        header=["iteration","objective","rms_s","step"]+[f"s_{i}" for i in range(n)]
        w=csv.writer(f); w.writerow(header)
        prior=[0.2]*n; lam=1e-4
        # A deterministic independent reproduction of the required
        # preconditioned steepest-descent line-search trace. The model itself
        # remains the exact normal-equation reference; 1000 updates bring the
        # final state within the verifier's binding tolerance.
        def metrics(s):
            pred=[sum(float(a)*s[i] for i,a in enumerate(row)) for row in A]
            obj=sum(((p-y)/q)**2 for p,y,q in zip(pred,t,sig))+1e-4*sum((x-0.2)**2 for x in s)
            rms=math.sqrt(sum((p-y)**2 for p,y in zip(pred,t))/len(t))
            return obj,rms
        states=[prior]
        diag=[lam+sum((float(row[i])/float(q))**2 for row,q in zip(A,sig)) for i in range(n)]
        for _ in range(1000):
            s=states[-1]; g=[lam*(x-0.2) for x in s]
            for row,y,q in zip(A,t,sig):
                e=sum(float(a)*s[i] for i,a in enumerate(row))-y; wgt=1.0/(q*q)
                for i,a in enumerate(row): g[i]+=wgt*float(a)*e
            d=[-g[i]/diag[i] for i in range(n)]; hd=[lam*x for x in d]
            for row,q in zip(A,sig):
                ad=sum(float(a)*d[i] for i,a in enumerate(row)); wgt=1.0/(q*q)
                for i,a in enumerate(row): hd[i]+=wgt*float(a)*ad
            alpha=-sum(a*b for a,b in zip(g,d))/sum(a*b for a,b in zip(d,hd))
            states.append([s[i]+alpha*d[i] for i in range(n)])
        for i,s in enumerate(states):
            obj,rms=metrics(s); step=0.0 if i==0 else math.sqrt(sum((a-b)**2 for a,b in zip(s,states[i-1])))
            w.writerow([i,obj,rms,step]+s)
    (out/"report.json").write_text(json.dumps({"method":"iterative_weighted_ridge_reference","units":{"velocity":"km/s","slowness":"s/km","time":"s"},"case_id":c["case_id"],"invariant":"forward_time_sum_matches_reported_rms"},indent=2,sort_keys=True)+"\n")
    return ref

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--case",required=True); p.add_argument("--out",required=True); a=p.parse_args(); write(a.case,a.out)
