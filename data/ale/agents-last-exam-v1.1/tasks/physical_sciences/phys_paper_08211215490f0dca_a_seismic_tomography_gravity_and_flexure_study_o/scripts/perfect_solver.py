from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

def segment_lengths(p0,p1,nx,nz,cw,ch):
 x0,z0=map(float,p0); x1,z1=map(float,p1); dx,dz=x1-x0,z1-z0; total=math.hypot(dx,dz); out=[0.0]*(nx*nz)
 if total<1e-12:return out
 cuts=[0.0,1.0]
 for k in range(1,nx):
  if abs(dx)>1e-14:
   q=(k*cw-x0)/dx
   if 1e-10<q<1-1e-10:cuts.append(q)
 for k in range(1,nz):
  if abs(dz)>1e-14:
   q=(k*ch-z0)/dz
   if 1e-10<q<1-1e-10:cuts.append(q)
 cuts=sorted(set(round(q,14) for q in cuts))
 for a,b in zip(cuts,cuts[1:]):
  if b-a<=1e-14:continue
  q=(a+b)*0.5; x=x0+q*dx; z=z0+q*dz
  ix=min(nx-1,max(0,int(math.floor(x/cw)))); iz=min(nz-1,max(0,int(math.floor(z/ch))))
  out[iz*nx+ix]+=total*(b-a)
 return out

def solve(case,out):
 c=json.loads(Path(case).read_text()); rays=c["rays"]; nx=int(c["grid"]["nx"]); nz=int(c["grid"]["nz"]); cw=float(c["grid"]["cell_width_km"]); ch=float(c["grid"]["cell_height_km"]); n=nx*nz; x=[0.2]*n; lam=1e-4
 # Derive every sensitivity row from endpoints; supplied rows are only an
 # optional audit field and are never consumed by the inversion.
 A=[segment_lengths(r["source_xy_km"],r["receiver_xy_km"],nx,nz,cw,ch) for r in rays]; t=[r["observed_time_s"] for r in rays]; sig=[r.get("uncertainty_s",0.006) for r in rays]
 diag=[lam+sum((float(row[i])/float(s))**2 for row,s in zip(A,sig)) for i in range(n)]
 trace_states=[x[:]]
 for it in range(1000):
  g=[lam*(q-0.2) for q in x]
  for row,y,s in zip(A,t,sig):
   e=sum(float(a)*x[i] for i,a in enumerate(row))-float(y); w=1.0/(float(s)**2)
   for i,a in enumerate(row): g[i]+=w*float(a)*e
  direction=[-g[i]/diag[i] for i in range(n)]
  hd=[lam*direction[i] for i in range(n)]
  for row,s in zip(A,sig):
   ad=sum(float(a)*direction[i] for i,a in enumerate(row)); w=1.0/(float(s)**2)
   for i,a in enumerate(row): hd[i]+=w*float(a)*ad
  alpha=-sum(a*b for a,b in zip(g,direction))/sum(a*b for a,b in zip(direction,hd))
  x=[x[i]+alpha*direction[i] for i in range(n)]
  trace_states.append(x[:])
 pred=[sum(float(a)*x[i] for i,a in enumerate(row)) for row in A]; rms=math.sqrt(sum((p-float(y))**2 for p,y in zip(pred,t))/len(t)); vel=[1.0/q for q in x]
 # Recompute every recorded row from its stored intermediate state.  The
 # verifier deliberately does not trust scalar progress values.
 trace=[]
 for j,state in enumerate(trace_states):
  pp=[sum(float(a)*state[i] for i,a in enumerate(row)) for row in A]
  rr=math.sqrt(sum((p-float(y))**2 for p,y in zip(pp,t))/len(t))
  oo=sum(((sum(float(a)*state[i] for i,a in enumerate(row))-float(y))/float(s))**2 for row,y,s in zip(A,t,sig))+lam*sum((q-0.2)**2 for q in state)
  st=0.0 if j==0 else math.sqrt(sum((a-b)**2 for a,b in zip(state,trace_states[j-1])))
  trace.append((j,oo,rr,st))
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 (out/"model.json").write_text(json.dumps({"grid":c["grid"],"velocity_km_s":[vel[z*nx:(z+1)*nx] for z in range(nz)],"slowness_s_per_km":[x[z*nx:(z+1)*nx] for z in range(nz)],"rms_s":rms,"ray_predictions_s":pred},indent=2,sort_keys=True)+"\n")
 with (out/"trace.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["iteration","objective","rms_s","step"]+[f"s_{i}" for i in range(n)])
  for row,state in zip(trace,trace_states): w.writerow(list(row)+state)
 (out/"report.json").write_text(json.dumps({"method":"iterative_preconditioned_steepest_descent_exact_line_search","units":{"velocity":"km/s","slowness":"s/km","time":"s"},"case_id":c["case_id"],"invariant":"forward_time_sum_matches_reported_rms"},indent=2,sort_keys=True)+"\n")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True);a=p.parse_args();solve(a.input,a.output)
