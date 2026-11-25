from __future__ import annotations
import argparse,ast,json,math,subprocess,sys,tempfile
from pathlib import Path
BANNED={'scipy','gillespy2','stochpy','copasi','stochkit','biosimulator'}
def num(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def close(a,b,tol): return num(a) and num(b) and abs(float(a)-float(b))<=tol
def load(p):
    try:
        x=json.loads(Path(p).read_text()); return x if isinstance(x,dict) else None
    except Exception:return None
def compare(actual,ref):
    pts=[]; reasons=[]
    if not isinstance(actual,dict): return 0.,['results.json must be an object']
    v=actual.get('validation',{});rv=ref.get('validation',{})
    pts += [0.08 if v.get('events')==rv.get('events') else 0,0.05 if close(v.get('mean_population'),rv.get('mean_population'),0.15) else 0,0.04 if close(v.get('std_population'),rv.get('std_population'),0.15) else 0]
    if len(actual.get('trajectories',[]))!=len(ref.get('trajectories',[])): reasons.append('trajectory count mismatch')
    else:
      good=0; total=0
      for a,b in zip(actual['trajectories'],ref['trajectories']):
        for k in ('mean_A','mean_B','mean_C','std_A','std_B','std_C'): total+=1;good+=close(a.get(k),b.get(k),0.15)
        for k in ('fraction_A','fraction_B','fraction_C','unassigned_fraction'): total+=1;good+=close(a.get(k),b.get(k),0.08)
        fs=[a.get(k) for k in ('fraction_A','fraction_B','fraction_C','unassigned_fraction')]
        if all(num(z) and 0<=z<=1 for z in fs) and abs(sum(fs)-1)<1e-9: good+=1
        else: reasons.append('fraction partition invalid')
      pts.append(0.28 if good>=0.9*total else 0)
    aa,rr=actual.get('alpha_scan'),ref.get('alpha_scan')
    if isinstance(aa,list) and len(aa)==len(rr):
      ok=True
      for a,b in zip(aa,rr):
       ok &= close(a.get('alpha'),b.get('alpha'),1e-9)
       for k in ('mean_A','mean_B','mean_C','fraction_A','fraction_B','fraction_C','unassigned_fraction'): ok &= close(a.get(k),b.get(k),0.15 if 'mean' in k else 0.08)
      pts.append(0.20 if ok else 0); reasons += [] if ok else ['alpha scan mismatch']
    else: reasons.append('alpha scan shape mismatch')
    ta,tr=actual.get('tau_comparison'),ref.get('tau_comparison')
    ok=isinstance(ta,list) and len(ta)==len(tr)
    if ok:
      for a,b in zip(ta,tr):
       ok &= close(a.get('alpha'),b.get('alpha'),1e-9) and close(a.get('exact_mean_A'),b.get('exact_mean_A'),0.15) and close(a.get('tauleap_mean_A'),b.get('tauleap_mean_A'),0.15) and close(a.get('relative_error_A'),b.get('relative_error_A'),0.12) and isinstance(a.get('leap_count'),int)
    pts.append(0.15 if ok else 0); 
    if not ok: reasons.append('tau comparison mismatch')
    inv=actual.get('invariants',{}); ok=close(inv.get('max_fraction_sum_error'),0.0,1e-9) and inv.get('all_states_nonnegative') is True
    pts.append(0.10 if ok else 0); 
    if not ok: reasons.append('invariant failed')
    return sum(pts),reasons
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);ap.add_argument('--reference-dir',required=True);a=ap.parse_args();o=Path(a.output_dir);r=Path(a.reference_dir); reasons=[]
 out=load(o/'results.json'); ref=load(r/'results.json')
 if out is None or ref is None: print(json.dumps({'score':0.0,'reasons':['missing/malformed results']}));return
 s,rs=compare(out,ref);reasons+=rs
 try:
  lines=(o/'trajectory.csv').read_text().splitlines(); header=lines[0].split(',') if lines else []
  if header==['event','time','A','B','C','basin'] and len(lines)==int(ref['validation']['events'])+1: s+=0.05
  else: reasons.append('trajectory.csv schema/count mismatch')
 except Exception: reasons.append('trajectory.csv missing')
 src=o/'gillespie_solver.py'
 try:
  tree=ast.parse(src.read_text()); imports={n.name.split('.')[0].lower() for n in ast.walk(tree) if isinstance(n,ast.Import) for n in n.names};imports|={n.module.split('.')[0].lower() for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module}
  if imports&BANNED: s=0.;reasons.append('banned import')
  text=src.read_text(); s+=0.05 if 'default_rng' in text and 'tau' in text and 'propens' in text else 0.;
 except Exception:s=0.;reasons.append('solver source missing or unparsable')
 # hidden rerun is a separate independent gate
 hidden_cfg=r/'hidden_case/model_config.json'; hidden_ref=r/'hidden_case/results.json';
 if hidden_cfg.exists() and hidden_ref.exists() and src.exists():
  with tempfile.TemporaryDirectory() as td:
   rr=subprocess.run([sys.executable,str(src),'--config',str(hidden_cfg),'--output',td],capture_output=True,text=True,timeout=30)
   h=load(Path(td)/'results.json')
   if rr.returncode==0 and h:
    hs,_=compare(h,load(hidden_ref)); s += 0.10 if hs>=0.75 else 0.; reasons += [] if hs>=0.75 else ['hidden case failed']
   else: reasons.append('hidden solver rerun failed')
 score=max(0.,min(1.,s));print(json.dumps({'score':score,'reasons':reasons}))
if __name__=='__main__':main()
