#!/usr/bin/env python3
import csv, json, math, os, subprocess, sys, tempfile
from pathlib import Path

def loadj(p):
 with open(p) as f: return json.load(f)
def close(a,b,t=1e-8): return abs(float(a)-float(b))<=t
def vec(a,b,t=1e-8): return len(a)==len(b) and all(close(x,y,t) for x,y in zip(a,b))
def score(input_dir, output_dir, reference_dir):
 out=Path(output_dir); ref=Path(reference_dir); inp=Path(input_dir)
 required={'mean_field.json','quasiparticle.json','band_path.csv','diagnostics.json','solve.py'}
 if not out.is_dir() or set(x.name for x in out.iterdir()) != required: return 0.0, ['file_gate']
 try:
  mf=loadj(out/'mean_field.json'); qp=loadj(out/'quasiparticle.json'); dg=loadj(out/'diagnostics.json')
  vr=loadj(ref/'visible_reference.json'); hr=loadj(ref/'hidden_reference.json'); hidden=loadj(ref/'hidden_case.json')
 except Exception: return 0.0,['parse']
 if mf.get('units')!='eV' or qp.get('units')!='eV' or dg.get('units')!='eV': return 0.0,['units']
 components=[]; failures=[]
 # visible mean-field 0.20
 ok=close(mf.get('indirect_gap_eV'),vr['mean_field']['indirect_gap_eV']) and vec(mf.get('vbm',{}).get('k',[]),vr['mean_field']['vbm']['k']) and vec(mf.get('cbm',{}).get('k',[]),vr['mean_field']['cbm']['k'])
 components.append(0.20 if ok else 0.0); failures += [] if ok else ['visible_mean_field']
 # visible qp topology 0.20
 ok=all(close(qp.get(k),vr['quasiparticle'][k]) for k in ['indirect_gap_eV','gap_opening_eV','direct_gamma_gap_eV']) and vec(qp.get('vbm',{}).get('k',[]),vr['quasiparticle']['vbm']['k']) and vec(qp.get('cbm',{}).get('k',[]),vr['quasiparticle']['cbm']['k'])
 components.append(0.20 if ok else 0.0); failures += [] if ok else ['visible_qp']
 # path score RMSE (0.15)
 try:
  with open(out/'band_path.csv',newline='') as f: rows=list(csv.DictReader(f))
  with tempfile.NamedTemporaryFile('w',delete=False) as tf: tf.write(vr['band_path_csv']); vp=tf.name
  with open(vp,newline='') as f: vrows=list(csv.DictReader(f))
  os.unlink(vp)
  vals=['ev_dft','ec_dft','ev_qp','ec_qp']; errs=[]
  if len(rows)!=len(vrows) or [r['segment'] for r in rows] != [r['segment'] for r in vrows] or [r['index'] for r in rows] != [r['index'] for r in vrows]: raise ValueError()
  for a,b in zip(rows,vrows): errs += [(float(a[k])-float(b[k]))**2 for k in vals]
  rmse=math.sqrt(sum(errs)/len(errs)); ok=rmse<=2e-6
 except Exception: ok=False
 components.append(0.15 if ok else 0.0); failures += [] if ok else ['path']
 # diagnostics/invariants 0.10
 ok=all(dg.get(k)==vr['diagnostics'].get(k) for k in ['vbm_at_gamma','cbm_near_declared_valley','qp_gap_opening_positive']) and dg.get('structure_atom_count')==2 and dg.get('path_rows')==vr['diagnostics']['path_rows']
 components.append(0.10 if ok else 0.0); failures += [] if ok else ['diagnostics']
 # hidden generic solver 0.35
 hidden_ok=False
 try:
  solver=out/'solve.py'; st=solver.stat();
  if not os.access(solver,os.X_OK): raise ValueError('not executable')
  with tempfile.TemporaryDirectory() as td:
   result=Path(td)/'result.json'; p=subprocess.run([str(solver),str(ref/'hidden_case.json'),str(result)],capture_output=True,text=True,timeout=20)
   if p.returncode!=0: raise ValueError(p.stderr)
   got=loadj(result)
   hidden_ok=(close(got['mean_field']['indirect_gap_eV'],hr['mean_field']['indirect_gap_eV']) and close(got['quasiparticle']['indirect_gap_eV'],hr['quasiparticle']['indirect_gap_eV']) and close(got['quasiparticle']['gap_opening_eV'],hr['quasiparticle']['gap_opening_eV']) and vec(got['quasiparticle']['cbm']['k'],hr['quasiparticle']['cbm']['k']) and got['diagnostics']['qp_gap_opening_positive'] is True)
 except Exception: hidden_ok=False
 components.append(0.35 if hidden_ok else 0.0); failures += [] if hidden_ok else ['hidden_generic']
 return round(sum(components),10), failures
if __name__=='__main__':
 if len(sys.argv)!=7 or sys.argv[1]!='--input-dir': print(json.dumps({'score':0.0,'failures':['args']})); raise SystemExit(0)
 # expected flags: --input-dir X --output-dir Y --reference-dir Z
 try: i=sys.argv.index('--input-dir'); o=sys.argv.index('--output-dir'); r=sys.argv.index('--reference-dir'); val,fail=score(sys.argv[i+1],sys.argv[o+1],sys.argv[r+1])
 except Exception: val,fail=0.0,['infrastructure']
 print(json.dumps({'score':val,'failures':fail},sort_keys=True))
