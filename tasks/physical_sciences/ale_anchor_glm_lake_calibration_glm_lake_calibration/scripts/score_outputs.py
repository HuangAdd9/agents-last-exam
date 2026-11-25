#!/usr/bin/env python3
"""Contract-first grader for the lake calibration task."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,shutil,subprocess,tempfile
from pathlib import Path
DEPTHS=(0.0,2.0,5.0,10.0,15.0)
REQUIRED=('simulation.csv','calibration_summary.json')
def fail(reason,**kw):
    print(json.dumps({'score':0.0,'passed':False,'reason':reason,**kw},sort_keys=True)); return 0
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read_rows(p):
    with p.open(newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['day','depth_m','temp_c']: raise ValueError('header')
        out=[]
        for x in r:
            d=int(x['day']); z=float(x['depth_m']); t=float(x['temp_c'])
            if not math.isfinite(t) or z not in DEPTHS: raise ValueError('value')
            out.append((d,z,t))
    return out
def rmse(rows,obs_path):
    obs=read_rows(obs_path); 
    if len(rows)!=len(obs) or [(d,z) for d,z,_ in rows]!=[(d,z) for d,z,_ in obs]: return math.inf
    return math.sqrt(sum((a-b)**2 for (_,_,a),(_,_,b) in zip(rows,obs))/len(obs))
def summary_ok(s,rows,nforcing):
    vals=[t for _,_,t in rows]; surf=[t for _,z,t in rows if z==0.0]; deep=[t for _,z,t in rows if z==15.0]
    checks=[s.get('schema_version')==1,s.get('rows')==len(rows),s.get('forcing_days')==nforcing,s.get('units')=={'day':'day_index','depth_m':'m','temp_c':'degC'},abs(float(s.get('min_temp_c',9e9))-min(vals))<1e-8,abs(float(s.get('max_temp_c',-9e9))-max(vals))<1e-8,abs(float(s.get('mean_surface_temp_c',9e9))-sum(surf)/len(surf))<1e-8,abs(float(s.get('mean_deep_temp_c',9e9))-sum(deep)/len(deep))<1e-8,abs(float(s.get('surface_deep_delta_c',9e9))-(sum(surf)/len(surf)-sum(deep)/len(deep)))<1e-8]
    return all(checks)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--reference-dir',required=True); a=ap.parse_args(); inp,out,ref=map(Path,(a.input_dir,a.output_dir,a.reference_dir))
    try:
        contract=json.loads((ref/'evaluation_contract.json').read_text())
        if not out.is_dir() or sorted(p.name for p in out.iterdir() if p.is_file())!=sorted(REQUIRED): return fail('exact output path gate')
        if sha(inp/'forcing_visible.csv')!=contract['visible_forcing_sha256'] or sha(inp/'observations_visible.csv')!=contract['visible_observations_sha256']: return fail('visible input integrity gate')
        rows=read_rows(out/'simulation.csv'); summ=json.loads((out/'calibration_summary.json').read_text())
        expected_rows=contract['visible_days']*len(DEPTHS)
        if len(rows)!=expected_rows or rows[0][:2]!=(0,0.0) or rows[-1][0]!=contract['visible_days']-1: return fail('shape/order gate')
        if not summary_ok(summ,rows,contract['visible_days']): return fail('summary invariant gate')
        if min(t for _,_,t in rows)<-5 or max(t for _,_,t in rows)>45 or abs(float(summ['surface_deep_delta_c']))>10: return fail('physical invariant gate')
    except Exception as e: return fail('malformed output',detail=str(e))
    with tempfile.TemporaryDirectory(prefix='lake_eval_') as td:
        rerun=Path(td)/'visible'; hidden=Path(td)/'hidden'; rerun.mkdir(); hidden.mkdir()
        try:
            subprocess.run(['/usr/bin/python3',str(Path(__file__).parents[1]/'software/model.py'),'--config',str(inp/'model.nml'),'--forcing',str(inp/'forcing_visible.csv'),'--output',str(rerun)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
            subprocess.run(['/usr/bin/python3',str(Path(__file__).parents[1]/'software/model.py'),'--config',str(inp/'model.nml'),'--forcing',str(ref/'forcing_hidden.csv'),'--output',str(hidden)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
            rerows=read_rows(rerun/'simulation.csv'); hrows=read_rows(hidden/'simulation.csv'); hobs=ref/'observations_hidden.csv'
            fidelity=math.sqrt(sum((a-b)**2 for (_,_,a),(_,_,b) in zip(rows,rerows))/len(rows)) if len(rows)==len(rerows) else math.inf
            vis_rmse=rmse(rows,inp/'observations_visible.csv'); hidden_rmse=rmse(hrows,hobs)
            hidden_summary=json.loads((hidden/'calibration_summary.json').read_text())
            invariant=summary_ok(hidden_summary,hrows,contract['hidden_days']) and abs(float(hidden_summary['surface_deep_delta_c']))<=10 and min(t for _,_,t in hrows)>=-5 and max(t for _,_,t in hrows)<=45
        except Exception as e: return fail('model rerun failure',detail=str(e))
    b={'output_schema':0.15,'rerun_fidelity':0.20 if fidelity<=1e-8 else 0.0,'visible_fit':0.25 if vis_rmse<=contract['visible_rmse_max'] else 0.0,'hidden_fit':0.30 if hidden_rmse<=contract['hidden_rmse_max'] else 0.0,'physical_invariant':0.10 if invariant else 0.0}
    score=round(sum(b.values()),6)
    print(json.dumps({'score':score,'passed':score>=0.999999,'visible_rmse_c':vis_rmse,'hidden_rmse_c':hidden_rmse,'rerun_rmse_c':fidelity,'breakdown':b},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
