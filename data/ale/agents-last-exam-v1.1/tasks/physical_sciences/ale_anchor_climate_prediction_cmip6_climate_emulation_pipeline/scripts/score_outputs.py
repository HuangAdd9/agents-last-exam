#!/usr/bin/env python3
"""Contract-first deterministic climate emulator grader."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np
REQUIRED=('processed/preprocessed.npz','processed/coefficients.npy','processed/test_predictions.npz','processed/metadata.json','summary.json','submissions/ssp245_predictions.csv')
def bad(reason,breakdown=None): print(json.dumps({'score':0.0,'passed':False,'reason':reason,'breakdown':breakdown or {}},sort_keys=True)); return 0
def finite(a): return np.all(np.isfinite(a))
def make_X(F,month,lat):
 F=np.asarray(F,float); month=np.asarray(month,float); lat=np.asarray(lat,float); co=(F[:,0]-400)/100; ch=(F[:,1]-.30)/.10; so=F[:,2]/1e-12; bc=F[:,3]/1e-13; rs=(F[:,4]-300)/50
 b=np.stack([np.ones(len(F)),co,ch,so,bc,rs,np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12)],1); X=np.empty((len(F),11,len(lat),8)); la=lat[:,None]/90; X[:,:8]=b[:,:,None,None]; X[:,8]=b[:,1,None,None]*b[:,2,None,None]; X[:,9]=b[:,1,None,None]*b[:,5,None,None]; X[:,10]=b[:,2,None,None]*b[:,5,None,None]; return X
def area_means(pred,lat):
 w=np.cos(np.deg2rad(lat)); w=w/w.sum(); return [float(np.sum(pred[:,j].mean(0)*w[:,None])/8) for j in range(2)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--reference-dir',required=True); a=ap.parse_args(); inp,out,ref=map(Path,(a.input_dir,a.output_dir,a.reference_dir))
 if not out.is_dir(): return bad('missing output directory')
 actual=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
 if sorted(REQUIRED)!=actual: return bad('exact output path gate')
 try:
  z=np.load(inp/'climate_fixture.npz',allow_pickle=False); pre=np.load(out/'processed/preprocessed.npz',allow_pickle=False); pp=np.load(out/'processed/test_predictions.npz',allow_pickle=False); coef=np.load(out/'processed/coefficients.npy',allow_pickle=False); meta=json.loads((out/'processed/metadata.json').read_text()); summ=json.loads((out/'summary.json').read_text()); rr=np.load(ref/'reference_arrays.npz',allow_pickle=False); contract=json.loads((ref/'evaluation_contract.json').read_text())
 except Exception as e: return bad(f'malformed artifact: {e}')
 if set(pre.files)!= {'X_train','Y_train','X_test'} or set(pp.files)!= {'predictions'}: return bad('archive key gate')
 if pre['X_train'].shape!=(108,11,6,8) or pre['Y_train'].shape!=(108,2,6,8) or pre['X_test'].shape!=(12,11,6,8) or pp['predictions'].shape!=(12,2,6,8) or coef.shape!=(2,11,6,8): return bad('shape gate')
 if not all(finite(x) for x in (pre['X_train'],pre['Y_train'],pre['X_test'],pp['predictions'],coef)): return bad('finite gate')
 expected_X=make_X(z['train_forcing'],z['train_month'],z['lat']); expected_Xt=make_X(z['test_forcing'],z['test_month'],z['lat'])
 if not (np.allclose(pre['X_train'],expected_X,atol=1e-12,rtol=0) and np.allclose(pre['X_test'],expected_Xt,atol=1e-12,rtol=0) and np.allclose(pre['Y_train'],z['train_targets'],atol=1e-12,rtol=0)): return bad('preprocessing leakage or convention gate')
 expected_norm={'forcing_mean':z['train_forcing'].mean(0).tolist(),'forcing_std':z['train_forcing'].std(0).tolist(),'target_mean':z['train_targets'].mean(axis=(0,2,3)).tolist(),'target_std':z['train_targets'].std(axis=(0,2,3)).tolist()}
 if meta.get('schema_version')!=1 or meta.get('target_units')!={'tas':'K','pr':'mm/day'} or meta.get('train_rows')!=108 or meta.get('test_rows')!=12 or meta.get('train_sha256')!=contract['train_sha256'] or any(not np.allclose(meta.get(k,[]),v,atol=1e-12,rtol=0) for k,v in expected_norm.items()): return bad('metadata/unit/hash/normalization gate')
 pred=pp['predictions']; truth=rr['test_truth']; hidden_truth=rr['hidden_truth']; hidden_pred=np.einsum('nikl,oikl->nokl',make_X(rr['hidden_forcing'],np.arange(12),z['lat']),coef)
 trm=np.sqrt(np.mean((pred-truth)**2,axis=(0,2,3))); hidden_rmse=float(np.sqrt(np.mean((hidden_pred-hidden_truth)**2)))
 coeff_rmse=float(np.sqrt(np.mean((coef-rr['coefficients'])**2))); csv_path=out/'submissions/ssp245_predictions.csv'; rows=list(csv.DictReader(csv_path.open()))
 csv_ok=len(rows)==1152 and [r.get('month') for r in rows[:2]]==['0','0'] and [r.get('variable') for r in rows[:2]]==['tas','tas']
 if csv_ok:
  try:
   vals=np.array([float(r['prediction']) for r in rows]); csv_ok=finite(vals) and np.allclose(vals,pred.reshape(-1),atol=5e-6,rtol=0)
  except Exception: csv_ok=False
 means=area_means(pred,z['lat']); global_mean={'tas_K':float(pred[:,0].mean()),'pr_mm_day':float(pred[:,1].mean())}; inv=summ.get('energy_balance_check',{}); rm=summ.get('rmse',{}); inv_ok=abs(float(inv.get('lat_weighted_tas_K',9e9))-means[0])<1e-8 and abs(float(inv.get('lat_weighted_pr_mm_day',9e9))-means[1])<1e-10 and abs(float(summ.get('global_mean',{}).get('tas_K',9e9))-global_mean['tas_K'])<1e-8 and abs(float(summ.get('global_mean',{}).get('pr_mm_day',9e9))-global_mean['pr_mm_day'])<1e-10 and abs(float(rm.get('tas_rmse',9e9))-float(trm[0]))<1e-8 and abs(float(rm.get('pr_rmse',9e9))-float(trm[1]))<1e-10
 breakdown={'csv':.10 if csv_ok else 0.0,'test_tas':.20 if trm[0]<=contract['rmse_max']['tas'] else 0.0,'test_pr':.20 if trm[1]<=contract['rmse_max']['pr'] else 0.0,'coefficients':.20 if coeff_rmse<=1e-8 else 0.0,'hidden_perturbation':.20 if hidden_rmse<=contract['hidden_rmse_max'] else 0.0,'physical_invariant':.10 if inv_ok else 0.0}
 score=round(sum(breakdown.values()),6); print(json.dumps({'score':score,'passed':score>=.999999,'tas_rmse':float(trm[0]),'pr_rmse':float(trm[1]),'hidden_rmse':hidden_rmse,'coeff_rmse':coeff_rmse,'breakdown':breakdown},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
