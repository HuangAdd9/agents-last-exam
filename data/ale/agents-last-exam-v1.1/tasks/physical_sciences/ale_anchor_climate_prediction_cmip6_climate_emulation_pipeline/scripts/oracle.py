#!/usr/bin/env python3
"""Independent deterministic oracle for the authored climate emulator."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

def make_X(F, month, lat):
    F=np.asarray(F,float); month=np.asarray(month,float); lat=np.asarray(lat,float)
    co=(F[:,0]-400)/100; ch=(F[:,1]-.30)/.10; so=F[:,2]/1e-12; bc=F[:,3]/1e-13; rs=(F[:,4]-300)/50
    b=np.stack([np.ones(len(F)),co,ch,so,bc,rs,np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12)],1)
    X=np.empty((len(F),11,len(lat),8),float); la=lat[:,None]/90
    X[:,:8]=b[:,:,None,None]; X[:,8]=b[:,1,None,None]*b[:,2,None,None]; X[:,9]=b[:,1,None,None]*b[:,5,None,None]; X[:,10]=b[:,2,None,None]*b[:,5,None,None]
    return X

def main(input_dir: Path, reference_dir: Path):
    z=np.load(input_dir/'climate_fixture.npz',allow_pickle=False)
    F,Y,Ft,lat,lon,tm=z['train_forcing'],z['train_targets'],z['test_forcing'],z['lat'],z['lon'],z['test_month']
    X=make_X(F,z['train_month'],lat); Xt=make_X(Ft,tm,lat)
    A=X.reshape(len(F),11,-1).transpose(0,2,1); T=Y.reshape(len(F),2,-1)
    coef=np.empty((2,11,6,8),float)
    for o in range(2):
      for c in range(48): coef[o,:,c//8,c%8]=np.linalg.lstsq(A[:,c,:],T[:,o,c],rcond=None)[0]
    pred=np.einsum('nikl,oikl->nokl',Xt,coef)
    # withheld perturbation: evaluator-only SSP245-like forcing with a second aerosol trajectory
    hm=np.arange(12); H=np.array([[414+1.1*m+2.5*np.sin(2*np.pi*m/12),.298+.001*m+.002*np.cos(2*np.pi*m/12),.88e-12*(1-.004*m),1.15e-13*(1+.15*np.sin(2*np.pi*m/12)),302+.2*m+1.5*np.cos(2*np.pi*m/12)] for m in hm])
    Hp=np.einsum('nikl,oikl->nokl',make_X(H,hm,lat),coef)
    reference_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(reference_dir/'reference_arrays.npz',test_truth=pred.astype('f8'),hidden_forcing=H.astype('f8'),hidden_truth=Hp.astype('f8'),X_test=Xt.astype('f8'),coefficients=coef.astype('f8'))
    np.savez_compressed(reference_dir/'test_outputs.npz',test_predictions=pred.astype('f8'))
    contract={'schema_version':1,'rmse_max':{'tas':1e-7,'pr':1e-9},'hidden_rmse_max':1e-7,'train_sha256':hashlib.sha256(z['train_targets'].tobytes()).hexdigest(),'coeff_shape':[2,11,6,8]}
    (reference_dir/'evaluation_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    diag={'train_rows':len(F),'design_shape':list(X.shape),'rank_min':int(min(np.linalg.matrix_rank(A[:,c,:]) for c in range(48))),'test_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'hidden_sha256':hashlib.sha256(Hp.tobytes()).hexdigest()}
    (reference_dir/'oracle_diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    print(json.dumps({'test_sha256':diag['test_sha256'],'hidden_sha256':diag['hidden_sha256'],'rank_min':diag['rank_min']}))
if __name__=='__main__':
    if len(sys.argv)!=3: raise SystemExit('usage: oracle.py INPUT_DIR REFERENCE_DIR')
    main(Path(sys.argv[1]),Path(sys.argv[2]))
