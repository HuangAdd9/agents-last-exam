from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np

def prop(x,c):
    a,K,h,b=c['alpha'],c['K'],c['hill'],c['basal']; g=np.asarray(c['gamma'],float); A,B,C=x.astype(float)
    return np.array([a/(1+(B/K)**h+(C/K)**h)+b,a/(1+(A/K)**h+(C/K)**h)+b,a/(1+(A/K)**h+(B/K)**h)+b,g[0]*A,g[1]*B,g[2]*C],float)
def basin(x):
    if x[0]>x[1] and x[0]>x[2]: return 0
    if x[1]>x[0] and x[1]>x[2]: return 1
    if x[2]>x[0] and x[2]>x[1]: return 2
    return 3
def birth_death(seed=101,events=20000,lam=30.0,gamma=1.0):
    rng=np.random.default_rng(seed); x=0; vals=[]
    for _ in range(events):
        birth=lam; death=gamma*x; total=birth+death
        if rng.random()*total < birth: x += 1
        elif x>0: x -= 1
        vals.append(x)
    arr=np.asarray(vals,dtype=float)
    return float(arr.mean()),float(arr.std()),float(abs(arr.mean()-lam/gamma))
def ssa(c,seed,events,write_csv=None):
    rng=np.random.default_rng(seed); x=np.zeros(3,dtype=int); t=0.; states=[]; times=[]; bas=[]
    changes=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],int)
    for i in range(events):
        p=prop(x,c); a0=float(p.sum()); r1=max(float(rng.random()),1e-15); t += -math.log(r1)/a0; r2=float(rng.random())*a0; j=int(np.searchsorted(np.cumsum(p),r2,side='right')); j=min(j,5); x=x+changes[j]
        states.append(x.copy()); times.append(t); bas.append(basin(x))
    return np.asarray(states),np.asarray(times),np.asarray(bas)
def tau(c,seed,events):
    rng=np.random.default_rng(seed); x=np.zeros(3,dtype=int); states=[]; leaps=0; changes=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],int)
    while len(states)<events:
        p=prop(x,c); a0=max(float(p.sum()),1e-12); eps=1e-4; deriv=[]
        for k in range(3):
            y=x.copy(); y[k]+=1; deriv.append(np.max(np.abs(prop(y,c)-p)))
        tau=min(0.25,0.03*a0/max(1e-12,max(deriv)))
        n=np.maximum(1,np.ceil(tau*1000)).astype(int); ok=False
        while not ok:
            draws=rng.poisson(p*tau); trial=x+draws@changes
            if np.all(trial>=0): ok=True
            else: tau*=0.5
        x=trial; leaps+=1
        for _ in range(min(int(n.max()),events-len(states))): states.append(x.copy())
    return np.asarray(states),leaps
def summarize(c,outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    cfgdig=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    s,t,b=ssa(c,c['seeds'][0],c['events'])
    rows=[]
    with (outdir/'trajectory.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['event','time','A','B','C','basin'])
        for i,(ti,xi,bi) in enumerate(zip(t,s,b),1): w.writerow([i,format(float(ti),'.12g'),*map(int,xi),int(bi)])
    tr=[]
    allstates=[]
    for seed in c['seeds']:
        st,_,ba=ssa(c,seed,c['events']); allstates.append(st)
        fr=[float(np.mean(ba==k)) for k in range(4)]
        tr.append({'seed':seed,'mean_A':float(st[:,0].mean()),'mean_B':float(st[:,1].mean()),'mean_C':float(st[:,2].mean()),'std_A':float(st[:,0].std()),'std_B':float(st[:,1].std()),'std_C':float(st[:,2].std()),'fraction_A':fr[0],'fraction_B':fr[1],'fraction_C':fr[2],'unassigned_fraction':fr[3]})
    # correct accidental typo robustly
    for r in tr:
        if not isinstance(r['mean_C'],float): r['mean_C']=float(r['mean_C'])
    def row_for(alpha,seed):
        cc=dict(c); cc['alpha']=float(alpha); st,_,ba=ssa(cc,seed,c['events']); fr=[float(np.mean(ba==k)) for k in range(4)]
        return {'alpha':float(alpha),'mean_A':float(st[:,0].mean()),'mean_B':float(st[:,1].mean()),'mean_C':float(st[:,2].mean()),'fraction_A':fr[0],'fraction_B':fr[1],'fraction_C':fr[2],'unassigned_fraction':fr[3]}
    scan=[]
    for a in c['scan_alphas']:
        rr=[row_for(a,sid) for sid in c['seeds']]; scan.append({'alpha':float(a),**{k:float(np.mean([z[k] for z in rr])) for k in rr[0] if k!='alpha'}})
    tc=[]
    for a in c['tau_alphas']:
        cc=dict(c);cc['alpha']=float(a); exact=[]; leaps=[]; tls=[]
        for seed in c['seeds']:
            st,_,_=ssa(cc,seed,c['tau_events']); tl,n=tau(cc,seed,c['tau_events']); exact.append(float(st[:,0].mean())); tls.append(float(tl[:,0].mean())); leaps.append(n)
        em=float(np.mean(exact)); tm=float(np.mean(tls)); tc.append({'alpha':float(a),'exact_mean_A':em,'tauleap_mean_A':tm,'relative_error_A':abs(tm-em)/max(abs(em),1e-9),'leap_count':int(np.mean(leaps))})
    ens={k:float(np.mean([r[k] for r in tr])) for k in ('mean_A','mean_B','mean_C')}; ens.update({k.replace('mean','std'):float(np.std([r[k] for r in tr])) for k in ('mean_A','mean_B','mean_C')})
    vals=np.asarray(s[:,0]); result={'config_digest':cfgdig,'validation':{'events':int(c['events']),'final_population':int(s[-1].sum()),'mean_population':float(vals.mean()),'std_population':float(vals.std()),'birth_death_mean':birth_death()[0],'birth_death_std':birth_death()[1],'poisson_mean_error':birth_death()[2]},'trajectories':tr,'ensemble':ens,'alpha_scan':scan,'tau_comparison':tc,'invariants':{'max_fraction_sum_error':float(max(abs(sum(r[k] for k in ('fraction_A','fraction_B','fraction_C','unassigned_fraction'))-1) for r in tr)),'all_states_nonnegative':bool(np.all(s>=0))}}
    (outdir/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(); summarize(json.loads(Path(a.config).read_text()),a.output)
