from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _arr(x): return np.asarray(x, dtype=float)

def dynamical_builder(cfg):
    a=float(cfg['a']); a1=np.array([a,0.0]); a2=np.array([a/2,a*np.sqrt(3)/2])
    tau=[np.zeros(2), np.array([a/2,a*np.sqrt(3)/6])]
    masses=np.array([float(cfg['m1']),float(cfg['m2'])])
    ab_near=[np.zeros(2),-a1,-a2]
    ab_third=[a1-a2,-a1+a2,-a1-a2]
    same=[a1,-a1,a2,-a2,a1-a2,-a1+a2]
    directed=[]
    for R in ab_near: directed.append((0,1,R,float(cfg['k1'])))
    for R in ab_third: directed.append((0,1,R,float(cfg['k3'])))
    for R in same:
        directed.append((0,0,R,float(cfg['k2'])))
        directed.append((1,1,R,float(cfg['k2'])))
    directed += [(b,a,-R,k) for a,b,R,k in directed[:6]]
    onsite=[np.zeros((2,2)),np.zeros((2,2))]; blocks={}
    for al,be,R,k in directed:
        d=R+tau[be]-tau[al]; e=d/np.linalg.norm(d); K=k*np.outer(e,e)
        onsite[al] += K
        key=(al,be,tuple(np.asarray(R,dtype=float)))
        blocks[key]=blocks.get(key,np.zeros((2,2)))-K
    def dyn(q):
        q=np.asarray(q,float); D=np.zeros((4,4),complex)
        for al in (0,1):
            sl=slice(2*al,2*al+2); D[sl,sl] += onsite[al]/masses[al]
        for (al,be,Rt),K in blocks.items():
            R=np.asarray(Rt); sl=slice(2*al,2*al+2); tl=slice(2*be,2*be+2)
            D[sl,tl] += K*np.exp(1j*np.dot(q,R+tau[be]-tau[al]))/np.sqrt(masses[al]*masses[be])
        return (D+D.conj().T)/2
    return dyn, a1, a2

def frequencies(cfg, q):
    dyn,_,_=dynamical_builder(cfg)
    vals=np.linalg.eigvalsh(dyn(q))
    vals[np.abs(vals)<1e-12]=0.0
    return np.sqrt(np.maximum(vals,0.0)), vals

def path_dispersion(cfg):
    _,a1,a2=dynamical_builder(cfg); pi=np.pi
    b1=2*pi/float(cfg['a'])*np.array([1.,-1/np.sqrt(3)])
    b2=2*pi/float(cfg['a'])*np.array([0.,2/np.sqrt(3)])
    G=np.zeros(2); M=b1/2; K=(2*b1+b2)/3
    def seg(x,y,n,include_end):
        # first two segments exclude their endpoint; final includes both endpoints
        t=np.linspace(0,1,n,endpoint=include_end)
        return x[None,:]*(1-t[:,None])+y[None,:]*t[:,None]
    q=np.vstack([seg(G,M,int(cfg['n_path_gm']),False),seg(M,K,int(cfg['n_path_mk']),False),seg(K,G,int(cfg['n_path_kg']),True)])
    dist=np.zeros(len(q)); dist[1:]=np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))
    fs=np.array([frequencies(cfg,x)[0] for x in q])
    vals=np.array([frequencies(cfg,x)[1] for x in q])
    return q,dist,fs,vals,(G,M,K)

def grid_frequencies(cfg):
    n=int(cfg['q_grid']); _,a1,a2=dynamical_builder(cfg)
    b1=2*np.pi/float(cfg['a'])*np.array([1.,-1/np.sqrt(3)])
    b2=2*np.pi/float(cfg['a'])*np.array([0.,2/np.sqrt(3)])
    frac=np.arange(n,dtype=float)/n
    out=np.empty((n,n,4))
    for i,x in enumerate(frac):
        for j,y in enumerate(frac): out[i,j]=frequencies(cfg,x*b1+y*b2)[0]
    return out,b1,b2

def _triangle_density(x,y,z,omega):
    v=np.sort([float(x),float(y),float(z)]); x,y,z=v; t=np.asarray(omega); out=np.zeros_like(t)
    eps=1e-12
    dxy=max(y-x,eps); dxz=max(z-x,eps); dyz=max(z-y,eps)
    m1=(t>=x)&(t<y); m2=(t>=y)&(t<=z)
    out[m1]=2*(t[m1]-x)/(dxy*dxz)
    out[m2]=2*(z-t[m2])/(dxz*dyz)
    if z-x<eps: out[np.abs(t-x)<eps]=0
    return out

def dos_from_grid(cfg, grid):
    n=int(cfg['q_grid']); nb=int(cfg['omega_bins']); maxf=float(np.max(grid)); ommax=float(np.ceil(maxf*10-1e-12)/10)
    omega=np.linspace(0.0,ommax,nb); dos=np.zeros(nb); area=1.0/(n*n*2.0)
    for i in range(n):
        ip=(i+1)%n
        for j in range(n):
            jp=(j+1)%n
            for br in range(4):
                w00,w10,w01,w11=grid[i,j,br],grid[ip,j,br],grid[i,jp,br],grid[ip,jp,br]
                dos += area*(_triangle_density(w00,w10,w11,omega)+_triangle_density(w00,w11,w01,omega))
    integ=float(np.trapezoid(dos,omega))
    if integ>0: dos*=4.0/integ
    return omega,dos

def thermo(omega,dos,n_t):
    T=np.geomspace(0.02,80.0,int(n_t)); w=np.maximum(omega,1e-14); x=w[None,:]/T[:,None]
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        nbe=1/np.expm1(np.minimum(x,700)); energy=np.trapezoid(dos[None,:]*w[None,:]*(nbe+0.5),omega,axis=1)
        cv_integrand=(x*x)*np.exp(-np.minimum(x,700))/np.square(-np.expm1(-np.minimum(x,700)))
        cv=np.trapezoid(dos[None,:]*cv_integrand,omega,axis=1)
        free=np.trapezoid(dos[None,:]*(0.5*w[None,:]+T[:,None]*np.log(-np.expm1(-np.minimum(x,700)))),omega,axis=1)
    return T,energy,cv,free

def solve_case(cfg):
    q,dist,freq,vals,pts=path_dispersion(cfg); grid,_,_=grid_frequencies(cfg); om,dos=dos_from_grid(cfg,grid); T,E,C,F=thermo(om,dos,cfg['temperatures'])
    # analytic diatomic chain
    nq=121; qc=np.linspace(0,np.pi,nq); k=float(cfg['chain_k']); m1=float(cfg['chain_m1']); m2=float(cfg['chain_m2']); s=(1/m1+1/m2)
    disc=np.maximum(s*s-4*np.sin(qc/2)**2/(m1*m2),0)
    wa=np.sqrt(k*(s-np.sqrt(disc))); wo=np.sqrt(k*(s+np.sqrt(disc)))
    results={'tier1':{'optical_frequency_at_gamma':float(wo[0]),'max_acoustic_at_gamma':float(wa[0]),'num_q_points':121},'tier2':{'num_q_points':int(len(q)),'num_branches':4,'freq_range':[float(np.min(freq[freq>1e-12])),float(np.max(freq))],'negative_eigenvalue_count':int(np.sum(vals < -1e-10)),'acoustic_at_gamma':[float(freq[-1,0]),float(freq[-1,1])]},'tier3':{'omega_max':float(om[-1]),'dos_integral':float(np.trapezoid(dos,om)),'cv_high_T':float(C[-1]),'cv_low_T':float(C[0]),'num_temperatures':int(len(T)),'q_grid_size':[int(cfg['q_grid']),int(cfg['q_grid'])]}}
    return {'q_points':q,'q_distance':dist,'frequencies':freq,'diatomic':(qc,wa,wo),'omega_bins':om,'dos':dos,'temperatures':T,'energy':E,'heat_capacity':C,'free_energy':F,'results':results}

def write_outputs(result,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    qc,wa,wo=result['diatomic']; np.savez_compressed(out/'diatomic_1d.npz',q_points=qc,omega_acoustic=wa,omega_optical=wo)
    np.savez_compressed(out/'dispersion_2d.npz',q_points=result['q_points'],q_distance=result['q_distance'],frequencies=result['frequencies'])
    np.savez_compressed(out/'dos.npz',omega_bins=result['omega_bins'],dos=result['dos'])
    np.savez_compressed(out/'thermodynamics.npz',temperatures=result['temperatures'],energy=result['energy'],heat_capacity=result['heat_capacity'],free_energy=result['free_energy'])
    (out/'results.json').write_text(json.dumps(result['results'],indent=2,sort_keys=True),encoding='utf-8')
