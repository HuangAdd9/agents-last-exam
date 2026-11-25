"""Independent host-side oracle for the Ka'ena analogue."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

G = 6.67430e-11
EPS = 500.0
UNITS = {"velocity":"km/s","density":"kg/m^3","gravity":"mGal","flexure":"m","te":"km"}

def _density(v):
    return 1000.0*(1.6612*v - 0.4721*v*v + 0.0671*v**3 - 0.0043*v**4 + 0.000106*v**5)

def _forward(p):
    x=np.asarray(p['x_km'],float); vp=np.asarray(p['vp_km_s'],float)
    top=np.asarray(p['top_depth_km'],float); bot=np.asarray(p['bottom_depth_km'],float)
    nl,nx=vp.shape; dx=float(p['dx_km'])*1000.0
    rho=_density(vp)
    zobs=-1000.0*float(p['observation_depth_km']); rb=float(p['background_density_kg_m3'])
    grav=np.zeros(nx)
    for i,xi in enumerate(x*1000.0):
        for l in range(nl):
            for j,xj in enumerate(x*1000.0):
                zmid=500.0*(top[l,j]+bot[l,j]); thick=1000.0*(bot[l,j]-top[l,j])
                dz=zobs-zmid; dd=xi-xj
                grav[i] += 2.0*G*(rho[l,j]-rb)*dx*thick*dz/(dd*dd+dz*dz+EPS*EPS)
    grav *= 1.0e5
    # Keep the full column thickness for the density-weighted mean, but the
    # specified flexural load acts over depth from sea level to the deepest
    # reflector (not merely the sum of layer intervals).
    thickness=(bot[-1]-top[0])*1000.0
    rmean=np.sum(rho*(bot-top)*1000.0,axis=0)/thickness
    q_thickness_m=bot[-1]*1000.0
    fac=math.exp(-((dx/(float(p['flexural_wavelength_km'])*1000.0))**2)/2.0)
    # The task's flexure observable is explicitly negative-down/positive-up.
    # Thus the buoyancy load keeps (rho_mean-rho_mantle): negative values and
    # the resulting negative w represent downward deflection in this contract.
    load=(rmean-float(p['mantle_density_kg_m3']))*float(p['gravity_accel_m_s2'])*q_thickness_m*fac
    return rho,grav,load

def _choose_te(trace):
    """Choose the smallest Te within the specified 1e-12 RMS tie band."""
    min_rms=min(rms for _,rms in trace)
    return min((te,rms) for te,rms in trace if rms <= min_rms+1e-12), min_rms

def _flexure(load,p,te_km):
    n=len(load); dx=float(p['dx_km'])*1000.0
    # Explicit DFT/IDFT, deliberately independent from a candidate's FFT call.
    qh=np.array([sum(load[j]*np.exp(-2j*math.pi*k*j/n) for j in range(n)) for k in range(n)])
    E=float(p['youngs_modulus_pa']); nu=float(p['poisson_ratio']); gm=float(p['gravity_accel_m_s2']); rm=float(p['mantle_density_kg_m3'])
    D=E*(float(te_km)*1000.0)**3/(12.0*(1.0-nu*nu)); wh=[]
    for k in range(n):
        signed=k if k <= n//2 else k-n
        kval=2.0*math.pi*signed/(n*dx)
        wh.append(qh[k]/(D*kval**4+rm*gm))
    return np.asarray([sum(wh[k]*np.exp(2j*math.pi*k*j/n) for k in range(n)).real/n for j in range(n)])

def solve_reference(problem:dict)->dict:
    rho,grav,load=_forward(problem)
    obs=np.asarray(problem['flexure_observed_m'],float)
    trace=[]
    for te in problem['te_grid_km']:
        w=_flexure(load,problem,float(te)); trace.append((float(te),float(np.sqrt(np.mean((w-obs)**2)))))
    best,_min_rms=_choose_te(trace)
    flex=_flexure(load,problem,best[0])
    vp=np.asarray(problem['vp_km_s'],float)
    invariant=float(np.min(np.diff(vp,axis=0)))
    return {'density_kg_m3':rho,'gravity_mgal':grav,'flexure_m':flex,
            'te_km':np.asarray([best[0]]),'gravity_rms_mgal':np.asarray([float(np.sqrt(np.mean((grav-np.asarray(problem['gravity_observed_mgal']))**2)))]),
            'flexure_rms_m':np.asarray([best[1]]),'trace':trace,'invariant':invariant}

def write_reference(problem,path):
    r=solve_reference(problem); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,**{k:r[k] for k in ('density_kg_m3','gravity_mgal','flexure_m','te_km','gravity_rms_mgal','flexure_rms_m')})
    with (path.parent/'trace.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['te_km','flexure_rms_m']); w.writerows(r['trace'])

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); write_reference(json.loads(Path(a.input).read_text()),Path(a.output))
