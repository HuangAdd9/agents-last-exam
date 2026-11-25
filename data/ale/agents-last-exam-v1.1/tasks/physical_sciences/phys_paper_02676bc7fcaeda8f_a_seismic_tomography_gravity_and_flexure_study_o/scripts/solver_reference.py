"""Reference-style candidate solver used only to build adversarial fixtures."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
G=6.67430e-11; EPS=500.0
UNITS={"velocity":"km/s","density":"kg/m^3","gravity":"mGal","flexure":"m","te":"km"}
def solve(p,out):
 x=np.asarray(p['x_km'],float);vp=np.asarray(p['vp_km_s'],float);top=np.asarray(p['top_depth_km'],float);bot=np.asarray(p['bottom_depth_km'],float)
 rho=1000*(1.6612*vp-0.4721*vp**2+0.0671*vp**3-0.0043*vp**4+0.000106*vp**5);nl,nx=vp.shape;dx=float(p['dx_km'])*1000
 grav=np.zeros(nx);zobs=-1000*float(p['observation_depth_km'])
 for i,xi in enumerate(x*1000):
  for l in range(nl):
   for j,xj in enumerate(x*1000):
    zm=500*(top[l,j]+bot[l,j]);th=1000*(bot[l,j]-top[l,j]);dz=zobs-zm;dd=xi-xj
    grav[i]+=2*G*(rho[l,j]-float(p['background_density_kg_m3']))*dx*th*dz/(dd*dd+dz*dz+EPS*EPS)
 grav*=1e5;thick=(bot[-1]-top[0])*1000;rmean=np.sum(rho*(bot-top)*1000,axis=0)/thick
 q_thickness_m=bot[-1]*1000
 # Flexure is negative-down/positive-up; retain the signed buoyancy load.
 load=(rmean-float(p['mantle_density_kg_m3']))*float(p['gravity_accel_m_s2'])*q_thickness_m*math.exp(-((dx/(float(p['flexural_wavelength_km'])*1000))**2)/2)
 qh=np.fft.fft(load); n=nx; E=float(p['youngs_modulus_pa']);nu=float(p['poisson_ratio']);gm=float(p['gravity_accel_m_s2']);rm=float(p['mantle_density_kg_m3'])
 rows=[]
 for te in p['te_grid_km']:
  D=E*(float(te)*1000)**3/(12*(1-nu*nu));den=[]
  for k in range(n):
   signed=k if k<=n//2 else k-n; kval=2*np.pi*signed/(n*dx);den.append(D*kval**4+rm*gm)
  w=np.fft.ifft(qh/np.asarray(den)).real;rms=float(np.sqrt(np.mean((w-np.asarray(p['flexure_observed_m']))**2)));rows.append((float(te),rms))
 min_rms=min(rms for _,rms in rows)
 best=min((te,rms) for te,rms in rows if rms<=min_rms+1e-12);D=E*(best[0]*1000)**3/(12*(1-nu*nu));den=[]
 for k in range(n):
  signed=k if k<=n//2 else k-n;kval=2*np.pi*signed/(n*dx);den.append(D*kval**4+rm*gm)
 flex=np.fft.ifft(qh/np.asarray(den)).real
 out.mkdir(parents=True,exist_ok=True);np.savez(out/'result.npz',density_kg_m3=rho,gravity_mgal=grav,flexure_m=flex,te_km=np.asarray([best[0]]),gravity_rms_mgal=np.asarray([float(np.sqrt(np.mean((grav-np.asarray(p['gravity_observed_mgal']))**2)))]),flexure_rms_m=np.asarray([best[1]]))
 with (out/'trace.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['te_km','flexure_rms_m']);w.writerows(rows)
 (out/'summary.json').write_text(json.dumps({'units':UNITS,'min_qp_minus_qsv':float(np.min(np.diff(vp,axis=0)))},separators=(',',':')))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();solve(json.loads(Path(a.input).read_text()),Path(a.output))
