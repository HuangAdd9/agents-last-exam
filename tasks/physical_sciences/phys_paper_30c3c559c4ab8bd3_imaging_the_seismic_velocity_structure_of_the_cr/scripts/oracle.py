"""Separate reference implementation used by the evaluator and local checks."""
import argparse,json
from pathlib import Path
import numpy as np
UNITS={"velocity":"km/s","slowness":"s/km","rms_residual":"s/km","roughness":"(s/km)^2"}
def _L(nx,ny):
 n=nx*ny; rows=[]
 for y in range(ny):
  for x in range(nx-1):
   q=np.zeros(n); q[y*nx+x]=1.; q[y*nx+x+1]=-1.; rows.append(q)
 for y in range(ny-1):
  for x in range(nx):
   q=np.zeros(n); q[y*nx+x]=1.; q[(y+1)*nx+x]=-1.; rows.append(q)
 return np.asarray(rows,dtype=np.float64) if rows else np.empty((0,n),dtype=np.float64)
def solve_reference(a):
 K=np.asarray(a['ray_matrix'],dtype=np.float64); d=np.asarray(a['observed_slowness_s_per_km'],dtype=np.float64); ny=int(a['grid']['ny']); nx=int(a['grid']['nx']); n=nx*ny; L=_L(nx,ny)
 vari=np.asarray(a['data_variance_s2_per_km2'],dtype=np.float64); pv=np.asarray(a['model_variance_s2_per_km2'],dtype=np.float64); m0=np.asarray(a['initial_model_slowness_s_per_km'],dtype=np.float64); lam=float(a['regularization_lambda']); mu=float(a['prior_weight']); alpha=float(a['update_damping']); nit=int(a['iterations'])
 vel=[]; pred=[]; rms=[]; rough=[]
 for p in range(d.shape[0]):
  A=K[p]; w=1.0/vari[p]; c=1.0/pv; m=m0.copy(); H=(A.T*w)@A+lam*(L.T@L)+mu*np.diag(c)+1e-10*np.eye(n)
  for _ in range(nit):
   g=A.T@(w*(d[p]-A@m))-mu*c*(m-m0)-lam*L.T@(L@m); m += alpha*np.linalg.solve(H,g)
  y=A@m; vel.append((1.0/m).reshape(ny,nx)); pred.append(y); rms.append(np.sqrt(np.mean((y-d[p])**2))); rough.append(float(np.mean((L@m)**2)) if L.shape[0] else 0.0)
 return {'velocity_km_s':np.array(vel),'predicted_slowness_s_per_km':np.array(pred),'rms_residual_s_per_km':np.array(rms),'roughness_slowness':np.array(rough)}
def run(inp,out):
 a=json.loads(Path(inp).read_text()); Path(out).mkdir(parents=True,exist_ok=True); r=solve_reference(a); np.savez(Path(out)/'result.npz',**r); obs=np.asarray(a['observed_slowness_s_per_km']); Path(out,'diagnostics.json').write_text(json.dumps({'max_abs_residual_s_per_km':float(np.max(np.abs(r['predicted_slowness_s_per_km']-obs))),'units':UNITS},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output'); p.add_argument('pos',nargs='*'); z=p.parse_args(); run(z.input or z.pos[0],z.output or z.pos[1])
