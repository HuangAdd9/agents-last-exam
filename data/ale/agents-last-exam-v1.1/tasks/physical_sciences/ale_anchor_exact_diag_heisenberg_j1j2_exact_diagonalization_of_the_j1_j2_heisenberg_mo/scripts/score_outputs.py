from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import hamiltonian

REQ=('ground_state.npz','correlations.npz','dynamical_sf.npz','results.json')
def fail(reason,**kw): print(json.dumps({'score':0.0,'reason':reason,**kw})); return 0
def load_npz(p):
  with np.load(p) as z: return {k:z[k] for k in z.files}
def compare_case(out,ref,hidden=False):
  try:
    g=load_npz(out/'ground_state.npz'); gr=load_npz(ref/'ground_state.npz'); c=load_npz(out/'correlations.npz'); cr=load_npz(ref/'correlations.npz'); d=load_npz(out/'dynamical_sf.npz'); dr=load_npz(ref/'dynamical_sf.npz'); js=json.loads((out/'results.json').read_text())
    for sec, keys in {'tier1':('E0','E1','spin_gap','E0_per_site','hilbert_dim','num_sites'),'tier2':('S_q_pi_pi','S_q_0_0','correlation_sum_rule','C_nearest_neighbor','C_next_nearest_neighbor'),'tier3':('omega_max','eta','num_lanczos_steps','num_omega_points','num_q_points','sum_rule_max_error')}.items():
      if sec not in js or any(k not in js[sec] for k in keys): return False,'results schema'
    for x in ('E0','E1','spin_gap','ground_state_vector'):
      if x not in g: return False,'missing ground key'
    v=np.asarray(g['ground_state_vector']);
    if v.shape!=(924,) or not np.all(np.isfinite(v)) or abs(np.linalg.norm(v)-1)>1e-6:return False,'bad ground vector'
    H,_,_=hamiltonian(0.62 if hidden else 0.35); res=np.linalg.norm(H@v-float(g['E0'])*v)
    if res>1e-7:return False,'eigen residual'
    if max(abs(float(g[k])-float(gr[k])) for k in ('E0','E1','spin_gap'))>1e-7:return False,'energy mismatch'
    if abs(float(js['tier1']['E0'])-float(g['E0']))>1e-6 or int(js['tier1']['hilbert_dim'])!=924:return False,'results tier1 mismatch'
    if c['correlation_matrix'].shape!=(12,12) or c['q_points'].shape!=(12,2) or c['structure_factor'].shape!=(12,):return False,'bad correlation shape'
    if np.max(abs(np.diag(c['correlation_matrix'])-.75))>1e-6 or np.min(c['structure_factor']) < -1e-8:return False,'static invariant'
    if abs(c['structure_factor'].sum()-9)>1e-5:return False,'static sum rule'
    if np.max(abs(c['correlation_matrix']-cr['correlation_matrix']))>1e-6 or np.max(abs(c['structure_factor']-cr['structure_factor']))>1e-6:return False,'static mismatch'
    if abs(float(js['tier2']['correlation_sum_rule'])-float(c['structure_factor'].sum()))>1e-6:return False,'results tier2 mismatch'
    if d['S_q_omega'].shape!=(12,240) or d['omega_grid'].shape!=(240,) or np.min(d['S_q_omega']) < -1e-8:return False,'bad dynamics'
    if abs(float(d['eta'])-.1)>1e-8 or not np.all(np.diff(d['omega_grid'])>0):return False,'grid/eta'
    if int(js['tier3']['num_lanczos_steps'])!=80 or int(js['tier3']['num_omega_points'])!=240:return False,'results tier3 mismatch'
    if np.max(abs(d['S_q_omega']-dr['S_q_omega']))>2e-5:return False,'dynamic mismatch'
    if abs(float(d['omega_max'])-float(dr['omega_max']))>.05:return False,'omega max'
    integ=np.trapezoid(d['S_q_omega'],d['omega_grid'],axis=1); rel=np.max(abs(integ-c['structure_factor'])/np.maximum(abs(c['structure_factor']),1e-12))
    if rel>.05:return False,'dynamic sum rule'
    for i in range(12):
      if f'a_{i}' not in d or f'b_{i}' not in d or len(d[f'a_{i}'])!=80 or len(d[f'b_{i}']) not in (79,80): return False,'Lanczos contract'
    return True,'ok'
  except Exception as e:return False,f'malformed: {e}'
def main():
  p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--reference-dir',required=True);a=p.parse_args(); out=Path(a.output_dir).resolve(); ref=Path(a.reference_dir).resolve()
  if any(not (out/x).is_file() for x in REQ) or not (out/'solver.py').is_file(): return fail('missing required output')
  if any((out/x).stat().st_size>8_000_000 for x in REQ+('solver.py',)): return fail('oversized output')
  ok,why=compare_case(out,ref,False)
  if not ok:return fail('visible case: '+why)
  with tempfile.TemporaryDirectory(prefix='heis_hidden_') as td:
    hidden_out=Path(td)/'out'; hidden_out.mkdir()
    cfg=ref/'hidden_config.json'; hidden_ref=ref/'hidden_case'
    try:
      r=subprocess.run([sys.executable,str(out/'solver.py'),'--config',str(cfg),'--output-dir',str(hidden_out)],cwd=str(out),capture_output=True,text=True,timeout=55)
    except Exception as e:return fail('hidden solver execution',error=str(e))
    if r.returncode!=0:return fail('hidden solver failed',stderr=r.stderr[-400:])
    ok2,why2=compare_case(hidden_out,hidden_ref,True)
    if not ok2:return fail('hidden case: '+why2)
  print(json.dumps({'score':1.0,'visible':'ok','hidden':'ok'}))
if __name__=='__main__': main()
