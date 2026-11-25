"""Deterministic local gates, including adversarial scientific/spec loopholes."""
from __future__ import annotations
import asyncio, json, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import importlib.util, sys

ROOT=Path(__file__).resolve().parents[1]
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; assert s.loader; s.loader.exec_module(m); return m
oracle=mod('oracle',ROOT/'scripts/oracle.py'); main=mod('task_main',ROOT/'main.py')

def run(inp,out):
    subprocess.run(['python3',str(ROOT/'scripts/solver_reference.py'),'--input',str(inp),'--output',str(out)],check=True)

class _PullSession:
    """Local session double for exercising the evaluator's guest-file gate."""
    async def run_command(self,command,**kwargs):
        p=subprocess.run(command,shell=True,capture_output=True,text=True)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    async def read_bytes(self,path):
        return Path(path).read_bytes()

def output_file_attack_test():
    """Symlinks and sparse oversized artifacts must be rejected before reads."""
    with tempfile.TemporaryDirectory() as td:
        stage=Path(td)/'stage'; run_dir=stage/'run'; run_dir.mkdir(parents=True)
        for name,data in (('result.npz',b'npz'),('trace.csv',b'trace'),('summary.json',b'{}')):
            (run_dir/name).write_bytes(data)
        local=Path(td)/'local'; local.mkdir()
        (run_dir/'result.npz').unlink(); (run_dir/'result.npz').symlink_to('/dev/zero')
        try:
            asyncio.run(main._pull(_PullSession(),str(stage),local))
        except RuntimeError:
            pass
        else:
            raise AssertionError('symlink output was accepted')
        (run_dir/'result.npz').unlink(); (run_dir/'result.npz').write_bytes(b'npz')
        with (run_dir/'trace.csv').open('wb') as f: f.truncate(main.PULL_LIMITS['trace.csv']+1)
        try:
            asyncio.run(main._pull(_PullSession(),str(stage),local))
        except RuntimeError:
            pass
        else:
            raise AssertionError('oversized sparse output was accepted')

def sign_convention_test():
    spec=(ROOT/'input/spec.md').read_text()
    assert 'negative-down/positive-up' in spec and '`flexure_observed_m` is downward' not in spec
    p=json.loads((ROOT/'input/problem.json').read_text()); _,_,load=oracle._forward(p)
    w=oracle._flexure(load,p,23.0)
    assert np.all(w < 0) and np.all(np.asarray(p['flexure_observed_m']) < 0)

class _EvalAttackSession:
    def __init__(self,kind): self.kind=kind; self.stage=None
    async def file_exists(self,path): return path=='output/solve.py' or Path(path).exists()
    async def read_file(self,path):
        if path=='output/solve.py': return '# isolated candidate\n'
        return (ROOT/path).read_text()
    async def write_file(self,path,content):
        Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(content)
        if path.endswith('/solve.py') and path.startswith('/tmp/kaena-'): self.stage=Path(path).parent
    async def run_command(self,command,**kwargs):
        if 'bwrap --die-with-parent' in command:
            run_dir=self.stage/'run'; run_dir.mkdir(parents=True,exist_ok=True)
            (run_dir/'trace.csv').write_bytes(b'trace')
            (run_dir/'summary.json').write_bytes(b'{}')
            if self.kind=='symlink':
                (run_dir/'result.npz').symlink_to('/dev/zero')
            else:
                (run_dir/'result.npz').write_bytes(b'npz')
                with (run_dir/'trace.csv').open('wb') as f: f.truncate(main.PULL_LIMITS['trace.csv']+1)
            return {'returncode':0}
        p=subprocess.run(command,shell=True,capture_output=True,text=True)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    async def read_bytes(self,path): return Path(path).read_bytes()

def evaluator_output_attack_test():
    """The full evaluator returns zero for both guest-file attack variants."""
    task=type('Task',(),{'metadata':{'remote_output_dir':'output','input_dir':'input','reference_dir':'reference'}})()
    for kind in ('symlink','sparse'):
        got=asyncio.run(main.evaluate(task,_EvalAttackSession(kind)))
        assert got==[0.0], (kind,got)

def science_regression_test():
    """The flexure load uses deepest bottom depth, not interval thickness."""
    p=json.loads((ROOT/'input/problem.json').read_text())
    rho,_,load=oracle._forward(p)
    top=np.asarray(p['top_depth_km'],float); bot=np.asarray(p['bottom_depth_km'],float)
    intervals=(bot-top)*1000.0; column=(bot[-1]-top[0])*1000.0
    rmean=np.sum(rho*intervals,axis=0)/column
    fac=np.exp(-((float(p['dx_km'])*1000.0/(float(p['flexural_wavelength_km'])*1000.0))**2)/2.0)
    expected=(rmean-float(p['mantle_density_kg_m3']))*float(p['gravity_accel_m_s2'])*bot[-1]*1000.0*fac
    old=(rmean-float(p['mantle_density_kg_m3']))*float(p['gravity_accel_m_s2'])*column*fac
    assert np.allclose(load,expected,rtol=0,atol=1e-6)
    assert not np.allclose(load,old,rtol=0,atol=1e-6)

def fixture_observation_regression_test():
    """Observations are independent perturbations, while the intended Te remains the grid minimum."""
    fixtures=(
        ('input/problem.json',23.0),
        ('reference/hidden_case.json',19.5),
        ('reference/hidden_geometry.json',22.5),
    )
    for rel, intended_te in fixtures:
        p=json.loads((ROOT/rel).read_text())
        _,_,load=oracle._forward(p)
        observed=np.asarray(p['flexure_observed_m'],float)
        regenerated=oracle._flexure(load,p,intended_te)
        assert np.max(np.abs(observed-regenerated)) > 2e-5, rel
        _,gravity,_=oracle._forward(p)
        gravity_observed=np.asarray(p['gravity_observed_mgal'],float)
        assert np.max(np.abs(gravity_observed-gravity)) > 2e-5, rel
        assert np.sqrt(np.mean((gravity-gravity_observed)**2)) > 2e-5, rel
        trace=[]
        for te in p['te_grid_km']:
            flex=oracle._flexure(load,p,float(te))
            trace.append((float(te),float(np.sqrt(np.mean((flex-observed)**2)))))
        (best,best_rms),min_rms=oracle._choose_te(trace)
        assert best==intended_te, (rel,best,intended_te)
        assert best_rms>2e-5 and abs(best_rms-min_rms)<=1e-12, (rel,best_rms,min_rms)

def observation_copy_shortcut_test():
    """Copying supplied gravity/flexure observations must not receive physical credit."""
    fixtures=('input/problem.json','reference/hidden_case.json','reference/hidden_geometry.json')
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for idx,rel in enumerate(fixtures):
            inp=ROOT/rel; p=json.loads(inp.read_text()); out=td/f'copy{idx}'
            run(inp,out)
            z=np.load(out/'result.npz'); d={k:z[k] for k in z.files}
            d['gravity_mgal']=np.asarray(p['gravity_observed_mgal'],float)
            d['flexure_m']=np.asarray(p['flexure_observed_m'],float)
            d['gravity_rms_mgal']=np.zeros_like(d['gravity_rms_mgal'])
            d['flexure_rms_m']=np.zeros_like(d['flexure_rms_m'])
            np.savez(out/'result.npz',**d)
            ref=oracle.solve_reference(p); s=main.score(out,ref)
            assert s['components']['physical']==0.0 and s['score']<0.2, (rel,s)

def tie_rule_test():
    # The larger-Te exact minimum is within 1e-12 RMS of the smaller grid value.
    best,_=oracle._choose_te([(20.0,1.0),(19.0,1.0+5e-13)])
    assert best==(19.0,1.0+5e-13)

def entrypoint_contract_test():
    spec=(ROOT/'input/spec.md').read_text(); assert 'python solve.py --input PATH --output DIR' in spec
    assert 'output/solve.py' in main.Config().task_description
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'output'; out.mkdir()
        shutil.copy(ROOT/'scripts/solver_reference.py',out/'solve.py')
        subprocess.run(['python3',str(out/'solve.py'),'--input',str(ROOT/'input/problem.json'),'--output',str(out/'run')],check=True)
        assert (out/'run/result.npz').exists()

def main_test():
 with tempfile.TemporaryDirectory() as td:
  td=Path(td); vis=json.loads((ROOT/'input/problem.json').read_text()); ref=oracle.solve_reference(vis)
  good=td/'good';run(ROOT/'input/problem.json',good); s=main.score(good,ref); assert s['score']>0.999 and s['components']['physical']==1.0
  empty=td/'empty';empty.mkdir(); assert main.score(empty,ref)['score']==0.0
  wrong=td/'wrong';shutil.copytree(good,wrong);z=np.load(wrong/'result.npz'); d={k:z[k] for k in z.files};d['gravity_mgal']=d['gravity_mgal']*1e5;np.savez(wrong/'result.npz',**d); assert main.score(wrong,ref)['components']['physical']==0.0 and main.score(wrong,ref)['score']<0.8
  copied=td/'copied';shutil.copytree(good,copied); hidden=json.loads((ROOT/'reference/hidden_case.json').read_text()); href=oracle.solve_reference(hidden); assert main.score(copied,href)['score']<0.5
  shortcut=td/'shortcut';run(ROOT/'reference/hidden_case.json',shortcut); z=np.load(shortcut/'result.npz');d={k:z[k] for k in z.files};d['te_km']=np.asarray([17.0]);np.savez(shortcut/'result.npz',**d); assert main.score(shortcut,href)['components']['physical']==0.0 and main.score(shortcut,href)['score']<0.8
  # Positive, correctly shaped dummy arrays plus the exact invariant earn no shortcut credit.
  dummy=td/'dummy'; dummy.mkdir()
  np.savez(dummy/'result.npz',density_kg_m3=np.ones_like(href['density_kg_m3']),gravity_mgal=np.ones_like(href['gravity_mgal']),flexure_m=np.ones_like(href['flexure_m']),te_km=np.ones_like(href['te_km']),gravity_rms_mgal=np.ones_like(href['gravity_rms_mgal']),flexure_rms_m=np.ones_like(href['flexure_rms_m']))
  with (dummy/'trace.csv').open('w') as f:
   f.write('te_km,flexure_rms_m\n')
   f.writelines(f'{te},{rms}\n' for te,rms in href['trace'])
  (dummy/'summary.json').write_text(json.dumps({'units':main.UNITS,'min_qp_minus_qsv':href['invariant']}))
  assert main.score(dummy,href)['score']<0.05
 science_regression_test(); fixture_observation_regression_test(); observation_copy_shortcut_test(); tie_rule_test(); entrypoint_contract_test(); output_file_attack_test(); evaluator_output_attack_test(); sign_convention_test()
 print('perfect=1.0 empty=0.0 wrong_units=0 visible_copy=0 observation_copy=0 on visible+hidden hidden_te_shortcut=0 dummy_positive_invariant<0.05 science_load=bottom_depth fixture_observations=independently_perturbed intended_te=23/19.5/22.5 tie=1e-12 entrypoint=solve.py symlink_zero=0 sparse_trace=0 evaluator_attack_score=0 sign=negative_down_positive_up')

if __name__=='__main__': main_test()
