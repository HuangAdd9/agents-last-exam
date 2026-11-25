from __future__ import annotations
import json,shlex,shutil,tempfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from model import solve_case,write_outputs
from oracle import solve_reference
from score_outputs import score
import main as task_main
cfg=json.loads((ROOT/'input/model_config.json').read_text()); ref=ROOT/'reference'
assert not (ref/'hidden_case').exists() and not (ref/'hidden_config.json').exists()

# The pinned snapshot's seccomp profile denies socketpair(), so asyncio.run()
# cannot create its selector loop under the snapshot validator.  These fixture
# coroutines perform only immediately-completing fake-session awaits and can be
# driven directly without opening a socket.
def run_fixture(coro):
    try:
        while True: coro.send(None)
    except StopIteration as done:
        return done.value
# Exercise inherited LinuxTaskConfig paths and the setup hook with a small
# DesktopSession double; this catches class-attribute/dataclass integration.
class _Interface:
    def __init__(self): self.dirs=[]
    async def create_dir(self,p): self.dirs.append(p)
class _Session:
    def __init__(self): self.interface=_Interface(); self.files={}; self.commands=[]
    async def write_file(self,p,c): self.files[p]=c
    async def run_command(self,c,**kw):
        self.commands.append(c)
        if '--output-dir' in c:
            from oracle import solve_reference,write_outputs
            tokens=shlex.split(c); cfg=json.loads(self.files[tokens[tokens.index('--config')+1]])
            out=Path(tokens[tokens.index('--output-dir')+1]); tmp=Path(tempfile.mkdtemp()); write_outputs(solve_reference(cfg),tmp)
            for name in ('diatomic_1d.npz','dispersion_2d.npz','dos.npz','thermodynamics.npz','results.json'):
                self.files[str(out/name)]=(tmp/name).read_bytes()
        return {'return_code':0}
    async def file_exists(self,p): return p in self.files
    async def read_bytes(self,p): return self.files[p]
meta=task_main.Config().to_metadata(); assert meta['domain_name']=='physical_sciences' and task_main.Config().task_dir.startswith('/media/user/data/agenthle/physical_sciences/')
fake_task=type('Task',(),{'metadata':meta})(); fake_session=_Session()
# ALE stages gated task-data before start().  Mirror that contract here so the
# task implementation never needs access to the authoring bundle's data tree.
for source, remote in ((ROOT/'input', meta['input_dir']), (ROOT/'reference', meta['reference_dir'])):
    for path in source.rglob('*'):
        if path.is_file():
            fake_session.files[f"{remote}/{path.relative_to(source).as_posix()}"] = path.read_bytes()
preexisting=set(fake_session.files); run_fixture(task_main.start(fake_task,fake_session))
assert meta['remote_output_dir'] in fake_session.interface.dirs
assert set(fake_session.files) == preexisting and meta['input_dir']+'/problem_spec.md' in fake_session.files
# Independent unit-cell geometry checks: the diagonal must partition a unit
# square into two disjoint half-area triangles, and an affine x field must
# integrate to mean x=1/2.  These checks do not call either implementation.
triangles = [((0.,0.),(1.,0.),(1.,1.)), ((0.,0.),(1.,1.),(0.,1.))]
areas=[]; x_integrals=[]
for tri in triangles:
    p,q,r = (np.asarray(v) for v in tri)
    areas.append(abs(np.linalg.det(np.stack((q-p,r-p))))/2.)
    x_integrals.append(areas[-1] * np.mean([p[0],q[0],r[0]]))
assert np.allclose(areas, [0.5,0.5]) and np.isclose(sum(areas),1.0)
assert np.isclose(sum(x_integrals)/sum(areas),0.5)
# Independent triangle-density normalization check (a nondegenerate test
# triangle integrates to one before the four-branch DOS normalization).
lo,mid,hi=0.1,0.4,0.9; w=np.linspace(lo,hi,100001); pdf=np.where(w<mid,2*(w-lo)/((mid-lo)*(hi-lo)),2*(hi-w)/((hi-lo)*(hi-mid)))
assert np.isclose(np.trapezoid(pdf,w),1.0,rtol=1e-6,atol=1e-8)
with tempfile.TemporaryDirectory(prefix='phonon_tests_') as td:
    base=Path(td); perfect=base/'perfect'; write_outputs(solve_case(cfg),perfect)
    # The same double crosses the evaluator boundary: visible artifacts are
    # pulled from the guest, the fake guest solver writes hidden artifacts,
    # and evaluate() scores them with the host oracle.
    for name in ('diatomic_1d.npz','dispersion_2d.npz','dos.npz','thermodynamics.npz','results.json'):
        fake_session.files[str(Path(meta['remote_output_dir'])/name)]=(perfect/name).read_bytes()
    assert run_fixture(task_main.evaluate(fake_task,fake_session)) == [1.0]
    # The reference path is independently coded.  Exercise it here and make
    # sure it reproduces every governed visible artifact before scoring.
    assert 'from model import' not in (ROOT/'scripts/oracle.py').read_text()
    independent=solve_reference(cfg)
    expected = {
        'diatomic_1d.npz': {'q_points': independent['diatomic'][0], 'omega_acoustic': independent['diatomic'][1], 'omega_optical': independent['diatomic'][2]},
        'dispersion_2d.npz': {k: independent[k] for k in ('q_points','q_distance','frequencies')},
        'dos.npz': {k: independent[k] for k in ('omega_bins','dos')},
        'thermodynamics.npz': {k: independent[k] for k in ('temperatures','energy','heat_capacity','free_energy')},
    }
    for name, values in expected.items():
        with np.load(ref/name) as archive:
            for key, value in values.items():
                assert np.allclose(archive[key], value, rtol=1e-8, atol=1e-9)
    # A self-contained candidate solver is the model plus its CLI.
    (perfect/'solver.py').write_text((ROOT/'scripts/model.py').read_text()+"\nif __name__=='__main__':\n import argparse, json\n p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args(); write_outputs(solve_case(json.loads(Path(a.config).read_text())),Path(a.output_dir))\n")
    pscore=score(perfect,ref); assert pscore['score']==1.0,pscore
    empty=base/'empty'; empty.mkdir(); escore=score(empty,ref); assert escore['score']==0.0
    wrong=base/'wrong'; shutil.copytree(perfect,wrong); 
    with np.load(wrong/'dispersion_2d.npz') as z: d={k:z[k] for k in z.files}
    d['frequencies']*=2.0; np.savez_compressed(wrong/'dispersion_2d.npz',**d)
    wscore=score(wrong,ref); assert wscore['components']['tier2']==0.0 and wscore['score']<1.0,wscore
    copy=base/'copy'; shutil.copytree(perfect,copy); (copy/'solver.py').write_text((perfect/'solver.py').read_text())
    # Solver still computes hidden from config; mutate it to ignore hidden k3/m2 by visible config.
    s=(copy/'solver.py').read_text(); s=s.replace("json.loads(Path(a.config).read_text())","json.loads((ROOT/'input/model_config.json').read_text())") if False else s
    # Instead use a visibly copied-output solver that always emits the visible reference.
    (copy/'solver.py').write_text("import argparse,shutil\np=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--output-dir'); a=p.parse_args(); shutil.copytree(r'"+str(perfect)+"',a.output_dir,dirs_exist_ok=True)\n")
    cscore=score(copy,ref); assert cscore['score']<1.0 and cscore['components']['hidden']==0.0,cscore
    # Principal historical loophole: use the config path to locate the hidden
    # reference directory and copy its answers.  The grader must stage only an
    # isolated config copy, so this solver cannot see reference/hidden_case.
    exfil=base/'exfil'; shutil.copytree(perfect,exfil)
    (exfil/'solver.py').write_text("import argparse,shutil\nfrom pathlib import Path\np=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--output-dir'); a=p.parse_args(); shutil.copytree(Path(a.config).parent/'hidden_case',a.output_dir,dirs_exist_ok=True)\n")
    exscore=score(exfil,ref); assert exscore['components']['hidden']==0.0 and exscore['score']<1.0,exscore
    # A second exfiltration path uses the candidate's original location:
    # before isolation, output/ sat beside reference/ and this solver could
    # copy reference/hidden_case through __file__.parents[1].
    sibling_pkg=base/'package_layout'; sibling_pkg.mkdir()
    shutil.copytree(ref,sibling_pkg/'reference')
    sibling_out=sibling_pkg/'output'; shutil.copytree(perfect,sibling_out)
    (sibling_out/'solver.py').write_text("import argparse,shutil\nfrom pathlib import Path\np=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--output-dir'); a=p.parse_args(); shutil.copytree(Path(__file__).resolve().parents[1]/'reference'/'hidden_case',a.output_dir,dirs_exist_ok=True)\n")
    sibling_score=score(sibling_out,sibling_pkg/'reference'); assert sibling_score['components']['hidden']==0.0 and sibling_score['score']<1.0,sibling_score
    # Absolute-path search attack: the solver probes the package's original
    # absolute location for a former hidden fixture and copies it if present.
    absolute=base/'absolute'; shutil.copytree(perfect,absolute)
    (absolute/'solver.py').write_text("""import argparse,shutil
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--output-dir'); a=p.parse_args()
for root in (Path('/data/ale'), Path('/workspace'), Path('/tmp')):
    for hit in (root/'reference'/'hidden_case', root/'hidden_case', root/'task'/'reference'/'hidden_case'):
        try:
            if hit.is_dir(): shutil.copytree(hit,a.output_dir,dirs_exist_ok=True); raise SystemExit
        except (PermissionError, OSError): pass
Path(a.output_dir).mkdir(parents=True,exist_ok=True)
""")
    absolute_score=score(absolute,ref); assert absolute_score['components']['hidden']==0.0 and absolute_score['score']<1.0,absolute_score
    # /proc-discovered attack: inspect every visible process cwd/root for a
    # reference sibling, covering namespace/path discovery rather than a
    # single hard-coded absolute path.
    procscan=base/'procscan'; shutil.copytree(perfect,procscan)
    (procscan/'solver.py').write_text("""import argparse,os,shutil
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--output-dir'); a=p.parse_args()
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    for field in ('cwd','root'):
        try: hit=Path(os.readlink('/proc/'+pid+'/'+field))/'reference'/'hidden_case'
        except (FileNotFoundError,PermissionError,OSError): continue
        if hit.is_dir(): shutil.copytree(hit,a.output_dir,dirs_exist_ok=True); raise SystemExit
Path(a.output_dir).mkdir(parents=True,exist_ok=True)
""")
    proc_score=score(procscan,ref); assert proc_score['components']['hidden']==0.0 and proc_score['score']<1.0,proc_score
    # The evaluator implementation is never staged beside the solver.  A
    # candidate that discovers and invokes a former _eval_oracle.py must fail.
    oracle_attack=base/'oracle_attack'; shutil.copytree(perfect,oracle_attack)
    (oracle_attack/'solver.py').write_text("""import argparse, pathlib, subprocess, sys
p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--output-dir'); a=p.parse_args()
for root in (pathlib.Path.cwd(), pathlib.Path('/tmp'), pathlib.Path('/data/ale')):
    for hit in list(root.glob('_eval_oracle.py')) + list(root.glob('*/_eval_oracle.py')) if root.exists() else ():
        subprocess.run([sys.executable, str(hit), '--config', a.config, '--output-dir', a.output_dir])
pathlib.Path(a.output_dir).mkdir(parents=True, exist_ok=True)
""")
    oracle_score=score(oracle_attack,ref); assert oracle_score['components']['hidden']==0.0 and oracle_score['score']<1.0,oracle_score
    # Schema is a gate, so one malformed field/archive cannot retain other
    # component credit.
    for label, mutate in (
        ('fractional_integer', lambda j: j['tier1'].__setitem__('num_q_points',121.9)),
        ('string_integer', lambda j: j['tier3'].__setitem__('num_temperatures','18'))):
        bad=base/label; shutil.copytree(perfect,bad); j=json.loads((bad/'results.json').read_text()); mutate(j); (bad/'results.json').write_text(json.dumps(j))
        result=score(bad,ref); assert result['score']==0.0 and all(v==0.0 for v in result['components'].values()),result
    missing=base/'missing_thermo'; shutil.copytree(perfect,missing)
    with np.load(missing/'thermodynamics.npz') as z: vals={k:z[k] for k in z.files if k!='free_energy'}
    np.savez_compressed(missing/'thermodynamics.npz',**vals)
    assert score(missing,ref)['score']==0.0
    nonfinite=base/'nonfinite'; shutil.copytree(perfect,nonfinite)
    with np.load(nonfinite/'dos.npz') as z: vals={k:z[k] for k in z.files}; vals['dos'][0]=np.nan
    np.savez_compressed(nonfinite/'dos.npz',**vals)
    assert score(nonfinite,ref)['score']==0.0
    print(json.dumps({'perfect':pscore,'empty':escore,'wrong_units':wscore,'visible_copy_hidden_probe':cscore,'config_parent_exfiltration':exscore,'sibling_reference_exfiltration':sibling_score,'absolute_path_exfiltration':absolute_score,'proc_filesystem_exfiltration':proc_score,'oracle_invocation':oracle_score,'schema_gate':'passed','cell_affine_check':'passed','triangle_density_normalization':'passed'},sort_keys=True))
